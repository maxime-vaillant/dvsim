import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import tqdm
from matplotlib import pyplot as plt

from backend.event_processing.event import UniformSamplingDynamicVisionProcessing
from backend.graphical_computation.utils.renderer import Renderer
from backend.optical_flow_processing.optical_flow import OpticalFlowProcessing
from backend.saving.saver import Saver
from backend.trajectories.trajectory import ImportedTrajectory, Trajectory
from backend.vizualisation.visualizer import Visualizer


class Computation:
    def __init__(
            self,
            height: int,
            width: int,
            ground_truth_sampling_rate: int,
            obj_path: str,
            texture_path: str,
            trajectory: Trajectory,
            timestep: int
    ):
        self.height = height
        self.width = width

        self.ground_truth_sampling_rate = ground_truth_sampling_rate

        self.dynamic_vision_processing = UniformSamplingDynamicVisionProcessing(
            c_plus=np.log(1.1),
            c_minus=np.log(1.1)
        )

        self.optical_flow_processing = OpticalFlowProcessing()

        self.trajectory = trajectory

        self.renderer = Renderer(
            height=height,
            width=width,
            obj_path=obj_path,
            texture_path=texture_path,
            trajectory=self.trajectory
        )

        self.timestep = timestep

    def compute(self):
        cur_timestamp = self.trajectory.initial_timestamp

        data: Dict[str, List] = {
            "events": [],
            "intensity": [],
            "depth": [],
            "optical_flow": [],
            "timestamp": []
        }

        step_count = 0

        pbar = tqdm.tqdm()

        while self.trajectory.is_trajectory_ongoing():
            pbar.update(1)

            step_count += 1
            color_frame, depth_frame, optical_flow = self.renderer.render(current_time=cur_timestamp, timestep=self.timestep)

            events = self.dynamic_vision_processing(
                intensity_img=color_frame,
                timestamp=cur_timestamp,
                time_step=self.timestep
            )

            if step_count % self.ground_truth_sampling_rate == 0:
                data['intensity'].append(np.copy(color_frame[:, :, :3]))
                data['depth'].append(np.copy(depth_frame))
                data['optical_flow'].append(np.copy(optical_flow))

                data['timestamp'].append(cur_timestamp)

            if len(events) > 0:
                data['events'].append(events)

            cur_timestamp += self.timestep

        pbar.close()

        print('Sorting events...')
        with ThreadPoolExecutor() as executor:
            sorted_futures = [executor.submit(lambda e: e[np.argsort(e[:, 0])], events) for events in data["events"]]
            sorted_events_list = [future.result() for future in sorted_futures]

        data["events"] = sorted_events_list

        return data

    def process_computation(self, output_dir, scenario, saving=True, visualizing=True):
        has_trajectory = True

        saver = Saver(output_dir=output_dir)
        visualizer = Visualizer(
            output_dir=output_dir,
            height=height,
            width=width,
            timestep=timestep,
            sample_rate=self.ground_truth_sampling_rate
        )

        scenario_idx = 0

        while has_trajectory:
            data = self.compute()

            cur_scenario_name = f'{scenario}{scenario_idx}'

            if saving:
                saver.save_events(events=data["events"], scenario=cur_scenario_name, filename='events.h5')
                saver.save_ground_truth(data=data, scenario=cur_scenario_name, filename='ground_truth.h5')

            if visualizing:
                visualizer.visualize_events(
                    events=data["events"],
                    scenario=cur_scenario_name,
                    filename='viz_events.mp4',
                    time_accumulation=self.timestep,
                )
                visualizer.visualize_frames(
                    frames=data["intensity"],
                    scenario=cur_scenario_name,
                    filename='viz_intensity.mp4'
                )
                visualizer.visualize_optical_flow_particles(
                    flows=data["optical_flow"],
                    scenario=cur_scenario_name,
                    filename='viz_optical_flow_particles.mp4'
                )
                visualizer.visualize_depth_frames(
                    depth_frames=data["depth"],
                    scenario=cur_scenario_name,
                    filename='viz_depth_frames.mp4'
                )

            scenario_idx += 1
            has_trajectory = self.trajectory.get_next_trajectory()



if __name__ == '__main__':
    argparse.ArgumentParser()

    height=260
    width=346
    timestep = 10000

    trajectory_paths = sorted(str(p) for p in Path('../../assets/trajectories').glob('*.json'))

    trajectory = ImportedTrajectory(trajectory_paths=trajectory_paths)

    computation = Computation(
        height=height,
        width=width,
        ground_truth_sampling_rate=1,
        obj_path='../../assets/scenes/bureauxtof.obj',
        texture_path='../../assets/scenes/bureauxtof.jpg',
        trajectory=trajectory,
        timestep=timestep
    )
    computation.process_computation(output_dir='../../outputs', scenario='bureauxtof', saving=True, visualizing=True)
