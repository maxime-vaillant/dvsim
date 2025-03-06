import os

import h5py
import hdf5plugin
import numpy as np

class Saver:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def _ensure_scenario_dir(self, scenario):
        scenario_dir = f"{self.output_dir}/{scenario}"
        os.makedirs(scenario_dir, exist_ok=True)

    def save_events(self, events, scenario, filename):
        print("Saving events...")
        self._ensure_scenario_dir(scenario=scenario)

        with h5py.File(f"{self.output_dir}/{scenario}/{filename}", 'w') as file:
            events_group = file.create_group('events')

            timestamp_dataset = events_group.create_dataset(
                name='t',
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint32
            )
            x_dataset = events_group.create_dataset(
                name='x',
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint16
            )
            y_dataset = events_group.create_dataset(
                name='y',
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint16
            )
            polarity_dataset = events_group.create_dataset(
                name='p',
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint8
            )

            for event in events:
                cur_size = event.shape[0]

                timestamp_dataset.resize((timestamp_dataset.shape[0] + cur_size,))
                x_dataset.resize((x_dataset.shape[0] + cur_size,))
                y_dataset.resize((y_dataset.shape[0] + cur_size,))
                polarity_dataset.resize((polarity_dataset.shape[0] + cur_size,))

                timestamp_dataset[-cur_size:] = event[:, 0]
                x_dataset[-cur_size:] = event[:, 1]
                y_dataset[-cur_size:] = event[:, 2]
                polarity_dataset[-cur_size:] = event[:, 3]

    def save_ground_truth(self, data, scenario, filename):
        print("Saving ground truth...")

        with h5py.File(f"{self.output_dir}/{scenario}/{filename}", 'w') as file:
            ground_truth_group = file.create_group('ground_truth')

            # Save depth maps
            if 'depth' in data.keys():
                print(" > Saving depth...")
                ground_truth_group.create_dataset(
                    name='depth',
                    data=np.array(data['depth'], dtype=np.float32),
                    compression='gzip',
                    compression_opts=9
                )

            # Save intensity images
            if 'intensity' in data.keys():
                print(" > Saving intensity...")
                ground_truth_group.create_dataset(
                    name='intensity',
                    data=np.array(data['intensity'], dtype=np.uint8),
                    compression='gzip',
                    compression_opts=9
                )

            # Save optical flow
            if 'optical_flow' in data.keys():
                print(" > Saving optical flow...")
                ground_truth_group.create_dataset(
                    name='optical_flow',
                    data=np.array(data['optical_flow'], dtype=np.float32),
                    compression='gzip',
                    compression_opts=9
                )

            # Save position data
            if 'position' in data.keys():
                print(" > Saving position...")
                ground_truth_group.create_dataset(
                    name='position',
                    data=np.array(data['position'], dtype=np.float32),
                    compression='gzip',
                    compression_opts=9
                )

            # Save timestamps
            if 'timestamp' in data.keys():
                print(" > Saving timestamp...")
                ground_truth_group.create_dataset(
                    name='timestamp',
                    data=np.array(data['timestamp'], dtype=np.uint32)
                )
