from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

import glm

from backend.graphical_computation.utils.loaders import load_trajectory

class Trajectory(ABC):
    camera_pos: glm.vec3
    camera_front: glm.vec3

    initial_timestamp: int

    @abstractmethod
    def get_next_point(self, current_time, timestep) -> Tuple[glm.vec3, glm.vec3]:
        pass

    @abstractmethod
    def is_trajectory_ongoing(self) -> bool:
        pass

    @abstractmethod
    def get_next_trajectory(self) -> bool:
        pass


class ImportedTrajectory(Trajectory):
    def __init__(self, trajectory_paths: List[str], initial_points: Optional[List]=None):
        self.trajectory_paths = trajectory_paths
        self.trajectory_index = 0

        self.initial_points = initial_points

        self.trace = load_trajectory(trajectory_paths[0])
        self.trace_index = 0

        self.initial_point = glm.vec3(0.0, 0.0, 1.0) if initial_points is None or initial_points[0] is None else initial_points[0]
        self.initial_timestamp = int(self.trace[0]['timestamp'] * 1000)

        self.camera_pos = self.initial_point
        self.camera_front = glm.vec3(0.0, 0.0, -1.0)

    def is_trajectory_ongoing(self) -> bool:
        # print(f'{self.index} / {len(self.trace)}')

        return self.trace_index + 2 < len(self.trace)

    def get_next_point(self, current_time, timestep):
        if self.is_trajectory_ongoing():
            # Get the current and next trajectory points
            current_point = self.trace[self.trace_index]
            next_point = self.trace[self.trace_index + 1]

            # Extract timestamps
            t_current = int(current_point['timestamp'] * 1000)
            t_next = int(next_point['timestamp'] * 1000)

            # Ensure we are within the time range
            if t_current <= current_time < t_next:
                # Compute the interpolation factor (lerp)
                alpha = (current_time - t_current) / (t_next - t_current)

                # Interpolate position
                pos_current = glm.vec3(current_point['x'], current_point['y'], current_point['z'])
                pos_next = glm.vec3(next_point['x'], next_point['y'], next_point['z'])
                self.camera_pos = self.initial_point + glm.mix(pos_current, pos_next, alpha)

                # Interpolate front vector (orientation)
                front_current = glm.vec3(current_point['rx'], current_point['ry'], current_point['rz'])
                front_next = glm.vec3(next_point['rx'], next_point['ry'], next_point['rz'])
                self.camera_front = glm.mix(front_current, front_next, alpha)

            while current_time + timestep >= t_next and self.trace_index + 2 < len(self.trace):
                # Move to the next trajectory point
                self.trace_index += 1

                next_point = self.trace[self.trace_index + 1]
                t_next = int(next_point['timestamp'] * 1000)

        return self.camera_pos, self.camera_front

    def get_next_trajectory(self):
        self.trajectory_index += 1

        if self.trajectory_index < len(self.trajectory_paths):
            self.trace = load_trajectory(self.trajectory_paths[self.trajectory_index])
            self.trace_index = 0

            if self.initial_points is None or self.initial_points[self.trajectory_index] is None:
                self.initial_point = glm.vec3(0.0, 0.0, 1.0)
            else:
                self.initial_point = self.initial_points[self.trajectory_index]

            self.initial_timestamp = int(self.trace[0]['timestamp'] * 1000)
            self.camera_pos = self.initial_point

            return True

        return False

