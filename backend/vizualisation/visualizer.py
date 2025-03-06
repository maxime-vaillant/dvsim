import os

import cv2
import numpy as np


class Visualizer:
    def __init__(self, output_dir, height, width, timestep, sample_rate):
        self.output_dir = output_dir
        self.height = height
        self.width = width
        self.sample_rate = sample_rate
        self.timestep = timestep

    def _ensure_scenario_dir(self, scenario):
        scenario_dir = f"{self.output_dir}/{scenario}"
        os.makedirs(scenario_dir, exist_ok=True)

    def visualize_events(self, events, scenario, filename, time_accumulation=10000):
        print("Visualize events...")
        if not events or len(events) == 0:
            raise ValueError("Events list is empty.")

        cur_t = events[0][0, 0]
        max_t = events[-1][-1, 0]

        self._ensure_scenario_dir(scenario=scenario)

        video = cv2.VideoWriter(
            f"{self.output_dir}/{scenario}/{filename}",
            cv2.VideoWriter_fourcc(*'mp4v'),
            1_000_000 // time_accumulation,
            (self.width, self.height)
        )

        cur_idx = 0
        num_batches = len(events)

        while cur_t < max_t:
            concerned_events = []

            # Collect events in the time window
            while cur_idx < num_batches and cur_t + time_accumulation > events[cur_idx][-1, 0]:
                cur_events = events[cur_idx]
                idx_start = np.searchsorted(cur_events[:, 0], cur_t)
                concerned_events.append(cur_events[idx_start:])
                cur_idx += 1

            if cur_idx < num_batches:
                cur_events = events[cur_idx]
                idx_start = np.searchsorted(cur_events[:, 0], cur_t)
                idx_end = np.searchsorted(cur_events[:, 0], cur_t + time_accumulation)
                concerned_events.append(cur_events[idx_start:idx_end])

            if concerned_events:
                concat_events = np.concatenate(concerned_events, axis=0)

                # Convert polarity 0 to -1
                concat_events[:, 3] = np.where(concat_events[:, 3] == 0, -1, concat_events[:, 3])

                # Efficiently accumulate event counts
                accumulated_events = np.zeros((self.height, self.width), dtype=np.int16)
                np.add.at(accumulated_events, (concat_events[:, 2], concat_events[:, 1]), concat_events[:, 3])

                # Normalize for visualization
                abs_accumulated = np.abs(accumulated_events)
                max_val = np.log1p(abs_accumulated.max()) if abs_accumulated.max() > 0 else 1
                normalized = (np.log1p(abs_accumulated) / max_val * 255).astype(np.uint8)

                events_array = np.zeros((self.height, self.width,  3), dtype=np.uint8)

                # Create event visualization
                events_array[:, :, 0] = np.where(accumulated_events > 0, normalized, 0)
                events_array[:, :, 2] = np.where(accumulated_events < 0, normalized, 0)
                events_array[accumulated_events == 0] = 255

                video.write(events_array)

            cur_t += time_accumulation

        video.release()

    def visualize_frames(self, frames, scenario, filename):
        print("Visualize frames...")
        self._ensure_scenario_dir(scenario=scenario)

        video = cv2.VideoWriter(
            f"{self.output_dir}/{scenario}/{filename}",
            cv2.VideoWriter_fourcc(*'mp4v'),
            1_000_000 // (self.timestep * self.sample_rate),
            (self.width, self.height)
        )

        for frame in frames:
            video.write(frame.astype(np.uint8))

        video.release()

    def visualize_optical_flow(self, flows, scenario, filename):
        print("Visualizing optical flow...")
        self._ensure_scenario_dir(scenario=scenario)

        video = cv2.VideoWriter(
            f"{self.output_dir}/{scenario}/{filename}",
            cv2.VideoWriter_fourcc(*'mp4v'),
            1_000_000 // (self.timestep * self.sample_rate),
            (self.width, self.height)
        )

        for flow in flows:
            hsv = np.zeros((self.height, self.width, 3), dtype=np.float32)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

            hsv[..., 0] = ang / 360.0
            hsv[..., 1] = 1.0
            hsv[..., 2] = np.clip(mag / (np.max(mag) + 1e-5), 0, 1)

            hsv_8bit = (hsv * 255).astype(np.uint8)
            flow_rgb = cv2.cvtColor(hsv_8bit, cv2.COLOR_HSV2BGR)

            video.write(flow_rgb)

        video.release()

    def visualize_optical_flow_particles(
            self,
            flows,
            scenario,
            filename,
            num_particles=1000,
            particle_life=20,
            fade_factor=0.95
    ):
        print("Visualizing optical flow with particles...")
        self._ensure_scenario_dir(scenario=scenario)

        video = cv2.VideoWriter(
            f"{self.output_dir}/{scenario}/{filename}",
            cv2.VideoWriter_fourcc(*'mp4v'),
            1_000_000 // (self.timestep * self.sample_rate * 2),
            (self.width, self.height)
        )

        random_generator = np.random.default_rng()

        particles = random_generator.standard_normal(size=(num_particles, 3))
        particles[:, 0] *= self.width
        particles[:, 1] *= self.height
        particles[:, 2] *= particle_life

        background = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for flow in flows:
            background = (background * fade_factor).astype(np.uint8)

            u = flow[..., 0]
            v = flow[..., 1]

            magnitude = np.sqrt(u ** 2 + v ** 2)
            max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
            angle = np.arctan2(v, u) * 180 / np.pi

            frame = background.copy()

            for i in range(num_particles):
                x, y, life = particles[i]

                if x < 0 or x >= self.width or y < 0 or y >= self.height or life <= 0:
                    particles[i, 0] = random_generator.standard_normal() * self.width
                    particles[i, 1] = random_generator.standard_normal() * self.height
                    particles[i, 2] = particle_life
                    continue

                ix, iy = int(min(max(0, x), self.width - 1)), int(min(max(0, y), self.height - 1))
                dx = u[iy, ix]
                dy = v[iy, ix]

                if abs(dx) < 0.1 and abs(dy) < 0.1:
                    continue

                flow_angle = angle[iy, ix]
                flow_mag = magnitude[iy, ix]

                hue = (flow_angle + 180) / 360.0 * 180
                saturation = 255
                value = min(255, int(flow_mag / max_mag * 255))

                color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0, 0]
                color = (int(color[0]), int(color[1]), int(color[2]))

                new_x = x + dx
                new_y = y + dy

                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    cv2.line(
                        frame,
                        (int(x), int(y)),
                        (int(new_x), int(new_y)),
                        color,
                        1,
                        cv2.LINE_AA
                    )

                    particles[i, 0] = new_x
                    particles[i, 1] = new_y
                    particles[i, 2] -= 1
                else:
                    particles[i, 0] = random_generator.standard_normal() * self.width
                    particles[i, 1] = random_generator.standard_normal() * self.height
                    particles[i, 2] = particle_life

            video.write(frame)

        video.release()

    def visualize_depth_frames(self, depth_frames, scenario, filename, min_depth=None, max_depth=None):
        print("Visualizing depth frames...")
        self._ensure_scenario_dir(scenario=scenario)

        video = cv2.VideoWriter(
            f"{self.output_dir}/{scenario}/{filename}",
            cv2.VideoWriter_fourcc(*'mp4v'),
            1_000_000 // (self.timestep * self.sample_rate),
            (self.width, self.height)
        )

        # Find global min/max for consistent scaling across all frames if not provided
        if min_depth is None or max_depth is None:
            all_valid = np.concatenate([frame[frame > 0].flatten() for frame in depth_frames])
            global_min = all_valid.min() if min_depth is None else min_depth
            global_max = all_valid.max() if max_depth is None else max_depth
            print(f"Depth range: {global_min:.2f} to {global_max:.2f}")
        else:
            global_min, global_max = min_depth, max_depth

        for frame in depth_frames:
            # Create mask for invalid depth values (0)
            valid_mask = frame > 0

            # Create normalized representation (0-255) for visualization
            # Note the inversion: (max - depth) makes near objects white and far objects black
            normalized = np.zeros_like(frame)
            normalized[valid_mask] = np.clip((global_max - frame[valid_mask]) / (global_max - global_min), 0, 1) * 255
            depth_vis = normalized.astype(np.uint8)

            # Convert to 3-channel for video output
            depth_vis_3ch = np.stack([depth_vis] * 3, axis=-1)

            video.write(depth_vis_3ch)

        video.release()