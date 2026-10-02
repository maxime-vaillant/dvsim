"""DiffusionMotionStyle: planned path -> the same path walked by a person."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from backend.motion.dataset import CONTEXT, HOP, WINDOW
from backend.motion.decompose import compose, conditioning, extract_path
from backend.motion.model import Diffusion, torch
from backend.motion.tracks import DT_US
from backend.motion.train import load_checkpoint, pick_device
from backend.trajectories.trajectory import WaypointTrajectory


class DiffusionMotionStyle:
    def __init__(self, model, cfg, stats, device, sample_steps: int = 50):
        self.model, self.cfg, self.stats, self.device = model, cfg, stats, device
        self.diffusion = Diffusion(cfg.diffusion_steps)
        self.sample_steps = sample_steps

    @classmethod
    def load(cls, path, device: Optional[str] = None, sample_steps: int = 50) -> "DiffusionMotionStyle":
        dev = pick_device(device)
        model, cfg, stats, _ = load_checkpoint(path, dev)
        return cls(model, cfg, stats, dev, sample_steps)

    def apply(self, traj: WaypointTrajectory, rng=None) -> WaypointTrajectory:
        """``traj``'s path (its positions) walked with sampled human motion; same start, end and number of seconds.
        The planner's own orientations are replaced. A seed (or Generator) makes the result reproducible."""
        rng = np.random.default_rng(rng)
        n = int(traj.duration_us // DT_US) + 1
        if n < 5:
            raise ValueError(f"the trajectory is too short for motion synthesis ({traj.duration_us} us)")
        times = traj.start_us + np.arange(n, dtype=np.int64) * DT_US
        path = extract_path(traj.poses_at(times)[:, :3])
        cond = conditioning(path)
        lo, hi = self.stats.speed_range
        if cond[:, 0].min() < lo or cond[:, 0].max() > hi:
            warnings.warn(f"path speed {cond[:, 0].min():.2f}-{cond[:, 0].max():.2f} m/s is outside the "
                          f"{lo:.2f}-{hi:.2f} m/s the model was trained on: clamped", stacklevel=2)
            cond = cond.copy()
            cond[:, 0] = np.clip(cond[:, 0], lo, hi)
        cond_n = self.stats.norm_cond(cond).astype(np.float32)

        k = 1 if n <= WINDOW else int(np.ceil((n - WINDOW) / HOP)) + 1
        total = (k - 1) * HOP + WINDOW
        cond_p = np.concatenate([cond_n, np.repeat(cond_n[-1:], total - n, axis=0)]) if total > n else cond_n
        gen = torch.Generator().manual_seed(int(rng.integers(2 ** 62)))
        res = np.zeros((total, 6), np.float32)
        for i in range(k):
            s = i * HOP
            c = torch.from_numpy(cond_p[s:s + WINDOW])[None].to(self.device)
            if i == 0:
                out = self.diffusion.sample(self.model, c, steps=self.sample_steps, generator=gen)
            else:
                ctx = torch.from_numpy(res[s:s + WINDOW].copy())[None].to(self.device)  # first CONTEXT = previous tail
                out = self.diffusion.sample(self.model, c, ctx, CONTEXT, self.sample_steps, gen)
            res[s:s + WINDOW] = out[0].cpu().numpy()
        positions, quats = compose(path, self.stats.denorm_res(res[:n].astype(np.float64)))
        return WaypointTrajectory(times, positions, quats)
