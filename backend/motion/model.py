"""Conditional 1D diffusion over residual windows (torch). Imported lazily: see ``require_torch``."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from backend.motion import require_torch
from backend.motion.dataset import CONTEXT
from backend.motion.decompose import N_COND, N_RESIDUAL


torch = require_torch()
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

IN_CH = N_RESIDUAL + N_COND + N_RESIDUAL + 1   # noisy residual, conditioning, context, context mask


def timestep_embedding(t, dim: int):
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = t.float()[:, None] * freq[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv1d(cin, cout, 5, padding=2)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv1d(cout, cout, 5, padding=2)
        self.film = nn.Linear(emb, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(emb).chunk(2, dim=1)
        h = self.n2(h) * (1 + scale[:, :, None]) + shift[:, :, None]
        return self.skip(x) + self.c2(F.silu(h))


class MotionUNet(nn.Module):
    """x0-prediction network: (noisy residual, step, conditioning, context) -> clean residual, all (B, W, C)."""

    def __init__(self, base: int = 64, mults=(1, 2, 4), emb: int = 128):
        super().__init__()
        self.emb = emb
        chans = [base * m for m in mults]
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv1d(IN_CH, chans[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = chans[0]
        for i, c in enumerate(chans):
            self.down.append(ResBlock(prev, c, emb))
            self.pool.append(nn.Conv1d(c, c, 3, stride=2, padding=1) if i < len(chans) - 1 else nn.Identity())
            prev = c
        self.mid = nn.ModuleList([ResBlock(prev, prev, emb), ResBlock(prev, prev, emb)])
        self.up = nn.ModuleList()
        for i in reversed(range(len(chans))):
            self.up.append(ResBlock(prev + chans[i], chans[i], emb))
            prev = chans[i]
        self.out = nn.Sequential(nn.GroupNorm(8, prev), nn.SiLU(), nn.Conv1d(prev, N_RESIDUAL, 3, padding=1))

    def forward(self, x_t, t, cond, ctx, mask):
        e = self.t_mlp(timestep_embedding(t, self.emb))
        h = self.inp(torch.cat([x_t, cond, ctx * mask[..., None], mask[..., None]], dim=-1).transpose(1, 2))
        skips = []
        for block, pool in zip(self.down, self.pool):
            h = block(h, e)
            skips.append(h)
            h = pool(h)
        for block in self.mid:
            h = block(h, e)
        for i, block in enumerate(self.up):
            skip = skips[-1 - i]
            if i > 0:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), e)
        return self.out(h).transpose(1, 2)


class Diffusion:
    """Cosine-schedule DDPM training and deterministic DDIM sampling, x0-parameterised."""

    def __init__(self, steps: int = 1000):
        s = 0.008
        x = np.linspace(0, steps, steps + 1)
        f = np.cos((x / steps + s) / (1 + s) * np.pi / 2) ** 2
        abar = np.clip(f[1:] / f[0], 1e-5, 1.0)
        self.steps = steps
        self.abar = torch.tensor(abar, dtype=torch.float32)

    def loss(self, model, x0, cond, use_ctx, generator: Optional["torch.Generator"] = None):
        """MSE on the part of the window the model has to generate. ``use_ctx`` (B,) bool: the first CONTEXT samples
        are given (and then not part of the loss)."""
        b, w, _ = x0.shape
        dev = x0.device
        t = torch.randint(0, self.steps, (b,), device="cpu", generator=generator).to(dev)
        noise = torch.randn(x0.shape, generator=generator).to(dev)
        a = self.abar.to(dev)[t][:, None, None]
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
        mask = (use_ctx[:, None] & (torch.arange(w, device=dev)[None] < CONTEXT)).float()
        pred = model(x_t, t, cond, x0, mask)
        err = ((pred - x0) ** 2).mean(-1) * (1 - mask)
        return err.sum() / (1 - mask).sum()

    @torch.no_grad()
    def sample(self, model, cond, ctx=None, ctx_len: int = 0, steps: int = 50,
               generator: Optional["torch.Generator"] = None):
        """(B, W, 6) residual windows. ``ctx`` (B, W, 6): its first ``ctx_len`` samples are given and kept."""
        b, w, _ = cond.shape
        dev = cond.device
        ctx = torch.zeros(b, w, N_RESIDUAL, device=dev) if ctx is None else ctx
        mask = torch.zeros(b, w, device=dev)
        mask[:, :ctx_len] = 1.0
        x = torch.randn((b, w, N_RESIDUAL), generator=generator).to(dev)
        abar = self.abar.to(dev)
        ts = np.linspace(self.steps - 1, 0, steps).round().astype(int)
        for k, ti in enumerate(ts):
            t = torch.full((b,), int(ti), device=dev, dtype=torch.long)
            x0 = model(x, t, cond, ctx, mask).clamp(-6, 6)
            if ctx_len:                          # the loss never trained the context part: use the given context
                x0[:, :ctx_len] = ctx[:, :ctx_len]
            a = abar[ti]
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            a_prev = abar[ts[k + 1]] if k + 1 < len(ts) else torch.tensor(1.0, device=dev)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        if ctx_len:
            x[:, :ctx_len] = ctx[:, :ctx_len]
        return x
