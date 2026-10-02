"""motion: the diffusion network and sampler (needs torch)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from backend.motion import require_torch
from backend.motion.dataset import CONTEXT, WINDOW

torch = pytest.importorskip("torch")
from backend.motion.model import Diffusion, MotionUNet  # noqa: E402


def tiny():
    return MotionUNet(base=16, mults=(1, 2))


def test_torch_is_required_with_a_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)             # `import torch` now fails
    with pytest.raises(ImportError, match=r"pip install -e \.\[torch\]"):
        require_torch()


def test_network_shapes_and_size():
    assert 2e6 < sum(p.numel() for p in MotionUNet().parameters()) < 6e6          # the default is a few million parameters
    out = tiny()(torch.randn(3, WINDOW, 6), torch.zeros(3, dtype=torch.long), torch.randn(3, WINDOW, 3),
                 torch.randn(3, WINDOW, 6), torch.zeros(3, WINDOW))
    assert out.shape == (3, WINDOW, 6)


def test_noise_schedule_goes_from_clean_to_noise():
    abar = Diffusion(200).abar.numpy()
    assert abar[0] > 0.99 and abar[-1] < 1e-3 and np.all(np.diff(abar) <= 0)


def test_loss_ignores_the_context_part_of_the_window():
    class Zero(torch.nn.Module):
        def forward(self, x_t, t, cond, ctx, mask):
            return torch.zeros_like(x_t)

    x0 = torch.ones(2, WINDOW, 6)
    x0[:, :CONTEXT] = 100.0
    cond = torch.zeros(2, WINDOW, 3)
    diffusion = Diffusion(50)
    assert diffusion.loss(Zero(), x0, cond, torch.tensor([True, True])).item() == pytest.approx(1.0)
    assert diffusion.loss(Zero(), x0, cond, torch.tensor([False, False])).item() > 1000.0


def test_sampling_is_reproducible_and_keeps_the_context():
    model, diffusion = tiny().eval(), Diffusion(100)
    cond, ctx = torch.randn(2, WINDOW, 3), torch.randn(2, WINDOW, 6)

    def sample(seed):
        return diffusion.sample(model, cond, ctx, CONTEXT, steps=4, generator=torch.Generator().manual_seed(seed))

    a = sample(5)
    assert torch.isfinite(a).all() and a.shape == (2, WINDOW, 6)
    assert torch.equal(a, sample(5)) and not torch.equal(a, sample(6))
    assert torch.equal(a[:, :CONTEXT], ctx[:, :CONTEXT])                      # given samples are kept as they are
    assert not torch.equal(a[:, CONTEXT:], ctx[:, CONTEXT:])
    free = diffusion.sample(model, cond, steps=4, generator=torch.Generator().manual_seed(5))   # no context at all
    assert free.shape == a.shape


def test_the_network_can_learn_a_fixed_batch():
    torch.manual_seed(0)
    model, diffusion = tiny(), Diffusion(100)
    x0 = torch.sin(torch.linspace(0, 20, WINDOW))[None, :, None].repeat(4, 1, 6)
    cond, no_ctx = torch.zeros(4, WINDOW, 3), torch.zeros(4, dtype=torch.bool)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(80):
        loss = diffusion.loss(model, x0, cond, no_ctx, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.5 * np.mean(losses[:10])
