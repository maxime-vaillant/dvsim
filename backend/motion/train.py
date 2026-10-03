"""Training loop and checkpoints. Checkpoints hold tensors, numbers and strings only, so ``torch.load`` can run with
``weights_only=True`` (a checkpoint from someone else cannot execute code)."""
from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence as Seq

import numpy as np

from backend.motion.dataset import Sequence, Stats, WindowSampler
from backend.motion.model import Diffusion, MotionUNet, torch

CHECKPOINT_VERSION = "dvsim-motion-1"


@dataclass
class TrainConfig:
    steps: int = 20000
    batch: int = 64
    lr: float = 2e-4
    ema: float = 0.999
    base: int = 64
    mults: tuple = (1, 2, 4)
    diffusion_steps: int = 1000
    ctx_prob: float = 0.5
    hop: int = 25
    seed: int = 0
    val_every: int = 1000
    val_batches: int = 16
    # how the corpus was split (``motion-eval`` recomputes the held-out recordings from these)
    val_frac: float = 0.1
    test_frac: float = 0.1
    split_seed: int = 0
    holdout_sources: tuple = ()


def pick_device(name: Optional[str] = None) -> "torch.device":
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ema_update(ema, model, decay: float) -> None:
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(decay).add_(pm.detach(), alpha=1.0 - decay)


def _batch(sampler: WindowSampler, rng, size: int, device, ctx_prob: float):
    b = sampler.batch(rng, size)
    x0 = torch.from_numpy(b["residual"]).to(device)
    cond = torch.from_numpy(b["cond"]).to(device)
    use_ctx = torch.from_numpy(rng.random(size) < ctx_prob).to(device)
    return x0, cond, use_ctx


def save_checkpoint(path, model, cfg: TrainConfig, stats: Stats, sources: Seq[str], step: int) -> None:
    """Written to a temporary file of the same folder, then renamed: ``path`` is never a half-written checkpoint."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save({
            "version": CHECKPOINT_VERSION,
            "config": json.dumps(asdict(cfg)),
            "stats": json.dumps(stats.to_dict()),
            "sources": json.dumps(sorted(set(sources))),
            "step": int(step),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }, tmp)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def load_checkpoint(path, device=None):
    """(model in eval mode, TrainConfig, Stats, sources)."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # unpickling errors, truncated files, not a torch file
        raise ValueError(f"{path}: cannot read this file as a motion checkpoint ({exc})") from None
    if not isinstance(ckpt, dict) or ckpt.get("version") != CHECKPOINT_VERSION:
        found = ckpt.get("version") if isinstance(ckpt, dict) else type(ckpt).__name__
        raise ValueError(f"{path}: not a {CHECKPOINT_VERSION} checkpoint (found {found!r})")
    try:
        raw = json.loads(ckpt["config"])
        cfg = TrainConfig(**{**raw, "mults": tuple(raw["mults"]),
                             "holdout_sources": tuple(raw.get("holdout_sources", ()))})
        stats, sources = Stats.from_dict(json.loads(ckpt["stats"])), json.loads(ckpt["sources"])
        model = MotionUNet(cfg.base, cfg.mults)
        model.load_state_dict(ckpt["state_dict"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{path}: the checkpoint is incomplete or does not match its own configuration "
                         f"({exc})") from None
    model.to(device or pick_device()).eval()
    return model, cfg, stats, sources


def train(train_seqs: Seq[Sequence], val_seqs: Seq[Sequence], stats: Stats, cfg: TrainConfig, out_path,
          device=None, init: Optional[str] = None, log: Callable[[str], None] = print) -> List[Dict[str, float]]:
    """Train and save the EMA weights to ``out_path`` at every validation and when interrupted. With ``init`` the
    training continues from that checkpoint, in its normalisation (``stats`` is then ignored). Returns the
    validation history [{step, val_loss}]."""
    device = pick_device(device) if not isinstance(device, torch.device) else device
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    prev = None
    if init:
        prev, prev_cfg, stats, _ = load_checkpoint(init, device)
        if (prev_cfg.base, prev_cfg.mults) != (cfg.base, cfg.mults):
            raise ValueError(f"--init model has base={prev_cfg.base} mults={prev_cfg.mults}, "
                             f"expected {cfg.base} {cfg.mults}")
        log("--init: training in the normalisation of the initial model (its statistics replace the corpus')")
    sampler = WindowSampler(train_seqs, stats, hop=cfg.hop)
    val_sampler = WindowSampler(val_seqs, stats, hop=cfg.hop) if val_seqs else None
    model = MotionUNet(cfg.base, cfg.mults).to(device)
    if prev is not None:
        model.load_state_dict(prev.state_dict())
    ema = copy.deepcopy(model).eval()
    diff = Diffusion(cfg.diffusion_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    history: List[Dict[str, float]] = []
    run = 0.0
    done = 0                                     # the last step whose update reached the EMA weights
    sources = [s.source for s in train_seqs]
    try:
        for step in range(1, cfg.steps + 1):
            x0, cond, use_ctx = _batch(sampler, rng, cfg.batch, device, cfg.ctx_prob)
            loss = diff.loss(model, x0, cond, use_ctx, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            _ema_update(ema, model, min(cfg.ema, (1 + step) / (10 + step)))
            done = step
            run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
            if step % cfg.val_every == 0 or step == cfg.steps:
                msg = f"step {step}/{cfg.steps} train {run:.4f}"
                if val_sampler is not None:
                    vrng, vgen = np.random.default_rng(1), torch.Generator().manual_seed(1)
                    with torch.no_grad():
                        v = np.mean([diff.loss(ema, *_batch(val_sampler, vrng, cfg.batch, device, 0.5), vgen).item()
                                     for _ in range(cfg.val_batches)])
                    history.append({"step": step, "val_loss": float(v)})
                    msg += f" val {v:.4f}"
                log(msg)
                save_checkpoint(out_path, ema, cfg, stats, sources, step)
    except KeyboardInterrupt:
        save_checkpoint(out_path, ema, cfg, stats, sources, done)
        raise
    if done == 0:                                # steps=0: the initial weights, as before
        save_checkpoint(out_path, ema, cfg, stats, sources, 0)
    return history
