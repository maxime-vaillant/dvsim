"""motion: training, checkpoints and what loading one accepts (needs torch)."""
from __future__ import annotations

import json

import pytest
from motion_helpers import synthetic_walk

torch = pytest.importorskip("torch")
from backend.motion.dataset import Stats, build_sequences  # noqa: E402
from backend.motion.train import CHECKPOINT_VERSION, TrainConfig, load_checkpoint, train  # noqa: E402

SEQS = build_sequences([synthetic_walk(i) for i in range(4)])
TINY = dict(batch=8, base=16, mults=(1, 2), diffusion_steps=100, val_batches=1)


def test_the_fixture_checkpoint_holds_only_plain_data(motion_checkpoint):
    ckpt = torch.load(motion_checkpoint, weights_only=True)                  # would raise on anything executable
    assert ckpt["version"] == CHECKPOINT_VERSION and ckpt["step"] == 120
    assert json.loads(ckpt["sources"]) == ["syn"] and json.loads(ckpt["config"])["base"] == 16
    assert all(isinstance(v, torch.Tensor) for v in ckpt["state_dict"].values())


def test_load_gives_an_eval_model_and_its_training_setup(motion_checkpoint):
    model, cfg, stats, sources = load_checkpoint(motion_checkpoint, "cpu")
    assert not model.training and (cfg.base, cfg.mults) == (16, (1, 2)) and sources == ["syn"]
    assert 0.5 < stats.speed_range[0] < stats.speed_range[1] < 2.0


def test_training_reports_the_validation_loss(tmp_path):
    cfg = TrainConfig(steps=20, val_every=10, **TINY)
    lines = []
    history = train(SEQS[:3], SEQS[3:], Stats.fit(SEQS[:3]), cfg, tmp_path / "m.pt", device="cpu", log=lines.append)
    assert [h["step"] for h in history] == [10, 20] and all(h["val_loss"] > 0 for h in history)
    assert len(lines) == 2 and "val" in lines[0] and (tmp_path / "m.pt").is_file()


def test_the_same_seed_trains_the_same_model(tmp_path):
    cfg = TrainConfig(steps=10, val_every=10, **TINY)
    for name in ("a.pt", "b.pt"):
        train(SEQS, [], Stats.fit(SEQS), cfg, tmp_path / name, device="cpu", log=lambda m: None)
    a, b = (torch.load(tmp_path / n, weights_only=True)["state_dict"] for n in ("a.pt", "b.pt"))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_training_can_continue_from_a_checkpoint(motion_checkpoint, tmp_path):
    cfg = TrainConfig(steps=5, val_every=5, **TINY)
    train(SEQS, [], Stats.fit(SEQS), cfg, tmp_path / "next.pt", device="cpu", init=str(motion_checkpoint), log=lambda m: None)
    assert load_checkpoint(tmp_path / "next.pt", "cpu")[1].steps == 5
    other = TrainConfig(steps=5, **{**TINY, "base": 32})
    with pytest.raises(ValueError, match="--init model has base=16"):
        train(SEQS, [], Stats.fit(SEQS), other, tmp_path / "x.pt", device="cpu", init=str(motion_checkpoint))


class _NotATensor:
    pass


def test_bad_files_are_refused_cleanly(tmp_path):
    torch.save({"version": "other"}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="not a dvsim-motion-1 checkpoint"):
        load_checkpoint(tmp_path / "old.pt", "cpu")
    torch.save(_NotATensor(), tmp_path / "pickle.pt")                         # arbitrary objects are never unpickled
    with pytest.raises(ValueError, match="cannot read this file"):
        load_checkpoint(tmp_path / "pickle.pt", "cpu")
    (tmp_path / "junk.pt").write_bytes(b"not a torch file")
    with pytest.raises(ValueError, match="cannot read this file"):
        load_checkpoint(tmp_path / "junk.pt", "cpu")


def _edited_copy(src, dst, edit):
    ckpt = torch.load(src, weights_only=True)
    edit(ckpt)
    torch.save(ckpt, dst)
    return dst


def _set_config(**changes):
    def edit(ckpt):
        ckpt["config"] = json.dumps({**json.loads(ckpt["config"]), **changes})
    return edit


@pytest.mark.parametrize("name, edit", [
    ("nostats", lambda c: c.pop("stats")),
    ("badjson", lambda c: c.__setitem__("config", "{")),
    ("unknown", _set_config(nonsense=1)),
    ("wider", _set_config(base=32)),                                         # the weights no longer fit
])
def test_an_incomplete_or_inconsistent_checkpoint_is_a_clean_error(motion_checkpoint, tmp_path, name, edit):
    path = _edited_copy(motion_checkpoint, tmp_path / f"{name}.pt", edit)
    with pytest.raises(ValueError, match=f"{name}\\.pt: the checkpoint is incomplete"):
        load_checkpoint(path, "cpu")


def test_checkpoints_are_saved_periodically_and_atomically(tmp_path):
    cfg = TrainConfig(steps=20, val_every=10, **TINY)
    train(SEQS, [], Stats.fit(SEQS), cfg, tmp_path / "m.pt", device="cpu", log=lambda m: None)
    assert torch.load(tmp_path / "m.pt", weights_only=True)["step"] == 20
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.pt"]           # no temporary file left behind

    calls = []

    def crash(message):
        calls.append(message)
        if len(calls) == 2:
            raise RuntimeError("power cut")
    with pytest.raises(RuntimeError, match="power cut"):
        train(SEQS, [], Stats.fit(SEQS), cfg, tmp_path / "crash.pt", device="cpu", log=crash)
    assert load_checkpoint(tmp_path / "crash.pt", "cpu")[1].steps == 20
    assert torch.load(tmp_path / "crash.pt", weights_only=True)["step"] == 10   # the save of step 10 survived


def test_an_interrupted_training_saves_where_it_stopped(tmp_path):
    def stop(message):
        raise KeyboardInterrupt
    cfg = TrainConfig(steps=20, val_every=10, **TINY)
    with pytest.raises(KeyboardInterrupt):
        train(SEQS, [], Stats.fit(SEQS), cfg, tmp_path / "m.pt", device="cpu", log=stop)
    load_checkpoint(tmp_path / "m.pt", "cpu")
    assert torch.load(tmp_path / "m.pt", weights_only=True)["step"] == 10
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.pt"]


def test_fine_tuning_keeps_the_normalisation_of_the_initial_model(motion_checkpoint, tmp_path):
    other = build_sequences([synthetic_walk(i + 50, n=900) for i in range(2)])
    cfg = TrainConfig(steps=3, val_every=3, **TINY)
    lines = []
    train(other, [], Stats.fit(other), cfg, tmp_path / "ft.pt", device="cpu", init=str(motion_checkpoint),
          log=lines.append)
    init_stats = load_checkpoint(motion_checkpoint, "cpu")[2]
    assert load_checkpoint(tmp_path / "ft.pt", "cpu")[2].to_dict() == init_stats.to_dict()
    assert init_stats.to_dict() != Stats.fit(other).to_dict() and any("--init" in line for line in lines)


def test_the_split_settings_are_stored_for_the_evaluation(tmp_path):
    cfg = TrainConfig(steps=2, val_every=2, val_frac=0.2, test_frac=0.3, split_seed=7, holdout_sources=("a", "b"), **TINY)
    train(SEQS, [], Stats.fit(SEQS), cfg, tmp_path / "m.pt", device="cpu", log=lambda m: None)
    loaded = load_checkpoint(tmp_path / "m.pt", "cpu")[1]
    assert (loaded.val_frac, loaded.test_frac, loaded.split_seed, loaded.holdout_sources) == (0.2, 0.3, 7, ("a", "b"))
