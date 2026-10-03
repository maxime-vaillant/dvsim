"""motion: the dvsim motion-import / motion-train / motion-eval commands."""
from __future__ import annotations

import pytest
from motion_helpers import synthetic_walk, write_tum_walk

from backend.cli import MOTION_FORMATS, _motion_ref, main
from backend.motion.corpus import load_corpus, save_corpus
from backend.motion.importers import FORMATS


def test_the_format_list_of_the_cli_matches_the_importers():
    assert MOTION_FORMATS == FORMATS


def test_import_builds_a_corpus_source_by_source(tmp_path, capsys):
    write_tum_walk(tmp_path / "tum" / "groundtruth.txt")
    corpus = tmp_path / "corpus.npz"
    assert main(["motion-import", "tumrgbd", "tum", str(tmp_path / "tum"), "--out", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert "tumrgbd: 1 file(s), 1 segment(s)" in out and "heading agreement 1.00" in out
    assert main(["motion-import", "second", "tum", str(tmp_path / "tum"), "--out", str(corpus)]) == 0
    assert main(["motion-import", "tumrgbd", "tum", str(tmp_path / "tum"), "--out", str(corpus)]) == 0   # replaces
    sources = sorted(t.source for t in load_corpus(corpus))
    assert sources == ["second", "tumrgbd"]


def test_a_wrong_frame_warns_and_an_empty_import_fails(tmp_path, capsys):
    write_tum_walk(tmp_path / "tum" / "groundtruth.txt")
    corpus = tmp_path / "corpus.npz"
    assert main(["motion-import", "x", "tum", str(tmp_path / "tum"), "--camera", "opengl", "--out", str(corpus)]) == 0
    assert "frame is probably wrong" in capsys.readouterr().err
    write_tum_walk(tmp_path / "short" / "groundtruth.txt", seconds=2.0)
    assert main(["motion-import", "s", "tum", str(tmp_path / "short"), "--out", str(tmp_path / "none.npz")]) == 1
    assert "nothing to import" in capsys.readouterr().err and not (tmp_path / "none.npz").exists()


def test_motion_ref_tells_files_from_ids():
    assert _motion_ref("walk-v1") == {"model": "walk-v1"}
    assert _motion_ref("some/dir/walk.pt") == {"path": "some/dir/walk.pt"}


def test_training_and_evaluating_a_model(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    monkeypatch.setenv("DVSIM_DATA_DIR", str(tmp_path / "data"))
    corpus = tmp_path / "corpus.npz"
    save_corpus(corpus, [synthetic_walk(i) for i in range(10, 18)])        # walks 10, 11, 12 and 17 are held out
    train = ["motion-train", "--corpus", str(corpus), "--steps", "20", "--batch", "8", "--base", "16", "--mults", "1", "2",
             "--diffusion-steps", "100", "--val-frac", "0", "--test-frac", "0.5", "--device", "cpu"]
    assert main(train + ["--name", "tiny"]) == 0
    assert "4 training, 0 validation, 4 held-out" in capsys.readouterr().out
    assert (tmp_path / "data" / "motion" / "tiny.pt").is_file()
    assert main(["motion-eval", "--corpus", str(corpus), "--model", "tiny", "--sample-steps", "4", "--device", "cpu",
                 "--max-sequences", "2"]) == 0
    out = capsys.readouterr().out
    assert "spectral distance" in out and "classifier accuracy" in out and "window seams" in out
    with pytest.raises(SystemExit, match="--name ID"):                      # exactly one of --name / --out
        main(train)


def test_motion_needs_the_street_generator(tmp_path, capsys):
    assert main(["run", "--scene", str(tmp_path / "s.obj"), "--generate", "spline", "--motion", "walk", "--out",
                 str(tmp_path / "o")]) == 2
    assert "--motion needs --generate street" in capsys.readouterr().err


def test_bad_inputs_are_reported_not_raised(tmp_path, capsys):
    assert main(["motion-import", "x", "tum", str(tmp_path / "nowhere")]) == 1
    assert "no such file or directory" in capsys.readouterr().err
    save_corpus(tmp_path / "c.npz", [synthetic_walk(0)])
    assert main(["motion-eval", "--corpus", str(tmp_path / "c.npz"), "--model", str(tmp_path / "gone.pt")]) == 1
    assert "motion model file not found" in capsys.readouterr().err
    assert main(["motion-train", "--corpus", str(tmp_path / "c.npz"), "--out", str(tmp_path / "m.pt"), "--test-frac", "0.99",
                 "--val-frac", "0"]) == 1
    assert "no training recording" in capsys.readouterr().err


@pytest.mark.usefixtures("gl_context")
def test_motion_eval_prints_the_event_comparison(demo_scene, motion_checkpoint, tmp_path, capsys):
    corpus = tmp_path / "corpus.npz"
    save_corpus(corpus, [synthetic_walk(i) for i in range(10, 18)])
    assert main(["motion-eval", "--corpus", str(corpus), "--model", str(motion_checkpoint), "--sample-steps", "4",
                 "--device", "cpu", "--max-sequences", "2", "--scene", str(demo_scene.scene_spec),
                 "--events-duration", "2", "--min-clearance", "0.5", "--edge-margin", "0"]) == 0
    out = capsys.readouterr().out
    assert "spectral distance" in out and "events on the scene" in out and "flow speed" in out


@pytest.mark.parametrize("name", ["../x", "a b", "a/b", "walk\n"])
def test_an_invalid_model_name_fails_before_anything_else(tmp_path, monkeypatch, capsys, name):
    monkeypatch.setenv("DVSIM_DATA_DIR", str(tmp_path / "data"))
    # the corpus does not even exist: the name is checked first
    assert main(["motion-train", "--corpus", str(tmp_path / "none.npz"), "--name", name]) == 1
    assert "invalid motion model id" in capsys.readouterr().err
    assert not any(tmp_path.rglob("*.pt")) and not (tmp_path / "data").exists()


def test_the_output_folder_is_made_before_training(tmp_path, capsys):
    save_corpus(tmp_path / "c.npz", [synthetic_walk(0)])
    out = tmp_path / "new" / "dir" / "m.pt"
    assert main(["motion-train", "--corpus", str(tmp_path / "c.npz"), "--out", str(out), "--test-frac", "0.99",
                 "--val-frac", "0"]) == 1                                  # fails after the split, before training
    assert "no training recording" in capsys.readouterr().err and out.parent.is_dir()
    (tmp_path / "afile").write_text("x")
    assert main(["motion-train", "--corpus", str(tmp_path / "c.npz"), "--out", str(tmp_path / "afile" / "m.pt"),
                 "--steps", "2"]) == 1                                     # a folder that cannot exist: no training
    assert "error:" in capsys.readouterr().err and not any(tmp_path.rglob("*.pt"))


def test_a_corpus_named_without_npz_accumulates_sources(tmp_path, capsys):
    write_tum_walk(tmp_path / "tum" / "groundtruth.txt")
    corpus = tmp_path / "corpus"
    assert main(["motion-import", "one", "tum", str(tmp_path / "tum"), "--out", str(corpus)]) == 0
    assert main(["motion-import", "two", "tum", str(tmp_path / "tum"), "--out", str(corpus)]) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last == f"{tmp_path / 'corpus.npz'}: 2 segment(s) from 2 source(s)" and (tmp_path / "corpus.npz").is_file()
    assert sorted(t.source for t in load_corpus(corpus)) == ["one", "two"]


def test_motion_with_a_trajectory_file_is_an_error(tmp_path, capsys):
    assert main(["run", "--scene", str(tmp_path / "s.obj"), "--trajectory", str(tmp_path / "t.json"), "--motion",
                 "walk", "--out", str(tmp_path / "o")]) == 2
    assert "--motion needs --generate street" in capsys.readouterr().err
