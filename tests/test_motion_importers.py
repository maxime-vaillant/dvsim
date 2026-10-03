"""motion: reading trajectory files of each format, directory discovery, and the import report."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from backend.motion.corpus import MIN_AGREEMENT, import_source, load_corpus, save_corpus
from backend.motion.importers import FORMATS, read
from backend.motion.registry import MODEL_ID, list_models, resolve_model
from backend.motion.tracks import FrameSpec

N = 1200
T = np.arange(N) * 0.0333                                                    # 30 Hz, 40 s
POS = np.c_[T * 1.3, np.zeros(N), np.full(N, 1.6)]                           # walks along +x in a z-up world
QUAT = Rotation.from_matrix(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]])).as_quat()   # optical camera looks +x


def write_tum(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    lines += [f"{t + 1000:.6f} {p[0]} {p[1]} {p[2]} {QUAT[0]} {QUAT[1]} {QUAT[2]} {QUAT[3]}" for t, p in zip(T, POS)]
    path.write_text("\n".join(lines) + "\n")


def test_tum_file(tmp_path):
    write_tum(tmp_path / "groundtruth.txt")
    (raw,) = read(tmp_path / "groundtruth.txt", "tum")
    assert raw.name == "groundtruth" and len(raw.times_us) == N
    assert raw.times_us[0] == 1_000_000_000 and raw.frame == FrameSpec(up=(0, 0, 1), camera="opencv")
    (forced,) = read(tmp_path, "tum", up=(0, 1, 0), camera="opengl")
    assert forced.frame == FrameSpec(up=(0, 1, 0), camera="opengl")
    tracks, report = import_source("tumrgbd", tmp_path, "tum")
    assert report.files == 1 and report.segments == 1 and report.agreement == pytest.approx(1.0)
    assert report.hours == pytest.approx(39.96 / 3600, rel=0.01) and report.warnings == []


def test_tumvi_csv_has_nanoseconds_and_w_first(tmp_path):
    lines = ["#timestamp [ns],px,py,pz,qw,qx,qy,qz"]
    lines += [f"{int(t * 1e9)},{p[0]},{p[1]},{p[2]},{QUAT[3]},{QUAT[0]},{QUAT[1]},{QUAT[2]}" for t, p in zip(T, POS)]
    (tmp_path / "data.csv").write_text("\n".join(lines))
    _, report = import_source("tumvi", tmp_path / "data.csv", "csv", layout="tumvi")
    assert report.agreement == pytest.approx(1.0)


def test_aria_takes_up_from_gravity(tmp_path):
    head = ("graph_uid,tracking_timestamp_us,tx_world_device,ty_world_device,tz_world_device,qx_world_device,"
            "qy_world_device,qz_world_device,qw_world_device,gravity_x_world,gravity_y_world,gravity_z_world")
    rows = [f"g,{int(t * 1e6)},{p[0]},{p[1]},{p[2]},{QUAT[0]},{QUAT[1]},{QUAT[2]},{QUAT[3]},0,0,-9.81"
            for t, p in zip(T, POS)]
    (tmp_path / "closed_loop_trajectory.csv").write_text("\n".join([head] + rows))
    (raw,) = read(tmp_path, "aria")
    assert raw.frame.up == pytest.approx((0, 0, 1))
    assert import_source("aria", tmp_path, "aria")[1].agreement == pytest.approx(1.0)


def test_headset_json_is_already_in_dvsim_frame(tmp_path):
    records = [{"timestamp": t * 1000, "x": -1.3 * t, "y": 1.6, "z": 0.0, "rx": -1.0, "ry": 0.0, "rz": 0.0} for t in T]
    (tmp_path / "walk.json").write_text(json.dumps(records))
    assert import_source("headset", tmp_path / "walk.json", "headset")[1].agreement == pytest.approx(1.0)


def test_directories_are_searched_recursively(tmp_path):
    write_tum(tmp_path / "a" / "groundtruth.txt")
    write_tum(tmp_path / "b" / "groundtruth.txt")
    assert sorted(r.name for r in read(tmp_path, "tum")) == ["a/groundtruth", "b/groundtruth"]
    write_tum(tmp_path / "loose" / "run1.txt")                               # no groundtruth.txt: any .txt
    assert [r.name for r in read(tmp_path / "loose", "tum")] == ["run1"]


def test_a_wrong_frame_is_reported(tmp_path):
    write_tum(tmp_path / "groundtruth.txt")
    _, report = import_source("bad", tmp_path, "tum", camera="opengl")
    assert report.agreement == pytest.approx(-1.0)
    assert len(report.warnings) == 1 and "frame is probably wrong" in report.warnings[0] and MIN_AGREEMENT == 0.5


def test_a_recording_without_a_long_enough_segment_is_reported(tmp_path):
    lines = [f"{t:.3f} {t} 0 1.6 {QUAT[0]} {QUAT[1]} {QUAT[2]} {QUAT[3]}" for t in np.arange(0, 2, 0.05)]
    (tmp_path / "short.txt").write_text("\n".join(lines))
    tracks, report = import_source("s", tmp_path / "short.txt", "tum")
    assert tracks == [] and report.segments == 0 and "no segment of at least 4 s" in report.warnings[0]


def test_errors_name_the_file_and_line(tmp_path):
    (tmp_path / "bad.txt").write_text("# header\n1 2 3\n")
    with pytest.raises(ValueError, match=r"bad\.txt:2: expected 8 numbers"):
        read(tmp_path / "bad.txt", "tum")
    (tmp_path / "bad.csv").write_text("h\n1,2,x,4,5,6,7,8\n")
    with pytest.raises(ValueError, match=r"bad\.csv:2"):
        read(tmp_path / "bad.csv", "csv")
    (tmp_path / "aria.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing column"):
        read(tmp_path / "aria.csv", "aria")
    with pytest.raises(ValueError, match="unknown format"):
        read(tmp_path, "rosbag")
    with pytest.raises(ValueError, match="unknown csv layout"):
        read(tmp_path / "bad.csv", "csv", "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no tum trajectory file"):
        read(empty, "tum")
    with pytest.raises(FileNotFoundError):
        read(tmp_path / "missing", "tum")
    assert FORMATS == ("tum", "aria", "headset", "csv")


def test_a_corpus_survives_saving(tmp_path):
    write_tum(tmp_path / "groundtruth.txt")
    tracks, _ = import_source("tumrgbd", tmp_path, "tum")
    save_corpus(tmp_path / "corpus.npz", tracks)
    (back,) = load_corpus(tmp_path / "corpus.npz")
    assert (back.source, back.name) == (tracks[0].source, tracks[0].name)
    assert np.array_equal(back.times_us, tracks[0].times_us) and np.allclose(back.positions, tracks[0].positions)


def test_registry(tmp_path):
    folder = tmp_path / "motion"
    folder.mkdir()
    (folder / "walk-v1.pt").write_bytes(b"x")
    (folder / "notes.txt").write_bytes(b"x")
    (folder / ".hidden.pt").write_bytes(b"x")
    assert resolve_model({"model": "walk-v1"}, tmp_path) == folder / "walk-v1.pt"
    assert [m["id"] for m in list_models(tmp_path)] == ["walk-v1"]
    assert list_models(tmp_path / "none") == []
    assert resolve_model({"path": str(folder / "walk-v1.pt")}) == folder / "walk-v1.pt"
    for bad in ("../walk-v1", "a/b", "", ".x", "x" * 65, None, 3, "walk-v1\n"):
        with pytest.raises(ValueError, match="invalid motion model id"):
            resolve_model({"model": bad}, tmp_path)
    with pytest.raises(FileNotFoundError, match="'other' not found"):
        resolve_model({"model": "other"}, tmp_path)
    with pytest.raises(FileNotFoundError, match="file not found"):
        resolve_model({"path": str(folder / "gone.pt")})
    assert MODEL_ID.match("walk_v1.2")
    (folder / "trail\n.pt").write_bytes(b"x")                              # "$" alone would accept the newline
    assert [m["id"] for m in list_models(tmp_path)] == ["walk-v1"]


def _tumvi_rows():
    lines = ["#timestamp [ns],px,py,pz,qw,qx,qy,qz"]
    return lines + [f"{int(t * 1e9)},{p[0]},{p[1]},{p[2]},{QUAT[3]},{QUAT[0]},{QUAT[1]},{QUAT[2]}" for t, p in zip(T, POS)]


def test_a_tumvi_tree_imports_only_the_mocap_files(tmp_path):
    for seq in ("room1", "room2"):
        mav = tmp_path / seq / "mav0"
        for sensor in ("cam0", "imu0", "mocap0"):
            (mav / sensor).mkdir(parents=True)
        (mav / "cam0" / "data.csv").write_text("#timestamp [ns],filename\n1,1.png\n2,2.png\n")
        (mav / "imu0" / "data.csv").write_text("#timestamp [ns],w_x,w_y,w_z\n1,0,0,0\n")
        (mav / "mocap0" / "data.csv").write_text("\n".join(_tumvi_rows()))
    raws = read(tmp_path, "csv")
    assert sorted(r.name for r in raws) == ["room1/mav0/mocap0/data", "room2/mav0/mocap0/data"]
    tracks, report = import_source("tumvi", tmp_path, "csv")
    assert report.files == 2 and report.segments == 2 and report.agreement == pytest.approx(1.0)


def test_headset_errors_name_the_file(tmp_path):
    (tmp_path / "broken.json").write_text("[{")
    with pytest.raises(ValueError, match=r"broken\.json: "):
        read(tmp_path / "broken.json", "headset")
    (tmp_path / "partial.json").write_text(json.dumps([{"timestamp": 0, "x": 0}, {"timestamp": 1, "x": 0}]))
    with pytest.raises(ValueError, match=r"partial\.json: record 0: missing y, z"):
        read(tmp_path / "partial.json", "headset")


def test_a_corpus_path_without_npz_is_found(tmp_path):
    write_tum(tmp_path / "groundtruth.txt")
    tracks, _ = import_source("tumrgbd", tmp_path, "tum")
    assert save_corpus(tmp_path / "corpus", tracks) == tmp_path / "corpus.npz"
    assert (tmp_path / "corpus.npz").is_file() and not (tmp_path / "corpus").exists()
    assert len(load_corpus(tmp_path / "corpus")) == 1 and len(load_corpus(tmp_path / "corpus.npz")) == 1
