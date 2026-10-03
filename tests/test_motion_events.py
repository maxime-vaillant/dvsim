"""motion: the event-statistics report on the demo scene (needs OpenGL and torch)."""
from __future__ import annotations

import numpy as np
import pytest

from backend.motion.events_report import compare_event_statistics, event_statistics, format_event_report
from backend.scene import load_scene
from backend.trajectories import WaypointTrajectory

pytestmark = pytest.mark.usefixtures("gl_context")
STREET = {"duration_s": 2.0, "seed": 1, "min_clearance": 0.5, "edge_margin": 0}


@pytest.fixture(scope="module")
def scene(demo_scene):
    return load_scene(demo_scene.scene_spec)


def test_statistics_of_a_known_trajectory(scene, demo_scene):
    stats = event_statistics(scene, WaypointTrajectory.from_json(demo_scene.trajectory_json))
    assert set(stats) == {"events_per_s", "events_per_step_p50", "events_per_step_p95", "flow_px_s_mean", "flow_px_s_p95"}
    assert stats["events_per_s"] > 0 and stats["flow_px_s_mean"] > 0 and stats["flow_px_s_p95"] >= stats["flow_px_s_mean"]


def test_plain_and_human_walks_are_compared_on_the_same_planned_path(scene, motion_checkpoint):
    pytest.importorskip("torch")
    report = compare_event_statistics(scene, {"path": str(motion_checkpoint)}, STREET)
    assert set(report) == {"plain", "human"}
    for stats in report.values():
        assert all(np.isfinite(v) for v in stats.values()) and stats["events_per_s"] > 0
    assert report["plain"] != report["human"]
    text = format_event_report(report)
    assert "events / s" in text and "plain" in text and "human" in text
