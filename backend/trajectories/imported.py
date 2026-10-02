from __future__ import annotations

import os
from typing import Any

from backend.trajectories.trajectory import WaypointTrajectory, _parse_records


class ImportedTrajectory(WaypointTrajectory):
    """Legacy name for ``WaypointTrajectory.from_json``: ``ImportedTrajectory(path_or_records, origin=(0, 0, 0))``.

    A one-element list of paths is accepted as before; the multi-path ``get_next_trajectory`` protocol is gone
    (build one trajectory per file).
    """

    def __init__(self, source: Any, origin=(0.0, 0.0, 0.0), on_duplicate: str = "raise"):
        if isinstance(source, (list, tuple)) and source and isinstance(source[0], (str, os.PathLike)):
            if len(source) != 1:
                raise ValueError("ImportedTrajectory holds one path; create one trajectory per file")
            source = source[0]
        super().__init__(*_parse_records(source, origin, on_duplicate))
