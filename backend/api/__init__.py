"""HTTP API of DVSim (FastAPI): ``uvicorn backend.api.main:app`` / ``dvsim serve``. Contract: docs/architecture.md.

Beyond the contract: ``GET|DELETE /api/trajectories/{id}`` (detail with points / delete), interactive docs at
``/api/docs``, ``DELETE`` answers 202 + job JSON when it cancels, 204 when it deletes. Errors are ``{"detail": ...}``.

Data root ``$DVSIM_DATA_DIR`` (default ./data), subdirectories created on first use::

    scenes/<id>/{scene.<ext>, texture.<ext>, meta.json}
    trajectories/<id>/{trajectory.json (legacy records, ms), meta.json}
    jobs/<id>/{request.json, job.json, cancel, error.log, output/seq_NNN/...}

Environment (plain variables): DVSIM_DATA_DIR, DVSIM_FRONTEND_DIR (default <repo>/frontend/dist, served at ``/`` when it
exists), DVSIM_MAX_UPLOAD_MB (512), DVSIM_MAX_JOBS (1 concurrent worker), DVSIM_MAX_PENDING (32 queued + running, then
429), DVSIM_CANCEL_GRACE_S (5), DVSIM_CORS_ORIGINS (comma list; none = same origin only).

Jobs: each runs in its own spawned process (``worker.run_job`` -> ``pipeline.run_to_disk``) and reports through
``job.json`` (atomic rewrites, at most every 0.25 s). Cancel = ``cancel`` flag file checked every step, then
``terminate()`` after the grace period. Jobs left queued / running by a previous server run become ``failed``
("server restarted"). One server process per data directory.

Security: ids are server-generated 16-hex strings, validated wherever they reach a path; client filenames are never
used on disk; uploads are extension-whitelisted (.obj .glb .ply / .png .jpg .jpeg), size-limited while streaming and
each lives in a directory of its own. Files that could make the scene loader read elsewhere are refused at upload:
``.gltf`` (external uris), ``.glb`` with a non-``data:`` uri, ``.obj`` containing ``mtllib`` or ``map_*``, ``.ply``
naming a ``TextureFile``. This is a scan, not a sandbox: ``load_scene`` should also restrict itself to the scene's
directory. Downloads resolve the path (no ``..``, absolute paths, backslashes, symlinks), stay inside
``jobs/<id>/output`` and are sent as attachments. There is no authentication: bind to localhost or put a proxy in front.
"""
