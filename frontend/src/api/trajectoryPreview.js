// Asks the simulator to generate trajectories on the loaded scene without simulating them, so the previewer can
// draw and play them first. DOM free (unit-tested in node).

import * as api from './client.js';
import { ensureScene } from './jobRunner.js';
import { withPlaybackTimes } from '../utils/trajectoryRecords.js';

/**
 * files  { obj, texture } as for JobRunner.run;  spec  a generated trajectory spec (buildTrajectorySpec).
 * Resolves to { seed, paths: [{ records, durationS }] }: the records carry normalizedTimestamp for playback, and
 * `seed` is the one the server used (it draws one when spec.seed is null): the same spec with that seed simulates
 * exactly these paths.
 */
export async function generatePreview({ files, spec, sceneStore, client = api, signal, maxPoints = 2000 }) {
    const call = scene => client.generateTrajectories(
        { scene_id: scene.id, trajectory: spec, max_points: maxPoints }, { signal }
    );

    let scene = await ensureScene({ files, sceneStore, client, signal });
    let result;

    try {
        result = await call(scene);
    } catch (error) {
        // The server may have lost a scene we re-used: upload it again once.
        if (scene.uploaded || error.status !== 404) throw error;

        scene = await ensureScene({ files, sceneStore, client, signal, forceUpload: true });
        result = await call(scene);
    }

    return {
        seed: result.seed,
        paths: result.trajectories.map(trajectory => withPlaybackTimes(trajectory.points))
    };
}
