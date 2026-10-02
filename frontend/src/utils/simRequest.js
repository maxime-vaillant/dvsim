// Simulation settings (what the GUI edits) -> the JSON body of POST /api/jobs. Pure functions.

import { pickRecord } from './trajectoryRecords.js';
import { ORIGIN_MODES } from './trajectoryOrigin.js';
import { sampleConstantSpeed } from './trajectorySampling.js';

export const GT_FIELDS = ['rgb', 'depth', 'flow', 'pose'];
export const GENERATED_KINDS = ['rectilinear', 'spline', 'street'];


export const DEFAULT_SIM_SETTINGS = {
    // camera / events / output
    width: 346,
    height: 260,
    fov: 60,
    dt_us: 10000,
    c_pos: 0.2,
    c_neg: 0.2,
    stereo: false,
    rgb: true,
    depth: true,
    flow: true,
    pose: true,
    gt_every: 1,
    visualize: false,

    // trajectory
    speed: 0.8,
    genKind: 'rectilinear',
    genDuration: 5,
    genCount: 1,
    genKeyPoints: 6,
    genSeed: '',

    // imported / live positions are relative to ...
    originMode: 'Object centre'
};

/** Copies the values of `saved` that have the right type into `target` (stored data may be stale or hand-edited). */
export function restoreSimSettings(target, saved) {
    if (!saved || typeof saved !== 'object') return target;

    Object.keys(DEFAULT_SIM_SETTINGS).forEach(key => {
        const value = saved[key];

        if (typeof value !== typeof DEFAULT_SIM_SETTINGS[key]) return;
        if (typeof value === 'number' && !Number.isFinite(value)) return;
        if (key === 'genKind' && !GENERATED_KINDS.includes(value)) return;
        if (key === 'originMode' && !ORIGIN_MODES.includes(value)) return;

        target[key] = value;
    });

    return target;
}

function requireInteger(value, name, min) {
    if (!Number.isInteger(value) || value < min) throw new Error(`${name} must be an integer >= ${min}.`);

    return value;
}

function requirePositive(value, name) {
    if (!(Number.isFinite(value) && value > 0)) throw new Error(`${name} must be a positive number.`);

    return value;
}

/** Empty -> null (the server picks a random seed); otherwise an integer. */
export function parseSeed(text) {
    const trimmed = String(text === undefined || text === null ? '' : text).trim();

    if (trimmed === '') return null;
    if (!/^-?\d+$/.test(trimmed)) throw new Error('Seed must be an integer (or empty for a random seed).');

    return Number(trimmed);
}

/** The partial SimConfig dict of the API. Throws an Error with a user-facing message when a value is invalid. */
export function buildSimConfig(sim) {
    const fov = Number(sim.fov);

    if (!(fov > 0 && fov < 180)) throw new Error('Camera FOV must be between 0 and 180 degrees.');

    return {
        camera: {
            width: requireInteger(sim.width, 'Width', 1),
            height: requireInteger(sim.height, 'Height', 1),
            fov_y_deg: fov,
            stereo: Boolean(sim.stereo)
        },
        events: {
            c_pos: requirePositive(sim.c_pos, 'c_pos'),
            c_neg: requirePositive(sim.c_neg, 'c_neg')
        },
        dt_us: requireInteger(sim.dt_us, 'dt_us', 1),
        gt_every: requireInteger(sim.gt_every, 'gt_every', 1),
        gt: GT_FIELDS.filter(field => sim[field])
    };
}

/**
 * The `trajectory` member of the job request.
 *
 * @param type  the Trajectory > Type dropdown value
 * @param context.curve    the drawn curve (THREE.CatmullRomCurve3) or null
 * @param context.records  imported legacy records or null
 * @param context.origin   {x, y, z} the origin the previewer offsets imported records by (see trajectoryOrigin), or null
 * @param context.sim      the simulation settings
 */
export function buildTrajectorySpec(type, { curve = null, records = null, origin = null, sim }) {
    switch (type) {
        case 'Custom':
            if (!curve) throw new Error('Draw a trajectory first (left-drag on the overview, top-left).');

            // Drawn positions are absolute world coordinates.
            return { type: 'inline', points: sampleConstantSpeed(curve, { speed: Number(sim.speed) }), origin: null };

        case 'Imported':
            if (!records || records.length < 2) throw new Error('Load a trajectory file first (Loading > Load Trajectory).');
            if (!origin) throw new Error('Load a scene first (Loading > Load scene): imported trajectories are relative to it.');

            return { type: 'inline', points: records.map(pickRecord), origin: [origin.x, origin.y, origin.z] };

        case 'Generated': {
            const spec = {
                type: sim.genKind,
                duration_s: requirePositive(sim.genDuration, 'Duration'),
                count: requireInteger(sim.genCount, 'Count', 1),
                seed: parseSeed(sim.genSeed)
            };

            if (!GENERATED_KINDS.includes(spec.type)) throw new Error(`Unknown generator "${spec.type}".`);
            if (spec.type === 'spline') spec.n_key_points = requireInteger(sim.genKeyPoints, 'Key points', 2);

            return spec;
        }

        default:
            throw new Error('Live trajectories cannot be simulated: choose Custom, Imported or Generated.');
    }
}

/** { trajectory, config, visualize }: everything of POST /api/jobs except scene_id. */
export function buildJobRequest(type, context) {
    return {
        trajectory: buildTrajectorySpec(type, context),
        config: buildSimConfig(context.sim),
        visualize: Boolean(context.sim.visualize)
    };
}
