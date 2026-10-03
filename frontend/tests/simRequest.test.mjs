// Settings -> POST /api/jobs body.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';

import {
    DEFAULT_SIM_SETTINGS, buildJobRequest, buildSimConfig, buildTrajectorySpec, motionOptions, parseSeed, restoreSimSettings
} from '../src/utils/simRequest.js';

const sim = (overrides = {}) => ({ ...DEFAULT_SIM_SETTINGS, ...overrides });
const curve = new THREE.CatmullRomCurve3([new THREE.Vector3(0, 1, 0), new THREE.Vector3(2, 1, 0), new THREE.Vector3(2, 1, 2)]);

test('defaults match the simulator defaults', () => {
    assert.deepEqual(buildSimConfig(sim()), {
        camera: { width: 346, height: 260, fov_y_deg: 60, stereo: false },
        events: { c_pos: 0.2, c_neg: 0.2 },
        dt_us: 10000,
        gt_every: 1,
        gt: ['rgb', 'depth', 'flow', 'pose']
    });
});

test('ground-truth toggles and numbers are forwarded', () => {
    const config = buildSimConfig(sim({ width: 640, height: 480, fov: 75, stereo: true, c_pos: 0.3, c_neg: 0.15, dt_us: 5000, gt_every: 4, rgb: false, flow: false }));

    assert.deepEqual(config.gt, ['depth', 'pose']);
    assert.equal(config.gt_every, 4);
    assert.deepEqual(config.camera, { width: 640, height: 480, fov_y_deg: 75, stereo: true });
    assert.deepEqual(config.events, { c_pos: 0.3, c_neg: 0.15 });
    assert.deepEqual(buildSimConfig(sim({ rgb: false, depth: false, flow: false, pose: false })).gt, []);
});

test('invalid settings give a readable error', () => {
    assert.throws(() => buildSimConfig(sim({ width: 0 })), /Width/);
    assert.throws(() => buildSimConfig(sim({ height: 12.5 })), /Height/);
    assert.throws(() => buildSimConfig(sim({ fov: 0 })), /FOV/);
    assert.throws(() => buildSimConfig(sim({ fov: 180 })), /FOV/);
    assert.throws(() => buildSimConfig(sim({ dt_us: 0 })), /dt_us/);
    assert.throws(() => buildSimConfig(sim({ gt_every: 0 })), /gt_every/);
    assert.throws(() => buildSimConfig(sim({ c_pos: 0 })), /c_pos/);
    assert.throws(() => buildSimConfig(sim({ c_neg: NaN })), /c_neg/);
});

test('Custom: constant-speed inline points, absolute positions, origin null', () => {
    const spec = buildTrajectorySpec('Custom', { curve, sim: sim({ speed: 0.8 }) });

    assert.equal(spec.type, 'inline');
    assert.equal(spec.origin, null);
    assert.ok(spec.points.length > 10);
    assert.deepEqual(Object.keys(spec.points[0]), ['timestamp', 'x', 'y', 'z', 'rx', 'ry', 'rz']);
    assert.equal(spec.points[0].x, 0);
    assert.ok(Math.abs(spec.points.at(-1).timestamp - curve.getLength() / 0.8 * 1000) < 1e-2);

    assert.throws(() => buildTrajectorySpec('Custom', { curve: null, sim: sim() }), /Draw a trajectory/);
});

test('Imported: inline records with the object centre as origin, preview fields stripped', () => {
    const records = [
        { timestamp: 100, x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: -1, normalizedTimestamp: 0 },
        { timestamp: 300, x: 1, y: 0, z: 0.5, rx: 1, ry: 0, rz: 0, normalizedTimestamp: 1 }
    ];
    const spec = buildTrajectorySpec('Imported', { records, origin: new THREE.Vector3(1.5, 0.25, -2), sim: sim() });

    assert.equal(spec.type, 'inline');
    assert.deepEqual(spec.origin, [1.5, 0.25, -2]);
    assert.equal(spec.points.length, 2);
    assert.equal(spec.points[1].timestamp, 300);
    assert.equal(spec.points[1].normalizedTimestamp, undefined);
    assert.equal(records[1].normalizedTimestamp, 1, 'the preview data is not modified');

    assert.throws(() => buildTrajectorySpec('Imported', { records: [], origin: { x: 0, y: 0, z: 0 }, sim: sim() }), /Load a trajectory/);
    assert.throws(() => buildTrajectorySpec('Imported', { records, origin: null, sim: sim() }), /Load a scene/);
});

test('Generated: server-side generators with an optional seed', () => {
    assert.deepEqual(
        buildTrajectorySpec('Generated', { sim: sim({ genKind: 'rectilinear', genDuration: 8, genCount: 3, genSeed: '' }) }),
        { type: 'rectilinear', duration_s: 8, count: 3, seed: null }
    );
    assert.deepEqual(
        buildTrajectorySpec('Generated', { sim: sim({ genKind: 'spline', genDuration: 4, genCount: 1, genKeyPoints: 7, genSeed: ' 42 ' }) }),
        { type: 'spline', duration_s: 4, count: 1, seed: 42, n_key_points: 7 }
    );
    assert.deepEqual(
        buildTrajectorySpec('Generated', { sim: sim({ genKind: 'street', genDuration: 20, genCount: 2, genKeyPoints: 7, genSeed: '1' }) }),
        { type: 'street', duration_s: 20, count: 2, seed: 1 }
    );

    assert.throws(() => buildTrajectorySpec('Generated', { sim: sim({ genSeed: 'abc' }) }), /Seed/);
    assert.throws(() => buildTrajectorySpec('Generated', { sim: sim({ genCount: 0 }) }), /Count/);
    assert.throws(() => buildTrajectorySpec('Generated', { sim: sim({ genDuration: 0 }) }), /Duration/);
    assert.throws(() => buildTrajectorySpec('Generated', { sim: sim({ genKind: 'spline', genKeyPoints: 1 }) }), /Key points/);
});

test('Generated street walks can name a human-motion model; other generators ignore it', () => {
    assert.equal(DEFAULT_SIM_SETTINGS.genMotion, '');
    assert.deepEqual(
        buildTrajectorySpec('Generated', { sim: sim({ genKind: 'street', genDuration: 20, genSeed: '1', genMotion: 'walk-v1' }) }),
        { type: 'street', duration_s: 20, count: 1, seed: 1, motion: { model: 'walk-v1' } }
    );
    assert.equal('motion' in buildTrajectorySpec('Generated', { sim: sim({ genKind: 'street', genMotion: '' }) }), false);
    assert.equal('motion' in buildTrajectorySpec('Generated', { sim: sim({ genKind: 'spline', genMotion: 'walk-v1' }) }), false);
    assert.equal(buildJobRequest('Generated', { sim: sim({ genKind: 'street', genMotion: 'walk-v1' }) }).trajectory.motion.model, 'walk-v1');
});

test('the Human motion dropdown offers the glide first, then the trained models', () => {
    assert.deepEqual(motionOptions([]), { 'None (planner glide)': '' });
    assert.deepEqual(motionOptions([{ id: 'walk-v1', size_bytes: 5 }, { id: 'run' }]), { 'None (planner glide)': '', 'walk-v1': 'walk-v1', run: 'run' });
    assert.deepEqual(motionOptions(undefined), { 'None (planner glide)': '' });
});

test('a stored motion model is restored, a wrongly typed one is not', () => {
    assert.equal(restoreSimSettings(sim(), { genMotion: 'walk-v1' }).genMotion, 'walk-v1');
    assert.equal(restoreSimSettings(sim(), { genMotion: 3 }).genMotion, '');
});

test('Live cannot be simulated', () => {
    assert.throws(() => buildTrajectorySpec('Live', { sim: sim() }), /Live/);
});

test('parseSeed', () => {
    assert.equal(parseSeed(''), null);
    assert.equal(parseSeed('   '), null);
    assert.equal(parseSeed(undefined), null);
    assert.equal(parseSeed('7'), 7);
    assert.equal(parseSeed('-3'), -3);
    assert.throws(() => parseSeed('1.5'), /integer/);
});

test('buildJobRequest has the shape of POST /api/jobs (minus scene_id)', () => {
    const request = buildJobRequest('Generated', { sim: sim({ visualize: true }) });

    assert.deepEqual(Object.keys(request).sort(), ['config', 'trajectory', 'visualize']);
    assert.equal(request.visualize, true);
});

test('restoreSimSettings only takes values of the right type, from known keys', () => {
    const target = sim();
    restoreSimSettings(target, {
        width: 640, height: '480', fov: NaN, stereo: true, rgb: 'yes', genKind: 'spline', unknown: 1, genSeed: '5'
    });

    assert.equal(target.width, 640);
    assert.equal(target.height, 260, 'string for a number is ignored');
    assert.equal(target.fov, 60, 'NaN is ignored');
    assert.equal(target.stereo, true);
    assert.equal(target.rgb, true, 'string for a boolean is ignored');
    assert.equal(target.genKind, 'spline');
    assert.equal(target.genSeed, '5');
    assert.equal('unknown' in target, false);

    restoreSimSettings(target, { genKind: 'teleport' });
    assert.equal(target.genKind, 'spline', 'unknown generator is ignored');
    assert.doesNotThrow(() => restoreSimSettings(target, null));
    assert.doesNotThrow(() => restoreSimSettings(target, 'garbage'));
});
