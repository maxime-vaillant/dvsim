// Phone page maths: device orientation -> camera direction, recentering, joystick walking, relay messages.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';

import { Rebaser, deviceQuaternion, horizontalHeading, toMessage, walkStep } from '../src/utils/phonePose.js';

const close = (actual, expected, tol = 1e-9, message = '') =>
    assert.ok(Math.abs(actual - expected) <= tol, `${message} ${actual} != ${expected}`);
const forward = q => new THREE.Vector3(0, 0, -1).applyQuaternion(q);
const closeVec = (v, [x, y, z], message = '') => {
    close(v.x, x, 1e-9, `${message} x`);
    close(v.y, y, 1e-9, `${message} y`);
    close(v.z, z, 1e-9, `${message} z`);
};

test('device orientation: upright looks at the horizon, flat face up looks down, alpha turns left', () => {
    closeVec(forward(deviceQuaternion(0, 90, 0)), [0, 0, -1], 'upright');
    closeVec(forward(deviceQuaternion(0, 0, 0)), [0, -1, 0], 'flat');
    closeVec(forward(deviceQuaternion(90, 90, 0)), [-1, 0, 0], 'alpha 90 turns towards -X');
    closeVec(forward(deviceQuaternion(0, 135, 0)), [0, Math.SQRT1_2, -Math.SQRT1_2], 'tilted back looks up');
});

test('heading: 0 along -Z, positive towards +X, defined when looking down or up', () => {
    close(horizontalHeading(deviceQuaternion(0, 90, 0)), 0);
    close(horizontalHeading(deviceQuaternion(-90, 90, 0)), Math.PI / 2, 1e-9, 'towards +X');
    close(horizontalHeading(deviceQuaternion(90, 90, 0)), -Math.PI / 2, 1e-9, 'towards -X');
    // flat on the table, screen top pointing to -X: that is "ahead"
    close(horizontalHeading(deviceQuaternion(90, 0, 0)), -Math.PI / 2, 1e-9, 'looking down');
    close(horizontalHeading(deviceQuaternion(90, 180, 0)), -Math.PI / 2, 1e-9, 'looking up');
});

test('recenter: that pose becomes the origin facing -Z; later motion is expressed in that frame', () => {
    const rebase = new Rebaser();
    const start = deviceQuaternion(90, 90, 0);                       // looking along -X
    const p0 = new THREE.Vector3(1, 2, 3);

    rebase.recenter(p0, start);

    const at = rebase.apply(p0, start);
    closeVec(new THREE.Vector3(at.x, at.y, at.z), [0, 0, 0], 'origin');
    closeVec(new THREE.Vector3(at.rx, at.ry, at.rz), [0, 0, -1], 'forward');

    // two metres further along where the phone looks (-X in the sensor frame) is two metres ahead (-Z)
    const ahead = rebase.apply(new THREE.Vector3(-1, 2, 3), start);
    closeVec(new THREE.Vector3(ahead.x, ahead.y, ahead.z), [0, 0, -2], 'moved ahead');

    // one metre to the phone's right (-Z when looking along -X) is +X after recentering, height is untouched
    const right = rebase.apply(new THREE.Vector3(1, 2.5, 2), start);
    closeVec(new THREE.Vector3(right.x, right.y, right.z), [1, 0.5, 0], 'moved right and up');

    // turning the phone 90 degrees to the right afterwards: now looking along +X
    const turned = rebase.apply(p0, deviceQuaternion(0, 90, 0));
    closeVec(new THREE.Vector3(turned.rx, turned.ry, turned.rz), [1, 0, 0], 'turned right');
});

test('recenter keeps the tilt: looking 45 degrees up stays 45 degrees up', () => {
    const rebase = new Rebaser();
    const q = deviceQuaternion(30, 135, 0);

    rebase.recenter(new THREE.Vector3(), q);

    const pose = rebase.apply(new THREE.Vector3(), q);
    close(pose.ry, Math.SQRT1_2, 1e-9, 'pitch');
    close(pose.rx, 0, 1e-9, 'heading x');
    close(pose.rz, -Math.SQRT1_2, 1e-9, 'heading z');
});

test('joystick walking: forward and strafe follow the heading, the stick length is capped, vertical raises', () => {
    const origin = { x: 0, y: 1.5, z: 0 };
    const facing = { rx: 0, rz: -1 };

    let p = walkStep(origin, facing, { x: 0, y: 1 }, 0, 1, 2);
    close(p.x, 0); close(p.z, -2); close(p.y, 1.5);

    p = walkStep(origin, facing, { x: 1, y: 0 }, 0, 0.5, 2);
    close(p.x, 1); close(p.z, 0, 1e-12);

    p = walkStep(origin, { rx: 1, rz: 0 }, { x: 0, y: 1 }, 0, 1, 1);            // facing +X
    close(p.x, 1); close(p.z, 0, 1e-12);
    p = walkStep(origin, { rx: 1, rz: 0 }, { x: 1, y: 0 }, 0, 1, 1);            // right of +X is +Z
    close(p.x, 0, 1e-12); close(p.z, 1);

    p = walkStep(origin, facing, { x: 1, y: 1 }, 0, 1, 1);                       // diagonal: not faster than 1 m/s
    close(Math.hypot(p.x, p.z), 1);

    p = walkStep(origin, { rx: 0.3, rz: -0.4 }, { x: 0, y: 1 }, 0, 1, 1);        // the heading need not be unit
    close(Math.hypot(p.x, p.z), 1);

    p = walkStep(origin, facing, { x: 0, y: 0 }, 1, 2, 0.5);
    close(p.y, 2.5);
    p = walkStep(origin, facing, { x: 0, y: 0 }, -5, 1, 1);                      // vertical capped to [-1, 1]
    close(p.y, 0.5);

    p = walkStep(origin, { rx: 0, rz: 0 }, { x: 0, y: 1 }, 0, 1, 1);             // no heading: assume -Z
    close(p.z, -1);
});

test('relay messages follow the /headset protocol', () => {
    const pose = { x: 1, y: 2, z: 3, rx: 0, ry: 0, rz: -1 };

    assert.deepEqual(JSON.parse(toMessage('stream_data', 1234.5, pose)), {
        action: 'stream_data', data: { timestamp: 1234.5, x: 1, y: 2, z: 3, rx: 0, ry: 0, rz: -1 }
    });
    assert.deepEqual(JSON.parse(toMessage('start_recording', 10)), { action: 'start_recording', data: { timestamp: 10 } });
});

// ---- origin of imported / live positions ----------------------------------------------------------

import { ORIGIN_MODES, PHONE_START_HEIGHT, trajectoryOrigin } from '../src/utils/trajectoryOrigin.js';
import { DEFAULT_SIM_SETTINGS, buildTrajectorySpec, restoreSimSettings } from '../src/utils/simRequest.js';

test('origin: the object centre (default) or the floor centre one phone height up', () => {
    const center = { x: 1, y: 22, z: -3 };

    assert.deepEqual(trajectoryOrigin('Object centre', center, 0), center);
    assert.notEqual(trajectoryOrigin('Object centre', center, 0), center, 'a copy');
    assert.deepEqual(trajectoryOrigin('Floor centre', center, 0.5), { x: 1, y: 0.5 + PHONE_START_HEIGHT, z: -3 });
    assert.deepEqual(trajectoryOrigin('anything else', center, 0), center, 'unknown modes fall back to the centre');
    assert.equal(DEFAULT_SIM_SETTINGS.originMode, ORIGIN_MODES[0]);
});

test('origin: saved settings keep a valid mode and drop an unknown one; imported spec carries the origin', () => {
    assert.equal(restoreSimSettings({ ...DEFAULT_SIM_SETTINGS }, { originMode: 'Floor centre' }).originMode, 'Floor centre');
    assert.equal(restoreSimSettings({ ...DEFAULT_SIM_SETTINGS }, { originMode: 'Moon' }).originMode, 'Object centre');

    const records = [0, 1000].map(t => ({ timestamp: t, x: t / 1000, y: 0, z: 0, rx: 1, ry: 0, rz: 0, normalizedTimestamp: t / 1000 }));
    const spec = buildTrajectorySpec('Imported', { records, origin: { x: 1, y: 2, z: 3 }, sim: DEFAULT_SIM_SETTINGS });

    assert.deepEqual(spec.origin, [1, 2, 3]);
    assert.equal(spec.points[0].normalizedTimestamp, undefined);
});

// ---- /api/live wire format --------------------------------------------------------------------------

import { LIVE_VIEWS, LatencyTracker, SIZES, parseFrame, poseMessage, startMessage, viewMessage } from '../src/utils/liveFrames.js';

test('live messages follow the /api/live protocol', () => {
    assert.deepEqual(JSON.parse(startMessage({ sceneId: 'abc', size: '640x480', origin: 'floor', view: 'grid' })), {
        type: 'start', scene_id: 'abc', width: 640, height: 480, origin: 'floor', view: 'grid', fps: 30
    });
    assert.deepEqual(JSON.parse(startMessage({ sceneId: 'abc', size: 'nonsense' })).width, SIZES['346x260'].width);
    assert.deepEqual(JSON.parse(poseMessage(5, { x: 1, y: 2, z: 3, rx: 0, ry: 0, rz: -1, extra: 9 })), {
        type: 'pose', seq: 5, x: 1, y: 2, z: 3, rx: 0, ry: 0, rz: -1
    });
    assert.deepEqual(JSON.parse(viewMessage('depth')), { type: 'view', view: 'depth' });
    assert.deepEqual(LIVE_VIEWS, ['rgb', 'events', 'depth', 'flow', 'grid']);
});

test('live frames: 4-byte big-endian sequence number then the JPEG', () => {
    const bytes = new Uint8Array([0, 0, 1, 2, 0xff, 0xd8, 0xff, 0xd9]);
    const frame = parseFrame(bytes.buffer);

    assert.equal(frame.seq, 258);
    assert.deepEqual([...frame.jpeg], [0xff, 0xd8, 0xff, 0xd9]);
    assert.equal(parseFrame(new Uint8Array([0, 0, 0, 1]).buffer), null, 'no image data');
    assert.equal(parseFrame(null), null);
    assert.equal(parseFrame(new Uint8Array([0xff, 0xff, 0xff, 0xff, 1]).buffer).seq, 4294967295);
});

test('latency: the round trip of the pose a frame shows; unknown or forgotten poses give null', () => {
    const tracker = new LatencyTracker(3);

    tracker.sent(1, 100); tracker.sent(2, 133); tracker.sent(3, 166); tracker.sent(4, 200);
    assert.equal(tracker.received(1, 250), null, 'seq 1 was forgotten (capacity 3)');
    assert.equal(tracker.received(3, 266), 100);
    assert.equal(tracker.average, 100);
    assert.equal(tracker.received(4, 280), 80);
    assert.ok(Math.abs(tracker.average - 96) < 1e-9, 'smoothed');
    assert.equal(tracker.received(99, 300), null);
});

// ---- the Live start point -------------------------------------------------------------------------

import { applyLiveStart, originBody, rotateY, startAtSurface } from '../src/utils/liveStart.js';

test('live start: heading rotates the pose about the vertical, like the simulator does', () => {
    const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);
    const start = { x: 10, y: 1.5, z: -4, headingDeg: 90 };
    const pose = applyLiveStart(start, { x: 0, y: 0.2, z: -2, rx: 0, ry: 0, rz: -1 });   // 2 m ahead, looking ahead

    near(pose.position.x, 10 - 2); near(pose.position.y, 1.7); near(pose.position.z, -4);   // heading 90: -Z becomes -X
    near(pose.direction.x, -1); near(pose.direction.y, 0); near(pose.direction.z, 0);

    const same = applyLiveStart({ ...start, headingDeg: 0 }, { x: 1, y: 2, z: 3, rx: 0.6, ry: 0, rz: -0.8 });
    assert.deepEqual([same.position.x, same.position.y, same.position.z], [11, 3.5, -1]);
    assert.deepEqual(same.direction, { x: 0.6, y: 0, z: -0.8 });

    const back = rotateY(rotateY({ x: 1, y: 5, z: 2 }, 37), -37);
    near(back.x, 1); near(back.y, 5); near(back.z, 2);
    near(rotateY({ x: 0, y: 0, z: -1 }, 180).z, 1);
});

test('live start: a clicked surface point starts one phone height above it; the server body wraps the heading', () => {
    assert.deepEqual(startAtSurface({ x: 1, y: 0.25, z: 2 }, 15), { x: 1, y: 1.75, z: 2, headingDeg: 15 });
    assert.deepEqual(startAtSurface({ x: 0, y: 0, z: 0 }), { x: 0, y: 1.5, z: 0, headingDeg: 0 });
    assert.equal(originBody({ x: 1, y: 2, z: 3, headingDeg: 190 }).heading_deg, -170);
    assert.equal(originBody({ x: 1, y: 2, z: 3, headingDeg: -190 }).heading_deg, 170);
    assert.equal(originBody({ x: 1, y: 2, z: 3, headingDeg: 45 }).heading_deg, 45);
    assert.deepEqual(originBody({ x: 1, y: 2, z: 3, headingDeg: 0 }), { x: 1, y: 2, z: 3, heading_deg: 0 });
});
