// Pure trajectory helpers: constant-speed sampling of a drawn curve and the legacy record format.
// Run with: npm test   (node --test tests/)
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';

import { sampleConstantSpeed } from '../src/utils/trajectorySampling.js';
import { parseTrajectoryRecords, toLegacyRecords } from '../src/utils/trajectoryRecords.js';

const curveThrough = (...points) => new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)));

const drawn = () => curveThrough([0, 1, 0], [1, 1, 0.2], [2, 1, 1], [2.5, 1, 2.5], [1, 1, 3.5], [0, 1, 3]);

test('sampling moves at the requested constant speed', () => {
    const curve = drawn();
    const speed = 0.8;
    const records = sampleConstantSpeed(curve, { speed });

    assert.equal(records[0].timestamp, 0);
    assert.ok(Math.abs(records.at(-1).timestamp - curve.getLength() / speed * 1000) < 1e-2);

    for (let i = 1; i < records.length; i++) {
        const a = records[i - 1];
        const b = records[i];
        const chord = Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z);
        const dt = (b.timestamp - a.timestamp) / 1000;

        assert.ok(dt > 0, 'timestamps strictly increase');
        // chord <= arc length; the curve is smooth at 1.6 cm spacing so they agree closely
        assert.ok(Math.abs(chord / dt - speed) < 0.02, `speed at sample ${i}: ${chord / dt}`);
    }
});

test('sampling starts and ends at the curve ends, with unit forward vectors along the tangent', () => {
    const curve = drawn();
    const records = sampleConstantSpeed(curve, { speed: 1, sampleHz: 100 });
    const first = curve.getPointAt(0);
    const last = curve.getPointAt(1);

    assert.deepEqual([records[0].x, records[0].y, records[0].z].map(v => +v.toFixed(5)), [first.x, first.y, first.z].map(v => +v.toFixed(5)));
    assert.ok(Math.abs(records.at(-1).x - last.x) < 1e-5 && Math.abs(records.at(-1).z - last.z) < 1e-5);

    for (let i = 1; i < records.length - 1; i++) {
        const r = records[i];
        assert.ok(Math.abs(Math.hypot(r.rx, r.ry, r.rz) - 1) < 1e-5, 'forward is unit length');

        // forward agrees with the direction of travel
        const next = records[i + 1];
        const prev = records[i - 1];
        const travel = new THREE.Vector3(next.x - prev.x, next.y - prev.y, next.z - prev.z).normalize();
        assert.ok(travel.dot(new THREE.Vector3(r.rx, r.ry, r.rz)) > 0.99);
    }
});

test('sampling positions are absolute world coordinates (no origin applied)', () => {
    const records = sampleConstantSpeed(curveThrough([10, 2, -5], [11, 2, -5]), { speed: 0.5 });

    assert.equal(records[0].x, 10);
    assert.equal(records[0].z, -5);
    assert.ok(Math.abs(records.at(-1).x - 11) < 1e-6);
    assert.deepEqual([records[0].rx, records[0].ry, records[0].rz], [1, 0, 0]);
});

test('sampling rejects a bad speed or a curve without length, and honours maxSamples', () => {
    assert.throws(() => sampleConstantSpeed(drawn(), { speed: 0 }), /Speed/);
    assert.throws(() => sampleConstantSpeed(drawn(), { speed: -1 }), /Speed/);
    assert.throws(() => sampleConstantSpeed(curveThrough([1, 1, 1], [1, 1, 1]), { speed: 1 }), /no length/);
    assert.equal(sampleConstantSpeed(drawn(), { speed: 0.01, maxSamples: 50 }).length, 50);
    assert.equal(sampleConstantSpeed(curveThrough([0, 0, 0], [0.001, 0, 0]), { speed: 100 }).length, 2);
});

test('legacy export is relative to the origin and re-imports to the same path', () => {
    const origin = [1.5, -0.25, 3];
    const world = sampleConstantSpeed(drawn(), { speed: 0.8 });
    const exported = toLegacyRecords(world, origin);

    assert.deepEqual(Object.keys(exported[0]), ['timestamp', 'x', 'y', 'z', 'rx', 'ry', 'rz']);
    assert.ok(Math.abs(exported[3].x - (world[3].x - origin[0])) < 1e-6);

    // what the previewer / backend do with an imported file: parse, then add the object centre back
    const imported = parseTrajectoryRecords(JSON.parse(JSON.stringify(exported)));
    imported.forEach((record, i) => {
        assert.ok(Math.abs(record.x + origin[0] - world[i].x) < 1e-5);
        assert.ok(Math.abs(record.y + origin[1] - world[i].y) < 1e-5);
        assert.ok(Math.abs(record.z + origin[2] - world[i].z) < 1e-5);
    });

    assert.equal(toLegacyRecords(world, null)[5].x, world[5].x);
});

test('parseTrajectoryRecords validates and sorts the imported file', () => {
    const rec = (timestamp, x = 0) => ({ timestamp, x, y: 0, z: 0, rx: 0, ry: 0, rz: -1 });

    assert.throws(() => parseTrajectoryRecords([]), /at least 2/);
    assert.throws(() => parseTrajectoryRecords(null), /array/);
    assert.throws(() => parseTrajectoryRecords({ nope: 1 }), /array/);
    assert.throws(() => parseTrajectoryRecords([rec(0), { ...rec(1), rz: undefined }]), /Record 1: "rz"/);
    assert.throws(() => parseTrajectoryRecords([rec(0), { ...rec(1), x: 'a' }]), /"x"/);
    assert.throws(() => parseTrajectoryRecords([rec(5), rec(5)]), /positive duration/);

    const sorted = parseTrajectoryRecords([rec(20, 2), rec(0, 0), rec(10, 1)]);
    assert.deepEqual(sorted.map(r => r.timestamp), [0, 10, 20]);

    // records wrapped in {points: [...]} are accepted, extra preview fields are dropped
    const wrapped = parseTrajectoryRecords({ points: [{ ...rec(0), normalizedTimestamp: 0 }, rec(1)] });
    assert.equal(wrapped[0].normalizedTimestamp, undefined);
});

test('quaternion fields are kept when the file has all four', () => {
    const base = { x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: -1 };
    const records = parseTrajectoryRecords([
        { timestamp: 0, ...base, qx: 0, qy: 0, qz: 0, qw: 1 },
        { timestamp: 1, ...base, qx: 0, qy: 0, qz: 0 }
    ]);

    assert.equal(records[0].qw, 1);
    assert.equal('qw' in records[1], false);
});
