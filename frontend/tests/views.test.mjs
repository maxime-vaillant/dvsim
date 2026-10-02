import { test } from 'node:test';
import assert from 'node:assert/strict';

import { COLOR_WHEEL, flowToRgb } from '../src/utils/flowColors.js';
import { DEFAULT_VIEWS, VIEW_KINDS, formatCameraInfo, headingDeg, restoreViews } from '../src/utils/viewLayout.js';

// [dx, dy, rgb] from the simulator's backend.vizualisation.visualizer.flow_to_rgb(flow, scale=5)
const PYTHON_REFERENCE = [
    [14.286, -17.89, [151, 0, 191]], [2.927, -3.974, [196, 3, 255]], [-3.169, -1.509, [76, 160, 255]],
    [-14.14, -1.624, [0, 140, 191]], [-6.056, 23.261, [170, 191, 0]], [1.581, -2.468, [213, 106, 255]],
    [-1.969, -4.676, [15, 0, 191]], [-7.386, -2.736, [0, 104, 191]], [3.374, -1.67, [255, 63, 222]],
    [6.704, -1.399, [191, 0, 89]], [0.17, 10.821, [191, 170, 0]], [3.816, -3.537, [170, 0, 191]],
    [0, 0, [255, 255, 255]], [5, 0, [255, 0, 0]], [0, 5, [255, 230, 0]], [-5, 0, [0, 209, 255]],
    [0, -5, [88, 0, 255]], [40, -3, [191, 0, 53]]
];

test('flow colours match the simulator videos (Middlebury wheel)', () => {
    assert.equal(COLOR_WHEEL.length, 55);
    for (const [dx, dy, rgb] of PYTHON_REFERENCE) {
        assert.deepEqual(flowToRgb(dx, dy, 5), rgb, `flow (${dx}, ${dy})`);
    }
});

test('restoreViews keeps valid choices and ignores the rest', () => {
    assert.deepEqual(restoreViews(null), DEFAULT_VIEWS);
    const views = restoreViews({ topRight: 'Optical flow', bottomLeft: 'Lidar', bottomRight: 'Camera', flowScale: -3, extra: 1 });
    assert.equal(views.topRight, 'Optical flow');
    assert.equal(views.bottomLeft, DEFAULT_VIEWS.bottomLeft);
    assert.equal(views.bottomRight, 'Camera');
    assert.equal(views.flowScale, DEFAULT_VIEWS.flowScale);
    assert.equal('extra' in views, false);
    assert.ok(VIEW_KINDS.includes('Optical flow'));
});

test('heading and camera readout', () => {
    assert.deepEqual(headingDeg({ x: 0, y: 0, z: -1 }), { yaw: 0, pitch: 0 });
    assert.equal(Math.round(headingDeg({ x: 1, y: 0, z: 0 }).yaw), 90);
    assert.equal(Math.round(headingDeg({ x: 0, y: 1, z: 0 }).pitch), 90);
    const text = formatCameraInfo({ x: 1.234, y: -0.001, z: 2 }, { x: 0, y: 0, z: -1 }, 1.5);
    assert.match(text, /x\s+1\.23\s+y\s+0\.00\s+z\s+2\.00 m/);
    assert.match(text, /yaw\s+0°/);
    assert.match(text, /t 1\.50 s/);
    assert.doesNotMatch(formatCameraInfo({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: -1 }), /t /);
});
