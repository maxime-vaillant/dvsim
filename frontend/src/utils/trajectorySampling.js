// Turns a drawn curve into timestamped camera records moving at a constant speed.

import { round } from './trajectoryRecords.js';

function normalised(vector) {
    const length = Math.hypot(vector.x, vector.y, vector.z);

    return length > 1e-9 ? { x: vector.x / length, y: vector.y / length, z: vector.z / length } : null;
}

/**
 * Samples `curve` (anything with getLength(), getPointAt(u) and getTangentAt(u), u in [0, 1] by arc
 * length, e.g. THREE.CatmullRomCurve3) into records { timestamp (ms), x, y, z, rx, ry, rz }.
 * Positions are the curve's own (world) coordinates; forward = normalised tangent.
 *
 * @param speed      metres (scene units) per second
 * @param sampleHz   records per second of trajectory time
 * @param maxSamples upper bound on the number of records
 */
export function sampleConstantSpeed(curve, { speed = 0.8, sampleHz = 50, maxSamples = 20000 } = {}) {
    if (!(speed > 0)) throw new Error('Speed must be positive.');

    const length = curve.getLength();
    if (!(length > 1e-6)) throw new Error('The trajectory has no length, draw a longer curve.');

    const durationS = length / speed;
    const count = Math.min(maxSamples, Math.max(2, Math.ceil(durationS * sampleHz) + 1));

    const records = [];
    let previousForward = { x: 0, y: 0, z: -1 };

    for (let i = 0; i < count; i++) {
        const u = i / (count - 1);
        const position = curve.getPointAt(u);
        const forward = normalised(curve.getTangentAt(u)) || previousForward;

        previousForward = forward;

        records.push({
            timestamp: round(u * durationS * 1000, 3),
            x: round(position.x, 6),
            y: round(position.y, 6),
            z: round(position.z, 6),
            rx: round(forward.x, 6),
            ry: round(forward.y, 6),
            rz: round(forward.z, 6)
        });
    }

    return records;
}
