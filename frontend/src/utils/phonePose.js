// Pose maths of the phone page (phone.html): DOM free, unit-tested in node.
// Conventions are three.js': +Y up, the camera looks along its -Z, metres. A pose is
// { x, y, z, rx, ry, rz }: position and unit forward vector, what the /headset relay carries.

import * as THREE from 'three';

const DEG = Math.PI / 180;
const ZEE = new THREE.Vector3(0, 0, 1);
const BACK_CAMERA = new THREE.Quaternion(-Math.sqrt(0.5), 0, 0, Math.sqrt(0.5));   // -90 deg around X
const MIN_HORIZONTAL = 0.2;   // below this the camera looks (nearly) straight up or down

/**
 * Orientation of the phone's back camera from DeviceOrientationEvent angles (degrees) and the screen rotation
 * (screen.orientation.angle): the same formula as three.js' DeviceOrientationControls. Holding the phone upright
 * and facing alpha = 0 looks along -Z; lying face up it looks straight down.
 */
export function deviceQuaternion(alpha, beta, gamma, screenAngle = 0) {
    const euler = new THREE.Euler(beta * DEG, alpha * DEG, -gamma * DEG, 'YXZ');
    const q = new THREE.Quaternion().setFromEuler(euler);

    q.multiply(BACK_CAMERA);
    q.multiply(new THREE.Quaternion().setFromAxisAngle(ZEE, -screenAngle * DEG));

    return q;
}

/** Horizontal heading (rad) of a camera: 0 along -Z, positive towards +X. Looking up / down: where the screen top points. */
export function horizontalHeading(quaternion) {
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion);

    if (Math.hypot(forward.x, forward.z) >= MIN_HORIZONTAL) return Math.atan2(forward.x, -forward.z);

    const top = new THREE.Vector3(0, 1, 0).applyQuaternion(quaternion);
    const sign = forward.y < 0 ? 1 : -1;   // looking down: the screen top points ahead; looking up: behind

    return Math.atan2(sign * top.x, -sign * top.z);
}

function rotateY(vector, angle) {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    return new THREE.Vector3(vector.x * cos + vector.z * sin, vector.y, -vector.x * sin + vector.z * cos);
}

/**
 * Makes poses relative to a start: after recenter(position, quaternion) that pose is the origin looking along -Z
 * (the tilt is not touched, only the heading and the position are re-based).
 */
export class Rebaser {
    constructor() {
        this.origin = new THREE.Vector3();
        this.heading = 0;
    }

    recenter(position, quaternion) {
        this.origin.copy(position);
        this.heading = horizontalHeading(quaternion);
    }

    /** Pose of (position, quaternion) relative to the recentered start. */
    apply(position, quaternion) {
        const place = rotateY(position.clone().sub(this.origin), this.heading);
        const look = rotateY(new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion), this.heading);

        return { x: place.x, y: place.y, z: place.z, rx: look.x, ry: look.y, rz: look.z };
    }
}

/**
 * One step of walking with the on-screen sticks: `stick` (x: strafe right, y: forward; each in [-1, 1], the
 * length is capped to 1) moves along the camera heading `dir` ({ rx, rz }), `vertical` in [-1, 1] raises / lowers.
 * Returns the new { x, y, z }; speed in m/s, dt in s.
 */
export function walkStep(position, dir, stick, vertical, dt, speed) {
    const length = Math.hypot(stick.x, stick.y);
    const scale = length > 1 ? 1 / length : 1;
    const norm = Math.hypot(dir.rx, dir.rz);
    const ahead = norm > 1e-6 ? { x: dir.rx / norm, z: dir.rz / norm } : { x: 0, z: -1 };
    const right = { x: -ahead.z, z: ahead.x };
    const step = speed * dt;

    return {
        x: position.x + step * scale * (stick.y * ahead.x + stick.x * right.x),
        y: position.y + step * Math.max(-1, Math.min(1, vertical)),
        z: position.z + step * scale * (stick.y * ahead.z + stick.x * right.z)
    };
}

/** The /headset relay message of a pose (timestamp in ms); action: stream_data | start_recording | stop_recording. */
export function toMessage(action, timestamp, pose = null) {
    const data = { timestamp };

    if (pose) Object.assign(data, { x: pose.x, y: pose.y, z: pose.z, rx: pose.rx, ry: pose.ry, rz: pose.rz });

    return JSON.stringify({ action, data });
}
