// The Live start point: where the live camera's origin (the phone after "Recenter") is in the scene, and which way
// its forward points. DOM free, unit-tested in node. Same formula as the simulator's (backend/live.py):
// a pose (p, f) relative to the origin is the world pose (origin + R(p), R(f)), R = rotation about +Y by the heading.

import { PHONE_START_HEIGHT } from './trajectoryOrigin.js';

/** Rotation about +Y by `degrees` (three.js convention: positive turns -Z towards -X). */
export function rotateY({ x, y, z }, degrees) {
    const angle = degrees * Math.PI / 180;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    return { x: x * cos + z * sin, y, z: -x * sin + z * cos };
}

/**
 * World position and forward of a live pose `rel` ({ x, y, z, rx, ry, rz }) for `start` ({ x, y, z, headingDeg }):
 * { position: {x,y,z}, direction: {x,y,z} }.
 */
export function applyLiveStart(start, rel) {
    const place = rotateY({ x: rel.x, y: rel.y, z: rel.z }, start.headingDeg);
    const look = rotateY({ x: rel.rx, y: rel.ry, z: rel.rz }, start.headingDeg);

    return {
        position: { x: start.x + place.x, y: start.y + place.y, z: start.z + place.z },
        direction: look
    };
}

/** The start for a clicked surface point: the phone starts one phone height above it. */
export function startAtSurface(hit, headingDeg = 0) {
    return { x: hit.x, y: hit.y + PHONE_START_HEIGHT, z: hit.z, headingDeg };
}

/** Body of PUT /api/live/origin (the server renders the phone's view from the same start). */
export function originBody(start) {
    return { x: start.x, y: start.y, z: start.z, heading_deg: ((start.headingDeg + 180) % 360 + 360) % 360 - 180 };
}
