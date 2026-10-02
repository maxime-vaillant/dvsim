// Imported trajectories and the Live (phone / headset) stream are relative to an origin in the scene: the centre of
// the loaded object (the original behaviour; right for a room), or the centre of its floor, one phone height up
// (right for a street or a campus, whose centre is high above the ground).

export const ORIGIN_MODES = ['Object centre', 'Floor centre'];
export const PHONE_START_HEIGHT = 1.5;   // m above the floor

/** { x, y, z } of the origin; `center` is the object's bounding-box centre, `floorY` its lowest point. */
export function trajectoryOrigin(mode, center, floorY) {
    if (mode === 'Floor centre') return { x: center.x, y: floorY + PHONE_START_HEIGHT, z: center.z };

    return { x: center.x, y: center.y, z: center.z };
}
