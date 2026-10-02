import * as THREE from "three";

import { THREE_VARIABLES } from "./variables";
import { trajectoryOrigin } from "../utils/trajectoryOrigin.js";

/** Where Imported and Live positions are measured from (a THREE.Vector3), or null while no object is loaded. */
export function currentOrigin() {
    const center = THREE_VARIABLES.initialTrajectoryPoint;

    if (!center) return null;

    const origin = trajectoryOrigin(THREE_VARIABLES.settings.sim.originMode, center, THREE_VARIABLES.objectFloorY);

    return new THREE.Vector3(origin.x, origin.y, origin.z);
}
