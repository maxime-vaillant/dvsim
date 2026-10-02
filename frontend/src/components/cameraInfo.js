import * as THREE from "three";

import { formatCameraInfo } from "../utils/viewLayout.js";
import { THREE_VARIABLES } from "./variables";
import { selectedGeneratedPath } from "../utils/trajectorySetup.js";

const forward = new THREE.Vector3();

// Current trajectory time of the preview, when there is one (Custom: distance / speed; Imported / Generated:
// time along the records).
function trajectoryTimeS() {
    const settings = THREE_VARIABLES.settings;

    switch (settings.trajectoryType) {
        case 'Custom':
            return THREE_VARIABLES.trajectoryCurve && settings.sim.speed > 0
                ? THREE_VARIABLES.trajectoryDistance / settings.sim.speed : null;
        case 'Imported':
            return THREE_VARIABLES.trajectoryData.length >= 2
                ? THREE_VARIABLES.animationProgress * THREE_VARIABLES.trajectoryDurationS : null;
        case 'Generated': {
            const path = selectedGeneratedPath();

            return path ? THREE_VARIABLES.animationProgress * path.durationS : null;
        }
        default:
            return null;
    }
}

/** Camera readout in the corner of the overview (top-left quadrant); hidden until there is a view camera. */
export function updateCameraInfo() {
    const element = document.getElementById('cameraInfo');
    const camera = THREE_VARIABLES.viewCamera;

    if (!element) return;
    if (!camera) {
        element.hidden = true;
        return;
    }

    camera.getWorldDirection( forward );
    element.textContent = formatCameraInfo( camera.position, forward, trajectoryTimeS() );
    element.hidden = false;
}
