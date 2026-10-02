import { THREE_VARIABLES } from "../components/variables.js";

export function updateViewCamera( position, lookAt ) {
    THREE_VARIABLES.viewCamera.position.copy( position );
    THREE_VARIABLES.viewCamera.lookAt( lookAt );
}

// Field of view (from the Simulation settings) and the aspect of the viewport the camera renders to.
export function applyViewCameraSettings() {
    const camera = THREE_VARIABLES.viewCamera;

    if (!camera) return;

    camera.fov = THREE_VARIABLES.settings.sim.fov;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
}

// The in-browser event preview uses the same contrast thresholds as the simulation.
export function applyEventThresholds() {
    const pass = THREE_VARIABLES.eventVisionPass;

    if (!pass) return;

    pass.uniforms.thresholdOn.value = THREE_VARIABLES.settings.sim.c_pos;
    pass.uniforms.thresholdOff.value = THREE_VARIABLES.settings.sim.c_neg;
}
