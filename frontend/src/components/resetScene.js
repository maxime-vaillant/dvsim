import * as THREE from "three";

import { THREE_VARIABLES } from "./variables";
import { clearCachedScene, disposeRoom, disposeTexture } from "./cache";
import { clearLiveStart, publishLiveOrigin } from "./liveStart";
import { frameOverview } from "./overview";
import { render, resetFlowView } from "./render";
import { clearGeneratedPaths, showDrawnTrajectory, showGeneratedPaths, updateImportedTrajectory } from "../utils/trajectorySetup.js";
import { notify } from "../utils/notify.js";

function disposeViewCamera() {
    const camera = THREE_VARIABLES.viewCamera;

    if (camera) {
        THREE_VARIABLES.scene.remove(camera);
        camera.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) [].concat(child.material).forEach(material => material.dispose());
        });
        THREE_VARIABLES.viewCamera = null;
    }

    if (THREE_VARIABLES.eventVisionComposer) {
        THREE_VARIABLES.eventVisionComposer.dispose();
        THREE_VARIABLES.eventVisionComposer = null;
        THREE_VARIABLES.eventVisionPass = null;
    }
}

/**
 * "Reset object": forgets the loaded scene (the OBJ, its texture, the other files of a multi-file scene) in the page
 * and in the browser cache, back to the empty previewer. Drawn and imported trajectories are kept: they come back
 * once an object is loaded. Previewed generated paths belonged to the object and go with it.
 */
export async function resetScene() {
    THREE_VARIABLES.isPlaying = false;
    THREE_VARIABLES.animationProgress = 0;
    THREE_VARIABLES.trajectoryDistance = 0;
    THREE_VARIABLES.currentTrajectoryIndex = 0;

    try {
        await clearCachedScene();
    } catch (error) {
        console.error('Error clearing the cached scene:', error);
        notify(`Cannot clear the browser cache: ${error.message}`, { level: 'error' });
        return;
    }

    disposeRoom();
    disposeTexture();
    disposeViewCamera();
    clearGeneratedPaths();
    clearLiveStart({ quiet: true });
    THREE_VARIABLES.initialTrajectoryPoint = null;
    THREE_VARIABLES.objectFloorY = 0;
    publishLiveOrigin();   // no scene: the simulator forgets the origin
    frameOverview(new THREE.Vector3(), 0);
    resetFlowView();

    switch (THREE_VARIABLES.settings.trajectoryType) {
        case 'Custom':
            showDrawnTrajectory();
            break;
        case 'Imported':
            updateImportedTrajectory();
            break;
        default:
            showGeneratedPaths();
    }

    render();
    notify('Scene cleared. Loading > Load scene picks a new one (select its .obj, .mtl and textures together).');
}
