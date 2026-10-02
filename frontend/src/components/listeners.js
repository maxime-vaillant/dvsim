import * as THREE from "three";

import { THREE_VARIABLES } from "./variables";
import { render } from "./render";
import { updateDrawnTrajectory } from "../utils/trajectorySetup.js";
import { updateViewCamera } from "../utils/cameraUpdater.js";
import { clearNotice } from "../utils/notify.js";


export function updateMouse(event) {
    THREE_VARIABLES.mouse.x = ( event.clientX / ( window.innerWidth ) ) * 4 - 1;
    THREE_VARIABLES.mouse.y = - ( event.clientY / window.innerHeight ) * 4 + 1;

    THREE_VARIABLES.raycaster.setFromCamera(THREE_VARIABLES.mouse, THREE_VARIABLES.overviewCamera);
}

// Horizontal plane at the camera height; trajectories are drawn on it.
function drawingPlane() {
    return new THREE.Plane(new THREE.Vector3(0, 1, 0), -THREE_VARIABLES.roomAvgHeight);
}

// Only the canvas' overview quadrant (top-left) draws: clicks on the GUI or the other views must not.
function isOverviewEvent(event) {
    return event.target === THREE_VARIABLES.renderer.domElement
        && event.clientX < window.innerWidth / 2
        && event.clientY < window.innerHeight / 2;
}

export function onMouseDown(event) {
    if (event.button !== 0 || !isOverviewEvent(event)) return;

    updateMouse(event);

    const intersectionPoint = new THREE.Vector3();

    if ( THREE_VARIABLES.raycaster.ray.intersectPlane(drawingPlane(), intersectionPoint) ) {
        THREE_VARIABLES.drawnTrajectoryPoints = [];
        THREE_VARIABLES.drawnTrajectoryPoints.push( intersectionPoint.clone() );
        THREE_VARIABLES.isDrawing = true;
    }
}

export function onMouseMove(event) {
    if (!THREE_VARIABLES.isDrawing) return;

    updateMouse(event);

    const intersectionPoint = new THREE.Vector3();

    if ( THREE_VARIABLES.raycaster.ray.intersectPlane(drawingPlane(), intersectionPoint) ) {
        if ( THREE_VARIABLES.drawnTrajectoryPoints.length === 0 || THREE_VARIABLES.drawnTrajectoryPoints[THREE_VARIABLES.drawnTrajectoryPoints.length - 1].distanceTo(intersectionPoint) > 0.2) {
            THREE_VARIABLES.drawnTrajectoryPoints.push( intersectionPoint.clone() );

            updateDrawnTrajectory();
        }
    }
}

export function onMouseUp() {
    if (!THREE_VARIABLES.isDrawing) return;

    if (THREE_VARIABLES.drawnTrajectoryPoints.length >= 2) {
        updateViewCamera( THREE_VARIABLES.drawnTrajectoryPoints[0], THREE_VARIABLES.drawnTrajectoryPoints[1] );
        clearNotice();
    }

    THREE_VARIABLES.isDrawing = false;
    render();
}

/**
 * Drawing mode (trajectory type "Custom"): left-drag draws on the overview, so the orbit controls move to
 * the right button; otherwise the controls get their default mapping back.
 */
export function setDrawingEnabled(enabled) {
    const method = enabled ? 'addEventListener' : 'removeEventListener';

    window[method]('mousedown', onMouseDown, false);
    window[method]('mousemove', onMouseMove, false);
    window[method]('mouseup', onMouseUp, false);

    THREE_VARIABLES.isDrawing = false;

    const controls = THREE_VARIABLES.controls;

    if (controls) {
        controls.mouseButtons.LEFT = enabled ? null : THREE.MOUSE.ROTATE;
        controls.mouseButtons.RIGHT = enabled ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
    }
}
