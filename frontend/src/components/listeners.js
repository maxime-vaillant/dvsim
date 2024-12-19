import * as THREE from "three";

import { THREE_VARIABLES } from "./variables";
import { render } from "./render";
import { updateDrawnTrajectory } from "../utils/trajectorySetup.js";
import { updateViewCamera } from "../utils/cameraUpdater.js";


export function updateMouse(event) {
    THREE_VARIABLES.mouse.x = ( event.clientX / ( window.innerWidth ) ) * 4 - 1;
    THREE_VARIABLES.mouse.y = - ( event.clientY / window.innerHeight ) * 4 + 1;

    THREE_VARIABLES.raycaster.setFromCamera(THREE_VARIABLES.mouse, THREE_VARIABLES.overviewCamera);
}

export function onMouseDown(event) {
    updateMouse(event);

    const plane = new THREE.Plane(new THREE.Vector3(0, THREE_VARIABLES.roomAvgHeight, 0), 0);
    const intersectionPoint = new THREE.Vector3();

    if ( THREE_VARIABLES.raycaster.ray.intersectPlane(plane, intersectionPoint) ) {
        THREE_VARIABLES.drawnTrajectoryPoints = [];
        THREE_VARIABLES.drawnTrajectoryPoints.push( intersectionPoint.clone() );
        THREE_VARIABLES.isDrawing = true;
    }
}

export function onMouseMove(event) {
    if (!THREE_VARIABLES.isDrawing) return;

    updateMouse(event);

    const plane = new THREE.Plane(new THREE.Vector3(0, THREE_VARIABLES.roomAvgHeight, 0), 0);
    const intersectionPoint = new THREE.Vector3();

    if ( THREE_VARIABLES.raycaster.ray.intersectPlane(plane, intersectionPoint) ) {
        if ( THREE_VARIABLES.drawnTrajectoryPoints.length === 0 || THREE_VARIABLES.drawnTrajectoryPoints[THREE_VARIABLES.drawnTrajectoryPoints.length - 1].distanceTo(intersectionPoint) > 0.2) {
            THREE_VARIABLES.drawnTrajectoryPoints.push( intersectionPoint.clone() );

            updateDrawnTrajectory();
        }
    }
}

export function onMouseUp() {
    if (!THREE_VARIABLES.isDrawing) return;

    if (THREE_VARIABLES.drawnTrajectoryPoints.length >= 2) updateViewCamera( THREE_VARIABLES.drawnTrajectoryPoints[0], THREE_VARIABLES.drawnTrajectoryPoints[1] );

    THREE_VARIABLES.isDrawing = false;
    render();
}