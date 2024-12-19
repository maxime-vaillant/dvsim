import * as THREE from "three";

import { THREE_VARIABLES } from "./variables";
import { updateViewCamera } from "../utils/cameraUpdater.js";


export function initiateWebsocket() {
    THREE_VARIABLES.websocket = new WebSocket(websocketAddress);

    THREE_VARIABLES.websocket.addEventListener('open', () => {
        console.log('WebSocket connection established');
        THREE_VARIABLES.websocket.send('spectator')
    })

    THREE_VARIABLES.websocket.addEventListener('message', (event) => {
        const parsedData = JSON.parse(event.data);

        const position = new THREE.Vector3(
            parsedData.x + THREE_VARIABLES.initialTrajectoryPoint.x,
            parsedData.y + THREE_VARIABLES.initialTrajectoryPoint.y,
            parsedData.z + THREE_VARIABLES.initialTrajectoryPoint.z
        );
        const lookAt = new THREE.Vector3(parsedData.rx, parsedData.ry, parsedData.rz);

        updateViewCamera(position, lookAt);
    })

    THREE_VARIABLES.websocket.addEventListener('close', () => {
        console.log('WebSocket connection closed');
    })

    THREE_VARIABLES.websocket.addEventListener('error', (error) => {
        THREE_VARIABLES.isWsOpen = false;
        console.error('WebSocket error:', error);
    })
}