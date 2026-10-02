import { THREE_VARIABLES } from "./variables";
import { worldLivePose } from "./liveStart.js";
import { updateViewCamera } from "../utils/cameraUpdater.js";
import { notify } from "../utils/notify.js";
import { getWsBase } from "../api/urls.js";
import { endpoints } from "../../endpoints.js";


export function initiateWebsocket() {
    if (!THREE_VARIABLES.initialTrajectoryPoint) {
        notify('Load a scene first (Loading > Load scene): live positions are relative to it.', { level: 'error' });
        return;
    }

    THREE_VARIABLES.websocket = new WebSocket(getWsBase() + endpoints.SPECTATOR);

    THREE_VARIABLES.websocket.addEventListener('open', () => {
        THREE_VARIABLES.isWsOpen = true;
    })

    THREE_VARIABLES.websocket.addEventListener('message', (event) => {
        if (!THREE_VARIABLES.viewCamera) return;

        let parsedData;
        try {
            parsedData = JSON.parse(event.data);
        } catch {
            return;
        }

        // the pose is relative to the live origin (the picked start point, or "Positions relative to");
        // (rx, ry, rz) is a direction: the camera looks at the point one unit ahead
        const pose = worldLivePose(parsedData);

        if (pose) updateViewCamera(pose.position, pose.lookAt);
    })

    THREE_VARIABLES.websocket.addEventListener('close', () => {
        THREE_VARIABLES.isWsOpen = false;
    })

    THREE_VARIABLES.websocket.addEventListener('error', () => {
        THREE_VARIABLES.isWsOpen = false;
        notify('Live connection failed: is the headset relay reachable?', { level: 'error' });
    })
}
