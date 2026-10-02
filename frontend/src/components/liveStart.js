import * as THREE from "three";

import { THREE_VARIABLES } from "./variables";
import { currentOrigin } from "./origin";
import { render } from "./render";
import { updateMouse } from "./listeners";
import { deleteLiveOrigin, putLiveOrigin } from "../api/client.js";
import { applyLiveStart, originBody, startAtSurface } from "../utils/liveStart.js";
import { clearNotice, notify } from "../utils/notify.js";

const MARKER_COLOR = 0xff8800;
const CLICK_TOLERANCE_PX = 5;
let downAt = null;

/**
 * The live camera's origin: the picked point with its heading, else the "Positions relative to" origin facing -Z.
 * { origin: THREE.Vector3, headingDeg } or null while no object is loaded.
 */
export function liveStart() {
    const start = THREE_VARIABLES.liveStart;

    if (start) return { origin: new THREE.Vector3(start.x, start.y, start.z), headingDeg: start.headingDeg };

    const origin = currentOrigin();

    return origin ? { origin, headingDeg: 0 } : null;
}

/** World position and look-at target of a live pose relative to the origin, or null while there is no origin. */
export function worldLivePose(rel) {
    const start = liveStart();

    if (!start) return null;

    const { position, direction } = applyLiveStart(
        { x: start.origin.x, y: start.origin.y, z: start.origin.z, headingDeg: start.headingDeg }, rel
    );
    const at = new THREE.Vector3(position.x, position.y, position.z);

    return { position: at, lookAt: at.clone().add(new THREE.Vector3(direction.x, direction.y, direction.z)) };
}

// ---- marker on the overview --------------------------------------------------------------------------------------

function disposeMarker() {
    const marker = THREE_VARIABLES.liveStartMarker;

    if (!marker) return;

    THREE_VARIABLES.scene.remove(marker);
    marker.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
    });
    THREE_VARIABLES.liveStartMarker = null;
}

// An arrow at the start (the phone's height) pointing the way it faces, and a line down to the ground.
function drawMarker() {
    disposeMarker();

    const start = THREE_VARIABLES.liveStart;

    if (!start) return;

    const scale = THREE_VARIABLES.markerScale;
    const forward = applyLiveStart(start, { x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: -1 }).direction;
    const origin = new THREE.Vector3(start.x, start.y, start.z);
    const group = new THREE.Group();

    group.add(new THREE.ArrowHelper(new THREE.Vector3(forward.x, forward.y, forward.z), origin, 1.2 * scale, MARKER_COLOR, 0.4 * scale, 0.25 * scale));
    group.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([origin, new THREE.Vector3(start.x, start.y - 1.5, start.z)]),
        new THREE.LineBasicMaterial({ color: MARKER_COLOR })
    ));
    group.traverse(child => child.layers.set(1));
    THREE_VARIABLES.liveStartMarker = group;
    THREE_VARIABLES.scene.add(group);
}

// ---- the server renders the phone's view from the same start ---------------------------------------------------

/**
 * Tells the simulator where the live camera starts: the picked point, else the "Positions relative to" origin (what
 * this page's Live view uses), so the phone's rendered view ("As in the interface") always agrees with it. Nothing
 * while no scene is loaded.
 */
export function publishLiveOrigin() {
    const start = liveStart();
    const call = start
        ? putLiveOrigin(originBody({ x: start.origin.x, y: start.origin.y, z: start.origin.z, headingDeg: start.headingDeg }))
        : deleteLiveOrigin();

    call.catch(error => console.warn('The simulator did not get the live start point:', error.message));
}

// ---- changing the start ------------------------------------------------------------------------------------------

function setStart(start) {
    THREE_VARIABLES.liveStart = start;
    drawMarker();
    publishLiveOrigin();
    render();
}

export function setLiveHeading(degrees) {
    const start = THREE_VARIABLES.liveStart || (liveStart() && {
        x: liveStart().origin.x, y: liveStart().origin.y, z: liveStart().origin.z, headingDeg: 0
    });

    if (!start) return;

    setStart({ ...start, headingDeg: degrees });
}

export function clearLiveStart({ quiet = false } = {}) {
    THREE_VARIABLES.pickingStart = false;

    if (!THREE_VARIABLES.liveStart) return;

    THREE_VARIABLES.liveStart = null;
    drawMarker();
    publishLiveOrigin();
    render();
    if (!quiet) notify('Live start point cleared: back to "Positions relative to".');
}

/** Arms the next click on the overview (top-left) to set the start point. */
export function armPicking() {
    if (!THREE_VARIABLES.room) {
        notify('Load a scene first (Loading > Load scene).', { level: 'error' });
        return;
    }

    THREE_VARIABLES.pickingStart = true;
    notify('Click on the scene in the overview (top-left) where the phone should start. Esc cancels.', { ms: 20000 });
}

function pickAt(event) {
    updateMouse(event);

    const raycaster = THREE_VARIABLES.raycaster;
    const hits = raycaster.intersectObject(THREE_VARIABLES.room, true);
    let point = hits.length ? hits[0].point : null;

    if (!point) {   // the click missed the object: the floor plane under it
        point = new THREE.Vector3();
        if (!raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -THREE_VARIABLES.objectFloorY), point)) return;
    }

    const heading = THREE_VARIABLES.liveStart ? THREE_VARIABLES.liveStart.headingDeg : 0;

    setStart(startAtSurface(point, heading));
    THREE_VARIABLES.pickingStart = false;
    clearNotice();
    notify('Live start set (orange arrow). Press "Recenter" on the phone where you stand; "Start heading" turns it.');
}

function isOverviewEvent(event) {
    return event.target === THREE_VARIABLES.renderer.domElement
        && event.clientX < window.innerWidth / 2
        && event.clientY < window.innerHeight / 2;
}

/** Click-to-pick on the overview, and Esc to cancel. A drag (orbiting the overview) is not a click. */
export function initLiveStartPicking() {
    window.addEventListener('mousedown', event => { downAt = { x: event.clientX, y: event.clientY }; });
    window.addEventListener('click', event => {
        if (!THREE_VARIABLES.pickingStart || event.button !== 0 || !isOverviewEvent(event) || !THREE_VARIABLES.room) return;
        if (downAt && Math.hypot(event.clientX - downAt.x, event.clientY - downAt.y) > CLICK_TOLERANCE_PX) return;

        pickAt(event);
    });
    window.addEventListener('keydown', event => {
        if (event.key === 'Escape' && THREE_VARIABLES.pickingStart) {
            THREE_VARIABLES.pickingStart = false;
            clearNotice();
        }
    });
}
