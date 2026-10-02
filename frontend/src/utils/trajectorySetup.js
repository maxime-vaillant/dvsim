import * as THREE from "three";
import { THREE_VARIABLES } from "../components/variables.js";
import { render } from "../components/render.js";
import { currentOrigin } from "../components/origin.js";
import { notify } from "./notify.js";

const SELECTED_COLOR = 0x0000ff;
const OTHER_COLOR = 0x8fa3ff;

function disposeObject(object) {
    THREE_VARIABLES.scene.remove(object);
    object.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
    });
}

// Removes the trajectory line(s) from the overview (the data they were built from is kept).
export function clearTrajectoryLine() {
    if (THREE_VARIABLES.trajectory) {
        disposeObject(THREE_VARIABLES.trajectory);
        THREE_VARIABLES.trajectory = null;
    }
    if (THREE_VARIABLES.generatedLines) {
        disposeObject(THREE_VARIABLES.generatedLines);
        THREE_VARIABLES.generatedLines = null;
    }
}

function showTrajectoryLine(points) {
    clearTrajectoryLine();

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0x0000ff });

    THREE_VARIABLES.trajectory = new THREE.Line(geometry, material);
    THREE_VARIABLES.trajectory.layers.set(1);
    THREE_VARIABLES.scene.add(THREE_VARIABLES.trajectory);
}

// Draws the curve that was last built from the mouse points.
export function showDrawnTrajectory() {
    if (THREE_VARIABLES.trajectoryCurve) {
        showTrajectoryLine(THREE_VARIABLES.trajectoryCurve.getPoints(100));
    } else {
        clearTrajectoryLine();
    }

    render();
}

export function updateDrawnTrajectory() {
    THREE_VARIABLES.trajectoryCurve = new THREE.CatmullRomCurve3(THREE_VARIABLES.drawnTrajectoryPoints.slice());

    showDrawnTrajectory();
}

// Imported records are relative to the object centre.
export function updateImportedTrajectory() {
    clearTrajectoryLine();

    const center = currentOrigin();
    const records = THREE_VARIABLES.trajectoryData;

    if (records.length < 2) {
        notify('Load a trajectory file first (Loading > Load Trajectory).');
    } else if (!center) {
        notify('Load a scene first (Loading > Load scene): imported trajectories are relative to it.');
    } else {
        showTrajectoryLine(records.map(data => new THREE.Vector3(center.x + data.x, center.y + data.y, center.z + data.z)));
    }

    render();
}

/** The generated path that Play follows, or null when nothing was previewed (or the object changed since). */
export function selectedGeneratedPath() {
    const paths = THREE_VARIABLES.generatedPaths;

    if (!paths.length || THREE_VARIABLES.generatedRoom !== THREE_VARIABLES.room) return null;

    return paths[Math.min(THREE_VARIABLES.generatedSelected, paths.length - 1)];
}

/** Keeps previewed paths ([{ records, durationS }], world coordinates) for the current object. */
export function setGeneratedPaths(paths) {
    THREE_VARIABLES.generatedPaths = paths;
    THREE_VARIABLES.generatedSelected = 0;
    THREE_VARIABLES.generatedRoom = THREE_VARIABLES.room;
}

export function clearGeneratedPaths() {
    THREE_VARIABLES.generatedPaths = [];
    THREE_VARIABLES.generatedSelected = 0;
    THREE_VARIABLES.generatedRoom = null;
}

// The view cameras show where the selected path starts, looking the way it goes.
function placeViewCameraAtStart(path) {
    const camera = THREE_VARIABLES.viewCamera;
    const first = path.records[0];

    if (!camera) return;

    camera.position.set(first.x, first.y, first.z);
    camera.lookAt(first.x + first.rx, first.y + first.ry, first.z + first.rz);
}

// Every previewed path on the overview, the one Play follows in dark blue, with a dot where each one starts.
export function showGeneratedPaths() {
    clearTrajectoryLine();

    const selected = selectedGeneratedPath();

    if (selected) {
        const group = new THREE.Group();
        const starts = [];

        THREE_VARIABLES.generatedPaths.forEach(path => {
            const points = path.records.map(r => new THREE.Vector3(r.x, r.y, r.z));
            const color = path === selected ? SELECTED_COLOR : OTHER_COLOR;
            const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color }));

            line.renderOrder = path === selected ? 1 : 0;
            group.add(line);
            starts.push(points[0]);
        });

        group.add(new THREE.Points(
            new THREE.BufferGeometry().setFromPoints(starts),
            new THREE.PointsMaterial({ color: SELECTED_COLOR, size: 6, sizeAttenuation: false })
        ));
        group.traverse(child => child.layers.set(1));

        THREE_VARIABLES.generatedLines = group;
        THREE_VARIABLES.scene.add(group);

        placeViewCameraAtStart(selected);
    }

    render();
}
