import * as THREE from "three";
import { THREE_VARIABLES } from "../components/variables.js";
import { render } from "../components/render.js";

export function updateDrawnTrajectory() {
    if (THREE_VARIABLES.trajectory) THREE_VARIABLES.scene.remove(THREE_VARIABLES.trajectory);

    THREE_VARIABLES.trajectoryCurve = new THREE.CatmullRomCurve3(THREE_VARIABLES.drawnTrajectoryPoints);
    const curvePoints = THREE_VARIABLES.trajectoryCurve.getPoints(100);

    const geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
    const material = new THREE.LineBasicMaterial({ color: 0x0000ff });

    THREE_VARIABLES.trajectory = new THREE.Line(geometry, material);
    THREE_VARIABLES.trajectory.layers.set(1);
    THREE_VARIABLES.scene.add(THREE_VARIABLES.trajectory);

    render();
}

export function updateImportedTrajectory() {
    if (THREE_VARIABLES.trajectory) THREE_VARIABLES.scene.remove(THREE_VARIABLES.trajectory);

    const trajectoryPoints = []

    THREE_VARIABLES.trajectoryData.forEach(data => {
        trajectoryPoints.push( new THREE.Vector3(
            THREE_VARIABLES.initialTrajectoryPoint.x + data.x,
            THREE_VARIABLES.initialTrajectoryPoint.y + data.y,
            THREE_VARIABLES.initialTrajectoryPoint.z + data.z
        ))
    })

    const geometry = new THREE.BufferGeometry().setFromPoints(trajectoryPoints);
    const material = new THREE.LineBasicMaterial({ color: 0x0000ff });

    THREE_VARIABLES.trajectory = new THREE.Line(geometry, material);
    THREE_VARIABLES.trajectory.layers.set(1);
    THREE_VARIABLES.scene.add(THREE_VARIABLES.trajectory);

    render();
}