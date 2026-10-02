import * as THREE from "three";

export function getObjectDimensions(object) {
    const boundingBox = new THREE.Box3().setFromObject(object);

    const size = new THREE.Vector3();
    boundingBox.getSize(size);

    return size;
}

export function getObjectFloor(object) {
    return new THREE.Box3().setFromObject(object).min.y;
}

export function getObjectCenter(object) {
    const boundingBox = new THREE.Box3().setFromObject(object);
    const center = new THREE.Vector3();

    boundingBox.getCenter(center);

    return center
}