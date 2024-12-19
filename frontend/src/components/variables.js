import * as THREE from "three";

export const THREE_VARIABLES = {
    // 3D scene
    scene: null,
    overviewCamera: null,
    viewCamera: null,
    renderer: null,
    room: null,
    roomAvgHeight: 0,
    texture: null,
    settings: null,
    objLoader: null,
    textureLoader: null,

    // Depth
    depthComposer: null,
    depthPass: null,

    // Event
    eventVisionComposer: null,
    eventVisionPass: null,
    renderTarget: new THREE.WebGLRenderTarget(window.innerWidth / 2, window.innerHeight / 2),

    // Interface
    mouse: new THREE.Vector2(),
    raycaster: new THREE.Raycaster(),
    drawnTrajectoryPoints: [],
    isDrawing: null,
    trajectory: null,
    trajectoryCurve: null,

    // Animation
    isPlaying: false,
    animationProgress: 0,
    trajectoryDistance: 0,

    // Imported Trajectory
    trajectoryData: [],
    currentTrajectoryIndex: 0,
    initialTrajectoryPoint: null,

    // Websocket
    websocket: null,
    isWsOpen: null,
}