import * as THREE from "three";

export const THREE_VARIABLES = {
    // 3D scene
    scene: null,
    overviewCamera: null,
    viewCamera: null,
    renderer: null,
    controls: null,
    room: null,
    roomHasMaterials: false,     // the object brought its own materials (MTL): no single texture on top
    roomAvgHeight: 0,
    markerScale: 1,              // size of the camera marker in the overview (grows with large scenes)
    texture: null,
    settings: null,
    objLoader: null,
    textureLoader: null,

    // Depth
    depthTarget: null,
    depthQuad: null,

    // Optical flow preview (see components/render.js)
    flowMaterial: null,
    flowState: { last: null, prev: null, changedAt: 0, simDt: 0 },

    // Which preview each quadrant shows (see components/views.js)
    views: null,

    // Event
    eventVisionComposer: null,
    eventVisionPass: null,
    renderTarget: new THREE.WebGLRenderTarget(1, 1),

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
    trajectoryDurationS: 0,
    currentTrajectoryIndex: 0,
    initialTrajectoryPoint: null,   // centre of the loaded object
    objectFloorY: 0,                // its lowest point

    // Generated trajectories previewed before simulating (see utils/trajectorySetup.js)
    generatedPaths: [],          // [{ records, durationS }], world coordinates
    generatedSelected: 0,        // index of the path that Play follows
    generatedRoom: null,         // the object they were generated on
    generatedLines: null,

    // Live start point picked on the overview (see components/liveStart.js)
    liveStart: null,             // { x, y, z, headingDeg } or null: then the "Positions relative to" origin
    liveStartMarker: null,
    pickingStart: false,

    // Websocket
    websocket: null,
    isWsOpen: null,
}
