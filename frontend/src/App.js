import * as THREE from 'three';

import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

import EventVisionShader from "./shaders/eventVisionShader";
import { initGUI } from "./components/gui";
import { getObjectCenter, getObjectDimensions, getObjectFloor } from "./utils/objects.js";
import { currentOrigin } from "./components/origin.js";
import { frameOverview } from "./components/overview.js";
import { clearLiveStart, initLiveStartPicking, publishLiveOrigin } from "./components/liveStart.js";
import { applyEventThresholds, applyViewCameraSettings } from "./utils/cameraUpdater.js";
import { selectedGeneratedPath, showGeneratedPaths, updateImportedTrajectory } from "./utils/trajectorySetup.js";
import { loadCachedObject, loadCachedTexture, loadCachedTrajectory } from "./components/cache";
import { initFileInputs } from "./components/fileInputs";
import { setDrawingEnabled } from "./components/listeners";
import { render, resetDepthView, resetFlowView, resizeViewTargets } from "./components/render";
import { THREE_VARIABLES } from "./components/variables";

const clock = new THREE.Clock();
const WORLD_ORIGIN = new THREE.Vector3();   // generated paths are in world coordinates


export function initApp() {
    initGUI();

    // Scene

    THREE_VARIABLES.scene = new THREE.Scene();

    const ambientLight = new THREE.AmbientLight( 0xffffff );
    THREE_VARIABLES.scene.add( ambientLight );

    // Overview camera

    THREE_VARIABLES.overviewCamera = new THREE.PerspectiveCamera(
        45,
        window.innerWidth / window.innerHeight,
        0.1,
        100
    );
    THREE_VARIABLES.overviewCamera.position.y = 5;
    THREE_VARIABLES.overviewCamera.layers.enable(1);

    THREE_VARIABLES.scene.add( THREE_VARIABLES.overviewCamera );

    // Renderer

    THREE_VARIABLES.renderer = new THREE.WebGLRenderer( { antialias: true } );
    THREE_VARIABLES.renderer.setPixelRatio( window.devicePixelRatio );
    THREE_VARIABLES.renderer.setSize( window.innerWidth, window.innerHeight );
    THREE_VARIABLES.renderer.setAnimationLoop( animate );
    document.body.appendChild( THREE_VARIABLES.renderer.domElement );

    // Control

    const controls = new OrbitControls(
        THREE_VARIABLES.overviewCamera,
        THREE_VARIABLES.renderer.domElement
    );
    controls.minDistance = 1;
    controls.maxDistance = 20;
    controls.addEventListener( 'change', render );

    THREE_VARIABLES.controls = controls;
    resizeViewTargets();

    // Loaders

    const manager = new THREE.LoadingManager( loadModel );

    THREE_VARIABLES.textureLoader = new THREE.TextureLoader( manager );
    THREE_VARIABLES.objLoader = new OBJLoader( manager );

    initFileInputs();

    loadCachedTexture().then();
    loadCachedObject().then();
    loadCachedTrajectory().then();

    // Listeners

    setDrawingEnabled( THREE_VARIABLES.settings.trajectoryType === 'Custom' );
    initLiveStartPicking();
    window.addEventListener( 'resize', onWindowResize );
}



function initViewCamera() {
    if (THREE_VARIABLES.viewCamera) THREE_VARIABLES.scene.remove(THREE_VARIABLES.viewCamera);
    if (THREE_VARIABLES.eventVisionComposer) THREE_VARIABLES.eventVisionComposer.dispose();

    // Clipping planes and depth range follow the size of the scene.
    const diagonal = getObjectDimensions( THREE_VARIABLES.room ).length();
    const far = Math.max( 10, diagonal * 1.5 );

    THREE_VARIABLES.viewCamera = new THREE.PerspectiveCamera(
        THREE_VARIABLES.settings.sim.fov,
        window.innerWidth / window.innerHeight,
        0.01,
        far
    );
    THREE_VARIABLES.viewCamera.position.copy( THREE_VARIABLES.initialTrajectoryPoint );
    THREE_VARIABLES.viewCamera.layers.disable(1);

    const viewCameraSphereGeometry = new THREE.SphereGeometry(0.05, 16, 16);
    const viewCameraSphereMaterial = new THREE.MeshBasicMaterial({ color: 0x8B0000 });
    const viewCameraSphere = new THREE.Mesh(viewCameraSphereGeometry, viewCameraSphereMaterial);
    viewCameraSphere.layers.set(1);
    viewCameraSphere.scale.setScalar( THREE_VARIABLES.markerScale );

    const viewCameraArrowDirection = new THREE.Vector3(0, 0, -1);
    const viewCameraArrowLength = 0.2;
    const viewCameraArrowColor =  0x8B0000;
    const viewCameraArrowHelper = new THREE.ArrowHelper(
        viewCameraArrowDirection,
        new THREE.Vector3(0, 0, 0),
        viewCameraArrowLength,
        viewCameraArrowColor
    );
    viewCameraArrowHelper.traverse( (node) => { node.layers.set(1) } )
    viewCameraArrowHelper.scale.setScalar( THREE_VARIABLES.markerScale );

    THREE_VARIABLES.viewCamera.add(viewCameraSphere);
    THREE_VARIABLES.viewCamera.add(viewCameraArrowHelper);

    THREE_VARIABLES.scene.add( THREE_VARIABLES.viewCamera );

    resetDepthView({ near: THREE_VARIABLES.viewCamera.near, far, range: Math.min( far, Math.max( diagonal, 1 ) ) });
    resetFlowView();

    THREE_VARIABLES.eventVisionComposer = new EffectComposer( THREE_VARIABLES.renderer, new THREE.WebGLRenderTarget( 1, 1 ) );

    THREE_VARIABLES.eventVisionPass = new ShaderPass( EventVisionShader );

    THREE_VARIABLES.eventVisionComposer.addPass( new RenderPass( THREE_VARIABLES.scene, THREE_VARIABLES.viewCamera ) );
    THREE_VARIABLES.eventVisionComposer.addPass( THREE_VARIABLES.eventVisionPass );

    applyViewCameraSettings();
    applyEventThresholds();
    resizeViewTargets();

    // The event shader compares each frame to the previous one: render a first "previous frame".
    THREE_VARIABLES.renderer.setRenderTarget(THREE_VARIABLES.renderTarget);
    THREE_VARIABLES.renderer.render(THREE_VARIABLES.scene, THREE_VARIABLES.viewCamera);

    THREE_VARIABLES.eventVisionPass.uniforms.uPrevFrame.value = THREE_VARIABLES.renderTarget.texture;
    THREE_VARIABLES.renderer.setRenderTarget(null);
}

function animate() {
    const deltaTime = Math.min( clock.getDelta(), 0.1 );

    switch (THREE_VARIABLES.settings.trajectoryType) {
        case 'Custom':
            animateCustomTrajectory( deltaTime );
            break
        case 'Imported':
            animateRecords( THREE_VARIABLES.trajectoryData, currentOrigin(), THREE_VARIABLES.trajectoryDurationS, deltaTime );
            break
        case 'Generated': {
            const path = selectedGeneratedPath();

            if (path) animateRecords( path.records, WORLD_ORIGIN, path.durationS, deltaTime );
            break
        }
        case 'Live':
            animateLiveTrajectory();
    }
}

// The previews play in real time: the drawn curve at its speed (m/s), records (imported file, generated
// path) at their own timestamps.
function animateCustomTrajectory( deltaTime ) {
    const curve = THREE_VARIABLES.trajectoryCurve;

    if ( THREE_VARIABLES.viewCamera && THREE_VARIABLES.isPlaying && curve ) {
        const progress = Math.min( THREE_VARIABLES.animationProgress, 1 );

        THREE_VARIABLES.viewCamera.position.copy( curve.getPointAt( progress ) );
        THREE_VARIABLES.viewCamera.lookAt( THREE_VARIABLES.viewCamera.position.clone().add( curve.getTangentAt( progress ) ) );

        render();

        THREE_VARIABLES.trajectoryDistance += THREE_VARIABLES.settings.sim.speed * THREE_VARIABLES.settings.animationSpeed * deltaTime;

        THREE_VARIABLES.animationProgress = THREE_VARIABLES.trajectoryDistance / curve.getLength();

        if (THREE_VARIABLES.animationProgress >= 1) {
            THREE_VARIABLES.isPlaying = false;
            THREE_VARIABLES.trajectoryDistance = 0;
            THREE_VARIABLES.animationProgress = 0;
        }
    }
}

// Records are relative to `origin` (imported files: the object centre; generated paths: the world origin).
function animateRecords( trajectoryData, origin, durationS, deltaTime ) {
    if ( THREE_VARIABLES.viewCamera && THREE_VARIABLES.isPlaying && trajectoryData.length >= 2 && origin ) {
        while (THREE_VARIABLES.currentTrajectoryIndex < trajectoryData.length - 1 && THREE_VARIABLES.animationProgress > trajectoryData[THREE_VARIABLES.currentTrajectoryIndex + 1].normalizedTimestamp) {
            THREE_VARIABLES.currentTrajectoryIndex++;
        }

        if (THREE_VARIABLES.currentTrajectoryIndex < trajectoryData.length - 1) {

            const currentTrajectoryPosition = trajectoryData[THREE_VARIABLES.currentTrajectoryIndex];
            const nextTrajectoryPosition = trajectoryData[THREE_VARIABLES.currentTrajectoryIndex + 1];

            const span = nextTrajectoryPosition.normalizedTimestamp - currentTrajectoryPosition.normalizedTimestamp;
            const interpolationFactor = span > 0
                ? THREE.MathUtils.clamp( (THREE_VARIABLES.animationProgress - currentTrajectoryPosition.normalizedTimestamp) / span, 0, 1 )
                : 0;

            THREE_VARIABLES.viewCamera.position.lerpVectors(
                new THREE.Vector3(
                    origin.x + currentTrajectoryPosition.x,
                    origin.y + currentTrajectoryPosition.y,
                    origin.z + currentTrajectoryPosition.z
                ),
                new THREE.Vector3(
                    origin.x + nextTrajectoryPosition.x,
                    origin.y + nextTrajectoryPosition.y,
                    origin.z + nextTrajectoryPosition.z
                ),
                interpolationFactor
            );

            const currentPositionDirection = new THREE.Vector3(
                currentTrajectoryPosition.rx,
                currentTrajectoryPosition.ry,
                currentTrajectoryPosition.rz
            )
            const nextPositionDirection = new THREE.Vector3(
                nextTrajectoryPosition.rx,
                nextTrajectoryPosition.ry,
                nextTrajectoryPosition.rz
            )

            const direction = currentPositionDirection.lerp( nextPositionDirection, interpolationFactor )
            const targetPosition = THREE_VARIABLES.viewCamera.position.clone().add(direction);
            THREE_VARIABLES.viewCamera.lookAt( targetPosition )
        }

        render();

        THREE_VARIABLES.animationProgress += THREE_VARIABLES.settings.animationSpeed * deltaTime / durationS;

        if (THREE_VARIABLES.animationProgress >= 1) {
            THREE_VARIABLES.isPlaying = false;
            THREE_VARIABLES.animationProgress = 0;
            THREE_VARIABLES.currentTrajectoryIndex = 0;
        }
    }
}

function animateLiveTrajectory() {
    render();
}

function onWindowResize() {
    THREE_VARIABLES.renderer.setSize( window.innerWidth, window.innerHeight );
    THREE_VARIABLES.overviewCamera.aspect = window.innerWidth / window.innerHeight;
    THREE_VARIABLES.overviewCamera.updateProjectionMatrix();

    // Each view is a quarter of the window: same aspect as the window.
    applyViewCameraSettings();
    resizeViewTargets();

    render();
}

function loadModel() {
    if ( !THREE_VARIABLES.room ) return;

    if ( THREE_VARIABLES.texture && !THREE_VARIABLES.roomHasMaterials ) {
        THREE_VARIABLES.room.traverse( function ( child ) {
            if ( child.isMesh ) {
                [].concat( child.material ).forEach( material => {
                    material.map = THREE_VARIABLES.texture;
                    material.needsUpdate = true;
                } );
            }
        } );
    }

    const objCenter = getObjectCenter(THREE_VARIABLES.room);

    clearLiveStart({ quiet: true });   // a start picked in the previous scene means nothing here

    frameOverview( objCenter, getObjectDimensions( THREE_VARIABLES.room ).length() );

    THREE_VARIABLES.roomAvgHeight = objCenter.y / 2
    THREE_VARIABLES.initialTrajectoryPoint = objCenter;
    THREE_VARIABLES.objectFloorY = getObjectFloor( THREE_VARIABLES.room );

    THREE_VARIABLES.scene.add( THREE_VARIABLES.room );

    initViewCamera();

    // Imported trajectories are drawn relative to the object centre; generated ones belong to one object
    if ( THREE_VARIABLES.settings.trajectoryType === 'Imported' ) updateImportedTrajectory();
    if ( THREE_VARIABLES.settings.trajectoryType === 'Generated' ) showGeneratedPaths();

    publishLiveOrigin();   // the simulator's phone view starts where this page's Live view does

    render();
}
