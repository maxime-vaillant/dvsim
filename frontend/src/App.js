import * as THREE from 'three';

import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer, RenderPass, ShaderPass } from "three/addons";

import EventVisionShader from "./shaders/eventVisionShader";
import DepthShader from "./shaders/depthShader";
import { initGUI } from "./components/gui";
import { getObjectCenter } from "./utils/objects.js";
import { loadCachedObject, loadCachedTexture, loadCachedTrajectory } from "./components/cache";
import { onMouseDown, onMouseMove, onMouseUp } from "./components/listeners";
import { render } from "./components/render";
import { THREE_VARIABLES } from "./components/variables";


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
        20
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
    controls.enableRotate = false;
    controls.minDistance = 1;
    controls.maxDistance = 10;
    controls.addEventListener( 'change', render );

    // Loaders

    const manager = new THREE.LoadingManager( loadModel );

    THREE_VARIABLES.textureLoader = new THREE.TextureLoader( manager );
    THREE_VARIABLES.objLoader = new OBJLoader( manager );

    loadCachedTexture().then();
    loadCachedObject().then();
    loadCachedTrajectory().then();

    // Listeners

    window.addEventListener( 'resize', onWindowResize );
    window.addEventListener('mousedown', onMouseDown, false);
    window.addEventListener('mousemove', onMouseMove, false);
    window.addEventListener('mouseup', onMouseUp, false);
}



function initViewCamera() {
    if (THREE_VARIABLES.viewCamera) THREE_VARIABLES.scene.remove(THREE_VARIABLES.viewCamera);

    THREE_VARIABLES.viewCamera = new THREE.PerspectiveCamera(
        45,
        2 * window.innerWidth / window.innerHeight,
        0.1,
        5
    );
    THREE_VARIABLES.viewCamera.position.copy( THREE_VARIABLES.initialTrajectoryPoint );
    THREE_VARIABLES.viewCamera.layers.disable(1);

    const viewCameraSphereGeometry = new THREE.SphereGeometry(0.05, 16, 16);
    const viewCameraSphereMaterial = new THREE.MeshBasicMaterial({ color: 0x8B0000 });
    const viewCameraSphere = new THREE.Mesh(viewCameraSphereGeometry, viewCameraSphereMaterial);
    viewCameraSphere.layers.set(1);

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

    THREE_VARIABLES.viewCamera.add(viewCameraSphere);
    THREE_VARIABLES.viewCamera.add(viewCameraArrowHelper);

    THREE_VARIABLES.scene.add( THREE_VARIABLES.viewCamera );

    const renderEventVisionTarget = new THREE.WebGLRenderTarget(
        window.innerWidth / 2,
        window.innerHeight / 2
    );
    const renderDepthTarget = new THREE.WebGLRenderTarget(
        window.innerWidth / 2,
        window.innerHeight / 2,
        {
            type: THREE.FloatType,
            depthTexture: new THREE.DepthTexture(),
        }
    );

    THREE_VARIABLES.eventVisionComposer = new EffectComposer( THREE_VARIABLES.renderer, renderEventVisionTarget );
    THREE_VARIABLES.depthComposer = new EffectComposer( THREE_VARIABLES.renderer, renderDepthTarget );

    const renderPass = new RenderPass( THREE_VARIABLES.scene, THREE_VARIABLES.viewCamera );
    THREE_VARIABLES.eventVisionPass = new ShaderPass( EventVisionShader );
    THREE_VARIABLES.depthPass = new ShaderPass( DepthShader );

    THREE_VARIABLES.eventVisionComposer.addPass(renderPass);
    THREE_VARIABLES.eventVisionComposer.addPass(THREE_VARIABLES.eventVisionPass);

    THREE_VARIABLES.depthComposer.addPass(renderPass);
    THREE_VARIABLES.depthComposer.addPass(THREE_VARIABLES.depthPass);

    THREE_VARIABLES.renderer.setRenderTarget(THREE_VARIABLES.renderTarget);
    THREE_VARIABLES.renderer.render(THREE_VARIABLES.scene, THREE_VARIABLES.viewCamera);

    THREE_VARIABLES.eventVisionPass.uniforms.uPrevFrame.value = THREE_VARIABLES.renderTarget.texture;
    THREE_VARIABLES.eventVisionPass.uniforms.threshold.value = THREE_VARIABLES.settings.eventVisionThreshold;
    THREE_VARIABLES.renderer.setRenderTarget(null);

    THREE_VARIABLES.depthPass.uniforms['tDepth'].value = renderDepthTarget.depthTexture;
    THREE_VARIABLES.depthPass.uniforms['uNear'].value = THREE_VARIABLES.viewCamera.near;
    THREE_VARIABLES.depthPass.uniforms['uFar'].value = THREE_VARIABLES.viewCamera.far;

}

function animate() {
    switch (THREE_VARIABLES.settings.trajectoryType) {
        case 'Custom':
            animateCustomTrajectory();
            break
        case 'Imported':
            animateImportedTrajectory();
            break
        case 'Live':
            animateLiveTrajectory();
    }
}

function animateCustomTrajectory() {
    if ( THREE_VARIABLES.viewCamera && THREE_VARIABLES.isPlaying ) {
        THREE_VARIABLES.viewCamera.position.copy( THREE_VARIABLES.trajectoryCurve.getPointAt( THREE_VARIABLES.animationProgress ) );

        if ( THREE_VARIABLES.animationProgress + 0.01 <= 1) {
            THREE_VARIABLES.viewCamera.lookAt( THREE_VARIABLES.trajectoryCurve.getPointAt( THREE_VARIABLES.animationProgress + 0.01 ) );
        }

        render();

        THREE_VARIABLES.trajectoryDistance += THREE_VARIABLES.settings.animationSpeed;

        THREE_VARIABLES.animationProgress = THREE_VARIABLES.trajectoryDistance / THREE_VARIABLES.trajectoryCurve.getLength();

        if (THREE_VARIABLES.animationProgress >= 1) {
            THREE_VARIABLES.isPlaying = false;
            THREE_VARIABLES.trajectoryDistance = 0;
            THREE_VARIABLES.animationProgress = 0;
        }
    }
}

function animateImportedTrajectory() {
    if ( THREE_VARIABLES.viewCamera && THREE_VARIABLES.isPlaying ) {
        while (THREE_VARIABLES.currentTrajectoryIndex < THREE_VARIABLES.trajectoryData.length - 1 && THREE_VARIABLES.animationProgress > THREE_VARIABLES.trajectoryData[THREE_VARIABLES.currentTrajectoryIndex + 1].normalizedTimestamp) {
            THREE_VARIABLES.currentTrajectoryIndex++;
        }

        if (THREE_VARIABLES.currentTrajectoryIndex < THREE_VARIABLES.trajectoryData.length - 1) {

            const currentTrajectoryPosition = THREE_VARIABLES.trajectoryData[THREE_VARIABLES.currentTrajectoryIndex];
            const nextTrajectoryPosition = THREE_VARIABLES.trajectoryData[THREE_VARIABLES.currentTrajectoryIndex + 1];

            const interpolationFactor = (THREE_VARIABLES.animationProgress - currentTrajectoryPosition.normalizedTimestamp) / (nextTrajectoryPosition.normalizedTimestamp - currentTrajectoryPosition.normalizedTimestamp);

            THREE_VARIABLES.viewCamera.position.lerpVectors(
                new THREE.Vector3(
                    THREE_VARIABLES.initialTrajectoryPoint.x + currentTrajectoryPosition.x,
                    THREE_VARIABLES.initialTrajectoryPoint.y + currentTrajectoryPosition.y,
                    THREE_VARIABLES.initialTrajectoryPoint.z + currentTrajectoryPosition.z
                ),
                new THREE.Vector3(
                    THREE_VARIABLES.initialTrajectoryPoint.x + nextTrajectoryPosition.x,
                    THREE_VARIABLES.initialTrajectoryPoint.y + nextTrajectoryPosition.y,
                    THREE_VARIABLES.initialTrajectoryPoint.z + nextTrajectoryPosition.z
                ),
                interpolationFactor
            );

            const currentPositionLookAt = new THREE.Vector3(
                currentTrajectoryPosition.rx,
                currentTrajectoryPosition.ry,
                currentTrajectoryPosition.rz
            )
            const nextPositionLookAt = new THREE.Vector3(
                nextTrajectoryPosition.rx,
                nextTrajectoryPosition.ry,
                nextTrajectoryPosition.rz
            )

            THREE_VARIABLES.viewCamera.lookAt( currentPositionLookAt.lerp( nextPositionLookAt, interpolationFactor ))
        }

        render();

        THREE_VARIABLES.animationProgress += THREE_VARIABLES.settings.animationSpeed;

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

    if ( THREE_VARIABLES.viewCamera ) {
        THREE_VARIABLES.viewCamera.aspect = window.innerWidth / window.innerHeight;
        THREE_VARIABLES.viewCamera.updateProjectionMatrix();
    }

    render();
}

function loadModel() {
    THREE_VARIABLES.room.traverse( function ( child ) {
        if ( child.isMesh ) child.material.map = THREE_VARIABLES.texture;
    } );

    THREE_VARIABLES.room.scale.setScalar( 0.5 );

    const objCenter = getObjectCenter(THREE_VARIABLES.room);

    THREE_VARIABLES.roomAvgHeight = objCenter.y / 2
    THREE_VARIABLES.initialTrajectoryPoint = objCenter;

    THREE_VARIABLES.scene.add( THREE_VARIABLES.room );

    initViewCamera();

    render();
}

const objFileInput = document.getElementById('objFileInput');
objFileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];

    if (file) {
        try {
            const cache = await caches.open('obj-cache')

            await cache.put('/obj', new Response(file));
        } catch (error) {
            console.log('Error caching file:', error)
        }
    }

    await loadCachedObject();
});

const textureFileInput = document.getElementById('textureFileInput');
textureFileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];

    if (file) {
        try {
            const cache = await caches.open('obj-cache')

            await cache.put('/texture', new Response(file));
        } catch (error) {
            console.log('Error caching file:', error)
        }
    }

    await loadCachedTexture();
});

const trajectoryFileInput = document.getElementById('trajectoryFileInput');
trajectoryFileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];

    if (file) {
        try {
            const cache = await caches.open('obj-cache')

            await cache.put('/trajectory', new Response(file));
        } catch (error) {
            console.log('Error caching file:', error)
        }
    }

    await loadCachedTrajectory();
})