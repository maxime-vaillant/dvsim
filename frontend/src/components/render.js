import * as THREE from "three";
import { FullScreenQuad } from "three/addons/postprocessing/Pass.js";

import DepthShader from "../shaders/depthShader";
import { createFlowMaterial } from "../shaders/flowShader.js";
import { SLOTS } from "../utils/viewLayout.js";
import { updateCameraInfo } from "./cameraInfo.js";
import { THREE_VARIABLES } from "./variables";

// The window is split in four viewports: [column, row] with row 0 at the bottom. The overview is always
// top-left; the three other quadrants show the preview chosen in the Views folder (THREE_VARIABLES.views).
const OVERVIEW = [0, 1];
const MAX_FRAME_DT_S = 0.1;   // a long pause must not make the next flow frame look frozen

function useViewport([column, row]) {
    const width = window.innerWidth / 2;
    const height = window.innerHeight / 2;

    THREE_VARIABLES.renderer.setViewport(column * width, row * height, width, height);
    THREE_VARIABLES.renderer.setScissor(column * width, row * height, width, height);
    THREE_VARIABLES.renderer.setScissorTest(true);
}

function overviewRender() {
    useViewport(OVERVIEW);
    THREE_VARIABLES.renderer.render( THREE_VARIABLES.scene, THREE_VARIABLES.overviewCamera );
}

function cameraRender(viewport) {
    useViewport(viewport);
    THREE_VARIABLES.renderer.render( THREE_VARIABLES.scene, THREE_VARIABLES.viewCamera );
}

function eventsRender(viewport) {
    useViewport(viewport);
    THREE_VARIABLES.eventVisionComposer.render();
}

// The depth buffer of the view camera is rendered to a texture, then shown linearised by DepthShader.
function depthRender(viewport) {
    const renderer = THREE_VARIABLES.renderer;

    renderer.setRenderTarget( THREE_VARIABLES.depthTarget );
    renderer.render( THREE_VARIABLES.scene, THREE_VARIABLES.viewCamera );
    renderer.setRenderTarget( null );

    useViewport(viewport);
    THREE_VARIABLES.depthQuad.render( renderer );
}

function flowRender(viewport) {
    const scene = THREE_VARIABLES.scene;
    const previous = scene.overrideMaterial;

    useViewport(viewport);
    scene.overrideMaterial = flowMaterial();
    THREE_VARIABLES.renderer.render( scene, THREE_VARIABLES.viewCamera );
    scene.overrideMaterial = previous;
}

const RENDERERS = {
    'Camera': cameraRender,
    'Events': eventsRender,
    'Depth': depthRender,
    'Optical flow': flowRender
};

function flowMaterial() {
    if (!THREE_VARIABLES.flowMaterial) THREE_VARIABLES.flowMaterial = createFlowMaterial();

    return THREE_VARIABLES.flowMaterial;
}

// The flow preview compares the current view-camera pose with the pose it had when it last MOVED, so a paused
// camera keeps showing its last motion (instead of flickering to zero) and irregular live updates stay consistent.
// The displacement is rescaled to one simulation step (dt_us) of simulated time.
function updateFlowUniforms() {
    const camera = THREE_VARIABLES.viewCamera;
    const state = THREE_VARIABLES.flowState;
    const sim = THREE_VARIABLES.settings.sim;
    const now = performance.now() / 1000;

    camera.updateMatrixWorld();
    const current = new THREE.Matrix4().multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );

    if (!state.last) {
        state.last = current.clone();
        state.prev = current.clone();
        state.changedAt = now;
    } else if (!current.equals( state.last )) {
        const playback = THREE_VARIABLES.settings.trajectoryType === 'Live' ? 1 : THREE_VARIABLES.settings.animationSpeed;

        state.prev.copy( state.last );
        state.last.copy( current );
        state.simDt = Math.min( now - state.changedAt, MAX_FRAME_DT_S ) * playback;
        state.changedAt = now;
    }

    const uniforms = flowMaterial().uniforms;

    uniforms.uPrevViewProj.value.copy( state.prev );
    uniforms.uSimSize.value.set( sim.width, sim.height );
    uniforms.uStepRatio.value = state.simDt > 0 ? (sim.dt_us * 1e-6) / state.simDt : 1;
    uniforms.uScale.value = THREE_VARIABLES.views.flowScale;
}

function viewRender() {
    const views = THREE_VARIABLES.views;

    updateFlowUniforms();

    Object.entries(SLOTS).forEach(([slot, viewport]) => {
        const draw = RENDERERS[views[slot]];

        if (draw) draw(viewport);
    });

    // The event preview compares each frame with the previous one: keep the current frame for next time.
    THREE_VARIABLES.renderer.setRenderTarget( THREE_VARIABLES.renderTarget );
    THREE_VARIABLES.renderer.render( THREE_VARIABLES.scene, THREE_VARIABLES.viewCamera );

    THREE_VARIABLES.eventVisionPass.uniforms.uPrevFrame.value = THREE_VARIABLES.renderTarget.texture;
    THREE_VARIABLES.renderer.setRenderTarget(null);
}

export function render() {
    overviewRender();

    if ( THREE_VARIABLES.viewCamera ) {
        viewRender();
    }

    updateCameraInfo();
}

/** Forget the flow history (new view camera): the next frame shows zero flow instead of a jump. */
export function resetFlowView() {
    THREE_VARIABLES.flowState = { last: null, prev: null, changedAt: 0, simDt: 0 };
}

/** (Re)creates the depth render target and points the depth shader at it. */
export function resetDepthView({ near, far, range }) {
    if (THREE_VARIABLES.depthTarget) {
        THREE_VARIABLES.depthTarget.depthTexture.dispose();
        THREE_VARIABLES.depthTarget.dispose();
    }

    const depthTexture = new THREE.DepthTexture();

    THREE_VARIABLES.depthTarget = new THREE.WebGLRenderTarget(1, 1, { depthTexture });

    if (!THREE_VARIABLES.depthQuad) {
        THREE_VARIABLES.depthQuad = new FullScreenQuad(new THREE.ShaderMaterial({
            uniforms: THREE.UniformsUtils.clone(DepthShader.uniforms),
            vertexShader: DepthShader.vertexShader,
            fragmentShader: DepthShader.fragmentShader
        }));
    }

    const uniforms = THREE_VARIABLES.depthQuad.material.uniforms;

    uniforms.tDepth.value = depthTexture;
    uniforms.uNear.value = near;
    uniforms.uFar.value = far;
    uniforms.uRange.value = range;
}

/** Sizes the off-screen targets to one viewport (a quarter of the window, in device pixels). */
export function resizeViewTargets() {
    const ratio = THREE_VARIABLES.renderer.getPixelRatio();
    const cssWidth = Math.max(1, Math.floor(window.innerWidth / 2));
    const cssHeight = Math.max(1, Math.floor(window.innerHeight / 2));
    const width = Math.max(1, Math.floor(cssWidth * ratio));
    const height = Math.max(1, Math.floor(cssHeight * ratio));

    THREE_VARIABLES.renderTarget.setSize(width, height);
    if (THREE_VARIABLES.depthTarget) THREE_VARIABLES.depthTarget.setSize(width, height);
    if (THREE_VARIABLES.eventVisionComposer) THREE_VARIABLES.eventVisionComposer.setSize(cssWidth, cssHeight);
}
