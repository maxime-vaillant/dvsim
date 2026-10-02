import * as THREE from 'three';

import { ARButton } from "three/examples/jsm/webxr/ARButton.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { XRControllerModelFactory } from "three/addons/webxr/XRControllerModelFactory.js";

import { getWsBase } from "./api/urls.js";
import { endpoints } from "../endpoints.js";

// 3D scene

let scene, camera, renderer;
let controller1, controller2;
let controllerGrip1, controllerGrip2;

// Websocket

let websocket;
let isWsOpen;

// Recording

let isRecording;
let recordingIndicator;

export function initHeadsetApp() {
    initiateWebsocket();

    // Scene

    scene = new THREE.Scene();

    const ambientLight = new THREE.AmbientLight( 0xffffff );
    scene.add( ambientLight );

    // Camera

    camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        20
    );
    camera.position.z = -5;

    // Renderer

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.xr.enabled = true;
    renderer.setAnimationLoop( animate );
    document.body.appendChild(renderer.domElement);

    // AR Button

    document.body.appendChild( ARButton.createButton(renderer) );

    // Controllers

    const controllerModelFactory = new XRControllerModelFactory();

    controller1 = renderer.xr.getController( 0 );
    controllerGrip1 = renderer.xr.getControllerGrip( 0 );
    controllerGrip1.add( controllerModelFactory.createControllerModel( controllerGrip1 ) );

    controller2 = renderer.xr.getController( 1 );
    controllerGrip2 = renderer.xr.getControllerGrip( 1 );
    controllerGrip2.add( controllerModelFactory.createControllerModel( controllerGrip2 ) );

    controller1.addEventListener('selectstart', handleControllerInput);
    controller2.addEventListener('selectstart', handleControllerInput);

    scene.add( controller1 );
    scene.add( controllerGrip1 );

    scene.add( controller2 );
    scene.add( controllerGrip2 );

    // Control

    const controls = new OrbitControls( camera, renderer.domElement );
    controls.minDistance = -10;
    controls.maxDistance = 10;
    controls.addEventListener( 'change', render );

    // Recording indicator

    const circleRadius = 0.05;
    const circleSegments = 32;

    const indicatorGeometry = new THREE.CircleGeometry(circleRadius, circleSegments);
    const indicatorMaterial = new THREE.MeshBasicMaterial( { color: 0xff0000 } ); // Red color
    recordingIndicator = new THREE.Mesh( indicatorGeometry, indicatorMaterial );
    recordingIndicator.position.set( 1, 1, -1 );

    const cubeGeometry = new THREE.BoxGeometry(1, 1, 1);
    const cubeMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 }); // Green color

    const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);

    scene.add(cube);

    // Listeners

    window.addEventListener( 'resize', onWindowResize );
}

function render() {
    renderer.render( scene, camera );
}

function handleControllerInput( event ) {
    const session = renderer.xr.getSession();

    session.inputSources.forEach( (inputSource) => {
        if (inputSource.gamepad) {
            const gamepad = inputSource.gamepad;
            if (gamepad.buttons[0].pressed) {
                handleControllerButtonPressed();
            }
        }
    });
}

function handleControllerButtonPressed() {
    if (isRecording) {
        sendEndTrigger();
        camera.remove( recordingIndicator );
    } else {
        sendStartTrigger()
        camera.add( recordingIndicator );
    }

    isRecording = !isRecording;
}

function initiateWebsocket(){
    websocket = new WebSocket(getWsBase() + endpoints.HEADSET);

    websocket.addEventListener('open', () => {
        isWsOpen = true;
    })

    websocket.addEventListener('close', () => {
        isWsOpen = false;
    })

    websocket.addEventListener('error', (error) => {
        isWsOpen = false;
        alert('WebSocket error');
        console.error('WebSocket error:', error);
    })
}

function sendStartTrigger() {
    websocket.send(JSON.stringify({
        action: "start_recording",
        data: {
            timestamp: performance.now(),
        }
    }))
}

function sendEndTrigger() {
    websocket.send(JSON.stringify({
        action: "stop_recording",
        data: {
            timestamp: performance.now(),
        }
    }))
}

function sendData() {
    const position = camera.position;
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);

    websocket.send(JSON.stringify({
        action: "stream_data",
        data: {
            timestamp: performance.now(),
            x: position.x,
            y: position.y,
            z: position.z,
            rx: direction.x,
            ry: direction.y,
            rz: direction.z
        }
    }))
}

function animate() {
    if (isWsOpen) sendData();

    render();
}

function onWindowResize() {
    renderer.setSize( window.innerWidth, window.innerHeight );
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    render();
}