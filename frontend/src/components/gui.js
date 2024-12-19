import { GUI } from 'three/addons/libs/lil-gui.module.min.js';

import { THREE_VARIABLES } from "./variables";
import { onMouseDown, onMouseMove, onMouseUp } from "./listeners";
import { initiateWebsocket } from "./websocket";
import { updateDrawnTrajectory, updateImportedTrajectory } from "../utils/trajectorySetup.js";


export function initGUI( ) {
    const guiContainer = document.getElementById('guiContainer');
    const gui = new GUI({ container: guiContainer });

    gui.title('Settings');

    THREE_VARIABLES.settings = {
        'play': () => {
            THREE_VARIABLES.isPlaying = !THREE_VARIABLES.isPlaying;
        },
        'animationSpeed': 0.005,
        'trajectoryType': 'Custom',
        'generateRandomTrajectory': () => {

        },
        'eventVisionThreshold': 0.1,
        'loadObject': () => {
            document.getElementById('objFileInput').click();
        },
        'loadTexture': () => {
            document.getElementById('textureFileInput').click();
        },
        'loadTrajectory': () => {
            document.getElementById('trajectoryFileInput').click();
        }
    }

    const animationFolder = gui.addFolder( 'Animation' );

    animationFolder.add( THREE_VARIABLES.settings, 'play' ).name('Play | Pause');
    animationFolder.add( THREE_VARIABLES.settings, 'animationSpeed', 0.001, 0.5, 0.001 ).name( 'Speed' );

    const trajectoryFolder = gui.addFolder( 'Trajectory' );

    trajectoryFolder.add( THREE_VARIABLES.settings, 'trajectoryType', [ 'Custom', 'Imported', 'Live' ]).name('Type')
        .onChange( handleTrajectoryChange )
    trajectoryFolder.add( THREE_VARIABLES.settings, 'generateRandomTrajectory' ).name('Generate').disable();

    const eventVisionFolder = gui.addFolder( 'Event Vision' );

    eventVisionFolder.add( THREE_VARIABLES.settings, 'eventVisionThreshold', 0.01, 1.0, 0.01).name('Luminance Threshold')
        .onChange(() => { if ( THREE_VARIABLES.viewCamera ) THREE_VARIABLES.eventVisionPass.uniforms.threshold.value = THREE_VARIABLES.settings.eventVisionThreshold });

    const loadingFolder = gui.addFolder( 'Loading' );

    loadingFolder.add( THREE_VARIABLES.settings, 'loadObject' ).name('Load Object');
    loadingFolder.add( THREE_VARIABLES.settings, 'loadTexture' ).name('Load Texture');
    loadingFolder.add( THREE_VARIABLES.settings, 'loadTrajectory' ).name('Load Trajectory');

    animationFolder.close();
    eventVisionFolder.close();
    loadingFolder.close();
}

function handleTrajectoryChange() {
    switch (THREE_VARIABLES.settings.trajectoryType) {
        case 'Custom':
            if (THREE_VARIABLES.websocket) THREE_VARIABLES.websocket.close();

            window.addEventListener('mousedown', onMouseDown, false);
            window.addEventListener('mousemove', onMouseMove, false);
            window.addEventListener('mouseup', onMouseUp, false);

            if ( THREE_VARIABLES.drawnTrajectoryPoints.length >=2 ) updateDrawnTrajectory();

            break
        case 'Imported':
            if (THREE_VARIABLES.websocket) THREE_VARIABLES.websocket.close();

            window.removeEventListener('mousedown', onMouseDown);
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);

            updateImportedTrajectory();
            break
        case 'Live':
            window.removeEventListener('mousedown', onMouseDown);
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);

            initiateWebsocket();
    }
}