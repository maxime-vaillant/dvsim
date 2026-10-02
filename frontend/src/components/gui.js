import { GUI } from 'three/addons/libs/lil-gui.module.min.js';

import { THREE_VARIABLES } from "./variables";
import { setDrawingEnabled } from "./listeners";
import { initiateWebsocket } from "./websocket";
import { addSimulationFolder, loadSimSettings, saveSimSettings } from "./simulation";
import { addTrajectoryOptions } from "./trajectoryOptions";
import { addViewsFolder } from "./views";
import { resetScene } from "./resetScene";
import { selectedGeneratedPath, showDrawnTrajectory, showGeneratedPaths, updateImportedTrajectory } from "../utils/trajectorySetup.js";
import { notify } from "../utils/notify.js";


const GENERATED_HINT = 'Click "Preview paths" to see generated trajectories (and play them) before simulating; or pick another Type.';
const DRAW_HINT = 'Custom trajectory: left-drag on the overview (top-left) to draw, right-drag to rotate, wheel to zoom.';

let updateTrajectoryOptions = () => {};

export function initGUI( ) {
    const guiContainer = document.getElementById('guiContainer');
    const gui = new GUI({ container: guiContainer });

    gui.title('Settings');

    THREE_VARIABLES.settings = {
        'play': () => {
            if ( !canPlay() ) return;

            THREE_VARIABLES.isPlaying = !THREE_VARIABLES.isPlaying;
        },
        'animationSpeed': 1,
        'trajectoryType': 'Generated',
        'sim': loadSimSettings(),
        'loadScene': () => {
            document.getElementById('objFileInput').click();
        },
        'loadTrajectory': () => {
            document.getElementById('trajectoryFileInput').click();
        },
        'resetScene': () => resetScene()
    }

    const animationFolder = gui.addFolder( 'Animation' );

    animationFolder.add( THREE_VARIABLES.settings, 'play' ).name('Play | Pause');
    animationFolder.add( THREE_VARIABLES.settings, 'animationSpeed', 0.1, 4, 0.1 ).name( 'Playback speed (x)' );

    const trajectoryFolder = gui.addFolder( 'Trajectory' );

    trajectoryFolder.add( THREE_VARIABLES.settings, 'trajectoryType', [ 'Custom', 'Imported', 'Generated', 'Live' ]).name('Type')
        .onChange( handleTrajectoryChange )
    updateTrajectoryOptions = addTrajectoryOptions( trajectoryFolder );

    addSimulationFolder( gui );
    addViewsFolder( gui );

    const loadingFolder = gui.addFolder( 'Loading' );

    loadingFolder.add( THREE_VARIABLES.settings, 'loadScene' ).name('Load scene (.obj + textures)');
    loadingFolder.add( THREE_VARIABLES.settings, 'loadTrajectory' ).name('Load Trajectory');
    loadingFolder.add( THREE_VARIABLES.settings, 'resetScene' ).name('Reset scene');

    animationFolder.close();

    // remember the simulation settings between visits
    gui.onChange( saveSimSettings );

    notify( GENERATED_HINT );
}

// Whether the current trajectory type has something to play; tells the user what is missing otherwise.
function canPlay() {
    switch (THREE_VARIABLES.settings.trajectoryType) {
        case 'Custom':
            if ( THREE_VARIABLES.trajectoryCurve ) return true;
            notify('Draw a trajectory first (left-drag on the overview, top-left).');
            return false
        case 'Imported':
            if ( THREE_VARIABLES.trajectoryData.length >= 2 ) return true;
            notify('Load a trajectory file first (Loading > Load Trajectory).');
            return false
        case 'Generated':
            if ( selectedGeneratedPath() ) return true;
            notify('Click "Preview paths" first (Trajectory folder) to generate the paths to play.');
            return false
        default:
            notify('Live trajectories follow the headset: there is nothing to play.');
            return false
    }
}

function handleTrajectoryChange() {
    const type = THREE_VARIABLES.settings.trajectoryType;

    THREE_VARIABLES.isPlaying = false;
    THREE_VARIABLES.animationProgress = 0;
    THREE_VARIABLES.trajectoryDistance = 0;
    THREE_VARIABLES.currentTrajectoryIndex = 0;

    if (THREE_VARIABLES.websocket) {
        THREE_VARIABLES.websocket.close();
        THREE_VARIABLES.websocket = null;
    }

    setDrawingEnabled( type === 'Custom' );
    updateTrajectoryOptions();

    switch (type) {
        case 'Custom':
            showDrawnTrajectory();
            notify( DRAW_HINT );
            break
        case 'Imported':
            updateImportedTrajectory();
            break
        case 'Generated':
            showGeneratedPaths();
            notify( selectedGeneratedPath()
                ? 'Play follows the dark blue path; "Path to play" picks another one.'
                : GENERATED_HINT );
            break
        case 'Live':
            initiateWebsocket();
    }
}
