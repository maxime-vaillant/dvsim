import { THREE_VARIABLES } from "./variables";
import { render } from "./render";
import { currentOrigin } from "./origin";
import { getCachedExtras, getCachedFile } from "./cache";
import { initJobPanel, renderJobPanel } from "./jobPanel";
import { health } from "../api/client.js";
import { JobRunner } from "../api/jobRunner.js";
import { getApiBase } from "../api/urls.js";
import { applyEventThresholds, applyViewCameraSettings } from "../utils/cameraUpdater.js";
import { DEFAULT_SIM_SETTINGS, GT_FIELDS, buildJobRequest, restoreSimSettings } from "../utils/simRequest.js";
import { loadJSON, saveJSON } from "../utils/storage.js";
import { clearNotice, notify } from "../utils/notify.js";

const SETTINGS_KEY = 'dvsim.settings.v1';
const SCENE_KEY = 'dvsim.scene.v1';

/** Simulation settings: the defaults, overridden by what was saved in localStorage. */
export function loadSimSettings() {
    return restoreSimSettings({ ...DEFAULT_SIM_SETTINGS }, loadJSON(SETTINGS_KEY));
}

export function saveSimSettings() {
    saveJSON(SETTINGS_KEY, THREE_VARIABLES.settings.sim);
}

// The id of the last uploaded scene is remembered (with the identity of the files) across reloads;
// the runner checks that the server still has it.
export const sceneStore = {
    get: key => {
        const saved = loadJSON(SCENE_KEY);

        return saved && saved.key === key ? saved.id : null;
    },
    set: (key, id) => saveJSON(SCENE_KEY, { key, id })
};

export async function getSceneFiles() {
    const obj = await getCachedFile('obj');

    if (!obj) throw new Error('Load a scene first (Loading > Load scene).');

    // a multi-file scene takes its textures from its .mtl files: no separate texture
    const extras = await getCachedExtras();

    return { obj, texture: extras.length ? null : await getCachedFile('texture'), extras };
}

/** Adds the Simulation folder (parameters + Run button) and wires the job panel. */
export function addSimulationFolder(gui) {
    const sim = THREE_VARIABLES.settings.sim;
    const folder = gui.addFolder('Simulation');
    let runButton = null;
    let preparing = false;

    const runner = new JobRunner({
        sceneStore,
        onState: (state) => {
            renderJobPanel(state);

            if (runButton) {
                runButton.disable(runner.active);
                runButton.name(runner.active ? 'Running...' : 'Run simulation');
            }
        }
    });

    async function runSimulation() {
        if (preparing || runner.active) return;

        preparing = true;
        try {
            const request = buildJobRequest(THREE_VARIABLES.settings.trajectoryType, {
                curve: THREE_VARIABLES.trajectoryCurve,
                records: THREE_VARIABLES.trajectoryData,
                origin: currentOrigin(),
                sim
            });
            const files = await getSceneFiles();

            clearNotice();
            preparing = false;
            await runner.run({ files, request });
        } catch (error) {
            notify(error.message, { level: 'error' });
        } finally {
            preparing = false;
        }
    }

    const onThresholdChange = () => {
        applyEventThresholds();
        render();
    };

    folder.add(sim, 'width', 16, 2048, 1).name('Width (px)');
    folder.add(sim, 'height', 16, 2048, 1).name('Height (px)');
    folder.add(sim, 'fov', 10, 120, 1).name('Camera FOV (deg)')
        .onChange(() => {
            applyViewCameraSettings();
            render();
        });
    folder.add(sim, 'dt_us', 100, 100000, 100).name('dt (us)');
    folder.add(sim, 'c_pos', 0.01, 1, 0.01).name('c_pos (ON)').onChange(onThresholdChange);
    folder.add(sim, 'c_neg', 0.01, 1, 0.01).name('c_neg (OFF)').onChange(onThresholdChange);
    folder.add(sim, 'stereo').name('Stereo');

    const groundTruthFolder = folder.addFolder('Ground truth');

    GT_FIELDS.forEach(field => groundTruthFolder.add(sim, field).name(field));
    groundTruthFolder.add(sim, 'gt_every', 1, 100, 1).name('Every N steps');

    folder.add(sim, 'visualize').name('Visualize (video)');
    runButton = folder.add({ run: runSimulation }, 'run').name('Run simulation');

    initJobPanel({ onCancel: () => runner.cancel() });

    // Tell early, and without blocking the previewer, when there is no simulator to talk to.
    health().catch(error => {
        notify(`Simulator API not reachable at ${getApiBase()}: previewing works, but simulations cannot run.`, { level: 'warn' });
        console.error(error);
    });

    return folder;
}
