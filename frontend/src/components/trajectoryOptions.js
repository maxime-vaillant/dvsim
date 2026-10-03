import { THREE_VARIABLES } from "./variables";
import { currentOrigin } from "./origin";
import { armPicking, clearLiveStart, publishLiveOrigin, setLiveHeading } from "./liveStart";
import { getSceneFiles, saveSimSettings, sceneStore } from "./simulation";
import { generatePreview } from "../api/trajectoryPreview.js";
import { listMotionModels } from "../api/client.js";
import { GENERATED_KINDS, NO_MOTION, buildTrajectorySpec, motionOptions } from "../utils/simRequest.js";
import { ORIGIN_MODES } from "../utils/trajectoryOrigin.js";
import { clearGeneratedPaths, setGeneratedPaths, showGeneratedPaths, updateImportedTrajectory } from "../utils/trajectorySetup.js";
import { sampleConstantSpeed } from "../utils/trajectorySampling.js";
import { toLegacyRecords } from "../utils/trajectoryRecords.js";
import { downloadRecords } from "../utils/download.js";
import { notify } from "../utils/notify.js";


// Exports the drawn curve as legacy records at the current speed. `relative`: positions relative to the
// object centre, which is what "Load Trajectory" expects (so an exported file re-imports to the same path).
function exportDrawnTrajectory(relative) {
    const curve = THREE_VARIABLES.trajectoryCurve;
    const center = currentOrigin();

    if (!curve) {
        notify('Draw a trajectory first (left-drag on the overview, top-left).');
        return;
    }

    if (relative && !center) {
        notify('Load a scene first (Loading > Load scene): the export is relative to it.');
        return;
    }

    try {
        const records = sampleConstantSpeed(curve, { speed: THREE_VARIABLES.settings.sim.speed });

        downloadRecords(
            relative ? 'trajectory.json' : 'trajectory_world.json',
            toLegacyRecords(records, relative ? [center.x, center.y, center.z] : null)
        );
    } catch (error) {
        notify(error.message, { level: 'error' });
    }
}

/**
 * Adds the per-type trajectory controls to the Trajectory folder and returns the function that shows the
 * ones of the current type (call it whenever the type changes).
 */
export function addTrajectoryOptions(folder) {
    const sim = THREE_VARIABLES.settings.sim;
    const actions = {
        exportRelative: () => exportDrawnTrajectory(true),
        exportWorld: () => exportDrawnTrajectory(false),
        previewPaths: () => previewPaths(),
        pathToPlay: 1,
        pickStart: () => armPicking(),
        clearStart: () => { clearLiveStart(); heading.updateDisplay(); },
        // the heading of the live start point: read from, and written to, the picked start
        get startHeading() { return THREE_VARIABLES.liveStart ? THREE_VARIABLES.liveStart.headingDeg : 0; },
        set startHeading(degrees) { setLiveHeading(degrees); }
    };
    let generating = false;

    const custom = [
        folder.add(sim, 'speed', 0.05, 5, 0.05).name('Speed (m/s)'),
        folder.add(actions, 'exportRelative').name('Export JSON (import format)'),
        folder.add(actions, 'exportWorld').name('Export JSON (world coords)')
    ];

    // Imported files, the Live phone / headset stream and the "import format" export are relative to this origin.
    const origin = folder.add(sim, 'originMode', ORIGIN_MODES).name('Positions relative to').onChange(() => {
        if (THREE_VARIABLES.settings.trajectoryType === 'Imported') updateImportedTrajectory();
        publishLiveOrigin();
    });

    origin.domElement.title = 'Object centre: the original behaviour (right for a room). '
        + 'Floor centre: the middle of the floor, 1.5 m up (right for a street or a campus).';

    // Live: where the phone starts in the scene (its origin after "Recenter"); picked on the overview, shared with
    // the simulator so that the phone's rendered view starts there too.
    const pick = folder.add(actions, 'pickStart').name('Pick live start (click overview)');
    const heading = folder.add(actions, 'startHeading', -180, 180, 1).name('Start heading (deg)');
    const clearStart = folder.add(actions, 'clearStart').name('Clear live start');
    const liveControls = [pick, heading, clearStart];

    const kind = folder.add(sim, 'genKind', GENERATED_KINDS).name('Generator');
    const duration = folder.add(sim, 'genDuration', 1, 600, 1).name('Duration (s)');
    const count = folder.add(sim, 'genCount', 1, 100, 1).name('Sequences');
    const keyPoints = folder.add(sim, 'genKeyPoints', 2, 20, 1).name('Key points');
    const seed = folder.add(sim, 'genSeed').name('Seed (empty = random)');
    // Street walks can be played by a trained human-motion model; the list comes from the server.
    let motion = folder.add(sim, 'genMotion', motionOptions([])).name('Human motion');
    const preview = folder.add(actions, 'previewPaths').name('Preview paths');
    const pathToPlay = folder.add(actions, 'pathToPlay', 1, 2, 1).name('Path to play').onChange(selectPath);
    const generated = [kind, duration, count, seed, preview];

    // Different settings give different paths: the preview no longer shows what would be simulated.
    const onGeneratorSettingChange = () => {
        clearGeneratedPaths();
        resetPlayback();
        updateVisibility();
        if (THREE_VARIABLES.settings.trajectoryType === 'Generated') showGeneratedPaths();
    };

    [kind, duration, count, keyPoints, seed].forEach(controller => controller.onChange(onGeneratorSettingChange));
    kind.onChange(() => { if (sim.genKind === 'street') refreshMotionModels(); });
    motion.onChange(onGeneratorSettingChange);

    // lil-gui rebuilds a dropdown to change its options: the new one is put back before "Preview paths".
    async function refreshMotionModels() {
        let models;

        try {
            models = await listMotionModels();
        } catch (error) {
            return; // no server (or an old one): the dropdown keeps offering the planner's glide only
        }

        if (!models.some(model => model.id === sim.genMotion)) sim.genMotion = NO_MOTION;

        motion = motion.options(motionOptions(models));
        motion.onChange(onGeneratorSettingChange);
        folder.$children.insertBefore(motion.domElement, preview.domElement);
        updateVisibility();
    }

    function resetPlayback() {
        THREE_VARIABLES.isPlaying = false;
        THREE_VARIABLES.animationProgress = 0;
        THREE_VARIABLES.currentTrajectoryIndex = 0;
    }

    function selectPath(value) {
        THREE_VARIABLES.generatedSelected = Math.round(value) - 1;
        resetPlayback();
        showGeneratedPaths();
    }

    async function previewPaths() {
        if (generating) return;

        generating = true;
        preview.name('Generating...').disable();
        try {
            const spec = buildTrajectorySpec('Generated', { sim });
            const files = await getSceneFiles();

            notify('Generating the paths... The first preview of a scene uploads it and maps its free space: '
                + 'this can take a few seconds.', { ms: 120000 });

            const result = await generatePreview({ files, spec, sceneStore });

            // Run simulation sends the same spec: with this seed it simulates exactly these paths.
            if (spec.seed === null) {
                sim.genSeed = String(result.seed);
                seed.updateDisplay();
                saveSimSettings();
            }

            setGeneratedPaths(result.paths);
            resetPlayback();
            actions.pathToPlay = 1;
            pathToPlay.max(Math.max(2, result.paths.length)).updateDisplay();
            updateVisibility();
            showGeneratedPaths();

            const seconds = result.paths[0].durationS;
            notify(`${result.paths.length} path(s) of ${seconds} s, seed ${result.seed} (now in the Seed field: Run `
                + 'simulation simulates exactly these paths). Play follows the dark blue one.', { ms: 12000 });
        } catch (error) {
            notify(error.message, { level: 'error' });
        } finally {
            generating = false;
            preview.name('Preview paths').enable();
        }
    }

    function updateVisibility() {
        const type = THREE_VARIABLES.settings.trajectoryType;

        custom.forEach(controller => controller.show(type === 'Custom'));
        origin.show(type !== 'Generated');
        liveControls.forEach(controller => controller.show(type === 'Live'));
        heading.updateDisplay();
        generated.forEach(controller => controller.show(type === 'Generated'));
        keyPoints.show(type === 'Generated' && sim.genKind === 'spline');
        motion.show(type === 'Generated' && sim.genKind === 'street');
        pathToPlay.show(type === 'Generated' && THREE_VARIABLES.generatedPaths.length > 1);
    }

    updateVisibility();
    refreshMotionModels();

    return updateVisibility;
}
