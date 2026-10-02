import { THREE_VARIABLES } from "./variables";
import { render } from "./render";
import { flowToRgb } from "../utils/flowColors.js";
import { SLOT_LABELS, SLOTS, VIEW_KINDS, restoreViews } from "../utils/viewLayout.js";
import { loadJSON, saveJSON } from "../utils/storage.js";

const VIEWS_KEY = 'dvsim.views.v1';
const LEGEND_SIZE = 44;

/** Views folder: which preview each quadrant shows (the overview stays top-left) and the flow colour scale. */
export function addViewsFolder(gui) {
    THREE_VARIABLES.views = restoreViews(loadJSON(VIEWS_KEY));

    const views = THREE_VARIABLES.views;
    const folder = gui.addFolder('Views');
    const changed = () => {
        saveJSON(VIEWS_KEY, views);
        updatePanelLabels();
        render();
    };

    Object.keys(SLOTS).forEach(slot => {
        folder.add(views, slot, VIEW_KINDS).name(SLOT_LABELS[slot]).onChange(changed);
    });
    folder.add(views, 'flowScale', 0.5, 50, 0.5).name('Flow scale (px / step)').onChange(changed);

    createPanelLabels();
    updatePanelLabels();
    folder.close();
}

function createPanelLabels() {
    const container = document.getElementById('panelLabels');

    if (!container || container.childElementCount) return;

    Object.keys(SLOTS).forEach(slot => {
        const label = document.createElement('div');

        label.className = `panel-label slot-${slot}`;
        label.dataset.slot = slot;
        label.append(document.createElement('span'), flowLegend());
        container.appendChild(label);
    });
}

// Small colour wheel: a flow of `scale` pixels in each direction, as in the flow preview and the simulator videos.
function flowLegend() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const half = LEGEND_SIZE / 2;

    canvas.width = canvas.height = LEGEND_SIZE;
    canvas.title = 'Flow colours: direction = hue, white = no motion, full colour = flow scale';
    if (!context) return canvas;

    const image = context.createImageData(LEGEND_SIZE, LEGEND_SIZE);

    for (let y = 0; y < LEGEND_SIZE; y++) {
        for (let x = 0; x < LEGEND_SIZE; x++) {
            const [r, g, b] = flowToRgb(x + 0.5 - half, y + 0.5 - half, half - 2);
            const i = 4 * (y * LEGEND_SIZE + x);

            image.data.set([r, g, b, 255], i);
        }
    }
    context.putImageData(image, 0, 0);

    return canvas;
}

function updatePanelLabels() {
    const views = THREE_VARIABLES.views;

    document.querySelectorAll('#panelLabels .panel-label').forEach(label => {
        const kind = views[label.dataset.slot];
        const isFlow = kind === 'Optical flow';

        label.querySelector('span').textContent = isFlow ? `${kind} · ${views.flowScale} px / step` : kind;
        label.querySelector('canvas').hidden = !isFlow;
    });
}
