import { THREE_VARIABLES } from "./variables";
import {
    deleteCachedFile, loadCachedObject, loadCachedTexture, loadCachedTrajectory, putCachedExtras, putCachedFile
} from "./cache";
import { render } from "./render";
import { clearTrajectoryLine, updateImportedTrajectory } from "../utils/trajectorySetup.js";
import { classifyFiles } from "../utils/sceneBundle.js";
import { notify } from "../utils/notify.js";

// Stores the picked file (name included) in the cache, then loads it into the scene.
function bindFileInput(inputId, key, load) {
    const input = document.getElementById(inputId);

    input.addEventListener('change', async (event) => {
        const file = event.target.files[0];

        if (!file) return;

        try {
            await putCachedFile(key, file);
        } catch (error) {
            console.error('Error caching file:', error);
            notify(`Cannot keep "${file.name}" in the browser cache: ${error.message}`, { level: 'error' });
            return;
        }

        await load();

        // let the same file be picked again
        input.value = '';
    });
}

// "Load scene": everything in one selection. The .obj file(s) with their .mtl file(s) and textures (a textured scene),
// or one .obj with one image (the image covers the whole object), or just an .obj (grey). The first .obj is kept as
// the object, the other files next to it.
async function loadSceneFiles(files) {
    const { objs, mtls, images, others } = classifyFiles(files);

    if (!objs.length) {
        notify('Pick an .obj file (and, for a textured scene, its .mtl and texture images in the same selection).', { level: 'error' });
        return;
    }

    if (others.length) notify(`Ignored: ${others.map(file => file.name).join(', ')} (only .obj, .mtl and images are used).`, { level: 'warn' });

    const [first, ...otherObjs] = objs;
    let extras = [];
    let texture = null;

    if (mtls.length) {
        extras = [...otherObjs, ...mtls, ...images];     // the materials bring their textures
    } else {
        extras = otherObjs;

        if (images.length === 1 && !otherObjs.length) {
            texture = images[0];
        } else if (images.length) {
            notify('Textures need an .mtl file to say which surface they cover: select it too. No texture is used.', { level: 'warn', ms: 12000 });
        }
    }

    try {
        await putCachedFile('obj', first);
        await putCachedExtras(extras);
        if (texture) await putCachedFile('texture', texture);
        else await deleteCachedFile('texture');
    } catch (error) {
        console.error('Error caching files:', error);
        notify(`Cannot keep the files in the browser cache: ${error.message}`, { level: 'error' });
        return;
    }

    await Promise.all([loadCachedObject(), loadCachedTexture()]);
}

function bindSceneInput() {
    const input = document.getElementById('objFileInput');

    input.addEventListener('change', async (event) => {
        const files = [...event.target.files];

        if (files.length) await loadSceneFiles(files);

        input.value = '';
    });
}

export function initFileInputs() {
    bindSceneInput();
    bindFileInput('trajectoryFileInput', 'trajectory', async () => {
        const loaded = await loadCachedTrajectory();

        if (THREE_VARIABLES.settings.trajectoryType !== 'Imported') return;

        // an invalid file already told the user why: only redraw (or clear) the preview line
        if (loaded) {
            updateImportedTrajectory();
        } else {
            clearTrajectoryLine();
            render();
        }
    });
}
