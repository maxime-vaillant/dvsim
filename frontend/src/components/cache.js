import * as THREE from "three";
import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

import { THREE_VARIABLES } from "./variables";
import { classifyFiles, fileName, missingFiles, referencedMaterialLibraries } from "../utils/sceneBundle.js";
import { withPlaybackTimes } from "../utils/trajectoryRecords.js";
import { clearNotice, notify } from "../utils/notify.js";

const CACHE_NAME = 'obj-cache';

// The Cache API only exists in secure contexts (https or localhost): keep files in memory otherwise.
const memoryFiles = new Map();

const IMAGE_EXTENSIONS = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/bmp': 'bmp' };

function defaultName(key, blob) {
    if (key === 'obj') return 'scene.obj';
    if (key === 'texture') return `texture.${IMAGE_EXTENSIONS[blob.type] || 'png'}`;

    return `${key}.json`;
}

function decodeName(value) {
    try {
        return value ? decodeURIComponent(value) : '';
    } catch {
        return value || '';
    }
}

/** Stores a user-picked file under `key` ('obj' | 'texture' | 'trajectory') together with its name. */
export async function putCachedFile(key, file) {
    if (typeof caches === 'undefined') {
        memoryFiles.set(key, file);
        return;
    }

    const cache = await caches.open(CACHE_NAME);

    await cache.delete(`/${key}`);
    await cache.put(`/${key}`, new Response(file, {
        headers: {
            'Content-Type': file.type || 'application/octet-stream',
            'X-File-Name': encodeURIComponent(file.name),
            'X-File-Modified': String(file.lastModified)
        }
    }));
}

export async function deleteCachedFile(key) {
    if (typeof caches === 'undefined') {
        memoryFiles.delete(key);
        return;
    }

    const cache = await caches.open(CACHE_NAME);

    await cache.delete(`/${key}`);
}

/** Resolves to { blob, name, size, lastModified }, or null when nothing is stored under `key`. */
export async function getCachedFile(key) {
    if (typeof caches === 'undefined') {
        const file = memoryFiles.get(key);

        return file ? { blob: file, name: file.name, size: file.size, lastModified: file.lastModified } : null;
    }

    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(`/${key}`);

    if (!response) return null;

    const blob = await response.blob();

    return {
        blob,
        name: decodeName(response.headers.get('X-File-Name')) || defaultName(key, blob),
        size: blob.size,
        lastModified: Number(response.headers.get('X-File-Modified')) || 0
    };
}

// The other files of a multi-file scene (more OBJ files, the MTL files, the textures) are kept next to the first OBJ,
// each under `extra/<index>` with its own name: [{ blob, name, size, lastModified }], in the order they were stored.
const EXTRA_PREFIX = '/extra/';
const memoryExtras = [];

/** Replaces the extra files of the scene with `files` (an empty list removes them). */
export async function putCachedExtras(files) {
    if (typeof caches === 'undefined') {
        memoryExtras.splice(0, memoryExtras.length, ...files);
        return;
    }

    const cache = await caches.open(CACHE_NAME);

    for (const request of await cache.keys()) {
        if (new URL(request.url).pathname.startsWith(EXTRA_PREFIX)) await cache.delete(request);
    }

    for (const [index, file] of files.entries()) {
        await cache.put(`${EXTRA_PREFIX}${index}`, new Response(file, {
            headers: {
                'Content-Type': file.type || 'application/octet-stream',
                'X-File-Name': encodeURIComponent(file.name),
                'X-File-Modified': String(file.lastModified)
            }
        }));
    }
}

export async function getCachedExtras() {
    if (typeof caches === 'undefined') {
        return memoryExtras.map(file => ({ blob: file, name: file.name, size: file.size, lastModified: file.lastModified }));
    }

    const cache = await caches.open(CACHE_NAME);
    const extras = [];

    for (const request of await cache.keys()) {
        const path = new URL(request.url).pathname;

        if (!path.startsWith(EXTRA_PREFIX)) continue;

        const response = await cache.match(request);
        const blob = await response.blob();

        extras.push({
            index: Number(path.slice(EXTRA_PREFIX.length)),
            blob,
            name: decodeName(response.headers.get('X-File-Name')),
            size: blob.size,
            lastModified: Number(response.headers.get('X-File-Modified')) || 0
        });
    }

    return extras.sort((a, b) => a.index - b.index);
}

/** Forgets the loaded scene: the OBJ, its texture and the other files of a multi-file scene. */
export async function clearCachedScene() {
    await deleteCachedFile('obj');
    await deleteCachedFile('texture');
    await putCachedExtras([]);
}

export function disposeTexture() {
    if (THREE_VARIABLES.texture) {
        THREE_VARIABLES.texture.dispose();
        THREE_VARIABLES.texture = null;
    }
}

export function disposeRoom() {
    const room = THREE_VARIABLES.room;

    THREE_VARIABLES.roomHasMaterials = false;

    if (!room) return;

    THREE_VARIABLES.scene.remove(room);
    room.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) [].concat(child.material).forEach(material => material.dispose());
    });
    THREE_VARIABLES.room = null;
}

// A multi-file scene: the OBJ files get the materials (and textures) of their MTL files. The textures are found by
// file name through the loading manager, whatever directories the MTL names them with.
async function loadBundle(first, extras, loadingElement) {
    const { objs, mtls, images } = classifyFiles([first, ...extras]);
    const manager = THREE_VARIABLES.objLoader.manager;
    const urls = new Map(images.map(image => [image.name.toLowerCase(), URL.createObjectURL(image.blob)]));
    const previousOnLoad = manager.onLoad;
    const texts = new Map();

    loadingElement.style.display = 'block';
    manager.setURLModifier(url => urls.get(fileName(decodeName(url)).toLowerCase()) || url);
    // runs when the textures are in: the same hook that finishes loading a single OBJ (loadModel in App.js)
    manager.onLoad = () => {
        manager.onLoad = previousOnLoad;
        manager.setURLModifier(undefined);
        urls.forEach(url => URL.revokeObjectURL(url));
        loadingElement.style.display = 'none';
        if (previousOnLoad) previousOnLoad();
    };
    manager.itemStart('bundle');   // keeps the manager "loading" until the objects are parsed, even without textures

    try {
        const libraries = new Map();

        for (const mtl of mtls) {
            const text = await mtl.blob.text();
            const creator = new MTLLoader(manager).parse(text, '');

            creator.preload();
            libraries.set(fileName(mtl.name).toLowerCase(), creator);
            texts.set(mtl.name, text);
        }

        const room = new THREE.Group();
        const objTexts = [];

        for (const obj of objs) {
            const text = await obj.blob.text();
            const loader = new OBJLoader(manager);
            const [library] = referencedMaterialLibraries(text.slice(0, 1 << 16));
            const creator = library && libraries.get(library.toLowerCase());

            if (creator) loader.setMaterials(creator);
            room.add(loader.parse(text));
            objTexts.push({ name: obj.name, text: text.slice(0, 1 << 16) });
        }

        const missing = missingFiles({
            objs: objTexts,
            mtls: mtls.map(mtl => ({ name: mtl.name, text: texts.get(mtl.name) })),
            available: [first, ...extras].map(file => file.name)
        });

        if (missing.length) {
            notify(`Missing file${missing.length > 1 ? 's' : ''}: ${missing.map(m => `${m.name} (used by ${m.neededBy})`).join(', ')}. `
                + 'Select them together with the .obj in the file dialog.', { level: 'warn', ms: 15000 });
        }

        THREE_VARIABLES.room = room;
        THREE_VARIABLES.roomHasMaterials = true;
    } catch (error) {
        console.error('Error loading the scene files:', error);
        notify(`Cannot load "${first.name}" with its materials: ${error.message}`, { level: 'error' });
    } finally {
        manager.itemEnd('bundle');
    }
}

export async function loadCachedObject() {
    const loadingElement = document.getElementById('loading');

    disposeRoom();

    try {
        const file = await getCachedFile('obj');

        if (!file) return;

        const extras = await getCachedExtras();

        if (extras.length) {
            await loadBundle(file, extras, loadingElement);
            return;
        }

        const blobUrl = URL.createObjectURL(file.blob);
        const done = () => {
            URL.revokeObjectURL(blobUrl);
            loadingElement.style.display = 'none';
        };

        loadingElement.style.display = 'block';

        THREE_VARIABLES.objLoader.load(blobUrl, (obj) => {
            THREE_VARIABLES.room = obj;
            done();
        }, undefined, (error) => {
            console.error('Error loading object from cache:', error);
            notify(`Cannot load "${file.name}" as an OBJ file.`, { level: 'error' });
            done();
        });
    } catch (error) {
        console.error('Failed to load cached object:', error);
        loadingElement.style.display = 'none';
    }
}

export async function loadCachedTexture() {
    try {
        disposeTexture();

        const file = await getCachedFile('texture');

        if (!file) return;

        const blobUrl = URL.createObjectURL(file.blob);

        THREE_VARIABLES.textureLoader.load(blobUrl, (texture) => {
            THREE_VARIABLES.texture = texture;
            URL.revokeObjectURL(blobUrl);
        }, undefined, (error) => {
            console.error('Error loading texture from cache:', error);
            notify(`Cannot load "${file.name}" as an image.`, { level: 'error' });
            URL.revokeObjectURL(blobUrl);
        });
    } catch (error) {
        console.error('Failed to load cached texture:', error);
    }
}

/** Loads and validates the cached trajectory file. Resolves to true when usable trajectory data is set. */
export async function loadCachedTrajectory() {
    THREE_VARIABLES.trajectoryData = [];
    THREE_VARIABLES.trajectoryDurationS = 0;

    try {
        const file = await getCachedFile('trajectory');

        if (!file) return false;

        let data;
        try {
            data = JSON.parse(await file.blob.text());
        } catch {
            throw new Error('the file is not valid JSON.');
        }

        const { records, durationS } = withPlaybackTimes(data);

        THREE_VARIABLES.trajectoryData = records;
        THREE_VARIABLES.trajectoryDurationS = durationS;
        clearNotice();

        return true;
    } catch (error) {
        console.error('Failed to load cached trajectory:', error);
        notify(`Cannot use the trajectory file: ${error.message}`, { level: 'error' });
        await deleteCachedFile('trajectory').catch(() => {});

        return false;
    }
}
