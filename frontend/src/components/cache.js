import { THREE_VARIABLES } from "./variables";

export async function loadCachedObject() {
    const loadingElement = document.getElementById('loading');

    try {
        loadingElement.style.display = 'block';

        const cache = await caches.open('obj-cache');
        const cachedResponse = await cache.match('/obj');

        if (cachedResponse) {
            const blob = await cachedResponse.blob();
            const blobUrl = URL.createObjectURL(blob);

            THREE_VARIABLES.objLoader.load(blobUrl, function (obj) {
                THREE_VARIABLES.room = obj;

                URL.revokeObjectURL(blobUrl);
            }, undefined, function (error) {
                console.error('Error loading object from cache:', error);
            });
        }
    } catch (error) {
        console.error('Failed to load cached object:', error);
    }

    loadingElement.style.display = 'none';
}

export async function loadCachedTexture() {
    try {
        const cache = await caches.open('obj-cache');
        const cachedResponse = await cache.match('/texture');

        if (cachedResponse) {
            const blob = await cachedResponse.blob();
            const blobUrl = URL.createObjectURL(blob);

            THREE_VARIABLES.textureLoader.load(blobUrl, (txt => {
                THREE_VARIABLES.texture = txt;

                URL.revokeObjectURL(blobUrl);
            }));
        }
    } catch (error) {
        console.error('Failed to load cached object:', error);
    }
}

export async function loadCachedTrajectory(){
    try {
        const cache = await caches.open('obj-cache');
        const cachedResponse = await cache.match('/trajectory');

        if (cachedResponse) {
            THREE_VARIABLES.trajectoryData = await cachedResponse.json();

            const minTimestamp = THREE_VARIABLES.trajectoryData[0].timestamp;
            const maxTimestamp = THREE_VARIABLES.trajectoryData.at(-1).timestamp;

            THREE_VARIABLES.trajectoryData.forEach(data => {
                data.normalizedTimestamp = (data.timestamp - minTimestamp) / (maxTimestamp - minTimestamp);
            });
        }
    } catch (error) {
        console.error('Failed to load cached object:', error);
    }
}