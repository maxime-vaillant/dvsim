// Thin fetch wrappers around the simulator HTTP API (see docs/architecture.md, "HTTP API contract").
// Every call resolves the base URL through getApiBase(), so this is the only place that knows it.

import { getApiBase } from './urls.js';

export class ApiError extends Error {
    constructor(message, { status = 0, detail = null, network = false } = {}) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.detail = detail;
        this.network = network;
    }
}

function describeDetail(detail) {
    if (typeof detail === 'string') return detail;

    // FastAPI validation errors: [{ loc: [...], msg: "..." }, ...]
    if (Array.isArray(detail)) {
        return detail
            .map(item => (item && item.msg ? `${(item.loc || []).join('.')}: ${item.msg}` : JSON.stringify(item)))
            .join('; ');
    }

    return detail && typeof detail === 'object' ? JSON.stringify(detail) : '';
}

async function request(path, { method = 'GET', json, form, signal } = {}) {
    const init = { method, signal, headers: {} };

    if (json !== undefined) {
        init.headers['Content-Type'] = 'application/json';
        init.body = JSON.stringify(json);
    } else if (form) {
        init.body = form;
    }

    let response;
    try {
        response = await fetch(getApiBase() + path, init);
    } catch (error) {
        if (error && error.name === 'AbortError') throw error;

        throw new ApiError(
            `Cannot reach the simulator API at ${getApiBase()} (${error && error.message ? error.message : error})`,
            { network: true }
        );
    }

    const text = await response.text();
    let data = null;
    if (text) {
        try { data = JSON.parse(text); } catch { data = text; }
    }

    if (!response.ok) {
        const detail = data && typeof data === 'object' && !Array.isArray(data) && 'detail' in data ? data.detail : data;
        const reason = describeDetail(detail) || 'is the backend running?';

        throw new ApiError(`API error ${response.status}: ${reason}`, { status: response.status, detail });
    }

    return data;
}

export function health(options) {
    return request('/api/health', options);
}

export function listScenes(options) {
    return request('/api/scenes', options);
}

/** Trained human-motion models: [{ id, size_bytes }]. */
export function listMotionModels(options) {
    return request('/api/motion-models', options);
}

/**
 * Upload an OBJ (+ optional texture image) as a multipart form, or a multi-file scene: `extras` are the other files
 * ({ blob, name }: more OBJ files, the MTL files, the textures). Resolves to { id, name, files }.
 */
export function uploadScene({ objBlob, objName, textureBlob = null, textureName = 'texture', extras = [], name }, options = {}) {
    const form = new FormData();

    form.append('obj', objBlob, objName);
    if (textureBlob) form.append('texture', textureBlob, textureName);
    extras.forEach(extra => form.append('files', extra.blob, extra.name));
    form.append('name', name || objName);

    return request('/api/scenes', { ...options, method: 'POST', form });
}

/** payload = { scene_id, trajectory, config, visualize }. Resolves to { id, status }. */
export function createJob(payload, options = {}) {
    return request('/api/jobs', { ...options, method: 'POST', json: payload });
}

/** payload = { scene_id, trajectory (a generated spec), max_points }. Resolves to { seed, trajectory, trajectories }. */
export function generateTrajectories(payload, options = {}) {
    return request('/api/trajectories/generate', { ...options, method: 'POST', json: payload });
}

/** The live camera's origin picked in the interface: { x, y, z, heading_deg }. The phone's rendered view uses it. */
export function putLiveOrigin(origin, options = {}) {
    return request('/api/live/origin', { ...options, method: 'PUT', json: origin });
}

export function deleteLiveOrigin(options = {}) {
    return request('/api/live/origin', { ...options, method: 'DELETE' });
}

export function getJob(id, options) {
    return request(`/api/jobs/${encodeURIComponent(id)}`, options);
}

/** Cancels a running job. On a finished job the server deletes its record and files instead. */
export function cancelJob(id, options = {}) {
    return request(`/api/jobs/${encodeURIComponent(id)}`, { ...options, method: 'DELETE' });
}

export function jobFileUrl(id, path) {
    const encodedPath = String(path).split('/').map(encodeURIComponent).join('/');

    return `${getApiBase()}/api/jobs/${encodeURIComponent(id)}/files/${encodedPath}`;
}
