// Drives one simulation job: upload the scene (once per file set), create the job, poll it until it
// ends. DOM free: progress is reported through the onState callback, so it is unit-testable in node.

import * as api from './client.js';

export const TERMINAL_STATUSES = ['done', 'failed', 'cancelled'];

const ACTIVE_PHASES = ['uploading', 'submitting', 'queued', 'running'];

export function isActivePhase(phase) {
    return ACTIVE_PHASES.includes(phase);
}

/** Identity of the loaded files: the same key means the same scene can be re-used on the server. */
export function sceneKey(files) {
    const part = file => (file ? `${file.name}:${file.size}:${file.lastModified}` : '-');
    const extras = (files.extras || []).map(part).join(',');

    return `${part(files.obj)}|${part(files.texture)}${extras ? `|${extras}` : ''}`;
}

export function memorySceneStore() {
    const scenes = new Map();

    return { get: key => scenes.get(key) || null, set: (key, id) => scenes.set(key, id) };
}

/**
 * Id of the scene of `files` on the server: the remembered one when the server still has it, else a fresh upload.
 * Resolves to { id, uploaded }.
 */
export async function ensureScene({ files, sceneStore, client = api, signal, forceUpload = false }) {
    const key = sceneKey(files);
    const known = forceUpload ? null : sceneStore.get(key);

    if (known && await sceneExists(client, known, signal)) return { id: known, uploaded: false };

    const scene = await client.uploadScene({
        objBlob: files.obj.blob,
        objName: files.obj.name,
        textureBlob: files.texture ? files.texture.blob : null,
        textureName: files.texture ? files.texture.name : undefined,
        extras: files.extras || [],
        name: files.obj.name
    }, { signal });

    sceneStore.set(key, scene.id);

    return { id: scene.id, uploaded: true };
}

async function sceneExists(client, id, signal) {
    try {
        const scenes = await client.listScenes({ signal });

        return Array.isArray(scenes) && scenes.some(scene => scene.id === id);
    } catch (error) {
        if (isAbort(error) || error.network) throw error;

        return false;
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function isAbort(error) {
    return error && error.name === 'AbortError';
}

export class JobRunner {
    /**
     * @param sceneStore  { get(key) -> id | null, set(key, id) }
     * @param onState     called with { phase, jobId, job, message, error } on every change
     * @param client      the API wrappers (injectable for tests)
     */
    constructor({ sceneStore = memorySceneStore(), onState = () => {}, client = api, pollMs = 500, maxPollFailures = 20 } = {}) {
        this.sceneStore = sceneStore;
        this.onState = onState;
        this.client = client;
        this.pollMs = pollMs;
        this.maxPollFailures = maxPollFailures;

        this.state = { phase: 'idle', jobId: null, job: null, message: '', error: null };
        this.abort = null;
        this.cancelRequested = false;
    }

    get active() {
        return isActivePhase(this.state.phase);
    }

    _emit(phase, extra = {}) {
        this.state = {
            phase,
            jobId: this.state.jobId,
            job: this.state.job,
            message: '',
            error: null,
            ...extra
        };
        this.onState(this.state);
    }

    /**
     * files   { obj: { blob, name, size, lastModified }, texture: same | null }
     * request { trajectory, config, visualize }   (scene_id is added here)
     * Resolves once the job reached a final state; never rejects, errors are reported through onState.
     */
    async run({ files, request }) {
        if (this.active) return;

        this.abort = new AbortController();
        this.cancelRequested = false;
        this.state = { phase: 'idle', jobId: null, job: null, message: '', error: null };

        try {
            this._emit('uploading');
            const scene = await this._ensureScene(files, false);

            this._emit('submitting');
            const created = await this._createJob(scene, files, request);

            this._emit('queued', { jobId: created.id });

            await this._poll(created.id);
        } catch (error) {
            if (isAbort(error)) this._emit('cancelled');
            else this._emit('error', { error: error && error.message ? error.message : String(error) });
        } finally {
            this.abort = null;
        }
    }

    async _createJob(scene, files, request) {
        const signal = this.abort.signal;

        try {
            return await this.client.createJob({ scene_id: scene.id, ...request }, { signal });
        } catch (error) {
            // The server may have lost a scene we re-used: upload it again once.
            if (scene.uploaded || ![404, 422].includes(error.status)) throw error;

            this._emit('uploading', { message: 'Re-uploading scene...' });
            const fresh = await this._ensureScene(files, true);

            this._emit('submitting');
            return this.client.createJob({ scene_id: fresh.id, ...request }, { signal });
        }
    }

    _ensureScene(files, forceUpload) {
        return ensureScene({
            files, sceneStore: this.sceneStore, client: this.client, signal: this.abort.signal, forceUpload
        });
    }

    async _poll(id) {
        let failures = 0;

        for (;;) {
            let job;

            try {
                job = await this.client.getJob(id);
                failures = 0;
            } catch (error) {
                if (error.status === 404) {
                    if (this.cancelRequested) return this._emit('cancelled');
                    return this._emit('error', { error: 'The job no longer exists on the server.' });
                }

                if (!(error.network || error.status >= 500)) throw error;

                failures += 1;
                if (failures > this.maxPollFailures) {
                    return this._emit('error', { error: `Lost connection to the simulator: ${error.message}` });
                }

                this._emit(this.state.phase, { message: 'Connection lost, retrying...' });
                await sleep(this.pollMs);
                continue;
            }

            const finished = TERMINAL_STATUSES.includes(job.status);

            this._emit(finished || job.status === 'queued' ? job.status : 'running', { job });
            if (finished) return;

            await sleep(this.pollMs);
        }
    }

    /** Aborts the upload, or asks the server to cancel the running job. */
    async cancel() {
        if (!this.active) return;

        if (this.state.phase === 'uploading') {
            this.abort.abort();
            return;
        }

        if (!this.state.jobId || this.cancelRequested) return;
        this.cancelRequested = true;

        try {
            // DELETE removes a finished job's files, so make sure it is still running first.
            const job = await this.client.getJob(this.state.jobId);

            if (TERMINAL_STATUSES.includes(job.status)) {
                this.cancelRequested = false;
                return;
            }

            await this.client.cancelJob(this.state.jobId);
        } catch (error) {
            // 404: the job is already gone, the poll loop reports it as cancelled.
            if (error.status === 404) return;

            this.cancelRequested = false;
            this._emit(this.state.phase, { message: `Cancel failed: ${error.message}` });
        }
    }
}
