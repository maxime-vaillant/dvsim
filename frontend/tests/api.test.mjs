// API client + job runner against a small in-process fake of the HTTP contract (docs/architecture.md).
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';

import { getApiBase, getWsBase, setApiBase } from '../src/api/urls.js';
import * as client from '../src/api/client.js';
import { JobRunner, memorySceneStore, sceneKey } from '../src/api/jobRunner.js';
import { generatePreview } from '../src/api/trajectoryPreview.js';

// ---- fake server ---------------------------------------------------------------------------------

function makeFake({ steps = 3 } = {}) {
    const state = {
        scenes: new Map(), jobs: new Map(), uploads: [], jobRequests: [], generateRequests: [], cancels: 0, finishedDeletes: 0,
        dropRequests: 0, failJob: false, steps
    };

    const json = (res, status, body) => {
        res.writeHead(status, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(body));
    };
    const body = req => new Promise(resolve => {
        const chunks = [];
        req.on('data', c => chunks.push(c));
        req.on('end', () => resolve(Buffer.concat(chunks)));
    });

    const server = http.createServer(async (req, res) => {
        const { pathname } = new URL(req.url, 'http://x');

        if (state.dropRequests > 0 && pathname.startsWith('/api/jobs/')) {
            state.dropRequests -= 1;
            return req.socket.destroy();
        }

        if (pathname === '/api/health') return json(res, 200, { status: 'ok', version: 'fake' });

        if (pathname === '/api/scenes' && req.method === 'POST') {
            const text = (await body(req)).toString('latin1');
            const fields = [...text.matchAll(/name="([^"]+)"(?:; filename="([^"]*)")?\r\n(?:Content-Type:[^\r]*\r\n)?\r\n([\s\S]*?)\r\n--/g)]
                .map(m => ({ field: m[1], filename: m[2] || null, value: m[3] }));
            const id = `scene${state.scenes.size + 1}`;

            state.uploads.push(fields);
            state.scenes.set(id, { id, name: fields.find(f => f.field === 'name').value });
            return json(res, 201, { id, name: state.scenes.get(id).name, files: fields.filter(f => f.filename).map(f => f.filename) });
        }
        if (pathname === '/api/scenes') return json(res, 200, [...state.scenes.values()]);

        if (pathname === '/api/trajectories/generate' && req.method === 'POST') {
            const payload = JSON.parse((await body(req)).toString());

            state.generateRequests.push(payload);
            if (state.loseSceneOnGenerate) {          // listed a moment ago, gone now (server restart)
                state.loseSceneOnGenerate = false;
                state.scenes.delete(payload.scene_id);
            }
            if (!state.scenes.has(payload.scene_id)) return json(res, 404, { detail: 'scene not found' });
            if (payload.trajectory.duration_s > 100) return json(res, 422, { detail: 'cannot generate the trajectory: too long' });

            const seed = payload.trajectory.seed ?? 77;
            const path = k => ({
                duration_s: 2,
                points: [0, 1000, 2000].map((t, i) => ({ timestamp: 5000 + t, x: k + i, y: 1.6, z: 0, rx: 1, ry: 0, rz: 0 })).reverse()
            });
            return json(res, 200, {
                seed, trajectory: { ...payload.trajectory, seed },
                trajectories: Array.from({ length: payload.trajectory.count }, (_, k) => path(k))
            });
        }

        if (pathname === '/api/jobs' && req.method === 'POST') {
            const payload = JSON.parse((await body(req)).toString());

            state.jobRequests.push(payload);
            if (!Array.isArray(payload.config && payload.config.gt)) {
                return json(res, 422, { detail: [{ loc: ['body', 'config', 'gt'], msg: 'value is not a valid list' }] });
            }
            if (!state.scenes.has(payload.scene_id)) return json(res, 404, { detail: 'scene not found' });

            const id = `job${state.jobs.size + 1}`;
            state.jobs.set(id, { id, polls: 0, status: 'queued', cancelled: false });
            return json(res, 202, { id, status: 'queued' });
        }

        let m = /^\/api\/jobs\/([^/]+)$/.exec(pathname);
        if (m) {
            const job = state.jobs.get(m[1]);
            if (!job) return json(res, 404, { detail: 'job not found' });

            if (req.method === 'DELETE') {
                if (job.status === 'done' || job.status === 'failed') {
                    state.finishedDeletes += 1;
                    state.jobs.delete(job.id);
                } else {
                    state.cancels += 1;
                    job.cancelled = true;
                }
                return json(res, 200, {});
            }

            job.polls += 1;
            if (!job.cancelled) {
                if (job.polls > state.steps + 1) job.status = state.failJob ? 'failed' : 'done';
                else if (job.polls > 1) job.status = 'running';
            } else job.status = 'cancelled';

            const done = job.status === 'done';
            return json(res, 200, {
                id: job.id,
                status: job.status,
                error: job.status === 'failed' ? 'boom: out of GPU memory' : null,
                progress: { sequence: 0, num_sequences: 1, step: job.polls, total_steps: state.steps + 1, fraction: Math.min(1, job.polls / (state.steps + 1)) },
                outputs: done ? [{ name: 'left_events.h5', path: 'seq_000/left_events.h5', size_bytes: 12 }] : []
            });
        }

        m = /^\/api\/jobs\/([^/]+)\/files\/(.+)$/.exec(pathname);
        if (m) {
            res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
            return res.end(Buffer.from(`file:${decodeURIComponent(m[2])}`));
        }

        return json(res, 404, { detail: 'not found' });
    });

    return new Promise(resolve => server.listen(0, '127.0.0.1', () => {
        state.url = `http://127.0.0.1:${server.address().port}`;
        state.close = () => new Promise(done => { server.closeAllConnections(); server.close(done); });
        resolve(state);
    }));
}

const files = (name = 'room.obj', texture = true) => ({
    obj: { blob: new Blob(['v 0 0 0\n']), name, size: 8, lastModified: 1 },
    texture: texture ? { blob: new Blob(['PNGDATA']), name: 'tex.png', size: 7, lastModified: 2 } : null
});
const request = () => ({
    trajectory: { type: 'rectilinear', duration_s: 2, count: 1, seed: null },
    config: { gt: ['rgb'] },
    visualize: false
});
const runnerFor = (fake, extra = {}) => {
    const states = [];
    const runner = new JobRunner({ pollMs: 2, onState: s => states.push(s), sceneStore: memorySceneStore(), ...extra });

    return { runner, states, phases: () => states.map(s => s.phase).filter((p, i, all) => all[i - 1] !== p) };
};

// ---- urls --------------------------------------------------------------------------------------

test('base URL: automatic default, override, ws scheme', () => {
    setApiBase(null);
    assert.equal(getApiBase(), 'http://localhost:8000', 'outside a browser: local backend');

    globalThis.window = { location: { origin: 'https://box.local:3000' } };
    try {
        assert.equal(getApiBase(), 'https://box.local:3000', 'same origin as the page');
        assert.equal(getWsBase(), 'wss://box.local:3000');
    } finally {
        delete globalThis.window;
    }

    setApiBase('http://api.example:9000///');
    assert.equal(getApiBase(), 'http://api.example:9000');
    assert.equal(getWsBase(), 'ws://api.example:9000');
    setApiBase(null);
});

// ---- client ------------------------------------------------------------------------------------

test('client: health, scene upload (multipart), job creation, file URLs', async (t) => {
    const fake = await makeFake();
    t.after(fake.close);
    setApiBase(fake.url);

    assert.deepEqual(await client.health(), { status: 'ok', version: 'fake' });

    const scene = await client.uploadScene({
        objBlob: new Blob(['v 1 2 3']), objName: 'my room.obj', textureBlob: new Blob(['x']), textureName: 'wall.jpg', name: 'my room.obj'
    });
    assert.equal(scene.id, 'scene1');
    assert.deepEqual(scene.files, ['my room.obj', 'wall.jpg']);

    const fields = fake.uploads[0];
    assert.deepEqual(fields.map(f => f.field).sort(), ['name', 'obj', 'texture']);
    assert.equal(fields.find(f => f.field === 'obj').value, 'v 1 2 3');
    assert.equal(fields.find(f => f.field === 'name').value, 'my room.obj');

    await client.uploadScene({ objBlob: new Blob(['v']), objName: 'a.obj', name: 'a.obj' });
    assert.deepEqual(fake.uploads[1].map(f => f.field).sort(), ['name', 'obj'], 'texture is optional');

    assert.deepEqual(await client.createJob({ scene_id: 'scene1', ...request() }), { id: 'job1', status: 'queued' });
    assert.equal((await client.listScenes()).length, 2);

    assert.equal(client.jobFileUrl('job1', 'seq 0/a+b.h5'), `${fake.url}/api/jobs/job1/files/seq%200/a%2Bb.h5`);
    const download = await fetch(client.jobFileUrl('job1', 'seq_000/left_events.h5'));
    assert.equal(await download.text(), 'file:seq_000/left_events.h5');
});

test('client: HTTP errors carry the server detail, network errors are flagged', async (t) => {
    const fake = await makeFake();
    setApiBase(fake.url);

    await assert.rejects(client.createJob({ scene_id: 'x', trajectory: {}, config: {} }), (error) => {
        assert.ok(error instanceof client.ApiError);
        assert.equal(error.status, 422);
        assert.match(error.message, /config\.gt: value is not a valid list/);
        return true;
    });
    await assert.rejects(client.getJob('nope'), error => error.status === 404 && /job not found/.test(error.message));

    await fake.close();
    await assert.rejects(client.health(), (error) => {
        assert.equal(error.network, true);
        assert.match(error.message, /Cannot reach the simulator API/);
        return true;
    });
    setApiBase(null);
});

// ---- job runner --------------------------------------------------------------------------------

test('runner: upload -> job -> poll -> done, outputs downloadable', async (t) => {
    const fake = await makeFake({ steps: 3 });
    t.after(fake.close);
    setApiBase(fake.url);
    const { runner, states, phases } = runnerFor(fake);

    await runner.run({ files: files(), request: request() });

    assert.deepEqual(phases(), ['uploading', 'submitting', 'queued', 'running', 'done']);
    const last = states.at(-1);
    assert.equal(last.phase, 'done');
    assert.equal(last.jobId, 'job1');
    assert.equal(runner.active, false);

    const sent = fake.jobRequests[0];
    assert.equal(sent.scene_id, 'scene1');
    assert.deepEqual(sent.trajectory, request().trajectory);

    const [output] = last.job.outputs;
    const body = await (await fetch(client.jobFileUrl(last.jobId, output.path))).text();
    assert.equal(body, 'file:seq_000/left_events.h5');

    // progress fractions never decrease and stay within [0, 1]
    const fractions = states.filter(s => s.job).map(s => s.job.progress.fraction);
    assert.ok(fractions.every((f, i) => f >= 0 && f <= 1 && (i === 0 || f >= fractions[i - 1])));
});

test('runner: the scene is uploaded once while the same files are loaded, again when they change', async (t) => {
    const fake = await makeFake({ steps: 1 });
    t.after(fake.close);
    setApiBase(fake.url);
    const { runner } = runnerFor(fake);

    await runner.run({ files: files('room.obj'), request: request() });
    await runner.run({ files: files('room.obj'), request: request() });
    assert.equal(fake.uploads.length, 1, 'same files: no second upload');
    assert.deepEqual(fake.jobRequests.map(r => r.scene_id), ['scene1', 'scene1']);

    await runner.run({ files: files('other.obj'), request: request() });
    assert.equal(fake.uploads.length, 2, 'different files: new upload');
    assert.equal(fake.jobRequests[2].scene_id, 'scene2');

    assert.notEqual(sceneKey(files('a.obj')), sceneKey(files('a.obj', false)));
});

test('runner: a scene the server lost is uploaded again', async (t) => {
    const fake = await makeFake({ steps: 1 });
    t.after(fake.close);
    setApiBase(fake.url);
    const { runner, states } = runnerFor(fake);

    await runner.run({ files: files(), request: request() });
    fake.scenes.clear();
    await runner.run({ files: files(), request: request() });

    assert.equal(fake.uploads.length, 2);
    assert.equal(states.at(-1).phase, 'done');
});

test('runner: a failed job shows the server error', async (t) => {
    const fake = await makeFake({ steps: 2 });
    fake.failJob = true;
    t.after(fake.close);
    setApiBase(fake.url);
    const { runner, states } = runnerFor(fake);

    await runner.run({ files: files(), request: request() });

    assert.equal(states.at(-1).phase, 'failed');
    assert.equal(states.at(-1).job.error, 'boom: out of GPU memory');
    assert.equal(runner.active, false);
});

test('runner: request validation errors end in an error state (no crash)', async (t) => {
    const fake = await makeFake();
    t.after(fake.close);
    setApiBase(fake.url);
    const { runner, states } = runnerFor(fake);

    await runner.run({ files: files(), request: { ...request(), config: {} } });

    assert.equal(states.at(-1).phase, 'error');
    assert.match(states.at(-1).error, /422/);
    assert.equal(runner.active, false);
});

test('runner: API not running -> error state with a message, Run usable again', async () => {
    setApiBase('http://127.0.0.1:1');
    const { runner, states } = runnerFor(null);

    await runner.run({ files: files(), request: request() });

    assert.equal(states.at(-1).phase, 'error');
    assert.match(states.at(-1).error, /Cannot reach the simulator API/);
    assert.equal(runner.active, false);
});

test('runner: transient connection loss while polling is survived, a long one gives up', async (t) => {
    const fake = await makeFake({ steps: 3 });
    t.after(fake.close);
    setApiBase(fake.url);

    let { runner, states } = runnerFor(fake);
    const running = runner.run({ files: files(), request: request() });
    fake.dropRequests = 3;
    await running;
    assert.equal(states.at(-1).phase, 'done');
    assert.ok(states.some(s => /Connection lost/.test(s.message)));

    ({ runner, states } = runnerFor(fake, { maxPollFailures: 2 }));
    const giveUp = runner.run({ files: files(), request: request() });
    fake.dropRequests = 100;
    await giveUp;
    assert.equal(states.at(-1).phase, 'error');
    assert.match(states.at(-1).error, /Lost connection/);
});

test('runner: cancel sends DELETE to a running job and ends cancelled', async (t) => {
    const fake = await makeFake({ steps: 1000 });
    t.after(fake.close);
    setApiBase(fake.url);
    const { runner, states } = runnerFor(fake);

    const running = runner.run({ files: files(), request: request() });
    while (!states.some(s => s.phase === 'running')) await new Promise(r => setTimeout(r, 2));

    await runner.cancel();
    await running;

    assert.equal(fake.cancels, 1);
    assert.equal(states.at(-1).phase, 'cancelled');
    assert.equal(runner.active, false);
});

test('runner: cancel never deletes a job that already finished', async (t) => {
    const fake = await makeFake({ steps: 1 });
    t.after(fake.close);
    setApiBase(fake.url);
    const { runner, states } = runnerFor(fake);

    const running = runner.run({ files: files(), request: request() });
    while (!states.some(s => s.phase === 'queued')) await new Promise(r => setTimeout(r, 1));
    // let the fake job finish behind the runner's back, then click Cancel
    fake.jobs.get('job1').polls = 100;
    await runner.cancel();
    await running;

    assert.equal(fake.finishedDeletes, 0, 'the finished job and its files are kept');
    assert.equal(fake.cancels, 0);
    assert.equal(states.at(-1).phase, 'done');
});

test('runner: a second run() while active is ignored', async (t) => {
    const fake = await makeFake({ steps: 5 });
    t.after(fake.close);
    setApiBase(fake.url);
    const { runner } = runnerFor(fake);

    const first = runner.run({ files: files(), request: request() });
    await runner.run({ files: files(), request: request() });
    await first;

    assert.equal(fake.jobRequests.length, 1);
});

test.after(() => setApiBase(null));

// ---- trajectory preview ------------------------------------------------------------------------

test('preview: uploads the scene once, returns the seed and playable paths', async (t) => {
    const fake = await makeFake();
    t.after(fake.close);
    setApiBase(fake.url);
    const sceneStore = memorySceneStore();
    const spec = { type: 'street', duration_s: 2, count: 2, seed: null };

    const first = await generatePreview({ files: files(), spec, sceneStore, maxPoints: 500 });
    assert.equal(first.seed, 77, 'the seed the server drew');
    assert.equal(first.paths.length, 2);
    assert.equal(first.paths[1].durationS, 2);
    assert.deepEqual(first.paths[1].records.map(r => r.normalizedTimestamp), [0, 0.5, 1], 'sorted, normalised');
    assert.deepEqual(first.paths[1].records.map(r => r.x), [1, 2, 3]);
    assert.deepEqual(fake.generateRequests[0], { scene_id: 'scene1', trajectory: spec, max_points: 500 });

    await generatePreview({ files: files(), spec: { ...spec, seed: 4 }, sceneStore });
    assert.equal(fake.uploads.length, 1, 'the same files reuse the uploaded scene');
    assert.equal(fake.generateRequests[1].trajectory.seed, 4);

    // the runner shares the store: simulating after a preview does not upload again
    const { runner } = runnerFor(fake, { sceneStore });
    await runner.run({ files: files(), request: request() });
    assert.equal(fake.uploads.length, 1);
});

test('preview: a scene the server lost is uploaded again once; generation errors are reported', async (t) => {
    const fake = await makeFake();
    t.after(fake.close);
    setApiBase(fake.url);
    const sceneStore = memorySceneStore();
    const spec = { type: 'spline', duration_s: 2, count: 1, seed: 1, n_key_points: 4 };

    await generatePreview({ files: files(), spec, sceneStore });
    fake.loseSceneOnGenerate = true;
    const result = await generatePreview({ files: files(), spec, sceneStore });
    assert.equal(result.paths.length, 1);
    assert.equal(fake.uploads.length, 2, 'uploaded again');
    assert.equal(fake.generateRequests.length, 3, 'asked again after the upload');
    assert.ok(fake.scenes.has(sceneStore.get(sceneKey(files()))), 'the new id is remembered');

    await assert.rejects(
        generatePreview({ files: files(), spec: { ...spec, duration_s: 500 }, sceneStore }),
        /API error 422: cannot generate the trajectory: too long/
    );
});

// ---- multi-file scenes ----------------------------------------------------------------------------

test('multi-file scene: the other files go up as repeated "files" fields; the scene key follows them', async (t) => {
    const fake = await makeFake();
    t.after(fake.close);
    setApiBase(fake.url);

    const extras = [
        { blob: new Blob(['newmtl a\nmap_Kd t.jpg\n']), name: 'scene.mtl', size: 22, lastModified: 3 },
        { blob: new Blob(['JPEGDATA']), name: 'Textures/t.jpg', size: 8, lastModified: 4 }
    ];
    await client.uploadScene({ objBlob: new Blob(['v 1 2 3']), objName: 'a.obj', extras, name: 'a.obj' });

    const fields = fake.uploads[0];
    assert.deepEqual(fields.map(f => f.field), ['obj', 'files', 'files', 'name']);
    assert.deepEqual(fields.filter(f => f.field === 'files').map(f => f.filename), ['scene.mtl', 'Textures/t.jpg']);
    assert.equal(fields.filter(f => f.field === 'files')[1].value, 'JPEGDATA');

    const base = files('a.obj', false);
    assert.equal(sceneKey(base), sceneKey({ ...base, extras: [] }), 'no extras: the key of a single OBJ');
    assert.notEqual(sceneKey({ ...base, extras }), sceneKey(base));
    assert.notEqual(sceneKey({ ...base, extras }), sceneKey({ ...base, extras: [extras[0]] }));
    assert.notEqual(sceneKey({ ...base, extras }), sceneKey({ ...base, extras: [extras[0], { ...extras[1], lastModified: 5 }] }));

    // the runner uploads the extras too, once per file set
    const { runner } = runnerFor(fake);
    await runner.run({ files: { ...files('b.obj', false), extras }, request: request() });
    assert.deepEqual(fake.uploads[1].filter(f => f.field === 'files').map(f => f.filename), ['scene.mtl', 'Textures/t.jpg']);
});
