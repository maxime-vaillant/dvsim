// Phone as a pose source for the simulator's Live view: it streams the camera pose to the /headset relay (the same
// protocol as the AR headset page), and the interface's Live mode (Trajectory > Type > Live) follows it.
//   AR (Android Chrome, ARCore): full 6DoF tracking through WebXR, metres, origin where the session starts.
//   Sensors (any phone, e.g. iPhone where there is no WebXR): orientation from the motion sensors, position from the
//   on-screen sticks (integrating the accelerometer drifts by metres within seconds, so it is not used).
// Recenter makes the current pose the origin looking along -Z; Record saves the walk as a trajectory on the server.
// Simulator view: the poses also go to /api/live, which renders the scene with the simulator's own renderer
// (exact RGB, events, depth, flow) and sends JPEG frames back, drawn full screen behind the controls.

import * as THREE from 'three';

import { listScenes } from './api/client.js';
import { getWsBase } from './api/urls.js';
import { endpoints } from '../endpoints.js';
import {
    LIVE_VIEWS, LatencyTracker, SIZES, VIEW_LABELS, parseFrame, poseMessage, startMessage, viewMessage
} from './utils/liveFrames.js';
import { Rebaser, deviceQuaternion, horizontalHeading, toMessage, walkStep } from './utils/phonePose.js';

const SEND_INTERVAL_MS = 1000 / 30;
const READOUT_INTERVAL_MS = 100;
const RECONNECT_MS = 2000;
const SENSOR_TIMEOUT_MS = 2500;
const SETTINGS_KEY = 'dvsim.phone.v1';

const $ = id => document.getElementById(id);
const ZERO = new THREE.Vector3();

const state = {
    mode: 'idle',                   // 'idle' | 'sensors' | 'ar'
    socket: null,
    connected: false,
    recording: false,
    recordStartedAt: 0,
    rebase: new Rebaser(),
    needsRecenter: true,
    orientation: null,              // latest { alpha, beta, gamma } (sensors)
    walker: { x: 0, y: 0, z: 0 },   // position integrated from the sticks (sensors)
    stick: { x: 0, y: 0 },
    vertical: 0,
    speed: 1.4,
    lastSentAt: 0,
    lastReadoutAt: 0,
    lastFrameAt: 0,
    wakeLock: null,
    xr: null,                       // { renderer, scene, camera, session }
    live: {                         // the simulator view
        socket: null,
        wanted: false,              // keep (re)connecting while a session is running
        ready: false,
        seq: 0,
        view: 'rgb',
        stats: null,
        decoding: false,
        retry: null,
        latency: new LatencyTracker()
    }
};

function loadSettings() {
    try {
        return JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {};
    } catch {
        return {};
    }
}

function saveSettings() {
    try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify({
            scene: $('liveScene').value, view: $('liveView').value, origin: $('liveOrigin').value, size: $('liveSize').value
        }));
    } catch {
        // storage blocked (private mode): the choices are just not remembered
    }
}

// ---- relay -----------------------------------------------------------------------------------------------------

function connect() {
    const socket = new WebSocket(getWsBase() + endpoints.HEADSET);

    state.socket = socket;
    socket.addEventListener('open', () => { state.connected = true; refresh(); });
    socket.addEventListener('close', () => {
        state.connected = false;
        state.recording = false;
        refresh();
        setTimeout(connect, RECONNECT_MS);
    });
    socket.addEventListener('error', () => socket.close());
}

function send(action, pose = null) {
    if (!state.connected) return false;

    state.socket.send(toMessage(action, performance.now(), pose));

    return true;
}

// ---- pose output -----------------------------------------------------------------------------------------------

function publish(pose, now) {
    if (now - state.lastSentAt >= SEND_INTERVAL_MS) {
        state.lastSentAt = now;
        send('stream_data', pose);
        sendLivePose(pose, now);
    }

    if (now - state.lastReadoutAt >= READOUT_INTERVAL_MS) {
        state.lastReadoutAt = now;
        $('phonePose').textContent = formatPose(pose);
        if (state.recording) $('phoneRecording').textContent = `REC ${formatClock((now - state.recordStartedAt) / 1000)}`;
    }
}

function formatPose(pose) {
    const heading = Math.atan2(pose.rx, -pose.rz) * 180 / Math.PI;
    const pitch = Math.asin(Math.max(-1, Math.min(1, pose.ry))) * 180 / Math.PI;
    const f = value => value.toFixed(2).padStart(6);

    return `x ${f(pose.x)}  y ${f(pose.y)}  z ${f(pose.z)} m\nheading ${Math.round(heading)}°  pitch ${Math.round(pitch)}°`;
}

function formatClock(seconds) {
    return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

// ---- sensors + sticks mode -------------------------------------------------------------------------------------

function screenAngle() {
    if (screen.orientation && typeof screen.orientation.angle === 'number') return screen.orientation.angle;

    return window.orientation || 0;
}

function onOrientation(event) {
    if (event.alpha === null || event.beta === null || event.gamma === null) return;

    state.orientation = { alpha: event.alpha, beta: event.beta, gamma: event.gamma };
}

async function startSensors() {
    if (!window.isSecureContext) {
        return help('The motion sensors are blocked on a page that is not HTTPS: open this page with https://.');
    }

    const Orientation = window.DeviceOrientationEvent;

    if (!Orientation) return help('This browser has no orientation sensor.');

    if (typeof Orientation.requestPermission === 'function') {   // iOS asks on a tap
        const answer = await Orientation.requestPermission().catch(() => 'denied');

        if (answer !== 'granted') {
            return help('Motion access was denied. Quit the browser, reopen this page and allow "Motion & Orientation".');
        }
    }

    window.addEventListener('deviceorientation', onOrientation);
    state.orientation = null;
    state.walker = { x: 0, y: 0, z: 0 };
    state.needsRecenter = true;
    enter('sensors');
    setTimeout(() => {
        if (state.mode === 'sensors' && !state.orientation) {
            help('No orientation data is coming in. Is "Motion & Orientation" allowed for this site, and the page HTTPS?');
            leave();
        }
    }, SENSOR_TIMEOUT_MS);
    state.lastFrameAt = performance.now();
    requestAnimationFrame(sensorFrame);
}

function sensorFrame(now) {
    if (state.mode !== 'sensors') return;

    requestAnimationFrame(sensorFrame);

    const dt = Math.min((now - state.lastFrameAt) / 1000, 0.1);

    state.lastFrameAt = now;

    if (!state.orientation) return;

    const { alpha, beta, gamma } = state.orientation;
    const quaternion = deviceQuaternion(alpha, beta, gamma, screenAngle());

    if (state.needsRecenter) {
        state.rebase.recenter(ZERO, quaternion);
        state.walker = { x: 0, y: 0, z: 0 };
        state.needsRecenter = false;
    }

    const pose = state.rebase.apply(ZERO, quaternion);
    // walk along where the screen top points when looking at the ground: the view direction itself is unstable there
    const heading = horizontalHeading(quaternion) - state.rebase.heading;

    state.walker = walkStep(state.walker, { rx: Math.sin(heading), rz: -Math.cos(heading) }, state.stick, state.vertical, dt, state.speed);
    Object.assign(pose, state.walker);

    publish(pose, now);
}

// ---- AR (WebXR) mode -------------------------------------------------------------------------------------------

async function arSupported() {
    if (!navigator.xr || !window.isSecureContext) return false;

    try {
        return await navigator.xr.isSessionSupported('immersive-ar');
    } catch {
        return false;
    }
}

// Not three's ARButton: it hides the overlay root when the session ends, and the page UI is that root.
async function startAr() {
    try {
        if (!state.xr) {
            const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });

            renderer.xr.enabled = true;
            renderer.xr.setReferenceSpaceType('local');
            state.xr = { renderer, scene: new THREE.Scene(), camera: new THREE.PerspectiveCamera(), session: null };
        }

        const session = await navigator.xr.requestSession('immersive-ar', {
            optionalFeatures: ['dom-overlay'],
            domOverlay: { root: $('phoneUi') }
        });

        await state.xr.renderer.xr.setSession(session);
        state.xr.session = session;
        session.addEventListener('end', () => { state.xr.session = null; leave(); });
        state.needsRecenter = true;
        enter('ar');
        state.xr.renderer.setAnimationLoop(arFrame);
    } catch (error) {
        help(`Could not start AR: ${error && error.message ? error.message : error}`);
    }
}

function arFrame(time, frame) {
    if (!frame || state.mode !== 'ar') return;

    const { renderer, scene, camera } = state.xr;

    renderer.render(scene, camera);   // updates the camera to the tracked head pose

    if (state.needsRecenter) {
        state.rebase.recenter(camera.position, camera.quaternion);
        state.needsRecenter = false;
    }

    publish(state.rebase.apply(camera.position, camera.quaternion), performance.now());
}

// ---- simulator view (/api/live) --------------------------------------------------------------------------------

function liveSettings() {
    return { sceneId: $('liveScene').value, size: $('liveSize').value, origin: $('liveOrigin').value, view: state.live.view };
}

function setLiveStatus(text, bad = false) {
    const chip = $('phoneLive');

    chip.hidden = !text;
    chip.textContent = text;
    chip.className = `chip${bad ? ' bad' : ''}`;
}

function openLive() {
    const live = state.live;
    const settings = liveSettings();

    if (!settings.sceneId) return;

    live.wanted = true;
    live.ready = false;
    live.view = $('liveView').value;
    setLiveStatus('loading the scene...');

    const socket = new WebSocket(getWsBase() + '/api/live');

    socket.binaryType = 'arraybuffer';
    live.socket = socket;
    socket.addEventListener('open', () => socket.send(startMessage({ ...settings, view: live.view })));
    socket.addEventListener('message', event => (typeof event.data === 'string' ? onLiveText(event.data) : onLiveFrame(event.data)));
    socket.addEventListener('close', () => {
        live.ready = false;
        if (live.socket === socket) live.socket = null;
        if (live.wanted) {
            setLiveStatus('simulator view lost, retrying...', true);
            live.retry = setTimeout(() => { if (live.wanted && !live.socket) openLive(); }, RECONNECT_MS);
        }
    });
    socket.addEventListener('error', () => socket.close());
    $('liveFrame').hidden = false;
    $('phoneUi').classList.add('live');
    showViews();
}

function closeLive() {
    const live = state.live;

    live.wanted = false;
    live.ready = false;
    clearTimeout(live.retry);
    if (live.socket) live.socket.close();
    live.socket = null;
    live.stats = null;
    setLiveStatus('');
    $('liveFrame').hidden = true;
    $('phoneUi').classList.remove('live');
    $('phoneViews').hidden = true;
}

function onLiveText(text) {
    let message;

    try {
        message = JSON.parse(text);
    } catch {
        return;
    }

    const live = state.live;

    if (message.type === 'loading') setLiveStatus('loading the scene...');
    else if (message.type === 'ready') {
        live.ready = true;
        setLiveStatus('simulator view');
    } else if (message.type === 'stats') live.stats = message;
    else if (message.type === 'error') {
        live.wanted = false;               // fatal: the server stopped this session, do not retry in a loop
        setLiveStatus(message.message, true);
    }
}

async function onLiveFrame(buffer) {
    const frame = parseFrame(buffer);
    const live = state.live;

    if (!frame || live.decoding) return;   // still drawing the previous one: drop this frame

    live.decoding = true;
    try {
        const ms = live.latency.received(frame.seq, performance.now());
        const bitmap = await createImageBitmap(new Blob([frame.jpeg], { type: 'image/jpeg' }));
        const canvas = $('liveFrame');

        if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
        }
        canvas.getContext('2d').drawImage(bitmap, 0, 0);
        bitmap.close();

        const average = live.latency.average;
        const stats = live.stats ? ` · ${live.stats.fps} fps · render ${live.stats.render_ms} ms` : '';

        if (ms !== null && average !== null) setLiveStatus(`${VIEW_LABELS[live.view]} · ${Math.round(average)} ms${stats}`);
    } catch {
        // a damaged frame: the next one replaces it
    } finally {
        live.decoding = false;
    }
}

function sendLivePose(pose, now) {
    const live = state.live;

    if (!live.ready || !live.socket || live.socket.readyState !== WebSocket.OPEN) return;

    live.seq += 1;
    live.latency.sent(live.seq, now);
    live.socket.send(poseMessage(live.seq, pose));
}

function showViews() {
    const row = $('phoneViews');

    if (!row.childElementCount) {
        LIVE_VIEWS.forEach(view => {
            const button = document.createElement('button');

            button.type = 'button';
            button.dataset.view = view;
            button.textContent = VIEW_LABELS[view];
            button.addEventListener('click', () => {
                state.live.view = view;
                if (state.live.socket && state.live.socket.readyState === WebSocket.OPEN) state.live.socket.send(viewMessage(view));
                showViews();
            });
            row.appendChild(button);
        });
    }

    row.hidden = false;
    [...row.children].forEach(button => button.classList.toggle('active', button.dataset.view === state.live.view));
}

async function fillLiveSetup() {
    const saved = loadSettings();
    const scenes = $('liveScene');
    const none = new Option('none: only stream the pose', '');

    scenes.add(none);
    $('liveView').replaceChildren(...LIVE_VIEWS.map(view => new Option(VIEW_LABELS[view] + (view === 'grid' ? ' (RGB, events, depth, flow)' : ''), view)));
    $('liveSize').replaceChildren(...Object.entries(SIZES).map(([key, size]) => new Option(size.label, key)));

    try {
        const list = await listScenes();

        list.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
        list.forEach(scene => scenes.add(new Option(scene.name, scene.id)));
    } catch {
        none.text = 'none (the simulator is not reachable)';
    }

    const known = [...scenes.options].some(option => option.value === saved.scene);

    scenes.value = known ? saved.scene : (scenes.options[1] ? scenes.options[1].value : '');
    if (saved.view) $('liveView').value = saved.view;
    $('liveOrigin').value = ['shared', 'centre', 'floor'].includes(saved.origin) ? saved.origin : 'shared';
    if (saved.size && SIZES[saved.size]) $('liveSize').value = saved.size;
    $('phoneLiveSetup').addEventListener('change', saveSettings);
}

// ---- screens and buttons ---------------------------------------------------------------------------------------

function help(message) {
    $('phoneHelp').textContent = message;
}

function enter(mode) {
    state.mode = mode;
    document.body.classList.toggle('ar', mode === 'ar');
    $('phoneIntro').hidden = true;
    $('phoneHud').hidden = false;
    $('phonePads').hidden = mode !== 'sensors';
    $('phoneSpeedRow').hidden = mode !== 'sensors';
    requestWakeLock();
    openLive();
    refresh();
}

function leave() {
    if (state.recording) toggleRecording();

    closeLive();
    state.mode = 'idle';
    window.removeEventListener('deviceorientation', onOrientation);
    document.body.classList.remove('ar');
    if (state.xr && state.xr.session) state.xr.session.end().catch(() => {});
    if (state.xr) state.xr.renderer.setAnimationLoop(null);
    $('phoneIntro').hidden = false;
    $('phoneHud').hidden = true;
    $('phonePads').hidden = true;
    refresh();
}

function refresh() {
    const chip = $('phoneConnection');

    chip.textContent = state.connected ? 'connected to the simulator' : 'connecting...';
    chip.className = `chip ${state.connected ? 'ok' : 'bad'}`;
    $('phoneRecord').disabled = !state.connected || state.mode === 'idle';
    $('phoneRecord').textContent = state.recording ? 'Stop recording' : 'Record';
    $('phoneRecord').classList.toggle('rec-on', state.recording);
    $('phoneRecording').hidden = !state.recording;
}

function toggleRecording() {
    if (state.recording) {
        send('stop_recording');
        state.recording = false;
    } else if (send('start_recording')) {
        state.recording = true;
        state.recordStartedAt = performance.now();
    }

    refresh();
}

async function requestWakeLock() {
    try {
        if (navigator.wakeLock && !state.wakeLock) {
            state.wakeLock = await navigator.wakeLock.request('screen');
            state.wakeLock.addEventListener('release', () => { state.wakeLock = null; });
        }
    } catch {
        // not allowed (battery saver): the screen may dim, nothing else
    }
}

// A stick: drag inside the pad; `onChange({ x, y })` gets values in [-1, 1] (y up). Springs back on release.
function setupPad(element, onChange, { vertical = false } = {}) {
    const knob = element.querySelector('.knob');
    let active = null;

    function place(x, y, travel) {
        knob.style.transform = `translate(${x * travel}px, ${-y * travel}px)`;
    }

    function move(event) {
        const rect = element.getBoundingClientRect();
        const travel = (vertical ? rect.height : rect.width) / 2 - 28;
        const dx = vertical ? 0 : event.clientX - (rect.left + rect.width / 2);
        const dy = event.clientY - (rect.top + rect.height / 2);
        const length = Math.hypot(dx, dy);
        const k = length > travel ? travel / length : 1;
        const x = (dx * k) / travel;
        const y = (-dy * k) / travel;

        place(x, y, travel);
        onChange({ x, y });
    }

    element.addEventListener('pointerdown', event => {
        active = event.pointerId;
        element.setPointerCapture(active);
        move(event);
    });
    element.addEventListener('pointermove', event => { if (event.pointerId === active) move(event); });
    const release = event => {
        if (event.pointerId !== active) return;

        active = null;
        knob.style.transform = '';
        onChange({ x: 0, y: 0 });
    };
    element.addEventListener('pointerup', release);
    element.addEventListener('pointercancel', release);
}

export async function initPhoneApp() {
    const start = $('phoneStart');
    const addButton = (label, onClick, secondary = false) => {
        const button = document.createElement('button');

        button.type = 'button';
        button.textContent = label;
        button.className = secondary ? 'secondary' : '';
        button.addEventListener('click', onClick);
        start.appendChild(button);
    };

    if (await arSupported()) {
        help('Start AR for full 6DoF tracking (walk around with the phone). Without it, use sensors and the on-screen sticks.');
        addButton('Start AR (6DoF tracking)', startAr);
        addButton('Sensors + joysticks', startSensors, true);
    } else {
        help(window.isSecureContext
            ? 'No AR tracking in this browser (iPhone: Safari and Chrome have no WebXR): the phone gives the viewing direction, the sticks move you.'
            : 'This page is not HTTPS, so the motion sensors and AR are blocked: open it with https://.');
        addButton('Start with sensors + joysticks', startSensors);
    }

    $('phoneRecenter').addEventListener('click', () => { state.needsRecenter = true; });
    $('phoneRecord').addEventListener('click', toggleRecording);
    $('phoneSpeed').addEventListener('input', event => {
        state.speed = Number(event.target.value);
        $('phoneSpeedValue').textContent = state.speed.toFixed(1);
    });
    setupPad($('padMove'), value => { state.stick = value; });
    setupPad($('padHeight'), value => { state.vertical = value.y; }, { vertical: true });
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible' && state.mode !== 'idle') requestWakeLock();
    });

    connect();
    refresh();
    fillLiveSetup();
}
