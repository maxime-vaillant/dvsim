// The #jobPanel status box (see index.html): progress bar, status, errors, cancel button, output links.
// Server-provided strings only ever go through textContent.

import { jobFileUrl } from "../api/client.js";
import { isActivePhase } from "../api/jobRunner.js";

const PHASE_LABELS = {
    uploading: 'Uploading scene',
    submitting: 'Starting job',
    queued: 'Queued',
    running: 'Running',
    done: 'Done',
    failed: 'Failed',
    cancelled: 'Cancelled',
    error: 'Error'
};

const element = id => document.getElementById(id);

function formatSize(bytes) {
    if (!Number.isFinite(bytes)) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;

    return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function progressDetail(progress) {
    if (!progress) return '';

    const parts = [];

    if (progress.num_sequences > 1) {
        parts.push(`sequence ${Math.min(progress.sequence + 1, progress.num_sequences)}/${progress.num_sequences}`);
    }
    if (progress.total_steps > 0) parts.push(`step ${progress.step}/${progress.total_steps}`);
    if (Number.isFinite(progress.fraction)) parts.push(`${Math.round(progress.fraction * 100)}%`);

    return parts.join(' - ');
}

function renderProgressBar(state, progress) {
    const bar = element('jobProgress');

    bar.hidden = state.phase === 'error';

    if (state.phase === 'done') {
        bar.value = 1;
    } else if (progress && Number.isFinite(progress.fraction) && (state.phase === 'running' || state.phase === 'failed' || state.phase === 'cancelled')) {
        bar.value = Math.min(1, Math.max(0, progress.fraction));
    } else if (isActivePhase(state.phase)) {
        bar.removeAttribute('value'); // indeterminate
    }
}

function renderOutputs(state) {
    const list = element('jobFiles');
    const outputs = state.phase === 'done' && state.job && Array.isArray(state.job.outputs) ? state.job.outputs : [];

    list.replaceChildren();

    outputs.forEach(output => {
        const link = document.createElement('a');
        link.href = jobFileUrl(state.job.id, output.path);
        link.download = output.name || output.path;
        link.textContent = output.path;

        const size = document.createElement('span');
        size.textContent = formatSize(output.size_bytes);

        const item = document.createElement('li');
        item.append(link, size);
        list.append(item);
    });

    if (state.phase === 'done' && outputs.length === 0) {
        const item = document.createElement('li');
        item.textContent = 'No output files.';
        list.append(item);
    }
}

export function initJobPanel({ onCancel }) {
    element('jobCancel').addEventListener('click', onCancel);
    element('jobDismiss').addEventListener('click', () => { element('jobPanel').hidden = true; });
}

export function renderJobPanel(state) {
    const panel = element('jobPanel');
    const job = state.job;
    const progress = job ? job.progress : null;
    const active = isActivePhase(state.phase);

    panel.hidden = state.phase === 'idle';
    panel.dataset.phase = state.phase;

    element('jobStatus').textContent = PHASE_LABELS[state.phase] || state.phase;
    element('jobDetail').textContent = [state.message, active ? progressDetail(progress) : ''].filter(Boolean).join(' - ');

    const error = state.error || (job && job.error) || '';
    element('jobError').textContent = error;
    element('jobError').hidden = !error;

    renderProgressBar(state, progress);
    renderOutputs(state);

    // A submitted job can be cancelled; the upload can be aborted.
    element('jobCancel').hidden = !(state.phase === 'uploading' || state.phase === 'queued' || state.phase === 'running');
    element('jobDismiss').hidden = active;
}
