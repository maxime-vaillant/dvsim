// Transient message shown in #notice (bottom of the page). Click it to dismiss.

let timer = null;

export function notify(message, { level = 'info', ms = 6000 } = {}) {
    const element = document.getElementById('notice');

    if (!element) return;

    element.textContent = message;
    element.className = level;
    element.hidden = false;
    element.onclick = () => { element.hidden = true; };

    clearTimeout(timer);
    timer = setTimeout(() => { element.hidden = true; }, level === 'error' ? ms * 2 : ms);
}

// Hides a message that is no longer true (e.g. "draw a trajectory first" once one is drawn).
export function clearNotice() {
    const element = document.getElementById('notice');

    if (element) element.hidden = true;
}
