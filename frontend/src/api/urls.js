// Where the simulator API lives. By default it is the origin that served the page
// (the Python backend serves the built frontend, and `vite dev` proxies /api, /spectate
// and /headset to it). Override at build/dev time with VITE_API_URL, e.g.
//   VITE_API_URL=https://my-server:8000 npm run dev

const FALLBACK_BASE = 'http://localhost:8000';

let override = null;

function trimSlashes(url) {
    return String(url).trim().replace(/\/+$/, '');
}

function envBase() {
    const env = typeof import.meta !== 'undefined' ? import.meta.env : undefined;
    return env && env.VITE_API_URL ? trimSlashes(env.VITE_API_URL) : '';
}

/** Force a base URL (used by tests); pass null to go back to the automatic default. */
export function setApiBase(url) {
    override = url ? trimSlashes(url) : null;
}

export function getApiBase() {
    if (override) return override;

    const fromEnv = envBase();
    if (fromEnv) return fromEnv;

    if (typeof window !== 'undefined' && window.location && window.location.origin !== 'null') {
        return window.location.origin;
    }

    return FALLBACK_BASE;
}

/** http(s)://host -> ws(s)://host */
export function getWsBase() {
    return getApiBase().replace(/^http/i, 'ws');
}
