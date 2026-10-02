// Wire format of /api/live (see backend/api/live.py): DOM free, unit-tested in node.

export const LIVE_VIEWS = ['rgb', 'events', 'depth', 'flow', 'grid'];
export const VIEW_LABELS = { rgb: 'RGB', events: 'Events', depth: 'Depth', flow: 'Flow', grid: 'All' };
export const SIZES = {
    '346x260': { width: 346, height: 260, label: '346 x 260 (simulator default)' },
    '640x480': { width: 640, height: 480, label: '640 x 480' },
    '128x96': { width: 128, height: 96, label: '128 x 96 (fastest)' }
};

/** The first message: which scene to render and how. */
export function startMessage({ sceneId, size = '346x260', origin = 'centre', view = 'rgb', fps = 30 }) {
    const { width, height } = SIZES[size] || SIZES['346x260'];

    return JSON.stringify({ type: 'start', scene_id: sceneId, width, height, origin, view, fps });
}

/** A camera pose relative to the origin; `seq` comes back with the frame rendered from it. */
export function poseMessage(seq, pose) {
    const { x, y, z, rx, ry, rz } = pose;

    return JSON.stringify({ type: 'pose', seq, x, y, z, rx, ry, rz });
}

export function viewMessage(view) {
    return JSON.stringify({ type: 'view', view });
}

/** A binary frame: 4 bytes big-endian pose sequence number, then the JPEG. */
export function parseFrame(buffer) {
    if (!buffer || buffer.byteLength <= 4) return null;

    const view = new DataView(buffer);

    return { seq: view.getUint32(0, false), jpeg: new Uint8Array(buffer, 4) };
}

/** Round-trip time pose -> frame: remember when each pose left, look it up when its frame arrives. */
export class LatencyTracker {
    constructor(capacity = 256) {
        this.capacity = capacity;
        this.sentAt = new Map();
        this.average = null;
    }

    sent(seq, now) {
        this.sentAt.set(seq, now);
        if (this.sentAt.size > this.capacity) this.sentAt.delete(this.sentAt.keys().next().value);
    }

    /** Milliseconds for the frame of pose `seq` (smoothed value in `average`), or null if unknown. */
    received(seq, now) {
        const start = this.sentAt.get(seq);

        if (start === undefined) return null;

        const ms = now - start;

        this.average = this.average === null ? ms : 0.8 * this.average + 0.2 * ms;

        return ms;
    }
}
