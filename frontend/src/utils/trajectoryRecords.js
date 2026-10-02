// Legacy trajectory records: { timestamp (ms), x, y, z, rx, ry, rz } where (rx, ry, rz) is the unit
// forward vector of the camera. This is the format of the JSON files the previewer imports.

const REQUIRED_KEYS = ['timestamp', 'x', 'y', 'z', 'rx', 'ry', 'rz'];
const QUATERNION_KEYS = ['qx', 'qy', 'qz', 'qw'];

export const round = (value, decimals) => {
    const factor = 10 ** decimals;

    return Math.round(value * factor) / factor;
};

/** Keeps only the fields of the legacy format (drops preview-only fields like normalizedTimestamp). */
export function pickRecord(record) {
    const picked = {};

    REQUIRED_KEYS.forEach(key => { picked[key] = record[key]; });
    if (QUATERNION_KEYS.every(key => Number.isFinite(record[key]))) {
        QUATERNION_KEYS.forEach(key => { picked[key] = record[key]; });
    }

    return picked;
}

/**
 * Validates parsed JSON and returns the records sorted by timestamp.
 * Throws an Error with a message meant to be shown to the user.
 */
export function parseTrajectoryRecords(data) {
    const list = Array.isArray(data) ? data : (data && Array.isArray(data.points) ? data.points : null);

    if (!list) throw new Error('Trajectory JSON must be an array of records.');
    if (list.length < 2) throw new Error('A trajectory needs at least 2 records.');

    list.forEach((record, index) => {
        REQUIRED_KEYS.forEach(key => {
            if (!record || !Number.isFinite(record[key])) {
                throw new Error(`Record ${index}: "${key}" is missing or not a number.`);
            }
        });
    });

    const records = list.map(pickRecord).sort((a, b) => a.timestamp - b.timestamp);

    if (records[records.length - 1].timestamp <= records[0].timestamp) {
        throw new Error('Trajectory timestamps must span a positive duration.');
    }

    return records;
}

/**
 * Sorted records with `normalizedTimestamp` in [0, 1] (what the previewer's playback interpolates on) and their
 * duration in seconds: { records, durationS }.
 */
export function withPlaybackTimes(list) {
    const records = parseTrajectoryRecords(list);
    const start = records[0].timestamp;
    const durationMs = records[records.length - 1].timestamp - start;

    records.forEach(record => {
        record.normalizedTimestamp = (record.timestamp - start) / durationMs;
    });

    return { records, durationS: durationMs / 1000 };
}

/**
 * Records in the legacy import format. Positions are made relative to `origin` (the previewer adds the
 * object centre back when it loads such a file), so exporting and re-importing gives the same path.
 */
export function toLegacyRecords(records, origin = null) {
    const [ox, oy, oz] = origin || [0, 0, 0];

    return records.map(record => ({
        timestamp: record.timestamp,
        x: round(record.x - ox, 6),
        y: round(record.y - oy, 6),
        z: round(record.z - oz, 6),
        rx: record.rx,
        ry: record.ry,
        rz: record.rz
    }));
}
