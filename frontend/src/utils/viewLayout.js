// Which preview is shown in each of the three configurable quadrants (the overview is always top-left).

export const VIEW_KINDS = ['Camera', 'Events', 'Depth', 'Optical flow'];

// [column, row] of each quadrant, row 0 at the bottom (WebGL viewport convention)
export const SLOTS = {
    topRight: [1, 1],
    bottomLeft: [0, 0],
    bottomRight: [1, 0]
};

export const SLOT_LABELS = { topRight: 'Top right', bottomLeft: 'Bottom left', bottomRight: 'Bottom right' };

export const DEFAULT_VIEWS = { topRight: 'Camera', bottomLeft: 'Depth', bottomRight: 'Events', flowScale: 2 };

/** Defaults overridden by the valid parts of a saved object (unknown kinds / slots / bad numbers are ignored). */
export function restoreViews(saved) {
    const views = { ...DEFAULT_VIEWS };

    if (!saved || typeof saved !== 'object') return views;

    Object.keys(SLOTS).forEach(slot => {
        if (VIEW_KINDS.includes(saved[slot])) views[slot] = saved[slot];
    });
    if (Number.isFinite(saved.flowScale) && saved.flowScale > 0) views.flowScale = saved.flowScale;

    return views;
}

/** Heading of a forward vector: yaw (deg, 0 = -Z, positive towards +X) and pitch (deg, positive up). */
export function headingDeg(forward) {
    const yaw = Math.atan2(forward.x, -forward.z) * 180 / Math.PI;
    const pitch = Math.atan2(forward.y, Math.hypot(forward.x, forward.z)) * 180 / Math.PI;

    return { yaw, pitch };
}

/** One-line camera readout: position (m), heading (deg) and, when known, the trajectory time (s). */
export function formatCameraInfo(position, forward, timeS = null) {
    const f = value => (Math.abs(value) < 0.005 ? 0 : value).toFixed(2).padStart(6);
    const { yaw, pitch } = headingDeg(forward);
    const parts = [
        `x ${f(position.x)}  y ${f(position.y)}  z ${f(position.z)} m`,
        `yaw ${yaw.toFixed(0).padStart(4)}°  pitch ${pitch.toFixed(0).padStart(3)}°`
    ];

    if (Number.isFinite(timeS)) parts.push(`t ${timeS.toFixed(2)} s`);

    return parts.join('\n');
}
