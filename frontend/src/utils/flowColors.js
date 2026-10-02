// Middlebury / Baker et al. 2007 optical-flow colours, identical to the simulator's videos
// (backend/vizualisation/visualizer.py): hue = direction, saturation = |flow| / scale, zero flow = white,
// beyond the scale the full hue is dimmed by 0.75.

/** The 55-hue colour wheel (RGB 0..255): red -> yellow -> green -> cyan -> blue -> magenta. */
export function makeColorWheel() {
    const segments = [
        [15, [255, null, 0], true], [6, [null, 255, 0], false], [4, [0, 255, null], true],
        [11, [0, null, 255], false], [13, [null, 0, 255], true], [6, [255, 0, null], false]
    ];
    const wheel = [];

    for (const [n, template, rising] of segments) {
        for (let i = 0; i < n; i++) {
            const ramp = Math.floor(255 * i / n);

            wheel.push(template.map(value => (value === null ? (rising ? ramp : 255 - ramp) : value)));
        }
    }

    return wheel;
}

export const COLOR_WHEEL = makeColorWheel();

/** RGB (0..255 integers) of one flow vector (dx right, dy down, pixels) on `scale` pixels. */
export function flowToRgb(dx, dy, scale) {
    const u = dx / scale;
    const v = dy / scale;
    const rad = Math.hypot(u, v);
    const ncols = COLOR_WHEEL.length;
    const fk = (Math.atan2(-v, -u) / Math.PI + 1) / 2 * (ncols - 1);
    const k0 = Math.floor(fk);
    const k1 = (k0 + 1) % ncols;
    const f = fk - k0;

    return [0, 1, 2].map(c => {
        const col = ((1 - f) * COLOR_WHEEL[k0][c] + f * COLOR_WHEEL[k1][c]) / 255;
        const out = rad <= 1 ? 1 - rad * (1 - col) : col * 0.75;

        return Math.min(255, Math.max(0, Math.round(out * 255)));
    });
}
