import numpy as np
from numba import jit

luminance_weights = np.array([0.299, 0.587, 0.114, 0], dtype=np.float16)
eps = 1e-6

def get_log_luminance(img):
    luminance = np.dot(img, luminance_weights) / 255
    return np.log(luminance + eps)

@jit(nopython=True)
def process_events(log_luminance, threshold, c_plus, c_minus, timestamp, time_step):
    events = []
    polarity = np.sign(log_luminance - threshold)

    for y in range(polarity.shape[0]):
        for x in range(polarity.shape[1]):
            pol = polarity[y, x]

            if pol == 0:
                continue

            C = c_plus if pol > 0 else c_minus

            coefficient = np.abs(log_luminance[y, x] - threshold[y, x]) / C
            n_events = int(coefficient)

            for cur_event in range(n_events):
                events.append(
                    (int(timestamp + time_step * cur_event / coefficient), x, y, pol > 0)
                )

            threshold[y, x] += n_events * pol * C

    return np.array(events, dtype=np.uint32), threshold


def compute_frac_mat(length):
    a = np.arange(length) + 1
    b = a[None, :] / a[:, None]

    c = b[np.tril_indices(length)]
    d = np.argsort(np.argsort(c))

    e = np.zeros((length, length))
    e[np.tril_indices(length)] = d

    return e

frac_mat = compute_frac_mat(144).astype(np.int16)

def process_events_new(log_luminance, threshold, c_plus, c_minus, timestamp, time_step):
    polarity = np.sign(log_luminance - threshold)

    C = np.where(polarity > 0, c_plus, c_minus)

    coefficients = np.abs(log_luminance - threshold) // C
    coefficients[polarity == 0] = 0

    coefficient_max = np.max(coefficients).astype(np.uint16)

    output = np.empty((coefficient_max * coefficient_max), dtype=list)

    for cur_denominator in range(1, coefficient_max + 1):
        coord = np.argwhere(coefficients == cur_denominator)
        frac = np.arange(1, cur_denominator + 1) / cur_denominator

        tuples = np.concatenate(
            np.broadcast_arrays(
                frac[:, None, None],
                coord[:, None, 0],
                coord[:, None, 1],
                polarity[coord[:, 0], coord[:, 1]][None, :, None]
            ), axis=-1
        )

        output[frac_mat[cur_denominator - 1, :cur_denominator]] = list(tuples)

    mask = [np.any(out) for out in output]

    threshold += coefficients * polarity * C

    outputs = np.concatenate(output[np.array(mask)])
    outputs[:, 0] = timestamp + outputs[:, 0] * time_step

    return outputs.astype(np.uint32), threshold


class UniformSamplingDynamicVisionProcessing:
    def __init__(self, c_plus, c_minus):
        self.c_plus = c_plus
        self.c_minus = c_minus

        self.threshold = None

    def __call__(self, intensity_img, timestamp, time_step):
        log_luminance = get_log_luminance(intensity_img)

        if self.threshold is None:
            self.threshold = log_luminance
            return np.array([])

        events, self.threshold = process_events(
            log_luminance,
            self.threshold,
            self.c_plus,
            self.c_minus,
            timestamp,
            time_step
        )

        return events

