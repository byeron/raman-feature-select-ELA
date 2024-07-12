import numpy as np
from scipy.signal import find_peaks


def detect_peaks(series, top_n, distance, delta, top, bottom, reverse=False):
    if top is None or bottom is None:
        raise ValueError("top and bottom must be specified")

    _iterator = np.arange(top, bottom, -delta)
    if reverse:
        _iterator = reversed(_iterator)

    for h in _iterator:
        peaks, _ = find_peaks(series, height=h, distance=distance)

        peaks = peaks.tolist()
        print(peaks)
        if len(peaks) >= top_n:
            return peaks

    return None
