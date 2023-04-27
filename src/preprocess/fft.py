import numpy as np
import cqtPython


def standard(x):
    x = (x - np.mean(x)) / np.std(x)
    return x


def minmax(x, axis=None, min=None, max=None):
    x_min = min
    x_max = max

    if x_min is None:
        x_min = x.min(axis=axis, keepdims=True)

    if x_max is None:
        x_max = x.max(axis=axis, keepdims=True)

    return (x - x_min) / (x_max - x_min)


def cqt(
        y,
        sr=22050,
        n_bins=12 * 3 * 7,
        bins_per_octave=12 * 3,
        hop_length=256 + 128 + 64,
        fmin=32.7,
        window="hamm",
        Qfactor=25.0,
        norm=minmax):
    mono = True if len(y.shape) == 1 else False

    if mono:
        S = np.abs(
            cqtPython.cqt(
                y,
                sr=sr,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                hop_length=hop_length,
                Qfactor=Qfactor,
                fmin=fmin,
                window=window)).astype("float32").T

    else:
        S_l = np.abs(
            cqtPython.cqt(
                y[0],
                sr=sr,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                hop_length=hop_length,
                Qfactor=Qfactor,
                fmin=fmin,
                window=window)).astype("float32")

        S_r = np.abs(
            cqtPython.cqt(
                y[1],
                sr=sr,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                hop_length=hop_length,
                Qfactor=Qfactor,
                fmin=fmin,
                window=window)).astype("float32")

        S = np.array((S_l.T, S_r.T))

    if norm is not None:
        S = norm(S)

    return S
