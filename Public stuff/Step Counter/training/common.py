"""Shared pieces for training the step-counting network: loading recordings
and their step labels, cutting them into 2 s windows, and the network.

The board (python/main.py) must turn its live buffer into the network's
input exactly the way window_inputs() does here, so keep the two in sync.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

FS = 50              # samples per second from the board
WINDOW = 100         # 2 s of samples per input, same as the board's buffer
HOP = 10             # the board re-runs the network every 10 samples (0.2 s)
DECIMATE = 2         # keep every 2nd sample: 25 Hz is plenty for walking
IMU = ("ax", "ay", "az", "gx", "gy", "gz")
DIST_FAR_CM = 100.0  # anything farther (or no echo) just means "no leg nearby"

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "recordings"
LABEL_DIR = ROOT / "data" / "labels"
RESULTS_DIR = ROOT / "training" / "results"
MODEL_DIR = ROOT / "python" / "models"   # the board app loads the trained nets from here


def dist_feature(dist_cm):
    """HC-SR04 reading -> 0..1, where 1 = far / nothing in range / no echo.
    Clipping at 1 m makes a far wall at home and an empty gym look the same;
    only the other leg passing close by stands out."""
    d = np.asarray(dist_cm, dtype=np.float32)
    return np.where(d >= 0, np.minimum(d, DIST_FAR_CM), DIST_FAR_CM) / DIST_FAR_CM


def load_recording(name, use_dist=True):
    """(samples x channels) array, step sample indices, and the label info."""
    with open(REC_DIR / f"{name}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = [np.array([float(r[k]) for r in rows], dtype=np.float32) for k in IMU]
    if use_dist:
        cols.append(dist_feature([float(r["dist"]) for r in rows]))
    label = json.loads((LABEL_DIR / f"{name}.json").read_text())
    return np.stack(cols, axis=1), np.array(label["steps"], dtype=int), label


def labeled_recordings():
    """Names of recordings whose labels matched the spoken count."""
    out = []
    for p in sorted(LABEL_DIR.glob("*.json")):
        lab = json.loads(p.read_text())
        if lab["status"] == "ok":
            out.append((p.stem, lab["activity"]))
    return out


def window_inputs(x):
    """One 2 s window (WINDOW x channels) -> the network's flat input."""
    return x[::DECIMATE].reshape(-1)


def make_windows(x, steps, hop):
    """Every 2 s window, sliding by `hop` samples, and how many steps are in it.

    The recording is padded with its first/last sample (you stand still at
    both ends), so windows also cover the start and end. Then every step is
    inside exactly WINDOW / hop windows, which is what lets the counts be
    summed back up to a total in count_from_windows().
    """
    n = len(x)
    padded = np.concatenate([np.repeat(x[:1], WINDOW, 0), x, np.repeat(x[-1:], WINDOW, 0)])
    starts = np.arange(0, n + WINDOW, hop)
    xs = np.stack([window_inputs(padded[s:s + WINDOW]) for s in starts])
    # Window starting at padded index s covers original samples [s - WINDOW, s).
    ys = np.array([np.sum((steps >= s - WINDOW) & (steps < s)) for s in starts], dtype=np.float32)
    return xs, ys


def count_from_windows(window_counts, hop):
    """Total steps from per-window step counts: each step is in WINDOW / hop windows."""
    return float(np.sum(np.maximum(window_counts, 0)) * hop / WINDOW)


class StepNet(nn.Module):
    """350 inputs (25 samples/s x 2 s x 7 channels) -> 32 -> 16 -> 1:
    how many steps happened in the last 2 s."""

    def __init__(self, n_in, hidden=(32, 16)):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Normalizer:
    """Per-channel mean/std from the training data, so accel (~10 m/s^2),
    gyro (~few rad/s) and distance (0..1) all reach the network at a similar
    scale."""

    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.mean = np.zeros(n_channels, dtype=np.float32)
        self.std = np.ones(n_channels, dtype=np.float32)

    def fit(self, recordings):
        allx = np.concatenate(recordings)
        self.mean = allx.mean(0).astype(np.float32)
        self.std = (allx.std(0) + 1e-6).astype(np.float32)
        return self

    def apply_flat(self, xs):
        """Normalize flat window inputs (time-major: t0c0, t0c1, ..., t1c0, ...)."""
        reps = xs.shape[1] // self.n_channels
        return (xs - np.tile(self.mean, reps)) / np.tile(self.std, reps)
