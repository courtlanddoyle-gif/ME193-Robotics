"""Runs the trained step-counting network on the board with numpy.

PyTorch isn't installed on the board, and it isn't needed just to *use* a
trained network: the forward pass is three matrix multiplies. This is the
same math as StepNet in training/common.py --
    input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> steps in last 2 s
with the weights PyTorch learned, loaded from the JSON that train.py saves.
"""
import json
from pathlib import Path

import numpy as np


class StepNet:
    def __init__(self, path):
        m = json.loads(Path(path).read_text())
        self.channels = m["channels"]
        self.window = m["window"]
        self.decimate = m["decimate"]
        self.hop = m["hop"]
        self.dist_far_cm = m["dist_far_cm"]
        # Flat input is time-major (t0 ax, t0 ay, ..., t1 ax, ...), so the
        # per-channel mean/std repeat once per kept sample.
        reps = self.window // self.decimate
        self.mean = np.tile(np.array(m["mean"], dtype=np.float32), reps)
        self.std = np.tile(np.array(m["std"], dtype=np.float32), reps)
        self.layers = [(np.array(l["W"], dtype=np.float32), np.array(l["b"], dtype=np.float32))
                       for l in m["layers"]]

    def inputs(self, window):
        """window: {channel: last `window` raw samples} -> flat, normalized input.
        Must match training/common.py: dist_feature(), window_inputs(), Normalizer."""
        cols = []
        for c in self.channels:
            v = np.asarray(window[c], dtype=np.float32)
            if c == "dist":
                # -1 (no answer) / -2 (nothing in range) / > 1 m all mean "no leg nearby"
                v = np.where(v >= 0, np.minimum(v, self.dist_far_cm), self.dist_far_cm) / self.dist_far_cm
            cols.append(v)
        x = np.stack(cols, axis=1)[::self.decimate].reshape(-1)
        return (x - self.mean) / self.std

    def steps_in_window(self, window):
        """How many steps the network thinks happened in this 2 s window."""
        h = self.inputs(window)
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b                 # each neuron: weighted sum of the layer below + bias
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)    # ReLU: negative -> 0
        return float(h[0])
