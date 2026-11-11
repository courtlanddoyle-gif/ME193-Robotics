"""The finger-counting CNN in plain numpy (no PyTorch), so it can run on the
UNO Q.  Same layers as train.py's FingerCNN, using the weights it exported
to models/finger_cnn.npz.

    net = FingerNet("models/finger_cnn.npz")
    probs = net.predict(net.prepare(frame, roi))   # 6 probabilities, 0-5 fingers
"""

import cv2
import numpy as np


def conv3x3(x, w, b):
    """x: (C, H, W), w: (F, C, 3, 3) -> (F, H, W), 'same' padding.
    Each output pixel = sum over a 3x3 patch of every channel x the kernel."""
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    patches = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))
    # patches: (C, H, W, 3, 3) -> multiply with each kernel and add up
    return np.einsum("chwij,fcij->fhw", patches, w, optimize=True) + b[:, None, None]


def relu(x):
    return np.maximum(0, x)


def maxpool2(x):
    """Keep the biggest value in each 2x2 block, halving the height and width."""
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


class FingerNet:
    SIZE = 64

    def __init__(self, path):
        p = np.load(path)
        self.layers = [(p[f"conv{i}.weight"], p[f"conv{i}.bias"]) for i in (1, 2, 3)]
        self.fc_w, self.fc_b = p["fc.weight"], p["fc.bias"]

    @staticmethod
    def prepare(frame, roi):
        """Camera frame (BGR) -> the network's input, exactly like training:
        crop the red-square box, resize to 128 (as collect.py saved it), then
        to 64x64, RGB, 0..1, channels first."""
        x, y, w, h = roi
        H, W = frame.shape[:2]
        im = frame[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
        im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im.transpose(2, 0, 1).astype(np.float32) / 255.0

    def features(self, x):
        """Run the 3 conv -> ReLU -> pool layers. Returns every stage."""
        stages = [x]
        for w, b in self.layers:
            x = maxpool2(relu(conv3x3(x, w, b)))
            stages.append(x)
        return stages

    def predict(self, x):
        feat = self.features(x)[-1]                  # (64, 8, 8)
        scores = self.fc_w @ feat.reshape(-1) + self.fc_b
        return softmax(scores)
