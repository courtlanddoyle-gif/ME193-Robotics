"""Train the finger-counting CNN in PyTorch.

    python train.py              # train, test on held-out photos, save the model

Network (3 conv layers, each followed by ReLU and 2x2 max pooling, then one
linear layer down to 6 answers = 0-5 fingers):

    64x64 RGB image
    conv 3->16, 3x3  -> ReLU -> maxpool  -> 16 x 32 x 32
    conv 16->32, 3x3 -> ReLU -> maxpool  -> 32 x 16 x 16
    conv 32->64, 3x3 -> ReLU -> maxpool  -> 64 x 8 x 8
    flatten (4096) -> dropout -> linear -> 6 scores -> softmax

Loss is cross-entropy, since this is picking a category, not predicting a
number.

Testing: photos come in bursts (4 per second), so neighbors are near
copies.  To get an honest score, each digit's photos are cut into 5-second
chunks and every 5th chunk (20%) is held out as a test set the network
never trains on.  After reporting that
score, the final model is trained on everything.

Saves models/finger_cnn.pt (PyTorch) and models/finger_cnn.npz (plain
numpy weights for the UNO Q), plus results/confusion.png and
results/training.png.
"""

import glob
import math
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
SIZE = 64
EPOCHS = 60
BATCH = 64
LR = 1e-3
CHUNK = 20        # 20 photos = 5 seconds of burst
TEST_EVERY = 5    # every 5th chunk is held out (20%)
torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------- data ----
def load_image(path):
    """128x128 BGR png -> 64x64 RGB float array in 0..1, channels first."""
    im = cv2.imread(str(path))
    im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im.transpose(2, 0, 1).astype(np.float32) / 255.0


def load_data():
    train, test = [], []
    for k in range(6):
        files = sorted(glob.glob(str(HERE / "data" / str(k) / "*.png")))
        for i, f in enumerate(files):
            chunk = i // CHUNK
            (test if chunk % TEST_EVERY == TEST_EVERY - 1 else train).append((f, k))
    to_arrays = lambda items: (
        torch.tensor(np.stack([load_image(f) for f, _ in items])),
        torch.tensor([k for _, k in items]))
    return to_arrays(train), to_arrays(test)


def augment(x):
    """Random rotate, shift, zoom, mirror and brightness for each image, so
    the network sees new versions of every photo every epoch."""
    n = x.shape[0]
    angle = (torch.rand(n) - 0.5) * 2 * math.radians(25)
    scale = 1 + (torch.rand(n) - 0.5) * 0.2
    tx, ty = [(torch.rand(n) - 0.5) * 0.25 for _ in range(2)]
    flip = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)  # mirror = other hand
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1),
                         torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    bright = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.4
    shift = (torch.rand(n, 1, 1, 1) - 0.5) * 0.15
    return (x * bright + shift).clamp(0, 1)


# --------------------------------------------------------------- model ----
class FingerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 8 * 8, 6)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 16 x 32 x 32
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 32 x 16 x 16
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)   # 64 x 8 x 8
        return self.fc(self.drop(x.flatten(1)))      # 6 scores


def train(x, y, x_test=None, y_test=None):
    model = FingerCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # Counts with fewer photos get a bigger weight in the loss, so the net
    # doesn't learn to lean toward whichever count we photographed most.
    per_class = torch.bincount(y, minlength=6).float()
    class_weight = per_class.sum() / (6 * per_class)
    history = []
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(x))
        total, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(x), BATCH):
            idx = order[i:i + BATCH]
            xb, yb = augment(x[idx]), y[idx]
            scores = model(xb)                       # forward pass
            loss = F.cross_entropy(scores, yb, weight=class_weight)  # how wrong
            opt.zero_grad()
            loss.backward()                          # back-propagation
            opt.step()                               # gradient descent step
            loss_sum += loss.item() * len(idx)
            correct += (scores.argmax(1) == yb).sum().item()
            total += len(idx)
        row = {"epoch": epoch + 1, "loss": loss_sum / total,
               "train_acc": correct / total}
        if x_test is not None:
            row["test_acc"] = (predict(model, x_test) == y_test).float().mean().item()
        history.append(row)
        print("  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                        for k, v in row.items()))
    return model, history


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).argmax(1)


# ---------------------------------------------------------------- main ----
def main():
    (x_tr, y_tr), (x_te, y_te) = load_data()
    print(f"train {len(x_tr)} photos, test {len(x_te)} photos (held out)")
    os.makedirs(HERE / "results", exist_ok=True)
    os.makedirs(HERE / "models", exist_ok=True)

    print("\n--- training on 80%, testing on the held-out 20% ---")
    model, hist = train(x_tr, y_tr, x_te, y_te)
    pred = predict(model, x_te)
    conf = np.zeros((6, 6), int)
    for t, p in zip(y_te.tolist(), pred.tolist()):
        conf[t, p] += 1
    acc = np.trace(conf) / conf.sum()
    print(f"\nheld-out accuracy: {acc:.1%}")
    print("confusion (rows = true count, columns = predicted):")
    print(conf)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(conf, cmap="Blues")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, conf[i, j], ha="center", va="center",
                    color="white" if conf[i, j] > conf.max() / 2 else "black")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xlabel("predicted fingers")
    ax.set_ylabel("true fingers")
    ax.set_title(f"Held-out photos: {acc:.1%} correct")
    fig.tight_layout()
    fig.savefig(HERE / "results" / "confusion.png", dpi=150)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    ep = [h["epoch"] for h in hist]
    a1.plot(ep, [h["loss"] for h in hist], color="#2a78d6", lw=2)
    a1.set_xlabel("epoch")
    a1.set_ylabel("cross-entropy loss")
    a1.set_title("Training loss")
    a2.plot(ep, [h["train_acc"] for h in hist], color="#2a78d6", lw=2,
            label="training photos")
    a2.plot(ep, [h["test_acc"] for h in hist], color="#eb6834", lw=2,
            label="held-out photos")
    a2.set_xlabel("epoch")
    a2.set_ylabel("accuracy")
    a2.set_title("Accuracy")
    a2.legend()
    fig.tight_layout()
    fig.savefig(HERE / "results" / "training.png", dpi=150)

    print("\n--- final model: training on all photos ---")
    final, _ = train(torch.cat([x_tr, x_te]), torch.cat([y_tr, y_te]))
    torch.save(final.state_dict(), HERE / "models" / "finger_cnn.pt")
    np.savez(HERE / "models" / "finger_cnn.npz",
             **{k: v.numpy() for k, v in final.state_dict().items()})
    print("saved models/finger_cnn.pt and models/finger_cnn.npz")


if __name__ == "__main__":
    main()
