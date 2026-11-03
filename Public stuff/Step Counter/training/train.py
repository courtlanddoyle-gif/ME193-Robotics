"""Train the step-counting network with PyTorch and measure how good it is.

    python training/train.py

1. 5-fold cross-validation, split by recording: every recording is counted
   by a network that never saw it during training, so the error numbers are
   honest. Done for IMU only and IMU + ultrasonic, to see whether the
   distance sensor helps, and repeated with several random seeds, since one
   run can come out a bit better or worse just from its starting weights.
2. Trains final networks on all recordings and saves them as JSON (weights +
   normalization) in python/models/, for the board to run with numpy
   (python/step_net.py).

Cross-validation results go to training/results/.
"""
import csv
import json
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from common import (HOP, IMU, MODEL_DIR, RESULTS_DIR, WINDOW, DECIMATE, DIST_FAR_CM, Normalizer, StepNet,
                    count_from_windows, labeled_recordings, load_recording, make_windows)

N_FOLDS = 5
TRAIN_HOP = 2        # slide training windows 2 samples at a time = more examples
# Held-out error bottoms out around epoch 20 and creeps up after (the net
# starts memorizing the training recordings), so stop at 25.
EPOCHS = 25
SEEDS = (0, 1, 2)
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4  # gently pulls weights toward 0 so the net doesn't memorize
VARIANTS = {"imu": False, "imu_dist": True}


def assign_folds(recs):
    """Deal each activity's recordings round-robin into folds, so every fold
    tests a mix of walking, jogging, stairs, standing..."""
    by_act = defaultdict(list)
    for name, act in recs:
        by_act[act].append(name)
    folds, k = {}, 0
    for act in sorted(by_act):
        for name in by_act[act]:
            folds[name] = k % N_FOLDS
            k += 1
    return folds


def train_net(train_data, n_channels, seed=0, log_every=0, val_data=None):
    """train_data: list of (x, steps). Returns (net, normalizer)."""
    torch.manual_seed(seed)
    norm = Normalizer(n_channels).fit([x for x, _ in train_data])
    xs, ys = zip(*(make_windows(x, s, TRAIN_HOP) for x, s in train_data))
    X = torch.tensor(norm.apply_flat(np.concatenate(xs)), dtype=torch.float32)
    Y = torch.tensor(np.concatenate(ys), dtype=torch.float32)
    if val_data:
        vx, vy = zip(*(make_windows(x, s, HOP) for x, s in val_data))
        VX = torch.tensor(norm.apply_flat(np.concatenate(vx)), dtype=torch.float32)
        VY = torch.tensor(np.concatenate(vy), dtype=torch.float32)

    net = StepNet(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    for epoch in range(1, EPOCHS + 1):
        net.train()
        order = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), BATCH):
            idx = order[i:i + BATCH]
            pred = net(X[idx])               # forward pass
            loss = loss_fn(pred, Y[idx])     # how wrong, on average (squared)
            opt.zero_grad()
            loss.backward()                  # back-propagation: gradient of loss for every weight
            opt.step()                       # nudge every weight downhill
            total += loss.item() * len(idx)
        if log_every and epoch % log_every == 0:
            msg = f"    epoch {epoch:3d}  train MSE {total / len(X):.3f}"
            if val_data:
                net.eval()
                with torch.no_grad():
                    msg += f"  held-out MSE {loss_fn(net(VX), VY).item():.3f}"
            print(msg)
    net.eval()
    return net, norm


def predict_total(net, norm, x, steps):
    xs, _ = make_windows(x, steps, HOP)
    with torch.no_grad():
        pred = net(torch.tensor(norm.apply_flat(xs), dtype=torch.float32)).numpy()
    return count_from_windows(pred, HOP)


def export(net, norm, use_dist, path):
    """Weights + normalization as plain JSON, for numpy on the board."""
    linears = [m for m in net.net if isinstance(m, nn.Linear)]
    path.write_text(json.dumps({
        "window": WINDOW,
        "decimate": DECIMATE,
        "hop": HOP,
        "channels": list(IMU) + (["dist"] if use_dist else []),
        "dist_far_cm": DIST_FAR_CM,
        "mean": norm.mean.tolist(),
        "std": norm.std.tolist(),
        "layers": [{"W": m.weight.detach().numpy().tolist(),
                    "b": m.bias.detach().numpy().tolist()} for m in linears],
    }))


def metrics(rows, key):
    walk = [r for r in rows if r["counted"] > 0]
    err = np.array([r[key] - r["counted"] for r in walk])
    pct = 100 * err / np.array([r["counted"] for r in walk])
    total = sum(r["counted"] for r in walk)
    return {
        "total_pct": 100 * (sum(r[key] for r in walk) - total) / total,
        "mae": np.abs(err).mean(),
        "mape": np.abs(pct).mean(),
        "worst_pct": np.abs(pct).max(),
        "spread": err.std(),
        "false_steps": np.mean([r[key] for r in rows if r["counted"] == 0]),
    }


def summarize(rows, key):
    walk = [r for r in rows if r["counted"] > 0]
    still = [r for r in rows if r["counted"] == 0]
    err = np.array([r[key] - r["counted"] for r in walk])
    pct = 100 * err / np.array([r["counted"] for r in walk])
    total_counted = sum(r["counted"] for r in walk)
    print(f"  recordings with steps ({len(walk)}):")
    print(f"    total steps   counted {total_counted}, predicted {sum(r[key] for r in walk):.1f} "
          f"({100 * (sum(r[key] for r in walk) - total_counted) / total_counted:+.1f}%)")
    print(f"    per recording  mean abs error {np.abs(err).mean():.2f} steps, "
          f"mean abs % error {np.abs(pct).mean():.1f}%, worst {np.abs(pct).max():.0f}%")
    print(f"    bias           {err.mean():+.2f} steps (+ = overcounts), spread (std) {err.std():.2f}")
    print(f"  0-step recordings ({len(still)}): false steps "
          + ", ".join(f"{r['activity']} {r[key]:.1f}" for r in still))
    by_act = defaultdict(list)
    for r in walk:
        by_act[r["activity"]].append(100 * (r[key] - r["counted"]) / r["counted"])
    print("  by activity (mean % error):  "
          + "  ".join(f"{a} {np.mean(v):+.0f}%" for a, v in sorted(by_act.items())))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    recs = labeled_recordings()
    folds = assign_folds(recs)
    acts = dict(recs)
    print(f"{len(recs)} labeled recordings, {N_FOLDS}-fold cross-validation\n")

    rows = {name: {"recording": name, "activity": acts[name], "fold": folds[name]} for name, _ in recs}
    keys = []
    for variant, use_dist in VARIANTS.items():
        data = {name: load_recording(name, use_dist) for name, _ in recs}
        for name, (x, steps, lab) in data.items():
            rows[name]["counted"] = lab["steps_counted"]
        n_ch = len(IMU) + use_dist
        for seed in SEEDS:
            key = f"{variant}_s{seed}"
            keys.append(key)
            t0 = time.time()
            for k in range(N_FOLDS):
                train = [(x, s) for name, (x, s, _) in data.items() if folds[name] != k]
                test = {name: (x, s) for name, (x, s, _) in data.items() if folds[name] == k}
                curve = k == 0 and seed == SEEDS[0]
                if curve:
                    print(f"[{variant}] learning curve, fold 0:")
                net, norm = train_net(train, n_ch, seed=seed, log_every=5 if curve else 0,
                                      val_data=list(test.values()) if curve else None)
                for name, (x, s) in test.items():
                    rows[name][key] = predict_total(net, norm, x, s)
            print(f"[{variant}] seed {seed}: cross-validation done in {time.time() - t0:.0f} s")
        print()

    out = sorted(rows.values(), key=lambda r: r["recording"])
    with open(RESULTS_DIR / "cv_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["recording", "activity", "fold", "counted", *keys])
        w.writeheader()
        for r in out:
            w.writerow({**r, **{k: round(r[k], 2) for k in keys}})

    print("=== Across seeds (mean ± std), counted by networks that never saw the recording ===")
    names = {"total_pct": "total steps error %", "mae": "mean abs error (steps)",
             "mape": "mean abs % error", "worst_pct": "worst recording %",
             "spread": "error spread (steps)", "false_steps": "false steps per 0-step rec."}
    table = {v: [metrics(out, f"{v}_s{s}") for s in SEEDS] for v in VARIANTS}
    print(f"  {'':30}" + "".join(f"{v:>20}" for v in VARIANTS))
    for m, label in names.items():
        cells = []
        for v in VARIANTS:
            vals = np.array([t[m] for t in table[v]])
            cells.append(f"{vals.mean():+8.2f} ± {vals.std():.2f}" if m == "total_pct"
                         else f"{vals.mean():8.2f} ± {vals.std():.2f}")
        print(f"  {label:30}" + "".join(f"{c:>20}" for c in cells))
    print()
    for variant in VARIANTS:
        print(f"=== {variant}, seed {SEEDS[0]} in detail ===")
        summarize(out, f"{variant}_s{SEEDS[0]}")
        print()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for variant, use_dist in VARIANTS.items():
        data = [load_recording(name, use_dist)[:2] for name, _ in recs]
        net, norm = train_net(data, len(IMU) + use_dist)
        export(net, norm, use_dist, MODEL_DIR / f"step_model_{variant}.json")
        print(f"saved final {variant} model to python/models/ (trained on all {len(recs)} recordings)")


if __name__ == "__main__":
    main()
