"""Render docs/how_it_works.mp4: how the step counter works, from the sensors
through the neural net and PyTorch training to back-propagation.

    pip install torch numpy matplotlib imageio-ffmpeg
    python docs/make_animation.py                 # full video (~2-5 min)
    python docs/make_animation.py --preview 30 47 # just PNG frames at those seconds

Everything shown is computed here from the repo, not typed in:
  * the sensor traces and step labels of one real recording,
  * the deployed network (python/models/step_model_imu_dist.json) run on it,
    neuron by neuron, the same way the board does,
  * a fresh StepNet trained with the exact loop from training/train.py, with
    the real gradients from its first loss.backward() and the real loss curve.
The accuracy numbers in the last scene are the cross-validation results in
README.md / training/results/.
"""
import argparse
import json
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.collections import LineCollection     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "training"))
from common import (FS, HOP, MODEL_DIR, REC_DIR, WINDOW, Normalizer, StepNet,  # noqa: E402
                    labeled_recordings, load_recording, make_windows)
from train import BATCH, EPOCHS, LR, TRAIN_HOP, WEIGHT_DECAY  # noqa: E402

EXAMPLE = "20261103-024327_walk-fast"
EX_END = 300            # example window = samples [200, 300) of the recording
FPS = 30
W_PX, H_PX = 1280, 720
OUT = HERE / "how_it_works.mp4"
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "dist"]
IN_TIMES = (0, 25, 49)  # input time steps drawn in the network diagram
IN_IDX = np.array([t * 7 + c for t in IN_TIMES for c in range(7)])

# ---- style -----------------------------------------------------------------
BG = "#0e1320"
PANEL = "#182033"
FG = "#eef2f8"
MUTED = "#8d98ad"
BLUE = "#4ea8ff"        # forward pass / predictions
ORANGE = "#ff9a3c"      # backward pass / gradients
GREEN = "#56d39b"       # truth (labels)
YELLOW = "#f5d04c"      # hardware
PURPLE = "#a68bff"      # negative sums before ReLU
DARK = "#283146"        # a neuron at 0
MONO = "DejaVu Sans Mono"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#3a4560"})
FWD_CMAP = LinearSegmentedColormap.from_list("fwd", [DARK, BLUE, "#dff0ff"])
BWD_CMAP = LinearSegmentedColormap.from_list("bwd", [DARK, ORANGE, "#fff0dc"])
IN_CMAP = LinearSegmentedColormap.from_list("inp", ["#1d2a44", "#5b7fb8", "#e6eefb"])
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#2f6fd0", DARK, "#d8e6ff"])


# ---------------------------------------------------------------------------
# Data: everything the video shows, computed from the repo
# ---------------------------------------------------------------------------
def linears(net):
    return [m for m in net.net if isinstance(m, nn.Linear)]


def forward_acts(net, xin):
    """Every layer's values for one normalized input, like the board's numpy loop."""
    l1, l2, l3 = linears(net)
    with torch.no_grad():
        v = torch.tensor(np.asarray(xin)[None], dtype=torch.float32)
        z1 = l1(v); h1 = torch.relu(z1)
        z2 = l2(h1); h2 = torch.relu(z2)
        out = l3(h2)
    return {"inp": np.asarray(xin, dtype=np.float32), "z1": z1[0].numpy(), "h1": h1[0].numpy(),
            "z2": z2[0].numpy(), "h2": h2[0].numpy(), "out": float(out[0, 0])}


def prepare():
    D = {}
    x, steps, lab = load_recording(EXAMPLE, use_dist=True)
    meta = json.loads((REC_DIR / f"{EXAMPLE}.json").read_text())
    raw = np.genfromtxt(REC_DIR / f"{EXAMPLE}.csv", delimiter=",", names=True)
    n = len(x)
    D.update(n=n, t=np.arange(n) / FS, gz=raw["gz"], dist_cm=np.where(raw["dist"] >= 0, raw["dist"], np.nan),
             x=x, steps=steps, right=np.array(lab["right"]), left=np.array(lab["left"]),
             counted=lab["steps_counted"], distance_m=meta["distance_m"])
    win_raw = x[EX_END - WINDOW:EX_END]
    D["win_raw"] = win_raw
    D["win_true"] = int(np.sum((steps >= EX_END - WINDOW) & (steps < EX_END)))

    # ---- the deployed network, loaded from the JSON the board uses ----------
    m = json.loads((MODEL_DIR / "step_model_imu_dist.json").read_text())
    dep = StepNet(len(m["mean"]) * WINDOW // m["decimate"])
    with torch.no_grad():
        for l, p in zip(linears(dep), m["layers"]):
            l.weight.copy_(torch.tensor(p["W"])); l.bias.copy_(torch.tensor(p["b"]))
    dnorm = Normalizer(7)
    dnorm.mean = np.array(m["mean"], np.float32); dnorm.std = np.array(m["std"], np.float32)
    D["dep_mean"], D["dep_std"] = dnorm.mean, dnorm.std
    D["dep_W"] = [l.weight.detach().numpy().copy() for l in linears(dep)]
    D["json_W00"] = m["layers"][0]["W"][0][:3]
    D["json_mean"] = m["mean"][:3]
    win_flat = win_raw[::2].reshape(-1)
    D["win_norm"] = dnorm.apply_flat(win_flat[None])[0]
    D["acts"] = forward_acts(dep, D["win_norm"])

    # ---- replay the walk through the board's logic (python/main.py) --------
    xs, _ = make_windows(x, steps, HOP)
    padded = np.concatenate([np.repeat(x[:1], WINDOW, 0), x, np.repeat(x[-1:], WINDOW, 0)])
    starts = np.arange(0, n + WINDOW, HOP)       # window ends at original sample s
    dm = json.loads((MODEL_DIR / "distance_model.json").read_text())
    coef = dict(zip(dm["terms"], dm["coef"]))
    D["coef"] = coef
    outs, totals, dists, lengths, swings, cads, live_acts = [], [], [], [], [], [], []
    tot = dist = 0.0
    for k, s in enumerate(starts):
        w = padded[s:s + WINDOW]
        gmag = np.linalg.norm(w[:, 3:6], axis=1)
        amag = np.linalg.norm(w[:, 0:3], axis=1)
        still = gmag.max() < 0.15 and amag.std() < 0.1
        a = forward_acts(dep, dnorm.apply_flat(xs[k:k + 1])[0])
        out = 0.0 if still else max(0.0, a["out"])
        cad = out * 60.0 / 2.0
        length = coef["1"] + coef["cadence_spm"] * cad + coef["swing_rad_s"] * gmag.mean()
        length = min(max(length, dm["min_m"]), dm["max_m"])
        tot += out * HOP / WINDOW
        dist += out * HOP / WINDOW * length
        outs.append(out); totals.append(tot); dists.append(dist); lengths.append(length)
        swings.append(gmag.mean()); cads.append(cad); live_acts.append(a)
    D.update(rep_end_t=starts / FS, rep_out=np.array(outs), rep_total=np.array(totals),
             rep_dist=np.array(dists), rep_len=np.array(lengths), rep_swing=np.array(swings),
             rep_cad=np.array(cads), rep_acts=live_acts)

    # ---- train a fresh StepNet with train.py's exact loop ------------------
    recs = [name for name, _ in labeled_recordings() if name != EXAMPLE]
    data = [load_recording(name, True)[:2] for name in recs]
    torch.manual_seed(0)
    norm = Normalizer(7).fit([xx for xx, _ in data])
    txs, tys = zip(*(make_windows(xx, s, TRAIN_HOP) for xx, s in data))
    X = torch.tensor(norm.apply_flat(np.concatenate(txs)), dtype=torch.float32)
    Y = torch.tensor(np.concatenate(tys), dtype=torch.float32)
    exX = torch.tensor(norm.apply_flat(xs), dtype=torch.float32)
    _, exY = make_windows(x, steps, HOP)
    exY = torch.tensor(exY)
    ex_win = torch.tensor(norm.apply_flat(win_flat[None]), dtype=torch.float32)
    net = StepNet(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    D.update(n_train_recs=len(recs), n_train_windows=len(X))

    def evaluate():
        net.eval()
        with torch.no_grad():
            p = net(exX)
            r = (loss_fn(p, exY).item(), float(torch.clamp(p, min=0).sum()) * HOP / WINDOW,
                 float(net(ex_win)[0]))
        net.train()
        return r

    hist = {"epoch": [0], "ex_mse": [], "ex_total": [], "ex_win": [], "train": [np.nan],
            "batch_x": [], "batch_loss": []}
    e = evaluate()
    hist["ex_mse"].append(e[0]); hist["ex_total"].append(e[1]); hist["ex_win"].append(e[2])
    n_batches = (len(X) + BATCH - 1) // BATCH
    for epoch in range(1, EPOCHS + 1):
        net.train()
        order = torch.randperm(len(X))
        total = 0.0
        for bi, i in enumerate(range(0, len(X), BATCH)):
            idx = order[i:i + BATCH]
            if epoch == 1 and bi == 0:
                # The very first step, taken apart for scene 5. Same math as
                # net(X[idx]); the intermediate values are kept so their
                # gradients can be shown too.
                l1, l2, l3 = linears(net)
                xb = X[idx]
                h1 = torch.relu(l1(xb)); h1.retain_grad()
                h2 = torch.relu(l2(h1)); h2.retain_grad()
                pred = l3(h2).squeeze(-1); pred.retain_grad()
                loss = loss_fn(pred, Y[idx])
                opt.zero_grad()
                loss.backward()
                D["first"] = {
                    "pred": pred.detach().numpy().copy(), "truth": Y[idx].numpy().copy(),
                    "loss": loss.item(),
                    "gW": [l.weight.grad.numpy().copy() for l in (l1, l2, l3)],
                    "gh1": h1.grad.abs().mean(0).numpy(), "gh2": h2.grad.abs().mean(0).numpy(),
                    "gout": float(pred.grad.abs().mean()),
                    "W_before": [l.weight.detach().numpy().copy() for l in (l1, l2, l3)],
                }
                opt.step()
                D["first"]["W_after"] = [l.weight.detach().numpy().copy() for l in (l1, l2, l3)]
                with torch.no_grad():
                    D["first"]["loss_after"] = loss_fn(net(xb), Y[idx]).item()
            else:
                pred = net(X[idx])               # forward pass
                loss = loss_fn(pred, Y[idx])     # how wrong, on average (squared)
                opt.zero_grad()
                loss.backward()                  # back-propagation
                opt.step()                       # nudge every weight downhill
            total += loss.item() * len(idx)
            hist["batch_x"].append(epoch - 1 + (bi + 1) / n_batches)
            hist["batch_loss"].append(loss.item())
        e = evaluate()
        hist["epoch"].append(epoch); hist["train"].append(total / len(X))
        hist["ex_mse"].append(e[0]); hist["ex_total"].append(e[1]); hist["ex_win"].append(e[2])
    D["hist"] = {k: np.array(v) for k, v in hist.items()}
    D["n_batches"] = n_batches
    return D


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def clip01(v):
    return float(min(max(v, 0.0), 1.0))


def prog(t, a, b):
    return clip01((t - a) / (b - a))


def ease(p):
    p = clip01(p)
    return p * p * (3 - 2 * p)


def fade(t, a, b, f=0.4):
    """1 inside [a, b], fading in/out over f seconds."""
    return clip01(min((t - a) / f + 1 if f else 1, (b - t) / f + 1 if f else 1))


def mix(c1, c2, f):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * clip01(f))


def text(ax, x, y, s, size=18, color=FG, alpha=1.0, **kw):
    if alpha <= 0.01:
        return None
    return ax.text(x, y, s, fontsize=size, color=color, alpha=alpha, transform=ax.transAxes, **kw)


def panel(ax, x, y, w, h, color=PANEL, alpha=1.0, ec="#33405c"):
    if alpha <= 0.01:
        return
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                transform=ax.transAxes, fc=color, ec=ec, lw=1.2, alpha=alpha))


def row(ax, y, parts, size, x=0.5, ha="center", gap=0.012, **kw):
    """Draw [(text, color, weight)] side by side, measured so nothing overlaps."""
    r = ax.figure.canvas.get_renderer()
    objs, widths = [], []
    for s_, c, wgt in parts:
        o = ax.text(0, y, s_, fontsize=size, color=c, weight=wgt, transform=ax.transAxes, va="center", **kw)
        objs.append(o)
        widths.append(o.get_window_extent(r).width / W_PX)
    total = sum(widths) + gap * (len(parts) - 1)
    x0 = x - total / 2 if ha == "center" else (x - total if ha == "right" else x)
    for o, w in zip(objs, widths):
        o.set_x(x0)
        x0 += w + gap
    return objs


def header(ax, num, title):
    text(ax, 0.03, 0.895, f"{num}", size=20, color=MUTED, va="center", weight="bold")
    text(ax, 0.062, 0.895, title, size=27, va="center", weight="bold")
    names = ["Sensors", "2 s window", "Forward pass", "Training", "Back-prop", "On the board",
             "Distance", "Accuracy"]
    active = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}.get(num, -1)
    parts = []
    for i, nm in enumerate(names):
        on = i == active
        parts.append((nm, (ORANGE if i == 4 else FG) if on else "#56617a", "bold" if on else "normal"))
        if i < len(names) - 1:
            parts.append(("›", "#3d4660", "normal"))
    row(ax, 0.965, parts, 11.5, x=0.97, ha="right", gap=0.008)


def caption(ax, t, items):
    """items: list of (start, end, text); drawn in the bottom band."""
    for a, b, s in items:
        al = fade(t, a, b, 0.35)
        if al > 0.01:
            text(ax, 0.5, 0.065, s, size=20, alpha=al, ha="center", va="center", linespacing=1.35)


def style(a, xlabel=None, ylabel=None):
    a.set_facecolor(PANEL)
    for s in a.spines.values():
        s.set_color("#3a4560")
    a.tick_params(labelsize=12, colors=MUTED)
    if xlabel:
        a.set_xlabel(xlabel, fontsize=13, color=MUTED)
    if ylabel:
        a.set_ylabel(ylabel, fontsize=13, color=MUTED)


def code_panel(ax, x, y, active=None, alpha=1.0, size=15):
    lines = ["pred = net(X[idx])", "loss = loss_fn(pred, Y[idx])", "opt.zero_grad()",
             "loss.backward()", "opt.step()"]
    notes = ["forward pass", "how wrong (MSE)", "clear old gradients", "back-propagation",
             "nudge every weight"]
    h = 0.05
    panel(ax, x, y - h * 5 - 0.075, 0.335, h * 5 + 0.105, alpha=alpha, color="#121a2b")
    text(ax, x + 0.012, y + 0.0, "PyTorch, training/train.py", size=12, color=MUTED, alpha=alpha)
    for i, (ln, nt) in enumerate(zip(lines, notes)):
        yy = y - 0.045 - i * h
        on = active is not None and i == active
        col = ORANGE if i >= 3 else BLUE
        if on:
            ax.add_patch(Rectangle((x + 0.004, yy - h / 2 + 0.004), 0.327, h - 0.008, transform=ax.transAxes,
                                   fc=col, alpha=0.2 * alpha, ec="none"))
            text(ax, x + 0.014, y - 0.045 - 5 * h + 0.002, "▲ " + nt, size=12.5, color=col, va="center",
                 alpha=alpha)
        text(ax, x + 0.014, yy, ln, size=size, color=FG if on else "#8a96ad", family=MONO,
             va="center", alpha=alpha)


# ---- the network diagram ----------------------------------------------------
def net_geom(x0, x1, y0, y1):
    xs = np.linspace(x0, x1, 4)
    slots = np.array(list(range(7)) + list(range(8, 15)) + list(range(16, 23)))
    yin = y1 - slots / 22 * (y1 - y0)
    yh1 = np.linspace(y1, y0, 32)
    m = 0.22 * (y1 - y0)
    yh2 = np.linspace(y1 - m, y0 + m, 16)
    yout = np.array([(y0 + y1) / 2])
    return xs, [yin, yh1, yh2, yout]


def edges(ax, xa, ya, xb, yb, alpha, color, lw=1.0):
    """alpha: (len(yb), len(ya)) opacity of each edge."""
    jj, ii = np.nonzero(alpha > 0.015)
    if len(jj) == 0:
        return
    segs = np.stack([np.stack([np.full(len(ii), xa), ya[ii]], 1), np.stack([np.full(len(jj), xb), yb[jj]], 1)], 1)
    rgb = to_rgb(color)
    cols = np.column_stack([np.tile(rgb, (len(jj), 1)), np.clip(alpha[jj, ii], 0, 1)])
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, transform=ax.transAxes, zorder=2))


def draw_net(ax, geom, node_colors, edge_alpha, edge_color, labels=True, in_labels=True, s_hidden=80):
    xs, ys = geom
    # faint structure
    for L in range(3):
        a = np.full((len(ys[L + 1]), len(ys[L])), 0.05)
        edges(ax, xs[L], ys[L], xs[L + 1], ys[L + 1], a, "#8d98ad", lw=0.6)
    for L in range(3):
        if edge_alpha[L] is not None:
            edges(ax, xs[L], ys[L], xs[L + 1], ys[L + 1], edge_alpha[L], edge_color, lw=1.0)
    sizes = [95, s_hidden, s_hidden + 25, 520]
    for L in range(4):
        ax.scatter(np.full(len(ys[L]), xs[L]), ys[L], s=sizes[L], c=node_colors[L], edgecolors="#6b7690",
                   linewidths=0.8, transform=ax.transAxes, zorder=3)
    yin = ys[0]
    for gap in (6.5, 14.5):   # the "..." between input groups
        yg = np.interp(gap, np.arange(21) + np.array([0] * 7 + [1] * 7 + [2] * 7), yin)
        text(ax, xs[0], yg - 0.004, "⋮", size=15, color=MUTED, ha="center", va="center")
    if in_labels:
        for g, tstep in enumerate(IN_TIMES):
            text(ax, xs[0] - 0.045, yin[g * 7 + 3], f"t{tstep}\n{tstep * 0.04:.2f} s", size=10.5,
                 color=MUTED, ha="right", va="center")
            for c in range(7):
                text(ax, xs[0] - 0.012, yin[g * 7 + c], CHANNELS[c], size=8.5, color="#6f7a91",
                     ha="right", va="center")
    if labels:
        top = max(ys[0].max(), ys[1].max()) + 0.035
        for L, s in enumerate(["350 inputs\n(21 drawn)", "32 neurons\n+ ReLU", "16 neurons\n+ ReLU", "1 output"]):
            text(ax, xs[L], top, s, size=12.5, color=MUTED, ha="center", va="bottom", linespacing=1.15)


def fwd_edge_alpha(W, a_in, idx_in=None):
    """How much signal travels along each edge: |weight x input|, scaled per layer."""
    Wd = W if idx_in is None else W[:, idx_in]
    a = a_in if idx_in is None else a_in[idx_in]
    c = np.abs(Wd * a[None, :])
    return np.clip(c / (np.percentile(c, 99.5) + 1e-9), 0, 1) ** 0.8 * 0.9


def grad_edge_alpha(g, idx_in=None):
    gd = np.abs(g if idx_in is None else g[:, idx_in])
    return np.clip(gd / (np.percentile(gd, 99.5) + 1e-12), 0, 1) ** 0.7 * 0.95


def fwd_colors(acts, show_z1=0, relu1=0, show_z2=0, relu2=0, show_out=0, vmax=None):
    """Node colors for the forward pass. z shown first; after ReLU negatives go dark."""
    inp = acts["inp"][IN_IDX]
    c_in = [IN_CMAP(clip01((v + 2.5) / 5)) for v in inp]
    v1 = vmax or max(acts["h1"].max(), 1e-6)

    def hid(z, show, relu, vm):
        out = []
        for v in z:
            if show <= 0:
                out.append(to_rgb(DARK)); continue
            if v > 0:
                c = mix(DARK, FWD_CMAP(clip01(v / vm))[:3], show)
            else:
                c = mix(DARK, mix(PURPLE, DARK, relu), show)
            out.append(c)
        return out
    c1 = hid(acts["z1"], show_z1, relu1, v1)
    c2 = hid(acts["z2"], show_z2, relu2, max(acts["h2"].max(), 1e-6))
    c3 = [mix(DARK, BLUE, show_out)]
    return [c_in, c1, c2, c3]


# ---------------------------------------------------------------------------
# Scenes. Each draws one frame at local time t (seconds).
# ---------------------------------------------------------------------------
def scene_title(ax, t, D):
    a = ease(prog(t, 0.0, 0.7))
    text(ax, 0.5, 0.64, "How the step counter works", size=46, weight="bold", ha="center", alpha=a)
    parts = [("sensors", YELLOW), ("neural net", BLUE), ("PyTorch training", BLUE), ("back-propagation", ORANGE)]
    items = []
    for i, (p, c) in enumerate(parts):
        items.append((p, c, "bold"))
        if i < 3:
            items.append(("→", MUTED, "normal"))
    objs = row(ax, 0.50, items, 25, gap=0.016)
    for k, o in enumerate(objs):
        o.set_alpha(ease(prog(t, 0.6 + 0.45 * ((k + 1) // 2), 1.0 + 0.45 * ((k + 1) // 2))))
    text(ax, 0.5, 0.33, "A wearable that counts steps with a small neural net trained in PyTorch",
         size=19, color=MUTED, ha="center", alpha=ease(prog(t, 2.2, 2.8)))
    text(ax, 0.5, 0.26, "ME-193, Tufts  ·  real data and the real trained network throughout",
         size=15, color="#66718a", ha="center", alpha=ease(prog(t, 2.4, 3.0)))


def draw_leg_sketch(ax, t):
    # Front view: the wearer's right leg is on the viewer's left.
    lw = 20
    col = "#3b4868"
    hip_y, knee_y, ank_y = 0.70, 0.49, 0.27
    R, L = 0.075, 0.150
    ax.plot([R + 0.005, R], [hip_y, knee_y], color=col, lw=lw, solid_capstyle="round", transform=ax.transAxes)
    ax.plot([R, R + 0.008], [knee_y, ank_y], color=col, lw=lw - 4, solid_capstyle="round", transform=ax.transAxes)
    ax.plot([L - 0.005, L], [hip_y, knee_y], color=col, lw=lw, solid_capstyle="round", transform=ax.transAxes)
    ax.plot([L, L - 0.008], [knee_y, ank_y], color=col, lw=lw - 4, solid_capstyle="round", transform=ax.transAxes)
    ax.plot([R - 0.03, R + 0.03], [ank_y - 0.035] * 2, color=col, lw=14, solid_capstyle="round", transform=ax.transAxes)
    ax.plot([L - 0.03, L + 0.03], [ank_y - 0.035] * 2, color=col, lw=14, solid_capstyle="round", transform=ax.transAxes)
    ax.plot([R - 0.01, L + 0.01], [hip_y + 0.02] * 2, color=col, lw=lw, solid_capstyle="round", transform=ax.transAxes)
    text(ax, R, 0.175, "right\nleg", size=12, color=MUTED, ha="center", linespacing=1.1)
    text(ax, L, 0.175, "left\nleg", size=12, color=MUTED, ha="center", linespacing=1.1)
    # IMU on the inside of the right ankle
    ax.add_patch(Rectangle((R + 0.016, ank_y + 0.015), 0.024, 0.045, transform=ax.transAxes, fc=YELLOW, ec="none", zorder=4))
    # ultrasonic on the inside of the right calf, aimed at the left leg
    uy = 0.41
    ax.add_patch(Rectangle((R + 0.016, uy - 0.025), 0.02, 0.05, transform=ax.transAxes, fc=YELLOW, ec="none", zorder=4))
    for k in range(3):
        ph = ((t * 1.2 + k / 3) % 1.0)
        r = 0.012 + 0.05 * ph
        th = np.linspace(-0.6, 0.6, 20)
        ax.plot(R + 0.036 + r * np.cos(th), uy + r * 1.7 * np.sin(th), color=YELLOW, lw=1.6,
                alpha=0.8 * (1 - ph), transform=ax.transAxes)
    text(ax, 0.215, uy + 0.06, "HC-SR04 ultrasonic\non the calf: distance\nto the left leg, 20 Hz", size=12.5, color=YELLOW,
         va="center", linespacing=1.2)
    text(ax, 0.215, ank_y + 0.02, "MPU6050 IMU\non the ankle: accel\n+ gyro, 3 axes, 50 Hz", size=12.5, color=YELLOW,
         va="center", linespacing=1.2)
    ax.add_patch(FancyArrowPatch((0.21, ank_y + 0.02), (R + 0.042, ank_y + 0.037), transform=ax.transAxes,
                                 arrowstyle="-", color=YELLOW, lw=1, alpha=0.5))
    text(ax, 0.03, 0.80, "An Arduino UNO Q on the leg reads\nboth: 7 numbers, 50 times a second",
         size=12.5, color=MUTED, linespacing=1.2)


def plot_traces(fig, D, rect_gz, rect_d, upto=None, box=None, ticks=True, dist=True):
    t = D["t"]
    k = len(t) if upto is None else int(np.clip(upto * FS, 2, len(t)))
    a1 = fig.add_axes(rect_gz)
    style(a1, ylabel="gz (rad/s)")
    a1.plot(t[:k], D["gz"][:k], color=BLUE, lw=2)
    a1.set_xlim(0, t[-1]); a1.set_ylim(-9, 6)
    a1.axhline(0, color="#3a4560", lw=0.8)
    if ticks:
        for s in D["right"]:
            if s < k:
                a1.plot(t[s], D["gz"][s], marker="v", color=GREEN, ms=11, zorder=5)
        for s in D["left"]:
            if s < k:
                a1.axvline(t[s], color=GREEN, lw=1.2, ls=(0, (3, 3)), alpha=0.6)
    axes = [a1]
    if dist:
        a2 = fig.add_axes(rect_d)
        style(a2, xlabel="time (s)", ylabel="distance (cm)")
        a2.plot(t[:k], D["dist_cm"][:k], color=YELLOW, lw=1.8)
        a2.set_xlim(0, t[-1]); a2.set_ylim(0, 310)
        a2.set_yticks([0, 100, 200, 300])
        a1.set_xticklabels([])
        axes.append(a2)
    else:
        a1.set_xlabel("time (s)", fontsize=13, color=MUTED)
    if box is not None:
        for a in axes:
            yl = a.get_ylim()
            a.add_patch(Rectangle((box - 2.0, yl[0]), 2.0, yl[1] - yl[0], fc=BLUE, alpha=0.16, ec=BLUE, lw=2, zorder=1))
    return axes


def scene_sensors(ax, t, D):
    header(ax, 1, "The sensors")
    draw_leg_sketch(ax, t)
    upto = D["t"][-1] * ease(prog(t, 0.8, 9.5))
    plot_traces(ax.figure, D, [0.46, 0.53, 0.51, 0.27], [0.46, 0.21, 0.51, 0.27], upto=upto)
    k = int(np.sum(D["steps"] < upto * FS))
    text(ax, 0.97, 0.885, f"labeled steps: {k} / {D['counted']}", size=15, ha="right", color=GREEN, weight="bold")
    text(ax, 0.46, 0.825, "▼ right step (bottom of gz dip)   ┆ left step (halfway)", size=12.5, color=GREEN)
    caption(ax, t, [(0, 6.0, "Each right step, the shin swings forward and the gyro (gz) dips.\n"
                             "The labels (green) mark every step, matched to the count said out loud."),
                    (6.0, 99, "The ultrasonic reads short distances when the left leg passes close by.\n"
                              "This walk: 19 steps over 14.33 m, in 11.8 s.")])


def scene_window(ax, t, D):
    header(ax, 2, "A 2-second window")
    fig = ax.figure
    end_t = EX_END / FS
    box = end_t - 0.2 * (15 - min(15, int(t / 0.2))) if t < 3.2 else end_t
    axs = plot_traces(fig, D, [0.07, 0.62, 0.90, 0.21], None, box=box, dist=False)
    axs[0].set_xlabel("")
    text(ax, 0.075, 0.845, f"window = the last 2 s = 100 samples        a new window every 0.2 s",
         size=13, color=BLUE)
    w = D["win_raw"]
    if t < 4.0:
        al = fade(t, -1, 4.0, 0.4)
        text(ax, 0.5, 0.38, "The board keeps the last 2 s of all 7 channels.\nEvery 0.2 s, it hands that window to the net.",
             size=22, ha="center", va="center", alpha=al, linespacing=1.4)
    elif t < 7.2:
        al = fade(t, 4.0, 7.2, 0.3)
        thin = prog(t, 5.0, 6.0)
        for c in range(7):
            a = fig.add_axes([0.13, 0.475 - c * 0.046, 0.80, 0.042])
            a.set_facecolor(PANEL); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            v = w[:, c]
            a.plot(np.arange(100), v, color="#5b6a8a" if thin > 0 else BLUE, lw=1.3, alpha=al)
            if thin > 0:
                kk = np.arange(0, 100, 2)[: int(50 * thin) + 1]
                a.plot(kk, v[kk], "o", color=BLUE, ms=3.2, alpha=al)
            a.set_xlim(-1, 100)
            pad = (v.max() - v.min()) * 0.15 + 1e-3
            a.set_ylim(v.min() - pad, v.max() + pad)
            text(ax, 0.12, 0.475 - c * 0.046 + 0.021, CHANNELS[c], size=12, color=MUTED, ha="right",
                 va="center", alpha=al)
        text(ax, 0.5, 0.545, "cut out the window  →  keep every 2nd sample (25 Hz)  →  50 time steps",
             size=16, ha="center", alpha=al)
    else:
        al = ease(prog(t, 7.2, 7.6))
        grid = D["win_norm"].reshape(50, 7).T
        a = fig.add_axes([0.13, 0.25, 0.80, 0.25])
        a.imshow(grid, aspect="auto", cmap=DIV_CMAP, vmin=-2.5, vmax=2.5, alpha=al)
        a.set_yticks(range(7)); a.set_yticklabels(CHANNELS, fontsize=11, color=MUTED)
        a.set_xticks([0, 10, 20, 30, 40, 49]); a.tick_params(labelsize=11, colors=MUTED)
        a.set_xlabel("time step (t0 = oldest, t49 = now)", fontsize=12, color=MUTED)
        for s in a.spines.values():
            s.set_visible(False)
        text(ax, 0.5, 0.545, "normalize each channel:  (value − training mean) / training std",
             size=16, ha="center", alpha=al)
        text(ax, 0.5, 0.135, "7 channels × 50 time steps = 350 numbers  →  the net's input",
             size=19, ha="center", weight="bold", color=BLUE, alpha=ease(prog(t, 8.2, 8.7)))
    caption(ax, t, [])


def scene_forward(ax, t, D):
    header(ax, 3, "Forward pass: the trained net")
    geom = net_geom(0.12, 0.60, 0.20, 0.76)
    W1, W2, W3 = D["dep_W"]
    live = t >= 7.5
    if not live:
        A = D["acts"]
        p_e1, p_z1, p_r1 = prog(t, 0.8, 2.0), prog(t, 1.8, 2.4), prog(t, 2.8, 3.5)
        p_e2, p_z2, p_r2 = prog(t, 3.6, 4.4), prog(t, 4.3, 4.8), prog(t, 4.9, 5.4)
        p_e3, p_o = prog(t, 5.5, 6.1), prog(t, 6.0, 6.4)
    else:
        k = min(len(D["rep_acts"]) - 1, int((t - 7.5) / 7.0 * len(D["rep_acts"])))
        A = D["rep_acts"][k]
        p_e1 = p_z1 = p_r1 = p_e2 = p_z2 = p_r2 = p_e3 = p_o = 1.0
    cols = fwd_colors(A, p_z1, p_r1, p_z2, p_r2, p_o, vmax=max(D["acts"]["h1"].max(), 1e-6))
    ea = [fwd_edge_alpha(W1, A["inp"], IN_IDX) * p_e1,
          fwd_edge_alpha(W2, A["h1"]) * p_e2 * (p_r1 > 0.5 or live),
          fwd_edge_alpha(W3, A["h2"]) * p_e3]
    draw_net(ax, geom, cols, ea, BLUE)
    xs = geom[0]
    n0 = int(np.sum(A["z1"] <= 0)); n2 = int(np.sum(A["z2"] <= 0))
    if p_r1 > 0.3:
        text(ax, xs[1], 0.155, f"{n0} of 32 → 0", size=12, color=PURPLE, ha="center", alpha=p_r1)
    if p_r2 > 0.3:
        text(ax, xs[2], 0.155, f"{n2} of 16 → 0", size=12, color=PURPLE, ha="center", alpha=p_r2)
    if p_o > 0:
        text(ax, xs[3], 0.43, f"{max(A['out'], 0):.1f}", size=26, weight="bold", color=FG,
             ha="center", va="center", alpha=p_o, zorder=6)
        text(ax, xs[3], 0.38, "steps in\nlast 2 s", size=12, color=MUTED, ha="center", va="top", alpha=p_o)

    # right panel
    fig = ax.figure
    if not live:
        al = ease(prog(t, 6.2, 6.8))
        panel(ax, 0.66, 0.30, 0.31, 0.46)
        text(ax, 0.815, 0.70, f"window {(EX_END - WINDOW) / FS:.1f}–{EX_END / FS:.1f} s of the walk", size=14,
             color=MUTED, ha="center")
        text(ax, 0.815, 0.58, f"net says {A['out']:.2f} steps", size=24, color=BLUE, ha="center",
             weight="bold", alpha=al)
        text(ax, 0.815, 0.48, f"labels say {D['win_true']}", size=24, color=GREEN, ha="center",
             weight="bold", alpha=al)
        text(ax, 0.815, 0.37, "(each neuron's color =\nits real value on this window)", size=12.5,
             color=MUTED, ha="center", alpha=al, linespacing=1.2)
        caption(ax, t, [(0, 2.7, "Each neuron multiplies its inputs by learned weights, adds them up, plus a bias.\n"
                                  "Brighter lines carry more signal (|weight × input|)."),
                        (2.7, 5.4, "ReLU: a neuron whose sum is negative (purple) outputs 0 (dark).\n"
                                   "Only the positive ones pass their signal on."),
                        (5.4, 99, f"The single output is the number of steps in that 2 s window: here {A['out']:.2f}.")])
    else:
        tt = D["rep_end_t"]
        a = fig.add_axes([0.68, 0.50, 0.29, 0.26])
        style(a, ylabel="steps in window")
        a.bar(tt[:k + 1], D["rep_out"][:k + 1], width=0.16, color=BLUE)
        a.set_xlim(0, tt[-1] + 0.2); a.set_ylim(0, 5.5)
        a.set_xlabel("window end (s)", fontsize=12, color=MUTED)
        panel(ax, 0.68, 0.17, 0.29, 0.22)
        text(ax, 0.825, 0.335, "running total += output × 0.1", size=14, color=MUTED, ha="center", family=MONO)
        text(ax, 0.825, 0.245, f"{D['rep_total'][k]:.1f} steps", size=34, weight="bold", ha="center", va="center")
        caption(ax, t, [(7.5, 99, "Every 0.2 s the window slides and the net runs again.\n"
                                  "Each step is inside 10 windows, so adding output × 0.1 each time counts it once.")])


def scene_training(ax, t, D):
    F, H = D["first"], D["hist"]
    fig = ax.figure
    if t < 6.0:
        header(ax, 4, "Training with PyTorch")
        code_panel(ax, 0.03, 0.82, active=0 if t < 3.0 else 1)
        panel(ax, 0.03, 0.17, 0.335, 0.27)
        text(ax, 0.045, 0.40, "Start: fresh StepNet, random weights", size=13, weight="bold")
        text(ax, 0.045, 0.335, f"Training data: {D['n_train_windows']:,} windows\nfrom {D['n_train_recs']} recordings, "
             f"each with its\ntrue step count from the labels", size=13, color=MUTED, va="top", linespacing=1.3)
        a = fig.add_axes([0.43, 0.34, 0.54, 0.46])
        style(a, ylabel="steps in window")
        nb = 12
        xx = np.arange(nb)
        p, y = F["pred"][:nb], F["truth"][:nb]
        al1 = ease(prog(t, 0.3, 1.2))
        a.bar(xx - 0.2, y, 0.38, color=GREEN, alpha=al1, label="truth (labels)")
        a.bar(xx + 0.2, p, 0.38, color=BLUE, alpha=ease(prog(t, 0.8, 1.7)), label="net's guess")
        a.set_ylim(min(-0.5, p.min() - 0.3), max(5.5, y.max()) + 1.6)
        a.axhline(0, color="#3a4560", lw=0.8)
        a.set_xticks(xx); a.set_xticklabels([f"#{i + 1}" for i in xx], fontsize=11)
        a.set_xlabel(f"12 of the {BATCH} windows in the first batch", fontsize=12, color=MUTED)
        leg = a.legend(loc="upper center", ncol=2, fontsize=12, facecolor=PANEL, edgecolor="#3a4560", labelcolor=FG)
        if t > 3.0:
            al = ease(prog(t, 3.0, 3.6))
            for i in range(nb):
                a.plot([i + 0.2, i + 0.2], [p[i], y[i]], color=ORANGE, lw=2.5, alpha=al)
            text(ax, 0.705, 0.22, f"loss = mean of (guess − truth)² over {BATCH} windows = {F['loss']:.2f}",
                 size=14.5, ha="center", color=FG, alpha=al, family=MONO)
        caption(ax, t, [(0, 3.0, "Forward pass on a batch of 256 windows.\nThe untrained net's guesses are all near 0."),
                        (3.0, 99, "The loss measures how wrong it is: the mean squared error.\n"
                                  "Orange = the error on each window. Training = making the loss smaller.")])
    elif t < 14.0:
        header(ax, 5, "loss.backward(): back-propagation")
        tb = t - 6.0
        code_panel(ax, 0.03, 0.82, active=2 if tb < 0.8 else 3)
        geom = net_geom(0.52, 0.93, 0.19, 0.76)
        p3, ph2 = prog(tb, 1.0, 1.8), prog(tb, 1.7, 2.2)
        p2, ph1 = prog(tb, 2.3, 3.3), prog(tb, 3.2, 3.7)
        p1 = prog(tb, 3.8, 5.0)
        gh1 = F["gh1"] / F["gh1"].max(); gh2 = F["gh2"] / F["gh2"].max()
        cols = [[IN_CMAP(0.25)] * 21,
                [mix(DARK, BWD_CMAP(clip01(v ** 0.7))[:3], ph1) for v in gh1],
                [mix(DARK, BWD_CMAP(clip01(v ** 0.7))[:3], ph2) for v in gh2],
                [mix(DARK, ORANGE, prog(tb, 0.6, 1.0))]]
        ea = [grad_edge_alpha(F["gW"][0], IN_IDX) * p1, grad_edge_alpha(F["gW"][1]) * p2,
              grad_edge_alpha(F["gW"][2]) * p3]
        draw_net(ax, geom, cols, ea, ORANGE, in_labels=False, s_hidden=60)
        xs = geom[0]
        text(ax, xs[3], 0.42, f"loss\n{F['loss']:.2f}", size=13, color=FG, ha="center", va="top",
             alpha=prog(tb, 0.6, 1.0), linespacing=1.1)
        # backward arrow
        al = prog(tb, 0.8, 1.2)
        ax.add_patch(FancyArrowPatch((0.93, 0.135), (0.52, 0.135), transform=ax.transAxes, arrowstyle="-|>",
                                     mutation_scale=22, color=ORANGE, lw=2.5, alpha=al))
        text(ax, 0.725, 0.15, "error signal flows backwards", size=13, color=ORANGE, ha="center", alpha=al)
        gmax = [np.abs(g).max() for g in F["gW"]]
        panel(ax, 0.03, 0.17, 0.335, 0.30)
        text(ax, 0.045, 0.43, "Real gradients from this first batch", size=14, weight="bold")
        text(ax, 0.045, 0.385, "largest |∂loss/∂weight| in each layer:", size=12.5, color=MUTED)
        for i, (lab, g, pp) in enumerate(zip(["16 → 1", "32 → 16", "350 → 32"], gmax[::-1], (p3, p2, p1))):
            text(ax, 0.06, 0.34 - i * 0.04, f"{lab:>9}   {g:.4f}", size=14, family=MONO,
                 color=ORANGE if pp > 0 else "#5b667d")
        text(ax, 0.045, 0.185, f"{sum(g.size for g in F['gW']):,} weight gradients in all,\nplus one per bias",
             size=12, color=MUTED, linespacing=1.2)
        caption(ax, t, [(6.0, 8.6, "opt.zero_grad() clears old gradients. loss.backward() starts at the loss\n"
                                   "and works backwards through the net, one layer at a time."),
                        (8.6, 11.2, "The chain rule: a weight's effect on the loss = its effect on the next layer\n"
                                    "× that layer's effect on the loss, already worked out one step later."),
                        (11.2, 99, "Every weight gets a gradient: which way, and how strongly, it pushes the loss.\n"
                                   "Brighter orange = bigger gradient (scaled per layer).")])
    elif t < 19.5:
        header(ax, 5, "opt.step(): nudge every weight downhill")
        tb = t - 14.0
        code_panel(ax, 0.03, 0.82, active=4)
        Wb, Wa, G = F["W_before"][1], F["W_after"][1], F["gW"][1]
        dW = Wa - Wb
        specs = [(Wb, "weights before", DIV_CMAP, np.abs(Wb).max(), BLUE),
                 (np.abs(G), "|gradient|", BWD_CMAP, np.abs(G).max(), ORANGE),
                 (dW, "change = after − before", DIV_CMAP, np.abs(dW).max(), FG)]
        for i, (M, lab, cm, vm, c) in enumerate(specs):
            al = ease(prog(tb, 0.3 + 0.9 * i, 0.9 + 0.9 * i))
            a = fig.add_axes([0.40 + i * 0.195, 0.47, 0.175, 0.30])
            a.set_facecolor(BG)
            a.imshow(M, cmap=cm, vmin=0 if cm is BWD_CMAP else -vm, vmax=vm, aspect="auto", alpha=al)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_color("#3a4560")
            text(ax, 0.40 + i * 0.195 + 0.0875, 0.79, lab, size=13, color=c, ha="center", alpha=al)
            if i == 0:
                text(ax, 0.40, 0.445, "layer 32 → 16 (16 × 32 weights)", size=11, color=MUTED, alpha=al, va="top")
            if i == 2:
                text(ax, 0.985, 0.445, f"every change is about ±{np.median(np.abs(dW)):.4f}", size=11,
                     color=MUTED, alpha=al, va="top", ha="right")
        # a few real weights
        al = ease(prog(tb, 2.8, 3.4))
        idx = [(0, 0), (3, 7), (9, 20)]
        text(ax, 0.40, 0.375, "weight         before      gradient      after", size=13.5, family=MONO,
             color=MUTED, alpha=al)
        for r, (j, i) in enumerate(idx):
            text(ax, 0.40, 0.33 - r * 0.042,
                 f"W[{j:>2},{i:>2}]   {Wb[j, i]:+.5f}   {G[j, i]:+.6f}   {Wa[j, i]:+.5f}",
                 size=13.5, family=MONO, alpha=al)
        panel(ax, 0.03, 0.17, 0.335, 0.27, alpha=1)
        text(ax, 0.045, 0.40, "Same batch, after one step", size=14, weight="bold")
        text(ax, 0.045, 0.33, f"loss {F['loss']:.3f} → {F['loss_after']:.3f}", size=20, family=MONO,
             alpha=ease(prog(tb, 3.4, 4.0)))
        text(ax, 0.045, 0.27, f"Adam: lr = {LR:g}, so each weight\nmoves about {LR:g} this first step", size=12.5,
             color=MUTED, va="top", linespacing=1.25)
        caption(ax, t, [(14.0, 99, "Each weight moves a tiny step opposite its gradient (Adam optimizer).\n"
                                   "Opposite = downhill: the direction that makes the loss smaller.")])
    else:
        header(ax, 4, "Repeat: thousands of steps")
        tb = t - 19.5
        steps_total = D["n_batches"] * EPOCHS
        cyc = int(tb * 2.5) % 5
        code_panel(ax, 0.03, 0.82, active=cyc)
        ep = EPOCHS * ease(prog(tb, 0.5, 7.0))
        a = fig.add_axes([0.44, 0.50, 0.53, 0.30])
        style(a, ylabel="loss (MSE)")
        bx, bl = H["batch_x"], H["batch_loss"]
        m = bx <= ep
        a.plot(bx[m], bl[m], color=BLUE, lw=0.6, alpha=0.35)
        e = H["epoch"]; me = e <= ep
        a.plot(e[me][1:], H["train"][me][1:], color=BLUE, lw=2.5, label="training windows (epoch mean)")
        a.plot(e[me], H["ex_mse"][me], color=GREEN, lw=2.5, label="example walk (never trained on)")
        a.set_yscale("log"); a.set_xlim(0, EPOCHS); a.set_ylim(0.05, max(bl.max(), H["ex_mse"].max()) * 1.3)
        a.set_xlabel("epoch (one pass over all training windows)", fontsize=12, color=MUTED)
        a.legend(loc="upper right", fontsize=11.5, facecolor=PANEL, edgecolor="#3a4560", labelcolor=FG)
        ei = int(np.clip(np.floor(ep), 0, EPOCHS))
        frac = ep - ei
        nxt = min(ei + 1, EPOCHS)
        ex_tot = H["ex_total"][ei]     # value after the last finished epoch
        ex_w = H["ex_win"][ei]
        nstep = int(min(steps_total, round(ep * D["n_batches"])))
        panel(ax, 0.03, 0.17, 0.335, 0.27)
        text(ax, 0.045, 0.40, f"epoch {ep:4.1f} / {EPOCHS}", size=17, weight="bold", family=MONO)
        text(ax, 0.045, 0.35, f"{nstep:,} of {steps_total:,} weight updates", size=13, color=MUTED)
        text(ax, 0.045, 0.285, f"example window: {ex_w:4.2f}  (truth {D['win_true']})", size=14, family=MONO, color=BLUE)
        text(ax, 0.045, 0.215, f"whole example walk: {ex_tot:4.1f}  (truth {D['counted']})", size=14,
             family=MONO, color=GREEN)
        text(ax, 0.44, 0.36, f"{EPOCHS} epochs × {D['n_batches']} batches = {steps_total:,} trips through the 5 lines",
             size=15, color=MUTED)
        text(ax, 0.44, 0.30, f"after {ei:2d} epochs   loss: training " + (f"{H['train'][ei]:.3f}" if ei else "  –  ")
             + f", example walk {H['ex_mse'][ei]:.3f}", size=13, color=FG, family=MONO)
        caption(ax, t, [(19.5, 99, "The same five lines run again and again. The loss drops, and the net\n"
                                   "learns to count a walk it never saw during training.")])


def scene_deployed(ax, t, D):
    header(ax, 6, "On the board")
    al = ease(prog(t, 0.2, 0.8))
    panel(ax, 0.03, 0.42, 0.40, 0.40, color="#121a2b", alpha=al)
    text(ax, 0.045, 0.785, "python/models/step_model_imu_dist.json", size=12.5, color=MUTED, family=MONO, alpha=al)
    mean = ", ".join(f"{v:.3f}" for v in D["json_mean"])
    w0 = ", ".join(f"{v:.4f}" for v in D["json_W00"])
    js = ('{\n  "mean": [' + mean + ', …],\n  "std":  [ … ],\n  "layers": [\n'
          '    {"W": [[' + w0 + ', …], …],\n     "b": [ … ]},   # 350 → 32\n'
          '    { … },                # 32 → 16\n    { … }  ]              # 16 → 1\n}')
    text(ax, 0.045, 0.745, js, size=11.5, family=MONO, va="top", alpha=al, linespacing=1.35)
    al2 = ease(prog(t, 1.2, 1.8))
    ax.add_patch(FancyArrowPatch((0.23, 0.41), (0.23, 0.36), transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=20, color=MUTED, lw=2, alpha=al2))
    panel(ax, 0.03, 0.15, 0.40, 0.20, color="#121a2b", alpha=al2, ec=YELLOW)
    text(ax, 0.045, 0.32, "Arduino UNO Q · python/step_net.py (numpy, no PyTorch)", size=12, color=YELLOW, alpha=al2)
    text(ax, 0.045, 0.265, "for W, b in layers:\n    h = W @ h + b\n    h = max(h, 0)   # ReLU, not on the last",
         size=12.5, family=MONO, va="top", alpha=al2, linespacing=1.3)
    # live replay
    p = prog(t, 2.5, 9.0)
    tt = D["rep_end_t"]
    k = int(min(len(tt) - 1, p * (len(tt) - 1)))
    upto = min(tt[k], D["t"][-1])
    fig = ax.figure
    a = fig.add_axes([0.48, 0.56, 0.49, 0.24])
    style(a, ylabel="gz (rad/s)")
    kk = int(upto * FS)
    a.plot(D["t"][:max(kk, 2)], D["gz"][:max(kk, 2)], color=BLUE, lw=1.8)
    a.set_xlim(0, D["t"][-1] + 2); a.set_ylim(-9, 6); a.set_xlabel("time (s)", fontsize=12, color=MUTED)
    a.axvspan(tt[k] - 2, tt[k], color=BLUE, alpha=0.15)
    panel(ax, 0.48, 0.17, 0.235, 0.28)
    panel(ax, 0.735, 0.17, 0.235, 0.28)
    text(ax, 0.5975, 0.40, "steps", size=15, color=MUTED, ha="center")
    text(ax, 0.5975, 0.29, f"{D['rep_total'][k]:.1f}", size=44, weight="bold", ha="center", va="center")
    text(ax, 0.5975, 0.20, f"counted out loud: {D['counted']}", size=12.5, color=GREEN, ha="center")
    text(ax, 0.8525, 0.40, "distance", size=15, color=MUTED, ha="center")
    text(ax, 0.8525, 0.29, f"{D['rep_dist'][k]:.1f} m", size=44, weight="bold", ha="center", va="center")
    text(ax, 0.8525, 0.20, f"measured: {D['distance_m']:.2f} m", size=12.5, color=GREEN, ha="center")
    caption(ax, t, [(0, 2.6, "PyTorch saves the learned weights to JSON. They are copied to the board."),
                    (2.6, 99, "The board does the same math with numpy every 0.2 s (matches PyTorch to 1e-6).\n"
                              "Replaying the example walk through the board's code:")])


def scene_distance(ax, t, D):
    header(ax, 7, "Distance")
    c = D["coef"]
    text(ax, 0.5, 0.81, f"step length = {c['1']:.3f} + {c['swing_rad_s']:.3f} × swing − "
                         f"{-c['cadence_spm']:.5f} × cadence", size=20, ha="center", family=MONO)
    text(ax, 0.5, 0.76, "swing = average |gyro| in the window (rad/s)     cadence = steps per minute (from the net)",
         size=12.5, color=MUTED, ha="center")
    tt = D["rep_end_t"]
    p = ease(prog(t, 0.6, 6.0))
    k = int(min(len(tt) - 1, p * (len(tt) - 1)))
    fig = ax.figure
    walking = D["rep_out"] > 0.5
    a1 = fig.add_axes([0.07, 0.20, 0.40, 0.48])
    style(a1, "window end (s)", "step length (m)")
    m = walking.copy(); m[k + 1:] = False
    a1.plot(tt[m], D["rep_len"][m], "o", color=BLUE, ms=5)
    a1.set_xlim(0, tt[-1]); a1.set_ylim(0.4, 1.0)
    a1.set_title("step length, each 0.2 s (while walking)", fontsize=13, color=MUTED)
    a2 = fig.add_axes([0.56, 0.20, 0.40, 0.48])
    style(a2, "window end (s)", "distance (m)")
    a2.plot(tt[:k + 1], D["rep_dist"][:k + 1], color=BLUE, lw=3, label="board's estimate")
    a2.axhline(D["distance_m"], color=GREEN, ls="--", lw=2, label=f"measured {D['distance_m']:.2f} m")
    a2.set_xlim(0, tt[-1]); a2.set_ylim(0, 17)
    a2.legend(loc="upper left", fontsize=12, facecolor=PANEL, edgecolor="#3a4560", labelcolor=FG)
    a2.set_title("distance = Σ (new steps × step length)", fontsize=13, color=MUTED)
    if p >= 1:
        a2.text(tt[-1] - 0.3, D["rep_dist"][-1] - 3.2, f"{D['rep_dist'][-1]:.1f} m", fontsize=18,
                color=BLUE, ha="right", weight="bold")
    caption(ax, t, [(0, 99, "A faster shin swing means a longer step. At the same swing speed, a quicker rhythm\n"
                            "leaves less time per swing, so a shorter step. Fitted on 25 measured walks.")])


def scene_accuracy(ax, t, D):
    header(ax, 8, "How good is it?")
    al = ease(prog(t, 0.2, 0.8))
    panel(ax, 0.03, 0.22, 0.45, 0.60, alpha=al)
    text(ax, 0.05, 0.775, "Step count", size=20, weight="bold", alpha=al)
    text(ax, 0.05, 0.735, "5-fold cross-validation by recording × 3 seeds\n(every recording counted by a net that never saw it)",
         size=12, color=MUTED, va="top", alpha=al, linespacing=1.25)
    rows = [("Typical error per recording", "4.4%"), ("90% of recordings within", "±11%"),
            ("All 1,099 steps added up", "−0.3%"), ("Hardest: running", "+8%"),
            ("IMU only vs IMU + ultrasonic", "tie")]
    for i, (a_, b_) in enumerate(rows):
        al_i = ease(prog(t, 0.6 + 0.25 * i, 1.0 + 0.25 * i))
        text(ax, 0.05, 0.60 - i * 0.07, a_, size=16, alpha=al_i)
        text(ax, 0.46, 0.60 - i * 0.07, b_, size=17, weight="bold", ha="right", color=BLUE, alpha=al_i)
    al = ease(prog(t, 1.5, 2.1))
    panel(ax, 0.51, 0.22, 0.46, 0.60, alpha=al)
    text(ax, 0.53, 0.775, "Distance", size=20, weight="bold", alpha=al)
    text(ax, 0.53, 0.735, "leave-one-walk-out over 25 measured walks\ntypical error, by how step length is found",
         size=12, color=MUTED, va="top", alpha=al, linespacing=1.25)
    bars = [("one fixed length", 24.5), ("cadence", 10.8), ("swing speed", 6.2), ("cadence + swing (used)", 5.5)]
    for i, (lab, v) in enumerate(bars):
        g = ease(prog(t, 2.0 + 0.3 * i, 2.8 + 0.3 * i))
        y = 0.60 - i * 0.075
        text(ax, 0.53, y, lab, size=14, va="center", alpha=al)
        ax.add_patch(Rectangle((0.745, y - 0.018), 0.15 * v / 24.5 * g, 0.036, transform=ax.transAxes,
                               fc=GREEN if i == 3 else "#5b6a8a", ec="none"))
        if g > 0.5:
            text(ax, 0.75 + 0.15 * v / 24.5 * g, y, f" {v}%", size=14, va="center", weight="bold")
    text(ax, 0.53, 0.28, "Live check: 14.33 m stretch read 13.8 m, twice", size=13.5, color=MUTED, alpha=al)
    caption(ax, t, [(3.6, 99, "Limitations: fitted to one person, tested on short walks (10–75 steps),\n"
                              "and no GPS, so distance errors add up.")])


SCENES = [(4.0, scene_title), (12.0, scene_sensors), (10.0, scene_window), (15.0, scene_forward),
          (29.0, scene_training), (10.0, scene_deployed), (8.5, scene_distance), (9.0, scene_accuracy)]
STARTS = np.cumsum([0] + [d for d, _ in SCENES])
DURATION = float(STARTS[-1])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_D = None
_FIG = None


def _init(D):
    global _D
    _D = D


def render(tg):
    global _FIG
    if _FIG is None:
        _FIG = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100)
    fig = _FIG
    fig.clf()
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    i = int(np.searchsorted(STARTS, tg, side="right") - 1)
    i = min(i, len(SCENES) - 1)
    tl = tg - STARTS[i]
    SCENES[i][1](ax, tl, _D)
    # fade in/out between scenes
    d = SCENES[i][0]
    f = min(clip01(tl / 0.35) if i > 0 else 1.0, clip01((d - tl) / 0.35) if i < len(SCENES) - 1 else 1.0)
    if f < 1:
        ov = fig.add_axes([0, 0, 1, 1], zorder=100)
        ov.axis("off")
        ov.add_patch(Rectangle((0, 0), 1, 1, transform=ov.transAxes, fc=BG, alpha=1 - f))
    fig.canvas.draw()
    return bytes(fig.canvas.buffer_rgba())


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    global _D
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=float, nargs="*", help="write PNG frames at these times (s) instead")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--preview-dir", type=Path, default=HERE / "preview")
    args = ap.parse_args()
    print("computing data, replaying the walk, training a fresh net ...")
    _D = prepare()
    F = _D["first"]
    print(f"  example window: net {_D['acts']['out']:.2f}, labels {_D['win_true']}")
    print(f"  board replay: {_D['rep_total'][-1]:.2f} steps, {_D['rep_dist'][-1]:.2f} m")
    print(f"  first batch loss {F['loss']:.3f} -> {F['loss_after']:.3f}; "
          f"final held-out walk total {_D['hist']['ex_total'][-1]:.2f}")
    print(f"  video length {DURATION:.1f} s")
    if args.preview is not None:
        outdir = args.preview_dir
        outdir.mkdir(parents=True, exist_ok=True)
        for tg in args.preview:
            buf = render(tg)
            img = np.frombuffer(buf, np.uint8).reshape(H_PX, W_PX, 4)
            plt.imsave(outdir / f"frame_{tg:06.2f}.png", img)
        print(f"wrote {len(args.preview)} frames to {outdir}")
        return
    times = np.arange(int(DURATION * FPS)) / FPS
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{W_PX}x{H_PX}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "28",
           "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    jobs = args.jobs or None
    with Pool(jobs, initializer=_init, initargs=(_D,)) as pool:
        for k, buf in enumerate(pool.imap(render, times, chunksize=8)):
            proc.stdin.write(buf)
            if k % 300 == 0:
                print(f"  frame {k}/{len(times)}")
    proc.stdin.close()
    proc.wait()
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {DURATION:.0f} s)")


if __name__ == "__main__":
    main()
