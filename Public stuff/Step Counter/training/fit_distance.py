"""Fit the distance estimate: how long a step is, as a function of cadence.

    python training/fit_distance.py

Steps aren't all the same length (your measured walks go from ~0.5 m per
step walking slowly to ~1.1 m running), so one fixed step length is wrong at
most speeds. Candidates, each a straight line:

    step length (m) = a                                  (fixed)
                    = a + b * cadence (steps/min)
                    = a + b * swing   (avg |gyro|, rad/s: how fast the shin swings)
                    = a + b * cadence + c * swing

The board runs the step net every 0.2 s and adds `inc` steps (1/10 of the
net's count for the last 2 s). Distance is then, e.g. for swing,

    D = sum(inc * (a + b * swing)) = a * sum(inc) + b * sum(inc * swing)

which is linear in a and b, so they're fitted by least squares straight from
the walks over a measured distance -- using exactly what the board computes,
start and end of each walk included. Each walk is predicted by a fit that
left it out, for an honest error estimate, and the simplest version that's
about as good as the best one is saved for the board.

Saves python/models/distance_model.json and, in training/results/,
distance_results.csv and step_length.png.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

from common import FS, HOP, MODEL_DIR, RESULTS_DIR, ROOT, WINDOW, labeled_recordings, load_recording

sys.path.insert(0, str(ROOT / "python"))
from step_net import StepNet  # noqa: E402  -- the board's own numpy code

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz", "dist")
# m; only catches wild inputs. Kept outside what the fitted line gives for real
# walking (~0.4-1.1 m), so the board computes what was tested here.
STEP_LEN_MIN, STEP_LEN_MAX = 0.2, 1.6


def board_windows(name, net):
    """(inc, cad, swing) for every 0.2 s update, computed the way main.py does.
    swing = average |gyro| over the window (main.py's gyro_mag_mean): a
    longer step needs a bigger, faster leg swing, even at the same cadence."""
    with open(ROOT / "data" / "recordings" / f"{name}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    raw = {c: np.array([float(r[c]) for r in rows], np.float32) for c in CHANNELS}
    # Pad with the standing-still first/last sample, like training/common.py.
    pad = {c: np.concatenate([np.repeat(v[:1], WINDOW), v, np.repeat(v[-1:], WINDOW)]) for c, v in raw.items()}
    starts = range(0, len(rows) + WINDOW, HOP)
    preds = np.array([max(0.0, net.steps_in_window({c: v[s:s + WINDOW] for c, v in pad.items()}))
                      for s in starts])
    gmag = np.sqrt(pad["gx"] ** 2 + pad["gy"] ** 2 + pad["gz"] ** 2)
    swing = np.array([gmag[s:s + WINDOW].mean() for s in starts])
    inc = preds * HOP / WINDOW
    cad = preds * 60.0 / (WINDOW / FS)
    return inc, cad, swing


def fit(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def main():
    net = StepNet(MODEL_DIR / "step_model_imu_dist.json")
    walks = []
    for name, act in labeled_recordings():
        meta = json.loads((ROOT / "data" / "recordings" / f"{name}.json").read_text())
        if meta.get("distance_m", 0) > 0:
            _, steps, lab = load_recording(name)
            inc, cad, swing = board_windows(name, net)
            t = steps / FS
            walks.append({
                "name": name, "activity": act, "distance": meta["distance_m"],
                "counted": lab["steps_counted"], "net_steps": inc.sum(),
                # sum(inc * term) for each term of the step-length formula
                "terms": {"1": inc.sum(), "cadence_spm": (inc * cad).sum(), "swing_rad_s": (inc * swing).sum()},
                # For the plot: the walk's actual step length, cadence and swing speed.
                "step_len": meta["distance_m"] / lab["steps_counted"],
                "cadence": 60 * (len(steps) - 1) / (t[-1] - t[0]),
                "swing": float((inc * swing).sum() / inc.sum()),
            })
    print(f"{len(walks)} walks over a measured distance\n")
    y = np.array([w["distance"] for w in walks])

    # Step length = c0 + c1 * term1 + ... ; each version tested with every walk
    # predicted by a fit that left that walk out.
    versions = {
        "fixed": ["1"],
        "cadence": ["1", "cadence_spm"],
        "swing": ["1", "swing_rad_s"],
        "cadence+swing": ["1", "cadence_spm", "swing_rad_s"],
    }
    scores = {}
    for v, terms in versions.items():
        X = np.array([[w["terms"][t] for t in terms] for w in walks])
        for i, w in enumerate(walks):
            keep = np.arange(len(walks)) != i
            w[f"pred_{v}"] = float(X[i] @ fit(X[keep], y[keep]))
        pct = np.array([100 * (w[f"pred_{v}"] - w["distance"]) / w["distance"] for w in walks])
        scores[v] = pct

    print(f"{'walk':32} {'activity':14} {'true m':>7} {'steps':>5} {'net':>6}" + "".join(f"{v:>15}" for v in versions))
    for w in walks:
        print(f"{w['name']:32} {w['activity']:14} {w['distance']:7.2f} {w['counted']:5d} {w['net_steps']:6.1f}"
              + "".join(f"{w[f'pred_{v}']:15.2f}" for v in versions))
    print("\nStep length from ... (each walk left out of its own fit):")
    for v, pct in scores.items():
        print(f"  {v:15} mean abs error {np.abs(pct).mean():4.1f}%, bias {pct.mean():+5.1f}%, "
              f"spread {pct.std():4.1f}%, worst {np.abs(pct).max():4.0f}%")

    # With only ~17 walks, a version that's barely better isn't really better:
    # take the one with the fewest terms within 0.5% of the lowest error.
    mae = {v: np.abs(p).mean() for v, p in scores.items()}
    best = min(versions, key=lambda v: (mae[v] > min(mae.values()) + 0.5, len(versions[v]), mae[v]))
    terms = versions[best]
    coef = fit(np.array([[w["terms"][t] for t in terms] for w in walks]), y)
    print(f"\nbest: {best}. Final fit on all walks: step length = "
          + " + ".join(f"{c:.4f}" + ("" if t == "1" else f" x {t}") for c, t in zip(coef, terms)) + " m")

    (MODEL_DIR / "distance_model.json").write_text(json.dumps({
        "terms": terms, "coef": [float(c) for c in coef], "min_m": STEP_LEN_MIN, "max_m": STEP_LEN_MAX,
        "fitted_on": len(walks),
        "loo_mean_abs_pct": round(float(np.abs(scores[best]).mean()), 1)}, indent=1))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "distance_results.csv", "w", newline="") as f:
        cols = ["name", "activity", "distance", "counted", "net_steps", "cadence", "swing", "step_len",
                *[f"pred_{v}" for v in versions]]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for w in walks:
            wr.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in w.items()})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for act in sorted({w["activity"] for w in walks}):
        ws = [w for w in walks if w["activity"] == act]
        ax1.plot([w["cadence"] for w in ws], [w["step_len"] for w in ws], "o", label=act)
        ax2.plot([w["distance"] for w in ws], [w[f"pred_{best}"] for w in ws], "o", label=act)
    ax1.set_xlabel("cadence (steps/min)")
    ax1.set_ylabel("actual step length (m)")
    ax1.set_title("Faster steps are longer, but not on one line")
    ax1.legend(fontsize=8)
    lim = [0, max(y.max(), max(w[f"pred_{best}"] for w in walks)) * 1.1]
    ax2.plot(lim, lim, "k-", lw=1, label="perfect")
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("measured distance (m)")
    ax2.set_ylabel("estimated distance (m)")
    ax2.set_title(f"Distance estimate ({best}), each walk left out of its own fit")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "step_length.png", dpi=120)
    print("\nsaved python/models/distance_model.json, training/results/distance_results.csv, step_length.png")


if __name__ == "__main__":
    main()
