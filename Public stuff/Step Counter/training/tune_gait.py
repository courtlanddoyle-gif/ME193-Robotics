"""Re-fit the Gait ID rule models (from the last project) to the new mounting.

    python training/tune_gait.py

The IMU moved from the outside to the inside of the right ankle, which flips
its axes, so the old thresholds no longer apply. The rules keep their shape:

  Model 1 (one number): peak-to-peak accel on ONE axis, sorted into bands
      stand < t1 <= bike < t2 <= walk < t3 <= run
  Model 2 (all of the IMU), questions in order, first "yes" wins:
      1. still?             accel_mag_std < T_std                         -> standing
      2. bike-like?         tilt < T_bike_tilt
                            OR (gyro_mag_max < T_gyro AND accel_mag_ptp < T_bike_ptp)  -> biking
      3. run-like?          tilt > T_run_tilt OR accel_mag_ptp > T_run_ptp  -> running
      4. otherwise                                                        -> walking

Searched: the thresholds, Model 1's axis, and which accel axis/sign is the
"tilt" (the old ay_mean). Windows are the board's 2 s windows from the step
recordings, labeled by activity. 5-fold cross-validation by recording gives
honest accuracy; the final fit on everything is saved to
python/models/gait_rules.json for python/main.py.
"""
import json
from collections import defaultdict
from itertools import product

import numpy as np

from common import FS, HOP, MODEL_DIR, RESULTS_DIR, WINDOW, labeled_recordings, load_recording

CLASSES = ["standing", "walking", "running", "biking"]
ACTIVITY_CLASS = {
    "stand": "standing", "shuffle": "standing",
    "walk-normal": "walking", "walk-slow": "walking", "walk-fast": "walking", "turns": "walking",
    "distance-walk": "walking", "stairs-up": "walking", "stairs-down": "walking",
    "jog": "running", "run": "running",
    "bike": "biking",
}
EDGE = 2 * FS        # skip the first/last 2 s of each recording (standing at Start/Stop)
N_FOLDS = 5
GRID = 25            # candidate thresholds per parameter (quantiles of the feature)
TILTS = ("ax_mean", "ay_mean", "az_mean", "-ax_mean", "-ay_mean", "-az_mean")


def window_features(w):
    """Same formulas as compute_features() in python/main.py."""
    acc, gyr = w[:, 0:3], w[:, 3:6]
    amag = np.sqrt((acc ** 2).sum(1))
    gmag = np.sqrt((gyr ** 2).sum(1))
    f = {
        "ax_ptp": np.ptp(acc[:, 0]), "ay_ptp": np.ptp(acc[:, 1]), "az_ptp": np.ptp(acc[:, 2]),
        "accel_mag_std": amag.std(), "accel_mag_ptp": np.ptp(amag),
        "gyro_mag_max": gmag.max(),
        "ax_mean": acc[:, 0].mean(), "ay_mean": acc[:, 1].mean(), "az_mean": acc[:, 2].mean(),
    }
    for t in ("ax_mean", "ay_mean", "az_mean"):
        f["-" + t] = -f[t]
    return f


def build_windows():
    feats, labels, groups = defaultdict(list), [], []
    for name, act in labeled_recordings():
        cls = ACTIVITY_CLASS[act]
        x, steps, _ = load_recording(name, use_dist=False)
        for s in range(EDGE, len(x) - EDGE - WINDOW + 1, HOP):
            # In walking/running recordings, only windows with real stepping
            # in them (at least 2 steps) count as that activity.
            if cls in ("walking", "running") and np.sum((steps >= s) & (steps < s + WINDOW)) < 2:
                continue
            for k, v in window_features(x[s:s + WINDOW]).items():
                feats[k].append(v)
            labels.append(CLASSES.index(cls))
            groups.append(name)
    return {k: np.array(v) for k, v in feats.items()}, np.array(labels), np.array(groups)


def balanced_acc(pred, y, classes=None):
    classes = np.unique(y) if classes is None else classes
    return np.mean([np.mean(pred[y == c] == c) for c in classes if np.any(y == c)])


def grid(v):
    """Candidate thresholds: quantiles of the feature, plus the midpoints
    between them, so a big gap between classes has candidates inside it."""
    q = np.unique(np.quantile(v, np.linspace(0.01, 0.99, GRID)))
    return np.unique(np.concatenate([q, (q[:-1] + q[1:]) / 2]))


def center(params, key, candidates, score_fn):
    """Slide one threshold to the middle of the range of values that score as
    well as its current one (holding the others fixed). Several thresholds
    usually tie on the training data; the middle of the gap between classes
    leaves the most room for a strap that sits a little differently."""
    cur = params[key]
    if not np.isfinite(cur):
        return
    cands = np.unique(np.append(candidates, cur))
    scores = np.array([score_fn({**params, key: c}) for c in cands])
    i = int(np.searchsorted(cands, cur))
    best = scores[i]
    lo = i
    while lo > 0 and scores[lo - 1] >= best - 1e-12:
        lo -= 1
    hi = i
    while hi < len(cands) - 1 and scores[hi + 1] >= best - 1e-12:
        hi += 1
    params[key] = float((cands[lo] + cands[hi]) / 2)


# --- Model 1 ----------------------------------------------------------------

def predict_m1(f, p):
    v = f[p["axis"] + "_ptp"]
    return np.select([v < p["t1"], v < p["t2"], v < p["t3"]],
                     [CLASSES.index("standing"), CLASSES.index("biking"), CLASSES.index("walking")],
                     CLASSES.index("running"))


def fit_m1(f, y):
    best = (-1, None)
    for axis in ("ax", "ay", "az"):
        g = grid(f[axis + "_ptp"])
        for t1, t2, t3 in product(g, g, g):
            if not t1 < t2 < t3:
                continue
            p = {"axis": axis, "t1": t1, "t2": t2, "t3": t3}
            score = balanced_acc(predict_m1(f, p), y)
            if score > best[0]:
                best = (score, p)
    p = best[1]
    g = grid(f[p["axis"] + "_ptp"])
    for key in ("t1", "t2", "t3"):
        center(p, key, g, lambda q: balanced_acc(predict_m1(f, q), y)
               if q["t1"] < q["t2"] < q["t3"] else -1)
    return p


# --- Model 2 ----------------------------------------------------------------

def predict_m2(f, p):
    tilt = f[p["tilt"]]
    still = f["accel_mag_std"] < p["T_std"]
    bike = (tilt < p["T_bike_tilt"]) | ((f["gyro_mag_max"] < p["T_gyro"]) & (f["accel_mag_ptp"] < p["T_bike_ptp"]))
    run = (tilt > p["T_run_tilt"]) | (f["accel_mag_ptp"] > p["T_run_ptp"])
    return np.select([still, bike, run],
                     [CLASSES.index("standing"), CLASSES.index("biking"), CLASSES.index("running")],
                     CLASSES.index("walking"))


def fit_m2(f, y):
    """One question at a time, in the order the model asks them."""
    S, W, R, B = (CLASSES.index(c) for c in ("standing", "walking", "running", "biking"))
    # 1. still?
    T_std = max(grid(f["accel_mag_std"]),
                key=lambda t: balanced_acc((f["accel_mag_std"] < t).astype(int), (y == S).astype(int)))
    rest = f["accel_mag_std"] >= T_std
    # 2. bike-like? (on what's left). -inf/+inf lets a sub-rule switch itself off.
    best = (-1, None)
    for tilt in TILTS:
        tv = f[tilt][rest]
        yb = (y[rest] == B).astype(int)
        gm, ap = f["gyro_mag_max"][rest], f["accel_mag_ptp"][rest]
        for tt in np.append(grid(tv), -np.inf):
            a = tv < tt
            for tg, tp in product(np.append(grid(gm), -np.inf), grid(ap)):
                score = balanced_acc((a | ((gm < tg) & (ap < tp))).astype(int), yb)
                if score > best[0]:
                    best = (score, (tilt, tt, tg, tp))
    tilt, T_bike_tilt, T_gyro, T_bike_ptp = best[1]
    p = {"tilt": tilt, "T_std": T_std, "T_bike_tilt": T_bike_tilt, "T_gyro": T_gyro, "T_bike_ptp": T_bike_ptp}
    bike = (f[tilt] < T_bike_tilt) | ((f["gyro_mag_max"] < T_gyro) & (f["accel_mag_ptp"] < T_bike_ptp))
    rest2 = rest & ~bike & ((y == W) | (y == R))
    # 3. run-like? (walk vs run on what's left)
    yr = (y[rest2] == R).astype(int)
    tv, ap = f[tilt][rest2], f["accel_mag_ptp"][rest2]
    best = (-1, None)
    for tt, tp in product(np.append(grid(tv), np.inf), np.append(grid(ap), np.inf)):
        score = balanced_acc(((tv > tt) | (ap > tp)).astype(int), yr)
        if score > best[0]:
            best = (score, (tt, tp))
    p["T_run_tilt"], p["T_run_ptp"] = best[1]

    # Center every threshold in its tie range, judged on the whole 4-class model.
    def score(q):
        return balanced_acc(predict_m2(f, q), y)
    for key, feat in (("T_std", "accel_mag_std"), ("T_bike_tilt", tilt), ("T_gyro", "gyro_mag_max"),
                      ("T_bike_ptp", "accel_mag_ptp"), ("T_run_tilt", tilt), ("T_run_ptp", "accel_mag_ptp")):
        center(p, key, grid(f[feat]), score)
    return p


def confusion(pred, y):
    m = np.zeros((4, 4), int)
    for t, p in zip(y, pred):
        m[t, p] += 1
    return m


def report(name, pred, y):
    m = confusion(pred, y)
    print(f"  {name}: {100 * np.mean(pred == y):.0f}% of windows right, "
          f"balanced {100 * balanced_acc(pred, y):.0f}%")
    print(f"    {'true \\ predicted':18}" + "".join(f"{c:>10}" for c in CLASSES) + "   recall")
    for i, c in enumerate(CLASSES):
        print(f"    {c:18}" + "".join(f"{v:10d}" for v in m[i]) + f"   {100 * m[i, i] / max(1, m[i].sum()):4.0f}%")


def main():
    f, y, groups = build_windows()
    print(f"{len(y)} windows from {len(set(groups))} recordings: "
          + ", ".join(f"{c} {np.sum(y == i)}" for i, c in enumerate(CLASSES)) + "\n")

    # Folds by recording, each activity spread over the folds.
    names = sorted(set(groups))
    by_cls = defaultdict(list)
    for n in names:
        by_cls[y[groups == n][0]].append(n)
    fold_of, k = {}, 0
    for c in sorted(by_cls):
        for n in by_cls[c]:
            fold_of[n] = k % N_FOLDS
            k += 1
    folds = np.array([fold_of[g] for g in groups])

    p1 = np.zeros_like(y)
    p2 = np.zeros_like(y)
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        ftr = {n: v[tr] for n, v in f.items()}
        fte = {n: v[te] for n, v in f.items()}
        p1[te] = predict_m1(fte, fit_m1(ftr, y[tr]))
        p2[te] = predict_m2(fte, fit_m2(ftr, y[tr]))
        print(f"  fold {k + 1}/{N_FOLDS} done")
    print("\nCross-validated (each recording judged by rules fitted without it):")
    report("Model 1", p1, y)
    report("Model 2", p2, y)

    m1, m2 = fit_m1(f, y), fit_m2(f, y)
    print("\nFinal rules, fitted on all recordings:")
    print(f"  Model 1: {m1['axis']} peak-to-peak: standing < {m1['t1']:.2f} <= biking < {m1['t2']:.2f} "
          f"<= walking < {m1['t3']:.2f} <= running")
    print(f"  Model 2: standing if accel_mag_std < {m2['T_std']:.2f}; "
          f"biking if {m2['tilt']} < {m2['T_bike_tilt']:.2f} OR (gyro_mag_max < {m2['T_gyro']:.2f} "
          f"AND accel_mag_ptp < {m2['T_bike_ptp']:.2f}); running if {m2['tilt']} > {m2['T_run_tilt']:.2f} "
          f"OR accel_mag_ptp > {m2['T_run_ptp']:.2f}; else walking")
    print("  (in-sample) ", end="")
    report("Model 1", predict_m1(f, m1), y)
    print("  (in-sample) ", end="")
    report("Model 2", predict_m2(f, m2), y)

    def clean(p):  # JSON has no infinity
        return {k: (v if isinstance(v, str) else (None if not np.isfinite(v) else float(v))) for k, v in p.items()}
    (MODEL_DIR / "gait_rules.json").write_text(json.dumps({"model1": clean(m1), "model2": clean(m2)}, indent=1))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(RESULTS_DIR / "gait_confusion_model2.csv", confusion(p2, y), fmt="%d", delimiter=",",
               header=",".join(CLASSES))
    print("\nsaved python/models/gait_rules.json")


if __name__ == "__main__":
    main()
