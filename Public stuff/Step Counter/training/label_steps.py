"""Mark when each step happened in every recording, to use as training labels.

Right-leg steps come from the IMU on the right ankle: the gz gyro (rotation
of the shin forward/back) dips hard once per right-leg swing. Left-leg steps
can't be felt directly, so they go halfway between right steps, plus one
before the first or after the last when the spoken count says so -- at the
end with more leg motion that no right swing explains.

(The ultrasonic doesn't mark left steps: its dips line up with the *right*
swings, when the right calf swings forward past the left leg.)

The total always has to equal the count said out loud. If the detector's
right-step count can't be made to fit, the recording is flagged instead of
guessed at.

    python training/label_steps.py

Writes data/labels/<recording>.json and a check plot per recording in
data/label_plots/.
"""
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FS = 50  # samples per second
ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "recordings"
LABEL_DIR = ROOT / "data" / "labels"
PLOT_DIR = ROOT / "data" / "label_plots"

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz", "dist")
MIN_STRIDE_S = 0.35   # two right swings can't be closer than this (fast run ~0.55 s)
MOVING_GYRO = 0.5     # rad/s; |gyro| above this = the leg is moving


def load(name):
    with open(REC_DIR / f"{name}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    data = {k: np.array([float(r[k]) for r in rows]) for k in CHANNELS}
    meta = json.loads((REC_DIR / f"{name}.json").read_text())
    return data, meta


SHALLOW_END_FRAC = 0.6  # end swing shallower than this x the typical swing = "maybe" step


def right_swings(gz):
    """Bottoms of the gz dips = middles of right-leg swings.

    Returns (full, start, end): swings that always count, and the "maybe"
    swings at each end (innermost first). Maybe swings are the shallow
    half-steps of starting off or bringing the feet together -- sometimes
    counted out loud, sometimes not -- so label() decides from the spoken
    count. Shallow dips in the middle of a walk are ignored.
    """
    smooth = np.convolve(gz, np.ones(5) / 5, mode="same")  # 0.1 s moving average
    depth = -smooth
    # Scale the thresholds to this recording, so slow walks and runs both work.
    typical = np.percentile(depth, 98)
    peaks, _ = find_peaks(
        depth,
        height=max(1.0, 0.15 * typical),
        prominence=max(0.5, 0.15 * typical),
        distance=int(MIN_STRIDE_S * FS),
    )
    is_full = depth[peaks] >= max(1.0, 0.35 * typical)
    full = list(peaks[is_full])
    if not full:
        return np.array([], dtype=int), [], []
    shallow = peaks[~is_full]
    before = shallow[shallow < full[0]]
    after = shallow[shallow > full[-1]]
    # Only the shallow swing right next to the walk can be a real step.
    start = [int(before[-1])] if len(before) else []
    end = [int(after[0])] if len(after) else []
    # A first/last swing much shallower than this walk's typical swing is a
    # start-off or feet-together half-step too, even if it passed above.
    typical_swing = np.median(depth[full])
    if len(full) > 2 and depth[full[0]] < SHALLOW_END_FRAC * typical_swing:
        start = [int(full.pop(0))] + start
    if len(full) > 2 and depth[full[-1]] < SHALLOW_END_FRAC * typical_swing:
        end = [int(full.pop())] + end
    return np.array(full), start, end


def fit_right(full, start, end, counted):
    """Pick how many "maybe" swings to include at each end (innermost first)
    so the right-step count fits the spoken count -- right and left steps
    alternate, so they differ by at most one. Prefers including real-looking
    motion. None if nothing fits."""
    options = []
    for k_start in range(len(start) + 1):
        for k_end in range(len(end) + 1):
            r = start[:k_start][::-1] + list(full) + end[:k_end]
            if abs(counted - 2 * len(r)) <= 1:
                options.append(np.array(r))
    if not options:
        return None
    return max(options, key=len)


def place_left(right, n_left, gyro_mag, n):
    """Left-step indices given right-step indices, or None if they can't fit."""
    if len(right) == 0:
        return None
    mids = list((right[:-1] + right[1:]) // 2)
    half = int(np.median(np.diff(right)) / 2) if len(right) > 1 else FS // 2
    before = max(0, right[0] - half)
    after = min(n - 1, right[-1] + half)
    extra = n_left - len(mids)
    if extra == 0:
        return mids
    if extra == 2:
        return [before] + mids + [after]
    if extra == 1:
        # The extra left step goes where the leg was moving longest before the
        # first / after the last right swing.
        moving = np.flatnonzero(gyro_mag > MOVING_GYRO)
        if moving.size == 0:
            return mids + [after]
        lead = right[0] - moving[0]
        trail = moving[-1] - right[-1]
        return [before] + mids if lead > trail else mids + [after]
    return None


def label(name):
    data, meta = load(name)
    n = len(data["gz"])
    counted = meta["steps_counted"]
    full, start, end = right_swings(data["gz"])
    n_shallow = len(start) + len(end)
    out = {
        "name": name,
        "activity": meta["activity"],
        "steps_counted": counted,
        "samples": n,
        "right_detected": f"{len(full)}+{n_shallow}" if n_shallow else str(len(full)),
        "right": [],
        "left": [],
        "status": "ok",
    }
    if counted < 0:
        out["status"] = "no spoken count"
    elif counted == 0:
        pass  # stand / shuffle / bike: no steps, whatever the gyro did
    else:
        right = fit_right(full, start, end, counted)
        gyro_mag = np.sqrt(data["gx"] ** 2 + data["gy"] ** 2 + data["gz"] ** 2)
        left = None if right is None else place_left(right, counted - len(right), gyro_mag, n)
        if left is None:
            out["status"] = (f"mismatch: {len(full)} full + {n_shallow} shallow right swings "
                             f"can't make {counted} steps")
        else:
            out["right"] = [int(i) for i in right]
            out["left"] = [int(i) for i in left]
    out["steps"] = sorted(out["right"] + out["left"])
    return out, data


def plot(lab, data):
    t = np.arange(lab["samples"]) / FS
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    a1.plot(t, data["gz"], lw=0.8, color="tab:green")
    full, start, end = right_swings(data["gz"])
    a1.plot(full / FS, data["gz"][full], "v", color="tab:red", label="full swing")
    ends = start + end
    if ends:
        a1.plot(np.array(ends) / FS, data["gz"][ends], "v", mfc="none", color="tab:red", label="shallow end swing")
    a1.set_ylabel("gz rad/s")
    a1.legend(loc="lower right")
    d = data["dist"]
    a2.plot(t, np.where(d >= 0, d, np.nan), ".", ms=2, color="tab:green")
    for i in lab["left"]:
        a2.axvline(i / FS, color="tab:blue", lw=1)
    for i in lab["right"]:
        a2.axvline(i / FS, color="tab:red", lw=1, ls=":")
    a2.set_ylim(0, 150)
    a2.set_ylabel("dist cm")
    a2.set_xlabel("s  (blue = left step, red dotted = right step)")
    a1.set_title(f"{lab['name']}  counted {lab['steps_counted']}, labeled {len(lab['steps'])}  [{lab['status']}]")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{lab['name']}.png", dpi=70)
    plt.close(fig)


def main():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in REC_DIR.glob("*.csv"))
    flagged = 0
    print("Recordings with shallow end swings or problems (the rest fit cleanly):")
    print(f"{'recording':32} {'activity':14} {'said':>4} {'right':>5} {'label':>5}  status")
    for name in names:
        lab, data = label(name)
        (LABEL_DIR / f"{name}.json").write_text(json.dumps(lab, indent=1))
        plot(lab, data)
        flagged += lab["status"] != "ok"
        if lab["status"] != "ok" or "+" in lab["right_detected"]:
            print(f"{name:32} {lab['activity']:14} {lab['steps_counted']:>4} "
                  f"{lab['right_detected']:>5} {len(lab['steps']):>5}  {lab['status']}")
    print(f"\n{len(names) - flagged}/{len(names)} recordings labeled, {flagged} flagged")


if __name__ == "__main__":
    main()
