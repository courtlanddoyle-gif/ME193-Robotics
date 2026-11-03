# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import csv
import json
import math
import re
import statistics
import threading
import time
from collections import deque
from pathlib import Path

from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI

from step_net import StepNet

logger = Logger("step-counter")

# Live page at http://<board IP>:7000 (assets/index.html), polling
# /api/gait -- easier to read on a phone than the LED matrix when the board
# is strapped to an ankle.
ui = WebUI()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAMPLE_RATE_HZ = 50          # must match `interval` in sketch.ino
WINDOW_SEC = 2.0             # long enough to cover a full gait/pedal cycle
WINDOW_SIZE = int(SAMPLE_RATE_HZ * WINDOW_SEC)
UPDATE_EVERY = 10            # re-classify every 10 samples (~0.2s)

STAND, WALK, RUN, BIKE = 0, 1, 2, 3
LABELS = {STAND: "standing", WALK: "walking", RUN: "running", BIKE: "biking"}

# Gait ID thresholds (Model 1 and Model 2) are loaded from
# models/gait_rules.json further down -- re-fitted by training/tune_gait.py
# after the IMU moved to the inside of the ankle (2026-11-03).

CALIBRATE_LOG = True         # print live feature values to help tune thresholds

# /app is the app folder on the board, so recordings land in
# /home/arduino/ArduinoApps/step-counter/data/recordings.
RECORD_DIR = Path("/app/data/recordings")

# --- Step-counting neural nets (trained on the laptop by training/train.py) --
# Both take the same 2 s window; the IMU + ultrasonic one is used whenever the
# HC-SR04 is answering, the IMU-only one when it isn't (e.g. a loose wire),
# since the dist net undercounts ~5% with its distance input dead.
MODEL_DIR = Path(__file__).resolve().parent / "models"
NET_IMU_DIST = StepNet(MODEL_DIR / "step_model_imu_dist.json")
NET_IMU = StepNet(MODEL_DIR / "step_model_imu.json")
DIST_ALIVE_MIN = 0.8   # fraction of the window the sensor must answer to trust it
# A completely motionless sensor (e.g. set down on a table) is something the
# nets never saw -- they were only trained on it worn on the ankle -- and they
# guess ~1 step / 2 s. A 2 s window with no motion can't hold a step, so skip
# the net. Every labeled window with a step had gyro max >= 0.33 rad/s; these
# limits are well below that.
STILL_GYRO_MAX = 0.15  # rad/s
STILL_ACCEL_STD = 0.1  # m/s^2

# --- Distance (fitted on the laptop by training/fit_distance.py) -------------
# step length = coef . [1, term, ...]; the fit picked the leg's swing speed
# (average |gyro|), since a longer step takes a faster swing.
DIST_MODEL = json.loads((MODEL_DIR / "distance_model.json").read_text())


# --- Gait ID rules (fitted on the laptop by training/tune_gait.py) -----------
# Same rule shapes as the last project, new numbers for the new mounting.
# Model 1: peak-to-peak accel on one axis, in bands (m/s^2).
# Model 2: still? -> bike-like? -> run-like? -> else walking. "tilt" is the
# mean accel along one axis = which way gravity pulls = how the shin is
# angled: ~10 m/s^2 along x when upright, ~0.5 with the shin horizontal
# (pedaling on your back / in a chair). A threshold of None turns that
# sub-rule off. Only valid if the IMU is strapped on the same way.
GAIT = json.loads((MODEL_DIR / "gait_rules.json").read_text())
M1, M2 = GAIT["model1"], GAIT["model2"]
MODEL1_AXIS = M1["axis"]


def _tilt(f):
    name = M2["tilt"]
    return -f[name[1:]] if name.startswith("-") else f[name]


def _below(value, threshold):
    return threshold is not None and value < threshold


def _above(value, threshold):
    return threshold is not None and value > threshold


def step_length_m(features, cadence_spm):
    values = {"1": 1.0, "cadence_spm": cadence_spm, "swing_rad_s": features["gyro_mag_mean"]}
    length = sum(c * values[t] for c, t in zip(DIST_MODEL["coef"], DIST_MODEL["terms"]))
    return min(max(length, DIST_MODEL["min_m"]), DIST_MODEL["max_m"])
assert NET_IMU_DIST.window == NET_IMU.window == WINDOW_SIZE
assert NET_IMU_DIST.hop == NET_IMU.hop == UPDATE_EVERY

# ---------------------------------------------------------------------------
# Gyro bias calibration
# ---------------------------------------------------------------------------
# MPU6050's gyro has a factory zero-rate offset (datasheet allows up to
# +-20 deg/s = ~0.35 rad/s) that doesn't average out to zero on its own.
# Left uncorrected it makes "gyro magnitude" look nonzero even at rest,
# which inflates the gyro features Model 2 depends on. Average the
# first couple seconds of samples (board must be held still) and subtract
# that offset from every sample afterwards.
GYRO_CALIB_SAMPLES = SAMPLE_RATE_HZ * 2

gyro_bias = {"gx": 0.0, "gy": 0.0, "gz": 0.0}
_calib_sums = {"gx": 0.0, "gy": 0.0, "gz": 0.0}
_calib_count = 0
calibrated = False

# ---------------------------------------------------------------------------
# Sliding window buffers
# ---------------------------------------------------------------------------

# dist = HC-SR04 reading in cm, from the inside of the right calf aimed at
# the left leg. -1 = sensor not responding (wiring), -2 = nothing in range.
# Recordings made before 2026-11-03 used -1 for both.
DIST_NO_RESPONSE = -1.0
buf = {k: deque(maxlen=WINDOW_SIZE) for k in ("ax", "ay", "az", "gx", "gy", "gz", "dist")}
sample_count = 0

last_result = {
    "model1": None,
    "model2": None,
    "features": {},
    "t": 0.0,
}

# Running step count from the neural net. Updated on the Bridge thread, reset
# from the web server's, so both go through step_lock.
step_lock = threading.Lock()
steps = {"total": 0.0, "distance_m": 0.0, "cadence_spm": 0.0, "step_len_m": 0.0,
         "model": None, "since": time.time()}

# ---------------------------------------------------------------------------
# Recording (training data for the step counter)
# ---------------------------------------------------------------------------
# One CSV per recording, every sample after gyro bias correction, plus a JSON
# file with what you were doing and how many steps you counted out loud.
# imu_sample runs on the Bridge thread and the API on the web server's, so
# both go through rec_lock.

CSV_HEADER = ["i", "t", "ax", "ay", "az", "gx", "gy", "gz", "dist"]

rec_lock = threading.Lock()
rec = {"file": None, "writer": None, "name": None, "meta": None, "n": 0, "t0": 0.0,
       "paused": False, "t_end": 0.0, "echo_n": 0, "alive_n": 0}


def _safe(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip())[:40] or "unnamed"


def record_start(activity: str = "walk", distance_m: float = 0.0, note: str = ""):
    """POST /api/record/start -- begin writing samples to a new CSV."""
    with rec_lock:
        if rec["file"] is not None:
            return {"ok": False, "error": f"already recording {rec['name']}"}
        if not calibrated:
            return {"ok": False, "error": "gyro still calibrating, hold still"}
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{_safe(activity)}"
        f = open(RECORD_DIR / f"{name}.csv", "w", newline="")
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        rec.update(file=f, writer=writer, name=name, n=0, t0=time.time(), paused=False, echo_n=0, alive_n=0)
        rec["meta"] = {
            "name": name,
            "activity": activity,
            "distance_m": distance_m,
            "note": note,
            "start_time": rec["t0"],
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "gyro_bias": dict(gyro_bias),
            "mount": "IMU inside right ankle; HC-SR04 inside right calf, upside down, aimed at left leg",
        }
    logger.info(f"Recording started: {name}")
    return {"ok": True, "name": name}


def record_pause():
    """POST /api/record/pause -- the Stop button: stop writing samples right
    away, so typing in the step count doesn't add standing time to the end."""
    with rec_lock:
        if rec["file"] is None:
            return {"ok": False, "error": "not recording"}
        if not rec["paused"]:
            rec["paused"] = True
            rec["t_end"] = time.time()
        return {"ok": True, "samples": rec["n"]}


def record_stop(steps: int = -1, distance_m: float = -1.0, keep: bool = True):
    """POST /api/record/stop -- close the CSV and save the step count you
    counted (steps=-1 means unknown). keep=false throws the recording away."""
    with rec_lock:
        if rec["file"] is None:
            return {"ok": False, "error": "not recording"}
        rec["file"].close()
        t_end = rec["t_end"] if rec["paused"] else time.time()
        name, meta, n = rec["name"], rec["meta"], rec["n"]
        echo_n, alive_n = rec["echo_n"], rec["alive_n"]
        rec.update(file=None, writer=None, name=None, meta=None, n=0, paused=False)
    csv_path = RECORD_DIR / f"{name}.csv"
    if not keep:
        csv_path.unlink(missing_ok=True)
        logger.info(f"Recording discarded: {name}")
        return {"ok": True, "discarded": name}
    meta["steps_counted"] = steps
    if distance_m >= 0:
        meta["distance_m"] = distance_m
    meta["samples"] = n
    # echo_frac = got a real distance (lower in big rooms, that's fine).
    # alive_frac = sensor answered at all; well below 1.0 means the HC-SR04
    # wiring came loose during the recording.
    meta["echo_frac"] = round(echo_n / n, 3) if n else 0.0
    meta["alive_frac"] = round(alive_n / n, 3) if n else 0.0
    meta["duration_s"] = round(t_end - meta["start_time"], 2)
    (RECORD_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Recording saved: {name} ({n} samples, {steps} steps)")
    return {"ok": True, "saved": name, "samples": n, "steps": steps}


def list_recordings():
    """GET /api/recordings -- newest first, for the page's list."""
    out = []
    for p in sorted(RECORD_DIR.glob("*.json"), reverse=True):
        try:
            m = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        out.append({k: m.get(k) for k in ("name", "activity", "steps_counted", "distance_m", "duration_s", "echo_frac", "alive_frac")})
    return {"recordings": out}


def rec_status():
    with rec_lock:
        if rec["file"] is None:
            return {"active": False}
        t_end = rec["t_end"] if rec["paused"] else time.time()
        return {"active": True, "paused": rec["paused"], "name": rec["name"],
                "samples": rec["n"], "elapsed_s": round(t_end - rec["t0"], 1),
                "echo_frac": round(rec["echo_n"] / rec["n"], 3) if rec["n"] else 1.0,
                "alive_frac": round(rec["alive_n"] / rec["n"], 3) if rec["n"] else 1.0}


def _std(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def compute_features():
    ax, ay, az = buf["ax"], buf["ay"], buf["az"]
    gx, gy, gz = buf["gx"], buf["gy"], buf["gz"]

    axis_series = {"ax": ax, "ay": ay, "az": az}[MODEL1_AXIS]
    axis_ptp = max(axis_series) - min(axis_series)

    accel_mag = [math.sqrt(x * x + y * y + z * z) for x, y, z in zip(ax, ay, az)]
    accel_mag_std = _std(accel_mag)
    accel_mag_ptp = max(accel_mag) - min(accel_mag)

    # How fast the shin rotates: average and peak over the window (rad/s).
    gyro_mag = [math.sqrt(x * x + y * y + z * z) for x, y, z in zip(gx, gy, gz)]
    gyro_mag_mean = statistics.fmean(gyro_mag)
    gyro_mag_max = max(gyro_mag)

    # Closest the left leg came to the sensor in this window; ignore no-echo.
    echoes = [d for d in buf["dist"] if d >= 0]
    dist_min = min(echoes) if echoes else -1.0
    # Fraction of readings where the sensor answered at all, in range or not.
    alive = sum(1 for d in buf["dist"] if d != DIST_NO_RESPONSE)

    # Average accel per axis ~= gravity direction, i.e. which way the shin
    # points (upright, leaning forward, or tilted back while biking).
    return {
        "dist_now": buf["dist"][-1],
        "dist_min": dist_min,
        "dist_echo_frac": len(echoes) / len(buf["dist"]),
        "dist_alive_frac": alive / len(buf["dist"]),
        "axis_ptp": axis_ptp,
        "accel_mag_std": accel_mag_std,
        "accel_mag_ptp": accel_mag_ptp,
        "gyro_mag_mean": gyro_mag_mean,
        "gyro_mag_max": gyro_mag_max,
        "ax_mean": statistics.fmean(ax),
        "ay_mean": statistics.fmean(ay),
        "az_mean": statistics.fmean(az),
    }


def classify_model1(f):
    """Uses only the peak-to-peak acceleration on one axis."""
    ptp = f["axis_ptp"]
    if ptp < M1["t1"]:
        return STAND
    if ptp < M1["t2"]:
        return BIKE
    if ptp < M1["t3"]:
        return WALK
    return RUN


def classify_model2(f):
    """Uses features built from all 6 IMU channels (3 accel + 3 gyro)."""
    # 1. Overall motion intensity separates standing from everything else.
    if f["accel_mag_std"] < M2["T_std"]:
        return STAND

    # 2. Biking: the shin is angled far from upright, or the leg turns in a
    #    smooth circle (low gyro peak) with no footstrike.
    if _below(_tilt(f), M2["T_bike_tilt"]):
        return BIKE
    if _below(f["gyro_mag_max"], M2["T_gyro"]) and _below(f["accel_mag_ptp"], M2["T_bike_ptp"]):
        return BIKE

    # 3. Running: the shin leans more, or the footstrike hits harder.
    #    Model 1 can't see the lean, which is why it mixes up walk and run.
    if _above(_tilt(f), M2["T_run_tilt"]) or _above(f["accel_mag_ptp"], M2["T_run_ptp"]):
        return RUN
    return WALK


def imu_sample(ax: float, ay: float, az: float, gx: float, gy: float, gz: float, dist: float):
    global sample_count, _calib_count, calibrated

    if not calibrated:
        _calib_sums["gx"] += gx
        _calib_sums["gy"] += gy
        _calib_sums["gz"] += gz
        _calib_count += 1
        if _calib_count == 1:
            logger.info("Calibrating gyro bias -- hold the sensor still...")
        if _calib_count >= GYRO_CALIB_SAMPLES:
            gyro_bias["gx"] = _calib_sums["gx"] / _calib_count
            gyro_bias["gy"] = _calib_sums["gy"] / _calib_count
            gyro_bias["gz"] = _calib_sums["gz"] / _calib_count
            calibrated = True
            logger.info(
                f"Gyro calibrated: bias=({gyro_bias['gx']:.3f}, "
                f"{gyro_bias['gy']:.3f}, {gyro_bias['gz']:.3f}) rad/s"
            )
        return

    gx -= gyro_bias["gx"]
    gy -= gyro_bias["gy"]
    gz -= gyro_bias["gz"]

    buf["ax"].append(ax)
    buf["ay"].append(ay)
    buf["az"].append(az)
    buf["gx"].append(gx)
    buf["gy"].append(gy)
    buf["gz"].append(gz)
    buf["dist"].append(dist)
    sample_count += 1

    with rec_lock:
        if rec["writer"] is not None and not rec["paused"]:
            rec["writer"].writerow([rec["n"], f"{time.time():.3f}",
                                    f"{ax:.4f}", f"{ay:.4f}", f"{az:.4f}",
                                    f"{gx:.4f}", f"{gy:.4f}", f"{gz:.4f}", f"{dist:.1f}"])
            rec["n"] += 1
            if dist >= 0:
                rec["echo_n"] += 1
            if dist != DIST_NO_RESPONSE:
                rec["alive_n"] += 1
            if rec["n"] % 50 == 0:   # ~1 s of data at most is lost if the app dies
                rec["file"].flush()

    if len(buf["ax"]) < WINDOW_SIZE or sample_count % UPDATE_EVERY != 0:
        return

    features = compute_features()
    m1 = classify_model1(features)
    m2 = classify_model2(features)

    last_result["model1"] = m1
    last_result["model2"] = m2
    last_result["features"] = features
    last_result["t"] = time.time()

    # The net estimates how many steps are in the last 2 s. This runs every
    # 0.2 s, so each step is seen by 10 windows: add 1/10 of each estimate.
    still = (features["gyro_mag_max"] < STILL_GYRO_MAX
             and features["accel_mag_std"] < STILL_ACCEL_STD)
    use_dist = features["dist_alive_frac"] >= DIST_ALIVE_MIN
    net = NET_IMU_DIST if use_dist else NET_IMU
    in_window = 0.0 if still else max(0.0, net.steps_in_window(buf))
    new_steps = in_window * UPDATE_EVERY / WINDOW_SIZE
    cadence = in_window * 60.0 / WINDOW_SEC
    length = step_length_m(features, cadence)
    with step_lock:
        steps["total"] += new_steps
        steps["distance_m"] += new_steps * length
        steps["cadence_spm"] = cadence
        steps["step_len_m"] = length
        steps["model"] = "still" if still else ("imu_dist" if use_dist else "imu")

    if CALIBRATE_LOG:
        logger.info(
            f"steps={steps['total']:.1f} dist={steps['distance_m']:.2f}m step_len={steps['step_len_m']:.2f} "
            f"cadence={steps['cadence_spm']:.0f}/min net={steps['model']} "
            f"model1={LABELS[m1]:<8} model2={LABELS[m2]:<8} "
            f"axis_ptp={features['axis_ptp']:.2f} "
            f"accel_mag_std={features['accel_mag_std']:.2f} "
            f"accel_mag_ptp={features['accel_mag_ptp']:.2f} "
            f"gyro_mean={features['gyro_mag_mean']:.2f} "
            f"gyro_max={features['gyro_mag_max']:.2f} "
            f"ax_mean={features['ax_mean']:.2f} "
            f"ay_mean={features['ay_mean']:.2f} "
            f"az_mean={features['az_mean']:.2f} "
            f"dist_now={features['dist_now']:.1f} "
            f"dist_min={features['dist_min']:.1f} "
            f"echo_frac={features['dist_echo_frac']:.2f} "
            f"alive_frac={features['dist_alive_frac']:.2f}"
        )

    # Drive the LED matrix with Model 2's call since it's the more reliable
    # of the two -- lets you demo "what the wearable thinks" with no laptop.
    try:
        Bridge.call("set_activity", m2)
    except RuntimeError as e:
        logger.debug(f"set_activity call failed: {e}")


def steps_status():
    with step_lock:
        return {"total": round(steps["total"], 1), "distance_m": round(steps["distance_m"], 2),
                "cadence_spm": round(steps["cadence_spm"]), "step_len_m": round(steps["step_len_m"], 2),
                "model": steps["model"], "since": steps["since"]}


def steps_reset():
    """POST /api/steps/reset -- the page's Reset button."""
    with step_lock:
        steps.update(total=0.0, distance_m=0.0, since=time.time())
    logger.info("Step count reset")
    return {"ok": True}


def get_gait():
    """GET /api/gait -- the page polls this. Plain HTTP instead of the
    socket, because iPhone Safari wouldn't hold the socket open on the
    board's own hotspot (no internet)."""
    if last_result["model1"] is None:
        return {"ready": False, "rec": rec_status()}
    return {
        "ready": True,
        "rec": rec_status(),
        "steps": steps_status(),
        "model1": LABELS[last_result["model1"]],
        "model2": LABELS[last_result["model2"]],
        "features": {k: round(v, 2) for k, v in last_result["features"].items()},
        "age_s": round(time.time() - last_result["t"], 2),
    }


ui.expose_api("GET", "/api/gait", get_gait)
ui.expose_api("POST", "/api/record/start", record_start)
ui.expose_api("POST", "/api/record/pause", record_pause)
ui.expose_api("POST", "/api/record/stop", record_stop)
ui.expose_api("GET", "/api/recordings", list_recordings)
ui.expose_api("POST", "/api/steps/reset", steps_reset)

Bridge.provide("imu_sample", imu_sample)

logger.info(f"Step counter running: {WINDOW_SIZE} samples/window @ {SAMPLE_RATE_HZ}Hz")
App.run()
