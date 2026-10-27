# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import math
import statistics
import time
from collections import deque

from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI

logger = Logger("gait-id")

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

# Which single accelerometer axis Model 1 is allowed to look at.
# Pick the axis that swings most on your mount; for an ankle mount the
# anteroposterior (direction of travel) or vertical axis usually has the
# clearest footstrike peaks. Check calibration logs and adjust if needed.
MODEL1_AXIS = "az"

STAND, WALK, RUN, BIKE = 0, 1, 2, 3
LABELS = {STAND: "standing", WALK: "walking", RUN: "running", BIKE: "biking"}

# --- Model 1 thresholds (peak-to-peak accel on MODEL1_AXIS, in m/s^2) ------
# Ordered by footstrike size, tuned on 20 s ankle recordings (2026-10-26):
# standing ~0.6, biking ~4 (no footstrike, chair or on your back),
# walking ~12, jogging ~17.5. Walking and jogging overlap a lot on this one
# axis -- that's the ambiguity Model 1 can't resolve with one number.
M1_TH_STAND = 2.0            # below -> standing
M1_TH_WALK = 6.5             # below -> biking
M1_TH_RUN = 14.0             # below -> walking, above -> running

# --- Model 2 thresholds (all 6 IMU channels) -------------------------------
# Tuned on 20 s ankle recordings (2026-10-26). With the sensor strapped on
# as worn, gravity runs along +x (the shin) when upright, and ay_mean tells
# how far the shin tilts: standing ~-1.7, walking ~0.6, jogging ~3.2,
# chair biking ~-3.9, biking on your back ~-9.5. ay_mean only means this if
# the sensor is worn the same way every time.
M2_TH_STAND_STD = 0.8        # accel magnitude std below this -> not moving
M2_TH_BIKE_TILT = -2.0       # ay_mean below -> shin tilted back -> biking
M2_TH_BIKE_GYRO = 4.2        # gyro_mag_max below (smooth pedal circle)...
M2_TH_BIKE_PTP = 17.0        # ...and accel_mag_ptp below (no footstrike) -> biking
M2_TH_RUN_TILT = 1.7         # ay_mean above -> shin leaning forward -> running
M2_TH_RUN_PTP = 22.0         # accel_mag_ptp above -> running impact

CALIBRATE_LOG = True         # print live feature values to help tune thresholds

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

buf = {k: deque(maxlen=WINDOW_SIZE) for k in ("ax", "ay", "az", "gx", "gy", "gz")}
sample_count = 0

last_result = {
    "model1": None,
    "model2": None,
    "features": {},
    "t": 0.0,
}


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

    # Average accel per axis ~= gravity direction, i.e. which way the shin
    # points (upright, leaning forward, or tilted back while biking).
    return {
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
    if ptp < M1_TH_STAND:
        return STAND
    if ptp < M1_TH_WALK:
        return BIKE
    if ptp < M1_TH_RUN:
        return WALK
    return RUN


def classify_model2(f):
    """Uses features built from all 6 IMU channels (3 accel + 3 gyro)."""
    # 1. Overall motion intensity separates standing from everything else.
    if f["accel_mag_std"] < M2_TH_STAND_STD:
        return STAND

    # 2. Biking: the shin tilts back (sitting or on your back), or -- if the
    #    strap is rotated and tilt can't be trusted -- the leg turns in a
    #    smooth circle (low gyro peak) with no footstrike.
    if f["ay_mean"] < M2_TH_BIKE_TILT:
        return BIKE
    if f["gyro_mag_max"] < M2_TH_BIKE_GYRO and f["accel_mag_ptp"] < M2_TH_BIKE_PTP:
        return BIKE

    # 3. Running: the shin leans forward more, or the footstrike hits harder.
    #    Model 1 can't see the lean, which is why it mixes up walk and run.
    if f["ay_mean"] > M2_TH_RUN_TILT or f["accel_mag_ptp"] > M2_TH_RUN_PTP:
        return RUN
    return WALK


def imu_sample(ax: float, ay: float, az: float, gx: float, gy: float, gz: float):
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
    sample_count += 1

    if len(buf["ax"]) < WINDOW_SIZE or sample_count % UPDATE_EVERY != 0:
        return

    features = compute_features()
    m1 = classify_model1(features)
    m2 = classify_model2(features)

    last_result["model1"] = m1
    last_result["model2"] = m2
    last_result["features"] = features
    last_result["t"] = time.time()

    if CALIBRATE_LOG:
        logger.info(
            f"model1={LABELS[m1]:<8} model2={LABELS[m2]:<8} "
            f"axis_ptp={features['axis_ptp']:.2f} "
            f"accel_mag_std={features['accel_mag_std']:.2f} "
            f"accel_mag_ptp={features['accel_mag_ptp']:.2f} "
            f"gyro_mean={features['gyro_mag_mean']:.2f} "
            f"gyro_max={features['gyro_mag_max']:.2f} "
            f"ax_mean={features['ax_mean']:.2f} "
            f"ay_mean={features['ay_mean']:.2f} "
            f"az_mean={features['az_mean']:.2f}"
        )

    # Drive the LED matrix with Model 2's call since it's the more reliable
    # of the two -- lets you demo "what the wearable thinks" with no laptop.
    try:
        Bridge.call("set_activity", m2)
    except RuntimeError as e:
        logger.debug(f"set_activity call failed: {e}")


def get_gait():
    """GET /api/gait -- the page polls this. Plain HTTP instead of the
    socket, because iPhone Safari wouldn't hold the socket open on the
    board's own hotspot (no internet)."""
    if last_result["model1"] is None:
        return {"ready": False}
    return {
        "ready": True,
        "model1": LABELS[last_result["model1"]],
        "model2": LABELS[last_result["model2"]],
        "features": {k: round(v, 2) for k, v in last_result["features"].items()},
        "age_s": round(time.time() - last_result["t"], 2),
    }


ui.expose_api("GET", "/api/gait", get_gait)

Bridge.provide("imu_sample", imu_sample)

logger.info(f"Gait ID running: {WINDOW_SIZE} samples/window @ {SAMPLE_RATE_HZ}Hz")
App.run()
