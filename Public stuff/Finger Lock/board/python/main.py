"""Finger Lock - runs on the UNO Q's Linux side.

Every loop: grab a webcam frame -> crop the red square -> CNN (numpy, the
same finger_net.py tested on the laptop) -> lock policy (lock_policy.py)
-> LED matrix + latch motor (sketch.ino does the hardware).

Web page at http://<board IP>:7000 shows the camera, the CNN's 6
probabilities and the lock state, and has the two buttons (until the real
buttons on D4 / D7 are wired; those work too).
"""

import base64
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge, Frame, Logger

from finger_net import FingerNet
from lock_policy import LockPolicy, load_code

logger = Logger("finger-lock")
HERE = Path(__file__).resolve().parent

CAMERA = 0              # not used directly any more; see open_camera()
LATCH_SPEED = 150       # 0-255 PWM; lower = gentler on the board's 5 V
# How long both lid motors run to open / close. They only turn ~1/4 turn
# into a hard stop, so this should be just long enough to get there; any
# longer and they sit stalled, drawing the most current. Tuned from the web
# page (longer / shorter) and saved to data/latch.json.
MOVE_MS_FILE = Path("/app/data/latch.json")
try:
    MOVE_MS = json.loads(MOVE_MS_FILE.read_text())["move_ms"]
except Exception:
    MOVE_MS = 350

r = json.loads((HERE / "roi.json").read_text())
ROI = (r["x"], r["y"], r["w"], r["h"])
net = FingerNet(HERE / "models" / "finger_cnn.npz")
policy = LockPolicy(load_code())
lock = threading.Lock()   # policy is touched by the loop and the web page

ui = WebUI()


def open_camera():
    """Find the webcam. It's usually /dev/video2, but if it drops out (the
    motor's start-up current once browned it out) it comes back as a new
    number, so try each one."""
    # The number changes between boots (video0 or video2), and the board's
    # own video codec also shows up as /dev/videoN, so skip anything named
    # "Venus" and try the rest.
    def name(n):
        try:
            return Path(f"/sys/class/video4linux/video{n}/name").read_text()
        except OSError:
            return ""
    for i in [n for n in range(10) if "venus" not in name(n).lower()]:
        c = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, _ = c.read()
            if ok:
                logger.info(f"camera opened at /dev/video{i}")
                return c
        c.release()
    return None


cap = open_camera()

latest = {"frame": None, "probs": [1, 0, 0, 0, 0, 0], "fps": 0.0}
latch_state = {"position": "closed", "moving": False}
last_inputs = 0


# ------------------------------------------------------------ LED matrix ----
DIGITS = {  # 3 wide x 5 tall
    1: ["010", "110", "010", "010", "111"],
    2: ["111", "001", "111", "100", "111"],
    3: ["111", "001", "111", "001", "111"],
    4: ["101", "101", "111", "001", "001"],
    5: ["111", "100", "111", "001", "111"],
}
CHECK = ["0000000000001", "0000000000010", "0000000000100", "1000000001000",
         "0100000010000", "0010000100000", "0001001000000", "0000110000000"]
CROSS = ["".join("1" if c in (2 + i, 9 - i) else "0" for c in range(13)) for i in range(8)]


def glyph(rows, bright=7):
    return np.array([[bright if c == "1" else 0 for c in row] for row in rows], np.uint8)


def led_frame():
    f = np.zeros((8, 13), np.uint8)
    s = policy.state
    if s == "unlocked":
        return glyph(CHECK)
    if s in ("denied", "lockout"):
        return glyph(CROSS, 7 if s == "denied" else 2)
    if s in ("entering", "clear"):
        for i in range(3):                      # three slots at columns 0, 5, 10
            col = i * 5
            if i < len(policy.entered):
                f[1:6, col:col + 3] = glyph(DIGITS[policy.entered[i]])
            f[7, col:col + 3] = 2               # slot marker
        slot = len(policy.entered)
        if s == "entering" and slot < 3:
            f[7, slot * 5:slot * 5 + 3] = 7     # current slot: bright
            if policy.candidate:                # holding a digit: preview it, dim
                f[1:6, slot * 5:slot * 5 + 3] = glyph(DIGITS[policy.candidate],
                                                     1 + int(5 * policy.hold_progress))
    return f


# ------------------------------------------------------------ latch ----------
# Only the main loop talks to the sketch: two threads calling Bridge at once
# (web page + loop) froze the app, so everyone else just queues a request.
latch_requests = []


def open_latch():
    latch_requests.append(1)


def close_latch():
    latch_requests.append(-1)


def run_latch_requests():
    while latch_requests:
        d = latch_requests.pop(0)
        latch_state["moving"] = True
        latch_state["position"] = "open" if d > 0 else "closed"
        Bridge.call("latch", d, LATCH_SPEED, MOVE_MS)


def latch_done(direction, hit_switch):
    latch_state["moving"] = False
    logger.info(f"latch {'opened' if direction > 0 else 'closed'}"
                f" ({'limit switch' if hit_switch else 'time limit'})")


def press_start():
    with lock:
        policy.start(time.time())
    logger.info("button 1: start")


def press_close():
    with lock:
        was_open = policy.state == "unlocked"
        policy.close(time.time())
    if was_open or latch_state["position"] == "open":
        close_latch()
    logger.info("button 2: close")


def inputs(bits):
    """Sketch tells us when a button / limit switch changes (bit set = pressed)."""
    global last_inputs
    newly = bits & ~last_inputs
    last_inputs = bits
    if newly & 1:
        press_start()
    if newly & 2:
        press_close()


# ------------------------------------------------------------ main loop ------
last_t = time.time()


def loop():
    global last_t, cap
    ok, frame = cap.read() if cap is not None else (False, None)
    if not ok:
        logger.warning("camera lost - reconnecting")
        if cap is not None:
            cap.release()
        time.sleep(0.5)
        cap = open_camera()
        run_latch_requests()
        return
    probs = net.predict(net.prepare(frame, ROI))
    now = time.time()
    with lock:
        event = policy.update(probs, now)
    if event:
        logger.info(f"{event}: entered {policy.entered}")
    if event == "unlock":
        open_latch()

    latest["frame"], latest["probs"] = frame, probs.tolist()
    latest["fps"] = 0.9 * latest["fps"] + 0.1 / max(1e-3, now - last_t)
    last_t = now
    run_latch_requests()
    Bridge.call("draw", Frame(led_frame()).to_board_bytes())
    if now - latest.get("logged", 0) > 30:     # heartbeat, to spot freezes
        latest["logged"] = now
        logger.info(f"running {latest['fps']:.1f} fps, state {policy.state}")


# ------------------------------------------------------------ web page -------
def get_state():
    with lock:
        return {
            "state": policy.state, "message": policy.message,
            "entered": list(policy.entered), "candidate": policy.candidate,
            "hold": round(policy.hold_progress, 2), "leds": policy.leds_on(),
            "tries_left": 3 - policy.wrong_tries,
            "probs": [round(p, 3) for p in latest["probs"]],
            "fps": round(latest["fps"], 1), "latch": dict(latch_state), "move_ms": MOVE_MS,
        }


def get_image():
    f = latest["frame"]
    if f is None:
        return {"jpg": None}
    f = f.copy()
    x, y, w, h = ROI
    cv2.rectangle(f, (x, y), (x + w, y + h), (0, 255, 0), 2)
    ok, buf = cv2.imencode(".jpg", cv2.resize(f, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return {"jpg": base64.b64encode(buf).decode()}


ui.expose_api("GET", "/api/state", get_state)
ui.expose_api("GET", "/api/image", get_image)
ui.expose_api("POST", "/api/button/start", lambda: (press_start(), {"ok": True})[1])
ui.expose_api("POST", "/api/button/close", lambda: (press_close(), {"ok": True})[1])
ui.expose_api("POST", "/api/latch/open", lambda: (open_latch(), {"ok": True})[1])
ui.expose_api("POST", "/api/latch/close", lambda: (close_latch(), {"ok": True})[1])


def change_move_ms(delta):
    global MOVE_MS
    MOVE_MS = max(100, min(1500, MOVE_MS + delta))
    MOVE_MS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MOVE_MS_FILE.write_text(json.dumps({"move_ms": MOVE_MS}))
    logger.info(f"latch run time now {MOVE_MS} ms")
    return {"move_ms": MOVE_MS}


ui.expose_api("POST", "/api/latch/longer", lambda: change_move_ms(50))
ui.expose_api("POST", "/api/latch/shorter", lambda: change_move_ms(-50))

Bridge.provide("inputs", inputs)
Bridge.provide("latch_done", latch_done)

logger.info(f"Finger lock running, code has {len(policy.code)} digits, camera {CAMERA}")
App.run(user_loop=loop)
