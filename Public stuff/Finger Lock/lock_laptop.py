"""Run the finger lock on the laptop with the USB webcam.

    python lock_laptop.py                          # camera 0, no motors
    python lock_laptop.py --board 10.247.137.116   # also opens the real lid

With --board, the laptop does all the camera + CNN + lock work and just
tells the UNO Q (running the finger-lock app) to open / close the lid over
WiFi, using the same page buttons the phone uses.

Keys (click the window first):
    space    button 1 - start entering the code
    c        button 2 - close / lock the box
    q / Esc  quit

Uses the numpy CNN (finger_net.py) and the lock logic (lock_policy.py), the
same code that runs on the UNO Q.
"""

import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from finger_net import FingerNet
from lock_policy import LockPolicy, load_code

HERE = Path(__file__).parent
GREEN, RED, YELLOW, WHITE, GRAY = (80, 200, 80), (60, 60, 230), (0, 210, 255), (255, 255, 255), (120, 120, 120)
STATE_COLOR = {"locked": GRAY, "entering": YELLOW, "clear": YELLOW,
               "unlocked": GREEN, "denied": RED, "lockout": RED}


def draw_panel(probs, policy, fps):
    """Side panel: lock state, entered digits, and the CNN's 6 probabilities."""
    p = np.full((480, 360, 3), 30, np.uint8)
    color = STATE_COLOR[policy.state]
    cv2.putText(p, policy.state.upper(), (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(p, policy.message, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    # LEDs + code slots
    cv2.circle(p, (330, 35), 14, YELLOW if policy.leds_on() else (60, 60, 60), -1)
    for i in range(3):
        x = 15 + i * 70
        cv2.rectangle(p, (x, 100), (x + 55, 165), WHITE, 2)
        if i < len(policy.entered):
            cv2.putText(p, str(policy.entered[i]), (x + 14, 152),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 3)
    if policy.candidate:
        cv2.rectangle(p, (15, 175), (15 + int(195 * policy.hold_progress), 185), YELLOW, -1)

    # probability bars
    cv2.putText(p, "CNN says (probability):", (15, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    best = int(np.argmax(probs))
    for k in range(6):
        y = 245 + k * 34
        cv2.putText(p, f"{k}", (15, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.rectangle(p, (45, y), (45 + int(250 * probs[k]), y + 24),
                      GREEN if k == best else (180, 130, 60), -1)
        cv2.putText(p, f"{probs[k]:.0%}", (300, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(p, f"{fps:.0f} fps   space = button   c = close   q = quit",
                (15, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
    return p


def tell_board(board, action):
    """POST /api/latch/open or /close to the UNO Q, in the background so the
    video doesn't pause while it waits."""
    def send():
        try:
            req = urllib.request.Request(f"http://{board}:7000/api/latch/{action}",
                                         method="POST")
            urllib.request.urlopen(req, timeout=3)
            print(f"board: lid {action}")
        except Exception as e:
            print(f"board: couldn't {action} the lid ({e})")
    if board:
        threading.Thread(target=send, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--board", help="UNO Q IP address, to drive the lid motors")
    args = ap.parse_args()

    roi = tuple(json.loads((HERE / "roi.json").read_text())[k] for k in "xywh")
    net = FingerNet(HERE / "models" / "finger_cnn.npz")
    policy = LockPolicy(load_code())
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps, last = 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break
        now = time.time()
        probs = net.predict(FingerNet.prepare(frame, roi))
        event = policy.update(probs, now)
        if event:
            print(f"{event}: entered {policy.entered}  ({policy.message})")
        if event == "unlock":
            tell_board(args.board, "open")

        fps = 0.9 * fps + 0.1 / max(1e-3, now - last)
        last = now
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), STATE_COLOR[policy.state], 3)
        cv2.imshow("finger lock", np.hstack([frame, draw_panel(probs, policy, fps)]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            policy.start(now)
        elif key == ord("c"):
            if policy.state == "unlocked":
                tell_board(args.board, "close")
            policy.close(now)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
