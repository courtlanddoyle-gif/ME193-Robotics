"""Collect finger-count photos from the USB webcam.

    python collect.py --person courtland            # use camera 1 (USB webcam)
    python collect.py --person alex --cam 2
    python collect.py --list                        # preview each camera index

Keys (click the video window first):
    r        drag a box over the red square, then Enter (saved to roi.json)
    0-5      choose which finger count you're recording
    space    start / stop burst capture (about 4 photos a second)
    s        save a single photo
    q / Esc  quit

Photos are the cropped red-square area, saved as 128x128 PNGs in
data/<count>/<person>_<time>.png.  The person's name is kept so we can test
the model on hands it never trained on.
"""

import argparse
import json
import time
from pathlib import Path

import cv2

HERE = Path(__file__).parent
DATA = HERE / "data"
ROI_FILE = HERE / "roi.json"
SAVE_SIZE = 128
BURST_INTERVAL = 0.25  # seconds between burst photos


def open_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"Camera {index} didn't open. Try --list.")
    return cap


def list_cameras():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if not ok:
            print(f"camera {i}: nothing")
            continue
        print(f"camera {i}: {frame.shape[1]}x{frame.shape[0]} "
              "- showing it, press any key for the next one")
        cv2.imshow(f"camera {i}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def load_roi():
    if ROI_FILE.exists():
        r = json.loads(ROI_FILE.read_text())
        return r["x"], r["y"], r["w"], r["h"]
    return None


def crop(frame, roi):
    """Cut out the box and resize it to a square (a wide box gets squeezed
    a little, the same way every time, so the model doesn't care)."""
    x, y, w, h = roi
    H, W = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    return cv2.resize(frame[y:min(H, y + h), x:min(W, x + w)],
                      (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)


def counts():
    return {k: len(list((DATA / str(k)).glob("*.png"))) for k in range(6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="courtland")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        list_cameras()
        return

    for k in range(6):
        (DATA / str(k)).mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.cam)
    roi = load_roi()
    label = 1
    burst = False
    last_shot = 0.0
    n = counts()
    flash = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break
        view = frame.copy()

        if roi:
            x, y, w, h = roi
            color = (0, 0, 255) if burst else (0, 255, 0)
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
            now = time.time()
            if burst and now - last_shot >= BURST_INTERVAL:
                save(frame, roi, label, args.person)
                n[label] += 1
                last_shot = now
                flash = 2
            cv2.imshow("crop (what the model sees)", crop(frame, roi))
        else:
            cv2.putText(view, "press r and drag a box over the red square",
                        (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        status = (f"{args.person}  recording: {label} fingers  "
                  f"{'BURST ON' if burst else 'burst off'}")
        cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if burst else (255, 255, 255), 2)
        cv2.putText(view, "  ".join(f"{k}:{n[k]}" for k in range(6)), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if flash:
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1),
                          (255, 255, 255), 6)
            flash -= 1
        cv2.imshow("finger collector", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if ord("0") <= key <= ord("5"):
            label = key - ord("0")
            burst = False
        elif key == ord(" ") and roi:
            burst = not burst
        elif key == ord("s") and roi:
            save(frame, roi, label, args.person)
            n[label] += 1
            flash = 2
        elif key == ord("r"):
            box = cv2.selectROI("finger collector", frame, showCrosshair=False)
            if box[2] > 10 and box[3] > 10:
                roi = tuple(map(int, box))
                ROI_FILE.write_text(json.dumps(dict(zip("xywh", roi))))
                print("saved roi.json:", roi)

    cap.release()
    cv2.destroyAllWindows()
    print("photos per class:", counts())


def save(frame, roi, label, person):
    name = f"{person}_{time.strftime('%Y%m%d-%H%M%S')}_{int(time.time()*1000) % 1000:03d}.png"
    cv2.imwrite(str(DATA / str(label) / name), crop(frame, roi))


if __name__ == "__main__":
    main()
