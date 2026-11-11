"""Live view from each camera, to find which index is the USB webcam.

    python preview.py          # start at camera 0
    python preview.py 1        # start at camera 1

Keys: n = next camera, q / Esc = quit.
"""

import sys

import cv2

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
while index < 5:
    cap = cv2.VideoCapture(index)
    ok, frame = cap.read()
    if not ok:
        print(f"camera {index}: nothing")
        cap.release()
        index += 1
        continue
    print(f"camera {index}: {frame.shape[1]}x{frame.shape[0]}  (n = next, q = quit)")
    while ok:
        cv2.putText(frame, f"camera {index}   n = next, q = quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("webcam preview", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            sys.exit()
        if key == ord("n"):
            break
        ok, frame = cap.read()
    cap.release()
    index += 1
print("no more cameras")
