#!/usr/bin/env python3
"""Finds which step breaks macOS speech. Run it and note the LAST number you hear."""
import subprocess
import sys
import time

import cv2


def say(n, note):
    print(f"[{n}] {note} ...", end=" ", flush=True)
    try:
        subprocess.run(["say", str(n)], timeout=15, capture_output=True)
        print("spoke", flush=True)
    except subprocess.TimeoutExpired:
        print("HUNG  <-- speech broke at this step", flush=True)
        sys.exit(1)


backend = cv2.CAP_AVFOUNDATION if "--avf" in sys.argv else cv2.CAP_ANY
say(1, "before anything")
cap = cv2.VideoCapture(0, backend)
say(2, "camera opened")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ok, frame = cap.read()
say(3, f"first frame read ({frame.shape[1]}x{frame.shape[0]})" if ok else "first frame FAILED")
cv2.namedWindow("audio test", cv2.WINDOW_NORMAL)
t = time.time()
while time.time() - t < 3:
    ok, frame = cap.read()
    cv2.imshow("audio test", frame)
    cv2.waitKey(1)
say(4, "window showing video for 3 s")
import mediapipe  # noqa: E402,F401
say(5, "mediapipe imported")
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
say(6, "camera closed")
print("All steps spoke fine.")
