"""
detector.py — P2: motion detector.

Algorithm source (consecutive-frame differencing):
  The detection pipeline below is adapted from the reference code provided
  for this exercise and the pyimagesearch article:
  https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

Pipeline per frame:
  1. Convert BGR → grayscale.
  2. Gaussian blur to suppress sensor noise.
  3. Absolute difference against the previous frame's blurred gray.
  4. Binary threshold → dilate to fill gaps.
  5. Find external contours; filter by minimum area.
  6. Emit one bounding rect (x, y, w, h) per surviving contour.

P2 always emits exactly one Detection message per FrameMeta consumed — even
when no motion is found (boxes=[]) — so P3's join logic never stalls waiting
for a detection that will never arrive.
"""
from __future__ import annotations

import queue
import signal

import cv2
import imutils
import numpy as np

from ipc import Detection, FrameMeta, RingBuffer

# --- Tuning constants (mirror pyimagesearch defaults) ---
_BLUR_KERNEL: tuple[int, int] = (21, 21)
_THRESH_VALUE: int = 25
_DILATE_ITERS: int = 2
_MIN_CONTOUR_AREA: int = 500   # pixels² — filters out sensor noise


# --------------------------------------------------------------------------- #
# Detection pipeline                                                           #
# --------------------------------------------------------------------------- #

def _to_gray_blurred(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to a blurred grayscale image ready for differencing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, _BLUR_KERNEL, 0)


def _detect_boxes(
    current_gray: np.ndarray,
    prev_gray: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Diff current vs previous gray frame; return bounding rects of motion regions."""
    diff = cv2.absdiff(current_gray, prev_gray)
    thresh = cv2.threshold(diff, _THRESH_VALUE, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=_DILATE_ITERS)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    boxes: list[tuple[int, int, int, int]] = []
    for c in cnts:
        if cv2.contourArea(c) < _MIN_CONTOUR_AREA:
            continue
        boxes.append(cv2.boundingRect(c))   # (x, y, w, h)

    return boxes


# --------------------------------------------------------------------------- #
# Process entry point                                                          #
# --------------------------------------------------------------------------- #

def run_detector(
    shm_name: str,
    frame_shape: tuple[int, int, int],
    meta_to_det,
    det_to_view,
    stop_event
) -> None:
    """P2 entry point — forked by main.py.

    Attaches to the shared-memory ring, then loops:
      - Pull FrameMeta from meta_to_det (timeout-based so stop_event is checked
        periodically in case P1 dies without sending its sentinel).
      - Read the frame from the ring slot (zero-copy view — safe because the
        bounded queue guarantees P1 cannot overwrite this slot while P2 holds it).
      - Run the detection pipeline.
      - Push one Detection to det_to_view, matching frame_id.

    On sentinel (None) or stop_event: send None downstream so P3 also unblocks,
    close the ring handle, and exit.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ring = RingBuffer(shape=frame_shape, name=shm_name)
    print("[P2] Detector started")

    prev_gray: np.ndarray | None = None

    try:
        while not stop_event.is_set():
            try:
                meta: FrameMeta | None = meta_to_det.get(timeout=0.1)
            except queue.Empty:
                continue

            if meta is None:
                break

            frame = ring.read_slot(meta.slot_id)
            current_gray = _to_gray_blurred(frame)

            if prev_gray is None:
                boxes: list[tuple[int, int, int, int]] = []
            else:
                boxes = _detect_boxes(current_gray, prev_gray)

            prev_gray = current_gray
            det_to_view.put(Detection(frame_id=meta.frame_id, boxes=boxes))

    finally:
        det_to_view.put(None)   # forward sentinel so P3 unblocks
        ring.close()
        print("[P2] Detector exiting")
