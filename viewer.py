"""
viewer.py — P3: detection viewer.

Drains meta_to_view (FrameMeta) and det_to_view (Detection) concurrently,
joins them by frame_id, and renders each frame with bounding boxes and a
video-time timestamp at the video's native FPS.
"""
from __future__ import annotations

import queue
import signal
import time

import cv2

from ipc import Detection, FrameMeta, RingBuffer

_BLUR_KERNEL: tuple[int, int] = (51, 51)


def _pts_to_str(pts_ms: float) -> str:
    """Format millisecond PTS as HH:MM:SS.mmm."""
    total_ms = int(pts_ms)
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60_000) % 60
    h = total_ms // 3_600_000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _drain(q, dest: dict, done: list[bool]) -> None:
    """Non-blocking burst drain of one queue into dest, keyed by frame_id."""
    if done[0]:
        return
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if item is None:
            done[0] = True
            break
        dest[item.frame_id] = item


def run_viewer(shm_name, frame_shape, fps, blur_detections, meta_to_view, det_to_view, stop_event):
    """P3 entry point (forked by main.py)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ring = RingBuffer(shape=frame_shape, name=shm_name)
    print("[P3] Viewer started")

    pending_frames: dict[int, FrameMeta] = {}
    pending_dets: dict[int, Detection] = {}
    meta_done = [False]
    det_done = [False]

    frame_period = 1.0 / fps
    display_deadline: float | None = None
    next_id = 0

    try:
        while not stop_event.is_set():
            _drain(meta_to_view, pending_frames, meta_done)
            _drain(det_to_view, pending_dets, det_done)

            if next_id in pending_frames and next_id in pending_dets:
                meta = pending_frames.pop(next_id)
                det = pending_dets.pop(next_id)

                # Copy so drawing does not corrupt shared memory
                frame = ring.read_slot(meta.slot_id).copy()

                for x, y, w, h in det.boxes:
                    if blur_detections:
                        frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], _BLUR_KERNEL, 0)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                ts = _pts_to_str(meta.pts_ms)
                cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2, cv2.LINE_AA)

                if display_deadline is None:
                    display_deadline = time.monotonic()

                sleep_s = display_deadline - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)

                cv2.imshow("axon_vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

                display_deadline += frame_period
                next_id += 1

            elif meta_done[0] and det_done[0] and not pending_frames and not pending_dets:
                break
            else:
                time.sleep(0.001)

    finally:
        cv2.destroyAllWindows()
        ring.close()
        print("[P3] Viewer exiting")
