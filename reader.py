"""
reader.py — P1: video reader.

Spawned by main.py.  Receives all IPC primitives as arguments.
Attaches to the already-allocated shared-memory ring, opens the video,
and feeds frames into the ring at full decode speed.

Backpressure: meta_to_det and meta_to_view are bounded queues; .put() blocks
when downstream is slow, preventing P1 from overwriting a ring slot still in use.

On EOF (or stop_event set by P3 on user quit):
  - Sends sentinel None to meta_to_det and meta_to_view.
  - Sets stop_event so P2/P3 abort any blocking wait immediately.
  - Closes its ring handle and exits.

main.py is responsible for joining all processes and calling ring.unlink().
"""
from __future__ import annotations

import signal
import sys
import time

import cv2

from ipc import FrameMeta, RingBuffer, N_SLOTS


def run_reader(
    video_path: str,
    shm_name: str,
    frame_shape: tuple[int, int, int],
    fps: float,
    meta_to_det,
    meta_to_view,
    stop_event,
) -> None:
    """P1 entry point (forked by main.py)."""

    # ------------------------------------------------------------------ #
    # 1. Attach to shared-memory ring (created by main.py)                 #
    # ------------------------------------------------------------------ #
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ring = RingBuffer(shape=frame_shape, name=shm_name)

    # ------------------------------------------------------------------ #
    # 2. Open video                                                         #
    # ------------------------------------------------------------------ #
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[P1] ERROR: cannot open '{video_path}'", file=sys.stderr)
        _signal_done(meta_to_det, meta_to_view, stop_event)
        ring.close()
        return

    print(f"[P1] Reader started — '{video_path}'  {fps:.3f} fps")

    # ------------------------------------------------------------------ #
    # 3. Read loop                                                          #
    # ------------------------------------------------------------------ #
    frame_id: int = 0
    frame_period: float = 1.0 / fps
    next_deadline: float = time.monotonic()

    def _push(frame) -> bool:
        """Write frame to ring slot and publish metadata to both queues.

        Blocks on backpressure (bounded queues).  Returns False if a queue
        times out, indicating a downstream process has died.
        """
        slot_id = frame_id % N_SLOTS
        ring.write_slot(slot_id, frame)
        pts_ms = frame_id * 1000.0 / fps
        meta = FrameMeta(frame_id=frame_id, slot_id=slot_id, pts_ms=pts_ms)
        try:
            meta_to_det.put(meta, timeout=5.0)
            meta_to_view.put(meta, timeout=5.0)
        except Exception as exc:
            print(f"[P1] queue stall (consumer dead?): {exc}", file=sys.stderr)
            return False
        return True

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break          # clean EOF
            if not _push(frame):
                break
            frame_id += 1
            next_deadline += frame_period
            sleep_s = next_deadline - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        cap.release()

    print(f"[P1] read loop done — {frame_id} frame(s) published")

    # ------------------------------------------------------------------ #
    # 4. Signal shutdown to P2 and P3                                      #
    # ------------------------------------------------------------------ #
    _signal_done(meta_to_det, meta_to_view, stop_event)
    ring.close()
    print("[P1] Reader exiting")


def _signal_done(meta_to_det, meta_to_view, stop_event) -> None:
    """Send EOF sentinels on both queues, then set the stop event.

    Each put() is guarded individually so a full or closed queue on one
    path does not prevent the sentinel from reaching the other consumer.
    """
    for q, name in ((meta_to_det, "meta_to_det"), (meta_to_view, "meta_to_view")):
        try:
            q.put(None, timeout=2.0)
        except Exception as exc:
            print(f"[P1] WARNING: could not send sentinel on {name}: {exc}", file=sys.stderr)
    stop_event.set()
