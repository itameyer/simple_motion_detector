"""
main.py — pipeline orchestrator.

Usage:
    python main.py --video path/to/clip.mp4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys

import cv2

from ipc import RingBuffer, N_SLOTS

QUEUE_MAXSIZE: int = N_SLOTS - 2
WATCHDOG_SECS: float = 15.0


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motion detection pipeline: reader → detector → viewer"
    )
    parser.add_argument(
        "--video", required=True, metavar="PATH",
        help="Path to the input video file",
    )
    parser.add_argument(
        "-b", "--blur-detections", action="store_true",
        help="Blur detected regions instead of drawing bounding boxes",
    )
    return parser.parse_args()


def _join_proc(proc: mp.Process, label: str) -> None:
    proc.join(timeout=WATCHDOG_SECS)
    if proc.is_alive():
        print(f"[main] WARNING: {label} still alive after {WATCHDOG_SECS}s — terminating",
              file=sys.stderr)
        proc.terminate()
        proc.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# Pipeline setup steps                                                         #
# --------------------------------------------------------------------------- #

def _probe_video(path: str) -> tuple[float, tuple[int, int, int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[main] ERROR: cannot open '{path}'", file=sys.stderr)
        sys.exit(1)

    fps: float = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
        print(f"[main] WARNING: FPS not in file header — defaulting to {fps}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("[main] ERROR: video has no readable frames", file=sys.stderr)
        sys.exit(1)

    h, w = frame.shape[:2]
    frame_shape: tuple[int, int, int] = (h, w, 3)
    print(f"[main] probed '{path}'  {w}×{h}  {fps:.3f} fps")
    return fps, frame_shape




def _allocate_ring(frame_shape: tuple[int, int, int]) -> RingBuffer:
    _BYTES_PER_MB = 1024 * 1024
    ring = RingBuffer(shape=frame_shape, create=True)
    slot_mb = ring.stride / _BYTES_PER_MB
    print(f"[main] ring '{ring.name}'  "
          f"{N_SLOTS} slots × {slot_mb:.1f} MB = {N_SLOTS * slot_mb:.0f} MB")
    return ring


def _spawn_processes(
    video_path: str,
    ring: RingBuffer,
    fps: float,
    frame_shape: tuple[int, int, int],
    blur_detections: bool,
    meta_to_det: mp.queues.Queue,
    meta_to_view: mp.queues.Queue,
    det_to_view: mp.queues.Queue,
    stop_event: mp.synchronize.Event,
) -> tuple[mp.Process, mp.Process, mp.Process]:
    from reader import run_reader
    from detector import run_detector
    from viewer import run_viewer

    p1 = mp.Process(
        target=run_reader,
        name="Reader",
        args=(video_path, ring.name, frame_shape, fps,
              meta_to_det, meta_to_view, stop_event),
        daemon=False,
    )
    p2 = mp.Process(
        target=run_detector,
        name="Detector",
        args=(ring.name, frame_shape, meta_to_det, det_to_view, stop_event),
        daemon=True,
    )
    p3 = mp.Process(
        target=run_viewer,
        name="Viewer",
        args=(ring.name, frame_shape, fps, blur_detections, meta_to_view, det_to_view, stop_event),
        daemon=True,
    )

    for proc in (p1, p2, p3):
        proc.start()
    print("[main] Reader, Detector, and Viewer spawned")
    return p1, p2, p3


def _await_shutdown(
    p1: mp.Process,
    p2: mp.Process,
    p3: mp.Process,
    ring: RingBuffer,
) -> None:
    _join_proc(p1, "Reader")
    _join_proc(p2, "Detector")
    _join_proc(p3, "Viewer")

    ring.close()
    ring.unlink()
    print("[main] shutdown complete")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    try:
        args = _parse_args()

        fps, frame_shape = _probe_video(args.video)
        ring = _allocate_ring(frame_shape)

        meta_to_det  = mp.Queue(maxsize=QUEUE_MAXSIZE)
        meta_to_view = mp.Queue(maxsize=QUEUE_MAXSIZE)
        det_to_view  = mp.Queue(maxsize=QUEUE_MAXSIZE)
        stop_event   = mp.Event()

        p1, p2, p3 = _spawn_processes(
            args.video, ring, fps, frame_shape, args.blur_detections,
            meta_to_det, meta_to_view, det_to_view, stop_event,
        )
        p1.join()
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt received — shutting down")
        stop_event.set()
    finally:
        _await_shutdown(p1, p2, p3, ring)


if __name__ == "__main__":
    main()
