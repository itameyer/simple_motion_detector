"""
ipc.py — shared data contracts and shared-memory ring buffer helper.

All three processes import this module; keep it import-side-effect free.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np

N_SLOTS: int = 16  # ring depth — 16 × ~6 MB ≈ 96 MB at 1080p


@dataclass(frozen=True)
class FrameMeta:
    frame_id: int   # monotonic counter assigned by P1; join key for detections
    slot_id: int    # which ring slot holds this frame's bytes
    pts_ms: float   # video presentation timestamp in milliseconds


@dataclass(frozen=True)
class Detection:
    frame_id: int                            # matches FrameMeta.frame_id
    boxes: list[tuple[int, int, int, int]]   # (x, y, w, h) in pixel coords


class RingBuffer:
    """Zero-copy shared-memory ring — one slot per BGR frame.

    Creator (P1):  RingBuffer(shape, create=True)  — allocates the block.
    Consumers:     RingBuffer(shape, name=shm_name) — attaches by name.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        *,
        name: str | None = None,
        create: bool = False,
    ) -> None:
        h, w, c = shape
        self.shape = shape
        self.stride: int = h * w * c          # bytes per slot
        total: int = N_SLOTS * self.stride
        self._shm = SharedMemory(name=name, create=create, size=total)

    @property
    def name(self) -> str:
        return self._shm.name

    def write_slot(self, slot_id: int, frame: np.ndarray) -> None:
        view = np.ndarray(
            self.shape, dtype=np.uint8,
            buffer=self._shm.buf, offset=slot_id * self.stride,
        )
        np.copyto(view, frame)

    def read_slot(self, slot_id: int) -> np.ndarray:
        """Return a zero-copy view of slot_id.  Valid until P1 next writes that slot."""
        return np.ndarray(
            self.shape, dtype=np.uint8,
            buffer=self._shm.buf, offset=slot_id * self.stride,
        )

    def close(self) -> None:
        """Release handle — call in every process that opened this buffer."""
        self._shm.close()

    def unlink(self) -> None:
        """Destroy the underlying segment — call once, in P1, after children exit."""
        try:
            self._shm.unlink()
        except Exception:
            pass  # no-op on Windows; harmless if already gone on POSIX
