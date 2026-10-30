"""BADPIX-REPAIR-2026-10: lists of repaired pixels (no photometry imports - read by the alignment and the
aperture photometry). The repair itself is ``badpix_repair``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: FITS extension of the calibrated frame listing the repaired pixels (x, y native 0-based; kind S/P).
REPAIR_HDU = "VYBPFIX"


def read_repair_table(hdul: Any) -> np.ndarray:
    """(n, 3) array x, y, is_star_model of the repaired pixels of a calibrated frame (empty when none)."""
    if REPAIR_HDU not in hdul:
        return np.zeros((0, 3))
    t = hdul[REPAIR_HDU].data
    if t is None or len(t) == 0:
        return np.zeros((0, 3))
    kind = np.char.decode(np.asarray(t["KIND"]).astype("S1"), "ascii") if np.asarray(t["KIND"]).dtype.kind == "S" else np.asarray(t["KIND"])
    return np.c_[np.asarray(t["X"], float), np.asarray(t["Y"], float), (kind == "P").astype(float)]



#: Sidecar next to the aligned frames: {aligned file name: [[x_ref, y_ref, is_star_model], ...]}.
REPAIRED_SIDECAR = "badpix_repaired.json"


def repaired_xy_on_reference(bp_native: Any, hdr: Any) -> list[list[float]]:
    """Repaired native pixels mapped to the reference grid through the frame's VY_AM transform
    (frame -> reference, ``frame_footprint.stamp_alignment_transform``)."""
    a = np.zeros((0, 3)) if bp_native is None else np.asarray(bp_native, dtype=float).reshape(-1, 3)
    if len(a) == 0:
        return []
    m = np.array([float(hdr[f"VY_AM{i}"]) for i in range(9)]).reshape(3, 3)
    p = m @ np.c_[a[:, 0], a[:, 1], np.ones(len(a))].T
    x, y = p[0] / p[2], p[1] / p[2]
    return [[round(float(u), 2), round(float(v), 2), int(k)] for u, v, k in zip(x, y, a[:, 2], strict=True)]


def write_repaired_sidecar(aligned_dir: Any, frames: dict[str, list[list[float]]]) -> None:
    """Merge this run's frames into ``REPAIRED_SIDECAR`` of ``aligned_dir`` (frames re-aligned now replace
    their old entry; a frame without repaired pixels is stored with an empty list)."""
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    p = Path(aligned_dir) / REPAIRED_SIDECAR
    old = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    old.update(frames)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(old, separators=(",", ":")), encoding="utf-8")


def load_repaired_sidecar(aligned_dir: Any) -> dict[str, np.ndarray]:
    """Frame stem -> (n, 3) array of repaired pixels on the reference grid (empty dict without a sidecar)."""
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    p = Path(aligned_dir) / REPAIRED_SIDECAR
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {Path(k).stem: np.asarray(v, dtype=float).reshape(-1, 3) for k, v in raw.items()}


def repaired_in_aperture(bp_ref: np.ndarray | None, x: float, y: float, r: float) -> int:
    """Repaired pixels whose pixel square touches the circle of radius ``r`` around (x, y)."""
    if bp_ref is None or len(bp_ref) == 0:
        return 0
    return int(np.count_nonzero(np.hypot(bp_ref[:, 0] - x, bp_ref[:, 1] - y) <= float(r) + math.sqrt(0.5)))
