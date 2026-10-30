"""Field maps of a setup: the whole MASTERSTAR with the targets and their comparison stars.

FIELD-MAP-2026-10 (Milan 2026-10-21): JPEG (quality 90; -70 % against PNG, no visible difference),
a fine RA / Dec grid, a scale bar and the N / E orientation, all from the MASTERSTAR WCS. Grid step
and scale bar follow the field size, so every telescope / camera gets a readable map. The image keeps
the chip orientation (the arrows show where north and east are).

The background (stretch, image, grid, scale, compass) is drawn once; each map adds its markers and
title, is saved and the markers are removed again (OUTPUT-REVIEW-2026-10 step 7).

Files: ``photometry/field_map.jpg`` (all targets) and ``photometry/lightcurves/field_map_<id>.jpg``.
Drafts made before keep ``.png``; :func:`field_map_path` finds either.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

FIELD_MAP_JPEG_QUALITY = 90
FIELD_MAP_NAME = "field_map.jpg"

#: grid steps: Dec in arcmin, RA in seconds of time; scale bar lengths in arcmin
_DEC_STEPS_ARCMIN = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0)
_RA_STEPS_SEC = (1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1200.0, 1800.0, 3600.0,
                 7200.0, 14400.0)
_BAR_ARCMIN = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0, 300.0)
_GRID_LINES = 6  # about this many lines across the field in each coordinate


def target_field_map_path(lc_dir: Path | str, cid: str) -> Path:
    """Where the per-target field map is written (``lightcurves/field_map_<id>.jpg``)."""
    return Path(lc_dir) / f"field_map_{cid}.jpg"


def field_map_path(directory: Path | str, cid: str | None = None) -> Path | None:
    """The field map of a setup (``cid`` None) or of a target; ``.jpg`` or the ``.png`` of older drafts."""
    d = Path(directory)
    stem = "field_map" if cid is None else f"field_map_{cid}"
    for ext in (".jpg", ".png"):
        p = d / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def grid_steps(fov_arcmin: float, dec_deg: float) -> tuple[float, float]:
    """(Dec step arcmin, RA step seconds of time) giving about ``_GRID_LINES`` lines over the field."""
    target = max(float(fov_arcmin), 1e-3) / _GRID_LINES
    dec_step = next((s for s in _DEC_STEPS_ARCMIN if s >= target), _DEC_STEPS_ARCMIN[-1])
    cos_d = max(math.cos(math.radians(float(dec_deg))), 0.05)
    ra_target_sec = target / cos_d * 4.0  # 1 arcmin of RA = 4 s of time
    ra_step = next((s for s in _RA_STEPS_SEC if s >= ra_target_sec), _RA_STEPS_SEC[-1])
    return dec_step, ra_step


def scale_bar_arcmin(width_arcmin: float) -> float:
    """Scale bar length: the listed value closest to a fifth of the image width."""
    want = float(width_arcmin) / 5.0
    return min(_BAR_ARCMIN, key=lambda v: abs(math.log(v / max(want, 1e-6))))


def _bar_label(arcmin: float) -> str:
    if arcmin >= 60.0:
        return f"{arcmin / 60.0:g}\u00b0"
    if arcmin < 1.0:
        return f"{arcmin * 60.0:g}\u2033"
    return f"{arcmin:g}\u2032"


def _celestial_wcs(header: Any) -> Any | None:
    if header is None:
        return None
    try:
        from astropy.wcs import WCS, FITSFixedWarning  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(header).celestial
        return w if w.has_celestial and w.naxis == 2 else None
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[FIELD MAP] no usable WCS (map without grid / scale / orientation): %s", exc)
        return None


class FieldMapCanvas:
    """The MASTERSTAR background of the field maps, drawn once."""

    def __init__(self, data: np.ndarray, header: Any = None, *,
                 percentile_lo: float = 5.0, percentile_hi: float = 99.5) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = np.asarray(data, dtype=np.float64)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            raise ValueError("MASTERSTAR has no finite pixels")
        vmin = float(np.percentile(finite, percentile_lo))
        vmax = float(np.percentile(finite, percentile_hi))
        self._plt = plt
        self.wcs = _celestial_wcs(header)
        self.fig = plt.figure(figsize=(18, 12))
        if self.wcs is not None:
            self.ax = self.fig.add_subplot(projection=self.wcs)
        else:
            self.ax = self.fig.add_subplot()
        self.ax.imshow(data, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax, aspect="equal")
        # the view is the image: markers never widen it
        self.ax.set_xlim(-0.5, data.shape[1] - 0.5)
        self.ax.set_ylim(-0.5, data.shape[0] - 0.5)
        self.ax.set_autoscale_on(False)
        self.subtitle = ""
        if self.wcs is not None:
            try:
                self._decorate(data.shape)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("[FIELD MAP] grid / scale / orientation not drawn: %s", exc)
        else:
            self.ax.axis("off")

    def _decorate(self, shape: tuple[int, ...]) -> None:
        import astropy.units as u  # noqa: PLC0415
        from astropy.wcs.utils import proj_plane_pixel_scales  # noqa: PLC0415

        ax, w = self.ax, self.wcs
        ny, nx = int(shape[0]), int(shape[1])
        scale = float(np.mean(proj_plane_pixel_scales(w))) * 3600.0  # arcsec / px
        w_am, h_am = nx * scale / 60.0, ny * scale / 60.0
        center = w.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
        dec_step, ra_step = grid_steps(max(w_am, h_am), float(center.dec.deg))
        ra, de = ax.coords[0], ax.coords[1]
        ra.set_ticks(spacing=(ra_step * 15.0 / 3600.0) * u.deg)
        de.set_ticks(spacing=dec_step * u.arcmin)
        ra.set_major_formatter("hh:mm" if ra_step >= 60.0 and ra_step % 60.0 == 0 else "hh:mm:ss")
        de.set_major_formatter("dd" if dec_step >= 60.0 and dec_step % 60.0 == 0 else
                               ("dd:mm" if dec_step >= 1.0 and dec_step % 1.0 == 0 else "dd:mm:ss"))
        ra.set_axislabel("RA (J2000)")
        de.set_axislabel("Dec (J2000)")
        ax.coords.grid(color="#1d4ed8", alpha=0.6, linestyle=(0, (4, 4)), linewidth=0.7)

        # scale bar, lower left
        bar_am = scale_bar_arcmin(w_am)
        bar_px = bar_am * 60.0 / scale
        x0, y0 = 0.04 * nx, 0.05 * ny
        ax.plot([x0, x0 + bar_px], [y0, y0], color="black", lw=3)
        ax.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=1.2)
        ax.text(x0 + bar_px / 2.0, y0 + 0.015 * ny, _bar_label(bar_am), ha="center", va="bottom", fontsize=10,
                fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

        # N / E arrows from the WCS, in a small panel right of the image (never over a marker)
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
        c0 = w.pixel_to_world(cx, cy)
        self.ax.apply_aspect()
        pos = self.ax.get_position()
        inset = self.fig.add_axes([pos.x1 + 0.008, pos.y1 - 0.09, 0.07, 0.09])
        inset.set_xlim(-1.7, 1.7)
        inset.set_ylim(-1.7, 1.7)
        inset.set_aspect("equal")
        inset.axis("off")
        for lab, pa in (("N", 0.0), ("E", 90.0)):
            px, py = w.world_to_pixel(c0.directional_offset_by(pa * u.deg, 60.0 * scale * u.arcsec))
            dx, dy = float(px) - cx, float(py) - cy
            n = math.hypot(dx, dy) or 1.0
            dx, dy = dx / n, dy / n
            inset.annotate("", xy=(dx, dy), xytext=(0.0, 0.0),
                           arrowprops=dict(arrowstyle="-|>", color="#c2410c", lw=2))
            inset.text(1.4 * dx, 1.4 * dy, lab, color="#c2410c", fontsize=12, fontweight="bold",
                       ha="center", va="center")
        self.subtitle = f'scale {scale:.2f}"/px, field {w_am / 60.0:.2f} x {h_am / 60.0:.2f} deg'

    def save(self, output_path: Path, title: str, draw: Callable[[Any], list[Any]]) -> Path:
        """Draw the markers (``draw(ax)`` returns the artists it added), save as JPEG, remove them."""
        added: list[Any] = []
        try:
            added = list(draw(self.ax) or [])
            self.ax.set_title(title + (f"\n{self.subtitle}" if self.subtitle else ""), fontsize=10)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(output_path, dpi=100, bbox_inches="tight", format="jpeg",
                             pil_kwargs={"quality": FIELD_MAP_JPEG_QUALITY, "optimize": True})
        finally:
            for art in added:
                art.remove()
        return output_path

    def close(self) -> None:
        self._plt.close(self.fig)


def _inside(ax: Any, x: float, y: float) -> bool:
    """A marker is drawn only on the image (the view never widens; labels stay inside)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1)


def _target_markers(target_row: pd.Series, comp_rows: pd.DataFrame) -> Callable[[Any], list[Any]]:
    def draw(ax: Any) -> list[Any]:
        import matplotlib.patches as mpatches  # noqa: PLC0415

        out: list[Any] = []
        try:
            tx, ty = float(target_row["x"]), float(target_row["y"])
            if not _inside(ax, tx, ty):
                raise ValueError("target outside the image")
            out.append(ax.add_patch(mpatches.Rectangle(
                (tx - 15, ty - 15), 30, 30, linewidth=2.0, edgecolor="red", facecolor="none")))
            from star_page_m3 import target_display_name  # noqa: PLC0415

            tname = target_display_name(target_row)[:20]  # G11-22: the export STARID name, never "nan"
            out.append(ax.text(tx + 18, ty, f"T: {tname}", color="red", fontsize=7, va="center", fontweight="bold",
                               clip_on=True))
        except (KeyError, TypeError, ValueError):
            pass
        for i, (_, crow) in enumerate(comp_rows.iterrows(), 1):
            try:
                cx, cy = float(crow["x"]), float(crow["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if not _inside(ax, cx, cy):
                continue
            out.append(ax.add_patch(mpatches.Circle(
                (cx, cy), radius=14, linewidth=1.5, edgecolor="#008a2e", facecolor="none")))
            out.append(ax.text(cx + 16, cy, f"C{i:02d}", color="#008a2e", fontsize=7, va="center", fontweight="bold",
                               clip_on=True))
        return out

    return draw


class TargetFieldMap:
    """Per-target field maps (``lightcurves/field_map_<id>.jpg``): the target as a red square, its
    comparison stars as numbered green circles, on the shared MASTERSTAR background."""

    def __init__(self, data: np.ndarray, header: Any = None) -> None:
        self.canvas = FieldMapCanvas(data, header)

    def save(self, output_path: Path, target_row: pd.Series, comp_rows: pd.DataFrame) -> Path:
        from star_page_m3 import target_display_name  # noqa: PLC0415

        name = target_display_name(target_row)  # G11-22: the name of the export files (VSX, exoplanet host, Gaia DR3)
        return self.canvas.save(output_path, f"VYVAR - {name}   (red = VSX target, green = comparison stars)",
                                _target_markers(target_row, comp_rows))

    def close(self) -> None:
        self.canvas.close()


_TFM_WORKER: dict[str, Any] = {}


def _tfm_worker_init(ctx: dict[str, Any]) -> None:
    """Pool initializer: the worker's map on its own copy of the background; ``{}`` closes it."""
    old = _TFM_WORKER.pop("map", None)
    if old is not None:
        old.close()
    if ctx:
        _TFM_WORKER["map"] = TargetFieldMap(ctx["data"], ctx["header"])


def _tfm_worker_save(job: tuple[str, pd.Series, pd.DataFrame]) -> str:
    """Draw one target map in a worker; returns "" or the error text."""
    try:
        _TFM_WORKER["map"].save(Path(job[0]), job[1], job[2])
        return ""
    except Exception as exc:  # noqa: BLE001
        return str(exc)


class DeferredTargetFieldMaps:
    """SPEED-2026-10: the per-target maps are queued during the target loop and drawn at ``close``,
    in a process pool when ``workers`` > 1 (each worker draws on its own copy of the background).

    Same files as :class:`TargetFieldMap`; a map that fails is logged and skipped, as before.
    """

    def __init__(self, data: np.ndarray, header: Any = None, *, workers: int | None = None) -> None:
        self._data = np.asarray(data)
        self._header = header
        self._workers = workers  # None: the run's resource budget decides at close()
        self._jobs: list[tuple[str, pd.Series, pd.DataFrame]] = []
        if not np.isfinite(np.asarray(self._data, dtype=np.float64)).any():
            raise ValueError("MASTERSTAR has no finite pixels")
        import matplotlib  # noqa: F401, PLC0415  (ImportError here, as for the canvas)

    def save(self, output_path: Path, target_row: pd.Series, comp_rows: pd.DataFrame) -> Path:
        self._jobs.append((str(output_path), target_row.copy(), comp_rows.copy()))
        return Path(output_path)

    def close(self) -> None:
        jobs, self._jobs = self._jobs, []
        if not jobs:
            return
        from parallel_frames import frame_bytes, map_frames, stage_workers  # noqa: PLC0415

        if self._workers is None:
            nw = stage_workers(len(jobs), frame_bytes(*self._data.shape[:2]), stage="target field maps")
        else:
            nw = max(1, min(int(self._workers), len(jobs)))
        errors = list(map_frames(_tfm_worker_save, jobs, workers=nw, initializer=_tfm_worker_init,
                                 ctx={"data": self._data, "header": self._header}, stage="target field maps"))
        for job, err in zip(jobs, errors, strict=True):
            if err:
                LOGGER.warning("[APERTURE] Optional artifact write failed (field map %s): %s", Path(job[0]).name, err)


def save_field_map(
    output_path: Path,
    data: np.ndarray,
    header: Any,
    active_targets: pd.DataFrame,
    comp_df: pd.DataFrame,
) -> Path | None:
    """``photometry/field_map.jpg``: every active target (red, named) and comparison star (green)."""
    from photometry_shared import _normalize_gaia_id  # noqa: PLC0415
    from star_page_m3 import target_display_name  # noqa: PLC0415

    try:
        canvas = FieldMapCanvas(data, header)
    except (ValueError, ImportError) as exc:
        LOGGER.warning("[FIELD MAP] not drawn: %s", exc)
        return None
    target_ids = {t for t in (_normalize_gaia_id(r.get("catalog_id", "")) for _, r in active_targets.iterrows()) if t}

    def draw(ax: Any) -> list[Any]:
        import matplotlib.patches as mpatches  # noqa: PLC0415

        out: list[Any] = []
        seen: set[str] = set()
        skipped = 0
        for _, row in comp_df.iterrows():
            cid = _normalize_gaia_id(row.get("catalog_id", ""))
            if not cid or cid in seen:
                continue
            if cid in target_ids:  # never a comparison marker on a known target
                skipped += 1
                continue
            seen.add(cid)
            try:
                xc, yc = float(row["x"]), float(row["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if not _inside(ax, xc, yc):
                continue
            out.append(ax.add_patch(mpatches.Rectangle(
                (xc - 8, yc - 8), 16, 16, linewidth=1.0, edgecolor="#008a2e", facecolor="none")))
        if skipped:
            logging.info("[FIELD MAP] Skipped %d comp markers - star is a known target", skipped)
        for _, row in active_targets.iterrows():
            try:
                xc, yc = float(row["x"]), float(row["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if not _inside(ax, xc, yc):
                continue
            out.append(ax.add_patch(mpatches.Rectangle(
                (xc - 12, yc - 12), 24, 24, linewidth=1.5, edgecolor="red", facecolor="none")))
            out.append(ax.text(xc + 14, yc, target_display_name(row)[:20], color="red", fontsize=5, va="center",
                               clip_on=True))
        return out

    try:
        return canvas.save(output_path, "VYVAR - Field Map (red = VSX target, green = comparison star)", draw)
    finally:
        canvas.close()
