"""
VYVAR lokálny astrometry.net solver cez WSL2.

Fallback plate-solver pre MASTERSTAR.fits keď VYVAR solver
produkuje zlý WCS (anizotropná mierka, nízky match rate).

Prerekvizity (WSL Ubuntu):
  sudo apt-get install -y astrometry.net
  mkdir -p ~/astrometry/index
  wget -P ~/astrometry/index http://data.astrometry.net/4200/index-4208.fits
  wget -P ~/astrometry/index http://data.astrometry.net/4200/index-4209.fits
  echo "add_path /home/<user>/astrometry/index" > ~/.config/astrometry.cfg
  echo "inparallel" >> ~/.config/astrometry.cfg

Voliteľne na Windows: nastav ``VYVAR_WSL_ASTROMETRY_CFG`` na WSL cestu k ``astrometry.cfg``.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from infolog import log_event
from utils import header_key_is_celestial_wcs, strip_celestial_wcs_keys

# --- Konfigurácia ---
_DEFAULT_WSL_CFG = os.environ.get("VYVAR_WSL_ASTROMETRY_CFG", "/home/uhlar/.config/astrometry.cfg")
_DEFAULT_TIMEOUT = 180
_ANISOTROPY_LIMIT = 1.3  # max ratio sx/sy pred zamietnutím


def _wsl_path(windows_path: str | Path) -> str:
    """Konvertuje Windows cestu na WSL /mnt/ cestu."""
    p = str(windows_path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]
        return f"/mnt/{drive}{rest}"
    return p


def _windows_path_from_wsl(wsl_path: str) -> Path:
    """Konvertuje WSL /mnt/ cestu späť na Windows cestu."""
    if wsl_path.startswith("/mnt/") and len(wsl_path) > 6:
        drive = wsl_path[5].upper()
        rest = wsl_path[6:].replace("/", "\\")
        return Path(f"{drive}:{rest}")
    return Path(wsl_path)


def is_wsl_available() -> bool:
    """Skontroluje či WSL a solve-field sú dostupné."""
    try:
        r = subprocess.run(
            ["wsl", "solve-field", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def solve_field_wsl(
    fits_path: str | Path,
    ra_hint_deg: float,
    dec_hint_deg: float,
    scale_arcsec_per_px: float,
    *,
    scale_err_pct: float = 15.0,
    radius_deg: float = 3.0,
    wsl_cfg: str = _DEFAULT_WSL_CFG,
    timeout_sec: int = _DEFAULT_TIMEOUT,
    downsample: int = 2,
    no_tweak: bool = True,
    anisotropy_max_ratio: float | None = None,
) -> dict[str, Any]:
    """
    Spustí astrometry.net solve-field v WSL na MASTERSTAR.fits.

    Returns dict:
        success (bool)
        wcs (WCS) — ak success=True
        solved_fits_win (str) — Windows cesta k .new
        error (str) — ak success=False
    """
    fits_path = Path(fits_path).resolve()
    if not fits_path.exists():
        return {"success": False, "error": f"FITS neexistuje: {fits_path}"}

    wsl_fits = _wsl_path(fits_path)
    wsl_out_dir = str(PurePosixPath(wsl_fits).parent)

    scale_low = scale_arcsec_per_px * (1.0 - scale_err_pct / 100.0)
    scale_high = scale_arcsec_per_px * (1.0 + scale_err_pct / 100.0)

    cmd = [
        "wsl",
        "solve-field",
        "--no-plots",
        "--no-verify",
        "--overwrite",
        "--dir",
        wsl_out_dir,
        "--ra",
        f"{ra_hint_deg:.6f}",
        "--dec",
        f"{dec_hint_deg:.6f}",
        "--radius",
        f"{radius_deg:.2f}",
        "--scale-units",
        "arcsecperpix",
        "--scale-low",
        f"{scale_low:.4f}",
        "--scale-high",
        f"{scale_high:.4f}",
        "--downsample",
        str(downsample),
        "--crpix-center",
        "--config",
        wsl_cfg,
    ]

    if no_tweak:
        cmd.append("--no-tweak")

    cmd.append(wsl_fits)

    log_event(f"astrometry.net WSL: {' '.join(cmd[2:])}")  # bez 'wsl' prefixu

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"solve-field timeout {timeout_sec}s"}
    except FileNotFoundError:
        return {"success": False, "error": "WSL nie je dostupný (FileNotFoundError)"}
    except Exception as exc:
        return {"success": False, "error": f"solve-field výnimka: {exc}"}

    # astrometry.net zapíše <stem>.new do toho istého priečinka
    solved_fits_win = fits_path.parent / (fits_path.stem + ".new")

    if not solved_fits_win.exists():
        stderr = (proc.stderr or "")[-600:]
        stdout = (proc.stdout or "")[-300:]
        log_event(f"astrometry.net: žiadne riešenie. stderr: {stderr}")
        return {
            "success": False,
            "error": "solve-field: žiadne riešenie (.new neexistuje)",
            "stdout": stdout,
            "stderr": stderr,
        }

    # Načítaj WCS z .new
    try:
        with fits.open(solved_fits_win, memmap=False) as hdul:
            hdr_solved = hdul[0].header.copy()
        wcs = WCS(hdr_solved)
    except Exception as exc:
        return {"success": False, "error": f"WCS load zlyhal: {exc}"}

    if not getattr(wcs, "has_celestial", False):
        return {"success": False, "error": ".new nemá celestial WCS"}

    # Validácia mierky
    sx = sy = ratio = None
    try:
        pm = wcs.pixel_scale_matrix
        sx = abs(pm[0, 0]) * 3600
        sy = abs(pm[1, 1]) * 3600
        ratio = max(sx, sy) / max(min(sx, sy), 0.001)
        log_event(
            f"astrometry.net: scale={sx:.3f}x{sy:.3f} arcsec/px "
            f"ratio={ratio:.2f} hint_scale={scale_arcsec_per_px:.3f}"
        )
        _lim = float(anisotropy_max_ratio) if anisotropy_max_ratio is not None else float(_ANISOTROPY_LIMIT)
        if math.isfinite(_lim) and _lim > 0 and ratio > _lim:
            return {
                "success": False,
                "error": f"astrometry.net: anizotropná mierka ratio={ratio:.2f} > {_lim}",
            }
    except Exception as exc:
        log_event(f"astrometry.net: scale check warning: {exc}")

    # Skopíruj WCS z .new do pôvodného MASTERSTAR.fits
    _inject_wcs_into_masterstar(fits_path, hdr_solved)

    log_event(f"astrometry.net WSL: solve OK → {solved_fits_win.name}")
    return {
        "success": True,
        "wcs": wcs,
        "solved_fits_win": str(solved_fits_win),
        "scale_x_arcsec": sx,
        "scale_y_arcsec": sy,
        "anisotropy_ratio": ratio,
    }


def _inject_wcs_into_masterstar(
    masterstar_path: Path,
    hdr_solved: fits.Header,
) -> None:
    """Prepíše WCS v ``MASTERSTAR.fits`` z astrometry.net riešenia (cez Astropy ``WCS.to_header``)."""
    try:
        wh = WCS(hdr_solved).to_header(relax=True)
    except Exception as exc:
        log_event(f"astrometry.net: WCS to_header failed: {exc}")
        return

    try:
        with fits.open(masterstar_path, mode="update", memmap=False) as hdul:
            h = hdul[0].header
            strip_celestial_wcs_keys(h)
            for k in list(h.keys()):
                if k == "WCSAXES":
                    try:
                        del h[k]
                    except Exception:
                        pass
            for card in wh.cards:
                kw = str(card.keyword).strip()
                if kw in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                    continue
                if kw.startswith("NAXIS") and kw != "NAXIS":
                    continue
                if not (header_key_is_celestial_wcs(kw) or kw == "WCSAXES"):
                    continue
                try:
                    h[kw] = (card.value, str(card.comment or ""))
                except Exception:
                    try:
                        h[kw] = card.value
                    except Exception:
                        pass
            h.add_history("VYVAR: WCS z astrometry.net WSL solve-field")
            hdul.flush()
        log_event(f"astrometry.net: WCS injected → {masterstar_path.name}")
    except Exception as exc:
        log_event(f"astrometry.net: WCS inject failed: {exc}")


def masterstar_wcs_quality(
    wcs: WCS,
    expected_scale_arcsec: float,
    *,
    anisotropy_limit: float | None = None,
) -> dict[str, Any]:
    """
    Vypočíta kvalitu WCS riešenia.
    Použiť na rozhodnutie VYVAR solver vs astrometry.net.

    Returns dict s kľúčmi: ok, ratio, scale_x, scale_y, scale_err_pct
    """
    try:
        pm = wcs.pixel_scale_matrix
        sx = abs(pm[0, 0]) * 3600
        sy = abs(pm[1, 1]) * 3600
        ratio = max(sx, sy) / max(min(sx, sy), 0.001)
        mean_scale = (sx + sy) / 2.0
        scale_err = abs(mean_scale - expected_scale_arcsec) / max(expected_scale_arcsec, 1e-9) * 100.0
        _lim = float(anisotropy_limit) if anisotropy_limit is not None else float(_ANISOTROPY_LIMIT)
        if not math.isfinite(_lim) or _lim <= 0:
            _lim = float(_ANISOTROPY_LIMIT)
        ok = ratio <= _lim and scale_err < 20.0
        return {
            "ok": ok,
            "ratio": ratio,
            "scale_x": sx,
            "scale_y": sy,
            "mean_scale": mean_scale,
            "scale_err_pct": scale_err,
        }
    except Exception as exc:
        return {"ok": False, "ratio": 999.0, "error": str(exc)}
