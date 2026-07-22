# -*- coding: ascii -*-
"""OSC-2: solve WCS once on oneRGGB, reuse geometry for R/G/B; unified QC verdict."""
from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from astropy.io import fits

from osc_extract import OSC_CHANNELS, channel_obs_group_folder, is_channel_obs_group_folder

LOGGER = logging.getLogger(__name__)

OSC_RGB_CHANNELS: tuple[str, ...] = ("R", "G", "B")
OSC_BAND_TOKENS: dict[str, str] = {
    "R": "TR",
    "G": "TG",
    "B": "TB",
    "oneRGGB": "CLEAR",
}
OSC_REGISTRATION_HANDOFF = "osc_registration_handoff.json"
OSC_WCS_PROPAGATION_META = "osc_wcs_propagation.json"
from vyvar_platesolver import MASTERSTAR_PLATESOLVE_MIN_MATCH_RATE as PLATESOLVE_MATCH_RATE_WARN


def parse_osc_channel_from_setup(setup_name: str) -> tuple[str, str | None]:
    """Return ``(base_name, channel)``; channel is None when not an OSC channel folder."""
    name = str(setup_name or "").strip()
    if not name:
        return "", None
    for ch in OSC_CHANNELS:
        suffix = f"_{ch}"
        if name.endswith(suffix):
            return name[: -len(suffix)], ch
    return name, None


def osc_band_token_for_channel(channel: str | None) -> str | None:
    if channel is None:
        return None
    return OSC_BAND_TOKENS.get(str(channel))


def obs_group_band_token(obs_group: str) -> str:
    """Band/filter token for k'' policy and reports (D6)."""
    raw = str(obs_group or "").split("|")[0].strip()
    if not raw:
        return ""
    _base, ch = parse_osc_channel_from_setup(raw)
    tok = osc_band_token_for_channel(ch)
    if tok:
        return tok
    from band_classify import obs_group_first_token

    return obs_group_first_token(obs_group)


def parse_osc_channel(obs_group: str) -> str | None:
    """Return OSC channel token (oneRGGB/R/G/B) or None for mono obs-groups."""
    _base, ch = parse_osc_channel_from_setup(str(obs_group or "").split("|")[0].strip())
    return ch


def is_onerggb_internal_obs_group(obs_group: str) -> bool:
    """E1: oneRGGB is internal-only; never AAVSO/VarAstro eligible."""
    return parse_osc_channel(obs_group) == "oneRGGB"


def is_osc_export_eligible_obs_group(obs_group: str) -> bool:
    """True for exportable OSC channel obs-groups (R/G/B), false for oneRGGB."""
    ch = parse_osc_channel(obs_group)
    return ch in OSC_RGB_CHANNELS


def osc_multiband_summary_rows(draft_dir: Path, obs_group: str) -> list[dict[str, object]]:
    """Compact per-channel summary for PDF report block (JD-consistent frame set per OSC-02)."""
    raw = str(obs_group or "").split("|")[0].strip()
    base, _cur = parse_osc_channel_from_setup(raw)
    if not base or base == raw:
        return []
    root = Path(draft_dir) / "platesolve"
    rows: list[dict[str, object]] = []
    for ch in OSC_CHANNELS:
        og = channel_obs_group_folder(base, ch)
        lc_dir = root / og / "photometry" / "lightcurves"
        n_lc = len(list(lc_dir.glob("lightcurve_*_aperture.csv"))) if lc_dir.is_dir() else 0
        tok = osc_band_token_for_channel(ch) or ""
        rows.append(
            {
                "channel": ch,
                "obs_group": og,
                "band_token": tok,
                "aavso_filt": tok if ch in OSC_RGB_CHANNELS else "internal",
                "export_eligible": ch in OSC_RGB_CHANNELS,
                "n_lightcurves": n_lc,
            }
        )
    return rows


def replicate_qc_verdict_from_one_rggb(*, lights_root: Path, base_name: str) -> dict[str, Any]:
    """D1: copy oneRGGB ``status`` to R/G/B rows; keep per-channel diagnostics; add ``qc_source``."""
    root = Path(lights_root)
    src_dir = root / channel_obs_group_folder(base_name, "oneRGGB")
    src_csv = src_dir / "qc_metrics.csv"
    if not src_csv.is_file():
        raise FileNotFoundError(f"oneRGGB qc_metrics.csv missing: {src_csv}")
    src_df = pd.read_csv(src_csv)
    if src_df.empty or "status" not in src_df.columns:
        raise ValueError(f"Invalid oneRGGB qc_metrics.csv: {src_csv}")
    src_col = "src" if "src" in src_df.columns else "dst"
    verdict_by_name: dict[str, str] = {}
    for _, row in src_df.iterrows():
        p = Path(str(row.get(src_col) or ""))
        verdict_by_name[p.name.casefold()] = str(row["status"]).strip()

    out: dict[str, Any] = {"base": base_name, "channels": {}}
    for ch in OSC_RGB_CHANNELS:
        ch_dir = root / channel_obs_group_folder(base_name, ch)
        ch_csv = ch_dir / "qc_metrics.csv"
        if not ch_csv.is_file():
            raise FileNotFoundError(f"Channel qc_metrics.csv missing for {ch}: {ch_csv}")
        df = pd.read_csv(ch_csv)
        if df.empty:
            continue
        col = "src" if "src" in df.columns else "dst"
        statuses: list[str] = []
        for i, row in df.iterrows():
            fp = Path(str(row.get(col) or ""))
            key = fp.name.casefold()
            if key not in verdict_by_name:
                raise KeyError(f"Frame {fp.name} missing from oneRGGB QC for base {base_name}")
            df.at[i, "status"] = verdict_by_name[key]
            df.at[i, "qc_source"] = "oneRGGB"
            statuses.append(verdict_by_name[key])
        df.to_csv(ch_csv, index=False)
        out["channels"][ch] = {"n": int(len(df)), "statuses": statuses}
    return out


def merge_osc_qc_metrics_at_lights_root(lights_root: Path) -> Path | None:
    """Concatenate per-subfolder qc_metrics.csv into ``lights_root/qc_metrics.csv``."""
    root = Path(lights_root)
    frames: list[pd.DataFrame] = []
    for sub in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
        if not sub.is_dir():
            continue
        qc = sub / "qc_metrics.csv"
        if qc.is_file():
            try:
                frames.append(pd.read_csv(qc))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("[OSC] skip qc merge %s: %s", qc, exc)
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    out = root / "qc_metrics.csv"
    merged.to_csv(out, index=False)
    return out


def partition_jobs_for_osc_alignment(
    job_list: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group OSC channel jobs by base; return reordered jobs (oneRGGB first per bundle)."""
    bundles: dict[str, dict[str, dict[str, Any]]] = {}
    mono: list[dict[str, Any]] = []
    for job in job_list:
        gkey = str(job.get("gkey") or "")
        setup = Path(gkey).name if gkey else ""
        base, ch = parse_osc_channel_from_setup(setup)
        if ch is None:
            mono.append(job)
            continue
        bundles.setdefault(base, {})[ch] = job

    ordered: list[dict[str, Any]] = []
    meta_bundles: dict[str, Any] = {}
    for base in sorted(bundles.keys(), key=str.casefold):
        b = bundles[base]
        if "oneRGGB" not in b:
            raise RuntimeError(
                f"OSC channel group {base}_R/G/B present without oneRGGB sibling (fail-closed)"
            )
        ordered.append(b["oneRGGB"])
        for ch in OSC_RGB_CHANNELS:
            if ch not in b:
                raise RuntimeError(f"OSC bundle {base} missing channel {ch} (fail-closed)")
            ordered.append(b[ch])
        meta_bundles[base] = b
    ordered.extend(mono)
    return ordered, {"has_osc_bundles": bool(meta_bundles), "bundles": meta_bundles}


def unified_allowlist_frame_ids(
    files_by_channel: Mapping[str, Sequence[Path | str]],
) -> set[str]:
    """Frame stems present in every channel allowlist (post-QC ok filenames)."""
    ids: set[str] | None = None
    for paths in files_by_channel.values():
        stems = {Path(p).name.casefold() for p in paths}
        ids = stems if ids is None else ids & stems
    return ids or set()


def write_registration_handoff(
    platesolve_dir: Path,
    *,
    reference_file: str,
    frames: dict[str, dict[str, Any]],
) -> Path:
    """Persist oneRGGB per-frame registration for R/G/B reuse (D2 artifact)."""
    ps = Path(platesolve_dir)
    ps.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_file": str(reference_file),
        "frames": frames,
        "schema": 1,
    }
    out = ps / OSC_REGISTRATION_HANDOFF
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="ascii")
    return out


def load_registration_handoff(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"OSC registration handoff missing: {p}")
    return json.loads(p.read_text(encoding="ascii"))


def propagate_wcs_between_fits(donor_fits: Path, recipient_fits: Path) -> None:
    """Copy celestial WCS from donor MASTERSTAR to channel MASTERSTAR (D3)."""
    from vyvar_alignment_frame import strip_celestial_wcs_keys

    with fits.open(donor_fits, memmap=False) as hd_d:
        donor_hdr = hd_d[0].header
    with fits.open(recipient_fits, mode="update", memmap=False) as hd_r:
        hdr = hd_r[0].header
        strip_celestial_wcs_keys(hdr)
        for k, v in donor_hdr.items():
            from vyvar_alignment_frame import header_key_is_celestial_wcs

            if header_key_is_celestial_wcs(k):
                hdr[k] = v
        hdr["VY_OSCWCS"] = (True, "WCS propagated from oneRGGB MASTERSTAR (OSC-2)")
        hd_r.flush()


def write_wcs_propagation_meta(
    platesolve_dir: Path,
    *,
    donor_dir: Path,
    channel: str,
    match_rate: float | None,
) -> Path:
    ps = Path(platesolve_dir)
    meta = {
        "donor_platesolve_dir": str(Path(donor_dir).resolve()),
        "channel": str(channel),
        "match_rate": float(match_rate) if match_rate is not None and math.isfinite(match_rate) else None,
    }
    out = ps / OSC_WCS_PROPAGATION_META
    out.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="ascii")
    return out


def apply_registration_handoff_to_frame(
    *,
    frame_path: Path,
    frame_data: np.ndarray,
    frame_hdr: fits.Header,
    ref_data: np.ndarray,
    ref_hdr: fits.Header,
    handoff_entry: dict[str, Any],
) -> tuple[np.ndarray, fits.Header, str]:
    """Apply stored oneRGGB registration to a channel frame (D2)."""
    method = str(handoff_entry.get("method") or "astroalign")
    data = np.asarray(frame_data, dtype=np.float32)
    hdr = frame_hdr.copy()
    aligned: np.ndarray | None = None

    if method == "wcs_reproject":
        from vyvar_alignment_frame import _hdr_has_wcs
        from astropy.wcs import WCS
        import warnings
        from astropy.utils.exceptions import FITSFixedWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w_i = WCS(hdr) if _hdr_has_wcs(hdr) else None
            w_r = WCS(ref_hdr) if _hdr_has_wcs(ref_hdr) else None
        if w_i is not None and w_r is not None and getattr(w_i, "has_celestial", False):
            from reproject import reproject_interp

            aligned, _ = reproject_interp(
                (data, w_i.celestial),
                w_r.celestial,
                shape_out=ref_data.shape,
            )
            method = "wcs_reproject"

    if aligned is None and method == "astroalign" and handoff_entry.get("matrix"):
        import astroalign
        from skimage.transform import SimilarityTransform

        from vyvar_alignment_frame import _as_fits_float32_image

        mat = np.array(handoff_entry["matrix"], dtype=np.float64)
        t = SimilarityTransform()
        if mat.size == 9:
            t.params = mat.reshape(3, 3)
        else:
            t.params = np.vstack([mat.reshape(2, 3), [0.0, 0.0, 1.0]])
        aligned, _ = astroalign.apply_transform(t, data, ref_data)
        aligned = _as_fits_float32_image(aligned)
        method = "osc_handoff_astroalign"

    if aligned is None and method in {"phase_correlation", "wcs_shift"}:
        from scipy.ndimage import shift as ndimage_shift

        from vyvar_alignment_frame import _as_fits_float32_image

        dy = float(handoff_entry.get("dy") or 0.0)
        dx = float(handoff_entry.get("dx") or 0.0)
        if data.shape == ref_data.shape:
            cval = float(np.nanmedian(data))
            aligned = ndimage_shift(
                data,
                shift=[dy, dx],
                mode="nearest",
                cval=cval,
                order=1,
                prefilter=False,
            )
            aligned = _as_fits_float32_image(aligned)
            method = f"osc_handoff_{method}"

    if aligned is None:
        from vyvar_alignment_frame import _as_fits_float32_image

        aligned = _as_fits_float32_image(data)
        method = "identity"

    from vyvar_alignment_frame import strip_celestial_wcs_keys

    strip_celestial_wcs_keys(hdr)
    for k, v in ref_hdr.items():
        from vyvar_alignment_frame import header_key_is_celestial_wcs

        if header_key_is_celestial_wcs(k):
            hdr[k] = v
    hdr["VYALGOK"] = (True, "OSC-2 handoff registration from oneRGGB")
    hdr["VY_ALGN"] = (True, "Aligned via oneRGGB registration handoff")
    hdr["VYALGM"] = (method[:30], "Alignment method (OSC-2 handoff)")
    hdr["VY_OSCREG"] = (True, "Registration reused from oneRGGB")
    return np.asarray(aligned, dtype=np.float32), hdr, method


def log_channel_match_rate_verification(
    *,
    channel: str,
    match_rate: float | None,
    one_rggb_failed: bool,
    log_event: Any,
) -> None:
    """D5: WARN per-channel below threshold; FAIL only if oneRGGB itself failed."""
    if match_rate is None or not math.isfinite(float(match_rate)):
        return
    mr = float(match_rate)
    ch = str(channel)
    if mr < PLATESOLVE_MATCH_RATE_WARN:
        msg = (
            f"OSC-2 channel {ch}: DAO-Gaia match_rate={mr * 100.0:.1f}% "
            f"< {PLATESOLVE_MATCH_RATE_WARN * 100.0:.0f}% (expected for B; WARN only)"
        )
        if one_rggb_failed:
            raise RuntimeError(msg.replace("WARN only", "oneRGGB failed - FAIL"))
        log_event(f"VAROVANIE: {msg}")
    else:
        log_event(f"OSC-2 channel {ch}: match_rate={mr * 100.0:.1f}% OK (propagated WCS)")


def collect_handoff_from_alignment_results(
    *,
    reference_file: str,
    star_counts: Sequence[dict[str, Any]],
    registration_by_file: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    frames: dict[str, dict[str, Any]] = {}
    for sc in star_counts:
        fname = str(sc.get("file") or "")
        if not fname:
            continue
        entry = dict(registration_by_file.get(fname) or {})
        entry.setdefault("method", str(sc.get("alignment_method") or sc.get("aligned_method") or "astroalign"))
        entry["aligned"] = bool(sc.get("aligned", True))
        frames[fname] = entry
    return frames


def require_osc_donor_products(donor_platesolve_dir: Path) -> None:
    ps = Path(donor_platesolve_dir)
    for name in (OSC_REGISTRATION_HANDOFF, "MASTERSTAR.fits", "alignment_report.csv"):
        p = ps / name
        if not p.is_file():
            raise FileNotFoundError(f"OSC donor product missing: {p}")


def copy_alignment_report_with_channel_paths(
    donor_report: Path,
    recipient_report: Path,
    *,
    channel_files: Mapping[str, Path],
) -> None:
    """Mirror oneRGGB alignment_report rows for channel paths (geometry reuse metadata)."""
    df = pd.read_csv(donor_report)
    if "file" not in df.columns:
        shutil.copy2(donor_report, recipient_report)
        return
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        r = dict(row)
        fname = str(row.get("file") or "")
        ch_path = channel_files.get(fname)
        if ch_path is not None:
            r["file"] = Path(ch_path).name
            r["osc_registration_source"] = "oneRGGB"
        rows.append(r)
    pd.DataFrame(rows).to_csv(recipient_report, index=False)
