"""Internal diagnostic per-target PSF light curves (EPSF-LC-LOG-01).

These files are NEVER an AAVSO/VarAstro submission product (INV-PSF-SUBMIT-01).
They are written additively beside aperture LCs; catalogs, aperture LCs, and
export bytes are not rewritten by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from gaia_catalog_id import normalize_gaia_source_id
from mag_constants import MAG_ERR_SCALE
from report_methods import lc_csv_path

LOGGER = logging.getLogger(__name__)

INTERNAL_BANNER = "# INTERNAL DIAGNOSTIC PRODUCT - NOT FOR AAVSO/VARASTRO SUBMISSION"
TRUST_NOTE_LINE1 = "# PSF absolute scale untrusted pending EPSF-SHAPE-01;"
TRUST_NOTE_LINE2 = "# relative photometry only"
PRODUCT_NAME = "internal_psf_diagnostic"

# Substrings that T1 / header audits must find (every provenance line).
REQUIRED_HEADER_MARKERS: tuple[str, ...] = (
    INTERNAL_BANNER,
    "epsf_model_file=",
    "epsf_model_sha256=",
    "epsf_n_stars=",
    "epsf_build_timestamp=",
    "epsf_oversampling=",
    "epsf_smoothing_kernel=",
    "epsf_cutout_size=",
    "psf_weight_mode=",
    "psf_err_mode=",
    "gain_authority=",
    "ensemble_n_comp=",
    "ensemble_pinned_ids=",
    "git_hash=",
    "git_dirty=",
    TRUST_NOTE_LINE1,
    TRUST_NOTE_LINE2,
)

_PROC_USECOLS = (
    "catalog_id",
    "source_file",
    "bjd_tdb_mid",
    "hjd_mid",
    "jd_mid",
    "psf_flux",
    "psf_flux_err",
    "psf_chi2",
    "psf_fit_ok",
    "psf_group_n",
    "n_group",
    "psf_weight_mode",
    "psf_err_mode",
    "flux",
    "dao_flux",
)


def _norm_cid(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    try:
        return str(normalize_gaia_source_id(s)).strip()
    except Exception:  # noqa: BLE001
        return s


def _coerce_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return bool(raw)
    if raw is None or (isinstance(raw, float) and not math.isfinite(raw)):
        return False
    t = str(raw).strip().lower()
    return t in ("1", "true", "t", "yes", "y")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _proc_csv_key(source_file: str, proc_path: Path | None = None) -> str:
    """Join key matching aperture LC ``source_file`` (proc_*.csv names)."""
    raw = str(source_file or "").strip()
    if raw.lower().endswith(".csv") and raw.lower().startswith("proc_"):
        return Path(raw).name
    if proc_path is not None:
        return Path(proc_path).name
    if raw.lower().endswith(".fits"):
        return f"proc_{Path(raw).stem}.csv"
    if raw:
        return f"proc_{Path(raw).stem}.csv"
    return ""


def _flux_to_inst_mag(flux: np.ndarray) -> np.ndarray:
    f = np.asarray(flux, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def _load_gain_authority(photometry_dir: Path) -> tuple[float, str]:
    from gain_photon_transfer import GAIN_PT_SIDECAR_NAME  # noqa: PLC0415

    sidecar = Path(photometry_dir) / GAIN_PT_SIDECAR_NAME
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        auth = payload.get("authority") if isinstance(payload, dict) else None
        if isinstance(auth, dict):
            src = str(auth.get("source") or "").strip() or "unresolved"
            g = float(pd.to_numeric(auth.get("g_pt", auth.get("value_e_per_adu_container")), errors="coerce"))
            if math.isfinite(g) and g > 0:
                return g, src
        pt = payload.get("photon_transfer") if isinstance(payload, dict) else None
        if isinstance(pt, dict):
            g = float(pd.to_numeric(pt.get("g_pt"), errors="coerce"))
            if math.isfinite(g) and g > 0:
                return g, "g_pt"
    return float("nan"), "unresolved"


def load_epsf_build_meta(platesolve_dir: Path) -> dict[str, Any]:
    ps = Path(platesolve_dir)
    meta_path = ps / "masterstar_epsf_meta.json"
    model_path = ps / "masterstar_epsf.fits"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
        except (OSError, json.JSONDecodeError):
            meta = {}
    n_stars = meta.get("n_stars_used", meta.get("n_stars"))
    return {
        "model_path": model_path if model_path.is_file() else None,
        "model_name": model_path.name if model_path.is_file() else "",
        "model_sha256": sha256_file(model_path) if model_path.is_file() else "",
        "n_stars": int(n_stars) if n_stars is not None else -1,
        "build_timestamp": str(meta.get("created_utc") or meta.get("created") or ""),
        "oversampling": meta.get("oversampling", ""),
        "smoothing_kernel": str(meta.get("smoothing_kernel") or ""),
        "cutout_size": meta.get("cutout_size", ""),
        "raw_meta": meta,
    }


def _git_provenance() -> tuple[str, str]:
    from photometry_core import _resolve_git_provenance  # noqa: PLC0415

    git_hash, git_dirty, _files = _resolve_git_provenance()
    gh = str(git_hash) if git_hash else ""
    if git_dirty is True:
        dirty = "True"
    elif git_dirty is False:
        dirty = "False"
    else:
        dirty = "null"
    return gh, dirty


def resolve_ensemble_ids(target_cid: str, photometry_dir: Path) -> tuple[list[str], dict[str, float], str]:
    """Pinned ensemble if present; else aperture comparison_stars_per_target membership."""
    from pinned_ensembles import get_pinned_members_for_target  # noqa: PLC0415

    tid = _norm_cid(target_cid)
    members = get_pinned_members_for_target(tid)
    if members:
        ids = [_norm_cid(m.comp_catalog_id) for m in members]
        ids = [c for c in ids if c]
        weights = {_norm_cid(m.comp_catalog_id): float(m.comp_weight) for m in members}
        return ids, weights, "pinned"
    comp_pt = Path(photometry_dir) / "comparison_stars_per_target.csv"
    if not comp_pt.is_file():
        return [], {}, "none"
    df = pd.read_csv(comp_pt, low_memory=False, dtype={"catalog_id": str, "target_catalog_id": str})
    if "target_catalog_id" not in df.columns or "catalog_id" not in df.columns:
        return [], {}, "none"
    tcol = df["target_catalog_id"].map(_norm_cid)
    sub = df.loc[tcol == tid]
    ids = [_norm_cid(v) for v in sub["catalog_id"].tolist()]
    ids = [c for c in ids if c]
    weights: dict[str, float] = {}
    if "comp_weight" in sub.columns:
        for _, row in sub.iterrows():
            cid = _norm_cid(row.get("catalog_id"))
            w = float(pd.to_numeric(row.get("comp_weight"), errors="coerce"))
            if cid and math.isfinite(w) and w > 0:
                weights[cid] = w
    return ids, weights, "comparison_stars_per_target"


def build_provenance_header(
    *,
    epsf_meta: dict[str, Any],
    psf_weight_mode: str,
    psf_err_mode: str,
    gain_value: float,
    gain_source: str,
    n_comp: int,
    pinned_ids: Sequence[str],
    ensemble_source: str,
    git_hash: str,
    git_dirty: str,
) -> list[str]:
    gtxt = f"{gain_value:.6g}" if math.isfinite(float(gain_value)) else "nan"
    ids = ",".join(str(c) for c in pinned_ids)
    lines = [
        INTERNAL_BANNER,
        f"# epsf_model_file={epsf_meta.get('model_name') or ''}",
        f"# epsf_model_sha256={epsf_meta.get('model_sha256') or ''}",
        f"# epsf_n_stars={epsf_meta.get('n_stars')}",
        f"# epsf_build_timestamp={epsf_meta.get('build_timestamp') or ''}",
        f"# epsf_oversampling={epsf_meta.get('oversampling')}",
        f"# epsf_smoothing_kernel={epsf_meta.get('smoothing_kernel') or ''}",
        f"# epsf_cutout_size={epsf_meta.get('cutout_size')}",
        f"# psf_weight_mode={psf_weight_mode}",
        f"# psf_err_mode={psf_err_mode}",
        f"# gain_authority=g_pt={gtxt} source={gain_source}",
        f"# ensemble_n_comp={int(n_comp)}",
        f"# ensemble_pinned_ids={ids}",
        f"# ensemble_source={ensemble_source}",
        f"# git_hash={git_hash}",
        f"# git_dirty={git_dirty}",
        TRUST_NOTE_LINE1,
        TRUST_NOTE_LINE2,
        f"# product={PRODUCT_NAME}",
    ]
    return lines


def _read_proc_stack(frames_root: Path) -> pd.DataFrame:
    root = Path(frames_root)
    files = sorted(root.glob("proc_*.csv"))
    if not files:
        raise FileNotFoundError(f"No proc_*.csv under {root}")
    chunks: list[pd.DataFrame] = []
    for p in files:
        try:
            df = pd.read_csv(p, low_memory=False, dtype={"catalog_id": str})
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PSF-LC] skip unreadable %s: %s", p.name, exc)
            continue
        keep = [c for c in _PROC_USECOLS if c in df.columns]
        if "catalog_id" not in keep:
            continue
        df = df[keep].copy()
        df["catalog_id"] = df["catalog_id"].map(_norm_cid)
        df["proc_csv"] = p.name
        if "source_file" in df.columns:
            df["epoch_key"] = [_proc_csv_key(str(s), p) for s in df["source_file"].tolist()]
        else:
            df["epoch_key"] = p.name
        chunks.append(df)
    if not chunks:
        raise FileNotFoundError(f"No readable proc CSVs under {root}")
    return pd.concat(chunks, ignore_index=True)


def _star_frame_table(stack: pd.DataFrame, cid: str, epoch_keys: Sequence[str]) -> pd.DataFrame:
    sub = stack.loc[stack["catalog_id"] == cid]
    by_key: dict[str, pd.Series] = {}
    if not sub.empty:
        for _, row in sub.iterrows():
            by_key[str(row.get("epoch_key") or "")] = row
    rows = []
    for key in epoch_keys:
        rec = by_key.get(str(key))
        if rec is None:
            rows.append(
                {
                    "epoch_key": key,
                    "psf_flux": np.nan,
                    "psf_flux_err": np.nan,
                    "psf_chi2": np.nan,
                    "psf_fit_ok": False,
                    "n_group": np.nan,
                    "aperture_flux": np.nan,
                    "psf_weight_mode": "",
                    "psf_err_mode": "",
                }
            )
            continue
        flux = float(pd.to_numeric(rec.get("flux"), errors="coerce"))
        if not (math.isfinite(flux) and flux > 0):
            flux = float(pd.to_numeric(rec.get("dao_flux"), errors="coerce"))
        n_grp = rec.get("psf_group_n", rec.get("n_group", np.nan))
        rows.append(
            {
                "epoch_key": key,
                "psf_flux": float(pd.to_numeric(rec.get("psf_flux"), errors="coerce")),
                "psf_flux_err": float(pd.to_numeric(rec.get("psf_flux_err"), errors="coerce")),
                "psf_chi2": float(pd.to_numeric(rec.get("psf_chi2"), errors="coerce")),
                "psf_fit_ok": _coerce_bool(rec.get("psf_fit_ok")),
                "n_group": float(pd.to_numeric(n_grp, errors="coerce")),
                "aperture_flux": flux,
                "psf_weight_mode": str(rec.get("psf_weight_mode") or ""),
                "psf_err_mode": str(rec.get("psf_err_mode") or ""),
            }
        )
    return pd.DataFrame(rows)


def _write_csv_with_header(path: Path, header_lines: Sequence[str], df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        for line in header_lines:
            fh.write(str(line).rstrip() + "\n")
        df.to_csv(fh, index=False)


def write_one_internal_psf_lc(
    *,
    target_cid: str,
    stack: pd.DataFrame,
    aperture_lc: pd.DataFrame,
    lc_dir: Path,
    photometry_dir: Path,
    epsf_meta: dict[str, Any],
    gain_value: float,
    gain_source: str,
    git_hash: str,
    git_dirty: str,
) -> Path | None:
    from photometry_core import ensemble_normalize  # noqa: PLC0415
    from sigma_floor_core import combine_production_err_mag  # noqa: PLC0415

    tid = _norm_cid(target_cid)
    if not tid or aperture_lc is None or aperture_lc.empty:
        return None
    ap = aperture_lc.copy()
    if "source_file" not in ap.columns:
        LOGGER.warning("[PSF-LC] %s aperture LC missing source_file", tid)
        return None
    epoch_keys = [str(s).strip() for s in ap["source_file"].tolist()]
    n = len(epoch_keys)
    tgt = _star_frame_table(stack, tid, epoch_keys)
    comp_ids, weight_map, ens_source = resolve_ensemble_ids(tid, photometry_dir)
    if not comp_ids:
        LOGGER.warning("[PSF-LC] %s has no ensemble membership; skip", tid)
        return None

    psf_ok = tgt["psf_fit_ok"].to_numpy(dtype=bool)
    psf_flux = tgt["psf_flux"].to_numpy(dtype=np.float64)
    usable = psf_ok & np.isfinite(psf_flux) & (psf_flux > 0)
    target_mag = np.full(n, np.nan, dtype=np.float64)
    target_mag[usable] = _flux_to_inst_mag(psf_flux)[usable]

    comp_mag: dict[str, np.ndarray] = {}
    comp_quality: dict[str, dict] = {}
    for cid in comp_ids:
        ctab = _star_frame_table(stack, cid, epoch_keys)
        cf = ctab["psf_flux"].to_numpy(dtype=np.float64)
        cok = ctab["psf_fit_ok"].to_numpy(dtype=bool) & np.isfinite(cf) & (cf > 0)
        mag = np.full(n, np.nan, dtype=np.float64)
        mag[cok] = _flux_to_inst_mag(cf)[cok]
        comp_mag[cid] = mag
        comp_quality[cid] = {"quality": "good"}

    dummy_cat = {cid: 0.0 for cid in comp_ids}
    _mag_calib, psf_delta, ensemble_scatter = ensemble_normalize(
        target_mag,
        comp_mag,
        dummy_cat,
        comp_quality,
        comp_weight_map=weight_map or None,
    )
    _ = _mag_calib

    pfe = tgt["psf_flux_err"].to_numpy(dtype=np.float64)
    err_rel = np.full(n, np.nan, dtype=np.float64)
    good_err = usable & np.isfinite(pfe) & (pfe > 0)
    err_rel[good_err] = pfe[good_err] / psf_flux[good_err]
    psf_delta_err = np.full(n, np.nan, dtype=np.float64)
    sc = np.asarray(ensemble_scatter, dtype=np.float64)
    for i in range(n):
        if not usable[i] or not math.isfinite(float(err_rel[i])):
            continue
        sem = float(sc[i]) if i < len(sc) and math.isfinite(float(sc[i])) else 0.0
        psf_delta_err[i] = combine_production_err_mag(float(err_rel[i]), sem)

    ap_flux = tgt["aperture_flux"].to_numpy(dtype=np.float64)
    ratio = np.full(n, np.nan, dtype=np.float64)
    both = usable & np.isfinite(ap_flux) & (ap_flux > 0)
    ratio[both] = psf_flux[both] / ap_flux[both]

    ap_delta = pd.to_numeric(ap.get("delta_mag"), errors="coerce").to_numpy(dtype=np.float64)
    ap_err = pd.to_numeric(ap.get("err"), errors="coerce").to_numpy(dtype=np.float64)
    if len(ap_delta) != n:
        ap_delta = np.full(n, np.nan)
    if len(ap_err) != n:
        ap_err = np.full(n, np.nan)

    weight_mode = ""
    err_mode = ""
    for col, default in (("psf_weight_mode", "full_ccd"), ("psf_err_mode", "sandwich_full_ccd")):
        vals = [str(v).strip() for v in tgt[col].tolist() if str(v).strip()]
        if col == "psf_weight_mode":
            weight_mode = vals[0] if vals else default
        else:
            err_mode = vals[0] if vals else default

    header = build_provenance_header(
        epsf_meta=epsf_meta,
        psf_weight_mode=weight_mode,
        psf_err_mode=err_mode,
        gain_value=gain_value,
        gain_source=gain_source,
        n_comp=len(comp_ids),
        pinned_ids=comp_ids,
        ensemble_source=ens_source,
        git_hash=git_hash,
        git_dirty=git_dirty,
    )

    def _num(arr: np.ndarray, nd: int) -> list[Any]:
        out: list[Any] = []
        for v in np.asarray(arr, dtype=np.float64):
            if math.isfinite(float(v)):
                out.append(round(float(v), nd))
            else:
                out.append(float("nan"))
        return out

    bjd = pd.to_numeric(ap.get("bjd"), errors="coerce") if "bjd" in ap.columns else pd.Series([np.nan] * n)
    hjd = pd.to_numeric(ap.get("hjd"), errors="coerce") if "hjd" in ap.columns else pd.Series([np.nan] * n)
    jd = pd.to_numeric(ap.get("jd"), errors="coerce") if "jd" in ap.columns else pd.Series([np.nan] * n)

    out = pd.DataFrame(
        {
            "bjd": _num(np.asarray(bjd, dtype=np.float64), 10),
            "hjd": _num(np.asarray(hjd, dtype=np.float64), 10),
            "jd": _num(np.asarray(jd, dtype=np.float64), 10),
            "source_file": epoch_keys,
            "psf_fit_ok": [bool(v) for v in psf_ok],
            "psf_flux": _num(psf_flux, 6),
            "psf_flux_err": _num(pfe, 6),
            "psf_chi2": _num(tgt["psf_chi2"].to_numpy(dtype=np.float64), 6),
            "n_group": _num(tgt["n_group"].to_numpy(dtype=np.float64), 0),
            "psf_delta_mag": _num(psf_delta, 6),
            "psf_delta_mag_err": _num(psf_delta_err, 6),
            "delta_mag": _num(ap_delta, 6),
            "err": _num(ap_err, 6),
            "psf_ap_ratio": _num(ratio, 6),
            "product": [PRODUCT_NAME] * n,
        }
    )
    out_path = lc_csv_path(lc_dir, tid, "psf")
    _write_csv_with_header(out_path, header, out)
    return out_path


def write_internal_psf_lightcurves(
    *,
    platesolve_dir: Path | str,
    frames_root: Path | str,
    photometry_dir: Path | str | None = None,
    target_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Write ``lightcurve_<id>_psf.csv`` for aperture LC targets. Additive only."""
    ps = Path(platesolve_dir)
    root = Path(frames_root)
    phot = Path(photometry_dir) if photometry_dir is not None else ps / "photometry"
    lc_dir = phot / "lightcurves"
    if not lc_dir.is_dir():
        raise FileNotFoundError(f"Aperture LC directory missing: {lc_dir}")

    epsf_meta = load_epsf_build_meta(ps)
    gain_value, gain_source = _load_gain_authority(phot)
    git_hash, git_dirty = _git_provenance()
    stack = _read_proc_stack(root)

    if target_ids is None:
        wanted = []
        for p in sorted(lc_dir.glob("lightcurve_*.csv")):
            stem = p.stem
            if stem.endswith("_psf") or stem.endswith("_adaptive"):
                continue
            cid = stem.replace("lightcurve_", "", 1)
            if cid:
                wanted.append(_norm_cid(cid))
        target_ids = [c for c in wanted if c]
    else:
        target_ids = [_norm_cid(t) for t in target_ids if _norm_cid(t)]

    written: list[str] = []
    skipped: list[str] = []
    for tid in target_ids:
        ap_path = lc_csv_path(lc_dir, tid, "aperture")
        if not ap_path.is_file():
            skipped.append(tid)
            continue
        try:
            ap_lc = pd.read_csv(ap_path, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PSF-LC] cannot read aperture LC %s: %s", ap_path.name, exc)
            skipped.append(tid)
            continue
        path = write_one_internal_psf_lc(
            target_cid=tid,
            stack=stack,
            aperture_lc=ap_lc,
            lc_dir=lc_dir,
            photometry_dir=phot,
            epsf_meta=epsf_meta,
            gain_value=gain_value,
            gain_source=gain_source,
            git_hash=git_hash,
            git_dirty=git_dirty,
        )
        if path is not None:
            written.append(str(path))
        else:
            skipped.append(tid)

    LOGGER.info("[PSF-LC] wrote %d internal PSF LC file(s); skipped %d", len(written), len(skipped))
    return {
        "written": written,
        "skipped": skipped,
        "n_written": len(written),
        "n_skipped": len(skipped),
        "lc_dir": str(lc_dir),
    }


def write_internal_psf_lightcurves_after_epsf_job(
    *,
    platesolve_dir: Path | str,
    frames_root: Path | str,
) -> dict[str, Any] | None:
    """RUN ePSF hook: write diagnostic LCs when aperture LCs already exist."""
    ps = Path(platesolve_dir)
    lc_dir = ps / "photometry" / "lightcurves"
    if not lc_dir.is_dir():
        LOGGER.info("[PSF-LC] skip: no aperture LC dir yet at %s", lc_dir)
        return None
    return write_internal_psf_lightcurves(platesolve_dir=ps, frames_root=frames_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write internal diagnostic PSF light curves.")
    parser.add_argument("--platesolve-dir", required=True, type=Path)
    parser.add_argument("--frames-root", required=True, type=Path)
    parser.add_argument("--target-id", action="append", default=None)
    args = parser.parse_args(argv)
    out = write_internal_psf_lightcurves(
        platesolve_dir=args.platesolve_dir,
        frames_root=args.frames_root,
        target_ids=args.target_id,
    )
    print(json.dumps({"n_written": out["n_written"], "n_skipped": out["n_skipped"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
