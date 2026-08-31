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
INV_PSF_LC_PIN_01 = "INV-PSF-LC-PIN-01"
ZP_MEMBERSHIP_STRICT = "fit_ok_strict"
ZP_MEMBERSHIP_FOR_ZP = "fit_ok_for_zp"
# Draft 516 manifest: rig.equipment_id=1, rig.telescope_id=1 (pair, not scanning_id).
WIDE_RIG_IDENTITY_KEY = "1:1"
UNVALIDATED_MEMBERSHIP_LINE = (
    "# psf_zp_membership: fit_ok_strict (psf_fit_ok_for_zp not "
    "validated for this rig; see EPSF-ZP-OK-XRIG-01)"
)

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
    "psf_ac_policy=",
    "gain_authority=",
    "ensemble_n_comp=",
    "ensemble_pinned_ids=",
    "git_hash=",
    "git_dirty=",
    TRUST_NOTE_LINE1,
    TRUST_NOTE_LINE2,
    "psf_lc_n_epochs_full=",
    "psf_lc_n_epochs_dropped_pin=",
    "psf_ap_level_offset_mag=",
    "psf_zp_membership_effective=",
    "psf_zp_membership_rig_validated=",
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
    "psf_ac_policy",
    "flux",
    "dao_flux",
)


def rig_identity_key(equipment_id: Any, telescope_id: Any) -> str:
    """Draft 516 identity is the equipment_id:telescope_id pair, not scanning_id."""
    return f"{int(equipment_id)}:{int(telescope_id)}"


def load_rig_identity_from_manifest(start: Path | str) -> tuple[str | None, Path | None]:
    p = Path(start)
    try:
        p = p.resolve()
    except OSError:
        p = Path(start)
    if p.is_file():
        p = p.parent
    for cand in (p, *p.parents):
        man = cand / "draft_manifest.json"
        if not man.is_file():
            continue
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rig = data.get("rig") if isinstance(data, dict) else None
        if not isinstance(rig, dict):
            rig = data if isinstance(data, dict) else {}
        eq = rig.get("equipment_id")
        tel = rig.get("telescope_id")
        if eq is None or tel is None:
            continue
        try:
            return rig_identity_key(eq, tel), man
        except (TypeError, ValueError):
            continue
    return None, None


def psf_fit_ok_for_zp_mask(
    fit_ok: np.ndarray,
    flux: np.ndarray,
    chi2: np.ndarray,
) -> np.ndarray:
    """ZP membership: stored psf_fit_ok OR (finite flux>0 AND finite chi2). No refit."""
    ok = np.asarray(fit_ok, dtype=bool)
    fl = np.asarray(flux, dtype=np.float64)
    ch = np.asarray(chi2, dtype=np.float64)
    extra = np.isfinite(fl) & (fl > 0) & np.isfinite(ch)
    return ok | extra


def zp_membership_usable(
    *,
    mode: str,
    fit_ok: np.ndarray,
    flux: np.ndarray,
    chi2: np.ndarray,
) -> np.ndarray:
    fl = np.asarray(flux, dtype=np.float64)
    ok = np.asarray(fit_ok, dtype=bool)
    if str(mode).strip() == ZP_MEMBERSHIP_FOR_ZP:
        return psf_fit_ok_for_zp_mask(ok, fl, chi2)
    return ok & np.isfinite(fl) & (fl > 0)


def resolve_zp_membership(
    *,
    platesolve_dir: Path | str,
    cfg: Any | None = None,
) -> tuple[str, bool, list[str], str | None]:
    """Return (effective_mode, rig_validated, extra_header_lines, rig_key)."""
    requested = ZP_MEMBERSHIP_FOR_ZP
    allow: list[str] = [WIDE_RIG_IDENTITY_KEY]
    if cfg is not None:
        requested = str(getattr(cfg, "psf_zp_membership", requested) or requested).strip()
        raw = getattr(cfg, "psf_zp_for_zp_validated_rigs", allow)
        if isinstance(raw, str):
            allow = [x.strip() for x in raw.split(",") if x.strip()]
        elif isinstance(raw, (list, tuple)):
            allow = [str(x).strip() for x in raw if str(x).strip()]
        else:
            allow = [WIDE_RIG_IDENTITY_KEY]
    if requested not in (ZP_MEMBERSHIP_STRICT, ZP_MEMBERSHIP_FOR_ZP):
        requested = ZP_MEMBERSHIP_FOR_ZP
    rig_key, _man = load_rig_identity_from_manifest(platesolve_dir)
    validated = bool(rig_key and rig_key in set(allow))
    extra: list[str] = []
    if requested == ZP_MEMBERSHIP_FOR_ZP and not validated:
        extra.append(UNVALIDATED_MEMBERSHIP_LINE)
        return ZP_MEMBERSHIP_STRICT, False, extra, rig_key
    return requested, validated, extra, rig_key


def _norm_cid_gaia(raw: Any) -> str:
    """Gaia canonical id via ``normalize_gaia_source_id``; on exception return stripped raw."""
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

    tid = _norm_cid_gaia(target_cid)
    members = get_pinned_members_for_target(tid)
    if members:
        ids = [_norm_cid_gaia(m.comp_catalog_id) for m in members]
        ids = [c for c in ids if c]
        weights = {_norm_cid_gaia(m.comp_catalog_id): float(m.comp_weight) for m in members}
        return ids, weights, "pinned"
    comp_pt = Path(photometry_dir) / "comparison_stars_per_target.csv"
    if not comp_pt.is_file():
        return [], {}, "none"
    df = pd.read_csv(comp_pt, low_memory=False, dtype={"catalog_id": str, "target_catalog_id": str})
    if "target_catalog_id" not in df.columns or "catalog_id" not in df.columns:
        return [], {}, "none"
    tcol = df["target_catalog_id"].map(_norm_cid_gaia)
    sub = df.loc[tcol == tid]
    ids = [_norm_cid_gaia(v) for v in sub["catalog_id"].tolist()]
    ids = [c for c in ids if c]
    weights: dict[str, float] = {}
    if "comp_weight" in sub.columns:
        for _, row in sub.iterrows():
            cid = _norm_cid_gaia(row.get("catalog_id"))
            w = float(pd.to_numeric(row.get("comp_weight"), errors="coerce"))
            if cid and math.isfinite(w) and w > 0:
                weights[cid] = w
    return ids, weights, "comparison_stars_per_target"


def build_provenance_header(
    *,
    epsf_meta: dict[str, Any],
    psf_weight_mode: str,
    psf_err_mode: str,
    psf_ac_policy: str,
    gain_value: float,
    gain_source: str,
    n_comp: int,
    pinned_ids: Sequence[str],
    ensemble_source: str,
    git_hash: str,
    git_dirty: str,
    n_epochs_full: int = 0,
    n_epochs_dropped_pin: int = 0,
    psf_ap_level_offset_mag: float = float("nan"),
    zp_membership_effective: str = ZP_MEMBERSHIP_STRICT,
    zp_membership_rig_validated: bool = False,
    extra_header_lines: Sequence[str] = (),
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
        f"# psf_ac_policy={psf_ac_policy}",
        f"# gain_authority=g_pt={gtxt} source={gain_source}",
        f"# ensemble_n_comp={int(n_comp)}",
        f"# ensemble_pinned_ids={ids}",
        f"# ensemble_source={ensemble_source}",
        f"# git_hash={git_hash}",
        f"# git_dirty={git_dirty}",
        TRUST_NOTE_LINE1,
        TRUST_NOTE_LINE2,
        f"# product={PRODUCT_NAME}",
        f"# psf_lc_n_epochs_full={int(n_epochs_full)}",
        f"# psf_lc_n_epochs_dropped_pin={int(n_epochs_dropped_pin)}",
        f"# psf_ap_level_offset_mag={psf_ap_level_offset_mag:.6g}"
        if math.isfinite(float(psf_ap_level_offset_mag))
        else "# psf_ap_level_offset_mag=nan",
        f"# psf_zp_membership_effective={zp_membership_effective}",
        f"# psf_zp_membership_rig_validated={'true' if zp_membership_rig_validated else 'false'}",
    ]
    for extra in extra_header_lines:
        s = str(extra).rstrip()
        if s:
            lines.append(s)
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
        df["catalog_id"] = df["catalog_id"].map(_norm_cid_gaia)
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
                    "psf_ac_policy": "",
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
                "psf_ac_policy": str(rec.get("psf_ac_policy") or ""),
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
    zp_membership: str = ZP_MEMBERSHIP_STRICT,
    zp_membership_rig_validated: bool = False,
    extra_header_lines: Sequence[str] = (),
) -> Path | None:
    from photometry_core import ensemble_normalize  # noqa: PLC0415
    from sigma_floor_core import combine_production_err_mag  # noqa: PLC0415

    tid = _norm_cid_gaia(target_cid)
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
    psf_chi2 = tgt["psf_chi2"].to_numpy(dtype=np.float64)
    zp_mode = str(zp_membership).strip() or ZP_MEMBERSHIP_STRICT
    usable = zp_membership_usable(
        mode=zp_mode, fit_ok=psf_ok, flux=psf_flux, chi2=psf_chi2
    )
    target_mag = np.full(n, np.nan, dtype=np.float64)
    target_mag[usable] = _flux_to_inst_mag(psf_flux)[usable]

    comp_mag: dict[str, np.ndarray] = {}
    comp_quality: dict[str, dict] = {}
    for cid in comp_ids:
        ctab = _star_frame_table(stack, cid, epoch_keys)
        cf = ctab["psf_flux"].to_numpy(dtype=np.float64)
        cchi = ctab["psf_chi2"].to_numpy(dtype=np.float64)
        cok_fit = ctab["psf_fit_ok"].to_numpy(dtype=bool)
        cok = zp_membership_usable(mode=zp_mode, fit_ok=cok_fit, flux=cf, chi2=cchi)
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

    # INV-PSF-LC-PIN-01: full pinned ensemble membership or NaN. No partial ZP.
    drop_reason = [""] * n
    psf_delta = np.asarray(psf_delta, dtype=np.float64)
    for i in range(n):
        missing: list[str] = []
        for cid in comp_ids:
            mag_i = float(comp_mag[cid][i]) if i < len(comp_mag[cid]) else float("nan")
            if not math.isfinite(mag_i):
                missing.append(cid)
        if missing:
            psf_delta[i] = float("nan")
            drop_reason[i] = f"comp_psf_fail:{missing[0]}"

    n_dropped_pin = int(sum(1 for r in drop_reason if r))
    n_full = int(n - n_dropped_pin)
    ap_delta = pd.to_numeric(ap.get("delta_mag"), errors="coerce").to_numpy(dtype=np.float64)
    if len(ap_delta) != n:
        ap_delta = np.full(n, np.nan)
    level_vals = []
    for i in range(n):
        if drop_reason[i]:
            continue
        pdlt = float(psf_delta[i])
        adlt = float(ap_delta[i])
        if math.isfinite(pdlt) and math.isfinite(adlt):
            level_vals.append(pdlt - adlt)
    level_off = float(np.median(level_vals)) if level_vals else float("nan")

    pfe = tgt["psf_flux_err"].to_numpy(dtype=np.float64)
    err_rel = np.full(n, np.nan, dtype=np.float64)
    good_err = usable & np.isfinite(pfe) & (pfe > 0)
    err_rel[good_err] = pfe[good_err] / psf_flux[good_err]
    psf_delta_err = np.full(n, np.nan, dtype=np.float64)
    sc = np.asarray(ensemble_scatter, dtype=np.float64)
    for i in range(n):
        if not usable[i] or not math.isfinite(float(err_rel[i])):
            continue
        if drop_reason[i]:
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
    ac_policy = ""
    for col, default in (
        ("psf_weight_mode", "full_ccd"),
        ("psf_err_mode", "sandwich_full_ccd"),
        ("psf_ac_policy", "p4_none"),
    ):
        vals = [str(v).strip() for v in tgt[col].tolist() if str(v).strip()] if col in tgt.columns else []
        if col == "psf_weight_mode":
            weight_mode = vals[0] if vals else default
        elif col == "psf_err_mode":
            err_mode = vals[0] if vals else default
        else:
            ac_policy = vals[0] if vals else default
    if not ac_policy:
        try:
            from config import AppConfig as _AppConfig

            ac_policy = str(getattr(_AppConfig(), "psf_ac_policy", "p4_none") or "p4_none")
        except Exception:  # noqa: BLE001
            ac_policy = "p4_none"

    header = build_provenance_header(
        epsf_meta=epsf_meta,
        psf_weight_mode=weight_mode,
        psf_err_mode=err_mode,
        psf_ac_policy=ac_policy,
        gain_value=gain_value,
        gain_source=gain_source,
        n_comp=len(comp_ids),
        pinned_ids=comp_ids,
        ensemble_source=ens_source,
        git_hash=git_hash,
        git_dirty=git_dirty,
        n_epochs_full=n_full,
        n_epochs_dropped_pin=n_dropped_pin,
        psf_ap_level_offset_mag=level_off,
        zp_membership_effective=zp_mode,
        zp_membership_rig_validated=bool(zp_membership_rig_validated),
        extra_header_lines=extra_header_lines,
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
            "psf_epoch_drop_reason": drop_reason,
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
    output_directory: Path | str | None = None,
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Write ``lightcurve_<id>_psf.csv`` for aperture LC targets. Additive only."""
    ps = Path(platesolve_dir)
    root = Path(frames_root)
    phot = Path(photometry_dir) if photometry_dir is not None else ps / "photometry"
    lc_dir = phot / "lightcurves"
    if not lc_dir.is_dir():
        raise FileNotFoundError(f"Aperture LC directory missing: {lc_dir}")
    out_dir = Path(output_directory) if output_directory is not None else lc_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg is None:
        try:
            from config import AppConfig as _AppConfig

            cfg = _AppConfig()
        except Exception:  # noqa: BLE001
            cfg = None
    zp_mode, zp_validated, zp_extra, zp_rig = resolve_zp_membership(
        platesolve_dir=ps, cfg=cfg
    )

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
                wanted.append(_norm_cid_gaia(cid))
        target_ids = [c for c in wanted if c]
    else:
        target_ids = [_norm_cid_gaia(t) for t in target_ids if _norm_cid_gaia(t)]

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
            lc_dir=out_dir,
            photometry_dir=phot,
            epsf_meta=epsf_meta,
            gain_value=gain_value,
            gain_source=gain_source,
            git_hash=git_hash,
            git_dirty=git_dirty,
            zp_membership=zp_mode,
            zp_membership_rig_validated=zp_validated,
            extra_header_lines=zp_extra,
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
        "output_directory": str(out_dir),
        "psf_zp_membership_effective": zp_mode,
        "psf_zp_membership_rig_validated": zp_validated,
        "psf_zp_membership_rig": zp_rig,
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
    parser.add_argument("--output-directory", type=Path, default=None)
    args = parser.parse_args(argv)
    out = write_internal_psf_lightcurves(
        platesolve_dir=args.platesolve_dir,
        frames_root=args.frames_root,
        target_ids=args.target_id,
        output_directory=args.output_directory,
    )
    print(json.dumps({"n_written": out["n_written"], "n_skipped": out["n_skipped"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
