"""Blind index series: manifest load, scale-aware tier order, try-in-order verify."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from config import AppConfig
from infolog import log_event
from vyvar_blind_solver import find_blind_candidates, find_blind_hint


def target_density_deg2(*, cell_deg: float, stars_per_cell: int) -> float:
    c = max(float(cell_deg), 1e-6)
    return float(stars_per_cell) / (c * c)


def estimate_rho_img_deg2(
    *,
    plate_scale_arcsec_per_px: float,
    fov_deg: float,
    img_budget: int,
    log_L3_max: float | None = None,
) -> float:
    """ρ_img ≈ budget / central selection area (matches diagnose / runbook)."""
    scale = max(float(plate_scale_arcsec_per_px), 1e-6)
    l3_max = float(log_L3_max) if log_L3_max is not None else 2.75
    L3_max_arcsec = 10.0 ** l3_max
    R_px = (L3_max_arcsec / scale) / 2.0
    R_deg = R_px * scale / 3600.0
    area = math.pi * max(R_deg, 1e-9) ** 2
    return max(1, int(img_budget)) / max(area, 1e-12)


def density_hint_from_plate_scale(plate_scale_arcsec_per_px: float | None) -> float | None:
    if plate_scale_arcsec_per_px is None:
        return None
    try:
        ps = float(plate_scale_arcsec_per_px)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ps) or ps <= 0:
        return None
    if ps >= 5.0:
        return 4.0
    if ps <= 2.0:
        return 95.0
    return None


def load_series_manifest(manifest_path: Path | str) -> list[dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"blind index series manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    tiers = data.get("tiers") if isinstance(data, dict) else data
    if not isinstance(tiers, list) or not tiers:
        raise ValueError(f"manifest has no tiers: {path}")
    out: list[dict[str, Any]] = []
    for t in tiers:
        if not isinstance(t, dict) or not t.get("path"):
            continue
        p = Path(str(t["path"]))
        if not p.is_absolute():
            p = (path.parent / p).resolve()
        row = dict(t)
        row["path"] = str(p)
        if "target_density_deg2" not in row:
            row["target_density_deg2"] = target_density_deg2(
                cell_deg=float(row.get("cell_deg", 1.0)),
                stars_per_cell=int(row.get("stars_per_cell", 1)),
            )
        out.append(row)
    if not out:
        raise ValueError(f"no resolvable tiers in {path}")
    return out


def _tier_sort_key(tier: dict[str, Any], rho_img: float) -> float:
    td = float(tier.get("target_density_deg2", 1.0))
    return abs(math.log10(max(td, 1e-6)) - math.log10(max(rho_img, 1e-6)))


def order_tiers_for_image(
    tiers: list[dict[str, Any]],
    *,
    rho_img_deg2: float,
    plate_scale_arcsec_per_px: float | None = None,
) -> list[dict[str, Any]]:
    hint = density_hint_from_plate_scale(plate_scale_arcsec_per_px)
    rho = float(rho_img_deg2)
    if hint is not None:
        rho = 0.5 * rho + 0.5 * float(hint)
    ordered = sorted(tiers, key=lambda t: _tier_sort_key(t, rho))
    if plate_scale_arcsec_per_px is not None:
        try:
            _ps = float(plate_scale_arcsec_per_px)
            if math.isfinite(_ps) and _ps >= 5.0:
                wide = [t for t in ordered if str(t.get("name", "")).lower() == "wide"]
                rest = [t for t in ordered if str(t.get("name", "")).lower() != "wide"]
                ordered = wide + rest
            elif math.isfinite(_ps) and _ps <= 2.0:
                fine = [t for t in ordered if str(t.get("name", "")).lower() == "fine"]
                rest = [t for t in ordered if str(t.get("name", "")).lower() != "fine"]
                ordered = fine + rest
        except (TypeError, ValueError):
            pass
    return ordered


def peek_index_log_l3_max(index_path: Path | str) -> float | None:
    try:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        v = data.get("log_L3_max")
        return float(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def solve_blind_with_series(
    dao_df: pd.DataFrame,
    *,
    app_config: Any | None = None,
    plate_scale_arcsec_per_px: float | None = None,
    fov_deg: float | None = None,
    gaia_db_path: Path | str | None = None,
    naxis1: int | None = None,
    naxis2: int | None = None,
    pixel_pitch_um: float | None = None,
    focal_length_mm: float | None = None,
    max_cat_mag: float = 16.0,
    debug_sink: dict[str, Any] | None = None,
) -> tuple[float, float, str] | None:
    """Try blind tiers in scale-aware order; return first verified (ra, dec, tier_name)."""
    cfg = app_config or AppConfig()
    mode = str(getattr(cfg, "blind_index_select_mode", "auto") or "auto").strip().lower()
    if mode == "single":
        return _solve_single(
            dao_df,
            cfg=cfg,
            plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
            fov_deg=fov_deg,
            gaia_db_path=gaia_db_path,
            naxis1=naxis1,
            naxis2=naxis2,
            pixel_pitch_um=pixel_pitch_um,
            focal_length_mm=focal_length_mm,
            max_cat_mag=max_cat_mag,
            debug_sink=debug_sink,
        )

    manifest = getattr(cfg, "blind_index_series", None)
    if not manifest or not Path(str(manifest)).is_file():
        log_event("WARNING: blind_index_series missing — falling back to blind_index_path.")
        return _solve_single(
            dao_df,
            cfg=cfg,
            plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
            fov_deg=fov_deg,
            gaia_db_path=gaia_db_path,
            naxis1=naxis1,
            naxis2=naxis2,
            pixel_pitch_um=pixel_pitch_um,
            focal_length_mm=focal_length_mm,
            max_cat_mag=max_cat_mag,
            debug_sink=debug_sink,
        )

    tiers = load_series_manifest(manifest)
    budget = int(getattr(cfg, "blind_img_star_budget", 80))
    fov = float(fov_deg) if fov_deg is not None else 1.0
    rho = estimate_rho_img_deg2(
        plate_scale_arcsec_per_px=float(plate_scale_arcsec_per_px or 1.0),
        fov_deg=fov,
        img_budget=budget,
    )
    ordered = (
        list(tiers)
        if mode == "series_all"
        else order_tiers_for_image(
            tiers,
            rho_img_deg2=rho,
            plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
        )
    )
    if plate_scale_arcsec_per_px is not None:
        try:
            _ps = float(plate_scale_arcsec_per_px)
            if math.isfinite(_ps) and _ps >= 5.0:
                ordered = [t for t in ordered if str(t.get("name", "")).lower() == "wide"] or ordered
            elif math.isfinite(_ps) and _ps <= 2.5:
                ordered = [t for t in ordered if str(t.get("name", "")).lower() == "fine"] or ordered
        except (TypeError, ValueError):
            pass

    from vyvar_platesolver import _verify_blind_candidates

    gaia = Path(str(gaia_db_path or cfg.gaia_db_path)).expanduser()
    verify_on = bool(getattr(cfg, "blind_verify_enabled", True))

    for tier in ordered:
        idx_path = Path(str(tier["path"]))
        name = str(tier.get("name", idx_path.stem))
        if not idx_path.is_file():
            log_event(f"WARNING: blind tier {name} missing: {idx_path}")
            continue
        tier_cfg = cfg
        if plate_scale_arcsec_per_px is not None:
            try:
                _ps = float(plate_scale_arcsec_per_px)
                if math.isfinite(_ps) and _ps > 0:
                    _ref = 1.3
                    _tol = min(
                        20.0,
                        float(cfg.blind_verify_match_tol_px) * max(1.0, _ps / _ref),
                    )
                    _minm = int(cfg.blind_verify_min_matches)
                    if _ps >= 5.0:
                        _minm = max(6, _minm - 4)
                    tier_cfg = replace(
                        cfg,
                        blind_verify_match_tol_px=_tol,
                        blind_verify_min_matches=_minm,
                    )
            except (TypeError, ValueError):
                tier_cfg = cfg
        l3max = peek_index_log_l3_max(idx_path)
        if l3max is not None and mode == "auto":
            rho_t = estimate_rho_img_deg2(
                plate_scale_arcsec_per_px=float(plate_scale_arcsec_per_px or 1.0),
                fov_deg=fov,
                img_budget=budget,
                log_L3_max=l3max,
            )
            # Re-rank is expensive; only use per-tier l3 for logging
            if debug_sink is not None:
                debug_sink.setdefault("tier_rho_img", {})[name] = rho_t

        hint: tuple[float, float] | None = None
        if verify_on:
            cands = find_blind_candidates(
                dao_df,
                idx_path,
                n_top=30,
                top_n=int(getattr(tier_cfg, "blind_verify_top_n", 15)),
                plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
                fov_deg=fov_deg,
                app_config=tier_cfg,
                debug_sink=debug_sink,
            )
            if gaia.is_file() and cands:
                hint = _verify_blind_candidates(
                    cands,
                    dao_df=dao_df,
                    gaia_db_path=gaia,
                    fov_deg=fov,
                    naxis1=int(naxis1 or 0),
                    naxis2=int(naxis2 or 0),
                    pixel_pitch_um=pixel_pitch_um,
                    focal_length_mm=focal_length_mm,
                    max_cat_mag=float(max_cat_mag),
                    known_plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
                    app_config=tier_cfg,
                    debug_sink=debug_sink,
                )
            if hint is None:
                hint = find_blind_hint(
                    dao_df,
                    idx_path,
                    n_top=30,
                    min_votes=3,
                    plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
                    fov_deg=fov_deg,
                    app_config=tier_cfg,
                )
        else:
            hint = find_blind_hint(
                dao_df,
                idx_path,
                n_top=30,
                min_votes=3,
                plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
                fov_deg=fov_deg,
                app_config=tier_cfg,
            )
        if hint is not None:
            if debug_sink is not None:
                debug_sink["blind_series_winner"] = name
            log_event(f"INFO: Blind series winner tier={name} RA={hint[0]:.4f} Dec={hint[1]:.4f}")
            return float(hint[0]), float(hint[1]), name
    return None


def _solve_single(
    dao_df: pd.DataFrame,
    *,
    cfg: AppConfig,
    plate_scale_arcsec_per_px: float | None,
    fov_deg: float | None,
    gaia_db_path: Path | str | None,
    naxis1: int | None,
    naxis2: int | None,
    pixel_pitch_um: float | None,
    focal_length_mm: float | None,
    max_cat_mag: float,
    debug_sink: dict[str, Any] | None,
) -> tuple[float, float, str] | None:
    idx = getattr(cfg, "blind_index_path", None)
    if not idx or not Path(str(idx)).is_file():
        return None
    from vyvar_platesolver import _verify_blind_candidates

    gaia = Path(str(gaia_db_path or cfg.gaia_db_path)).expanduser()
    verify_on = bool(getattr(cfg, "blind_verify_enabled", True))
    hint: tuple[float, float] | None = None
    if verify_on:
        cands = find_blind_candidates(
            dao_df,
            idx,
            n_top=30,
            top_n=int(getattr(cfg, "blind_verify_top_n", 15)),
            plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
            fov_deg=fov_deg,
            app_config=cfg,
            debug_sink=debug_sink,
        )
        if gaia.is_file() and cands:
            hint = _verify_blind_candidates(
                cands,
                dao_df=dao_df,
                gaia_db_path=gaia,
                fov_deg=float(fov_deg or 1.0),
                naxis1=int(naxis1 or 0),
                naxis2=int(naxis2 or 0),
                pixel_pitch_um=pixel_pitch_um,
                focal_length_mm=focal_length_mm,
                max_cat_mag=float(max_cat_mag),
                known_plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
                app_config=cfg,
                debug_sink=debug_sink,
            )
        if hint is None:
            hint = find_blind_hint(
                dao_df,
                idx,
                n_top=30,
                min_votes=3,
                plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
                fov_deg=fov_deg,
                app_config=cfg,
            )
    else:
        hint = find_blind_hint(
            dao_df,
            idx,
            n_top=30,
            min_votes=3,
            plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
            fov_deg=fov_deg,
            app_config=cfg,
        )
    if hint is None:
        return None
    return float(hint[0]), float(hint[1]), "single"
