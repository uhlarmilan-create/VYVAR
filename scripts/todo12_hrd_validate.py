"""Real-data HRD validation for TODO-12 arc (draft_425 B/V/R + draft_424)."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import AppConfig
from hrd_analysis import (
    _LABEL_BINARY,
    _select_stage1_candidates,
    _stage1_net_masks,
    annotate_field_image,
    build_hrd_dataframe,
    ensure_clean_field_background_png,
    field_annotation_pixel_scale,
    get_top_interesting_stars,
    hrd_parallax_params_from_cfg,
    plot_hrd_matplotlib,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp" / "todo12_hrd"
PRE12E = OUT / "pre12e"
RS_PER_CID = "458407464445792384"


def _alignment_check_rsg(
    label: str,
    hrd_df: pd.DataFrame,
    top: pd.DataFrame,
    ann_png: Path,
    platesolve_dir: Path,
    *,
    png_from_fits: bool,
) -> dict:
    """Crop ~100 px around brightest RSG; verify star peak vs local background."""
    empty = bool(top.get("_empty_field", False).any()) if not top.empty else True
    if empty:
        return {"skipped": "empty field table"}
    rsg = top[top["category"].astype(str).str.contains("Red supergiant", na=False)]
    if rsg.empty:
        return {"skipped": "no RSG row in table"}
    rsg = rsg.copy()
    rsg["_mag"] = pd.to_numeric(rsg.get("mag_g"), errors="coerce")
    row = rsg.sort_values("_mag", ascending=True, na_position="last").iloc[0]
    cid = str(row.get("catalog_id", ""))
    match = hrd_df[hrd_df["catalog_id"].astype(str) == cid]
    if match.empty:
        return {"skipped": f"catalog_id {cid} not in hrd_df"}
    hr = match.iloc[0]
    x_raw = float(hr["x"])
    y_raw = float(hr["y"])
    img = np.asarray(Image.open(str(ann_png)).convert("L"), dtype=np.float64)
    h, w = img.shape
    sx, sy, ok = field_annotation_pixel_scale(
        platesolve_dir, w, h, png_from_fits=png_from_fits
    )
    if not ok:
        return {"skipped": "annotation scale guard failed"}
    x = int(round(x_raw * sx))
    y = int(round(y_raw * sy))
    half = 50
    x0, x1 = max(0, x - half), min(w, x + half)
    y0, y1 = max(0, y - half), min(h, y + half)
    crop = img[y0:y1, x0:x1]
    crop_path = OUT / f"{label.replace(' ', '_')}_rsg_align_crop.png"
    Image.fromarray(crop.astype(np.uint8)).save(str(crop_path))
    yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
    cx, cy = x - x0, y - y0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner = crop[r <= 10]
    outer = crop[(r > 15) & (r <= 45)]
    if inner.size == 0 or outer.size == 0:
        return {"skipped": "crop too small", "crop": str(crop_path), "x": x, "y": y}
    peak_inner = float(np.max(inner))
    bg_med = float(np.median(outer))
    ratio = peak_inner / bg_med if bg_med > 0 else math.inf
    return {
        "catalog_id": cid,
        "x_px": x,
        "y_px": y,
        "x_raw": x_raw,
        "y_raw": y_raw,
        "scale_x": sx,
        "scale_y": sy,
        "peak_inner": peak_inner,
        "bg_median_outer": bg_med,
        "peak_bg_ratio": ratio,
        "aligned": ratio >= 2.0,
        "crop": str(crop_path),
    }


def _run_setup(label: str, ms_csv: Path, platesolve_dir: Path, gdb: Path, cfg: AppConfig, *, offline: bool) -> dict:
    run_cfg = AppConfig()
    for attr in (
        "hrd_online_enrich_enabled",
        "hrd_simbad_enrich_enabled",
        "hrd_enrich_max_candidates",
        "hrd_max_per_category",
        "hrd_min_per_net",
        "hrd_parallax_min_mas",
        "hrd_parallax_snr_min",
        "hrd_nss_category_enabled",
        "hrd_dsc_confirm_prob",
    ):
        setattr(run_cfg, attr, getattr(cfg, attr))
    if offline:
        run_cfg.hrd_online_enrich_enabled = False
        run_cfg.hrd_simbad_enrich_enabled = False
    pmin, psnr = hrd_parallax_params_from_cfg(run_cfg)
    hrd_df = build_hrd_dataframe(ms_csv, gdb, parallax_min_mas=pmin, parallax_snr_min=psnr)
    cache = platesolve_dir / "_hrd_cache" / "hrd_enrich.json"
    top = get_top_interesting_stars(hrd_df, cfg=run_cfg, cache_path=cache)
    obs = platesolve_dir.name
    png = OUT / f"{label.replace(' ', '_')}_hrd.png"
    plot_hrd_matplotlib(hrd_df, top, output_path=png, obs_group=obs)
    empty = bool(top.get("_empty_field", False).any()) if not top.empty else True
    enrich_src: list[str] = []
    if not empty and "src" in top.columns:
        enrich_src = [str(x) for x in top["src"].tolist()]
    cats = [] if empty else top["category"].tolist()
    simbad_ids = [] if empty else top.get("simbad_id", pd.Series(dtype=str)).tolist()
    otype_conflicts: list[str] = []
    for cat, sid in zip(cats, simbad_ids):
        if "Very cool" in cat and "s*r" in str(sid):
            otype_conflicts.append(f"{cat} | {sid}")
    ann_path: str | None = None
    align: dict = {"skipped": "not draft425 or empty"}
    plot_stars = top[~top.get("_empty_field", False).astype(bool)] if not empty else pd.DataFrame()
    if not plot_stars.empty:
        cache_dir = platesolve_dir / "_hrd_cache"
        phot_dir = platesolve_dir / "photometry"
        if not phot_dir.is_dir():
            phot_dir = platesolve_dir
        bg_png, from_fits = ensure_clean_field_background_png(
            platesolve_dir, phot_dir, cache_dir=cache_dir
        )
        if bg_png is not None:
            ann_out = OUT / f"{label.replace(' ', '_')}_field_annotated.png"
            ann = annotate_field_image(
                bg_png,
                plot_stars,
                hrd_df,
                platesolve_dir=platesolve_dir,
                output_path=ann_out,
                nss_category_enabled=bool(run_cfg.hrd_nss_category_enabled),
                png_from_fits=from_fits,
            )
            if ann is not None:
                ann_path = str(ann)
                if label.startswith("draft425"):
                    align = _alignment_check_rsg(
                        label,
                        hrd_df,
                        top,
                        ann,
                        platesolve_dir,
                        png_from_fits=from_fits,
                    )
    ident_tiers = {"confirmed": 0, "likely": 0, "candidate": 0}
    ident_rows: list[dict] = []
    if not empty and "ident" in top.columns:
        for _, tr in top.iterrows():
            tier = str(tr.get("ident", "candidate") or "candidate")
            ident_tiers[tier] = ident_tiers.get(tier, 0) + 1
            ident_rows.append(
                {
                    "catalog_id": str(tr.get("catalog_id", "")),
                    "category": str(tr.get("category", "")),
                    "ident": tier,
                    "logg_source": str(tr.get("logg_source", "")),
                }
            )
    rs_per: dict = {"catalog_id": RS_PER_CID}
    if not empty:
        hit = top[top["catalog_id"].astype(str) == RS_PER_CID]
        if not hit.empty:
            rs_per = {
                "catalog_id": RS_PER_CID,
                "category": str(hit.iloc[0].get("category", "")),
                "ident": str(hit.iloc[0].get("ident", "")),
                "logg_source": str(hit.iloc[0].get("logg_source", "")),
            }
        else:
            rs_per = {"catalog_id": RS_PER_CID, "in_table": False}
    dsc_wd_probs: list[float | None] = []
    if not offline and cache.is_file():
        try:
            raw = json.loads(cache.read_text(encoding="utf-8"))
            entries = raw.get("entries", raw)
            for tr in ident_rows:
                sid = tr["catalog_id"]
                p = entries.get(sid, {}).get("classprob_dsc_combmod_whitedwarf")
                if p is not None:
                    try:
                        dsc_wd_probs.append(float(p))
                    except (TypeError, ValueError):
                        pass
        except (OSError, json.JSONDecodeError):
            pass
    result: dict = {
        "label": label,
        "obs_group": obs,
        "hrd_rows": len(hrd_df),
        "reliable": int(hrd_df["hrd_reliable"].sum()) if not hrd_df.empty else 0,
        "candidates_table": 0 if empty else len(top),
        "empty_field": empty,
        "categories": cats,
        "binary_rows": sum(_LABEL_BINARY in str(c) for c in cats),
        "enrich_src": enrich_src,
        "simbad_otype_conflicts": otype_conflicts,
        "png": str(png),
        "field_annotated_png": ann_path,
        "rsg_alignment": align,
        "offline": offline,
        "hrd_nss_category_enabled": bool(run_cfg.hrd_nss_category_enabled),
        "ident_tiers": ident_tiers,
        "ident_rows": ident_rows,
        "rs_per_row": rs_per,
        "dsc_wd_probs_sample": dsc_wd_probs[:5],
    }
    if label.startswith("draft425") and not offline:
        from hrd_enrich import enrich_candidates  # noqa: PLC0415

        nss_on = bool(run_cfg.hrd_nss_category_enabled)
        nets = _stage1_net_masks(hrd_df, nss_enabled=nss_on)
        lum = hrd_df.loc[nets["luminous"]].copy()
        if not lum.empty:
            cand = _select_stage1_candidates(
                hrd_df,
                int(run_cfg.hrd_enrich_max_candidates),
                min_per_net=int(run_cfg.hrd_min_per_net),
                nss_enabled=nss_on,
            )
            cand_ids = set(cand["catalog_id"].astype(str))
            enriched = enrich_candidates(
                cand,
                cache,
                enabled=True,
                simbad_enabled=bool(run_cfg.hrd_simbad_enrich_enabled),
            )
            teff_by_id = {
                str(r["catalog_id"]): r.get("teff_gspphot")
                for _, r in enriched.iterrows()
            }
            result["luminous_net_sample"] = [
                {
                    "catalog_id": str(r.get("catalog_id", "")),
                    "abs_mag_g": float(r["abs_mag_g"]) if pd.notna(r["abs_mag_g"]) else None,
                    "bp_rp": float(r["bp_rp"]) if pd.notna(r["bp_rp"]) else None,
                    "teff_gspphot": teff_by_id.get(str(r.get("catalog_id", ""))),
                    "in_stage1_pick": str(r.get("catalog_id", "")) in cand_ids,
                }
                for _, r in lum.sort_values("abs_mag_g", ascending=True).head(8).iterrows()
            ]
    return result


def main() -> int:
    gdb = Path(AppConfig().gaia_db_path)
    if OUT.is_dir() and any(OUT.iterdir()) and not PRE12E.is_dir():
        PRE12E.mkdir(parents=True, exist_ok=True)
        for item in OUT.iterdir():
            if item.name.startswith("pre12"):
                continue
            dest = PRE12E / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    setups = [
        ("draft425_B", ROOT / "Archive/Drafts/draft_000425/platesolve/B_20_2"),
        ("draft425_V", ROOT / "Archive/Drafts/draft_000425/platesolve/V_20_2"),
        ("draft425_R", ROOT / "Archive/Drafts/draft_000425/platesolve/R_20_2"),
        ("draft424", ROOT / "Archive/Drafts/draft_000424/platesolve/NoFilter_60_2"),
    ]
    summary = []
    for label, ps in setups:
        ms = ps / "masterstars_full_match.csv"
        summary.append(_run_setup(label, ms, ps, gdb, cfg, offline=False))
    summary.append(
        _run_setup(
            "draft425_B_offline",
            setups[0][1] / "masterstars_full_match.csv",
            setups[0][1],
            gdb,
            AppConfig(),
            offline=True,
        )
    )
    out_json = OUT / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
