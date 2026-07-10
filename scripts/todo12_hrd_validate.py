"""Real-data HRD validation for TODO-12 arc (draft_425 B/V/R + draft_424)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from config import AppConfig
from hrd_analysis import (
    _select_stage1_candidates,
    _stage1_net_masks,
    build_hrd_dataframe,
    get_top_interesting_stars,
    hrd_parallax_params_from_cfg,
    plot_hrd_matplotlib,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp" / "todo12_hrd"
PRE12C = OUT / "pre12c"


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
    result: dict = {
        "label": label,
        "obs_group": obs,
        "hrd_rows": len(hrd_df),
        "reliable": int(hrd_df["hrd_reliable"].sum()) if not hrd_df.empty else 0,
        "candidates_table": 0 if empty else len(top),
        "empty_field": empty,
        "categories": cats,
        "enrich_src": enrich_src,
        "simbad_otype_conflicts": otype_conflicts,
        "png": str(png),
        "offline": offline,
    }
    if label.startswith("draft425") and not offline:
        from hrd_enrich import enrich_candidates  # noqa: PLC0415

        nets = _stage1_net_masks(hrd_df)
        lum = hrd_df.loc[nets["luminous"]].copy()
        if not lum.empty:
            cand = _select_stage1_candidates(
                hrd_df,
                int(run_cfg.hrd_enrich_max_candidates),
                min_per_net=int(run_cfg.hrd_min_per_net),
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
    if OUT.is_dir() and any(OUT.iterdir()) and not PRE12C.is_dir():
        PRE12C.mkdir(parents=True, exist_ok=True)
        for item in OUT.iterdir():
            if item.name in ("pre12b", "pre12c"):
                continue
            dest = PRE12C / item.name
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
