"""Pre-CQ-C comp_qa with iterative dropped_global locus (order-coupled baseline)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from comp_qa_core import (
    build_locus,
    comp_axis_mag,
    flag_reasons,
    load_proc_pivot,
    locus_at,
    loo_diff_series,
    robust_thr,
    sokolovsky_indices,
    worst_flagged_score,
)
def compute_comp_qa_iterative_locus(
    *,
    photometry_dir: Path,
    proc_dir: Path,
    mad_k: float = 4.0,
    min_comps: int = 3,
    max_comps: int = 8,
    _target_processing_order: list[str] | None = None,
) -> dict[str, Any]:
    """Legacy order-coupled locus rebuild (pre-2026-06-09 CQ-C)."""
    # Reuse target_data / pass1 setup from production entry (same inputs).
    import pandas as pd

    from comp_qa_core import _normalize_id, flux_to_mag

    phot = Path(photometry_dir).expanduser()
    proc_dir = Path(proc_dir).expanduser()
    comp_path = phot / "comparison_stars_per_target.csv"
    if not comp_path.is_file():
        raise FileNotFoundError(f"missing {comp_path}")

    comps = pd.read_csv(comp_path, dtype={"catalog_id": str, "target_catalog_id": str})
    comps["catalog_id"] = comps["catalog_id"].astype(str).str.strip()
    comps["target_catalog_id"] = comps["target_catalog_id"].astype(str).str.strip()
    comps["_catalog_id_n"] = comps["catalog_id"].map(_normalize_id)
    comps["_target_n"] = comps["target_catalog_id"].map(_normalize_id)

    target_data: dict[str, dict] = {}
    for tid, grp in comps.groupby("_target_n"):
        if not tid:
            continue
        comp_ids = sorted(grp["_catalog_id_n"].unique().tolist())
        comp_ids = [c for c in comp_ids if c]
        if len(comp_ids) < min_comps:
            continue
        all_ids = set(comp_ids) | {tid}
        flux_w, _time_df = load_proc_pivot(proc_dir, all_ids)
        if flux_w.empty or tid not in flux_w.columns:
            continue
        mag = {cid: flux_to_mag(flux_w[cid].values.astype(float)) for cid in flux_w.columns}
        flux_raw = {cid: flux_w[cid].values.astype(float) for cid in flux_w.columns}
        comp_rows = {row["_catalog_id_n"]: row for _, row in grp.iterrows()}
        pool = [c for c in comp_ids if c in mag]
        if len(pool) < min_comps:
            continue
        target_data[tid] = {
            "grp": grp,
            "pool": pool,
            "mag": mag,
            "flux_raw": flux_raw,
            "comp_rows": comp_rows,
        }

    pass1: list[tuple[str, str, float, float]] = []
    for tid, td in target_data.items():
        for cid in td["pool"]:
            m = loo_diff_series(td["mag"], cid, td["pool"])
            idx = sokolovsky_indices(m)
            row = td["comp_rows"].get(cid)
            imag = comp_axis_mag(td["flux_raw"].get(cid, np.array([])), row)
            pass1.append((tid, cid, imag, idx["sigma_iqr"]))

    if pass1:
        mags1 = np.array([p[2] for p in pass1])
        sig1 = np.array([p[3] for p in pass1])
        lc_c, lc_m, lc_s = build_locus(mags1, sig1)
    else:
        lc_c = lc_m = lc_s = np.array([])

    per_comp_rows: list[dict[str, Any]] = []
    per_target: dict[str, dict[str, Any]] = {}
    n_flag_total = 0
    n_flag_amp = n_flag_inv = n_flag_spike = 0
    n_flag_amp_inv = 0
    dropped_global: set[tuple[str, str]] = set()

    if _target_processing_order is not None:
        _extra = [t for t in target_data if t not in _target_processing_order]
        _target_items = [
            (t, target_data[t])
            for t in list(_target_processing_order) + _extra
            if t in target_data
        ]
    else:
        _target_items = list(target_data.items())

    for tid, td in _target_items:
        grp = td["grp"]
        surviving = list(td["pool"])

        while len(surviving) >= min_comps:
            metrics = {}
            for cid in surviving:
                m = loo_diff_series(td["mag"], cid, surviving)
                metrics[cid] = sokolovsky_indices(m)
                row = td["comp_rows"].get(cid)
                metrics[cid]["inst_mag"] = comp_axis_mag(td["flux_raw"].get(cid, np.array([])), row)

            locus_pts_m, locus_pts_s = [], []
            for t2, td2 in target_data.items():
                surv2 = [c for c in td2["pool"] if (t2, c) not in dropped_global]
                for cid in surv2:
                    m = loo_diff_series(td2["mag"], cid, surv2)
                    ix = sokolovsky_indices(m)
                    row = td2["comp_rows"].get(cid)
                    imag = comp_axis_mag(td2["flux_raw"].get(cid, np.array([])), row)
                    if math.isfinite(imag) and math.isfinite(ix["sigma_iqr"]):
                        locus_pts_m.append(imag)
                        locus_pts_s.append(ix["sigma_iqr"])
            if len(locus_pts_m) >= 3:
                lc_c, lc_m, lc_s = build_locus(np.asarray(locus_pts_m), np.asarray(locus_pts_s))
            elif lc_c.size == 0:
                lc_c, lc_m, lc_s = build_locus(mags1, sig1)

            thr_inv = robust_thr([metrics[c]["inv_nv"] for c in surviving], mad_k)
            flagged: dict[str, list[str]] = {}
            for cid in surviving:
                fr = flag_reasons(
                    metrics[cid]["sigma_iqr"],
                    metrics[cid]["inv_nv"],
                    metrics[cid]["spike"],
                    metrics[cid]["inst_mag"],
                    lc_c,
                    lc_m,
                    lc_s,
                    thr_inv,
                    mad_k,
                )
                if fr:
                    flagged[cid] = fr
            if not flagged:
                break
            worst_id = max(
                flagged,
                key=lambda cid: worst_flagged_score(
                    metrics[cid],
                    flagged[cid],
                    metrics[cid]["inst_mag"],
                    lc_c,
                    lc_m,
                    lc_s,
                    thr_inv,
                    mad_k,
                ),
            )
            dropped_global.add((tid, worst_id))
            surviving = [c for c in surviving if c != worst_id]

        surv_final = [c for c in td["pool"] if (tid, c) not in dropped_global]
        if len(surv_final) < min_comps:
            surv_final = list(td["pool"])

        locus_pts_m, locus_pts_s = [], []
        for t2, td2 in target_data.items():
            surv2 = [c for c in td2["pool"] if (t2, c) not in dropped_global]
            for cid in surv2:
                m = loo_diff_series(td2["mag"], cid, surv2)
                ix = sokolovsky_indices(m)
                row = td2["comp_rows"].get(cid)
                imag = comp_axis_mag(td2["flux_raw"].get(cid, np.array([])), row)
                if math.isfinite(imag) and math.isfinite(ix["sigma_iqr"]):
                    locus_pts_m.append(imag)
                    locus_pts_s.append(ix["sigma_iqr"])
        if len(locus_pts_m) >= 3:
            lc_c, lc_m, lc_s = build_locus(np.asarray(locus_pts_m), np.asarray(locus_pts_s))

        thr_inv_f = robust_thr(
            [
                sokolovsky_indices(loo_diff_series(td["mag"], c, surv_final))["inv_nv"]
                for c in surv_final
            ],
            mad_k,
        )

        n_flag_t = 0
        comp_payload: dict[str, dict[str, Any]] = {}
        for cid in td["pool"]:
            peers = surv_final
            m = loo_diff_series(td["mag"], cid, peers)
            idx = sokolovsky_indices(m)
            row = td["comp_rows"].get(cid)
            imag = comp_axis_mag(td["flux_raw"].get(cid, np.array([])), row)
            loc, spr = locus_at(imag, lc_c, lc_m, lc_s)
            flags = flag_reasons(
                idx["sigma_iqr"],
                idx["inv_nv"],
                idx["spike"],
                imag,
                lc_c,
                lc_m,
                lc_s,
                thr_inv_f,
                mad_k,
            )
            flagged = len(flags) > 0
            reason = "+".join(flags) if flagged else ""
            if flagged:
                n_flag_total += 1
                n_flag_t += 1
                if flags == ["amplitude"]:
                    n_flag_amp += 1
                elif flags == ["invNV"]:
                    n_flag_inv += 1
                elif flags == ["spike"]:
                    n_flag_spike += 1
                elif set(flags) == {"amplitude", "invNV"}:
                    n_flag_amp_inv += 1

            comp_payload[cid] = {
                "sigma_iqr": idx["sigma_iqr"],
                "inv_nv": idx["inv_nv"],
                "spike": idx["spike"],
                "qa_flag": flagged,
                "reason": reason,
                "inst_mag": imag,
                "locus_sigma_iqr": loc,
                "locus_spread": spr,
                "n_frames": idx["n"],
            }
            row_df = grp[grp["_catalog_id_n"] == cid]
            r0 = row_df.iloc[0] if not row_df.empty else {}
            per_comp_rows.append({
                "target_catalog_id": tid,
                "target_vsx_name": str(r0.get("target_vsx_name", "")),
                "catalog_id": cid,
                "inst_mag": imag,
                "sigma_iqr": idx["sigma_iqr"],
                "inv_nv": idx["inv_nv"],
                "spike": idx["spike"],
                "locus_sigma_iqr": loc,
                "locus_spread": spr,
                "n_frames": idx["n"],
                "thr_inv_nv": thr_inv_f,
                "FLAG": flagged,
                "flag_reason": reason,
            })

        n_clean = len(td["pool"]) - n_flag_t
        per_target[tid] = {
            "target_catalog_id": tid,
            "n_comps": len(td["pool"]),
            "n_flagged": n_flag_t,
            "n_clean": n_clean,
            "comps": comp_payload,
        }

    tgt_df = __import__("pandas").DataFrame(
        [
            {
                "target_catalog_id": v["target_catalog_id"],
                "n_comps": v["n_comps"],
                "n_flagged": v["n_flagged"],
                "n_clean": v["n_clean"],
            }
            for v in per_target.values()
        ]
    )
    mn = max(1, int(min_comps))
    mx = max(mn, int(max_comps))
    strong = min(mn + 2, mx)
    n_ge_strong = int((tgt_df["n_clean"] >= strong).sum()) if not tgt_df.empty else 0
    n_thin = (
        int(((tgt_df["n_clean"] >= mn) & (tgt_df["n_clean"] < strong)).sum())
        if not tgt_df.empty
        else 0
    )
    n_lt_min = int((tgt_df["n_clean"] < mn).sum()) if not tgt_df.empty else 0

    return {
        "per_target": per_target,
        "per_comp_rows": per_comp_rows,
        "stats": {
            "n_flagged": n_flag_total,
            "n_flag_amp": n_flag_amp,
            "n_flag_inv": n_flag_inv,
            "n_flag_spike": n_flag_spike,
            "n_flag_amp_inv": n_flag_amp_inv,
            "n_clean_ge_strong": n_ge_strong,
            "n_clean_thin": n_thin,
            "n_clean_lt_min": n_lt_min,
            "min_comps": mn,
            "strong_comps": strong,
        },
        "comp_csv_path": comp_path,
    }
