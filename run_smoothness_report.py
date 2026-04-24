from __future__ import annotations

from pathlib import Path

import pandas as pd

from variability_detector import _norm_cid, compute_rms_variability, load_field_flux_matrix


def main() -> None:
    draft = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000249")
    obs = "NoFilter_60_2"
    per = draft / "detrended_aligned" / "lights" / obs
    ps = draft / "platesolve" / obs

    comp = pd.read_csv(ps / "comparison_stars.csv", low_memory=False)
    comp_ids = [_norm_cid(x) for x in comp["catalog_id"].astype(str).tolist()]

    fm, meta, _ = load_field_flux_matrix(per, flux_col="dao_flux", min_frames_frac=0.5)
    res = compute_rms_variability(
        fm,
        meta,
        comp_ids,
        sigma_threshold=3.0,
        vsx_targets_csv=ps / "variable_targets.csv",
    )

    cand = res[(res["is_variable_candidate"] == True) & (~res["vsx_known_variable"].fillna(False))].copy()  # noqa: E712
    print("candidates_new", len(cand))
    print("\nTop5 smoothness_ratio:")
    print(
        cand[
            ["catalog_id", "mag", "rms_pct", "smoothness_ratio", "variability_score"]
        ]
        .head(5)
        .to_string(index=False)
    )

    ids = [1497974233661289000, 1485338783474459600, 1497236186481686000]
    for cid in ids:
        row = res[res["catalog_id"].astype(str) == str(cid)]
        if row.empty:
            print("\n", cid, "not found")
            continue
        r = row.iloc[0]
        print(
            f"\n{cid}: cand={bool(r.get('is_variable_candidate'))} "
            f"vsx_known={bool(r.get('vsx_known_variable'))} vsx_match={bool(r.get('vsx_match'))} "
            f"rms={float(pd.to_numeric(r.get('rms_pct'), errors='coerce')):.2f}% "
            f"smooth={float(pd.to_numeric(r.get('smoothness_ratio'), errors='coerce')):.2f}"
        )


if __name__ == "__main__":
    main()

