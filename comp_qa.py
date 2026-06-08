#!/usr/bin/env python3
"""Comp-star LOO QA CLI — delegates to comp_qa_core (same math as pipeline stage)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from comp_qa_core import compute_comp_qa


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Comp-star LOO QA (Sokolovsky indices + mag locus)")
    ap.add_argument("--vyvar-photometry-dir", type=Path, required=True)
    ap.add_argument("--proc-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("./tmp/xval_out"))
    ap.add_argument("--mad-k", type=float, default=4.0)
    ap.add_argument("--min-comps", type=int, default=3)
    args = ap.parse_args()

    phot = args.vyvar_photometry_dir.expanduser()
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = compute_comp_qa(
        photometry_dir=phot,
        proc_dir=args.proc_dir.expanduser(),
        mad_k=float(args.mad_k),
        min_comps=int(args.min_comps),
    )

    per_df = pd.DataFrame(result["per_comp_rows"])
    tgt_df = pd.DataFrame(
        [
            {
                "target_catalog_id": v["target_catalog_id"],
                "n_comps": v["n_comps"],
                "n_flagged": v["n_flagged"],
                "n_clean": v["n_clean"],
            }
            for v in result["per_target"].values()
        ]
    )
    summ_path = phot / "photometry_summary.csv"
    if summ_path.is_file() and not tgt_df.empty:
        summ = pd.read_csv(summ_path, dtype={"catalog_id": str})
        summ["catalog_id"] = summ["catalog_id"].astype(str).str.strip()
        lc_map = dict(zip(tgt_df["target_catalog_id"], tgt_df["n_clean"]))
        tgt_df = tgt_df.merge(
            summ[["catalog_id", "lc_rms"]].rename(columns={"catalog_id": "target_catalog_id"}),
            on="target_catalog_id",
            how="left",
        )

    per_path = out_dir / "comp_qa_per_comp.csv"
    tgt_path = out_dir / "comp_qa_targets.csv"
    per_df.to_csv(per_path, index=False)
    tgt_df.to_csv(tgt_path, index=False)

    st = result["stats"]
    log(f"\nWrote {per_path} ({len(per_df)} rows)")
    log(f"Wrote {tgt_path} ({len(tgt_df)} targets)")
    log(f"\nflagged comps total: {st['n_flagged']}")
    log(f"  by amplitude only: {st['n_flag_amp']}")
    log(f"  by invNV only: {st['n_flag_inv']}")
    log(f"  by spike only: {st['n_flag_spike']}")
    log(f"  amp+invNV: {st.get('n_flag_amp_inv', 0)}")
    log(
        f"\nn_clean buckets (targets): >=5 comps: {st['n_clean_ge5']}  |  "
        f"3-4: {st['n_clean_3_4']}  |  <3: {st['n_clean_lt3']}"
    )
    if st["n_flagged"] and not per_df.empty:
        log("\nFLAGGED (sample):")
        log(per_df[per_df["FLAG"]].head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
