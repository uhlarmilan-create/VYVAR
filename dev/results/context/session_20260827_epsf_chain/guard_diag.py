# -*- coding: ascii -*-
"""Diagnose INV-EXPORT-READ-ONLY-01 on freeze proc CSVs. Writes only under tmp."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src_py"))

from config import AppConfig  # noqa: E402
from epsf_frame_accounting import list_epsf_science_light_fits  # noqa: E402
from epsf_psf_merge import (  # noqa: E402
    guarded_psf_sidecar_write,
    hash_non_psf_columns,
    merge_psf_into_sidecar,
    non_psf_columns,
)
from pipeline import _epsf_fit_catalog_ids, _export_catalog_psf_st_fields, _fill_psf_catalog_columns  # noqa: E402
from proc_frame_store import proc_csv_path_for_aligned_fits  # noqa: E402
from astropy.io import fits  # noqa: E402
import numpy as np  # noqa: E402

SNAP = REPO / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
# Use work-copy ePSF if present, else we cannot fit; diagnosis of hash uses fill+write.
WORK = REPO / "tmp" / "epsf_chain_m2_era04"
OUT = Path(__file__).resolve().parent


def _col_hash(df: pd.DataFrame, col: str) -> str:
    import hashlib
    import math

    h = hashlib.sha256()
    s = df[col]
    n = int(len(s))
    num = pd.to_numeric(s, errors="coerce")
    n_num = int(num.notna().sum())
    use_num = n_num >= max(1, (n + 1) // 2) if n else False
    h.update(b"N" if use_num else b"S")
    if use_num:
        for v in num.tolist():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                h.update(b"*")
                continue
            if not math.isfinite(fv):
                h.update(b"nan")
            else:
                h.update(f"{fv:.10g}".encode("ascii"))
    else:
        for v in s.tolist():
            h.update(str("" if v is None else v).encode("utf-8"))
    return h.hexdigest()[:16], use_num, n_num


def main() -> None:
    ps = WORK / "platesolve" / SETUP
    lights_src = SNAP / "detrended_aligned" / "lights" / SETUP
    files = list_epsf_science_light_fits(lights_src)
    print("n_files", len(files), "frame0", files[0].name, "frame3", files[3].name)
    cfg = AppConfig()
    cfg.psf_photometry_enabled = True
    cfg.photometry_mode = "both"
    target_ids = _epsf_fit_catalog_ids(ps, psf_photometry_enabled=True)
    st_base = _export_catalog_psf_st_fields(cfg, ps)
    st_base["_run_epsf"] = True
    st_base["_psf_merge_only"] = True
    # Diagnose write/read on frame index 3 (4th file) using a temp copy of freeze sidecar.
    fp = files[3]
    tmp = REPO / "tmp" / "epsf_chain_guard_diag"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    sidecar_src = proc_csv_path_for_aligned_fits(fp)
    sidecar = tmp / sidecar_src.name
    shutil.copy2(sidecar_src, sidecar)
    fits_copy = tmp / fp.name
    shutil.copy2(fp, fits_copy)
    before = pd.read_csv(sidecar, low_memory=False)
    st = dict(st_base)
    st["epsf_frame_name"] = fp.name
    with fits.open(fits_copy, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()
    after = _fill_psf_catalog_columns(before.copy(), data, hdr, st, target_ids=target_ids)
    from epsf_psf_merge import assert_inv_psf_additive_01

    assert_inv_psf_additive_01(before, after, frame_name=fp.name)
    print("additive in-memory OK")
    pre = {c: _col_hash(before, c) for c in non_psf_columns(before.columns)}
    try:
        guarded_psf_sidecar_write(sidecar, after, before)
        print("guarded write OK")
        rec = {"status": "ok", "file": fp.name}
    except Exception as exc:
        print("guarded write FAIL", exc)
        on_disk = pd.read_csv(sidecar, low_memory=False)
        # sidecar restored, so on_disk == before. Need to write without guard to compare.
        from pipeline import _vyvar_df_to_csv

        written = tmp / "after_write.csv"
        _vyvar_df_to_csv(after, written)
        on_disk = pd.read_csv(written, low_memory=False)
        post = {c: _col_hash(on_disk, c) for c in non_psf_columns(on_disk.columns)}
        diffs = []
        for c in sorted(set(pre) | set(post)):
            if pre.get(c) != post.get(c):
                diffs.append(
                    {
                        "col": c,
                        "before": pre.get(c),
                        "after": post.get(c),
                        "in_before": c in before.columns,
                        "in_after_disk": c in on_disk.columns,
                    }
                )
        extra_cols = [c for c in on_disk.columns if c not in before.columns]
        rec = {
            "status": "fail",
            "file": fp.name,
            "n_diff_cols": len(diffs),
            "diffs": diffs[:40],
            "extra_cols": extra_cols,
            "n_before_cols": int(len(before.columns)),
            "n_disk_cols": int(len(on_disk.columns)),
        }
        print("n_diff_cols", len(diffs))
        for d in diffs[:20]:
            print(" ", d)
    (OUT / "guard_diag.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
