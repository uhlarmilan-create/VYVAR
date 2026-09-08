# EPSF-XVAL-A2 - Milan Linux run (five steps)

No other interaction. Kit is `dev/xval_psfex/`. Snapshot is
`Archive/Drafts/draft_000516_snapshot_era04_20260826` (or a copy of
that tree). Output is a new empty directory you choose.

1. `git pull` on `consolidate-01` (this kit).
2. One apt line:
   `sudo apt install source-extractor psfex python3-astropy python3-pandas python3-numpy`
3. Copy the snapshot directory onto the Linux machine (the era04
   freeze, or the C-EXPORT-GAP sandbox layout: aligned lights under
   `detrended_aligned/lights/<setup>/` plus `calibrated/lights/qc_metrics.csv`).
4. From the repo:
   `bash dev/xval_psfex/run_all.sh /path/to/snapshot /path/to/out`
5. Copy `/path/to/out` back to the Windows tree (for example
   `dev/results/context/session_20260907_epsfxval_a2/linux_out/`)
   and/or `git push origin consolidate-01:consolidate-01`.
   Do not push `main`.

`run_all.sh` fails loud if `source-extractor|sextractor|sex` or
`psfex` is missing (prints the apt line). Per-frame `SEEING_FWHM`
is `qc_metrics.fwhm_px * WCS plate scale`. Do not edit the param
files. Degree-3 sensitivity lands in `OUT_DIR/deg3/`.
