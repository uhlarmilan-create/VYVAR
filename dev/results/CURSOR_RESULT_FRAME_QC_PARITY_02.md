CURSOR RESULT - 2026-09-07 FRAME-QC-PARITY-02

Date: 2026-09-07. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 96c6a84.
Part A: ae7f757. Part B: this commit.
Class: Part A wire (log text only); Part B MEASUREMENT ONLY then STOP.
Live 516/517 read-only for Part B. No science change.

Premise (Rule 0.1): Layer A `_post_calibration_qc_eval` log said
REJECTED; Layer B (qc_metrics status allowlist) is the drop path.
Compared log wording to whether qc_passed False skips a write or
downstream frame. n_stars under study is qc_metrics
`n_stars_detected` (robust helper), not Layer A capped n_star
(max_stars=50).

## Refute check

A0 line numbers: `_post_calibration_qc_eval` pipeline_calibrate.py:1480;
no-finite log :1502-1504; HFR/stars/RMS logs :1544-1553; header stamp
:1950-1960; `fits.writeto` always :1998. No stop.

Literature (raised before any n_stars wiring; Part B wires nothing):
- AIJ: visual quarantine / filename-range filter; no automated
  n_stars frame gate (Collins et al. 2017; AIJ user guide).
- C-Munipack/Muniwin community practice (astrojolo): operator MAY
  exclude frames whose detected-star count differs by more than
  ~50% from the set. Frame 29 is 263 vs median 98 (~2.7x). Not an
  automated default in the package API; noted for Milan.
- VaST: optional automated bad-image removal (matching failure);
  `-6 --notremovebadimages` disables it. Not an n_stars MAD gate.
- photutils DAOStarFinder: source-level sharpness/roundness/peak
  filters, not a per-frame n_stars drop.

None of that blocks Part A log honesty. C-Munipack 50% heuristic is
contrary practice for a future drop-authority wire; Part B STOPS.

## Part A - A1 per-site consequence

| site | log | after return | drop? |
|------|-----|--------------|-------|
| :1502 no finite pixels | annotate | caller stamps VYQCPASS=False and `writeto` :1998 | (a) kept/written |
| :1547 single HFR reason | annotate | same | (a) kept/written |
| :1550 multi-reason | annotate | same | (a) kept/written |

`qc_passed` False only increments `stats["qc_rejected"]`
(:2576, :2680). Downstream drop is Layer B:
`filter_files_by_qc_metrics_allowlist` (pipeline_preprocess.py:108-124)
keeps `status == 'ok'` only. VYQCPASS is not that allowlist.

All three sites annotate-only. REJECTED retired on all three.

## Part A - A2 rewording

New text: `Frame <name> QC FAIL (diagnostic; frame kept): <reasons>`
Reasons payload unchanged (`no finite pixels`; `HFR: ...`;
`stars: ...`; `background RMS: ...`; `'; '.join(reasons)`).
No header/threshold/return-value change.

Word-boundary grep REJECTED in `dev/tests` + `src_py`: no test
asserted the old Frame-REJECTED log. Hits were IS_REJECTED /
SEED_REJECTED / decision titles. No test retarget.

ePSF-graph: no ePSF module name edited. `--full-epsf` skipped.

## Part A - A3/A4

D-FRAME-QC-AUTH-01 appended at top of `docs/VYVAR_DECISIONS.md`
(verbatim). FRAME-QC-PARITY row updated, still OPEN.

Commit A: `ae7f757` FRAME-QC-PARITY-02A: Layer A diagnostic log
wording (D-FRAME-QC-AUTH-01).

## Gates

G1 `--fast --clean` after Part A: OVERALL PASS (1621 passed, 32
skipped; clean-tree PASS). HEAD ae7f757.

G2 `--full` aperture-only: OVERALL PASS. era04_aperture
`d55fcc9d` n=53 / ext `cc8b532e` n=157 byte-identical
(full-photometry-sha-core-aperture / -ext-aperture PASS).
full-pipeline 1345s. No `--full-epsf`.

G4 live 516 after G2: PASS csv `bfa24039` / fits `13e77cf8` /
epsf `172f9540`. Not written.

## Part B - B1 n_stars (qc_metrics n_stars_detected)

Column = `_robust_frame_fwhm_median` `n_stars_detected`
(pipeline_calibrate.py:2884; escalating DAO 5.0/3.5/2.5).
516 and 517 tables are identical (same QC artifacts).

| draft | subset | n | median | MAD | sigma_MAD | min | max |
|-------|--------|---|--------|-----|-----------|-----|-----|
| 516/517 | all | 150 | 98.0 | 2.0 | 2.9652 | 82 | 263 |
| 516/517 | ok | 134 | 98.0 | 1.0 | 1.4826 | 90 | 263 |

Frame 29: n_stars=263, status=ok (both drafts).

Two-sided median +/- k*sigma_MAD (mirror of
`compute_auto_fwhm_limit` MAD math, two-sided):

ok subset (n=134), both drafts:

| k | n_low | n_high | high includes 029? |
|---|-------|--------|-------------------|
| 1.5 | 11 | 29 | yes |
| 2.5 | 7 | 20 | yes |
| 3.5 | 3 | 5 | yes |
| 5.0 | 1 | 3 | yes |

all rows (n=150): k=5.0 is 1 low (Light_058) + 1 high (Light_029).

Full frame lists: `n_stars_k_bounds.csv`.

## Part B - B2 live 516 check residual (read-only)

Target BO CVn `1498613634033133184`; check
`1497613731286514432` via `check_kmag_*.csv` kmag (134 normal
epochs). Sidecar has no per-comp epoch residuals
(pseudo-check comps: none).

Demeaned residual = kmag - median(kmag)=8.596979.

| quantity | Light_029 |
|----------|-----------|
| residual (mag) | +0.013052 |
| robust z vs other 133 | **1.865** |
| signed percentile | 92.5 |

Other frames with |z|>=3 on check residual: Light_034
(resid +0.0287, z=4.095).

residual_flatness_p99_adu (ok n=134): median 34.44, MAD 2.09,
sigma_MAD 3.10, min 29.47, max 58.55. Frame 29: 47.79,
robust z vs other 133 = **4.295** (sky-flatness outlier; not the
H1 reading quantity).

## Part B - pre-registered reading (verbatim)

|z_029(check)| = 1.865 < 2.0 -> **R-B1 (contained)**.

Architect wording (not executed here): recommend DIAGNOSTIC-ONLY
warning (damage already contained by identity gating per C8-2;
dropping would force an era04 recut for no measured photometric
benefit).

**STOP for Milan.** Decide (i) diagnostic vs drop, (ii) k,
(iii) recut timing. Nothing wired in this task.

## Evidence

`dev/results/context/session_20260907_frameqc2/`
- frame_qc_parity_02b_measure.py
- n_stars_stats.csv
- n_stars_k_bounds.csv
- check_residual_frame29.csv
- summary.json

## Files

Part A: src_py/pipeline_calibrate.py; docs/VYVAR_DECISIONS.md;
docs/VYVAR_ROADMAP.md.
Part B: this report + session_20260907_frameqc2/.

Push only `git push origin consolidate-01:consolidate-01`.
