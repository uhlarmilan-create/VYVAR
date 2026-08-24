CURSOR RESULT - 2026-08-24 19:05 UTC+2

What I did
Measured which side of the draft 520 DAO-GAIA-CALIBRATION comparison
moved. Measurement only: no threshold tuning, no gate changes, no
photometry re-run, no catalog rollback, no re-lock, no code revert.
Rig: Brno AZ800 / C5A-150M. Draft 520 = SS Cam 2026-06-08, sets
g_60_4 / i_70_4 / r_60_4 / z_90_4. HEAD `2926a95` (+ dirty AC-02
unrelated to DAO/solve). Not pushed.

## Premise (Rule 0.1)

**What is compared:** STAGE-01 iter4 match fractions on the **draft 516
WIDE CVn sandbox** (MASTERSTAR + BO_CVn Light_001 / 076 / 148), scored
with **tolerances derived from each 520 Brno MASTERSTAR**, versus a
**2026-08-19 hand CSV** locked on those same 516 frames at
seed/pass2 = 2.0 px.

**How they differ:** the gate is not 520-frame vs 520-hand. Light_001 /
076 / 148 in the fail text are 516 sandbox labels. 520 g/i derive
centroid tols = 1.0 px (FWHM 1.25 px); the hand lock and the 516
ERA-03 PASS used 2.0-2.5 px (FWHM 5.2 px, plate scale 9.77 vs 0.56
arcsec/px).

## Named cause (split)

1. **g/i/r DAO-GAIA-CALIBRATION FAIL -- H-GATE-XFER (cross-rig
   derived-tol transfer).** Not H-KAT, not H-CODE, not H-DATA.
   Brno-derived `pass2_center_tol_px` / `forced_seed_centroid_max_px`
   = 1.0 (g, i) or 1.5 (r) are applied to the 516 WIDE sandbox; that
   cuts pass2/forced_seed on 516 and drops G1 by ~0.6-2.0 pp vs the
   516-era hand CSV. Preflight fail-loud was correct.

2. **z_90_4 MASTERSTAR solve FAIL -- separate.** Same Gaia DB, same
   `n_cat_in_frame=168` as g/i/r (which verified). z match_rate 19.5%,
   rms 6.72 px, catalog_recovery_tight 9.5%, gate 14.2%,
   n_matched_tight=16. Historical 2026-06-14 z ~34% rejected; r ~84%
   VERIFIED. Current r gate 87.2% still VERIFIED-class. Do not relax
   the solve gate. Not the same cause as (1).

H-KAT rejected for (1). H-CODE (today `dbb6967`..`2926a95` + dirty
AC-02) rejected for (1). H-DATA on 520 FITS rejected for (1) (those
FITS are not what the regression scores). D3 bisect skipped.

---

## D0 - evidence already on disk

### Preflight

`logs/run_preflight_error_20260824_164140.log` (copy in session dir).
Fail-closed raise:

- `src_py/dao_gaia_calibration.py:1182-1193`
  `write_calibration_certificate(..., fail_closed=True)` ->
  `InvariantViolation("DAO-GAIA-CALIBRATION", ...)`
- Called from `src_py/pipeline.py` `build_calibration_certificate` +
  write after MASTERSTAR (~13552-13583); outer abort
  `src_py/pipeline.py:16239` via `src_py/app.py:1006`.

Regression comparison (the 1 pp numbers):

- `src_py/dao_gaia_stage_validation.py:39-64` loads hand CSV
  `dev/results/context/session_20260819_daostage01_iter4/final_scores.csv`
- `run_validation_gate` (153-199) re-scores **iter4 sandbox frames**
  via `tmp/dao_gaia_stage_01_iter4.py` `score_validation_params`
  against **current** `AppConfig().gaia_db_path`, using
  **520-derived** `ValidationParams`, vs that CSV.
  `MAX_REGRESSION_PP = 0.005`.

`FRAMES` in `tmp/dao_gaia_stage_01.py:60-65` are hard-wired to
draft **516** `NoFilter_60_2`, not 520.

### Hand-lock era

| Item | Value |
|------|-------|
| Date | 2026-08-19 (iter4 result 13:45 UTC+2) |
| CSV | `session_20260819_daostage01_iter4/final_scores.csv` |
| Git SHA in CSV | **not recorded** |
| Catalog fingerprint in CSV | **not recorded** |
| Production fail-closed wire | `3aae487` 2026-08-20 ERA-03 A |
| Hand params | match 3.0 / pass2 2.0 / seed 2.0 / p1=4.5 / p2=4.0 |
| MASTERSTAR g1_strict_le145 | 0.9247889485801996 |

Same Gaia file as the 2026-07-29 CATALOG-PROVENANCE stamp
(`CURSOR_RESULT_session_close.md`): size 53,137,264,640,
rows 211,712,600, max_g 17.5, fp
`921ecb430eabd2f5d1c4815ea99bb08d2ee04734b8a45f66f60f0fe51126552d`.

Gate-design finding (not fixed here): the validation gate does **not**
check catalog fingerprint, plate scale, or FWHM before comparing
derived scores to the 516 hand CSV. The comparison is only meaningful
on the catalog **and the rig/FWHM** it was locked on.

### Previous PASSING Brno run of this observation (D0.2)

Same night/rig: SS Cam 2026-06-08, C5A-150M / AZ800 / bin4 3552x2664,
sets g/i/r/z. June 14 production path:

- draft_400 `g_60_4` passed solver (75.5% brightest-N); commit
  `70c23d0` (stale-hint cone recenter).
- `r_60_4` catalog-recovery **~84% VERIFIED**; `z_90_4` **~34%
  rejected** (`docs/VYVAR_STATE.md` Brno block; JOURNAL 2026-06-14).
- **DAO-GAIA-CALIBRATION did not exist** (wired 2026-08-19/20).
  "Previously PASSED" is solve+photometry, not this certificate gate.
- Drafts 399/400/402/407/426 **not on disk**. No pipeline_meta, no
  catalog fingerprint, no FITS SHA from that run.

2026-08-24 infolog (same 520 run): g/i/r **did solve** (verified):

| set | match | rms | rec_tight | rec_gate | n_cat | n_det |
|-----|------:|----:|----------:|---------:|------:|------:|
| g_60_4 | 100.0% | 1.44 px | 19.0% | **97.0%** | 168 | 33 |
| i_70_4 | 86.3% | 2.98 px | 24.4% | **80.4%** | 168 | 51 |
| r_60_4 | 94.9% | 1.49 px | 20.2% | **87.2%** | 168 | 39 |
| z_90_4 | 19.5% | 6.72 px | 9.5% | **14.2%** | 168 | 113 |

r gate 87.2% is the same metric class as historical ~84% VERIFIED.
g/i/r then failed the **new** calibration certificate, so the
pipeline treated every set as MASTERSTAR-fail.

Hand-params recompute **inside the 520 g certificate**
(`validation.hand_scores.recomputed`): Light_001/076/148 match the
hand CSV to all recorded digits; MASTERSTAR g1_strict_le145 =
0.92517 vs 0.92479 (**+0.038 pp**, 162 vs 163 forced_seed). Current
code + current Gaia + current 516 lights still reproduce the lock.

---

## D1 - input identity (H-DATA)

Gate-named frames are **516 sandbox**, not 520 lights.

516 sandbox SHA256:

| frame | live 516 | era03 2026-08-20 | cleanrebuild 2026-08-18 |
|-------|----------|------------------|-------------------------|
| MASTERSTAR | 13e77cf8... | identical | **different** 104700dd... (pre-ERA-03) |
| Light_001/076/148 | 2e085929 / be4c7efc / d762f151 | identical | identical |

Lights byte-identical since 2026-08-17 (before hand lock). MASTERSTAR
rewritten 2026-08-20 07:10 (after lock); that explains the 1 forced_seed
on MS hand-recompute, not the ~1 pp derived drop (lights also drop).

520 inputs (pre_calibrated `non_calibrated/lights`, 25 FITS/set):

| set | MASTERSTAR sha256 | DATE-OBS | naxis | n_lights |
|-----|-------------------|----------|-------|----------|
| g_60_4 | 3ab82d72... | 2026-06-08T20:02:41 | 3552x2664 | 25 |
| i_70_4 | 9b05a45d... | 2026-06-08T20:05:04 | 3552x2664 | 25 |
| r_60_4 | 55e41a2d... | 2026-06-08T20:03:52 | 3552x2664 | 25 |
| z_90_4 | da5c86a9... | 2026-06-08T20:06:25 | 3552x2664 | 25 |

First g light `SSCam_2026-06-08_20-03-48_g_0000.fits` sha256
`b93ea21f8011b3dc89519d842ceb02e2c7f47e3f62a4aecb9d59576d700d4252`.
No historical SHA from draft_400. **Byte-level identity vs the June
14 passing run cannot be established (provenance gap).** Header
identity: same night, same camera/telescope, same dimensions,
DATE-OBS 2026-06-08. 520 has no Light_001/076/148 names.

H-DATA cannot explain the 1 pp gate fail (gate does not score 520
FITS). 520 FITS remain relevant only for z solve (D4).

---

## D2 - catalog identity (H-KAT)

NOW (cheap fingerprint = sha256(size + first 1 MiB + last 1 MiB)):

| DB | size | mtime UTC | fingerprint |
|----|-----:|-----------|-------------|
| `GAIA_DR3/vyvar_gaia_dr3.db` | 53,137,264,640 | 2026-06-14T23:05:57 | `921ecb430eabd2f5d1c4815ea99bb08d2ee04734b8a45f66f60f0fe51126552d` |
| `VSX/vyvar_vsx_local_v2.db` | 908,324,864 | 2026-06-03T07:02:13 | `13b4753f97c16a23f079026d9beab3eab0a1ebf3ea917f302e19e2f41b5086c5` |
| zaloha Gaia | 10,066,063,360 | 2026-04-28 | `9dd58788ceec35c54609177e19c2c5c1ebebdad95f0830cc9dc4e701c1e14601` |

Live Gaia+VSX **identical** to 2026-07-29 session_close stamp (size,
rows 211,712,600 recorded 2026-08-21 EPSF-DB-01, fp). No WAL. File
mtime June 14/June 3 -- the in-progress DR3 rebuild did **not**
rewrite the DB the gate reads.

Hand-lock CSV has no fingerprint; identity to lock era is inferred
from (a) same path/size/mtime/fp as July 29, (b) hand-params
recompute matching the CSV.

**Verdict vs hand-lock era: identical. Vs zaloha: changed** (G<=16
10 GB vs G<=17.5 53 GB).

Cone r=0.40 deg (covers chip):

| footprint | live n / G<=14.5 / G<=16 / G<=17.5 | zaloha n / G<=14.5 / G<=16 |
|-----------|-------------------------------------|----------------------------|
| 520 SS Cam 109.11 +73.33 | 365 / **66** / 175 / 365 | 175 / **66** / 175 |
| 516 CVn 209.50 +41.19 | 235 / **39** / 116 / 235 | 116 / **39** / 116 |

G<=14.5 row counts (the gated metric) are **identical** live vs
zaloha in both fields. A ~1 pp G1_le145 drop is not a catalog
row-set change.

Gate-design finding: no catalog-identity check before hand compare
(`dao_gaia_stage_validation.py`). Not fixed here.

---

## D3 - code bisect (skipped)

D1 (gate inputs / 516 lights) identical for the scored lights; D2
identical; hand-params recompute matches CSV on current tip. Today's
series (`dbb6967`..`2926a95`) does not touch
`dao_gaia_calibration.py` / `dao_gaia_stage_validation.py` /
`vyvar_platesolver.py`; `6fd1452` adds 3 lines to `pipeline.py` (PSF
`x_fit`/`y_fit` persist). Dirty AC-02 does not diff those DAO/solve
files. No bisect.

---

## D4 - z_90_4 solve context

Read-only. z MASTERSTAR has **no WCS** (solve rejected; file is the
copied candidate `SSCam_2026-06-08_20-08-02_z_0003.fits`). Cannot
re-project Gaia onto z without a solve, so recovery vs zaloha was
not recomputed as a WCS match. Infolog already has the live-catalog
numbers (same `n_cat_in_frame=168` on all four sets).

Pixel contrast (MASTERSTAR, memmap=False):

| set | median | std | p99-med | max | WCS |
|-----|-------:|----:|--------:|----:|-----|
| g | 35747 | 86.4 | 40.0 | 88782 | yes |
| i | 33634 | 27.5 | 15.9 | 48939 | yes |
| r | 33759 | 63.2 | 16.7 | 71746 | yes |
| z | 33349 | 9.3 | 9.1 | 37431 | no |

z is the shallowest ADU contrast. n_det=113 with only 16 tight
matches and rms 6.72 px is a bad WCS / false-detection pattern, not
a missing-catalog pattern (g verified at the same n_cat=168).

**34% -> 9.5%:** historical 34% is the same *class* of reject (z
never VERIFIED). Current **gate** fraction is 14.2%
(`n_tight/min(n_cat,n_det)=16/113`). Historical ~34% was likely that
gate fraction (STATE "catalog recovery tight"). Same live catalog as
g (97% gate) so this is **not** H-KAT and **not** cause (1). Separate
z-band solve/data-quality effect. Solve gate not relaxed.

INV-PREP-01 on z: `large_small_ratio=0.06x` (warn>10) -- informational,
below warn.

---

## Mechanism (g/i/r 1 pp)

`derive_tolerances_from_diagnostic` (`dao_gaia_calibration.py:728-763`):
centroid = round_0.5(clamp(seed_p95, floor=1.0, cap=3.0)).

520 g seed p95 = 1.11 px -> **1.0 px**. 516 WIDE ERA-03 PASS used
seed p95 = 2.46 -> **2.5 px**. Hand lock used **2.0 px** by design.

Scoring 516 FWHM~5.2 px detections with a 1.0 px pass2/seed cap:

| MASTERSTAR | n_pass1 | n_pass2 | n_forced_seed | g1_strict_le145 |
|------------|--------:|--------:|--------------:|----------------:|
| hand CSV / hand recompute | 2636 | 217 | 163 / 162 | 0.9248 / 0.9252 |
| g_60_4 derived (1.0 px) | 2636 | **168** | **109** | **0.9129** |
| r_60_4 derived (1.5 px) | 2636 | 200 | 138 | 0.9202 (MS PASS; only Light_001 0.58 pp) |

g and i MASTERSTAR/Light_076/Light_148 scores match to 4 decimals
because both used seed/pass2 = 1.0; Light_001 differs slightly
(match_radius 3.5 vs 4.0). That is the "global 1 pp" signature:
one sandbox, two filters, same tight tols -- not a Gaia rewrite.

---

## Proposed fix task scope (not executed)

Working title: **DAO-GAIA-XFER-01** (Milan GO). Options:

1. **Pin the STAGE-01 validation gate to `ValidationParams.hand_validated()`**
   (do not substitute 520-derived centroid tols into the 516 sandbox
   rescore). Keep derived tols for **production photometry on that
   set only**. Stamp catalog fingerprint + plate_scale + FWHM on the
   certificate. This unblocks g/i/r 520 without re-lock and without
   touching the z solve gate.

2. **Refuse the sandbox gate when plate_scale or FWHM differs from the
   516 lock** (e.g. scale ratio > 2x: 9.77 vs 0.56 arcsec/px). Skip
   or PASS-diagnostic; still write derived tols for the Brno set.

3. **Re-lock hand references on a Brno sandbox** WITH catalog
   fingerprint + rig/FWHM stamp -- only if Milan wants a Brno-specific
   completeness gate. Do not re-lock the 516 CSV onto Brno tols.

4. **Catalog rollback:** not indicated for (1). Live DB matches lock.
5. **Code revert of `dbb6967`..`2926a95` / AC-02:** not indicated for (1).

z_90_4 follow-up (optional, separate): keep reject; diagnose why
n_tight=16 / rms=6.72 vs g rms=1.44 on the same catalog. Do not
lower `masterstar_catalog_recovery_min`.

---

## `--fast`

`python dev/scripts/session_baseline_check.py --fast` at HEAD
`2926a95`: **OVERALL PASS** (pytest 1524 passed, 32 skipped). Expected
WARNs: untracked, origin/main `b1af049`, db-quick-check waived,
ledger-todo, deps-outdated. No src_py edits in this task.

## Files changed

Measurement artifacts only (no src_py / config / gate edits):

- `dev/results/CURSOR_RESULT_DAO_GAIA_REGRESS_01.md` (this file)
- `dev/results/session_20260824_dao_gaia_regress_01/d0_scores_and_solve_qa.json`
- `dev/results/session_20260824_dao_gaia_regress_01/d1_d4_fits_identity.json`
- `dev/results/session_20260824_dao_gaia_regress_01/d2_catalog_identity.json`
- `dev/results/session_20260824_dao_gaia_regress_01/run_preflight_error_20260824_164140.log`
- `dev/results/session_20260824_dao_gaia_regress_01/hand_final_scores.csv`
- `dev/results/session_20260824_dao_gaia_regress_01/hand_final_config.json`

## Errors (if any)

None that block the diagnosis. Whole-table `COUNT(*)` on the 53 GB
Gaia DB was not re-run (size+fp match July 29; 211,712,600 from
2026-08-21). z recovery vs zaloha not re-projected (no z WCS).
Historical 520-input SHA256 absent (drafts 400/402 gone).
