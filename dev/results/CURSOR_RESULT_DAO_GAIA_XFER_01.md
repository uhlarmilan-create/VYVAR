CURSOR RESULT - 2026-08-24 20:25 UTC+2

What I did
Pinned the STAGE-01 sandbox validation gate to
`ValidationParams.hand_validated()` (match 3.0 / pass2 2.0 / seed 2.0 /
p1=4.5 / p2=4.0). Draft-derived centroid tols stay production-scope on
the certificate and are never substituted into the 516 rescore.
Certificate write now fail-loud-stamps catalog fingerprints, sandbox
SHAs, hand CSV, and 516 lock-rig plate scale/FWHM. Informational 2x
tol-drift WARN does not block. Draft 520 preflight re-run: g/i/r
certificates PASS; z_90_4 remains solve-rejected. Photometry not
started (Milan not watching in-session). Not pushed.

## Premise (Rule 0.1)

**What is compared:** STAGE-01 iter4 match fractions on the **draft 516
WIDE CVn sandbox** (MASTERSTAR + Light_001 / 076 / 148), scored with
**hand_validated params**, versus the **2026-08-19 hand CSV** locked on
those same frames at seed/pass2 = 2.0 px.

**How they differ from the 520 fail:** REGRESS-01 scored that sandbox
with **Brno-derived** tols (1.0 px from FWHM 1.25 px). This task stops
that substitution. Production photometry of the current 520 set still
uses the derived tols; they are recorded as separate certificate
fields.

## Gates

| Gate | Status | Evidence |
|------|--------|----------|
| G-GO | PASS | Milan CURSOR TASK is the GO (Option 1 + identity stamps). |
| G0 | PASS | REGRESS-01 committed `d66613ce45c483d40f7b7a9cfcb44a7c787b0d62`. Declared dirt (AC-02 wiring) left unstaged. |
| G1 | PASS | `d66613c` is a descendant of `2926a95` (`git merge-base --is-ancestor` exit 0). XFER commit SHA below. |
| G2 | PASS | Production ePSF SHA `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` unchanged. 128 guarded 516 files (ePSF FITS+meta, aperture LCs, AAVSO, VarAstro) hash-identical before vs after. Extra 60 `lightcurve_*_psf.csv` on 516 are pre-existing AC-02 dirt, not this sandbox rescore. |
| G3 | PASS | `--fast` OVERALL PASS: pytest 1530 passed, 32 skipped (~11 min). |

## W1 - pin the gate

`src_py/dao_gaia_stage_validation.py` `run_validation_gate` scores only
`ValidationParams.hand_validated()`. `derived` / pass1 / pass2 /
`seed_snr_min` arguments are unused for scoring (signature kept for
callers). `MAX_REGRESSION_PP=0.005` unchanged. Missing hand CSV raises
`InvariantViolation("DAO-GAIA-IDENTITY")`. Fail-closed raise on
regression still `DAO-GAIA-CALIBRATION`.

## W2 - certificate identity stamps

`write_calibration_certificate` calls `build_certificate_identity_stamps`
and fails loud if any of `IDENTITY_STAMP_KEYS` is empty.

Mandatory fields on the live g_60_4 certificate:

| Field | Live g_60_4 value |
|-------|-------------------|
| `gaia_fingerprint` | `921ecb430eabd2f5d1c4815ea99bb08d2ee04734b8a45f66f60f0fe51126552d` |
| `vsx_fingerprint` | `13b4753f97c16a23f079026d9beab3eab0a1ebf3ea917f302e19e2f41b5086c5` |
| `sandbox.draft_id` | 516 |
| `sandbox.masterstar_sha256` | `13e77cf8a1dcb4e73fae0558437d7234feeb70a5ae4aa85064a8316812b01345` |
| `sandbox.light_001_sha256` | `2e085929a8a99b49254852377726db0fd48011f399dd84abb05af30ebedd0c18` |
| `sandbox.light_076_sha256` | `be4c7efcfcdd4a51ebbb3006bc795176b1e964763cddcb1c7390da4854f1f714` |
| `sandbox.light_148_sha256` | `d762f151c861fc524763e342bec1bae191c5f3f53eee2dc4bafbcc1d1e97698f` |
| `hand_csv.path` | `.../session_20260819_daostage01_iter4/final_scores.csv` |
| `hand_csv.sha256` | `450f620901aa4e5e7b17a79e3e7034aebd4ec29ca89683b4f9322e0aa2aa88b2` |
| `lock_rig.plate_scale_arcsec_per_px` | 9.773890066483727 |
| `lock_rig.fwhm_px` | 5.19465 |
| `derived_pass2_center_tol_px` | 1.0 (production-scope) |
| `derived_forced_seed_centroid_max_px` | 1.0 (production-scope) |
| `production_tolerances.scope` | `production_photometry_current_set` |
| `sandbox_params` | `hand_validated` |

## W3 - drift WARN

g_60_4 and i_70_4: `tol_drift_warn.status=WARN`, `blocks=false`.
pass2/seed derived 1.00 px vs hand 2.00 px (ratio 2.00x). Lock FWHM
5.19 px / 9.77 arcsec/px vs current-set FWHM 1.25 px / 0.56 arcsec/px.
r_60_4 derived 1.5 px is 1.33x vs 2.0: `status=OK`. Never blocked.

## W1/W5 pre/post sandbox scores (MASTERSTAR)

Same 516 frames. Pre-fix numbers from REGRESS-01 (derived 1.0 px into
the gate). Post-fix from the 520 g certificate (`sandbox_params=hand_validated`).

| MASTERSTAR | n_pass1 | n_pass2 | n_forced_seed | g1_strict_le145 |
|------------|--------:|--------:|--------------:|----------------:|
| hand CSV | 2636 | 217 | 163 | 0.9247889485801996 |
| pre-fix derived 1.0 px | 2636 | **168** | **109** | **0.9129** |
| post-fix hand params | 2636 | **217** | **162** | **0.9251726784343822** |

Lights post-fix match the CSV to all recorded digits (gate regressions
0.0). MASTERSTAR +0.038 pp vs CSV is the known Aug 20 rewrite
(162 vs 163 forced_seed), under `MAX_REGRESSION_PP=0.005`.
`max_regression_pp` on all three PASS certs = 6.4e-6 (g3_g18).

## W4 - tests

`dev/tests/test_dao_gaia_xfer_01.py`: 6 passed (6.10 s).

1. `test_derived_1px_tols_do_not_change_sandbox_score` -- derived 1.0 px
   into `run_validation_gate` still scores with hand 2.0/2.0/3.0.
2. CSV digits on disk + gate PASS when sandbox returns lock digits.
3. `test_certificate_identity_stamps_present` -- every W2 field.
4. Missing hand CSV and missing Gaia fingerprint raise
   `DAO-GAIA-IDENTITY`.

## W5 - draft 520 preflight (no photometry)

`astrometry_align_and_build_masterstar` on `Archive/Drafts/draft_000520`
(equipment 4, ram align). Wall 26.4 min (`elapsed_ms=1583523`).
Infolog `dev/results/session_20260824_dao_gaia_xfer_01/infolog_20260824_195115.txt`.

| set | solve | catalog_recovery_gate | cert | derived pass2/seed px | tol_drift | validation |
|-----|-------|----------------------:|------|----------------------:|-----------|------------|
| g_60_4 | VERIFIED | 97.0% | **PASS** | 1.0 / 1.0 | WARN 2.00x | PASS 6.4e-6 |
| i_70_4 | VERIFIED | 80.4% | **PASS** | 1.0 / 1.0 | WARN 2.00x | PASS 6.4e-6 |
| r_60_4 | VERIFIED | 87.2% | **PASS** | 1.5 / 1.5 | OK | PASS 6.4e-6 |
| z_90_4 | **rejected** | 14.2% | none | -- | -- | -- |

z_90_4 unchanged: match 19.5%, rms 6.72 px, n_matched_tight=16,
catalog_recovery_tight 9.5%. `masterstar_catalog_recovery_min` not
lowered. Pipeline: "3 setov OK, 1 preskocenych: z_90_4".

Photometry not started (Milan call).

g certificate copy:
`dev/results/session_20260824_dao_gaia_xfer_01/g_60_4_dao_gaia_calibration.json`
and `dev/results/context/session_20260824_dao_gaia_xfer_01/`.

## W6 - docs

- `docs/VYVAR_DECISIONS.md` -- H-GATE-XFER + gate-design principle
  (pin AND stamp params, rig scale/FWHM, catalog identity, input SHAs;
  same lesson class as CATALOG-PROVENANCE and the census pattern note).
- `docs/VYVAR_ROADMAP.md` -- Z-SOLVE-520-01 LOW; hand-CSV re-lock
  deferred to the next natural STAGE-01 iteration.
- `docs/VYVAR_JOURNAL.md` / `docs/VYVAR_STATE.md` -- one-liners.

## `--fast`

`python dev/scripts/session_baseline_check.py --fast` after the wire:
**OVERALL PASS** (pytest 1530 passed, 32 skipped; was 1524 at `2926a95`,
+6 XFER tests). Expected WARNs: untracked (AC-02 + Archive + this
session dir until commit), origin/main `b1af049`, db-quick-check waived,
ledger-todo, deps-outdated. Wall ~11 min.

## Files changed

**XFER-01 wire (this commit):**

- `src_py/dao_gaia_stage_validation.py`
- `src_py/dao_gaia_calibration.py`
- `dev/tests/test_dao_gaia_xfer_01.py`
- `dev/results/CURSOR_RESULT_DAO_GAIA_XFER_01.md` (this file)
- `dev/results/session_20260824_dao_gaia_xfer_01/`
- `dev/results/context/session_20260824_dao_gaia_xfer_01/`
- `docs/VYVAR_DECISIONS.md`, `docs/VYVAR_ROADMAP.md`,
  `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_STATE.md`
  (these four also still contain uncommitted AC-02 / census one-liners
  from earlier dirt; included because W6 requires the XFER entries)

**Not staged (declared dirt, unrelated):**

- AC-02 wiring: `config.json`, `src_py/pipeline.py`, `src_py/epsf_psf_merge.py`,
  `src_py/psf_*.py`, `src_py/photometry_core.py`, PSF tests, PARAMS/INVARIANTS,
  CONFIG guides.
- Extra 516 `lightcurve_*_psf.csv` (hash-guard excluded; not XFER).

## Errors (if any)

None that block the wire. Photometry of 520 g/i/r not run (W5 stop).
z_90_4 diagnosis deferred to Z-SOLVE-520-01.
