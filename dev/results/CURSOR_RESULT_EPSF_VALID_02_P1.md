CURSOR RESULT - 2026-08-22 (EPSF-VALID-02 Phase 1)

What I did
Implemented G0 (db-quick-check waiver + docs), verified G1 PASS, launched G2 (`--full` recut,
still running at report time). Executed Phase 1 measurement sandbox on drafts 516 and 517.
No src_py science changes beyond G0 gate script.

## Prerequisite gate (rev 2)

| Gate | Status | Evidence |
|------|--------|----------|
| G0 waiver + docs | **PASS** | Commit `0e1a484`; `dev/validation/db_quick_check_waiver.json` |
| G1 `--fast` OVERALL | **PASS** | db-quick-check **WARN** (waived); pytest green |
| G2 `--full` recut | **PASS** | @ `0e1a484`: core **9902d918** n=121, extended **472bc9e4** n=179; science 60/0; OVERALL PASS (2097 s pipeline) |

---

## P1-A - 516 coverage collapse (E2)

### Infolog grep
Draft 516 infolog: `infolog_20260817_222127.txt` only (pre-ePSF). **Zero** lines matching
`[ePSF] per-frame PSF failed (non-fatal)` or `[ePSF] fitting`. UI RUN ePSF session
(2026-08-21 meta `created_utc`) did not write ePSF lines to draft infolog.

### Per-frame proc table (measurement set = `_epsf_lc_catalog_ids`, n=2505)
Artifact: `dev/results/context/session_20260822_epsf_valid_02_p1/p1a_per_frame_table.csv`

| Metric | Value |
|--------|-------|
| Total frames | 134 |
| Frames with partial PSF ok (0 < n_ok < n_fit) | 35 (idx 0-34) |
| Frames with zero PSF ok | 99 (idx 35-133) |
| Last frame with any ok | idx **34** (`BO_CVn_Light_038`), n_ok=907 / n_fit=2410 |
| First frame with zero ok | idx **35** (`BO_CVn_Light_039`), n_ok=0 / n_fit=2331 |

Example target `1498486880958321024` (CSS_J140918.7+423422): **35/134** psf_fit_ok
(artifact `p1a_target_1498486880958321024.csv`).

### Column signature (decisive)
| Frame | psf columns present |
|-------|----------------------|
| Light_038 (last ok) | 10: flux, err, chi2, fit_ok, quality, snr, ac_*, quality_fallback |
| Light_039+ (all fail) | **4 only**: flux, err, chi2, fit_ok |

The 4-column pattern matches the **exception swallow** branch in `pipeline.py:569-585`
(minimal columns on caught exception). Successful `psf_photometry_stars` path writes 10 columns.

Frame 039: 3509 rows, **all** `psf_fit_ok=false`, **zero** finite `psf_flux`.

### Classification
| Option | Verdict |
|--------|---------|
| (i) Job died at frame ~35 | **Rejected** - 134 proc CSVs exist through Light_148 |
| (ii) Per-frame swallow from frame ~36 | **Confirmed** (column signature + abrupt n_ok drop). Infolog text unavailable; exception string not captured (replay too slow; UI log not in draft infolog) |
| (iii) Fit-level failure only | **Rejected as sole cause** - would retain quality/ac columns from partial `psf_photometry_stars` |

**Mechanism (named):** Per-frame **exception swallow** (C1 confirmed) beginning at frame
**BO_CVn_Light_039** (index 35). Job continues; proc CSVs get NaN flux + fit_ok=False with
minimal column set. Produces E2 LC truncation pattern exactly.

**Architect finding C1: CONFIRMED.**

---

## P1-B - 517 build all-fail (E6)

Sandbox: `dev/sandbox/epsf_valid_02_p1_measure.py`, iteration hook on EPSFBuilder.

| Item | Result |
|------|--------|
| Full `build_epsf_model` draft 517 | **FAIL** `ValueError: The ePSF fitting failed for all stars.` |
| Pool size after `_epsf_prepare_stars` | 1561 stars (cutout_size=17) |
| C6 sky-injection vs pre-sub extract (200-star sample, iter 1) | Both **all status 0** - injection not primary |
| Iteration accounting (smoothing=quadratic, maxiters=15) | Iter 1-5: n_fail=0; iter 6: **1522/1561** fail; iter 10: **1559/1561** fail |
| Dominant photutils warning | **Status 3** - fitted position outside cutout (recentering divergence) |
| Status 1 (fit region beyond cutout) | Present at field edge (e.g. index 884) |

**517 vs 516:** 516 build **succeeds** (1475 stars, meta `masterstar_epsf_meta.json` on 516).
517 fails because the widened CSV pool + recentering iterations drive essentially all stars
to status 3 before stack completes - not because iter-1 extraction fails.

**Architect finding C5: CONFIRMED** (all-fail raise at `_process_iteration` when all
`_fit_error_status > 0`).

**Architect finding C6: REJECTED as root cause** - first iteration passes for both injection
paths; failure emerges at iteration 6+ (recentering). Sky-sub refactor (F4) may still help
marginally but is not the 517 all-fail explanation.

Artifacts: `p1b_517_status_histogram.json`, iteration capture in sandbox stdout.

**F4 pre-registration note:** Fix is **BUILD-star selection** (Part C gates + edge-safe
cutout sizing), not sky-injection alone.

---

## P1-C - Set census on 516

Artifact: `p1c_set_census.csv`

| Set | Count |
|-----|-------|
| `_epsf_lc_catalog_ids` (measurement when PSF enabled) | **2505** |
| `_epsf_target_catalog_ids` | **2505** (same; full `comparison_stars.csv` loaded) |
| Science set (DECISIONS: targets excl catalog_only + per-target comps + checks + blended) | **333** |
| Science subset of LC | yes (333 subset of 2505) |
| LC only (not science) | **2172** |
| Dashboard "Stars with PSF stats" (E1: 3944) | All catalog_ids in proc CSVs with psf columns (broader than LC set) |

**Architect finding C2: CONFIRMED** - measurement uses full LC pool (~2505), not science set
(333). F3 pre-registered scope reduction is justified.

---

## P1-D - Quality context (E5)

| Item | Finding |
|------|---------|
| E1 "32 stars >50% PSF frames" | 32 stars with n_psf_ok/n_frames > 0.5; these have **n_frames=34-35 only** (short-track subset), not full-night 134-frame variables |
| Full-night stars (n_frames>=100) with >50% PSF ok | **0** |
| Typical 134-frame star | n_psf_ok=**35**, pct=**26.1%** (matches E2/E3) |
| E1 32-star psf_dao_ratio | 0.26 - 1.16 (median quality mixed; not E5's 0.64-0.91 full-coverage band) |
| E1 32-star mean chi2 | median **14.4** |
| `psf_ac_applied` | **True** on all ok fits sampled (frame 001: 844/844 ok rows) |
| C3 display bug | pct_psf_ok=26.1 stored correctly; Streamlit ProgressColumn without format -> **2610%** display |

**Architect finding C3: CONFIRMED.**

E5 "full-coverage" ratios (0.64-0.91, chi2 20-36) likely refer to the 32 short-track stars
with 100% ok within their 35-frame span - partial overlap with measured 32-star set; chi2
order of magnitude consistent.

---

## Architect findings summary

| ID | Verdict |
|----|---------|
| C1 per-frame swallow | **CONFIRMED** |
| C2 measurement vs science set | **CONFIRMED** |
| C3 ProgressColumn double-x100 | **CONFIRMED** |
| C4 dashboard scope | **Not re-measured** (code inspection only; accepted) |
| C5 photutils all-fail taxonomy | **CONFIRMED** |
| C6 sky-injection stale attrs | **REJECTED as 517 root cause** |

---

## Phase 2 pre-registration (unchanged; authorize after review)

| Item | Scope |
|------|-------|
| F1 | UI pct fix + science-set dashboard scope |
| F2 | INV-PSF-FRAME-01 fail-loud accounting (addresses P1-A) |
| F3 | Measurement set := science set (333 on 516) |
| F4 | 517 build: BUILD-star selection (Part C), not injection-only |

Parts B-D unchanged from task brief.

---

## Docs impact (G0)

- `dev/validation/db_quick_check_waiver.json` - committed waiver marker
- `docs/VYVAR_DECISIONS.md` - Milan 2026-08-22 files-only addendum
- `docs/VYVAR_ROADMAP.md` - DB-RETIRE-01 FUTURE row; EPSF-VALID-02 entry point
- `docs/VYVAR_STATE.md` - waiver status, ePSF arc

---

## Errors

None blocking P1. G2 `--full` incomplete at report time.

## Files changed

- G0 commit `0e1a484`: gate script, waiver, tests, docs
- `dev/sandbox/epsf_valid_02_p1_measure.py` (sandbox, untracked)
- `dev/sandbox/epsf_valid_02_p1_frame_replay.py` (sandbox, untracked)
- `dev/results/context/session_20260822_epsf_valid_02_p1/*` (measurement artifacts)
- `dev/results/CURSOR_RESULT_EPSF_VALID_02_P1.md` (this file)

**STOP** - Phase 1 complete pending architect review. All rev-2 gates green (G0-G2).

---

## Addendum (sandbox replay, 2026-08-22)

`dev/sandbox/epsf_valid_02_p1_frame_replay.py` replayed `_fill_psf_catalog_columns` on
stored FITS+proc for Light_038 and Light_039 (~37 min):

| Frame | Replay n_ok | Exception |
|-------|-------------|-----------|
| BO_CVn_Light_038 | 912 / 3478 | none |
| BO_CVn_Light_039 | 912 / 3509 | none |

**Implication:** The collapse recorded in on-disk proc CSVs (0 ok from frame 39 onward,
4-column swallow signature) is **not reproducible** on current code + frozen model in
isolation. The original UI RUN ePSF job likely hit a **transient per-frame exception**
from frame 39 onward (swallow path still the best match for the persisted artifact), but
the root trigger (memory, stale worker state, concurrent access, etc.) was not captured
in infolog. F2 fail-loud accounting remains justified to prevent silent persistence.
