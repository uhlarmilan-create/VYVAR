CURSOR RESULT - 2026-08-15 TARGET-DEPTH-02

Register ID: TARGET-DEPTH-02
Follows: TARGET-DEPTH-01 (Item A accepted; Item B superseded here)
Status: COMMITTED tip **974e17a**; `--fast` OVERALL PASS on that tip (1360 passed, 27 skipped); push awaits Milan.

## Verdict

MASTERSTAR is a **single best-frame copy** (not a stack). Scale factor = 1
(T2-R0). Item A gates photometry on `zone=noise`. Item B replaces the
TARGET-DEPTH-01 NP half-SNR proxy with a MASTERSTAR zone linear-fraction depth
(3 named constants vs 6). Item A alone is the primary physical gate; propose
dropping the mag limit in a follow-up. Item C: BO CVn ensembles on 512 vs 513
are **disjoint** -- uncontrolled trial comparison (T2-R4).

---

## History -- the ~G 14.5 limit

| What | Detail |
|------|--------|
| Parameter | `vsx_variable_targets_mag_limit = 14.5` |
| Removed | commit `a0e3431` (2026-07-22) `feat(vsx)!: scope by DAO+Gaia detection; remove vsx_variable_targets_mag_limit` |
| Decision | `VSX-AUTO-MAGLIM` in `docs/VYVAR_DECISIONS.md` |
| Intent then | VSX scope by DAO+Gaia detection instead of a static mag_max |

That was a VSX planner cut, not a MASTERSTAR-scaled per-frame SNR limit. Its
removal is a regression relative to Milan's expected ~G14.5 programme depth
once SNR-GATE deepened MASTERSTAR.

---

## Item A -- `noise` must not enter photometry

### Downstream consumer check (before implement)

| Consumer | Depends on noise LCs? |
|----------|------------------------|
| `variability_detector` | No -- default `zone_filter=["linear"]` |
| `lc_quality` / `_classify_lc_quality` | Labels existing noise LCs as `noisy`; diagnostic only |
| UI aperture | Badge/display only |
| Phase 2A | Builds LC unless `skip_photometry`; gating is the intended change |

**Verdict:** safe to gate. No stop condition.

### Implementation

`select_active_targets`: `zone_flag==noise` -> `skip_photometry=True`,
`skip_reason=zone_noise` (row kept). Phase 2A also refuses `noise` when
per-frame sat is off.

### Mask counts (existing products)

| Draft | n_noise | of which have LC today |
|------:|--------:|-----------------------:|
| 513 | 45 | 12 |
| 512 | 26 | 6 |
| 510 | 26 | 6 |
| 435 | 0 | 0 |

---

## Item B -- MASTERSTAR depth scaled to a single frame

### 2.1 What MASTERSTAR is

`build_masterstar_from_detrended` docstring and body: **copy the single best
processed FITS** (lowest `VY_FWHM`) to `MASTERSTAR.fits`. Not a median/mean
stack. Headers on 512/513/510/435: `EXPTIME=60`, no `NCOMBINE`, `DATE-OBS`
matches one light. Candidate pool may list many frames; the product is one.

**T2-R0:** `n_combine=1` => `snr_scale_factor = 1`, `mag_offset = 0`. The
linear-on-MASTERSTAR / unusable-LC gap is **not** explained by `sqrt(N)`.
Remaining cause for residual gap: best-frame FWHM/sky can beat the median
night frame; forced photometry still returns flux on worse frames.

### 2.2 New criterion (replaces TARGET-DEPTH-01 proxy)

From MASTERSTAR `zone` labels (already `peak_dao/bg_sigma` vs
`dao_detection_n_equiv`):

- Walk 0.5-mag bins; last bin with `n>=8` and `frac(zone==linear) >= 0.5`
- `target_depth_g` = that upper edge minus `mag_offset` (0 when n_combine=1)

Physical requirement: at least half the field stars in the bin still sit above
the DAO peak-significance cut on the MASTERSTAR frame. Same family as catalogue
half-completeness (`g_lim_50`). AAVSO SNR~100 applies to **comps**; targets use
the existing DAO N-sigma cut (`dao_detection_n_equiv=3.78`), not an AAVSO-100
constant.

### Chosen constants (T2-R1)

| # | Name | Value | Justification |
|---:|------|------:|---------------|
| 1 | `bin_width` | 0.5 | COMP-POOL NP convention |
| 2 | `min_bin_n` | 8 | COMP-POOL NP convention |
| 3 | `linear_frac_thr` | 0.5 | half-linear / g_lim_50 family; not tuned to 14.5 |

TARGET-DEPTH-01 had **six** named choices; this has **three**.

### Derived depths

| Draft | depth_g | n_targets mag>limit | of those already noise | existing LC mag>limit |
|------:|--------:|--------------------:|-----------------------:|----------------------:|
| 513 | 15.0 | 15 | 12 | (see harness) |
| 512 | 11.5 | 27 | 25 | (shallow pre-SNR-GATE MS) |
| 510 | 11.5 | 27 | 25 | same |
| 435 | 15.5 | 0 | 0 | 0 |

vs old ~G14.5 VSX cut: 513 lands **0.5 mag fainter** (15.0); 512/510 are
shallower because their MASTERSTAR cliff is the broken-gate depth, not the VSX
cut.

### F2 cross-check (T2-R2)

SNR-GATE-01 F2: G14 frac_median 0.955, G15 0.507. Draft 513 MASTERSTAR bin
14.5-15.0 linear_frac ~0.55 (usable), 15.0-15.5 ~0.36 (fails) => depth 15.0.
**Agrees** with the F2 cliff within 0.5-mag binning. No adjustment.

### 2.3 Item A vs Item B redundancy

With factor=1, per-star `zone=noise` **is** significance at the right depth.
Item A is sufficient for Milan's "only linear goes to photometry" rule. The mag
limit is a population summary that also catches rare `linear` outliers past the
half-linear bin. **Propose dropping the magnitude limit** in a follow-up after
Item A is accepted. Not removed here.

---

## Item C -- BO CVn 512 vs 513 (finding only)

| | Draft 512 | Draft 513 |
|--|----------:|----------:|
| trust | GREEN | RED |
| check_scatter | 0.009300 | 0.011147 (+1.85 mmag) |
| n_clean | 5 | 4 |
| selected_tier | TIER1 | TIER3 |
| n_tier1/2/3/4 | 5/0/0/0 | 0/0/1/3 |

Comp sets are **disjoint** (intersection empty). 513 trust_reason: T3+ colour
mismatch, 0 T1/T2 (GREEN needs >=3).

**Comparability (T2-R4):** 513 is a trial under X-R3; differs in detection depth,
MASTERSTAR content, and possibly assignment path. Cannot attribute the regression
to a single cause from this uncontrolled pair. A committed-tree rebuild of
both drafts on one tip would settle it. No fix in this task.

---

## Item D

### zone names

Quantity: `peak_dao/bg_sigma` vs `dao_detection_n_equiv`. Proposed rename (not
implemented): `linear` -> `dao_detected`, `noise` -> `dao_subthreshold`,
`saturated` -> `peak_saturated`.

### INGESTED status on 512/513/510

Heal-on-open does not rewrite old manifests. Repair: re-run
`record_qc_processing_apply(draft_id, hash)` then
`update_obs_draft_status(draft_id, "PROCESSED")` (patches `draft_manifest.json`;
same path `night_run`/`app` use after DQF).

---

## Section 7 evidence

| Check | Measured |
|-------|----------|
| Aperture max \|delta\| vs archive | 0.10 px (n=23; pre-existing near-tie) |
| dao_flux six BO stars `proc_BO_CVn_Light_068.csv` | max rel = 0.0 |
| Exported errs | identical re-read |
| BO CVn masked? | No on 512/513/510/435 (linear, G~9.72) |
| Iron-gate + kwarg | fire (returncode 0) |
| `--fast` | OVERALL PASS on tip `974e17a` (1360 passed, 27 skipped) |

Impact inventory: zone_noise skip and MASTERSTAR depth gate intended; photometry
numbers / COMP-POOL Stage 2 / SNR-GATE untouched.

---

## Pre-registered rules

| Rule | Fired? |
|------|--------|
| T2-R0 | Yes -- single-frame MASTERSTAR; factor=1 |
| T2-R1 | Yes -- 3 chosen constants (was 6) |
| T2-R2 | Yes -- 513 depth agrees with F2 cliff |
| T2-R3 | Yes -- no tuning to 14.5 or counts |
| T2-R4 | Yes -- BO comparison uncontrolled; honest outcome |

---

## Deferred (one line each)

- LOCATION_OLD orphan heal (unchanged from TARGET-DEPTH-01).
- Propose removal of draft-level mag limit once Item A accepted (redundancy).
- zone rename (`dao_detected` / `dao_subthreshold`) -- own change.
- Controlled BO CVn rebuild on committed tip for Item C attribution.

---

## Register diff

- **TARGET-DEPTH-02**: FIXED (A+B+C finding+D records)
- **TARGET-DEPTH-01 Item B**: SUPERSEDED by TARGET-DEPTH-02 MASTERSTAR zone depth

Machine-readable: `dev/results/TARGET_DEPTH_02_results.json`

## Files changed

- `src_py/comp_pool_noise.py` -- MASTERSTAR zone depth
- `src_py/photometry_core.py` -- noise skip + wire depth from masterstars
- `dev/tests/test_target_depth_01.py`
- `dev/results/TARGET_DEPTH_02_results.json`
- `dev/results/CURSOR_RESULT_TARGET_DEPTH_02.md`
- `docs/VYVAR_AUDIT_2026_REGISTER.md`
