# VYVAR audit 2026 — Wave 2: gates register

**Date:** 2026-08-13  
**Scope:** Every admission rule, threshold, invariant, QC flag, and quality gate — including off-by-default and hardcoded values.  
**Anchor drafts for boundary counts:** 435, 509, 510 (BO CVn NoFilter_60_2 bin2).

Field comparison note: **AstroImageJ** typically flags outliers and leaves decisions to the observer; **VYVAR** often auto-excludes comps/epochs. Differences noted per gate where known.

---

## 2.1 Gate index (by pipeline stage)

| # | Gate ID / name | File:line | Default | Policy |
|---|----------------|-----------|---------|--------|
| G-CAL-01 | CAL-DIAG v2 (INV-CAL-01) | `cal_diag.py:301` | always on | FAIL/ABORT |
| G-CAL-02 | Cal stage stamp (INV-CAL-02) | `cal_stage.py`; `invariants_runtime.py:902` | always on | FAIL |
| G-CAL-03 | INV-FLUX-01 resample sum | `invariants_runtime.py:132` | on | FAIL |
| G-CAL-04 | INV-FLUX-02 flat median?1 | `invariants_runtime.py:183` | on | FAIL |
| G-CAL-05 | Master validity days | `importer.py` (plan) | 30/90 d | soft warn |
| G-CAL-06 | BPM dark MAD | `importer.py:1139` | ?=5 | UNVERIFIED fire |
| G-QC-01 | QC-01 qc_metrics.csv match | `pipeline.py` (align) | on | FAIL |
| G-QC-02 | FWHM auto limit (MAD) | `night_run.py:799` | cfg on | filter |
| G-QC-03 | FWHM manual limit | `ui_quality_dashboard.py:647` | cfg | filter |
| G-QC-04 | IS_REJECTED frame flag | DB + UI editor | manual | exclude frame |
| G-QC-05 | INV-PREP-01 gradient ratio | `invariants_runtime.py:446` | WARN@10× | WARN |
| G-QC-06 | INV-FLAT-01 post-preprocess | `invariants_runtime.py:390` | WARN | WARN |
| G-SAT-01 | SAT-DIAG admission fraction | `sat_diag.py:737` | >10% frames over admission | exclude comp |
| G-SAT-02 | SAT-DIAG placed aperture | `sat_diag.py` (INV-SAT-01) | on | FAIL invariant |
| G-SAT-03 | admission_sat_peak_frac (D5-2) | `config.py:756`; photometry | 0.70 | exclude comp |
| G-SAT-04 | per_frame_saturation_enabled | `photometry_core.py`; INV-CFG-01 | on | mask epoch |
| G-AST-01 | WCS invertibility (INV-WCS-00) | `wcs_invertibility.py` | on | FAIL |
| G-AST-02 | INV-WCS-01 identity p95 | `invariants_runtime.py:403` | WARN@2 px | WARN |
| G-AST-03 | SIP RMS guard reject | `vyvar_platesolver.py:761` | ratio 1.15 | reject SIP |
| G-AST-04 | catalog_match_max_sep | `pipeline.py:14290` | floor 10? | match cutoff |
| G-AST-05 | frame_align_residual_gate | `photometry_core.py:7854` | **off** | drop frames |
| G-DAO-01 | masterstar_dao_threshold_sigma | `config.py` | 3.8 | detection |
| G-DAO-02 | dao_detection_n_equiv | `pipeline.py` | 3.78 | zone boundary |
| G-DAO-03 | SIP clip edge recovery | `astrometry_optimizer.py:980` | comment | no NAXIS clip |
| G-MS-01 | MASTERSTAR TOP1 score | `night_run.py:817` | auto | pick frame |
| G-PHOT-01 | phase01_comparison_max_comp_rms | `config.py:847` | **0.1** mag | exclude comp |
| G-PHOT-02 | comp_pool max_comp_rms default | `comp_pool_rms.py:88` | **0.05** | prefilter |
| G-PHOT-03 | Hard p2p ceiling | `photometry_core.py:3015` | **0.10 hardcoded** | exclude comp |
| G-PHOT-04 | Stability slope gate | `photometry_core.py:3035` | 5 mmag/hr @ 3? | exclude comp |
| G-PHOT-05 | phase01_comparison_n_comp_min | `config.py` | 3 | fail phase |
| G-PHOT-06 | Colour tier limits | `comp_color_tiers` ? `comp_selection_per_target.py` | tiered | weight/exclude |
| G-PHOT-07 | VSX out-of-scope types | INV-CFG-01R | cfg list | skip target |
| G-PHOT-08 | err_scatter unmatched epochs | `photometry_core.py:3609` | on (I-04 fix) | exclude epoch |
| G-PHOT-09 | detect_outliers (variables) | `photometry_core.py:4529` | 3? (relaxed VSX) | flag epochs |
| G-PHOT-10 | COG night gate | `photometry_core.py:2126` | cfg | all-or-nothing |
| G-PHOT-11 | Colour term apply gate | `should_apply_color_term` | auto | fit/not |
| G-PHOT-12 | Ensemble completeness audit | `night_run.py:381` | 90% measurable | fail run |
| G-PHOT-13 | ZP MAD clip | `ensemble_normalize` | **removed** 2026-08-12 | — |
| G-TRUST-01 | Check-star scatter soft/hard | `trust_flag_core.py` | cfg | GREEN/YELLOW/RED |
| G-TRUST-02 | comp_trust_min_comps | `trust_flag_core.py:61` | 3/8 | trust tier |
| G-TRUST-03 | comp_stability_test p-value | `sparse_trust_core.py:212` | sparse path | marginal flag |
| G-EXP-01 | OSC-03 oneRGGB export block | `invariants_runtime.py:313` | on | FAIL |
| G-EXP-02 | BJD_TDB required for AAVSO | `export_reports.py` | on | refuse export |
| G-EXP-03 | time_base truth (T1 fix) | export path | on | refuse |
| G-INV-01 | INV-DAG-01 stage order | `invariants_runtime.py:481` | on | FAIL |
| G-INV-02 | INV-PHASE0-ID catalog join | `invariants_runtime.py:713` | on | FAIL |
| G-INV-03 | INV-PROV-01 schema | `invariants_runtime.py:798` | end-of-run | FAIL |
| G-INV-04 | INV-CFG-01 gating no-op | `invariants_runtime.py:575` | end-of-run | FAIL |
| G-INV-05 | INV-COMP-MEMBERSHIP | policy | on | review FAIL |

---

## 2.2 Known gates — detailed review

### G-PHOT-01 / G-PHOT-02 / G-PHOT-03 — comp RMS trinity

| Aspect | Detail |
|--------|--------|
| **Purpose** | Reject comparison stars with night-long scatter inconsistent with ensemble precision |
| **Condition** | (1) Phase-1: `comp_rms > phase01_comparison_max_comp_rms` (default **0.1**); (2) pool prefilter default **0.05**; (3) stability hard ceiling `_ABS_MAX_P2P = **0.10**` at `photometry_core.py:3015` |
| **Condition matches purpose?** | **Partially.** Three different ceilings; (3) is hardcoded and duplicates (1) with different default semantics |
| **Defect can disable?** | Setting (1) very loose does not disable (3) |
| **Boundary (510)** | comp_rms max **0.025** — **75% below** 0.05 and **4× below** 0.10; **no comps within 20% of any ceiling** |
| **Per draft vs frame** | Per draft (membership stable) ? |
| **Provenance** | `quality`, `note` in stability dict; comp CSV columns |
| **Ever fired?** | draft 509: ZP clip fired (removed); p2p ceiling UNVERIFIED on anchors |
| **Other tools** | AIJ: user inspects comp LC; VYVAR: auto-exclude |

**Status:** C-P2P-01, C-MAX-RMS-01 OPEN

---

### G-PHOT-04 — stability slope gate

| Aspect | Detail |
|--------|--------|
| **Purpose** | Catch comps with slow linear drift invisible to p2p |
| **Condition** | `slope_mmag_hr > max_comp_slope_mmag_hr` (default **5**) AND `slope_sig >= comp_slope_significance_k` (default **3**); ?20 finite points — `photometry_core.py:3035-3076` |
| **Matches purpose?** | Yes, if slopes calibrated |
| **Boundary (510)** | UNVERIFIED per-comp slope values on disk without re-run |
| **Per draft** | Yes |
| **Provenance** | Log line + `note` suffix on comp |
| **Ever fired?** | UNVERIFIED on 435/509/510 |
| **Other tools** | AIJ: manual; VYVAR: auto-exclude |

---

### G-SAT-01 / G-SAT-03 — saturation admission

| Aspect | Detail |
|--------|--------|
| **Purpose** | Keep non-linear/compromised stars out of ensemble (D5-2) |
| **Condition** | SAT-DIAG: `n_over_admission/n_frames > 0.10` — `sat_diag.py:737`; photometry: peak > `admission_sat_peak_frac × sat_adu` (0.70) |
| **Matches purpose?** | Yes post placed-aperture fix |
| **Boundary (510)** | sat_diag: `sat_adu=65535`, `lin_adu=55704.75` (DEFAULT_FRAC); **0 admission_rejects** in sat_diag.json |
| **Per draft** | Yes (membership) |
| **Provenance** | sat_diag.json; photometry skip_reason columns |
| **Ever fired?** | batch E physical re-cut excluded bright comps; 510 post-fix: none |
| **Other tools** | AIJ: saturation warning; VYVAR: hard exclusion |

**Note:** `lin_source=DEFAULT_FRAC` — Tier-3 must not exclude (spec); warning present in sat_diag.json.

---

### G-CAL-01 — INV-CAL-01 CAL-DIAG

| Aspect | Detail |
|--------|--------|
| **Purpose** | Verify dark resample SUM vs MEAN convention before science |
| **Condition** | Checks P/C/B pedestal/resolvability — `cal_diag.py:301` |
| **Matches purpose?** | Yes when `sigma_p > 0`; **degenerate when `pedestal_sigma_p=0`** |
| **Boundary (510)** | PASS SUM; `pedestal_p=24.47`; **`pedestal_sigma_p=0.0`** — register OPEN |
| **Per draft** | Per obs_group × dark × binning |
| **Provenance** | cal_diag.json + FITS VY_DKRSMP |
| **Ever fired?** | **Yes — always runs**; 435/509/510 PASS in cal_diag.json |
| **Other tools** | IRAF ccdproc: user chooses combine mean/sum; VYVAR: derived gate |

---

### G-SAT-02 — SAT-DIAG / INV-SAT-01 (2026-08-13)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Raw-frame peak measurement for saturation authority |
| **Condition** | Placed aperture on raw grid; no brightest-pixel search on comps |
| **Matches purpose?** | Yes (fixes peak-search contract bug) |
| **Boundary (510)** | 5 comps, check scatter **0.008629**, GREEN |
| **Ever fired?** | Gate runs; exclusion none on 510 |

---

### G-PHOT-13 — ZP MAD clip (removed)

| **Purpose** | Remove per-frame ZP outliers |
| **Condition** | Was `len(z) >= 4` ? 3×MAD clip |
| **Bug** | **Condition wrong:** purpose needs ?4 comps *after* clip stability; at N=4 one outlier breaks ensemble |
| **Status** | **FIXED** removed 2026-08-12; INV-COMP-MEMBERSHIP |

---

### G-PHOT-08 — I-04 err_scatter unmatched

| **Purpose** | Do not quote finite err when comp coverage incomplete |
| **Condition** | `_exclude_err_scatter_unmatched_epochs` — `photometry_core.py:3609` |
| **Status** | FIXED batch D; epochs ? NaN err |

---

### G-AST-05 — frame_align_residual_gate

| **Purpose** | Drop frames where alignment residual > frac × aperture radius |
| **Condition** | `frame_align_residual_gate_enabled` default **False** — `photometry_core.py:7871` |
| **Boundary** | UNTESTED on anchor (off) |
| **Other tools** | Most pipelines: manual blink |

---

### G-AST-03 — SIP RMS guard

| **Purpose** | Reject SIP distortion fit if RMS not improved vs linear |
| **Condition** | `_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO = 1.15` — `vyvar_platesolver.py:81` |
| **Boundary** | UNVERIFIED fire rate on 510 |
| **Other tools** | Astrometry.net: separate distortion step |

---

### G-PHOT-09 — detect_outliers (survivor)

| **Purpose** | Sigma-clip variable-star epochs after detrend |
| **Condition** | `detect_outliers` — `photometry_core.py:4529`; VSX-known skips clip |
| **Note** | Residual sigma-clip params **ignored** elsewhere (zero-clipping 2026-08-12); API surface partially dead |

---

### G-BPM — bpm_dark_mad_sigma

| **Purpose** | Flag hot pixels from dark stack |
| **Condition** | ? default 5 — `config.py:736`; read `importer.py:1139` |
| **Ever fired?** | **No `*_dark_bpm.json` on any draft** (STATE carry-forward) |
| **Status** | UNVERIFIED end-to-end |

---

### G-TRUST-01 — trust flag scatter

| **Purpose** | Observer-facing GREEN/YELLOW/RED from check-star RMS |
| **Condition** | Soft/hard thresholds in `trust_flag_core.py` |
| **Boundary (510)** | BO CVn check scatter **0.008629** ? GREEN |
| **Bug carry** | C-TRUST-01: color term misread in same module |

---

### G-INV-05 — INV-COMP-MEMBERSHIP

| **Purpose** | No per-frame comp membership changes |
| **Enforcement** | Policy + code review; ZP clip was violation |
| **Status** | Clip removed; frame_align gate could drop frames without changing comp set (different axis) |

---

### G-INV-00 — INV-ANCHOR-00 (`--full` coverage)

| **Purpose** | Regression gate |
| **Gap** | Does **not** exercise cal, preprocess, align, MASTERSTAR, detection |
| **Implication** | Gates in those stages invisible to `--full` |

---

## 2.3 Invariants quick reference

See `docs/VYVAR_INVARIANTS.md` for full wired set. FAIL-CLOSED: INV-FLUX-01/02, INV-DAG-01, INV-PROV-01, INV-CFG-01, INV-PHASE0-ID, QC-01, OSC-01/02/03, INV-SAT-01, INV-CAL-01/02.

---

## 2.4 Gate firing summary (drafts 435/509/510)

| Gate | 435 | 509 | 510 |
|------|-----|-----|-----|
| CAL-DIAG | PASS | PASS | PASS (?_p=0) |
| SAT-DIAG admission | UNVERIFIED json | UNVERIFIED | runs; 0 rejects |
| SAT peak 0.70 exclusion | batch E history | same raw as 435 | 0 exclusions post-fix |
| ZP MAD clip | N/A pre-fix | **fired** (bug) | removed |
| p2p 0.10 ceiling | no comps near | no comps near | max comp_rms 0.025 |
| max_comp_rms 0.1 | no comps near | no comps near | same |
| `--full` photometry SHA | P1 stale | — | check scatter 0.008629 |

---

## Wave 2 closing

**Surprised:** CAL-DIAG **does** fire on every anchor run (contrary to task hint "never had"); SAT-DIAG json on 510 shows **`raw_peaks_used: false`** — verify whether placed-aperture path logged separately.

**Could not determine:** Per-comp slope exclusions without recomputing stability; BPM gate end-to-end.

**Next (Wave 3):** Trace unit contracts at every boundary (gain, RN, pedestal, SATURATE_ADU, exposure time, filter identity).
