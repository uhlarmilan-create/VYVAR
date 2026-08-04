> **PROVENANCE WARNING (added 2026-07-30).** The numbers in this document derive from **draft_000451** (with comparisons to **draft_000450** and anchor **draft_000435_snapshot_skysurface_20260716**), produced after the in-place preprocess architecture landed (`013cb0c`, 2026-07-22) and before the sky-surface idempotency guard (`84174ae`, 2026-07-30). During that window a repeated preprocess pass could subtract the sky surface twice, at a measured cost of order 500 ADU. **draft_000451** and **draft_000450** are no longer available, so their status is UNKNOWN, not clean. Treat these numbers as indicative, not validated.

# CURSOR RESULT - DRAFT-451 ANALYSIS (2026-07-27)

Read-only diagnostic. No code changes, no commits. Scratch: `tmp/draft451_analysis.py`,
`tmp/draft451_analysis.json`.

Draft 451 is the first BO CVn run on the **full raw-to-photometry path** with the identity gate
and VSX->Gaia matcher live. It inherits the SKY-SURFACE regression (same DAO catalogue as 450).
**Not a reference candidate.**

---

## Executive verdict

**The section-2 hypothesis is supported for the extra targets; it is not supported for shared-target
photometry degradation.**

| Prediction | Result |
|------------|--------|
| Group B (451-only actives) fainter, noisier, worse comps, more missing/useless curves | **Confirmed** -- fainter by ~1.5 mag (p50), 3.7x higher median RMS, 50% RED trust vs 13% in Group A |
| Group A (shared with anchor 165) normal quality | **Confirmed** -- median RMS ratio 451/435 = **1.01**; BO CVn numerically identical to 450 |
| Shared group also degraded (SKY-SURFACE hurts well-detected targets) | **Not supported** at population level -- only **5.1%** of shared LCs worse by >20% RMS vs anchor |

Milan's field notes map cleanly: faint spurious actives and noisy RED curves dominate Group B;
BO CVn and the anchor-shared set look like prior runs.

**Caveat on "mag 16-18":** `active_targets.csv` `mag` tops at **15.25** (Gaia placeholder rotators in
Group B). Nothing in the active list exceeds G=16 in that column. The Summary Measure Report impression
of "16-18" likely reflects display rounding, VSX/catalog context, or curves that are present but
unusable (RED / high scatter) rather than literal active-row magnitudes. The detection-limit knee is
visible near **G 14-15** in the scatter table below.

---

## A1 - Run fingerprint

### Infolog (`infolog_20260727_104233.txt`)

**`VSX-GAIA XM:`** -- **not present** in the UI infolog (Python `logging.info` only). Captured by
read-only plan-regen replay on frozen 451 inputs (`tmp/draft451_analysis.py`):

```
VSX-GAIA XM: n_vsx=873 n_gaia=15092 rho=705.8 deg^-2 mean_nn=67.8" r_max=7.65" Q=0.99 w=0.96
sigma_n=0.18" sigma_b=0.50" accepted=717 contamination=0.011% cand_mult=0:156 1:694 2:23 3+:0
multi=2.63% pm_path=broadened pm_cols=False pm_finite=0 vsx_epoch=2000.0 gaia_epoch=2016.0
masterstars=278/282 outcome=ok gaia_db_max_g=17.5
```

**`FAZA 0 funnel:`** -- **not present** as the structured `logging.info` line in infolog. Closest
verbatim UI/log_event proxies:

```
07:40:23  select_active_targets: deduped to 242 unique catalog_ids (prefer real VSX name over Gaia placeholder)
07:40:23  select_active_targets: linear=119 noisy1=14 noisy2=26 noisy3=81 saturated=2
          no_catalog_id=0 no_gaia_id=0 no_dao_detection=25 out_of_frame=78
07:40:23  [RUN VYVAR] Faza 0 hotova: 242 aktivnych cielov
```

Computed fingerprint (`phase0_funnel.py` on disk):

| Field | Value |
|-------|-------|
| `variable_targets_rows` | 873 |
| `gaia_match_source` | masterstars=278, gaia_dr3_direct=439, no_match=156 |
| `active_targets_rows` | **242** |
| `skip_photometry_true` | 2 (saturated zone) |
| `zone_flag` | linear=119, noisy1=14, noisy2=26, noisy3=81, saturated=2 |

### MASTERSTAR DAO (infolog + measurement)

Infolog verbatim:

```
07:20:14  INFO  [pipeline]  [DAO pass 1] 8926 detections, 403 Gaia unmatched
07:20:22  post_match_identity_gate: ok=3177 warn=254 fail=5 (FWHM=3.23px)
07:20:14  DAO po SNR filtri (sumova podlaha median+1.8xsigma): 6698/9284 bodov
          (noise_floor~2066.4 ADU; pred matchom s katalogom).
```

| Quantity | draft 451 (measured) | draft 435 anchor | draft 450 |
|----------|---------------------:|-----------------:|----------:|
| Pass-1 DAO count | **8926** | 2552 | 8926 |
| `sigma_pp` (MASTERSTAR.fits) | **46.07 ADU** | 46.13 | 46.07 |
| `bg_std` estimator | **62.23 ADU** | 83.82 | 62.23 |
| Threshold @ 2.1 sigma | **130.7 ADU** | 175.4 | 130.2 |
| Effective sigma (`thresh/sigma_pp`) | **2.84** | 3.81 | 2.83 |
| Best-frame FWHM (identity gate line) | **3.23 px** | (435 post-match gate ~similar class) | 3.23 px |
| `DAO_ONLY` fraction (`masterstars_full_match.csv`) | **40.4%** (2705/6698) | **3.7%** (109/2951) | **40.4%** |

451 and 450 share the **identical** inflated MASTERSTAR catalogue (6698 rows, 2705 DAO_ONLY).

### `pipeline_meta.json` excerpt

- `gaia_dao_completeness_raw_pct`: 3.99
- `n_gaia_detected`: 3992
- `g_lim_50` / `g_lim_90`: 14.97 / 14.17
- `lc_quality_summary`: good=126, noisy=113, no_data=1, saturated=2, total=242
- `phase2a_empty_comp_drop`: 1 (R CVn class)

---

## A2 - Two-group comparison (core test)

Split by `catalog_id` membership in anchor snapshot active set (165).

| | **Group A** (shared, n=160) | **Group B** (451-only, n=82) |
|---|---------------------------:|-----------------------------:|
| **n active** | 160 | 82 |
| **Mag p10 / p50 / p90** | 11.28 / **13.14** / 14.20 | 13.52 / **14.61** / 15.11 |
| **Fraction with no LC** (`n_frames=0`) | 3/160 = **1.9%** | 0/82 = **0%** |
| **Median RMS** (LC CSV, mag_calib_final) | **0.081 mag** | **0.301 mag** |
| **Median per-point err** | **0.061 mag** | **0.255 mag** |
| **Median n_good_comp** | 8 | 8 |
| **Trust: GREEN / YELLOW / RED** | 70 / 67 / 20 | 4 / 37 / **41** |

Group B is **~1.5 mag fainter** (p50), **~3.7x noisier** (median RMS), and **82% RED or noisy**
quality flags vs 44% in Group A. All 82 Group-B actives received frame data; quality is the defect,
not file absence.

### Scatter vs magnitude (both groups, targets with usable LC points)

| Mag bin | n | Median RMS | p90 RMS | Group A | Group B |
|---------|--:|-----------:|--------:|--------:|--------:|
| 0-12 | 21 | 0.015 | 0.046 | 21 | 0 |
| 12-14 | 118 | 0.081 | 0.185 | 111 | 7 |
| 14-15 | 71 | 0.247 | 0.447 | 25 | **46** |
| 15-16 | 14 | 0.428 | 0.641 | 0 | **14** |
| 16-17 | 0 | -- | -- | 0 | 0 |

The knee where curves become scientifically useless is **G ~ 14-15**: median RMS rises from ~0.08 to
~0.25-0.43 mag, driven almost entirely by Group B.

**Group B composition:** 82 targets absent from anchor 165 -- overwhelmingly **Gaia DR3 placeholder**
VSX rows (`Gaia DR3 ...` names) at the DAO detection tail (G 14.6-15.3). Example faintest:

- Gaia DR3 1498670636838609664 (G=15.25, noisy3, RED)
- ZTF J135315.86+392846.7 (G=15.16, noisy3)
- TSVSC1 TN-N130302101-35-67-2 (G=15.10, noisy2)

**Five anchor actives missing from 451** (full-path VT/matcher differs from frozen anchor):

- TOI-3919 (exo -- not promoted on this full-path VT of 873 VSX-only rows)
- Gaia DR3 1497436125799224960, 1485913828055470592, 1497121459315202560
- HAT-188-0000323

---

## A3 - Missing curves (enumerated)

Only **3** actives with `n_frames=0` (same three as anchor CLOSE-2 C2):

| Target | zone_flag | skip_reason | n_frames | n_good_comp | Group |
|--------|-----------|-------------|----------|-------------|-------|
| **R CVn** | linear | *(empty)* | 0 | 0 | A |
| **CV CVn** | saturated | zone_flag | 0 | 0 | A |
| **Gaia DR3 1498278351706325248** | saturated | zone_flag | 0 | 0 | A |

**By cause:**

| Cause | Count | Targets |
|-------|------:|---------|
| `zone_flag` saturated skip | 2 | CV CVn, Gaia DR3 1498278351706325248 |
| Phase 2A silent drop (empty `skip_reason`, R CVn class) | **1** | R CVn (`ac_skip_reason=no_comps`) |
| Group B missing LC | **0** | -- |

**Silent Phase 2A drop count: 1** (R CVn) -- same defect class as CLOSE-2 C2, not a new regression.

**What Milan likely saw as "no light curve":** 132 actives with LC files but **RED or noisy /
no_data** quality (61 Group A + 71 Group B), including 64 RED curves with frames. The pipeline
produced 239 LCs from 139 frames; photometry ran, but many Group-B curves are not credible.

---

## A4 - Shared targets vs anchor history

157 shared targets have usable LCs in **both** 451 and anchor snapshot.

| Metric | Value |
|--------|------:|
| Median RMS ratio (451 / 435) | **1.007** |
| Fraction worse (>1.2x RMS) | 5.1% |
| Fraction better (<0.9x RMS) | 12.1% |

Shared-target photometry is **equivalent** to anchor at median; degradation is not the dominant story.

### BO CVn (`catalog_id=1498613634033133184`)

| | draft 435 anchor | draft 450 | draft 451 |
|---|-----------------:|----------:|----------:|
| Mean/median mag (LC) | 9.466 | 9.419 | **9.419** |
| RMS scatter | 0.1479 | **0.1479** | **0.1479** |
| Median per-point err | 0.00493 | 0.00495 | **0.00495** |
| n_good_comp | 3 | 4 | **4** |
| Trust | GREEN | GREEN | **GREEN** |
| Comp ensemble | 3 GAIA_MATCHED | 4 GAIA_MATCHED | **Same 4 IDs as 450** |

**Milan's visual reading confirmed:** 451 matches 450 exactly; 435 differs only in comp count
(3 vs 4 -- one comp added in post-regression runs, all GAIA_MATCHED, not DAO_ONLY).

---

## A5 - Comparison-star pool

Selected comparison stars (`comparison_stars_per_target.csv`, 451):

| Metric | Value |
|--------|------:|
| Unique comp stars used | 435 |
| Unique with `source_type=DAO_ONLY` | **0** |
| Unique with `peak_dao` below 435-equivalent threshold (175 ADU) | **0** |
| Targets whose ensemble is majority DAO_ONLY or below-threshold | **0** |

**Conclusion:** On this run the regression did **not** poison well-detected targets through the comp
pool. Inflated catalogue effect operates through **target admission** (identity join on spurious DAO
detections), not through comp-star contamination. Group A photometry staying near anchor is consistent
with this mechanism.

---

## Quantitative expectation after SKY-SURFACE + sigma-estimator fix

> **VOID (2026-07-30, Audit Tranche 3 / P-10).** This table assumed draft_435 had correctly
> flattened backgrounds and that restoring sky-surface subtract would return pass-1 DAO to ~2550.
> **P-10 sign error:** 435 `proc_*` had **doubled** gradient (`2g`), not flattened; 450 plain had
> natural `g`. The two defects (sign error + inflated `bg_std` estimator) **cancelled** on 435,
> yielding ~3.7% `DAO_ONLY` at an effective **3.8sigma**, not 2.1sigma. After P-10 + sigma_pp estimator
> + threshold recalibration (~3.8), expect threshold ~175 ADU and anchor-class DAO counts - **not**
> the table below. See `CURSOR_RESULT_audit_t3.md`.

~~After fix (restore sky-surface subtract + correct estimator, `bg_std` ~84 ADU, threshold ~175 ADU):~~

| Quantity | Observed 451 | ~~Expected post-fix~~ **VOID** |
|----------|-------------:|------------------:|
| Pass-1 DAO | 8926 | ~~**~2550**~~ |
| `DAO_ONLY` fraction | 40.4% | ~~**~3.7%**~~ |
| Active targets | 242 | ~~**~160-165**~~ |

The 82 Group-B extras are exactly the targets that should not survive a correct DAO threshold; they
are genuinely Gaia-identified but **marginally detected** spurious actives, not a separate catalogue
hygiene problem requiring a magnitude parameter.

---

## What this run is / is not

| | |
|---|---|
| **Is** | First full-path proof of identity gate + matcher; regression diagnostic; confirms DAO inflation -> faint spurious actives |
| **Is not** | Reference anchor candidate; evidence that shared-target photometry is broken |
| **Next** | SKY-SURFACE arc (restore subtract on skip-only path) + sigma-estimator fix; re-run full path and expect active count to collapse toward 165 without any mag limit |

---

## Files read (no modifications)

- `Archive/Drafts/draft_000451/infolog_20260727_104233.txt`
- `Archive/Drafts/draft_000451/platesolve/NoFilter_60_2/{MASTERSTAR.fits,masterstars_full_match.csv,variable_targets.csv}`
- `Archive/Drafts/draft_000451/platesolve/NoFilter_60_2/photometry/{active_targets.csv,photometry_summary.csv,pipeline_meta.json,comparison_stars_per_target.csv,lightcurves/*}`
- Anchor: `Archive/Drafts/draft_000435_snapshot_skysurface_20260716/.../photometry/*`
- Reference: `Archive/Drafts/draft_000450/...` (BO CVn compare)
