CURSOR RESULT - 2026-08-06 (A-6 DAO_ONLY magnitude classification)

What I did
Implemented A-6: per-row `dao_only_class` and `implied_g_mag` on `masterstars_full_match.csv`,
informational per-class census log, `pipeline_meta` flat keys, PDF resolved-facts row, and UI
caption. No runtime gate, no row removal. Offline classification on drafts 501/435/500 before
pipeline wiring. Local P1 headless A/B at same HEAD (with vs without A-6 src changes).

## 1. What `dao_reconcile.py` already provided vs added

**Already present (reused):**
- `get_gaia_db_max_g_mag` / right-censoring via `CensoredLimit`
- Fleming completeness fit (`fit_fleming_completeness`, `sigma_mag`)
- `compute_gaia_dao_reconcile`, `reconcile_to_pipeline_meta`
- Legacy `classify_unmatched_dao` (artifact heuristic for reconcile report only; not used for A-6 rows)

**Added for A-6:**
- `fit_instrumental_flux_to_g` ù median ZP from Gaia-matched `flux` + `phot_g_mean_mag`
- `derive_dao_only_class_margin` ù `hypot(2 * MAD, fleming_sigma_mag)`, no hardcoded floor
- `classify_dao_only_dataframe` ù assigns four classes; additive columns only
- `annotate_dao_only_magnitude_classes` ù pipeline entry point
- `format_dao_only_census_log`, `dao_only_report_lines`, `dao_only_class_meta_flat`
- `reconcile_to_pipeline_meta` extended to merge `dao_only_class_meta` when present

**Pipeline:** reconcile + classification run before CSV write; meta patch reuses cached `_recon_ms`.

## 2. Flux-to-G fit (draft_501 reference)

| quantity | value |
|----------|-------|
| method | median ZP: G = zp ? 2.5 log10(flux) |
| n_matched | 958 |
| zp | 22.310 |
| residual MAD | 0.101 mag |
| residual RMS | 0.431 mag |
| fleming_sigma_mag | 1.202 (from pipeline_meta curve when available) |
| derived margin | 1.219 mag |
| bright threshold | G < 16.281 (= max_g_mag ? margin) |

Formula logged: `margin = hypot(2 * flux_fit_MAD, fleming_sigma_mag)`.

## 3. Per-class tables (offline, pre-pipeline)

| draft | setup | DAO_ONLY | artifact_negative | below_catalogue | unconfirmed_bright | indeterminate |
|-------|-------|----------|-------------------|-----------------|--------------------|---------------|
| draft_501 | V_60_2 | 696 | **142** | 525 | 14 | 15 |
| draft_435 snapshot | NoFilter_60_2 | 109 | 8 | 0 | 98 | 3 |
| draft_500 | NoFilter_60_2 | 561 | 48 | 8 | 496 | 9 |

**Predictions vs data:**
- **draft_501:** `artifact_negative` = 142 ù **confirmed** (20.4%). Remainder dominated by
  `below_catalogue` (525) as predicted for shallow peak_dao / deep implied-G tail.
- **draft_435 / draft_500:** prediction of `below_catalogue` dominance ù **refuted**. Wide-rig
  DAO_ONLY are mostly **brighter** than `max_g_mag ? margin` (`unconfirmed_bright` 98/109 and
  496/561). These are spurious detections above the local catalogue depth, not faint sources the
  catalogue cannot see. The 3.7ù higher DAO_ONLY fraction on draft_500 vs 435 is explained by
  more `unconfirmed_bright` (496 vs 98), not by below-cap tail.

## 4. Census log line (verbatim, draft_501)

```
MASTERSTAR DAO_ONLY census: 696/1668 (fraction=0.417) [artifact_negative=142, below_catalogue=525, unconfirmed_bright=14, indeterminate=15] | Gaia DB cap G<17.50 | informational, not a gate
```

## 5. Local P1 A/B core SHAs

P1 mini headless chain (`draft_000435_p1mini`, setup `NoFilter_60_2`), two runs at same HEAD:

| run | core SHA | core n | ext SHA | ext n |
|-----|----------|--------|---------|-------|
| **with A-6** | `aa72e97979a74d5b8297c6bc3624bee668d8bd5f28624de0a708149e286c2636` | 325 | `05d7036d6cfd5a035fe725728bea9d89266ae8fc0b63604e0f912801d76b4c39` | 485 |
| **without A-6** | `aa72e97979a74d5b8297c6bc3624bee668d8bd5f28624de0a708149e286c2636` | 325 | `05d7036d6cfd5a035fe725728bea9d89266ae8fc0b63604e0f912801d76b4c39` | 485 |

**Identical** ù additive masterstar columns do not move photometry outputs on P1 mini.

## 6. What this does not resolve

- **`unconfirmed_bright`** rows may be real uncatalogued sources or artifacts; DAO-PHYS-1/2/2b found
  no detection-stage filter that separates them without unacceptable depth loss.
- Wide-rig inflation (`unconfirmed_bright` majority) is a separate open question from
  right-censored faint tail (`below_catalogue` on Newton draft_501).
- Consumption gate `snr50_ok` still controls photometry admission; 687/696 draft_501 DAO_ONLY
  failed that gate and never reached photometry.

## Errors (if any)

None blocking.

## Files changed

- `src_py/dao_reconcile.py` ù classification core
- `src_py/pipeline.py` ù census + columns before CSV write; meta reuse
- `src_py/photometry_report.py` ù resolved-facts DAO census row
- `src_py/ui_masterstar_qa.py` ù per-class caption
- `dev/tests/test_dao_reconcile.py` ù unit + draft_501 regression
- `dev/tools/a6_classify_offline.py` ù offline tables
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_LIMITATIONS.md`
- `dev/results/CURSOR_RESULT_a6_dao_only_classification.md`

## Acceptance checklist

| # | criterion | status |
|---|-----------|--------|
| 1 | Local A/B core SHAs identical | PASS |
| 2 | Per-class counts for 501/435/500 | PASS (offline) |
| 3 | artifact_negative draft_501 = 142 | PASS |
| 4 | No row removal; existing columns untouched | PASS (additive only) |
| 5 | No new WIRED_INV_IDS | PASS (grep) |
| 6 | Margin/fit logged, not hardcoded | PASS |
| 7 | Full suite | 1256 passed; 5 pre-existing failures (stale P1 ledger, ascii in unrelated result docs) |
| 8 | No new config key | PASS (margin derived) |
