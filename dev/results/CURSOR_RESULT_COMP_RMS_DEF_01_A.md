CURSOR RESULT - 2026-08-25 (COMP-RMS-DEF-01-A STOP C2)

What I did
Measure-only. Named what selector `comp_rms` actually computes,
recomputed four candidate definitions on 520 (7 stars) and 516
(68 live comps), checked suspected_variables, and measured
ZONE-SAT-01 on the 520 G=7.63 star. No wiring (C3 waits Milan GO
on C2f). Live 516/520 SHA unchanged. No Archive/Drafts writes.

HEAD `78b3495` / remote `sel-ghost-01`. Session JSON:
`dev/results/session_20260825_closeout/c2_measure.json`.

## Premise (Rule 0.1)

**What is compared:** B3 T4 selector `comp_rms` (0.17-4.33 mag on six
bright 520 pool stars; forced-7 ensemble lc_rms 0.053) versus four
definitions computed from the same 520 g_60_4 proc CSVs, and the
same four on the 67/68 live 516 comps. ZONE-SAT-01 compares the
zone classifier to pixel peak vs `saturate_limit_adu`.

**How they differ:** selector `comp_rms` is not leave-one-out
differential mag. Photon sigma is 1/snr_ap from MASTERSTAR; frozen
516 MS has no `snr` / `snr_ap_pixscaled`, so 516 k = iii/photon is
not computed here. Direct `compute_global_pool_rms_map` on 520
returned nulls (name vs catalog_id / dtype); column (i) is the T4
`rms_map` (same selector, already run).

## C2a - what the selector computes

File: `src_py/comp_pool_rms.py` `compute_global_pool_rms_map`
lines 74-387. Ceiling site:
`photometry_core._select_comps_by_rms_then_color` ~15464.

- Flux column: `dao_flux` (fallback `flux`).
- Frames: all loadable per-frame proc CSVs. No separate QC-admit
  filter inside the map. Empty / nonpositive flux skipped.
- Differential: relative flux vs mag-bin (0.5 mag) or the frame
  catalog median, then quadratic detrend. Not vs an ensemble.
- Statistic: `robust_comp_rms` = 1.4826 * MAD of detrended relative
  flux (`comp_frame_normalize.py:153`). `clip_sigma` is accepted and
  ignored (INV-NOCLIP).
- NaN: non-finite and <=0 dropped; stars with < min_frames
  (max(3, 0.3*n_loaded)) omitted.
- Units: fractional flux, stored as `comp_rms` and gated as if mag
  (fixed ceiling 0.1).

That is sky-pedestal / wrong-bin relative-flux scatter, not
differential magnitude against a comparison ensemble.

## C2b - 520 seven stars x four definitions

Proc CSVs: 25. Selector (i) from T4 `rms_map`. (ii) std of raw
instrumental mag. (iii) std of leave-one-out differential mag vs
the forced-7 ensemble excluding itself. (iv) MAD*1.4826 of (iii).
Photon = 1.0857 / snr_ap from the 520 MS (approx).

| catalog_id | G | (i) selector | (ii) std inst | (iii) std LOO | (iv) MAD LOO | photon |
|---|---|---|---|---|---|---|
| 1112113680298377344 | 7.63 | 0.969 | 0.383 | 0.184 | 0.037 | 0.00127 |
| 1111920204908702336 | 10.07 | 4.332 | 0.772 | 0.594 | 0.016 | 0.00511 |
| 1112110695298081664 | 10.90 | 0.758 | 0.583 | 0.373 | 0.014 | 0.00632 |
| 1111749157833870208 | 11.23 | 3.815 | 0.382 | 0.331 | 0.018 | 0.00683 |
| 1112121862213003648 | 11.10 | 0.207 | 0.389 | 0.028 | 0.012 | 0.00697 |
| 1112121067641532160 | 11.61 | 0.174 | 0.363 | 0.326 | 0.016 | 0.00824 |
| 1111737033143440768 | 13.87 | 0.051 pass | 0.634 | 0.391 | 0.187 | 0.0252 |

(iv) puts the bright six near a few times photon (still above
photon for G=7.63 because of clip outliers). (iii) is
outlier-sensitive (0.18-0.59). (i) is not a magnitude RMS: a 4 mag
selector number can sit in a 5% ensemble. Named defect: gating
fractional mag-bin relative flux as if it were mag scatter.

## C2c - 68 live 516 comps (one draft, wide rig)

68 unique comps (task said 67; T2-P1 was 67 on skip_photometry=False
targets). 134 proc frames. Frozen MS has no SNR column -> photon
NaN, k not computed.

| def | p50 | p90 | max |
|-----|-----|-----|-----|
| (iii) std LOO mag | 0.052 | 0.055 | 0.110 |
| (iv) MAD*1.4826 LOO | 0.031 | 0.041 | 0.072 |

(i) selector null on this recompute (same name/id issue as 520
direct map). Ceiling k from iii/photon needs a post-C0b MS with
`snr_ap_pixscaled` (not this freeze).

## C2d - suspected_variables.csv

Statistic: `photometry_core._write_suspected_variables` ~18560.
Same mag-bin relative flux + quadratic detrend, then
RMS = sqrt(mean((rel-1)^2)) -- not MAD. Flag if rms > median +
3 * MAD_of_the_rms_distribution.

5/7 of the 520 seven are flagged:
1112113680298377344, 1111920204908702336, 1111749157833870208,
1112121067641532160, 1111737033143440768.

Under definition (iii) all five still have std LOO > 0.05, so they
"survive" as high if the flag used (iii). That overstates
variability: (iv) of the bright four is 0.016-0.037. The G=13.87
star is honestly noisy on (iv)=0.187.

516 suspected_variables n=250; 0 of those 250 are in the live 68
comps.

## C2e - ZONE-SAT-01 (measure only)

Classifier: `src_py/pipeline.py` `_annotate_masterstars_flux_zones`
line 6490. Peak test 6577-6579: zone=saturated if `peak_s` >
`peak_sat_lim`. `peak_s` = `peak_max_adu` if present else `flux`,
**not** `peak_dao`. Limit from
`_resolve_peak_saturation_limit_adu` (empirical clip, else
equipment/header clip * saturate_limit_fraction / n_stack).

Star 1112113680298377344 (520 g_60_4, G=7.63):

| field | value |
|-------|-------|
| peak_max_adu | 88781.5 |
| peak_dao | 50134.7 |
| zone | linear |
| saturate_limit_adu | 65535 |
| saturate_limit_adu_85pct | NaN |

Peak test skipped because `peak_sat_lim` is None when 85pct is
NaN. linear then comes from `peak_dao/bg_sigma`. A clipped pixel
peak (88781 > 65535) passes as linear.

Task count (`peak_dao` > `saturate_limit_adu` and zone != saturated):
520 = 0; 516 = 0 (4 peak_dao>lim, all already zone=saturated).

Governing-pixel count (`peak_max_adu` > lim and zone != saturated):
520 = 1 (this star); 516 = 0 (12 peak_max>lim, all saturated).
516 zone=saturated n=24; lim85 finite on all 3610 frozen rows.
520 lim85 finite on 0 rows -- that is why the clipped peak is not
tested.

## C2f - named defects and fix direction (no wiring)

Defects:

1. `comp_rms` is mag-bin relative-flux scatter (fractional), gated
   with a 0.1 "mag" ceiling. Seventh "statistic under the gate".
2. `suspected_variables` uses the same relative-flux series with
   RMS not MAD.
3. Zone peak test reads `peak_max_adu`/`flux` but skips when
   `saturate_limit_adu_85pct` is NaN, so a clipped peak stays linear.

Fix direction for Milan GO (C3):

- `comp_rms` := leave-one-out differential scatter against the
  candidate pool, definition (iv) preferred (or (iii)), QC-admitted
  frames only, no clipping.
- Ceiling := k x photon-expected sigma of the star. k to be
  measured from 516 once `snr_ap_pixscaled` exists (this freeze
  cannot supply k).
- `suspected_variables` on the same LOO definition.
- zone := pixel peak (`peak_max_adu` or `peak_dao`) vs the effective
  saturation limit even when `peak_sat_lim` is None.

Predictions stay as the task stated; not tested here.

## Run time (Rule 0.3)

C2 measure script: seconds (CSV reads, no photometry rerun).

## Gates

`--fast --clean` OVERALL PASS at `78b3495` (C0c) before this STOP.
Live SHA unchanged. No sigma-clipping added.

## Errors

Direct selector recompute on 520/516 returned null `comp_rms` for
the seven / 68 (name vs catalog_id). Column (i) on 520 uses T4
`rms_map`. 516 photon/k not computed (frozen MS lacks SNR).

## Docs impact

DECISIONS COMP-RMS-DEF-01 findings; ROADMAP C3 waits GO;
ZONE-SAT-01 still OPEN (measure done, wiring is C3). STATE/JOURNAL.

## Recurrence

Do not gate a fractional relative-flux MAD as if it were mag RMS.
Do not skip the saturation peak test when 85pct is NaN -- read the
pixel peak against `saturate_limit_adu`.

## Files changed

- `dev/results/CURSOR_RESULT_COMP_RMS_DEF_01_A.md` (this STOP)
- `dev/results/context/session_20260825_closeout/c2_measure.json`
