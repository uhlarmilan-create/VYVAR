CURSOR RESULT - 2026-08-04

What I did
Read-only WIDE-ERR E4 per-comp excess measurement on draft_000435 / NoFilter_60_2.
Reused E2 field builder, E3 photon-err path (_photon_rel_median on proc rows; proc CSV
has no err_photon column). Check star from tmp/wide_err_w1w2/diag_check_lc LCs.
Pre manifest check PASS; harness dev/tools/wide_err_e4.py; JSON tmp/wide_err_e4/wide_err_e4.json.

## E4.1 -- Excess per comp

Rig: 166 fields with check-star ensemble; 1167 comp instances (field x comp).

Definition per comp:
  sigma_meas = 1.4826 * MAD(m - comp_ref_map) across frames
  err_photon_median = median photon err (rel flux) from proc rows at comp epochs
  excess = sqrt(max(0, (1000*sigma_meas)^2 - (1000*err_photon_median)^2)) mmag

Distribution of excess (all comps):
  median   20.18 mmag
  IQR      15.77 -- 20.18 -- 25.50 mmag
  min      0.00 mmag
  max      65.61 mmag

Counts:
  excess > 10 mmag:  1097 / 1167
  excess > 20 mmag:   593 / 1167

Excess vs G magnitude (median excess per bin):
  G  8-10   n=  57   11.92 mmag
  G 10-11   n= 117   15.52 mmag
  G 11-12   n= 370   18.50 mmag
  G 12-13   n= 514   24.03 mmag
  G 13-14   n= 109   30.53 mmag
  G 14-16   n=   0   (no comps)

Excess vs median peak ADU (quintile bins on comp population):
  ADU q0-20   n= 235   25.24 mmag
  ADU q20-40  n= 233   26.04 mmag
  ADU q40-60  n= 234   21.99 mmag
  ADU q60-80  n= 242   19.03 mmag
  ADU q80-100 n= 223   14.80 mmag

## E4.2 -- Check star excess

Star 1499906247391001088; 163 fields with diag LC.

Definition per field:
  sigma_meas = 1.4826 * MAD(mag_calib_final) over frames
  err_photon_median = median err_photon from LC
  excess = sqrt(max(0, (1000*sigma_meas)^2 - (1000*err_photon_median)^2)) mmag

Distribution across 163 fields:
  median excess   17.62 mmag
  IQR             11.48 -- 17.62 -- 23.99 mmag

Context:
  median err_photon(check) across fields   0.002467 rel flux  (~ 2.47 mmag)

## E4.3 -- Direct comparison

  median excess, all comps              20.18 mmag
  median excess, comps G bin 8-10       11.92 mmag  (n=57)
  median excess, check star (163 fld)   17.62 mmag
  ratio  check / bright-comp excess      1.48

WIDE-ERR-RIG-EXCESS: 17.6 mmag

## Errors (if any)
None.

## Files changed
dev/tools/wide_err_e4.py (harness)
tmp/wide_err_e4/wide_err_e4.json (output)
dev/results/CURSOR_RESULT_wide_err_e4.md (this report)
