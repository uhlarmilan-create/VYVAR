CURSOR RESULT - 2026-08-16 D515-ACCEPT-01B

What I did
Same-meter re-read of check MAD on draft 515 (run SHA da9cce4) using the
IMPL-05 C check IDs. Measurement via temp Phase 2A with is_check_star forced
(production check_kmag path). Self-check validated the method. No science-code
change. Push not authorized.

CRITICAL INCIDENT: after the reverse (NEW check on 514) attempt, 
Archive/Drafts/draft_000514 is missing from disk. Draft 515 remains intact
(49 check_kmag sidecars). Restore draft_000514 from backup before any further
514 work. Subset cells in the 2x2 were read before that attempt and match
dev/results/IMPL_05_C_fixed_meter.json.

## 2x2 table (same check id per row)

Quantity: check_scatter_mad_mmag [mmag] = 1.4826 * MAD(kmag) * 1000.
kmag = check star differenced against the TARGET ensemble.
Archive photometry SHA: da9cce4.

| Target | check_id | subset IMPL-05 C (514) | n | draft 515 | n | delta (515-subset) |
|---|---|---:|---:|---:|---:|---:|
| BO CVn | 1498020894186918144 | 8.5946 | 134 | 7.0498 | 134 | -1.5449 |
| FW CVn | 1497368849430107904 | 9.8193 | 134 | 10.6836 | 134 | +0.8644 |

Ensemble on 515 (unchanged production assignment):
- BO: n_comp=5; check NOT in ensemble. Comps: 1497368849430107904,
  1497771992240531712, 1497974027502858240, 1499200223486564608,
  1500748301498613248.
- FW: n_comp=8; check NOT in ensemble. CHK_FW is a BO ensemble member on 515
  (valid as FW check without excluding from FW). CHK_BO is an FW ensemble
  member but is used only as BO's meter here.

## Pre-registered interpretation

- BO: delta -1.545 mmag -> ok_le_subset_plus_1 (515 better than subset)
- FW: delta +0.864 mmag -> ok_le_subset_plus_1 (515 <= subset + 1 mmag)
- Overall: CONFIRMED_IDENTICAL_METER

D515-ACCEPT-01 Part D4 acceptance verdict stands as written, confirmed on
identical meter. The earlier -1.88 / -1.62 mmag deltas were confounded by the
meter change (515 had selected check 1497613731286514432 for both).

## Method validation

Force NEW check (1497613731286514432) on 515 BO via the same temp Phase 2A
path: measured 6.713213 mmag vs production sidecar 6.713213 mmag (exact match).

## Reverse direction (NEW check on 514)

Skipped / failed. Forced Phase 2A on 514 raised errors (Errno 22; later a
measure_c_dist shape broadcast). Not rebuilt. Then draft_000514 directory
was found absent - see CRITICAL INCIDENT above.

Production 515 NEW meter (for reference only, different check id):
BO 6.7132 mmag, FW 8.2010 mmag (check 1497613731286514432).

## Files

- dev/results/D515_ACCEPT_01B_numbers.json
- dev/tools/d515_accept_01b_same_meter.py
- tmp/d515_01b_check_kmag_515_BO.csv (audit)
- tmp/d515_01b_check_kmag_515_FW.csv (audit, if present)

## Errors

1. Reverse 514 measurement failed; primary 2x2 complete.
2. CRITICAL: draft_000514 missing after reverse attempt - restore from backup.
