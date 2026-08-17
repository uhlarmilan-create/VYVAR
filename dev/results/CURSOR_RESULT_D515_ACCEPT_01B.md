CURSOR RESULT - 2026-08-16 D515-ACCEPT-01B

SUPERSEDED-WITH-POINTER (ensemble membership only, 2026-08-17): the 515 BO
cell below (7.0498 mmag) was measured with C2 `1500748301498613248` in the
ensemble. SAT-LIMIT-01 reclassified C2 as saturated. Product-frame check MAD
without C2 (no replacement) is 8.580 mmag. See
`dev/results/CURSOR_RESULT_SAT_LIMIT_01.md`. This 2x2 vs 514 same-meter table
is not overwritten; it remains the 514-vs-515 identical-ensemble comparison.

What I did
Same-meter re-read of check MAD on draft 515 (run SHA da9cce4) using the
IMPL-05 C check IDs. Measurement via temp Phase 2A with is_check_star forced
(production check_kmag path). Self-check validated the method. Reverse
direction (NEW check on draft 514) completed after draft_000514 re-check.
No science-code change. Push not authorized.

Note on draft_000514: an earlier probe reported the directory missing; re-check
confirms it is present and intact (49 check_kmag sidecars; BO/FW meters match
IMPL-05 C). That absence was a transient path/visibility false alarm.

## 2x2 table (same check id per row)

Quantity: check_scatter_mad_mmag [mmag] = 1.4826 * MAD(kmag) * 1000.
kmag = check star differenced against the TARGET ensemble.
Archive photometry SHA: da9cce4.

| Target | check_id | subset IMPL-05 C (514) | n | draft 515 | n | delta (515-subset) |
|---|---|---:|---:|---:|---:|---:|
| BO CVn | 1498020894186918144 | 8.5946 | 134 | 7.0498 | 134 | -1.5449 |
| FW CVn | 1497368849430107904 | 9.8193 | 134 | 10.6836 | 134 | +0.8644 |

Ensemble on 515 (unchanged production assignment):
- BO: n_comp=5; check NOT in ensemble.
- FW: n_comp=8; check NOT in ensemble. CHK_FW is a BO ensemble member on 515
  (valid as FW check). CHK_BO is an FW ensemble member but is used only as BO's
  meter here.

## Pre-registered interpretation

- BO: delta -1.545 mmag -> ok_le_subset_plus_1
- FW: delta +0.864 mmag -> ok_le_subset_plus_1
- Overall: CONFIRMED_IDENTICAL_METER

D515-ACCEPT-01 Part D4 acceptance verdict stands as written, confirmed on
identical meter.

## Method validation

Force NEW check (1497613731286514432) on 515 BO via the same temp Phase 2A
path: measured 6.713213 mmag vs production sidecar 6.713213 mmag (exact match).

## Reverse direction (NEW check on 514)

check_id 1497613731286514432 against each target's 514 ensemble (carrier-target
inject so PERF-8 matrix includes NEW; not attached to the science ensemble):

| Target | NEW on 514 [mmag] | n | NEW on 515 (prod) [mmag] |
|---|---:|---:|---:|
| BO CVn | 12.7733 | 134 | 6.7132 |
| FW CVn | 14.0884 | 134 | 8.2010 |

The NEW meter is quieter on 515 than on 514; the D515-ACCEPT-01 deltas that
compared different check stars remain confounded in the reverse direction as
well. The same-meter 2x2 above is the acceptance-closing comparison.

## Files

- dev/results/D515_ACCEPT_01B_numbers.json
- dev/tools/d515_accept_01b_same_meter.py

## Errors

None blocking. Primary 2x2 + reverse complete.
