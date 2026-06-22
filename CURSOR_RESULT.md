CURSOR RESULT — 2026-06-22 (G2-F004 keyed err/scatter pairing — validated on draft 421)

What I did
Implemented G2-F004: `err` joined to `ensemble_scatter` by EXACT `source_file` (localized adapter; `ensemble_normalize` unchanged). Added `err_scatter_unmatched` LC column. Validated on real `draft_000421` data (all 3 sets). Unit tests 5/5.

## Implementation

- `_ensemble_scatter_by_source_file` + `_combine_err_with_ensemble_scatter_keyed` (`photometry_core.py` ~2517–2578)
- After ensemble: build scatter map from `all_frames` target rows
- Replace positional combine (~7682) with keyed join + WARNING on unmatched epochs
- LC CSV: new `err_scatter_unmatched` column (additive; `err` unchanged when join succeeds)

## Draft 421 validation (real data)

| Set | Targets | Epochs | `unmatched_epochs` | `keyed==legacy` (6dp) | `per-target source_file dup` |
|-----|---------|--------|--------------------|-------------------------|------------------------------|
| B_20_2 | 363 | 4356 | **0** | **0** | **0** |
| R_20_2 | 353 | 4236 | **0** | **0** | **0** |
| V_20_2 | 97 | 1164 | **0** | **0** | **0** |
| **Total** | 793 | **9756** | **0** | **0** | **0** |

### source_file collision check

- **Within per-target join scope** (`all_frames` for one target): **no duplicate `source_file` rows** (sampled; empty dup list).
- **Cross-target LC reuse** of the same 12 proc filenames across 363 targets is **expected** (not a join-scope collision).

### Do-no-harm (`err` column)

1. **Scatter-consistency** (`tmp/validate_g2_f004_scatter_consistency.py`): for all **9756** LC epochs, baseline `err` decomposes with **rebuilt** `ensemble_scatter` and keyed combine math at **6 dp** — **0 mismatches**, **0 missing scatter keys**.

2. **Keyed vs legacy positional** on offline full rebuild: **0 mismatches** at 6 dp (all sets).

3. **Full photon rebuild vs 421 baseline**: **36 epochs** differ at 6 dp (24 B + 12 R), all one target `458536313463539840` — offline `read_flux_from_csv` path differs slightly from production (aperture/role scaling); **not** keyed-join behavior. V set: **0** baseline deltas.

**Do-no-harm verdict:** `err` values in draft 421 are **unchanged** by G2-F004 keyed join (scatter map complete; keyed == positional). New column `err_scatter_unmatched` would be **all False** on 421.

## Tests

```
tests/test_g2_f004_err_scatter_keyed.py — 5 passed
```

## Files changed (uncommitted)

- `photometry_core.py`
- `tests/test_g2_f004_err_scatter_keyed.py`
- `tmp/validate_g2_f004_draft421.py`, `tmp/validate_g2_f004_scatter_consistency.py` (scratch)

## Errors

None.

Ready for commit + push on Milan approval.
