CURSOR RESULT - 2026-08-12 (ZP clip removal)

What I did
Removed the per-frame 3xMAD zeropoint rejection in `ensemble_normalize`
(`photometry_core.py`). Kept Broeg 1/sigma^2 (+ tier) weights. Documented
decision + INV-COMP-MEMBERSHIP. Re-ran draft 509 photometry. Did not push.

## Section 2 - Callers and side effects

Clip outputs before removal:
- Only a `logging.debug("[ZP] Frame sigma-clip: ...")` line when comps were dropped.
- No return value, no QC counter, no proc CSV column, no PDF field.

After removal: that debug line is gone. Nothing else referenced a rejected-comp
count from this clip. No retired field needed.

Config parameters that only served this clip: **none live**. The clip used a
hardcoded 3.0 and `_MAD_CONSISTENCY`. Historical `comp_clip_sigma` was already
removed in `c9e1f8f` (different, draft-level iterative ensemble clip path).

Untouched (as required): `detect_outliers`, SIP plate-solver clip, I-04 epoch
drop, `phase01_comparison_max_mag_diff`, Broeg weights, draft-level p2p/slope.

## Section 3-4 - Predictions

| ID | Prediction | Result | Measured |
|----|------------|--------|----------|
| 1 | 435 byte-identical / clip no-op at N=3 | **PASS** | Clip was gated on `len(z)>=4`; matrix D==C exact (max diff 0). Unit tests encode Broeg-all-comps equality. `--fast` failed on 4 **pre-existing** unrelated tests (ascii_policy, field_map, robust_fwhm x2) - not this edit. Did not run `--full` (P1-RECUT). |
| 2 | 509 check `...4892800` ~0.0085, unimodal | **PASS** | check_scatter=**0.00863**; target poly5 res std=0.0126, peaks=[-0.004] unimodal; ZP residual unimodal; trust **GREEN** |
| 3 | n_points stays 134 | **PASS** | n_points=**134** |
| 4 | Instrumental check scatter stays 0.008-0.009 | **PASS** | instrumental check509 vs 3-comp ens std=**0.00917** (unchanged) |

Validation: photometry re-run on draft 509 existing aligned frames (clip lives in
ensemble_normalize; full raw rebuild not required to test this edit). First 509
re-run wrote LCs then hit INV-DAG-01 (phase2a stamp after postprocess); stages
trimmed and re-run for report export.

## Diff summary

- `src_py/photometry_core.py`: delete lines that MAD-clipped `z`/`w` when `len(z)>=4`
- `dev/tests/test_ensemble_normalize_no_zp_clip.py`: new
- docs: DECISIONS, INVARIANTS, STATE, JOURNAL

## White-cores v2 (same session, read-only)

Closed at Step 1 letter **(a)**: cores are peaked maxima (often saturated
plateau ~68567 ADU). Sky ~2413 not 2.2e8. Display artifact. Report:
`dev/results/CURSOR_RESULT_white_cores_v2.md`.

## Errors
INV-DAG-01 on first 509 re-stamp (mitigated by trimming stages). Mistakenly
started a 435 full photometry re-run; killed and restored BO CVn LC from backup.

## Files changed
See git commit. Not pushed.
