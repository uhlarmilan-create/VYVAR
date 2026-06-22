CURSOR RESULT — 2026-06-19

What I did
Decoupled astroalign control-point count from the detection ladder (G1-F001/F002): new `alignment_max_control_points` config (default 80), plumbed through night_run/app/pipeline into `_align_ctx`, ladder uses plumbed cap while detection still uses max_stars 200–500. Added unit tests and Milan validation script. Ledger: FIXED pending Chi/h validation. Commit held locally; not pushed.

## Output / findings

### Root cause (fixed)
- `vyvar_alignment_frame.py`: `mcp = max(12, min(max_st, n_fit))` tied astroalign CP to ladder detection cap (up to 500).
- `pipeline.py`: `align_cp` was `min(cap, 1.5×N_ref)` and never passed into worker ctx; `max_control_points` param was dead at frame level.

### Changes
| Area | Change |
|------|--------|
| `config.py` | `alignment_max_control_points: int = 80`; clamp 12–500 in `__post_init__`; `to_dict` |
| `ui_settings.py` | Slider next to `alignment_max_stars` |
| `pipeline.py` | `align_cp` from `cfg.alignment_max_control_points`; `max_control_points` in `_align_ctx` + MP ctx |
| `vyvar_alignment_frame.py` | `mcp = max(12, min(max_control_points, n_fit))` |
| `night_run.py` / `app.py` | Source from `cfg.alignment_max_control_points` (removed hardcoded 180) |
| `scripts/validate_alignment_control_points.py` | Chi/h validation harness (200 vs 80 on same detections) |
| `tests/test_alignment_max_control_points.py` | Mechanism tests (default, clamp, json, mcp formula) |

### Unit tests
```
420 passed, 15 skipped (full suite)
tests/test_alignment_max_control_points.py: 6 passed
```
No wall-clock speedup measured in unit tests (synthetic points only). Theoretical triangle budget: C(200,3)/C(80,3) ≈ **16.2×**.

### Validation script (Milan — Chi/h draft 419/420)

From repo root, after draft has `processed/lights/<setup>/proc_*.fits`:

```powershell
python scripts/validate_alignment_control_points.py `
  --draft-root Archive/Drafts/draft_000419 `
  --setup B_60_2 `
  --max-frames 5
```

Adjust `--draft-root` (419 or 420), `--setup` (filter/exp/bin folder name under `processed/lights`), and `--max-frames` as needed.

Explicit FITS mode:

```powershell
python scripts/validate_alignment_control_points.py `
  --ref path/to/reference.fits `
  --frames path/to/light1.fits path/to/light2.fits
```

**Pass criteria (printed by script):** |dtranslation| < 0.05 px, |drotation| < 0.01°, |dscale| < 1e-4, |dRMS| < 0.05 px; NEW (80 CP) materially faster than OLD (200 CP). Frames exceeding thresholds are flagged FAIL.

**Do not mark G1-F001/F002 fully closed until Milan reports Chi/h result.** Alignment output is not byte-identical.

## Errors (if any)
None.

## Files changed
- `config.py`, `pipeline.py`, `vyvar_alignment_frame.py`, `night_run.py`, `app.py`, `ui_settings.py`
- `scripts/_build_vyvar_params.py`, `scripts/validate_alignment_control_points.py`
- `tests/test_alignment_max_control_points.py`
- `docs/VYVAR_FULL_AUDIT_LEDGER.md`
- Commit: `8198c45` (not pushed)
