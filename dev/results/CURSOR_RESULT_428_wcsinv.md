CURSOR RESULT - 2026-07-16 (F-428-WCS-INV fix batch)

What I did
Implemented WCS invertibility gate, SIP inverse regeneration, coordinate finalization, and
post-match pixel identity gate per architect task. Added `wcs_invertibility.py`, wired into
`pipeline.py`, `astrometry_optimizer.py`, `vyvar_platesolver.py`; FIX 5 Poisson note in
`diag_428_coord_forensics_v5.py`. Unit tests: `tests/test_wcs_invertibility.py`.
pytest **871 passed**, 16 skipped.

## Fixes shipped

| Fix | Module / site | Behavior |
|-----|---------------|----------|
| **FIX 1** | `wcs_invertibility.py`; `pipeline.py` post-solve + post-optimizer | 9x9 round-trip gate; p99 < 0.2 px PASS; initial solve WARN+flag; optimizer refit FAIL-CLOSED |
| **FIX 2** | `vyvar_platesolver._fit_sip_on_matches`; optimizer `_refit_wcs_sip5_once` | `ensure_sip_inverse_coefficients`; SIP fit uses **Gaia sky** not stale row ra/dec |
| **FIX 3** | `finalize_masterstar_sky_coords` | Matched -> Gaia coords + `coord_source=gaia_catalog`; unmatched -> `final_wcs` |
| **FIX 4** | `astrometry_optimizer._write_match`; `detect_stars` post-match gate | WARN >1.5xFWHM px; drop assignment >3xFWHM px |
| **FIX 5** | `diag_428_coord_forensics_v5.py` | Poisson 33.4 arcsec vs control 122.4 arcsec reconciled (non-uniform cone export) |

**pipeline_meta.json:** `wcs_roundtrip_p99_px`, `wcs_roundtrip_pass` persisted after finalization.

## Root cause (DECISIONS)

Optimizer SIP refit paired DAO pixels with stale `ra_deg`/`dec_deg` while catalog assignment and
apertures used the healthy Gaia branch -> `world2pix(Gaia)` ~ ms x/y but `pix2world(x,y)` ~12 arcsec off.
Internal WCS round-trip can still pass (numerical world2pix). Finalization + Gaia-sky SIP refit +
identity gate close the bookkeeping chain without touching photometry science columns.

## Validation status

| Check | Status |
|-------|--------|
| pytest (871 pass) | **PASS** |
| ruff BLE001/E722 | **PASS** |
| Milan UI re-run draft_428 (T4 checklist) | **PENDING Milan** |
| LC/proc byte-identical vs current draft_428 | **PENDING re-run** |
| diag v5 -> 0 CATALOG_PROJECTION_OFF post-fix | **PENDING re-run** |
| Anchor arc (2 fresh runs + snapshot) | **BLOCKED until T4 pass** |

## Milan T4 checklist (post UI re-run)

Cursor to validate after Milan single UI re-run on clean tree:

1. provenance `git_dirty=false`, entry_point, git_hash
2. LC/proc science columns byte-identical vs current draft_428 (err included)
3. `wcs_roundtrip_p99_px` PASS in pipeline_meta
4. post-match identity gate quiet (no fail drops on priority targets)
5. masterstars: violations vs Gaia <2 arcsec for matched rows; `coord_source` populated
6. diag v5 re-run -> 0 `CATALOG_PROJECTION_OFF` on former MISASSIGNED set
7. VSX flags ~197-207; AC columns; excluded_targets.csv; REPAIR summary; UTC infolog; ePSF once; PDF overflow 0

## Errors (if any)

None in implementation / pytest.

## Files changed

- `wcs_invertibility.py` (new)
- `pipeline.py`, `astrometry_optimizer.py`, `vyvar_platesolver.py`
- `scripts/diag_428_coord_forensics_v5.py`, `scripts/diag_428_coord_forensics_v4.py` (ruff)
- `tests/test_wcs_invertibility.py` (new)
- `docs/VYVAR_DECISIONS.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_JOURNAL.md`, `CHANGELOG.md`
- `CURSOR_RESULT_428_wcsinv.md`
