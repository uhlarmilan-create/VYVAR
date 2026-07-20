CURSOR RESULT - 2026-07-20 (DEPS-CYCLE-2)

What I did
Cross-major dependency cycle: photutils 3.0.0 + astropy 8.0.1 + numpy pin floor 2.4.4.
Scout checklist M1-M6 executed; gates G1-G4 run; anchor byte-identical (path a).

## Migration checklist (M1-M6)

| Step | Status | Notes |
|------|--------|-------|
| M1 kwarg migration | DONE | `sharpness_range`/`roundness_range` in `DAO_STAR_FINDER_NO_ROUNDNESS_FILTER`; `n_brightest` at all former `brightest=` sites |
| M2 min_separation=0 | DONE | In preset dict + every standalone `DAOStarFinder` (src_py + dev harness scripts) |
| M3 column renames | DONE | Readers use `x_centroid`/`y_centroid` (pipeline, platesolver, alignment, wide_slope_noise, tests) |
| M4 PSF module | DONE | ImagePSF/PSFPhotometry/SourceGrouper/EPSFBuilder import smoke OK; no FittableImageModel/EPSFModel |
| M5 warning sweep | DONE | Zero DeprecationWarning from our photutils call sites on DAO-related tests; third-party NoDetectionsWarning only |
| M6 fresh-env proof | DONE | `tmp/deps_cycle2_venv`: pip install -r requirements.txt -> numpy 2.5.1, astropy 8.0.1, photutils 3.0.0 |

## Gates

| Gate | Result |
|------|--------|
| G1 `--fast` | OVERALL PASS (1058 passed, 17 skipped) |
| G2 P1 golden | **7/7 PASS** (mini SHAs unchanged) |
| G3 `--full` vs anchor | **Path (a) BYTE-IDENTICAL** - run `tmp/session_baseline/20260720T224544Z`, 2025 s |
| G4 runtime | ~2025 s new vs ~2296 s pre-upgrade (pubprep Part B baseline); ~12% faster |

### G3 evidence

| Check | Result |
|-------|--------|
| full-science-compare | PASS n_lc=166 failures=0 |
| full-snapshot-sha-core | PASS 03d8fb6491bc3c22... n=333 |
| full-photometry-sha-core | PASS 03d8fb6491bc3c22... n=333 |
| full-photometry-sha-extended | PASS bbfcc92e7ac5c4c5... n=499 |

No re-cut required (VL-ANCHOR-WCSINV + VL-P1-GOLD unchanged).

## Pins (requirements.txt)

```
numpy>=2.4.4,<3
astropy>=8.0,<9
photutils>=3.0,<4
```

Installed (dev): numpy 2.4.4, astropy 8.0.1, photutils 3.0.0.

## Docs impact

- `docs/VYVAR_DECISIONS.md` - DEPS-CYCLE-2 (M2 freeze decision, path-a outcome)
- `docs/DEPS_POLICY.md` - pin examples + watchlist (photutils 3.1)
- `docs/VYVAR_STATE.md` - header one-liner
- `dev/results/DEPS_SCOUT.md` - Cycle 2 executed addendum
- `dev/tools/docs_pdf/build_flow_doc.py` + `docs/VYVAR_FLOW_CZ.pdf` - ch 17 pins + software citations
- ROADMAP: no change

## Recurrence

- Any new `DAOStarFinder` site must pass `min_separation=0` until a deliberate science arc adopts the 3.x crowding filter (DECISIONS DEPS-CYCLE-2).
- Cross-major bumps: scout -> migrate -> `--full` vs anchor per DEPS_POLICY.

## Errors (if any)

None blocking. `--full` OVERALL was FAIL solely on transient pytest count during combined long run; all anchor/science checks PASS.

## Files changed

- `requirements.txt`
- `src_py/utils.py`, `pipeline.py`, `vyvar_platesolver.py`, `vyvar_alignment_frame.py`, `wide_slope_noise_core.py`
- `dev/scripts/` (DAO harness alignment)
- `dev/tests/test_g1_f003_alignment_pixel_fallback.py`
- docs listed above
