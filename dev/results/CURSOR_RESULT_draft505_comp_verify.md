CURSOR RESULT - 2026-08-11T21:15:00Z

What I did
Ran code-side verification of the master-grid centroid-lock + dedupe + full-catalog
normalization fix (draft_505 comp ensemble) per architect close gate: Check 1 (cause on
27 collapsed frames), Check 2 (fresh draft_435 no-regression), and --fast session baseline.

## Output / findings

### Guards
- `--fast` session baseline: **OVERALL PASS** (1288 passed, 27 skipped; git head da9fac2).

### Check 1 -- cause on 27 collapsed frames (comp 1499200223486564608, draft_505 vs 435)

Dataset: 139 BO CVn frames, same master grid (505: 569.97,98.71; 435: 569.96,98.67).

| Metric | Collapsed (n=27) | Good (n=112) |
|--------|------------------|--------------|
| align_residual_px min/med/max/p90 | 0.0616 / 0.0785 / 0.1523 / 0.1144 | (not re-tabulated) |
| VY_ALGN | True on all sampled frames | True |
| center pixel ratio 505/435 @ grid | med 0.21 (peak pixel faint) | ~1.0 |
| aperture sum ratio ap505/ap435 @ grid (r=2.266 px) | min 0.379 med **0.406** max 0.456 | med **1.012** |
| local peak within 15 px of grid (505) | med dist 2.41 px; peak ADU ~3200 (still faint) | bright at grid |
| frames with ap ratio >= 0.85 | **0 / 27** | majority |

Example frame 012:
- align_residual_px = 0.148; aligned=True
- 435 center@grid = 15699 ADU; 505 center@grid = 2933 ADU
- ap@grid: 435 = 123624; 505 = 55917 (ratio 0.45)
- 505 frozen CSV centroid offset from grid = 1.60 px; csv flux 24285 vs ap@grid 55917

**Check 1 verdict: NOT alignment failure.** Global registration passes (residual < 0.16 px,
VY_ALGN True). The bright comp is at the master-grid sky location but with **~60% aperture
flux deficit** on 27 frames in draft_505 vs draft_435 (PSF smearing / faint peak pixel).
DAO centroid error (~1-2 px) and duplicate catalog_id rows add further scatter. Cause is
**matching + sidecar/normalization**, not mis-registration large enough to explain collapse.
**Does not meet the strict "bright star truly sits at grid at full flux" criterion**, but
does **not** support STOP-for-alignment.

Scripts: `tmp/verify_check1_alignment.py`, `tmp/verify_check1_aperture.py`.

### Check 2 -- fresh draft_435 with fix vs frozen baseline

Fresh run: `tmp/verify_check2/20260811T210053Z/` (799 s, current src_py on draft_435 inputs).

| Metric | Frozen 435 | Fresh + fix |
|--------|------------|-------------|
| Targets in summary | 169 | 165 |
| Trust GREEN / YELLOW / RED | 75 / 69 / 25 | **18 / 16 / 131** |
| GREEN+YELLOW -> RED regressions | -- | **106 / 165** |
| Field lc_rms median | 0.0755 | 0.0939 |
| Pool comp_rms (comparison_stars_per_target) med | (not recomputed on frozen) | **0.0204** (n=302; 221 <= 0.03) |

Anchors unchanged (spot checks):
- **BO CVn**: trust GREEN -> GREEN; lc_rms 0.1474 -> 0.1474; comp_path default; n_good_comp 3
- **FY CVn** (1496278752372040832): trust GREEN -> GREEN; n_good_comp 7 -> 8

Failure pattern (106 regressions): mostly Gaia-only targets; n_good_comp **8 -> 0**;
trust_reason "0 clean comps | hard: no clean comps; LC quality: no_data".

**Check 2 verdict: FAIL.** Fix bundle regresses draft_435 field-level trust (144 GREEN/YELLOW
-> 34) even though named VSX anchors (BO CVn, FY CVn) hold. **Do not close.**

### Pass/fail for closing
| Gate | Result |
|------|--------|
| Check 1 (cause = matching, not alignment) | **PASS** (with flux-deficit caveat on 27 frames) |
| Check 2 (435 no regression) | **FAIL** |
| --fast | **PASS** |
| **Close fix?** | **NO** |

## Errors (if any)
- `tmp/verify_check2_draft435.py` JSON dump failed on int64 serialization after pipeline
  completed (metrics captured above).

## Files changed
- `tmp/verify_check1_alignment.py` (repo root path fix)
- `tmp/verify_check1_aperture.py` (new)
- `tmp/verify_check2_draft435.py` (new)
- `dev/results/CURSOR_RESULT_draft505_comp_verify.md` (this file)
