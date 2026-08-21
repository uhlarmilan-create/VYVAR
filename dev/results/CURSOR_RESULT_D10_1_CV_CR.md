CURSOR RESULT - 2026-08-21 (D10-1 CV vs CR measurement)

What I did
Pre-registered slope comparison on frozen era-516 NoFilter snapshot
(`draft_000516_snapshot_era03_20260820`): ensemble-calibrated pre-CT mags vs
Gaia DR3 Table 5.9 Johnson V / Cousins Rc catalogs. Mechanical decision rule
applied. No `src_py/` changes.

## Baseline

| Ref | SHA | Note |
|-----|-----|------|
| Local HEAD | `5d015fc` | ERR-518 series tip (not yet on origin) |
| origin/main | `8dea595` | Expected post-push target per task brief |

## Pre-registration (unchanged)

**Method:** `resid_X = mag_ensemble_night - mag_catalog_X` for X in {V, Rc};
OLS `resid_X = a_X + b_X * (BP-RP)`; report slopes in mmag/mag BP-RP.

**Catalog transform:** `gaia_johnson.transform_gaia_to_johnson` with
`GDR3_TABLE59_COEFFS` (Gaia DR3 CU5 Table 5.9; `src_py/gaia_johnson.py:52-56`).

**Ensemble mag (pre-CT):** nightly median of `mag_inst + zp_frame` where
`zp_frame = median(Gaia_G_comp - mag_inst_comp)` across global comp pool
(same frame-shared ZP as `photometry_core.py:4546-4579`; pre colour-term).

**Decision rule:** if `max(|b_V|,|b_R|)/min(|b_V|,|b_R|) >= 2` AND
`||b_V|-|b_R|| > 3*stderr_combined` then smaller-|b| band wins (CV or CR);
else INCONCLUSIVE (default mapping stays CV by AAVSO prevalence).

**Expected sign (pre-run):** red-leaning CMOS + V catalog should give
`b_V - b_R < 0` (V residual slope more negative than R).

**Assumption:** single anchor night, shared frames; airmass/colour terms
common-mode within fit.

## Star set

Source: `masterstars_full_match.csv` on frozen snapshot, criteria:

- `zone == linear`
- not `vsx_known_variable`
- `source_state` in {DETECTED_P1, DETECTED_P2} (clean isolation)
- finite BP-RP and G within Table 5.9 validity (see D10-2)
- >= 33 finite ensemble frames (134 frames night; min = max(10, n/4))

| Metric | Value |
|--------|-------|
| n_stars in fit | **2076** |
| BP-RP span | **2.34 mag** (0.46 - 2.80) |
| Isolation widen | **not needed** (span > 1.0) |
| Airmass range | **1.013 - 1.219** |

Exclusions: **1508** star-criterion rows in
`dev/results/context/session_20260821_d10_1/exclusions.csv` (top: 1168
non-linear zone, 173 VSX, 108 no proc photometry, 43 non-clean source_state,
11 BP-RP out of range, 3 G out of range, 2 sparse frames).

Sandbox: `dev/sandbox/d10_1_cv_cr_measure.py`

## Slopes and verdict

| Band | b (mag/mag BP-RP) | b (mmag/mag) | stderr(b) |
|------|-------------------|--------------|-----------|
| V | +0.00164 | **+1.6** | 0.067 |
| Rc | +0.49547 | **+495.5** | 0.067 |

| Rule quantity | Value |
|---------------|-------|
| \|b\| ratio max/min | **302** (>= 2) |
| \|\|b_V\| - |b_R|\|\ | **0.494** |
| 3 * stderr_combined | **0.286** |
| sign(b_V - b_R) | **-0.494** (matches pre-registered expected sign) |

**Mechanical verdict: CV** (|b_V| << |b_R|).

Plot: `dev/results/context/session_20260821_d10_1/residual_vs_bprp.png`

## Interpretation (SUPERSEDED by D10-1b - 2026-08-21)

**The paragraph below is retracted.** D10-1b provenance audit and raw mag_inst
probe show: (1) no V-transform on the LHS - ZP uses Gaia G only; (2) flat b_V
is **not** explained by G-scale residuals being flat; (3) b_R ~ +495 tracks
**d(V-Rc)/dc**, not d(G-Rc)/dc alone. Raw counts (no ZP) confirm **V-like**
effective band (b_V ~ flat, b_R ~ +466). **Mechanical CV verdict stands** on
both fits. See `dev/results/CURSOR_RESULT_D10_1B_CV_CR.md`.

~~1. **G-anchored ensemble:** production ZP uses Gaia **G** catalog mags...~~

~~2. **Sign sanity:** `b_V - b_R` negative as pre-registered...~~

3. **GH CVn T1-abs (era arc):** baseline |calib - Gaia->Johnson V| **141 mmag**
   beats candidate **332 mmag** on GH check star (`CURSOR_RESULT_DAO_GAIA_ERA_02_EXECUTE.md`).
   GH target BP-RP **0.704** vs pinned comp median **0.758** (delta -0.054).
   **Consistent with CV:** V-catalog alignment already closer for GH; a CR export
   would not improve that axis.

## Pinned submission colour systematics (mmag)

Using measured slopes * delta(BP-RP) target - comp_median:

| Target | d(BP-RP) | @ b_V (CV) | @ b_R (CR) |
|--------|----------|------------|------------|
| BO CVn | -0.030 | **-0.05** | -15.0 |
| FW CVn | +0.034 | **+0.06** | +16.8 |
| GH CVn | -0.054 | **-0.09** | -26.7 |

FW is the only pinned target with target redder than comp ensemble; CR would
inject ~17 mmag colour systematic vs ~0.06 mmag under CV at the measured b.

## D10-2 (validity-range guard)

**Present.** `gaia_johnson.py:38-42` defines `BPRP_MIN=-0.5`, `BPRP_MAX=5.1`,
`G_MAG_MIN=8.0`, `G_MAG_MAX=16.0`; enforced in `transform_gaia_to_johnson`
lines 140-155 (returns `ok=False` with reason outside range). This run also
excluded 11 stars with BP-RP and 3 with G outside those bounds at selection.

## CR branch consequence (information only)

Choosing **CR** would change comp catalog magnitudes in export (Gaia G -> Cousins
Rc transform for ensemble reference), a **number-changing** step with anchor/pin
implications; separate task before any upload. Current mapping **CV** unchanged.

## Docs impact (DOCS-SYNC)

None (measurement-only task). No edits to `docs/`.

## Artifacts

- `dev/results/context/session_20260821_d10_1/summary.json`
- `dev/results/context/session_20260821_d10_1/star_residuals.csv`
- `dev/results/context/session_20260821_d10_1/exclusions.csv`
- `dev/results/context/session_20260821_d10_1/residual_vs_bprp.png`
- `dev/sandbox/d10_1_cv_cr_measure.py`

STOP - Milan decides the band letter; mapping/export change is out of scope.
