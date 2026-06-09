# VYVAR NEIGHBOR-SUB -- design (TODO-PSF-NEIGHBOR-SUB)

Status: **step 2 implemented** (core + A9 scoring; gated OFF; not wired to production measurement
sites yet -- step 2b). ASCII only.

---

## 1. Goal and why this shape

For a target whose aperture/annulus is contaminated by a bright NEIGHBOUR (a blend), measure the
target by: fit the ePSF to the NEIGHBOUR, subtract that model from the frame, then run the EXISTING
aperture photometry on the cleaned residual. The target is still measured by APERTURE (which VYVAR
trusts, cross-validates with SEP, and feeds comp_qa) -- PSF is used only to remove the contaminant.

Why not the grouper: simultaneous grouped PSF fitting (former "rule 2") was removed -- no precision
gain at 0.39"/px on draft 364, and is_blended (nn <= 1.5 FWHM) vs resolve_fwhm (>= 2.0) are mutually
exclusive (`photometry_core.py:5427-5430`). NEIGHBOR-SUB sidesteps that: it does not need PSF to BEAT
aperture; it only needs PSF to subtract a contaminant well enough that aperture sees clean sky+target.
That works at coarse bin2 (h & chi Per, ~1.3"/px) where the grouper did not.

---

## 2. Trigger / worklist (now on the CORRECTED FWHM)

Source: `crowding_targets.csv` (Part D), which after TODO-FWHM-CONSISTENCY uses `VY_FWHM_GAUSS`.
Per-row fields already present: `is_blended`, `nn_dist_fwhm`, `nn_catalog_id`, `nn_mag`,
`delta_mag_nn`, `n_neigh_2fwhm`.

Select a target for NEIGHBOR-SUB when ALL hold:

- `is_blended` (`nn_dist_fwhm <= 1.5`), with priority to hard blends (`nn_dist_fwhm < 1.0`); AND
- the neighbour is a CONTAMINANT, i.e. comparably bright or brighter:
  `delta_mag_nn <= nn_contam_dmag` (reuse `assess_psf_quality` criterion,
  `psf_photometry.py:2056-2063` -- the CSS_J161519.8 case: neighbour 3.5 mag brighter at
  1.46 FWHM; `nn_contam_dmag` default **2.5** allows neighbours up to 2.5 mag fainter); AND
- neither target nor neighbour is saturated; the neighbour has a finite `catalog_id`/position.

On h & chi Per (corrected FWHM): 375 L -> 39 hard / 58 blended; 380 L -> 34 / 53 -- a real worklist.

**Plumbing (step 2):** `_load_blend_worklist` + `BlendMapEntry` (`photometry_core.py:5290`) load the
full crowding row (`nn_catalog_id`, `delta_mag_nn`, neighbour ra/dec). Legacy
`_load_adaptive_blend_map` still returns `(is_blended, nn_dist_fwhm)` tuples.

---

## 3. Algorithm (per blended target, per frame)

### 3a. Gather

Target + contaminating neighbour(s) from the worklist (`delta_mag_nn <= nn_contam_dmag`) with
positions (catalog + WCS->pixel for production; stamp coords in A9).

### 3b. JOINT fit (prototype finding -- required)

**Prototype (synthetic blends):** fitting the NEIGHBOUR ALONE over-subtracts (target flux leaks into
the neighbour fit; residual bias -35% to -94%). **WRONG.**

**Correct core:** JOINT-fit target + neighbour(s) together (amplitudes + bounded sub-pixel centroids,
ePSF/Moffat SHAPE fixed), subtract ONLY the neighbour component(s) from a stamp copy, then aperture
the target in the residual. On ideal data this recovers target flux to ~0% across separations.

**Caveat:** ideal-data recovery is optimistic (exact PSF, clean partition). On real data the amplitude
split is ill-conditioned at full overlap (sep <~0.8 FWHM) and the ePSF will not match exactly.
Guards must be **fit-quality-driven** (not a blanket separation cut).

Implementation (step 2): `psf_neighbor_sub._joint_moffat_fit_subtract` (validation uses Moffat;
production step 2b will reuse grouped multi-source machinery in `psf_photometry_stars` ~2077).

### 3c. Subtract + aperture

Subtract neighbour model only from a **copy** of the local stamp (never mutate shared frames).
Aperture the target via `_catalog_only_fixed_aperture_flux` with `VY_FWHM_GAUSS`-based radii.

### 3d. Bookkeeping

`neighbor_subtracted`, `n_neighbors_subtracted`, `subtracted_neighbor_flux`, `joint_fit_chi2`,
`residual_rms`, `fit_condition` (amplitude-ratio proxy), fitted centroids.

Multiple contaminants: joint-fit all in the stamp together (open-Q3: joint, since prototype showed
joint is required even for one neighbour).

---

## 4. Insertion point

NEIGHBOR-SUB runs at the **per-frame measurement** stage, producing the target's (cleaned) aperture
flux + flags before LC assembly. `compute_lc_flux_method` (`photometry_core.py:5417`) is the
aperture-vs-psf **router** applied after `frame_results` are concatenated -- it does not measure flux.
The router and downstream LC build stay unchanged except they may see new bookkeeping columns.

**Likely hook sites (decide at implementation):**

| Site | When | Pros / cons |
|------|------|-------------|
| Pipeline per-frame catalog | `enhance_catalog_dataframe_aperture_bpm` (~9197) | Flux in CSV is already deblended; all Phase 2A paths benefit |
| Phase 2A remeasure | After `read_flux_from_csv` (~6225) for worklist targets | No pipeline rerun; re-opens aligned FITS per frame |
| Catalog-only path | `_catalog_only_merge_frame_flux` (~1655) | Covers catalog-only targets today |

Recommended: **pipeline hook** for standard DAO+aperture frames; **Phase 2A remeasure** fallback for
catalog-only / reprocess-without-pipeline. Both gated OFF by default.

The existing `blend_map` argument to `compute_lc_flux_method` is unused (`_ = blend_map` at 5441) --
NEIGHBOR-SUB does not revive rule 2; it is a separate measurement path upstream.

---

## 5. Reuse map (build mostly from existing pieces)

| Piece | Location |
|-------|----------|
| ePSF model + per-star fit (flux, position, model, chi2) | `psf_photometry_stars` (2067), `fit_moffat_psf_stars` (1565), `build_epsf_model` -> `masterstar_epsf.fits` |
| Worklist + neighbour identity/brightness | `crowding_targets.csv` via `_build_blend_targets_df` (195) |
| Contaminant criterion | `assess_psf_quality` `nn_contam_dmag` logic (2056-2063) |
| Aperture extraction on residual | `_catalog_only_fixed_aperture_flux` (1467) or `_aperture_flux_sky_per_star` (8983) |
| FWHM/aperture radii | `VY_FWHM_GAUSS` (post FWHM-CONSISTENCY), consistent with aperture path (`pipeline.py:9206`) |

**Genuinely new code (small):** render+subtract fitted neighbour model on a local stamp; per-target
orchestration; guards; bookkeeping columns.

---

## 6. Guards / failure modes (fail SAFE to plain aperture + flag)

**Step 2a (2026-06-08):** A9 realistic-mismatch diagnostic drove guard hardening. Goal: **FAIL-SILENT ~0**
(never emit a confident wrong flux; refuse and fall back to plain aperture).

**Separation floor (inclusive):** refuse when `nn_dist_fwhm <= neighbor_sub_refuse_sep_fwhm` (default
**0.8**). Covers sep 0.5 and 0.8 REFUSE-zone cells.

**Catalog-anchored sanity (production-ready; uses Gaia `mag` / `nn_mag`):**

- `neighbor_overfit`: joint-fit neighbour aperture flux brighter than catalog `nn_mag` by more than
  `neighbor_sub_max_neighbor_overmag` (default 0.3 mag).
- `target_undershoot`: recovered target flux fainter than catalog `mag` by more than
  `neighbor_sub_max_target_undermag` (default 0.2 mag after A9 calibration).
- `subtract_harmed`: mild-contamination case where cleaned flux < 95% of plain (subtraction hurt).
- `nonphysical_flux`: recovered flux <= 0; `low_recovered_snr` vs sky+read noise in aperture.

**Fit-quality (secondary):** centroid shift, target_amp <= 0, `no_improvement`, chi2/rms when no gain.

A9 realistic mismatch post-2a: **FAIL-SILENT 0**, HV PASS-RECOVER **~18%** -> **SAFE_LOW_YIELD**
(2b blocked; re-test at fine scale / improve ePSF first).

Never mutate a shared frame; always work on a per-target stamp copy.

---

## 7. Trust integration

New per-measurement columns: `neighbor_subtracted`, `n_neighbors_subtracted`,
`subtracted_neighbor_flux`, `neighbor_fit_chi2`. Surface a per-target summary into comp_quality /
trust inputs so the gate can down-weight or YELLOW a target whose result leaned on a poor subtraction.

`lc_quality_flag`: a target measured via NEIGHBOR-SUB carries a flag so downstream/AAVSO export is
transparent about the deblend.

---

## 8. Config / gating

- `psf_neighbor_sub_enabled` (default OFF), `neighbor_sub_chi2_max`, `neighbor_sub_residual_rms_max`,
  `neighbor_sub_refuse_sep_fwhm` (0.8, **inclusive** `<=`), `neighbor_sub_centroid_max_fwhm`,
  `neighbor_sub_nn_contam_dmag`, `neighbor_sub_max_neighbor_overmag` (0.3),
  `neighbor_sub_max_target_undermag` (0.2), `neighbor_sub_min_recovered_snr` (5.0).
- Applies ONLY to worklist (`is_blended` + contaminant) targets. Isolated-star photometry is
  byte-identical (untouched). Stays OFF in production until validated (mirrors PSF discipline).

Note: `psf_neighbor_include_fwhm` (config, default 3.0) is for the **grouper** neighbour catalog --
do not conflate with NEIGHBOR-SUB worklist selection.

---

## 9. Validation plan (proof before production enable)

**PRIMARY -- synthetic harness inject-and-recover (decisive, known truth):** inject a target of
KNOWN flux with a KNOWN bright neighbour at controlled separations (0.5, 0.8, 1.0, 1.3, 1.5 FWHM)
and delta_mag (-2..+2). Run NEIGHBOR-SUB. PASS: recovered target flux matches truth within noise AND
is closer to truth than plain aperture (contaminated). Sweep separation/delta to map where it helps
vs where guards refuse. **New harness item** (Tier-A family, e.g. **A9**) -- extends existing A1/A2
blend injection in `tests/validation/`.

**REAL DATA -- h & chi Per (375/380):** on the 39/34 hard blends, compare target LC scatter
(check-star RMS, comp_qa locus position) WITH vs WITHOUT NEIGHBOR-SUB. PASS: blended targets'
scatter improves (or unchanged) and constant comps stay flat. Reference case: CSS_J161519.8.

---

## 10. Byte-identity

With the flag OFF: numeric SHA `770966c3...` unchanged (no path touched). With it ON: only worklist
(blended) targets change; isolated stars remain byte-identical -> a SEPARATE baseline for the blended
subset, validated against synthetic truth and real-field scatter, not against old contaminated numbers.

---

## 11. Open design questions (decide before implementing)

- Fit the neighbour at its per-frame detected position, or at the masterstar/catalog position
  projected per frame? (Per-frame detection is more accurate but noisier on faint neighbours.)
- Single vs joint fit when 2+ contaminants share the annulus.
- Also subtract neighbours FAINTER than the target (`delta_mag_nn > 0`) when very close (< 1.0 FWHM)?
  They still bias annulus sky even if not the aperture core.
- Where to persist neighbor-subtracted residual stamps (debug/QA) without bloating drafts.

**Build order:**

1. ~~A9 envelope (step 1)~~ DONE.
2. ~~Joint-fit subtract + aperture core + A9 `neighbor_sub` scoring + PSF-mismatch variant (step 2)~~ DONE
   (`psf_neighbor_sub.py`, gated OFF).
3. Wire into per-frame measurement sites (step 2b): `enhance_catalog_dataframe_aperture_bpm` ~9197,
   Phase-2A remeasure ~6225, `_catalog_only_merge_frame_flux` ~1655.
4. Trust integration + h & chi Per real-field scatter validation (step 3).

All OFF in production until synthetic envelope + real-field scatter confirm gain.

---

## Cross-references

- **A9 acceptance envelope (steps 1-2, DONE):** `tests/validation/a9_core.py`, `psf_neighbor_sub.py`,
  `docs/VYVAR_VALIDATION.md` (ideal pass ~86%; mismatch ~21% on coarse grid)
- Crowding worklist (corrected FWHM): `docs/VYVAR_HCHIPER_CROWDING_RECOMPUTE.md`
- ePSF FWHM denominator context: `docs/VYVAR_EPSF_FWHM_TEST.md`
- Validation harness: `docs/VYVAR_VALIDATION.md`, `tests/validation/README.md`
