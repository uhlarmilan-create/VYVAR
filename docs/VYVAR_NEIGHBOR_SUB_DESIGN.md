# VYVAR NEIGHBOR-SUB -- design (TODO-PSF-NEIGHBOR-SUB)

Status: **DESIGN for review** (no code yet). Grounded read-only at HEAD fe8201c + post
FWHM-CONSISTENCY line drift noted below. ASCII only.

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

**Plumbing gap (pre-implementation):** `_load_adaptive_blend_map` (`photometry_core.py:5293`) currently
caches only `(is_blended, nn_dist_fwhm)` per `catalog_id`. NEIGHBOR-SUB needs the full worklist row
(`nn_catalog_id`, `delta_mag_nn`, positions). Extend the loader or read `crowding_targets.csv`
directly when `psf_neighbor_sub_enabled`.

---

## 3. Algorithm (per blended target, per frame)

a. Gather the target + its contaminating neighbour(s) positions (from the worklist + WCS->pixel).

b. Fit the ePSF to the NEIGHBOUR ONLY: amplitude + sub-pixel centroid, ePSF SHAPE FIXED. Reuse the
   existing fitter (`psf_photometry_stars`, `psf_photometry.py:2067`, run on a `star_positions`
   frame that contains the neighbour; it already returns flux + fitted position + model + chi2).

c. Render the fitted neighbour model and SUBTRACT it from a local stamp of the frame around the
   target (work on a stamp, not the whole frame, for speed and locality).

d. Run the EXISTING aperture extractor (`photometry_core` CircularAperture path,
   `_catalog_only_fixed_aperture_flux` / `enhance_catalog_dataframe_aperture_bpm`, ~1479 / ~9313)
   on the TARGET in the residual stamp -- same aperture/annulus radii (`VY_FWHM_GAUSS`-based) as normal.

e. Emit the target's aperture flux from the residual PLUS bookkeeping: `neighbor_subtracted=True`,
   `n_neighbors_subtracted`, `subtracted_neighbor_flux`, `neighbor_fit_chi2` / `residual_rms`.

If multiple contaminating neighbours fall within the annulus, fit+subtract each (or jointly) before
the aperture step.

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

- Bad neighbour fit (high chi2 / residual_rms above threshold, or fitted position far from catalog
  neighbour position) -> DO NOT subtract; fall back to plain aperture; flag `neighbor_sub_failed`.
- Over-subtraction risk: fitted neighbour centroid within ~0.5 FWHM of TARGET -> skip subtraction,
  flag `ambiguous_blend`.
- Saturated neighbour or target -> skip (PSF fit invalid on clipped cores).
- Neighbour is itself a science target / variable -> still fine to subtract for THIS target's
  measurement; never mutate a shared frame used by others -> always work on a per-target local stamp
  copy.

---

## 7. Trust integration

New per-measurement columns: `neighbor_subtracted`, `n_neighbors_subtracted`,
`subtracted_neighbor_flux`, `neighbor_fit_chi2`. Surface a per-target summary into comp_quality /
trust inputs so the gate can down-weight or YELLOW a target whose result leaned on a poor subtraction.

`lc_quality_flag`: a target measured via NEIGHBOR-SUB carries a flag so downstream/AAVSO export is
transparent about the deblend.

---

## 8. Config / gating

- New flag `psf_neighbor_sub_enabled` (default OFF), plus thresholds: `neighbor_sub_chi2_max`,
  `neighbor_sub_min_sep_fwhm` (~0.5), reuse `nn_contam_dmag` (or alias from `_PSF_QUALITY_THRESH`).
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

**Recommended build order once approved:**

1. Synthetic harness item (controlled blends) -- define acceptance envelope FIRST.
2. Subtract+aperture core reusing `psf_photometry_stars`.
3. Guards + bookkeeping.
4. Trust integration.
5. Validate on h & chi Per.

All OFF until synthetic envelope + real-field scatter confirm gain.

---

## Cross-references

- Crowding worklist (corrected FWHM): `docs/VYVAR_HCHIPER_CROWDING_RECOMPUTE.md`
- ePSF FWHM denominator context: `docs/VYVAR_EPSF_FWHM_TEST.md`
- Validation harness: `docs/VYVAR_VALIDATION.md`, `tests/validation/README.md`
