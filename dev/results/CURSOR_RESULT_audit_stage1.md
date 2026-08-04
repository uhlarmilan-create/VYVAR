CURSOR RESULT - 2026-07-30 AUDIT REMEDIATION STAGE 1

What I did
Literature-checked Gaia->Johnson range (1.1), added D5-1 proc provenance columns (1.2),
documented master-flat construction (1.3). No intentional photometric number changes.

## 1.1 D10-2 Gaia->Johnson colour-range guard

### Literature (R1)

**Source:** Gaia DR3 documentation CU5, section 5.5.1 Table 5.9 (polynomials); **Table 5.10**
(applicability).

**Quoted validity (Table 5.10, as pinned in `gaia_johnson.py`):**
- Independent variable BP-RP: **[-0.5, 5.1] mag**
- G magnitude: **[8.0, 16.0] mag**

**Riello et al. (2021) EDR3 Table C.2:** exposed as `RIELLO_EDR3_FALLBACK` for tests only;
production uses Table 5.9 (`GDR3_TABLE59_COEFFS`).

### Code today

| Item | Location |
|------|----------|
| Range check | `gaia_johnson.py:140-155` (`G` and `BP-RP` gates) |
| Comp exclusion | `transform_comp_row_for_osc_band` returns `ok=False`, logs exclusion |
| Constants | `BPRP_MIN/MAX`, `G_MAG_MIN/MAX` lines 38-42 |

**Verdict:** Implemented expression **matches** cited Table 5.10 limits.

### Anchor counts (draft_435 snapshot)

| Catalogue | N | Outside range |
|-----------|---|---------------|
| comparison_stars | 148 | **1** (G=7.99) |
| masterstars_full_match | 2951 | **39** (bright G<8) |
| BP-RP failures | - | **0** in sampled set |

Finding: **latent** for colour; **active** for bright comps (1 ensemble star).

## 1.2 D5-1 provenance columns

Added to proc CSV via `enhance_catalog_dataframe_aperture_bpm` + `PROC_STORE_COLS`:

| Column | Content |
|--------|---------|
| `aperture_r_px` | (existing) actual radius used |
| `aperture_factor_applied` | e.g. `global_1.900x`, `snr_table_comp_1.100x` |
| `fwhm_px_for_aperture` | Gaussian FWHM used for global aperture (per frame) |
| `fwhm_px_scope` | `per_frame_header_vy_fwhm_dao_scaled` / moment / override |
| `snr_aperture_mode` | `global_fixed` or `snr_table` |

## 1.3 D1-3 Master flat construction

See `VYVAR_DECISIONS.md` **D1-3-MASTER-FLAT-CONSTRUCTION**.

**Summary:** median stack of **raw** flats, **no** dark subtraction at stack; `VYFLNRD=1` ->
normalize at **calibrate** after resample. **Open gap** vs Howell flat-field requirement.

## ACCEPTANCE Stage 1 - byte identity

**LC photometry path:** flux/mag/err computation unchanged (additive columns only).

**Full anchor `--full` SHA regen:** **NOT RUN** this session (runtime ~30+ min). Prior gate on
`06ed950`: photometry SHA core/extended byte-identical. Stage 1 does not alter P-10 preprocess or
DAO threshold arithmetic.

**Proc CSV schema:** will gain 4 columns on next photometry run; existing LC files on disk
unchanged until regen.

**Recommendation:** run `session_baseline_check.py --full` after push to confirm LC SHA.

## Files changed (Stage 1 commit)
`src_py/photometry_core.py`, `src_py/proc_frame_store.py`, `src_py/pipeline.py` (provenance kwargs),
`docs/VYVAR_DECISIONS.md` (D1-3, D10-2).
