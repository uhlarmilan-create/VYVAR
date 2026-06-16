# VYVAR -- Canonical ensemble combination logic

Date: 2026-06-15  
Status: **CONDITIONAL HOLD** -- flux-sum `delta_mag` canonical until sigma budget validates

---

## Theory

Inverse-variance weighting (Broeg 2005 AN 326:134; SPECULOOS-South Murray et al. 2020
arXiv:2005.02423) is optimal **only** with a complete, validated per-measurement sigma.

Flux-sum equals IVW only in the photon-limited, all-constant limit (Howell 1989).

---

## Current production state

| Product | Method | Role |
|---------|--------|------|
| `delta_mag` | Flux-sum ensemble | **Canonical** science / AIJ-SIPS validation column |
| `mag_calib` | Differential + ensemble ZP offset | Reporting export (Workstream B) |
| Ensemble weights | Night-level `comp_rms` (stability) | Partial Broeg inflation via PyTICS; **not** per-frame IVW |

**Hold:** Do **not** flip ensemble combine to Broeg IVW until sigma = Howell + scintillation
(Young 1967 / Osborn et al. 2015) + Broeg inflation **and** reduced chi-squared/dof ~ 1 on a
verified-constant calibrator.

---

## Load-bearing next work

See `docs/VYVAR_SIGMA_BUDGET_SPEC.md` (PARKED sandbox). Same sigma machinery required for
TODO-GS8 (global ZP) and TODO-MULTISET multi-rig IVW.

---

## Audit anchors (2026-06-15)

- `ensemble_normalize` flux-sum: `photometry_core.py` (~2409+)
- `pytics_iterative_weights`: Broeg-like `comp_rms` inflation
- LC `err`: Howell photon+sky+read only (`_photometric_error`)
- No production chi-squared/dof gate (Mighell 1999 export-only)
