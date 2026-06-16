# VYVAR -- Sigma budget (complete per-measurement uncertainty)

Date: 2026-06-15  
Status: **PARKED** -- sandbox only; no production wiring until chi-squared/dof ~ 1 on constant calibrator  
Blocks: Broeg-canonical IVW ensemble combine; TODO-GS8; TODO-MULTISET multi-rig IVW

---

## Load-bearing gate

Inverse-variance weighting (Broeg 2005; SPECULOOS-South arXiv:2005.02423) is optimal **only**
with a correct, complete, validated sigma. Current state:

| Component | Production today | Gap |
|-----------|------------------|-----|
| Photon + sky + read | `_photometric_error` -> LC `err` | OK per frame |
| Scintillation | **Absent** | Dominant for bright comps (Osborn et al. 2015) |
| ZP / ensemble weights | Night-level `comp_rms` (empirical stability) | Not per-frame; not theoretical floor |
| Broeg inflation | PyTICS on `comp_rms` only | Not on per-frame photon+scint floor |
| chi-squared/dof validation | **None** | Mighell 1999 export-only |

**Consequence:** IVW on Howell-only sigma would over-weight bright, scintillation-limited comps.

**Canonical until gate passes:** `delta_mag` = flux-sum (AIJ `tot_C_cnts` parity).

---

## Target design

Per-measurement sigma = **per-frame theoretical floor** [Howell + modified Young/Osborn scintillation]
**inflated per comp by measured excess variability** (Broeg 2005 iteration).

**Differential caveat:** scintillation is partly common-mode; the term in the *differential* error is
the **residual**, not full single-aperture sigma. chi-squared/dof on a constant star empirically pins the
surviving fraction -- do **not** tune `differential_fraction` to force chi-squared ~ 1.

---

## chi-squared infrastructure audit (read-only, 2026-06-15)

| Location | Role | Promotable to Phase-2A gate? |
|----------|------|------------------------------|
| `citations.py:310` | `mighell1999` in PSF export block only | **No** |
| `trust_flag_core.check_star_scatter` | RMS only | **No** |
| DoD-B constant gate | No-regression + RMS ratio | **Insufficient** |

**Verdict:** Need **new** reduced-chi-squared/dof gate on verified-constant calibrator LC.
Sandbox: `tmp/phase12/chi2_sigma_gate.py` (not committed).

---

## Implementation sequence

1. Sandbox (`tmp/phase12/`) -- `sigma_budget.py`, `chi2_sigma_gate.py`, harness, unit tests.
2. Measure on verified-constant calibrator + V0612 regression (unchanged `delta_mag`).
3. Production (after chi-squared/dof ~ 1 only): extend `_photometric_error` with scintillation;
   per-frame IVW weights; wire gate into Phase-2A export QA.
4. Flip ensemble combine to Broeg IVW canonical only after gate passes.

---

## Sandbox files (not in git)

| File | Purpose |
|------|---------|
| `tmp/phase12/sigma_budget.py` | Howell + Osborn sigma; Broeg inflation |
| `tmp/phase12/chi2_sigma_gate.py` | Reduced chi-squared/dof gate |
| `tmp/phase12/test_sigma_budget.py` | Unit tests |

---

## Citations

| Key | Reference |
|-----|-----------|
| `howell1989` | CCD equation |
| `broeg2005` | Photon floor + variability inflation |
| `young1967` | Young scintillation approximation |
| `osborn2015` | Osborn, Fohring, Dhillon & Wilson 2015 MNRAS 452, 1707 |
| `dravins1998` | Small-aperture scintillation regime |
| `murray2020speculoos` | arXiv:2005.02423 optimal weighted ALC |
| `mighell1999` | chi-squared-gamma (export; gate uses reduced chi-squared) |

---

## Acceptance criteria

- chi-squared/dof in [0.8, 1.2] on verified-constant calibrator with complete sigma.
- V0612 `delta_mag` unchanged; corr vs AIJ/SIPS not regressed.
- Only then: Broeg IVW canonical for ensemble combine.
