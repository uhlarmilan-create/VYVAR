# VYVAR -- Audit closure (final register)

**Date:** 2026-08-04
**Commit:** `20dde2b` (batch E GATE 2 closure)
**Purpose:** Referee-facing deliverable. Every audit item, final disposition, evidence pointer,
and implementation batch. Limitations: `docs/VYVAR_LIMITATIONS.md`.

---

## Audit status

The **science audit is closed** (2026-08-04). Batch D (GATE 1) and batch E (GATE 2) re-cuts are
complete. Anchor fingerprints pushed after physical re-cut from calibrated lights. Two **future
threads** remain outside the audit: **WIDE-ERR** (wide-rig error bars) and **MASTERSTAR stacking
architecture** (enhancement, not a correctness blocker).

---

## Closure register (final)

| ID | Item | Final state | Evidence | Commit |
|----|------|-------------|----------|--------|
| P-10 | Sky-surface sign error | **FIXED** | `test_preprocess_sky_surface.py` | pre-closure |
| SKYSF | In-place guard | **FIXED** | `pipeline.py` | pre-closure |
| I-12 | PM logging | **FIXED** | audit t2 | pre-closure |
| T1 | Export time_base | **FIXED** | audit | pre-closure |
| D10-2 | Gaia-Johnson guard | **FIXED** | Stage 1 | pre-closure |
| D10-1 | CV->CR band | **FIXED** | Milan decision | pre-closure |
| sigma_pp | Estimator revert | **FIXED** | Milan decision | pre-closure |
| threshold 3.8 | DAO threshold | **FIXED** | P-10 bundle | pre-closure |
| D1-3 | Master flat docs | **CLOSED** | DECISIONS | pre-closure |
| **D5-1** | Aperture provenance | **DOCUMENTED** | Step 1g; A-1b | batch A |
| **A-1** | SNR-table differential | **DOCUMENTED** | Step 1d-1g; ~144 mmag fixture | batch A |
| **A-9** | PSF scale | **DOCUMENTED** | Step 1f | batch A |
| **D1-1 / CR-1** | Cosmic-ray rejection | **FIXED** | E.3 astroscrappy; physical re-cut | batch E |
| **D5-2** | Flux vs G compression | **FIXED** | B-revised + E.5 gate; slope -0.318 -> -0.491 | batch E |
| **I-11** | Howell sky term | **FIXED** | batch D `683fba1` | batch D |
| **I-04** | Ensemble scatter | **FIXED** | batch D | batch D |
| **I-03** | Omitted Howell terms | **DOCUMENTED** | legacy path unused on anchor | batch D |
| **P-02** | Scintillation | **FIXED** | batch D wired | batch D |
| **A-6** | sigma_sys floor | **DOCUMENTED** | fit anomaly; not applied | batch D |
| **U-09** | DATE-OBS per rig | **DOCUMENTED** | home rig verified | batch A |
| **Part 0c** | Delta pairing | **FIXED** | E.1 source_file merge | batch E |
| **DAO centroid** | Aperture placement | **FIXED** | E.2 WCS guard | batch E |
| **T4-1** | Detection on resampled | **FIXED** | E.4 N_equiv=3.78 | batch E |
| **C-1/C-2** | Saturation admission gate | **FIXED** | E.5 admission_sat_peak_frac | batch E |
| **29** | Anchor re-cut VL-ANCHOR-WCSINV | **FIXED** | physical re-cut; SHA 5bccd85a/7fdcdca4 | batch E |
| **D1-2** | Linearity curve | **DEFERRED** | dome-flat ramp per sensor | -- |
| **WIDE-ERR** | Wide-rig err underquote | **OPEN** | H1-global; future thread | -- |
| **TODO-B / Steps 1-6** | MASTERSTAR stack | **QUEUED** | enhancement thread | post-audit |

Reports: `dev/results/CURSOR_RESULT_batch_{A,B_revised,C,D,E,physical_recut,final_closure}.md`.

---

## Closing statement

The VYVAR photometry pipeline underwent a twelve-domain scientific audit (VYVAR_AUDIT_FINAL.md,
22 findings plus 8 raised during remediation). All findings are resolved, documented as
accepted limitations, or deferred with a stated reason and route. The differential-aperture
systematic (A-1) is documented with its mechanism and recommended curve-of-growth correction
(Stetson 1990). The bright-end flux compression (D5-2) was traced to detector
saturation/non-linearity on stars above ~70% full well and is resolved by a saturation
admission gate excluding such stars from the comparison ensemble; the physical re-cut confirmed
the fix moved the bright-bin flux-vs-magnitude slope from -0.318 to -0.491, removing the
compression. The error budget now includes per-epoch scintillation (Young 1967 / Osborn 2015).
Cosmic-ray rejection (L.A.Cosmic / van Dokkum 2001) was added to preprocessing. The anchor was
re-cut on the full production path from calibrated lights, and the published differential light
curves were verified stable (matched-star differential delta -23 mmag) against nightly sky
variation. The pipeline is scientifically sound for differential photometry on the validated
home rig.

---

## Known limitations

One sentence each; full detail in `docs/VYVAR_LIMITATIONS.md`.

- **A-1 aperture differential:** ~144 mmag expectation for G 8-9 without COG correction; Stetson
  1990 fix identified; deferred to a future wave (**DOCUMENTED**).
- **A-9 absolute PSF scale:** estimators disagree 2.4-4.9 px; not blocking differential results
  (**DOCUMENTED**).
- **D1-2 detector linearity:** per-sensor dome-flat ramp required; deferred (**DEFERRED**).
- **WIDE-ERR wide-rig error budget:** quoted err underquoted ~2x; error bars only, not fluxes;
  fix routed (Honeycutt 1992 LOO + photon-term audit) before wide-rig submission claims err bars
  (**OPEN**).
- **D1-1 / CR-1:** cosmic-ray rejection added in batch E (**FIXED**).
- **U-09 timing convention:** home rig (BO CVn) verified; other rigs need per-rig confirmation
  (**DOCUMENTED**).
- **Gaia proper-motion:** no-op on local DR3 build; awaits DR4 (**DEFERRED**).

---

## Out of scope (future work)

The **MASTERSTAR stacking architecture** (frame-selection metric, median/sigma-clip stack,
coaddition per Zackay & Ofek 2017) is an **enhancement**. The pipeline is scientifically sound
with a single-frame MASTERSTAR; stacking is not an audit-correctness item.

---

## Evidence index

| Report | Content |
|--------|---------|
| `dev/results/CURSOR_RESULT_batch_B_revised.md` | D5-2 mechanism |
| `dev/results/CURSOR_RESULT_batch_D.md` | GATE 1 re-cut |
| `dev/results/CURSOR_RESULT_batch_E_physical_recut.md` | Physical re-cut + GATE 2 verify |
| `dev/results/CURSOR_RESULT_wide_error_diag.md` | WIDE-ERR |
| `dev/results/CURSOR_RESULT_final_closure.md` | This closure execution |
| `docs/VYVAR_AUDIT_FINAL.md` | Domain synthesis |
| `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` | Live register mirror |
