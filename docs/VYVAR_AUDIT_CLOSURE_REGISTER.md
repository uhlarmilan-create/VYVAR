# VYVAR -- Audit closure register (30 items)

**Date:** 2026-07-31
**Source audit:** `docs/VYVAR_AUDIT_FINAL.md`
**Status legend:** CLOSED | FIXED | MEASURED | DECISION | QUEUED | BLOCKED

Steps **1--10** below are the active closure queue (ROADMAP). Remaining items are tracked but
not in the first execution wave.

---

## Closure queue (Steps 1--10)

| Step | ID | Item | Domain | Status | Depends on |
|------|-----|------|--------|--------|------------|
| **1** | **A-1** | Implement frame selection metric `I_j = F_j^2 / (sigma_j^2 * FWHM_j^2)` for MASTERSTAR stack ranking | 7 | **QUEUED** | -- |
| **2** | **A-2** | Selection rule: N_min=10, N_max=20, quality gate I_j >= 0.5 max(I_j) | 7 | QUEUED | A-1 |
| **3** | **A-3** | Median/sigma-clip stack combination (replace single-frame copy) | 7 | QUEUED | A-2 |
| **4** | **A-4** | Mandatory stack provenance in header + `pipeline_meta.json` | 7 | QUEUED | A-3 |
| **5** | **A-5** | Recalibrate `masterstar_dao_threshold_sigma` against stack noise/PSF | 7 | QUEUED | A-3, T4-1 |
| **6** | **A-6** | Split `DAO_ONLY` health metric by magnitude vs Gaia cap (17.5) | 7 | QUEUED | A-5 |
| **7** | **C-1** | Admission gate: predicted per-epoch SNR (`g_lim_*` + Labb sigma_bkg_ap) | 7, 8 | QUEUED | -- |
| **8** | **C-2** | Flag catalogue rows CONTEXT-ONLY vs PHOTOMETRY-CANDIDATE | 7 | QUEUED | C-1 |
| **9** | **CR-1** | Cosmic-ray rejection (L.A.Cosmic or equivalent) | 1 | QUEUED | -- |
| **10** | **T4-1** | Milan decision: detection noise on resampled frames (options A/B/C/D) | 2, 7 | **DECISION** | measurement in stage2 |

### Aperture closure Step 1 (finding A-1 -- SNR table radius)

**Status:** **A-1b CONFIRMED** (2026-08-01 Step 1g). F1 configuration (proxies at clamp 1.916 px,
disjoint from comps). G9 PASS; G6 FAIL; exact magnitude **open**.
**Step 1h diagnosis (2026-08-01):** G6 magnitude spread in numerator; denominator stable;
contamination excluded. Report: `dev/results/CURSOR_RESULT_closure_step1h.md`.
**Step 1i mechanism (2026-08-01):** I3 -- normalisation not placement (E5); HIGH EE from F(12)
issues; LOW EE rare placement. Report: `dev/results/CURSOR_RESULT_closure_step1i.md`.
**Step 1j (2026-08-01):** J-a -- per-star sky subtraction error in F(12); J1 slope **-0.285**
(vs -0.4); G 11.52/11.53 F(12) ratio **2.6x** at dG = 0.006 mag. Step 1i production annulus
claim withdrawn. Report: `dev/results/CURSOR_RESULT_closure_step1j.md`.
**Step 1k (2026-08-01):** Two F(12) defects proposed (additive sky + multiplicative compression).
K3: production slope **-0.296** (D5-2 opened). K-a (D1-2 non-linearity) **withdrawn in Step 1l**
(residual-vs-peak confounded; peak trend gone after aperture control).
**Step 1l (2026-08-01):** L-a proposed (aperture mechanism); K-a withdrawal stands. **Step 1l L-a
withdrawn in Step 1m** (COG normalisation does not reach -0.400).
**Step 1m (2026-08-01):** M-x -- normalised flux slope **-0.280** (H-ap -0.400 and H-2 ~-0.19
both rejected); measured `slope(log10 EE(r_ap) vs G)` = **-0.016**. D5-2 confirmed; mechanism
**open**. Report: `dev/results/CURSOR_RESULT_closure_step1m.md`.
**Reports:** `dev/results/CURSOR_RESULT_closure_step1g.md`; Step 1f **48.0 mmag VOID** (V11).
**Fixture:** `dev/tools/closure_a1_reference_fixture.py` (target-radius sweep in --emit)
**Harness:** `dev/tools/closure_step1f_differential_aperture.py` (Step 1g F1/G9)

| Finding | Verdict | Decisive measurement |
|---------|---------|----------------------|
| SNR-table differential (mmag) | **A-1b CONFIRMED** | Exact value **open** (G6 fail); proxy G 11.52 diagnostic **94 mmag** G 8-9 vs fixture **144.3 mmag**; Step 1f 48.0 mmag VOID |
| Absolute FWHM scale | **A-9 open** | estimators disagree 2.4-4.9 px; not blocking |
| SNR table clamp | binding (expected) | **2060/2649** on r_min; faint-end normal at 0.8 x FWHM |
| `aperture_snr_sizing` (S1) | **DEAD** | hardcoded 0.8/2.5 used |
| D5-1 Q1 per-frame FWHM | **No** | draft-constant `VY_FWHM_GAUSS = 2.395` |
| Role factors (S3) | label only | not applied on export path |

**Fix posture:** no mandatory patch on anchor differential grounds; option (iii) COG AC is the
direct D5-1 mechanism if Milan wants enclosed-flux normalisation. S2 ZP patch must not ship alone.

**ID note:** MASTERSTAR stack items **A-1..A-6** above are a separate queue.

---

## Register items 11--30

| ID | Item | Domain | Status | Notes |
|----|------|--------|--------|-------|
| 11 | P-10 sky-surface sign error | 3 | **FIXED** | `pipeline.py`; tests in `test_preprocess_sky_surface.py` |
| 12 | SKYSF-DOUBLE in-place guard | 3 | **FIXED** | Read `VY_SKYSF` before re-subtract |
| 13 | I-12 PM unavailable logging | 4 | **FIXED** | WARNING when pmra/pmdec absent |
| 14 | T1 export time_base truth | 12 | **FIXED** | Refuse non-BJD_TDB AAVSO export |
| 15 | D10-2 Gaia->Johnson range guard | 10 | **FIXED** | Stage 1; 1 comp outside range on anchor |
| 16 | D5-1 aperture provenance columns | 5 | **FIXED** | Step 1g: F1 config valid; G6 fail; exact delta_ap open; 48.0 mmag VOID |
| 16b | **D5-2** production flux vs G scaling | 5 | **MEASURED** | Step 1k: slope -0.296; Step 1m: COG-norm -0.280; mechanism **open** |
| 31 | **A-9** absolute PSF scale unresolved | 5, 7 | **MEASURED** | VY_FWHM_GAUSS 2.395 vs header 3.207 vs COG identities 4.0-4.9 px disagree; not blocking Steps 2-10; required before absolute flux/SNR claims |
| 17 | D1-3 master flat documentation | 1 | **CLOSED** | DECISIONS entry; builder gap noted |
| 18 | D10-1 unfiltered CV->CR band | 10 | **FIXED** | Milan decision; Stage 3 |
| 19 | sigma_pp drop / sigma_clipped_stats | 2 | **FIXED** | Milan decision Stage 3 |
| 20 | masterstar_dao_threshold 2.1->3.8 | 7 | **FIXED** | Bundled with P-10 |
| 21 | I-11 Howell sky on subtracted frames | 2 | **DECISION** | Options 1--3 documented; 0 prod epochs |
| 22 | I-04 ensemble scatter unmatched | 8 | **DECISION** | NaN+exclude vs inflate |
| 23 | I-03 omitted Howell terms | 2 | QUEUED | After I-11 decision |
| 24 | D1-2 linearity correction | 1 | **DEFERRED** | Step 1k K-a withdrawn Step 1l; peak-vs-residual confounded |
| 25 | P-02 scintillation in production err | 9 | **DECISION** | Do not wire without Milan |
| 26 | U-09 DATE-OBS convention per rig | 4 | MEASURED | BO CVn: shutter-open; others TBD |
| 27 | Part 0c delta pairing fix (source_file) | 7 | **QUEUED** | Harness bug; invalid tail stats |
| 28 | DAO centroid stability / aperture placement | 5, 7 | **QUEUED** | Part 0e M4; 19/156 targets > r_ap shift |
| 29 | Anchor re-cut (VL-ANCHOR-WCSINV) | all | **BLOCKED** | After T4-1 + A-5 + pairing fix |
| 30 | TODO-B proper coaddition (Zackay & Ofek) | 7 | QUEUED | After CR-1, A complete, per-frame PSF |

---

## Decision log (Milan, 2026-07-30)

| # | Decision |
|---|----------|
| 1 | Drop `sigma_pp`; revert to `sigma_clipped_stats` for DAO noise scalar |
| 2 | Unfiltered band: switch CV -> CR (Cousins R comparison mags) |
| 3 | Do NOT pick DAO threshold N from Part 2b sweep (R5) |
| 4 | GAIA-1/GAIA-2 remain deferred to DR4 |

---

## Evidence index

| Stage / part | Report |
|--------------|--------|
| Tranche 1 | `dev/results/CURSOR_RESULT_audit_t1.md` |
| Tranche 2 | `dev/results/CURSOR_RESULT_audit_t2.md` |
| Tranche 3 | `dev/results/CURSOR_RESULT_audit_t3.md` |
| Tranche 4 | `dev/results/CURSOR_RESULT_audit_t4.md` |
| Stage 0--2 | `dev/results/CURSOR_RESULT_audit_stage{0,1,2}.md` |
| Stage 3 Part 0a--0e | `dev/results/CURSOR_RESULT_audit_stage3_part*.md` |
| Closure Step 1 (aperture A-1) | `dev/results/CURSOR_RESULT_closure_step1.md` (VOID markers) |
| Closure Step 1b (A-1 repair) | `dev/results/CURSOR_RESULT_closure_step1b.md` (B.3/B.5/B.6 VOID) |
| Closure Step 1d (mmag + fixture) | `dev/results/CURSOR_RESULT_closure_step1d.md` (V8-V10 VOID) |
| Closure Step 1e (measurement repair) | `dev/results/CURSOR_RESULT_closure_step1e.md` (contamination VOID) |
| Closure Step 1f (admissibility + measure) | `dev/results/CURSOR_RESULT_closure_step1f.md` (V11-V14 VOID) |
| Closure Step 1g (F1 configuration) | `dev/results/CURSOR_RESULT_closure_step1g.md` |
| Closure Step 1h-1m (A-1 diagnosis) | `dev/results/CURSOR_RESULT_closure_step1{h,i,j,k,l,m}.md` |
| MASTERSTAR spec | `docs/VYVAR_TODO_MASTERSTAR_REFERENCE.md` |

---

*Register maintained at audit close 2026-07-31. Update item status in JOURNAL when steps complete.*
