# VYVAR handoff — 2026-08-14

Start here in a new session. Full audit closure: `docs/VYVAR_AUDIT_2026_CLOSURE.md`. Current snapshot: `docs/VYVAR_STATE.md`.

---

## Repository

- **Branch:** `main`
- **HEAD at handoff:** see `git log -1` after push (parent `d3b6ecd` + two session-close commits)
- **`--fast`:** 1325 passed, 27 skipped (2026-08-14 pre-push)

---

## Anchors

| Draft | Role | Checksum manifest | Notes |
|-------|------|-------------------|-------|
| **435** | P1 mini / July snapshot | `dev/validation/anchor_435_checksums_post_restore_20260813.json` | 150 calibrated FITS byte-stable; photometry from July generation |
| **509** | ZP-clip regression | ad hoc baseline in `tmp/_inv_cal02_p1_509_baseline.json` | Same raw as 435; 5 comps after mag-diff gate |
| **510** | Live BO CVn anchor | **`dev/validation/anchor_510_checksums_a1_dao_fwhm_20260814.json`** (current) | Supersedes `anchor_510_checksums_placed_aperture_20260813.json`; **237 files changed** (see `anchor_510_checksum_diff_20260814.json`) |

**Draft 510 post A-1:** `aperture_px` **4.261**, check_scatter **0.008638**, GREEN trust, 134 frames, 5 comps.

**Draft 435 re-cut:** not done. Aligned tree has **proc sidecars only** (139); science FITS in `calibrated/` (150). Needs align + SNR table + export + Phase 2A.

---

## Iron rules (do not violate)

1. **Science pixels:** no in-frame cosmic-ray cleaning (removed 2026-08-12).
2. **Gates:** do not remove a verification gate on byte-identity evidence alone (`INV-GATE-REMOVAL`).
3. **Comparison membership:** once per draft after Phase 0+1 (`INV-COMP-MEMBERSHIP`); not per-frame variable comps for ZP.
4. **Measurement over plausibility:** audit and closure items require measured values, not estimates.
5. **Anchor re-cut:** Milan authorization before changing P1 golden SHA or draft archive products.

---

## Three-role model

| Role | Responsibility |
|------|----------------|
| **Milan** | Observing, telescope decisions, authorizes re-cuts and pushes |
| **Claude (architect)** | STATE/ROADMAP/JOURNAL/DECISIONS, task specs, review |
| **Cursor (implementer)** | Code, measurements, `dev/results/CURSOR_RESULT_*.md`, tests |

Session init: read `docs/VYVAR_STATE.md`, `VYVAR_ROADMAP.md`, latest `VYVAR_JOURNAL.md`, `VYVAR_CLAUDE_OPERATING_PRINCIPLES.md`.

---

## Session arc (2026-08-13/14) — what happened

1. **ZP clip removal** — per-frame MAD on ensemble ZP at `len(z)>=4` caused 509 BO CVn failure; clip deleted, Broeg weights kept.
2. **SAT-DIAG** — derive saturation from data; `SATURATE_ADU=16384` wrong units; placed-aperture raw peaks.
3. **Placed aperture** — stop peak-search hijack on faint comps.
4. **INV-CAL-01** — CAL-DIAG v2; dark SUM resample convention derived from measured pedestal.
5. **INV-CAL-02** — `VY_CALSTAGE` + DATASUM on in-place `calibrated/`.
6. **INV-GATE-REMOVAL** — policy documented.
7. **Full audit** — seven waves; xval vs photutils/sep; eight deletions; four C-class fixes.
8. **A-1 decision (2)** — SNR FWHM authority ? per-draft median frame DAO moment; draft 510 re-export verified.

**Physics (measurement-backed):** QHY294MM 14-bit samples in 16-bit container (65535 = clip); pedestal ~24.5 ADU/bin1 vs header `OFFSET=0`; ?10 °C dark pedestal-dominated (60 s = 120 s); block-sum dark resample correct for CMOS software binning, not for CCD on-chip binning.

---

## Open items — act on these

### Exposure ramp (D1-2 linearity)

- **Known:** SAT-DIAG uses DEFAULT_FRAC for linearity knee; one-sided test.
- **Decided:** measured dome-flat / exposure ramp per sensor (Howell 2006 §4.4).
- **Next:** Milan at telescope; no software substitute.

### WIDE-ERR

- **Known:** wide-rig quoted errors ~2× underquoted vs check-star scatter (~20 mmag excess); fluxes OK.
- **Decided:** defer sigma fix; route Honeycutt LOO + photon-term audit (`docs/VYVAR_LIMITATIONS.md`).
- **Next:** implement fix on wide rig; re-run check-star chi2 before publication claims on error bars.

### Decision (4) — fixed enclosed fraction

- **Known:** decision (2) moved EE +0.8 pp; 90% needs ~5.0–5.75 px on 510 comps; current ~86%.
- **Decided:** deferred; likely next architectural step if publication needs stated EE.
- **Next:** design per-star r90 from growth curves; estimate re-cut cost on 435+510.

### Draft 435 re-cut

- **Known:** SNR FWHM 2.395 px (underradiused vs frame PSF); no aligned science FITS.
- **Decided:** report only until Milan approves.
- **Next:** align from `calibrated/` ? DAO SNR table ? export ? Phase 2A ? optional `--full`.

### INV-DAG-01

- **Known:** `stamp_pipeline_stage` blocks Phase 2A re-run when `postprocess` already stamped.
- **Decided:** no fix; workaround = bypass `enforce_upstream=False` or trim stages (dev only).
- **Next:** product fix if routine re-photometry needed.

### W6-PROP (authorized, not implemented)

| ID | Next step |
|----|-----------|
| W6-PROP-03 | `VY_QCBG_PRE` at cal QC + `VY_QCBG` at preprocess |
| W6-PROP-01 | `detect_outliers` clip constants only |
| W6-PROP-05 | Wire library delete guards |
| W6-PROP-02 | Rename preprocess shim |

### C-EXPORT-GAP

- **Known:** `night_run` does not call AAVSO/VarAstro exporters.
- **Next:** wire optional export step or document manual UI path only.

### U-XVAL-COMP-RMS

- **Known:** VYVAR comp RMS 0.0101 vs photutils 0.0078 — gap **open, unexplained**; fixed 3 px harness vs ~4.26 px VYVAR radii explains non-comparability, not the 2 mmag.
- **Next:** extend `xval_run.py` to use per-star `aperture_r_px` from proc CSVs.

### U-P5-PRED

- **Known:** P5 PASS did **not** verify saturation at photometry radius (`peak_max_adu` is 7×7 box at centroid on raw).
- **Next:** if needed, measure peak within circular mask of `aperture_r_px` at placed centroid on raw.

### Register OPEN (see `docs/VYVAR_AUDIT_2026_REGISTER.md`)

U-PED-01, INV-CAL-01 ?_p=0 edge, F-B01/F-B02, P1-RECUT ledger stale, QHY294MM RN double-count, BPM sidecars, I-DETECT-OUT, A-9 PSF scale, I-03 legacy Howell terms.

---

## Key paths

| Purpose | Path |
|---------|------|
| Audit closure | `docs/VYVAR_AUDIT_2026_CLOSURE.md` |
| Register | `docs/VYVAR_AUDIT_2026_REGISTER.md` |
| xval harness | `src_py/xval_run.py` |
| A-1 code | `src_py/photometry_core.py` (`resolve_fwhm_px_for_snr_aperture_table`) |
| Session baseline | `dev/scripts/session_baseline_check.py` |
| Draft 510 diff | `dev/validation/anchor_510_checksum_diff_20260814.json` |

---

## Honest session notes (for the next implementer)

- **P5 was a bad prediction** — recorded in register; do not cite it as saturation-at-radius verification.
- **xval comp gap remains open** — target agreement is strong; comp scatter mismatch is not closed.
- **Draft 510 archive on disk diverged from prior manifest** — new manifest written; P1 golden ledger still stale for photometry SHA.
- **`fwhm_px_scope` in proc CSVs may still read `per_draft_gaussian_override` for annulus FWHM** while SNR radii use DAO authority — cosmetic provenance gap, not re-measured this session.

*Handoff written 2026-08-14. No further push without Milan.*
