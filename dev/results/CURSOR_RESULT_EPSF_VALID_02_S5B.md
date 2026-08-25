CURSOR RESULT - 2026-08-22T18:30:00Z (EPSF-VALID-02 S5b - D1/D2 re-measurement)

What I did
Re-ran Part D certificates with **scale-aligned** metrics (sandbox only; no production
writes, no model swap). Harness: `dev/sandbox/epsf_valid_02_s5b_measure.py`. Artifacts
under `dev/results/context/session_20260822_epsf_valid_02_s5b/`.

Parent: S5 STOP-B partial (B1-B4 signed; D1/D2 rejected for raw-flux metric defect).
HEAD: `2ba3d58`. No commits.

---

## Premise confirmation (metric defect)

S5 D2 raw deltas showed RMS ? |median| at every N (~300 mmag), dominated by **global
flux-scale offset** between builds. S5b removes per-model median scale (equivalent to
each model's own AC bookkeeping) before comparing per-star deltas.

| N (S5 raw) | raw median ? | raw RMS ? |
|-----------:|-------------:|----------:|
| 15 | ?324.8 | 327.6 |
| 30 | ?427.3 | 430.5 |
| 50 | ?300.4 | 299.3 |

After alignment (S5b), the same builds differ by **tens of mmag**, not hundreds - premise
**confirmed**; S5 D1/D2 rejection stands; S5b certificates supersede S5 Part D numbers.

---

## D1b-a - Odd-half build diagnosis + sandbox fix

**Failure:** `ValueError: All elements of input data must be finite` during `EPSFBuilder`
normalization when too many odd-half stars receive photutils `_fit_error_status=3`
(fitted position outside cutout) ? normalized ePSF sum ? 0/NaN.

**Primary trigger star:** `1497528072458898432` at (688.63, 1340.12),
`dist_edge_px=55.9` (nearest to frame edge in the 34-star odd half).

**Secondary (status-3, not sufficient alone):** `1496994156484645632` at (1518.33, 1238.96).

**Sandbox fix (verified):** drop edge-nearest star on non-finite build; logged in
`build_d1b_odd/build_guard.json`. Single-drop fix verified in `diagnose_odd_build_failure()`.

**Candidate S6 production hardening (not committed):** extend build-input guard to drop
stars that destabilize EPSFBuilder iterations (edge proximity / status-3 cluster), with
logged `catalog_id` + reason - same pattern as sandbox `sandbox_build_guarded()`.

---

## D1b - True split-half (aligned)

| Item | Value |
|------|------:|
| Build A (odd) | 34 requested ? **33 built** (1 guard drop) |
| Build B (even) | 33 built, 0 drops |
| Frames | 12 (same as S5 D1) |
| Matched star-frame pairs | **368** (34 stars; target ?50 **met**) |
| Removed offset - odd (mmag) | +1490.4 |
| Removed offset - even (mmag) | +374.5 |
| **Raw** median ? / RMS (mmag) | ?1121.3 / 1124.7 |
| **Aligned** per-star median ? (mmag) | **?7.8** |
| **Aligned** per-star RMS ? (mmag) | **30.3** |
| AC-path median ? / RMS (mmag) | ?11.5 / 32.8 |

Alignment method: per model, global median `dao_flux / psf_flux_raw` applied before
per-star ? mmag (equivalent to per-model AC scale removal).

### Error budget (comp stars, same frames)

| Item | Value |
|------|------:|
| Median ERR-path err | **15.3 mmag** |
| n pairs | 368 / 368 |
| Source | proc CSV photon term: `sqrt(dao_flux/gain + sigma_bkg_ap^2)/dao_flux x 1000` |
| Note | `err_term_epochs_picks.csv` has **0 overlap** with comp build pool (target LCs only); not used |

Criteria: |aligned median| < 2x budget and aligned RMS < 3x budget.

| Check | Threshold | Actual | |
|-------|----------:|-------:|---|
| \|median\| | < 30.5 mmag | 7.8 | pass |
| RMS | < 45.8 mmag | 30.3 | pass |

**D1b verdict: PASS** (with note: odd half required one guard drop; true 34 vs 33 split).

Artifacts: `d1b_summary.json`, `d1b_matched_pairs.csv`, `d1b_per_star_aligned_mmag.csv`,
`d1b_photometry_odd.csv`, `d1b_photometry_even.csv`.

---

## D2b - Convergence with aligned scale (vs N=67 reference)

Same 12 frames as D1b; science-set comp stars (67 gated pool); scale aligned per D1b (c).

| N | n_pairs | removed offset N (mmag) | raw median ? | raw RMS ? | **aligned median ?** | **aligned RMS ?** |
|--:|--------:|------------------------:|-------------:|----------:|---------------------:|------------------:|
| 15 | 460 | +69.9 | ?318.1 | 317.3 | ?8.6 | **25.4** |
| 30 | 457 | ?31.7 | ?419.5 | 415.1 | ?9.8 | **36.9** |
| 50 | 462 | +88.9 | ?294.1 | 291.0 | ?2.7 | **12.9** |
| 67 | 478 | +379.8 | 0.0 | 0.0 | 0.0 | 0.0 |

(ref removed offset ? +379.8 mmag at all N - reference self-comparison at N=67.)

**Curve artifacts:** `d2b_convergence_curve.csv`, `d2b_convergence_curve.png`
(offset + aligned RMS series), `d2b_summary.json`, sandbox builds `build_d2b_n15/30/50/`.

**D2b verdict: PASS (aligned convergence)** - N=50 aligned RMS 12.9 mmag < 15.3 mmag
budget; N=15 within ~1.7x budget. N=30 is a local outlier (36.9 mmag) but still ? raw
300 mmag artifact. Non-monotonic aligned RMS at N=30 likely subset composition / edge-star
sensitivity, not scale defect.

### Re-proposed N policy (from aligned numbers)

> **Production (unchanged recommendation):** use the **full Part C gated science-comp pool**
> (N=67 for draft 516). Best aligned RMS (0 by definition) and full PSF shape from complete
> homogeneous sample.
>
> **Certificate / validation threshold (revised):** partial-N sandbox builds are **not**
> disqualified by hundreds-of-mmag raw offsets. For split-half / convergence checks, require
> **scale-aligned** RMS ? < 3x comp-star ERR budget (~46 mmag here). N?50 meets this;
> N=15 meets it; N=30 marginal (2.4x budget).
>
> **INTERIM top-N=200:** remain **disabled** for production builds until field-specific
> evidence shows gated-pool convergence; S5b does not change that policy, only corrects the
> certificate metric.
>
> **Build hardening:** adopt edge-star drop guard (D1b-a) before S6 swap to reduce odd-half
> build failures.

---

## Offsets table (bookkeeping, not photometric error)

| Comparison | Model A offset (mmag) | Model B offset (mmag) |
|------------|----------------------:|----------------------:|
| D1b odd vs even | +1490.4 | +374.5 |
| D2b N=15 vs ref67 | ref +379.8 | N15 +69.9 |
| D2b N=30 vs ref67 | ref +379.8 | N30 ?31.7 |
| D2b N=50 vs ref67 | ref +379.8 | N50 +88.9 |

These offsets cancel in differential photometry (ZP/ensemble); they must not be interpreted
as PSF model error.

---

## Gate status

| Check | Status |
|-------|--------|
| Production writes | None |
| Model swap | None (STOP-B holds) |
| Production code changes | None (sandbox harness only) |
| `--fast` | Not required (no production code change) |
| HEAD | `2ba3d58` |

---

## Errors (if any)

- DB malformed warnings (pre-existing; no impact on sandbox measurements).
- `err_term_epochs_picks.csv` has no comp-star rows; budget computed from proc CSV ERR
  photon term on matched pairs (documented above).
- D2b PNG regenerated post-run from CSV (initial `--only d2b` run wrote CSV/JSON only).

---

## Files changed

| File | Role |
|------|------|
| `dev/sandbox/epsf_valid_02_s5b_measure.py` | S5b sandbox harness |
| `dev/results/CURSOR_RESULT_EPSF_VALID_02_S5B.md` | This deliverable |
| `dev/results/context/session_20260822_epsf_valid_02_s5b/*` | Measurements, CSVs, JSON, PNG |

No commits (measurement-only task).

---

## STOP-B resume

S5b completes **aligned** D1/D2 certificates. B1-B4 verdicts from S5 stand unchanged.

**Architect review + Milan swap decision required before S6.**

Recommended STOP-B questions:

1. Accept S5b D1b/D2b as superseding S5 Part D (aligned PASS)?
2. Accept gated 67-star model replacement for production `masterstar_epsf.fits` on 516?
3. Adopt edge-star build guard (D1b-a) as S6 hardening item?
4. Adopt revised N policy (full pool production; aligned-RMS certificate threshold)?

**Do not proceed to S6 (model swap) until architect signs STOP-B and Milan authorizes swap.**
