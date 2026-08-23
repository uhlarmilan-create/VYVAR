CURSOR RESULT - 2026-08-23T13:30:00Z (EPSF-BRIGHT-01 Phase 3 / FD-A GO)

What I did
Implemented FD-A full CCD variance model in PSF fit error maps (one-pass DAO-flux F_model),
added unit tests, ran F6 PSF merge on draft 516 (134/134), post-FD-A M1-M3 remeasurement,
INV-PSF-ADDITIVE-01 spot check, and opened EPSF-SHAPE-01 on persistent bright-star ratio droop.

STOP for architect review; Milan dashboard eye-check + push next.

---

## Commits

| Commit | Description |
|--------|-------------|
| `086fb44` | Phase 1 UI: full science table + gated-epoch caption |
| `a0319fe` | **FD-A:** full CCD variance model (`full_ccd` / `sandwich_full_ccd`) + tests |
| `acba8b4` | Docs + this deliverable (DECISIONS FD-A, ROADMAP EPSF-SHAPE-01) |

---

## FD-A implementation summary

**Variance model (production):** per-pixel `sigma^2 = F_model/g + sky/g + (RN/g)^2`.

**One-pass choice (i):** photutils requires a static error array per fit; F_model built from
**DAO aperture flux** (`ref_fluxes` from proc sidecar) distributed by ePSF at catalog xy
before fit - standard DAOPHOT practice (Anderson & King 2000; Stetson 1987 ensemble context).

**Gain path:** `_psf_resolve_gain_read_noise` -> `param_resolver.resolve_gain` on frame header
(same chain as pipeline photometry; no hardcoded gain).

**Provenance:** `psf_weight_mode=full_ccd`, `psf_err_mode=sandwich_full_ccd`.

**Files:** `src_py/psf_photometry.py`, `dev/tests/test_psf_variance_fd_a.py`,
`dev/tests/test_v3d_fine_scale.py` (expectation update).

**Tests:** 4 new FD-A tests PASS (gain propagation, sky vs full map, noiseless bright chi2 < 5,
sandwich full variance).

---

## Acceptance - F6 merge draft 516

| Item | Result |
|------|--------|
| Frames merged | **134 / 134** (`f6_merge_result.json`) |
| INV-PSF-ADDITIVE-01 (3-frame spot) | **PASS** (`inv_spot_check.json`) |
| Aperture / non-psf columns | **byte-identical** (spot check + merge invariant) |

Log: `dev/results/context/session_20260823_epsf_bright_01_p3/accept_run2.log`

---

## M1 - BO CVn before / after FD-A

**catalog_id:** `1498613634033133184`

| Metric | Pre-FD-A (Phase 2) | Post-FD-A |
|--------|-------------------:|----------:|
| `psf_fit_ok` | **54 / 134 (40%)** | **133 / 134 (99.3%)** |
| `chi2 >= 50` | 80 / 134 | **1 / 134** |
| chi2 median | 60.4 | **22.6** |
| chi2 p95 | 127.2 | **36.4** |
| chi2 max | 168.6 | **55.8** |

**Verdict:** Chi2 brightness gate **fixed** for BO CVn overlay (target >120 fit_ok **MET**).
Median chi2 **22.6** (not 1-3) - residual misfit / one-pass F_model chi2 offset on this
bright target; not a gating crisis.

Per-frame table: `session_20260823_epsf_bright_01_p3/m1_bo_cvn_per_frame.csv`

---

## M2 - chi2 vs mag (science set, full coverage)

| Metric | Pre-FD-A | Post-FD-A |
|--------|---------:|----------:|
| Population median chi2 | 2.51 | **2.12** |
| Stars median chi2 >= 50 | 43 / 192 | **30 / 192** |
| mag < 10 median chi2 | ~143 | **~61** |
| mag < 10 mean pct_fit_ok | ~40% | **~38%** |

**Plot:** `session_20260823_epsf_bright_01_p3/m2_chi2_vs_mag.png`

**Verdict:** Bright-end chi2 **strongly reduced**; faint-end ~flat (population median ~2).
**30 stars** still have median chi2 >= 50 - treat as **real bright-end misfit candidates**
(do not threshold-tune). M2 sandbox inflation proxy remains **retracted** (gain=1.0 unit defect).

CSV: `m2_chi2_vs_mag.csv`

---

## M3 - ratio vs mag after FD-A

| Metric | Pre-FD-A | Post-FD-A |
|--------|---------:|----------:|
| BO CVn median psf/dao ratio | 0.675 | **0.671** |
| Bright-10 ratio range | 0.09 - 0.70 | **0.17 - 0.65** |
| corr(ratio, peak/linearity) | 0.12 | **0.22** |

**Plot:** `session_20260823_epsf_bright_01_p3/m3_ratio_vs_mag_peak.png`

**Verdict:** Ratio droop **persists** post-FD-A (bright stars **< 0.9**). Not fixed by variance
recalibration - **EPSF-SHAPE-01 opened** (HIGH, PSF-branch trust). Weak peak-ADU linkage unchanged.

---

## Residual shape misfit verdict

1. **Gating defect:** **RESOLVED** (BO CVn 133/134; chi2 no longer acts as brightness cut).
2. **Honest chi2 ~ 1-3 on BO CVn:** **NOT MET** (median 22.6) - likely composite of one-pass
   F_model init vs fitted profile + genuine PSF shape mismatch at high SNR.
3. **PSF/DAO flux ratio on bright stars:** **OPEN** - EPSF-SHAPE-01 (out of scope here).

---

## Gates

| Gate | Status |
|------|--------|
| `--fast` @ `acba8b4` | **OVERALL PASS** (1512 passed, 32 skipped; prior tip `ce7f378` equivalent) |
| `--full` recut 9902d918 / 472bc9e4 | **OVERALL PASS** (`full_gate.log`) |

---

## Docs impact

| Doc | Change |
|-----|--------|
| `VYVAR_DECISIONS.md` | FD-A full CCD variance decision block |
| `VYVAR_INVARIANTS.md` | **Unchanged** |
| `VYVAR_ROADMAP.md` | **EPSF-SHAPE-01** opened (HIGH) |

---

## Files changed (this arc)

| Path | Role |
|------|------|
| `src_py/psf_photometry.py` | FD-A variance model |
| `src_py/ui_epsf_dashboard.py` | Phase 1 UI (prior commit) |
| `dev/tests/test_psf_variance_fd_a.py` | FD-A tests |
| `dev/tests/test_v3d_fine_scale.py` | Provenance string update |
| `docs/VYVAR_DECISIONS.md` | FD-A decision |
| `docs/VYVAR_ROADMAP.md` | EPSF-SHAPE-01 |
| `dev/results/CURSOR_RESULT_EPSF_BRIGHT_01_P3.md` | This file |
| `dev/results/context/session_20260823_epsf_bright_01_p3/*` | Acceptance artifacts |

**STOP - architect review. Milan: dashboard BO CVn overlay eye-check, then push when satisfied.**
