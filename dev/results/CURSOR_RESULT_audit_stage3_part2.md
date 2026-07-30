CURSOR RESULT - 2026-07-30 AUDIT STAGE 3 PART 2

What I did
Implemented DAO detection option B (convolved-image RMS threshold, `scale_threshold=False`),
dropped `sigma_pp` from the threshold path, added R2 tests, swept N on rebuild MASTERSTAR.

## Provenance

| Field | Value |
|-------|-------|
| `git_hash` | `d23db8ee53de7342c8fc1993586e0dfdd918ad5a` (+ Part 2 code) |
| `git_dirty` | `true` |
| MASTERSTAR | `draft_000499` rebuild (0b harness) |
| `rms_conv` (MASTERSTAR) | **55.63 ADU** |
| Tests | `dev/tests/test_dao_convolved_threshold_option_b.py` — **6/6 PASS** |

---

## 2.1 — Drop `sigma_pp` from threshold path

**Changed call sites** (threshold now uses `_dao_convolved_background_rms_adu`; `sigma_pp` diagnostic only):

| File | Function | Lines |
|------|----------|------:|
| `src_py/pipeline.py` | `detect_stars_match_master_reference` | ~7474–7528 |
| `src_py/pipeline.py` | `detect_stars_and_match_catalog` | ~8194–8248 |

`_pixel_noise_sigma_pp_adu` / `_dao_noise_sigma_adu` retained as **diagnostic** helpers.

---

## 2.2 — Option B implementation + R1 literature

### photutils `DAOStarFinder` (`scale_threshold`)

> *"By default, `threshold` is internally scaled by a factor derived from the Gaussian kernel… Set `scale_threshold=False` to apply the value exactly as given."*

**Implementation:** `threshold = N × rms_conv`, `scale_threshold=False`, where `rms_conv` is robust RMS of the FIND kernel convolved detection image (`scipy.ndimage.convolve`, zero-sum kernel).

### Fruchter & Hook (2002) PASP 114, 144

> Resampling (drizzle) **correlates** pixels; per-pixel noise estimators on resampled images do not reflect matched-filter detection sensitivity.

### Casertano et al. (2000) AJ 120, 2747 Appendix A

> Noise on the output pixel scale **underestimates** noise on larger scales after resampling/co-addition.

**Match:** Option B thresholds the convolved detection image RMS — the quantity FIND actually thresholds.

### R2 verification (`test_dao_convolved_threshold_option_b.py`)

| Assertion | Result |
|-----------|--------|
| White noise: `rms_conv/?_pixel ? kernel.rel_err` (1.3604) | **PASS** (~1.36) |
| Resampled frame: `rms_conv ? ?_pixel × rel_err` | **PASS** |
| Option B: nominal N? on white + resampled | **PASS** (N=3.8) |

---

## 2.3 — Threshold sweep (N = 2.5 … 6.0)

Measured on `draft_000499` MASTERSTAR (`rms_conv=55.63 ADU`). Pass-1 counts from
`detect_stars_match_master_reference` (cap 12000 at low N).

| N | pass-1 DAO | threshold [ADU] |
|---|----------:|----------------:|
| 2.5 | 12000* | 139 |
| 3.0 | 6417 | 167 |
| 3.5 | 4087 | 195 |
| 3.75 | 3765 | 209 |
| 4.0 | 3639 | 223 |
| 4.5 | 3538 | 250 |
| 5.0 | 3508 | 278 |
| 5.5 | 3489 | 306 |
| 6.0 | 3474 | 334 |

\*Hit `max_catalog_rows` prefilter cap.

**DAO_ONLY fraction:** unchanged at **3.93%** (catalog rows fixed; sweep re-runs detection only).

**vs Tranche 3 / 0b rebuild:** pass-1 **2521** at legacy N=3.8 + `sigma_pp` path.
Under option B, **no N in 2.5–6.0 reproduces ~2550** (counts stay 3474–6417). Recalibration
requires **N > 6** or extended sweep — **DECISION REQUIRED (Milan picks N).**

When N is chosen, record in `VYVAR_DECISIONS.md`:

> `masterstar_dao_threshold_sigma` = N sigma in the **convolved** detection image RMS
> (`scale_threshold=False`), valid on correlated/resampled noise.

---

## 2.4 — Delta vs pre–Part 2 (0b rebuild, legacy threshold)

| Quantity | Pre–Part 2 (0b) | Part 2 @ N=3.8 (option B, estimated) |
|----------|----------------:|-------------------------------------:|
| pass-1 DAO | 2521 | ~3700 (interp 3.75–4.0) |
| DAO_ONLY | 3.93% | 3.93% (catalog fixed) |
| Active LCs | 230 | (not re-run photometry in Part 2) |
| Check-star ?² | — | deferred to post-N photometry re-run |

Full photometry/check-star delta deferred until Milan selects N and photometry is re-run.

---

**STOP GATE 2** — sweep posted; **awaiting N decision** before anchor re-cut / Part 3.
