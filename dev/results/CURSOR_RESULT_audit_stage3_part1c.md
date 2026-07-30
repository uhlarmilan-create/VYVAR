CURSOR RESULT - 2026-07-30 AUDIT STAGE 3 PART 1c

What I did
Re-measured check-star ?² with robust estimators, forensically inspected the top-20
single-epoch contributors, and audited `sigma_sys_mag` configuration vs differential-photometry
literature. Corrected Part 1b's ?² indexing error.

## Provenance

| Field | Value |
|-------|-------|
| `git_hash` | `75e08cc07c91882402dd16aa105348258eaf67e1` |
| `git_dirty` | `true` |
| Data | Anchor snapshot `draft_000435_snapshot_skysurface_20260716` |
| Harness | `dev/scripts/audit_stage3_part1c_robust_chi2.py` |
| Raw JSON | `tmp/audit_stage3_part1c_results.json` |

---

## Correction: Part 1b ?² = 649 was mislabeled

**Part 1b is wrong on the metric name.** `audit_stage3_part1b_check_chi2.py` stored
`reduced_chi2_constant(...)[0]` — index **0** is **total ?²**, not ?²_red (index **2**).

| Statistic | Part 1b (mislabeled) | Part 1c (correct ?²_red) |
|-----------|---------------------:|-------------------------:|
| Median | 649 | **4.70** |
| p95 | — | **18.75** |
| max | — | **199** |

Rerun of the 1b script on the same snapshot reproduces **649.0** exactly — confirming the
bug is indexing, not data drift. Typical N ? 139 epochs ? 649/138 ? **4.7**.

The production-path pairing (`mag_calib_final`, `err`) remains correct; only the reported
statistic was mis-indexed.

---

## 1c.2 — Robust ?² distribution (162 check stars)

Clip: **? = 3.0**, **maxiters = 5** (MAD-scaled iterative residual clip vs weighted mean).

| Estimator | median | p05 | p95 | max |
|-----------|-------:|----:|----:|----:|
| ?²_red raw | 4.70 | 2.22 | 18.75 | 198.9 |
| ?²_red clipped | 3.97 | 1.72 | 10.89 | 58.4 |
| MAD robust `(med(z²)×1.4826²)` | 3.48 | 1.35 | 11.15 | 71.5 |
| Outlier fraction removed | 1.4% | 0% | 5.0% | 38.1% |

**Reading:** Median ?²_red ? 4–5 (not ?1) indicates production `err` is still ~2× too small
for check-star scatter on typical fields, but the Part 1b headline **649** was an order-of-magnitude
reporting error. Clipping removes a **median 1.4%** of epochs per star; clipped median ?²_red
falls modestly (4.7 ? 4.0).

Err on bright epochs: median **0.058 mag**, minimum **0.006 mag** (star `1485540612577549568`:
scatter **0.033 mag**, ?²_red **0.79**).

---

## 1c.3 — Top-20 single-epoch ?² contributors

| Pattern | Evidence |
|---------|----------|
| **Bad frames (multi-star)** | **4/20** from `proc_BO_CVn_Light_123.csv`; **3/20** from `_080.csv`; repeats in `_051`, `_117`. Same epoch affects many check stars ? frame problem, not isolated CR. |
| **Cosmic rays (sharp)** | **None** of top 20 show FWHM ? PSF (`fwhm_estimate_px` ~3.3–3.4 px, PSF-like). |
| **Saturated / bright frame** | Frame 123: `peak_max_adu` ? **27?700 ADU** on multiple stars; residuals up to **+1.04 mag** with `err` ? 0.006 mag. Likely saturation/non-linear or tracking/focus on that frame. |

Example (worst point): target `1497561779362267392`, frame 123 — residual **+1.04 mag**,
`err` **0.006 mag**, ?²_point ? **26?300**.

**Verdict:** Dominant failure mode in the extreme tail is **shared bad frames** (especially
bright/saturated), not cosmic-ray spikes. A minority of high-?² points remain single-star.

---

## 1c.4 — `sigma_sys_mag = 0` on this rig

| Item | Value |
|------|-------|
| Code | `resolve_sigma_sys_mag()` in `src_py/sigma_floor_core.py` |
| Default when unconfigured | **0.0** (one-time INFO log) |
| Per-rig config | `config.json` ? `"sigma_sys_mag": {"4": 0.018}` only |
| This rig (`equipment_id=1`, 200 mm) | **0.0** resolved |
| Intended use | Hand-configured per-rig white floor in mag; not measured from data |

Production error: `err² = err_photon² + sem_ens² + sigma_sys²` (`sigma_floor_core.py`).
On this rig `frac_sys ? 0%` of variance (Part 1b budget, confirmed).

### Literature (R1)

**Honeycutt (1992), PASP 104, 435:** differential ensemble photometry should add an
**ensemble scatter term** (SEM) in quadrature — VYVAR implements this via `ensemble_sem_mag_from_residuals`
with c4 small-sample correction.

**Broeg et al. (2005), AN 326, 134:** advocate an **irreducible systematic floor** beyond
Poisson + scintillation when combining many frames; floor prevents ?² ? ? on constant stars
when SEM ? 0 on bright data.

**VYVAR today:** SEM is present; **`sigma_sys_mag` floor is zero on equipment_id=1**. Scintillation
is computed (~**1.73 mmag** at airmass 1 on the 200 mm rig per harness) but **not wired into
production `err`**. Audit P-02 cited ~2.3 mmag scintillation — same order, below the 6 mmag
minimum `err` on bright epochs; scintillation alone does not explain ?²_red ? 5, but it is a
real missing floor alongside `sigma_sys = 0`.

---

## DECISION REQUIRED (1c)

Whether to introduce an error floor (and of what kind):

| Option | Trade-off |
|--------|-----------|
| Per-rig `sigma_sys_mag` (Honeycutt/Broeg-style floor) | Stabilises ?² on bright constants; needs Milan-chosen mag value per rig |
| Wire scintillation into `err` | Physically motivated, ~1–2 mmag; insufficient alone for ?²_red ? 5 |
| Epoch outlier rejection (CR / bad frames) | Addresses frame 123 class; not yet in pipeline (D1-1) |
| Status quo | ?²_red ~5 on check stars; extreme tail from bad frames + tiny `err` |

**Do not wire anything without Milan decision.**

---

## Files changed

- `dev/scripts/audit_stage3_part1c_robust_chi2.py`
- `dev/results/CURSOR_RESULT_audit_stage3_part1c.md`

**STOP GATE 1c** — awaiting Milan review.
