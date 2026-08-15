# CURSOR RESULT - SEM-WEIGHT-01

Date: 2026-08-15
Baseline: 011fff7 (+ uncommitted A1/A2 from DRAFT-514-TRIAGE)
Type: CHECK + MEASUREMENT. Exported error bars unchanged.
Push: NO.

Machine: `dev/results/SEM_WEIGHT_01_summary.json`,
`dev/results/SEM_WEIGHT_01_results.json` (per-epoch).
Harness: `dev/tools/sem_weight_01_measure.py`
Commit SHA on every result block: **011fff7**

---

## Named outcome

**Explains none of it.**

`ratio = sem_weighted / sem_current` has median **0.677** (p16=0.529, p84=0.856)
across 29 targets / 3886 epochs on draft 514. The consistent weighted SEM is
*smaller* than what the code reports today. WIDE-ERR is an *under*-estimate of
scatter (R~2.05 at G10.25, R~2.31 at G8.25). Fixing the SEM the consistent way
would shrink the ensemble term and make WIDE-ERR larger, not smaller. The earlier
localization (deficit in photon/sky; ensemble SEM excluded) survives at N~1292.

---

## Estimator

Primary `sem_weighted` (reliability / inverse-variance style, empirical):

```
mu = sum(w x) / sum(w)
V1 = sum(w),  V2 = sum(w^2)
N_eff = V1^2 / V2
s_w^2 = sum(w (x-mu)^2) / (V1 - V2/V1)
SEM = s_w / c4(round(N_eff)) / sqrt(N_eff)
```

Same residual vector `x` as production: per-comp `(m_inst - median_night)`.
Same weights as Phase 2A ZP: `w = 1/sigma_eff^2` recomputed per target
(`c_col = 0.029485 mag/BP-RP`, `c_dist = 0`).

Why this form: it is the reliability-weight sample variance (Cochran 1977 /
Wikipedia "Weighted arithmetic mean") with the SE of the mean using `N_eff`
in place of `n`. When all weights are equal it reduces exactly to the current
`s_ddof1/c4/sqrt(n)`. Appropriate when `w=1/sigma_eff^2` are treated as
reliability weights on an empirical residual population (not frequency counts).

Secondary (reported, not used for the named outcome):

| estimator | median ratio vs `sem_current` | note |
|---|---:|---|
| ZP-offset reliability SEM on `(catalog_mag - m)` | ~0.84-1.2 by target | quantity actually averaged for ZP |
| Model IVW `1/sqrt(sum w)` | ~0.01 | useless here: `sigma_eff` is not the residual sigma of `(m-med)` |

The two conventions differ materially: model IVW is ~100x too small because
`sigma_eff` encodes Broeg/comp quality, not the night-detrended residual scale.

---

## Distribution (SHA 011fff7)

| quantity | value |
|---|---|
| n_targets with LC | 29 |
| n_epochs | 3886 |
| ratio median | **0.677** |
| ratio p16 / p84 | 0.529 / 0.856 |
| ratio min / max | 0.375 / 1.397 |
| corr(ratio, N_eff) | +0.14 (weak) |
| corr(ratio, target G) | -0.47 (mild; not the WIDE-ERR bright excess shape) |

By N_eff bin (epoch pool):

| N_eff | n_epochs | ratio median |
|---|---:|---:|
| 150-200 | 2682 | 0.665 |
| 200-300 | 668 | 0.699 |
| 300-1000 | 536 | 0.730 |

Extremes by target median ratio:

| | target | ratio | N_eff |
|---|---|---:|---:|
| lowest | ZTF J140640.73+413639.0 | 0.611 | 188 |
| highest | R CVn | 1.220 | 345 |

R CVn is the only target with median ratio > 1 (extreme red; flatter weights;
denom effect wins). Everywhere else current SEM is *too large*.

---

## Decomposition (the two opposing errors)

Median across targets:

| factor | median | direction |
|---|---:|---|
| denom-only `sqrt(n/N_eff)` | **2.59** | alone would raise SEM (current too small) |
| num-only `s_w / std_unweighted` | **0.248** | alone would lower SEM (current too large) |
| product | **0.642** | matches observed ratio ~0.68 |

Low-weight comps dominate the unweighted std. That numerator inflation beats the
N_eff denominator deficit. Net: reported SEM ~1.5x too high vs a consistent
weighted SEM on the same residuals.

---

## WIDE-ERR comparison

| | |
|---|---|
| WIDE-ERR R at G10.25 | 2.054 (WIDE_ERR_LOC_01) |
| WIDE-ERR R at G8.25 | 2.308 |
| median `ratio` | 0.677 |

`ratio` is not near 2.3, not near 1 in a cancelling sense, and does not track
magnitude the way WIDE-ERR does (WIDE-ERR is worse at the bright end; ratio is
slightly higher for some bright/red targets but median stays <1).

Naive `R_remaining ~ R_WIDE / ratio` is meaningless for "SEM was missing" because
ratio < 1. Correcting SEM would move model sigma the wrong way.

**Verdict name: Explains none of it.**

---

## Membership (forced-phot / sat)

Per epoch, SEM residual set and ZP-weighted set are identical after shared
filters (finite mag, not `likely_saturated`): **0 / 3886** mismatch epochs.
Saturated comps excluded from ZP are also excluded from the SEM residual list.
The SEM describes the same contributing set as the weighted mean membership.

Note: SEM residuals are `(m - med_night)`; the weighted mean averages
`(catalog_mag - m)`. Same stars, different scalar. Secondary ZP-residual SEM
is reported in JSON for completeness.

---

## c4

Production: `c4(n)` with n~1292 -> 0.99981 (irrelevant).
Weighted form should use `c4(N_eff)` (rounded). At median N_eff~192, c4 still
~0.9987. Only matters when N_eff is tens. Measurement applied `c4(round(N_eff))`.

---

## Incidental: CSV `comp_weight` is wrong

`comparison_stars_per_target.csv` stores identical `comp_weight` for every
target (N_eff=152.2 for all). Phase 2A recomputes weights in memory; the CSV
column is not trustworthy. Measurement recomputed `sigma_eff` weights per target
(matches B1 N_eff spread 150-590). Separate cleanup, not in scope.

---

## Proposed fix (DO NOT APPLY)

Isolated correctness fix in `sigma_floor_core.py` + call site. Does not change
products until authorized.

```python
# sigma_floor_core.py -- add alongside ensemble_sem_mag_from_residuals

def ensemble_sem_mag_from_residuals_weighted(
    residuals: list[float] | Any,
    weights: list[float] | Any,
) -> float:
    """SEM of a reliability-weighted mean; reduces to unweighted when w equal.

    s_w^2 = sum(w (x-mu)^2) / (V1 - V2/V1)
    SEM   = s_w / c4(round(N_eff)) / sqrt(N_eff)
    N_eff = V1^2 / V2
    """
    pairs = [
        (float(x), float(w))
        for x, w in zip(residuals, weights, strict=False)
        if math.isfinite(float(x)) and math.isfinite(float(w)) and float(w) > 0
    ]
    if len(pairs) < 2:
        return 0.0
    xs = [p[0] for p in pairs]
    ws = [p[1] for p in pairs]
    v1 = sum(ws)
    v2 = sum(w * w for w in ws)
    if v1 <= 0 or v2 <= 0:
        return float("nan")
    n_eff = (v1 * v1) / v2
    mu = sum(w * x for w, x in zip(ws, xs, strict=False)) / v1
    denom = v1 - (v2 / v1)
    if denom <= 0:
        return float("nan")
    s2 = sum(w * (x - mu) ** 2 for w, x in zip(ws, xs, strict=False)) / denom
    if s2 < 0:
        return float("nan")
    c4 = c4_small_sample(max(2, int(round(n_eff))))
    if not math.isfinite(c4) or c4 <= 0:
        return float("nan")
    return math.sqrt(s2) / c4 / math.sqrt(n_eff)
```

Call site in `ensemble_normalize` (~3602): build `comp_resid` and parallel
`comp_resid_w` from `comp_weight_map` (same cid filter), call the weighted
function. Keep the unweighted function for tests/compat until cutover.

Regression (when authorized): call the real weighted function with equal weights
and assert identity to `ensemble_sem_mag_from_residuals`; with two-weight toy
data assert N_eff and SEM match the closed form.

**Do not ship this into exported `err` without a separate authorization.** It
would change every quoted uncertainty, and on this measurement it shrinks the
ensemble term (~0.68x), which does not help WIDE-ERR.

---

## Files

- `dev/tools/sem_weight_01_measure.py` (new)
- `dev/results/SEM_WEIGHT_01_summary.json`
- `dev/results/SEM_WEIGHT_01_results.json`
- `dev/results/CURSOR_RESULT_SEM_WEIGHT_01.md` (this file)

No production code changed. No push.
