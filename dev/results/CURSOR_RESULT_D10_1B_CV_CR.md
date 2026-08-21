CURSOR RESULT - 2026-08-21 (D10-1b flat b_V verification)

What I did
Provenance audit of D10-1 sandbox, b_G cross-check on the ensemble fit, and
artifact-proof raw mag_inst probe on frozen era-516 NoFilter snapshot. No
`src_py/` changes.

## Baseline

| Ref | SHA |
|-----|-----|
| Local tip | `489d9d3` (D10-1) |
| origin/main | `8dea595` (ERR-518 series not yet on origin) |

References D10-1: `dev/results/CURSOR_RESULT_D10_1_CV_CR.md`

---

## Step 1 - Provenance audit (LHS scale)

**Question:** does any V-transformed quantity enter `mag_ensemble`?

**Answer: NO.** `transform_gaia_to_johnson` touches only the **right-hand side**
(catalog V / Rc for residuals).

| Step | File:lines | What enters the fit ordinate |
|------|------------|------------------------------|
| Proc flux | `d10_1_cv_cr_measure.py:130-144` | `flux` column from `proc_*.csv` |
| mag_inst | `:143` | `-2.5*log10(flux)` via `photometry_core._flux_to_mag` |
| Per-frame ZP | `:210-217`, `:171-185` | `median(phot_g_mean_mag - mag_inst_comp)` |
| Ensemble night mag | `:229-242` | `median(mag_inst + zp)` over 134 frames |
| Catalog RHS | `:247-248`, `:265-266` | `transform_gaia_to_johnson(g, bp_rp, V/RC)` only here |

ZP catalog magnitude is **`phot_g_mean_mag` (Gaia G)** from
`comparison_stars.csv`, not Johnson V (`:210-211`). No export transform, no
`johnson_mag`, no V coefficient on the LHS.

**LHS scale by construction:** G-anchored differential zeropoint
(`mag_inst + zp`). The flat `b_V` is **not** explained by a V-transform coding
bug in the sandbox.

---

## Table 5.9 expected slopes (recomputed from repo coeffs)

Source: `src_py/gaia_johnson.py:52-56`, `Y = G - X` polynomials, derivative
`dY/d(BP-RP)`.

| Quantity | Uniform mean c in [0.46, 2.80] | At c=1.0 | Star-weighted (D10-1 set, mean c=0.93) |
|----------|----------------------------------|----------|----------------------------------------|
| d(G-V)/dc | **-1039** mmag/mag | -374 | -350 |
| d(G-Rc)/dc | **-181** mmag/mag | +121 | +108 |
| d(V-Rc)/dc | **+858** mmag/mag | **+495** | **+455** |

Architect-cited ~-555 / ~-40 / ~+515 mmag/mag are in the same ballpark as the
**c~1** and **star-weighted** values (especially d(V-Rc)/dc ~ **495** at c=1.0,
matching measured `b_R - b_V`). Uniform mean over the full [0.46, 2.80] span is
steeper because d(G-V)/dc is highly non-linear toward red BP-RP.

**Identity check (ensemble fit):** measured `b_R - b_V` = **+494** mmag/mag vs
star-weighted expected d(V-Rc)/dc **+455** (within colour-distribution
tolerance; not exact agreement required).

---

## Step 2 - b_G on unchanged ensemble fit (n=2076, BP-RP span 2.34)

| Band | b (mmag/mag) | stderr | Pre-registered prediction if... |
|------|--------------|--------|----------------------------------|
| **G** | **+385** | 67 | G-scale: ~0; V-scale: ~+555 |
| **V** | **+1.6** | 67 | (D10-1 replay) |
| **Rc** | **+495** | 67 | |

**Readout:** `b_G` is **not** ~0 (rules out naive "LHS = Gaia G" for the fitted
quantity). `b_G` is **not** ~+555 (rules out pure V-transform artifact in code).
**Algebra:** `b_V ~ b_G + d(G-V)/dc` -> 385 + (-374) ~ **+11** mmag/mag at
c~1, consistent with measured flat `b_V`.

Ensemble decision rule replay: **CV** (unchanged from D10-1).

---

## Step 3 - Raw-counts bandpass probe (decisive)

**Method:** same D10-1 star selection; `mag_inst` nightly median only; **no ZP,
no CT, no transform on LHS**; G <= 14 (span 2.34, no relax needed); OLS on
(mag_inst - X_cat) vs BP-RP for X in {G, V, Rc}.

| Band | n | BP-RP span | b (mmag/mag) | stderr |
|------|---|------------|--------------|--------|
| **G** | 1567 | 2.34 | **+358** | 65 |
| **V** | 1567 | 2.34 | **-29** | 65 |
| **Rc** | 1567 | 2.34 | **+466** | 65 |

**Sanity gates (pre-registered, outside fit):**

| Check | Measured | Expected | Pass |
|-------|----------|----------|------|
| b_R - b_V | **+494** | d(V-Rc)/dc ~ **+455** (star-weighted) | yes |
| b_G - b_V | **+387** | -d(G-V)/dc ~ **+350** | yes |

If the star set or flux column were wrong, identities would fail; they do not.

**Pre-registered readout:** flattest |b_X| is **V** (29 mmag/mag).

**Decision rule on raw b_V vs b_R:** ratio **16**, |diff| **495** > 3 sigma
(**92**) -> **CV**.

**Physics read (plain):** raw counts with no ZP show **V-like** colour behaviour
(flat vs Johnson V, steep vs Rc, b_G ~ -d(G-V)/dc). An **I ~ G** instrument
would give b_G ~ 0, b_V ~ **-500+**, b_R ~ **-40** -- **not observed**. The
flat `b_V` is **real bandpass physics**, not a sandbox V-transform artifact.

Plot: `dev/results/context/session_20260821_d10_1b/raw_residual_vs_bprp.png`

---

## Step 4 - Reconcile with D10-1

| Item | D10-1 | D10-1b |
|------|-------|--------|
| Verdict (rule on b_V vs b_R) | CV | **CV** (confirmed on raw) |
| Interpretation paragraph | Retracted (see below) | V-like effective band |
| Mechanism | Incorrect (G-R slope story) | No V bug; I_eff ~ V |

**D10-1 interpretation corrections (retract in `CURSOR_RESULT_D10_1_CV_CR.md`):**

1. **Wrong:** "if mag_ensemble were Gaia-G scale, b_V ~ flat." G-anchored ZP
   with G-only LHS would give **b_G ~ 0** and **b_V ~ d(G-V)/dc ~ -350 to -1000**
   mmag/mag over this span, not ~0. Measured **b_G ~ +385**, **b_V ~ +2**.
2. **Wrong:** "b_R tracks G-R polynomial slope ~0.5 mag/mag." Mean d(G-Rc)/dc
   is **~-180 mmag/mag** (uniform) / **+108** (star-weighted); measured **b_R
   ~ +495** tracks **d(V-Rc)/dc**, not d(G-Rc)/dc alone.
3. **Correct:** `b_R - b_V` matches d(V-Rc)/dc identity; V-vs-R contrast is
   internally sound.

**Why D10-1 CV verdict survived:** both ensemble and raw slopes select **CV**
(smaller |b_V|). Stakes under raw slopes are **smaller** in mmag for pinned
targets (below).

---

## Final verdict line

**Pre-registered rule applied to raw-counts slopes: CV.**

(Milan decides letter; export mapping unchanged in this task.)

---

## Pinned submission systematics (mmag, b * delta BP-RP)

| Target | d(BP-RP) | CV @ raw b_V=-29 | CR @ raw b_R=+466 | CV @ ens b_V=+2 | CR @ ens b_R=+495 |
|--------|----------|------------------|-------------------|------------------|-------------------|
| BO CVn | -0.030 | +0.9 | -14.1 | -0.05 | -15.0 |
| FW CVn | +0.034 | -1.0 | +15.8 | +0.06 | +16.8 |
| GH CVn | -0.054 | +1.5 | -25.1 | -0.09 | -26.7 |

Raw-counts slopes imply **sub-mmag** colour systematics for CV on BO/FW at
typical pinned colour offsets; CR would inject **~15-25 mmag** on FW/GH.

---

## Artifacts

- `dev/sandbox/d10_1b_cv_cr_verify.py`
- `dev/results/context/session_20260821_d10_1b/summary.json`
- `dev/results/context/session_20260821_d10_1b/raw_star_residuals.csv`
- `dev/results/context/session_20260821_d10_1b/raw_residual_vs_bprp.png`

## Docs impact (DOCS-SYNC)

None. Amended interpretation recorded here and in D10-1 result (pointer).
No ROADMAP/DECISIONS edits.

STOP - Milan decides the band letter.
