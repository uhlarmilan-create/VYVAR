CURSOR RESULT - 2026-07-31 AUDIT STAGE 3 PART 0d

What I did
Forensic audit of the Part 0c delta tail. Established that Part 0c epoch pairing is invalid,
recomputed deltas on `source_file`, and traced the worst-target headline (`1498135552633294976`,
3.36 mag) to pairing error plus ensemble/flux effects on correctly paired epochs.

## Provenance

| Field | Value |
|-------|-------|
| `git_hash` | `086d45a6fbac1c30a71765ef77cbd53e816f7e4b` |
| `git_dirty` | `false` |
| Harness | `dev/scripts/audit_stage3_part0d_delta_forensics.py` |
| Raw JSON | `tmp/audit_stage3_part0d_results.json` (optional rerun) |

---

# SUMMARY

**Part 0c delta table is invalid.** Epochs are paired by **positional CSV index**, not by
`source_file` or BJD. The anchor has **139** photometry frames; the rebuild has **150**. Row
order also differs: for the focus target only **8/139** positional rows share the same
`source_file`. Positional pairing on a **2.3 mag ROT variable** produces spurious multi-magnitude
deltas. The **3.36 mag headline is not a valid measurement.**

**Correct pairing (`source_file`, 139 common frames):** cohort median per-target |?mag| p95 =
**0.104 mag** (vs **0.317 mag** positional). Focus target max |?| = **2.76 mag**, p95 =
**1.52 mag** (not 3.36). Residual tail on correct pairing is dominated by **ensemble
recomposition** (near-constant offset ~1.3-1.9 mag on worst targets) plus **raw flux
differences** on shared frames - not identity error, not astrophysical epoch mismatch.

**STOP GATE A: FAILED (A1). Part B skipped per task rules.** Part C uses corrected pairing.

---

# PART A - IS THE COMPARISON VALID?

## A1 - Epoch pairing in Part 0c (FAIL)

From `dev/scripts/audit_stage3_part0c_cohort_delta.py`, `_delta_table()`:

```python
n = min(len(rdf), len(adf))
mag_anchor = adf["mag_calib_final"].iloc[:n]
mag_rebuild = rdf["mag_calib_final"].iloc[:n]
```

**Pairing method: positional index** in CSV row order. Not `source_file`. Not BJD.

| Run | LC rows (focus target) | Unique `source_file` |
|-----|----------------------:|---------------------:|
| Anchor snapshot | 139 | 139 |
| Rebuild `draft_000499` | 150 | 150 |
| Common `source_file` | 139 | - |
| Positional order match (first 139 rows) | **8/139** same `source_file` | - |

Rebuild-only frames (11): e.g. `proc_BO_CVn_Light_009.csv`, `_056`, `_058`, `_074`, `_100`, ...

**Verdict: Part 0c delta table must be recomputed on `source_file` (or BJD). All Part 0c
tail statistics (p95 0.43 mag, bright-target 3.36 mag max, ensemble-changed 100%) are
unreliable.**

Note: `n_good_comp` is **not exported** in light-curve CSVs; Part 0c's ensemble-changed
stratification from that column is also invalid.

## A2 - Same star in both runs? (PASS)

Target `1498135552633294976`:

| field | anchor | rebuild |
|-------|--------|---------|
| `catalog_id` | 1498135552633294976 | 1498135552633294976 |
| RA, Dec | 212.232608 deg, 41.382321 deg | identical |
| pixel x, y (masterstars) | 281.13, 620.50 | 281.87, 620.61 |
| sky separation | **0.0 arcsec** | |
| pixel separation | **0.74 px** | |
| VSX | ROT (Gaia DR3 1498135552633294976) | same |
| `source_type` | GAIA_MATCHED | GAIA_MATCHED |

0.74 px ? aperture radius (~1.9-2 px). **Not PHASE0-IDENTITY-GATE failure.**

## A3 - Intrinsic variability (context)

Anchor LC for this target (139 epochs):

| Quantity | Value |
|----------|------:|
| `mag_calib_final` range | 15.39 - 17.73 mag (**2.34 mag** span) |
| std (ddof=1) | 0.31 mag |
| p05 / p50 / p95 | 15.62 / 15.91 / 16.47 |

Star is a **known ROT variable**. Positional mispairing on a 2.3 mag variable **can** produce
multi-magnitude spurious deltas without any pipeline bug. On **correctly paired** same-frame
epochs, variability does not explain the delta (same BJD, max BJD diff 0.0 days on merge).

## STOP GATE A

**FAILED on A1.** Part B not executed. Corrected analysis below under Part C and focus-target
annex.

---

# PART C - SCOPE (correct `source_file` pairing)

## C1 - Cohort distribution (156 common targets)

| Metric | Positional (0c method, invalid) | `source_file` paired (valid) |
|--------|--------------------------------:|-----------------------------:|
| Median per-target \|?\| p95 | 0.317 mag | **0.104 mag** |
| Cohort max per-target \|?\| p95 | 2.32 mag | **1.98 mag** |
| Targets with p95 > 0.1 mag | - | **78** |
| Targets with p95 > 0.5 mag | - | **12** |
| Targets with p95 > 1.0 mag | - | **5** |

Stratify by ensemble change (comp `catalog_id` set in `comparison_stars_per_target.csv`):

| Ensemble comp set changed? | n | median \|?\| p95 |
|--------------------------|--:|-----------------:|
| Yes | 134 | **0.138 mag** |
| No | 22 | **0.032 mag** |

Large deltas correlate with **comparison ensemble change**, not target faintness alone.

## C2 - Five worst targets (valid pairing)

| target_cid | \|?\| p95 | mean ? | std ? | comp intersection | anchor comps | rebuild comps |
|------------|----------:|-------:|------:|------------------:|-------------:|--------------:|
| 1498322916287022976 | 1.98 | +1.92 | 0.06 | 3 | 8 | 3 |
| 1485540612577549568 | 1.85 | ?1.80 | 0.03 | 1 | 4 | 8 |
| **1498135552633294976** | **1.52** | **?1.38** | **0.16** | **1** | **4** | **8** |
| 1498453414573142016 | 1.50 | ?1.39 | 0.13 | 0 | 3 | 8 |
| 1498341092588681856 | 1.41 | ?1.29 | 0.08 | 1 | 3 | 8 |

**Shared mechanism:** near-constant per-target offset (std ? |mean|) with **almost completely
disjoint comparison ensembles** (0-3 comps in common, rebuild typically 8 comps vs anchor 3-4).
This is **C2 ensemble zero-point recomposition**, not positional artefact. Positional method
inflated these further (focus target positional max 3.36 ? valid max 2.76).

### Focus-target annex (correct pairing; B-equivalent checks)

**1498135552633294976** - comparison comps:

| Run | comp IDs | mean comp catalog mag |
|-----|----------|----------------------:|
| Anchor | 1485995020117153792, 1497863492225150208, 1498352740539935744, 1498819861182631168 | 13.39 |
| Rebuild | 8 comps (intersection **1**: 1485995020117153792 only) | 12.95 |

Catalog mean comp mag shift ? **?0.45 mag** - insufficient alone for **?1.38 mag** mean target
offset; differential pipeline amplifies ensemble swap.

**Raw flux on worst common frame** (`proc_BO_CVn_Light_063.csv`):

| | anchor | rebuild |
|---|-------:|--------:|
| `dao_flux` | 134.4 | 458.1 |
| ratio | - | **3.41x** ? **?1.36 mag** instrumental |
| `aperture_r_px` | 1.916 | 1.901 |
| x, y | 282.18, 618.15 | 281.98, 621.63 |
| `peak_max_adu` | 1666 | 1709 |
| saturated | no | no |

Calibrated ? on that frame = **?2.76 mag**; raw flux accounts for ~**?1.36 mag** (C3);
remainder from ensemble calibration (C2).

**Cause classification (focus target, valid pairing):**

- **C4 pairing** - explains the **3.36 mag Part 0c headline** (invalid)
- **C2 ensemble** - primary cause of **?1.38 mag** mean offset on valid pairs
- **C3 flux/aperture** - flux ratio ~3.4x on worst frame; centroids within ~4 px
- **Not C1** identity (0.74 px sep)
- **Not C5** same-epoch variability (BJD matched)
- **Not C6**

## C3 - Six anchor-only targets

| catalog_id | In rebuild `variable_targets`? | In rebuild masterstars? | In rebuild photometry? | Drop stage |
|------------|--------------------------------|-------------------------|------------------------|------------|
| 1485913828055470592 | yes | yes | **no** | Photometry export (in catalogue, LC not written) |
| 1497121459315202816 | **no** | yes | no | **Planner/regen** (absent from rebuild `variable_targets`; anchor VSX name maps to nearby Gaia id) |
| 1497158396033960064 | yes | **no** | no | **MASTERSTAR / catalogue** (not in rebuild masterstars) |
| 1497960008729614976 | yes | **no** | no | **MASTERSTAR / catalogue** |
| 1498751450943601280 | yes | **no** | no | **MASTERSTAR / catalogue** |
| 1498878753773944576 | yes | **no** | no | **MASTERSTAR / catalogue** |

Anchor photometry summary shows all six had LCs in the snapshot (139 frames, various trust flags).
Rebuild wrote **230** LCs vs anchor **162**; these six are in the anchor set but not the rebuild
export set for reasons above - not a single uniform rule.

---

# Contradictions with earlier reports

| Earlier report | Finding | Verdict |
|----------------|---------|---------|
| Part 0c: ? p95 **0.43 mag**, max **2.56 mag** | Positional pairing | **Wrong** - recomputed median p95 **0.104 mag** |
| Part 0c: bright target max **3.36 mag** | Positional pairing on ROT star | **Wrong** - valid max **2.76 mag**, p95 **1.52 mag** |
| Part 0c: tail **100% ensemble_changed** via `n_good_comp` | Column absent from LC CSV | **Unreliable** - use comp-set diff (134/156 changed) |
| Part 0c: "bright targets moved materially" | Partially positional artefact | **Revised** - valid pairing shows 5 targets with p95 > 1 mag, all with ensemble swap |

---

# Implications for anchor re-cut

1. **Recompute all Stage 3 delta metrics** on `source_file` before any gate decision.
2. The **3.36 mag blocker** is **not confirmed** on valid pairing; the real tail is **~1-2 mag
   constant-offset** targets with **disjoint comparison ensembles** between anchor snapshot and
   rebuild - a plan/ensemble stability issue, not a single-star identity catastrophe.
3. Fix `audit_stage3_part0c_cohort_delta.py` before any future delta report.

**STOP GATE 0d** - awaiting Milan review.
