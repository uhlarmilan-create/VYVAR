CURSOR RESULT - 2026-08-13

What I did
Pre-push investigation for commit `7ec4b09`: traced BO CVn 5?4 comp drop, reconciled scatter
metrics, traced peak self-check consumers, measured fail/pass frame physics, checked lin_adu native
units and draft-435 anchor status. Corrected mislabeled metrics in
`dev/results/CURSOR_RESULT_sat_diag_implement.md` S3.2-3.3.

Artifacts: `tmp/_prepush_7ec4b09_investigate.py`, `tmp/_selfcheck_diag.py`

---

## 1 - Larger pool, smaller ensemble

### 1.1 Comparison stars by catalog_id

| Rank (comp_rms) | Draft 509 (5) | Draft 510 (4) |
|----------------:|---------------|---------------|
| 1 | 1497771992240531712 | 1497771992240531712 |
| 2 | 1499200223486564608 | 1499200223486564608 |
| 3 | **1497974027502858240** | - |
| 4 | 1499053747922698240 | 1499053747922698240 |
| 5 | 1497368849430107904 | 1497368849430107904 |

**Dropped star:** `1497974027502858240` (3rd-best comp_rms 0.013912 in 509; tier 1; 134 frames in both drafts).

Remaining four stars have **identical** comp_rms, comp_score, and comp_n_frames between drafts.

### 1.2 Where the star dropped out

| Stage | Draft 509 | Draft 510 |
|-------|-----------|-----------|
| Global static pool | 624 / 735 @ 16384 admission sim | 709 / 735 @ 65535 |
| Per-target candidates (debug) | 101 at per-target stage | 101 |
| **Saturation hard filter** | Not applied (no raw peaks; `saturate_limit_adu_85pct` absent on proc CSVs) | **Excluded here** |
| RMS / color / n_comp_max | Selected (5 comps) | 4 comps (drop already removed from `flux_map`) |

**Measured exclusion (draft 510, star `1497974027502858240`):**

| Quantity | Value |
|----------|------:|
| Frames with finite `peak_max_adu` (self-check pass) | 90 / 134 |
| Frames with `peak_max_adu` > admission threshold | **59 / 90** |
| Admission threshold | **45874.5 ADU** (= `65535 x 0.85 x 0.70/0.85`; `saturate_limit_adu_85pct` = 55704.75) |
| Fraction over threshold | **0.656** |
| Gate threshold | **> 0.10** when `total >= 10` |

**Code:** `src_py/comp_selection_per_target.py:1087-1099` - `_apply_comp_metric_hard_filters`, saturation branch:

```1087:1099:src_py/comp_selection_per_target.py
    # Filter SAT: reject candidates above admission gate (Tier 1 only; INV-SAT-01)
    _sat_rejected: set[str] = set()
    if sat_may_exclude:
        for cid in sorted(flux_map.keys()):
            total = int(peak_total_map.get(cid, 0) or 0)
            over = int(peak_over_map.get(cid, 0) or 0)
            if total >= 10 and total > 0 and (float(over) / float(total)) > 0.10:
                flux_map.pop(cid, None)
                _sat_rejected.add(cid)
```

Peak accumulation uses per-frame proc CSV `peak_max_adu` vs `saturate_limit_adu_85pct` at `comp_selection_per_target.py:798-842`.

This is **not** n_comp_max truncation (`n_comp_max=8`; only 4-5 comps ever reached final selection).

### 1.3 Cause: SAT-DIAG raw vs aligned peak - established

**Yes - caused by raw-versus-aligned peak divergence crossing the admission threshold.** Not by pool widening alone.

| | Draft 509 | Draft 510 (star `1497974027502858240`) |
|--|-----------|----------------------------------------|
| `peak_max_adu` source | Aligned-frame peak only | Raw FITS peak when self-check passes (`sat_diag.py:583`) |
| Median peak | **5211 ADU** | **49404 ADU** on pass frames; aligned median **5211 ADU** |
| Sat gate | No limit columns on proc ? gate inert | **59/90 > 45874.5 ? reject** |

On **pass** frames, `mag_guided_centroid` (45x45 search, `sat_diag.py:379-391`) locks onto a **bright raw feature** (~37-61 kADU) while aligned photometry at `(x,y)?(751,629)` measures the faint star (~5-7 kADU). Self-check passes because ring contrast is high (~20x) at the wrong pixel.

On **fail** frames (44/134), guided centroid lands on sky at `(758,665)` with centre **1780-1900 ADU** - below `PEAK_MIN_ADU=4000` and ring ratio **~1.14 < 1.8** (`sat_diag.py:360-376`).

Drift reference for export: brightest masterstar `1500347838748255360` (`pipeline.py:10540-10552`).

### 1.4 Stars near the boundary

Admission threshold **45874.5 ADU** (70% of full scale). Pattern 'aligned peak < 12 kADU but raw pass-frame peak > threshold' (mis-centre hijack):

| Count | Scope |
|------:|-------|
| **19** | Stars with hijack on **>10%** of self-check-pass frames (all proc stars, 134 frames) |
| **7** | Of those, also **>10% over admission** ? would fail sat gate (includes drop star) |
| **1** | BO CVn 509 comp ensemble member affected (`1497974027502858240`) |

Other 509 comps on draft 510: sat-gate pass (0/N over on their pass-frame peaks). Several have **fewer** pass frames (e.g. `1499200223486564608`: 44 pass / 90 fail) but **0** over-threshold on passes.

The threshold did not move for stars with trustworthy raw peaks; it **mis-fires** where mag-guided search hijacks to bright structure on faint comps.

### 1.5 Four-star vs five-star ensemble

| Evidence | Verdict |
|----------|---------|
| **check-star scatter** (primary pre-registered QC) | 510 **slightly worse**: 0.008946 vs 509 **0.008629** (+3.7%) |
| **ac_scatter** | **Identical** 0.009283 (AC ref-star path; unchanged by comp count) |
| TRUST | GREEN both |
| Dropped comp quality | 3rd by RMS (0.013912); reasonable tier-1 member |
| LC flux path | Uses aligned aperture flux regardless of peak self-check |

**Plain statement:** The four-star ensemble is **slightly worse** on the metric this session pre-registered (check-star scatter), **the same** on ac_scatter, and **different** in membership (lost a good RMS comp to a sat-gate artefact, not to RMS ranking). The wider pool did not shrink the ensemble; **SAT-DIAG saturation exclusion** did.

---

## 2 - Scatter metrics

### 2.1 Check-star scatter (same method as 509)

| Draft | Check star | check-star scatter | n_epochs |
|-------|------------|-------------------:|---------:|
| **509** | 1497313255374892800 | **0.008629** | 134 |
| **510** | 1497313255374892800 | **0.008946** | 134 |

Same check star both drafts (`check_kmag_1498613634033133184.csv`). No check-star swap.

Computed via `trust_flag_core.check_star_scatter()` ? `nanstd(kmag)` on `photometry/lightcurves/check_kmag_{target_id}.csv` (`trust_flag_core.py:84-97`).

### 2.2 ac_scatter (like-for-like)

| Draft | ac_scatter | n_good_comp |
|-------|----------:|------------:|
| **509** | **0.009283** | 5 |
| **510** | **0.009283** | 4 |

From `photometry_summary.csv`; sourced from aperture-correction ref-star residuals (`photometry_core.py:4982-4985`).

### 2.3 Standard comparator going forward

**Primary draft-vs-draft QC:** **check-star scatter** from the `check_kmag` sidecar - this is what ZP-clip predictions used and what TRUST hard-gates reference.

**Secondary / diagnostic:** `ac_scatter` - aperture-correction internal consistency; label explicitly as **AC scatter**, never as check-star scatter.

**Fix applied:** `dev/results/CURSOR_RESULT_sat_diag_implement.md` S3.2-3.3 now labels both metrics separately (was comparing 509 check scatter to 510 ac_scatter).

### 2.4 Correct 509 vs 510 verdict

On **check-star scatter**: draft 510 **regressed slightly** (+0.00032 mag, +3.7%). Still GREEN TRUST; not a large move but not an improvement.

On **ac_scatter**: **held level** (identical to round-off).

---

## 3 - Self-check without gate (design gap)

### 3.1 What a self-check failure does today

**Trace - all consumers of `peak_loc_ok` / `peak_loc_fail`:**

| Location | Effect |
|----------|--------|
| `sat_diag.py:557-559` | Sets columns on proc DataFrame |
| `sat_diag.py:583` | **`peak_max_adu = raw peak if ok else NaN`** (authoritative for saturation) |
| `sat_diag.py:591-594` | Increments `ctx.peak_loc_fail_count[cid]` |
| `sat_diag.py:499-515` | Optional persist in `sat_diag.json` (draft 510 file has `{}`; aggregate not written post-pass) |

**No other consumer.** Comp selection reads `peak_max_adu` only (`comp_selection_per_target.py:695-842`). Photometry flux/ZP/LC paths do **not** read self-check flags.

**Answer:** A failure **records a count** (in-memory; optionally JSON) and **nulls raw peak** for saturation. It does **not** remove the star from the ensemble, skip photometry, or flag the frame in the LC.

**Note on task premise:** In draft **510**, `1497974027502858240` is **not** in the admitted ensemble (excluded by sat gate on pass frames). It **was** in draft **509** (no raw peaks / no gate). The design gap is: failures (and false passes) do not gate **membership** or **photometry** - only whether raw peak counts toward saturation.

### 3.2 Why 44 frames fail and 90 pass (measured)

Drift ref: masterstar `1500347838748255360`. Star aligned photometry at **(751, 629)**; flux **~24-25k ADU** on all sampled frames.

| Frame | Self-check | Guided (x,y) | Centre ADU | Aligned peak | Ring med | Ratio | Fail reason |
|-------|------------|--------------|------------|--------------|----------|-------|-------------|
| Light_001 | **pass** | (732, 629) | **51652** | 6681 | 2616 | 19.7 | - (bright hijack) |
| Light_055 | **pass** | (736, 627) | **38232** | 5211 | 1800 | 21.2 | - (bright hijack) |
| Light_098 | **fail** | (758, 665) | **1900** | 5370 | 1666 | 1.14 | centre < 4000; ratio < 1.8 |
| Light_100 | **fail** | (758, 665) | **1780** | 4834 | 1608 | 1.11 | centre < 4000; ratio < 1.8 |
| Light_103 | **fail** | (757, 664) | **1780** | 5295 | 1568 | 1.14 | centre < 4000; ratio < 1.8 |

**Pass frames:** mag-guided search captures a **bright neighbour/artifact** ~20 px from the true star; high ring contrast satisfies self-check.

**Fail frames:** larger drift misplaces search window on **sky**; low ADU and low contrast.

Not edge-clipped (guided pixels well inside 1397x2082 chip).

### 3.3 Photometry value on failing frames

When self-check **fails**: `peak_max_adu` and `peak_max_adu_raw` are **NaN**; **`flux` is still the aligned aperture measurement** (e.g. frame 098 flux **24656 ADU**). Photometry uses flux for ensemble/ZP; peak is metadata for saturation only.

If the star were in the ensemble (509 case): **verified flux, unverified peak** on fail frames; on pass frames **wrong raw peak (~50 kADU) drives saturation** while flux remains ~24 kADU.

### 3.4 Proposed policy (not implemented - Milan decides)

Respecting **INV-COMP-MEMBERSHIP** (decided once per draft):

**Recommend: draft-level comp eligibility rule combining self-check quality and raw/aligned consistency.**

1. After catalog export, for each pool candidate compute over LC frames:
   - `fail_frac = peak_loc_fail / (pass + fail)`
   - `hijack_frac = frames where peak_loc_ok and peak_max_adu_raw > admission_thr and peak_max_adu_aligned < 12000` / pass frames
   - `raw_aligned_ratio = median(peak_raw / peak_aligned)` on frames with both finite

2. **Exclude from comp pool for the whole draft** if any:
   - `fail_frac > 0.25` **OR**
   - `hijack_frac > 0.10` **OR**
   - `raw_aligned_ratio > 3.0`

   Rationale: measured drop star has `fail_frac=0.33`, `hijack_frac=0.66`, ratio ~9x - would exclude with headroom. BO CVn target (134/0 fails) unaffected.

3. **Do not** apply per-frame comp membership changes (invariant).

4. **Saturation:** For stars passing eligibility, use **`min(peak_raw, peak_aligned)`** or aligned-only when `hijack_frac > 0` on the static master peak - avoids false sat exclusion until mag-guided search is tightened for faint comps.

5. **Spec language:** Self-check is a **quality input to draft-level comp eligibility**, not a per-frame photometry gate. False pass (bright hijack) is the higher risk than false fail.

**Alternatives considered:**

| Option | Issue |
|--------|-------|
| Per-star fail fraction only | Does not catch hijack **passes** (this case) |
| Exclude saturation only | Star stays in ensemble but false peaks still exclude via current gate |
| Failed self-check = unknown peak | Better than false pass; does not fix hijack passes |
| Inform-only | Current behaviour; sat gate then acts on wrong peaks |

---

## 4 - Smaller items

### 4.1 lin_adu native units

| Record | stored ADU | native ADU (= stored / 4) |
|--------|------------|----------------------------|
| `sat_diag.json` draft 510 | lin_adu **55704.75** | **~13926.2** (not stored) |
| Spec S4.2 prose | defaults in stored ADU | mentions native = stored/4 |
| `VY_LINADU` FITS header | 55704.75 | not duplicated |

**Gap:** Spec documents the conversion; **`sat_diag.json` and `SatDiagContext.to_json_dict()` carry stored ADU only** - no `lin_adu_native` / `xbinning` companion field for ramp comparison.

**Recommendation:** Add `lin_adu_native`, `sat_adu_native`, and `adu_unit_note` to JSON on next schema bump (ramp will report native).

### 4.2 Draft 435 not re-exported

| Item | Status |
|------|--------|
| Proc CSV peak columns | `peak_max_adu` only (aligned); **no** `peak_max_adu_raw`, no self-check |
| `sat_diag.json` | **Absent** on draft 435 archive |
| `--full` anchor | Uses **frozen draft-435 inputs** (INV-ANCHOR-00); does not re-export catalogs |
| Anchor comp pool | Still reflects **16384-era** aligned peaks if anchor photometry re-run without re-export |

**Impact:** `--full` regression against anchor is **unchanged** (byte-stable anchor path). Comparing **SAT-DIAG behaviour** on 435 requires re-export - otherwise 435 simulates as LEGACY_ALIGNED.

**Recommendation:** Do **not** re-export 435 before push unless anchor comp-pool update is explicitly desired. For SAT-DIAG validation, treat **509/510** as the before/after pair; note 435 limitation in any cross-draft saturation table. When anchor refresh is scheduled, re-export 435 raw catalogs first so anchor and `--full` see consistent peaks.

---

## 5 - Push readiness

| Item | Status |
|------|--------|
| Implementation / tests | Sound (`7ec4b09`) |
| Metric labelling | Fixed in implement result doc |
| Ensemble drop | Explained - sat gate on hijacked raw peaks |
| Scatter | 510 slightly regressed on check-star scatter |
| Self-check policy | **Unsettled design gap** - false passes drive real comp exclusion |

**Recommendation:** **Do not push until Milan decides point 3 policy** (or accepts inform-only + documents hijack risk). The code works as written, but **`7ec4b09` admits a known failure mode**: mag-guided raw peaks can pass self-check on the wrong bright pixel, excluding good comps (measured on BO CVn). Pushing without policy leaves comp selection non-deterministic in a way the spec did not close.

If Milan accepts **inform-only** with explicit spec wording and a follow-up fix for faint-comp peak search, push can proceed with that documented caveat.

---

## Files changed

- `dev/results/CURSOR_RESULT_prepush_7ec4b09.md` (this file)
- `dev/results/CURSOR_RESULT_sat_diag_implement.md` (metric labels S3.2-3.3)
- `tmp/_prepush_7ec4b09_investigate.py`, `tmp/_selfcheck_diag.py` (scratch)

No code commit. **Not pushed.**
