CURSOR RESULT - 2026-07-30 AUDIT STAGE 3 PART 2b

What I did
Confirmed Part 2 sweep pathology, re-ran threshold sweep on the **MASTERSTAR construction path**
(`detect_stars_and_match_catalog`), verified legacy N arithmetic, log-log sanity, and built
the decision table. **No N selected** (R5).

## Provenance

| Field | Value |
|-------|-------|
| `git_hash` | `75e08cc07c91882402dd16aa105348258eaf67e1` |
| `git_dirty` | `true` |
| Data | `Archive/Drafts/draft_000499` MASTERSTAR.fits |
| Harness | `dev/scripts/audit_stage3_part2b_threshold_sweep.py` |
| Raw JSON | `tmp/audit_stage3_part2b_sweep.json` |

---

## 2b.1 — Why Part 2 sweep was unusable (confirmed)

`detect_stars_match_master_reference` (Part 2 script) matches a **fixed** `masterstars_full_match.csv`
catalogue. Output row count is master-locked; `DAO_ONLY` fraction was read from the **frozen CSV**
(3.93% at every N). Pass-1 meta counts barely responded to threshold (3470–6417) because
pass-2 targeted Gaia fill-in dominates at high N.

**Second symptom confirmed:** `dao_only_fraction_baseline_csv` = **3.93%** constant across
Part 2 sweep — the metric the sweep was meant to optimise could not move.

---

## 2b.2 — Legacy N_equiv arithmetic

On rebuild MASTERSTAR (Option B path):

| Quantity | Value |
|----------|------:|
| `rms_conv` | **55.63 ADU** |
| Kernel `rel_err` (measured) | **1.090** |
| Nominal threshold @ 3.8? | **192.8 ADU** (= 3.8 × 55.63, pre-kernel) |

**Task formula (legacy `scale_threshold=True`):**

    N_equiv = (192.8 × rel_err) / rms_conv = (192.8 × 1.36) / 55.63 ? **4.71**

using **rel_err = 1.36** from the legacy convolved threshold scaling.

**Measured on this rebuild** (rel_err = **1.09** from `_dao_convolved_background_rms_adu`):

    N_equiv = (192.8 × 1.09) / 55.63 ? **3.78**

Part 2 sweep at N=4.0 reported pass-1 **3639** on the **wrong path**; correct path at N=4.0
gives **n_pass1_raw = 2578** (close to rebuild pass-1 **2521**). The Part 2 pass-1 numbers
are **not comparable** to legacy 2521 — different pipeline stage (master-locked vs catalogue rebuild).

---

## 2b.3 — Correct sweep (N = 2.5 … 8.0)

Path: `detect_stars_and_match_catalog` on `MASTERSTAR.fits`, cone catalog cached,
`prematch_peak_sigma_floor = 1.8`, `max_catalog_rows = 100000`. Catalogue rebuilt each N.

| N | threshold [ADU] | n_pass1_raw | n_after_snr | n_catalog_rows | dao_only_frac | G<16 | G 16–17.5 | G>17.5 | cap |
|---|----------------:|------------:|------------:|---------------:|--------------:|-----:|----------:|-------:|-----|
| 2.5 | 139 | 15641 | 13077 | 13077 | 70.5% | 5432 | 663 | 119 | — |
| 3.0 | 167 | 6013 | 5938 | 5938 | 38.5% | 1358 | 156 | 26 | — |
| 3.5 | 195 | 3307 | 4013 | 4013 | 12.8% | 328 | 28 | 5 | — |
| 3.75 | 209 | 2840 | 3724 | 3724 | 6.4% | 168 | 9 | 1 | — |
| **3.8** | **211** | **~2700** | **~3700** | **~3700** | **~5.5%** | — | — | — | (interp.) |
| 4.0 | 223 | 2578 | 3612 | 3612 | **4.87%** | 120 | 8 | 2 | — |
| 4.5 | 250 | 2289 | 3520 | 3520 | 4.83% | 117 | 10 | 2 | — |
| 5.0 | 278 | 2080 | 3494 | 3494 | 4.78% | 118 | 9 | 2 | — |
| 6.0 | 334 | 1837 | 3485 | 3485 | 4.68% | 114 | 9 | 2 | — |
| 8.0 | 445 | 1492 | 3491 | 3491 | 4.73% | 116 | 9 | 2 | — |

No `max_catalog_rows` or `masterstar_detection_cap_*` binding at any N (cap columns empty).

**Note:** `n_after_snr_filter` **exceeds** `n_pass1_raw` for N ? 3.25 because DAO pass-2
(targeted Gaia fill-in) adds detections after pass-1; pass-2 threshold is independent of sweep N.
Above N ? 4.5, `n_after_snr` plateaus ~**3490** while pass-1 continues to fall — SNR filter +
pass-2 set the floor, not pass-1 threshold alone.

Magnitude splits use zeropoint fit on GAIA_MATCHED rows (same method as `CURSOR_RESULT_dao_only_verify.md`).

---

## 2b.4 — Log-log sanity

Fit `log(n_pass1_raw)` vs `log(threshold_adu)` over 23 uncapped points:

| Metric | Value |
|--------|------:|
| Slope | **?1.58** |
| Passes (slope < ?0.5)? | **YES** |

Stellar-field expectation (~?1 to ?2) met. Part 2 flat curve would have **failed** this gate.

---

## 2b.5 — Decision table (no recommendation)

Operating region near current config (N ? 3.8–4.0):

| N | dao_only_frac | dao_only_G_lt_16 |
|---|--------------:|-----------------:|
| 3.75 | 6.4% | 168 |
| 4.0 | 4.9% | 120 |
| 4.5 | 4.8% | 117 |
| 5.0 | 4.8% | 118 |
| 6.0 | 4.7% | 114 |

Rebuild measured **3.93%** DAO_ONLY at config N=3.8 (full `generate_masterstar` with optimizer);
this sweep (detection only, no optimizer passes) gives **~4.9%** at N=4.0 — same ballpark.

**DECISION REQUIRED — Milan picks N.**

---

## Contradictions with Part 2 report

| Part 2 | Part 2b (correct) |
|--------|-------------------|
| DAO_ONLY 3.93% at all N | **4.7–70%** depending on N |
| pass-1 flat ~3470–6417 | pass-1 **1492–15641**, slope ?1.58 |
| No N reproduces 2521 | N=4.0 pass-1 **2578** on correct path |
| N selection table usable | Part 2 table **discarded**; use 2b table above |

---

## Files changed

- `dev/scripts/audit_stage3_part2b_threshold_sweep.py`
- `dev/results/CURSOR_RESULT_audit_stage3_part2b.md`

**STOP GATE 2b** — awaiting Milan N decision before Parts 3–5.
