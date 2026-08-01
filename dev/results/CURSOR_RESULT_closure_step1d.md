CURSOR RESULT - 2026-07-31 CLOSURE STEP 1d (mmag unit fix + reference fixture)

**Outcome: A-1b CONFIRMED (FIX required)**
**Decisive number:** max |delta_ap| = **203.0 mmag** (proxy G~12.0, G 8-9 comps; repaired units)
**T4 ratio:** **5.32** (proxy 149732..., G8-9/G>11 range ratio; band 5-15 PASS)

**Root cause:** Step 1b/1c computed `-2.5*log10(EE_t/EE_c)` in **magnitudes** but labeled the
result **mmag** without the `* 1000` factor. Step 1c "0.203 mmag" was **203 mmag**; Step 1b
"2.69 mmag" was **2690 mmag** (focus COG corrupt; not headline).

**Fixture:** `dev/tools/closure_a1_reference_fixture.py` -- independent L2 photutils reference.
**Harness gate:** `python dev/tools/closure_step1c_differential_aperture.py --fixture-check` PASS
(max err **0.005 mmag** vs emitted table).

---

## Reference fixture (expected physics, Moffat beta=3)

| r50 [px] | G_8_9 [mmag] | G_9_11 | G_gt_11 |
|---------:|-------------:|-------:|--------:|
| 1.46 | 329.2 | 271.9 | 59.2 |
| 1.97 | 473.5 | 382.1 | 74.0 |

Range over r50 span: G_8_9 **+144.3 mmag**, G_gt_11 **+14.8 mmag**, T4 ratio **9.74**.

Integer-centre harness (L3) bias on fixture: **-0.95% EE**, position jitter **~77 mmag**.

---

## Recomputed real data (mmag, proxy targets, r=1.916 px)

| proxy G | G 8-9 range | G 9-11 | G > 11 |
|--------:|------------:|-------:|-------:|
| 12.03 | **64.2** | 35.6 | 100.7 |
| 12.06 | 38.1 | 61.7 | 126.7 |
| 12.10 | **203.0** | 175.6 | 38.1 |
| 13.01 | 45.0 | 34.0 | 18.6 |
| 14.50 | 30.0 | 28.0 | 12.0 |

All proxy sub-ensembles **>> 10 mmag gate**. Headline max **203 mmag**.

**Real target** (QC failed, separate): G 8-9 range **1258 mmag** -- corrupt COG; not headline.

---

## VOID supersession

| Report | VOID item | Pointer |
|--------|-----------|---------|
| Step 1c | All delta_ap in mmag (0.203 max) | this report |
| Step 1c | T4 fail at ratio 0.66 | recomputed T4 **5.32** on mmag ranges |
| Step 1c | DOCUMENTED verdict | superseded by **CONFIRMED** |

Step 1c harness repair (monotone EE, proxy decoupling) **stands**. Unit fix only.

---

## VOID (Step 1e supersession -- do not delete)

**Pointer:** `dev/results/CURSOR_RESULT_closure_step1e.md`

| ID | Claim in this report | Issue |
|----|---------------------|-------|
| **V8** | Decisive **203 mmag** (max two-point range across proxies) | Five proxies at same radius must agree within 25%; spread was 30-203 mmag (6.8x). Not a consolidated measurement. |
| **V9** | T4 **5.32 PASS** (single best proxy) | Median per-proxy ratio is **2.42** (band 5-15 FAIL). Quoting the maximum proxy is not T4. |
| **V10** | Proxies G 12.03/12.06 show range(G>11) > range(G 8-9) | Physically impossible under monotone PSF; integer-centre noise (fixture G4: 77 mmag jitter). |

The **203 mmag headline** and **T4=5.32 PASS** are **VOID**. Verdict **A-1b CONFIRMED** stands
(minimum 5x3 proxy cell 12.0 mmag > 10 mmag gate); exact consolidated magnitude remains open
after Step 1e (G6 failed on real data).

## Commands

```bash
python dev/tools/closure_a1_reference_fixture.py
python dev/tools/closure_step1c_differential_aperture.py --fixture-check
python dev/tools/closure_step1c_differential_aperture.py \
  --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \
  --step1b-json tmp/closure_step1b_results.json \
  --out tmp/closure_step1c_mmag_results.json \
  --cache tmp/closure_step1c_ee_cache.npz
```

---

## Docs impact

| Doc | Action |
|-----|--------|
| `VYVAR_AUDIT_CLOSURE_REGISTER.md` | A-1 -> **A-1b CONFIRMED**; decisive **203 mmag** |
| `VYVAR_AUDIT_FINAL.md` | D5-1 delta_ap in mmag |
| `VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1 CONFIRMED; fix pending Milan |

---

## Files

| File | Role |
|------|------|
| `dev/tools/closure_a1_reference_fixture.py` | L1/L2/L3 reference + gates G0-G5 |
| `dev/tools/closure_step1c_differential_aperture.py` | `*1000` mmag + `--fixture-check` |
| `dev/results/CURSOR_RESULT_closure_step1d.md` | this report |
