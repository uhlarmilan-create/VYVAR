CURSOR RESULT - 2026-08-01 CLOSURE STEP 1g (restore differential configuration, measure A-1)

**G6: FAIL.** **G7: PASS.** **G8: PASS.** **G9: PASS.** **No decisive M2 number** (G6 failed).

---

## What I did

Applied F1 to `closure_step1f_differential_aperture.py`: proxies G 11.5-13.0 at clamp 1.916 px,
excluded from comparison subsets (`exclude=set(proxy_ids)`), added gate G9, re-ran measurement
on existing Step 1f cache. Added target-radius sweep to `closure_a1_reference_fixture.py`
(--emit). Marked V11-V14 VOID in Step 1f report.

## Output / findings

### Gates (first four lines)

1. **G6 FAIL** -- proxy p95-p5 spread exceeds 1.25x on all sub-ensembles (G 8-9 max/min **4.79**;
   ranges 94-452 mmag across five proxies G 11.52-12.68).
2. **G7 PASS** -- harness COG vs fixture L2 max err **0.005 mmag**.
3. **G8 PASS** -- predicted noise p95-p5 **5.0-9.9 mmag** (ceiling 48.1 mmag).
4. **G9 PASS** -- all proxies disjoint from comp lists (intersection **0**); G 8-9 and G 9-11
   differential (radius sep **1.10** and **0.73 px**); G > 11 **non-differential** (sep **0.15 px**
   < 0.3 px threshold, labelled not differential per gate spec).

### F1 configuration

**Selected proxies (G 11.5-13.0, ranked by predicted sigma delta_ap):**

| catalog_id | G | frames | pred noise p95-p5 [mmag] | intersection G8_9 / G9_11 / G>11 |
|------------|---|--------|-------------------------:|----------------------------------|
| 1497368849430107904 | 11.52 | 139/139 | 5.0 | 0 / 0 / 0 |
| 1497091703781835776 | 11.53 | 131/139 | 5.0 | 0 / 0 / 0 |
| 1499960535777095296 | 12.03 | 139/139 | 6.6 | 0 / 0 / 0 |
| 1499238946911605504 | 12.59 | 139/139 | 9.3 | 0 / 0 / 0 |
| 1498488702024456448 | 12.68 | 137/139 | 9.9 | 0 / 0 / 0 |

**Proxy radius audit:** `target_r_override=PROXY_R_AP` -> **1.916 px on all 139 frames** (confirmed).

**Comparison subset sizes after exclusion:** G 8-9 **6**, G 9-11 **12**, G > 11 **12** (not thin).

### M1 (per proxy; G > 11 non-differential)

**G 8-9 (differential, fixture expectation 144.3 mmag):**

| proxy G | p95-p5 [mmag] | slope x span [mmag] | Pearson r50 | vs fixture |
|--------:|--------------:|--------------------:|------------:|-----------:|
| 11.52 | **94.3** | 96.8 | 0.54 | -50 mmag |
| 11.53 | 171.8 | 171.8 | 0.54 | +28 mmag |
| 12.03 | 264.5 | 264.5 | 0.54 | +120 mmag |
| 12.59 | 451.9 | 451.9 | 0.54 | +308 mmag |
| 12.68 | 425.8 | 425.8 | 0.54 | +282 mmag |

Brightest proxy (G 11.52) lands **94 mmag**, ~65% of fixture 144.3 mmag. Fainter proxies show
larger ranges (proxy EE at 1.916 px varies with magnitude). G6 fails because proxies span G
11.5-12.7 and do not agree within 25%.

**G > 11:** reported for audit only; G9 marks **non-differential** (median comp radius 2.07 px
vs target 1.916 px).

### M2 consolidated

**Blocked** (G6 failure). No headline `X +/- Y mmag` registered.

Diagnostic median G 8-9 p95-p5 across proxies would be **~265 mmag** (dominated by fainter
proxies) -- not admissible under G6.

### M3 real target

`1498135552633294976`: **25/139** frames admissible (C1); QC **failed** (< 90%). G 8-9 p95-p5
**1732 mmag** on 25 frames -- not headline.

### M4 B.5 / B.6 (valid F1 configuration)

**T3 synthetic identity:** **0.0 mmag** PASS (all three proxies checked).

**B.5 real data (frozen k_i, scale = r50_frame):** range **188-416 mmag** per proxy. Non-zero
on real data vs synthetic T3: measures **star-to-star profile variation** and finite sample
r50 scaling on heterogeneous PSFs, not a broken identity.

**B.6 reopt:** G 8-9 p95-p5 **135 mmag**; sky correlation **-0.55**.

### T4

Median G 8-9 / G > 11 ratio **0.79** (band 5-15 FAIL; fixture 9.74 at r=1.916 px). G > 11
column is non-differential at this configuration; T4 not interpretable until G > 11 passes G9
radius separation or is excluded from the ratio.

### Step 1f VOID (V11-V14)

Marked in `dev/results/CURSOR_RESULT_closure_step1f.md`.

### Fixture target-radius sweep (added to fixture --emit)

At r_target = 1.916 px: G 8-9 **+144.3 mmag**, T4 **9.74** (matches A-1 design).
At r_target = 3.016 px (Step 1f proxy config): G 8-9 **-8.6 mmag**, T4 **~0.06** (matches Step 1f).

## Register wording (section 5 -- G6 failed)

> **A-1b CONFIRMED.** Seeing-correlated, magnitude-dependent differential aperture systematic,
> magnitude of order 10^2 mmag. Target on the `r_min` clamp at 1.916 px against comparisons at
> larger magnitude-binned radii, with no curve-of-growth correction. Exact value open.
> Physics expectation from `dev/tools/closure_a1_reference_fixture.py`: 144.3 mmag for G 8-9
> comparisons over the anchor's measured r50 span.

Step 1f **48.0 mmag** superseded (V11). Step 1g F1 configuration is valid (G9 PASS) but G6
requires proxy agreement; magnitude spread G 11.5-12.7 prevents consolidation.

## What Step 2 inherits

- **F1 configuration is correct:** disjoint proxies at clamp 1.916 px; G 8-9 is differential.
- **G6 blocker:** need proxies at similar G (narrow band) or a single canonical proxy G ~11.5-12.0
  with multiple independent validation paths, not five stars spanning 1.2 mag.
- **Brightest proxy (G 11.52):** G 8-9 p95-p5 **94 mmag** -- closest to fixture 144.3 mmag so far.
- **G > 11 sub-ensemble:** non-differential at clamp vs comp radii (~2.0 px); do not use for T4
  until radii are separated or comps are fainter/smaller-aperture.
- B.5/B.6 on valid configuration available for Milan's option choice.

## Docs impact

| Doc | Action |
|-----|--------|
| `VYVAR_AUDIT_CLOSURE_REGISTER.md` | A-1: exact value open; 48.0 mmag superseded |
| `VYVAR_AUDIT_FINAL.md` | Step 1g status |
| `VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1g; G6 open |
| `closure_a1_reference_fixture.py` | target-radius sweep added; G0-G5 pass |

## Errors (if any)

None.

## Files changed

| File | Role |
|------|------|
| `dev/tools/closure_step1f_differential_aperture.py` | F1 + G9 (Step 1g config) |
| `dev/tools/closure_a1_reference_fixture.py` | target-radius sweep |
| `dev/results/CURSOR_RESULT_closure_step1g.md` | this report |
| `dev/results/CURSOR_RESULT_closure_step1f.md` | V11-V14 VOID |

## Commands

```bash
python dev/tools/closure_step1f_differential_aperture.py \
  --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \
  --step1b-json tmp/closure_step1b_results.json \
  --out tmp/closure_step1g_results.json \
  --cache tmp/closure_step1f_ee_cache.npz
```
