CURSOR RESULT - 2026-08-01 CLOSURE STEP 1j (test F(12) normalisation and sky annulus)

**Outcome: J-a -- sky subtraction per star.** J1 fitted slope **-0.285** (expected **-0.4**); G 11.52
vs G 11.53 (dG = 0.006 mag) F(12) ratio **2.60x** with sky_ann-global offset **-33 ADU/px**.
J3 does not restore catalogue consistency under any annulus tested. J2 field-centre correlation
**-0.53** indicates a large-scale gradient component.

---

## What I did

Built and ran `dev/tools/closure_step1j_test_f12_normalisation.py` (J1-J4) on all 35 closure
stars and 4083 star-frames from the Step 1f cache. Withdrew incorrect Step 1i production annulus
claim from `VYVAR_AUDIT_FINAL.md`.

## VYVAR_AUDIT_FINAL.md withdrawal (section 1)

**Withdrawn:** Step 1i statement that production uses the same 25-45 px annulus as fixture L2.

Production uses `annulus_inner_fwhm = 4.75`, `annulus_outer_fwhm = 9.0` scaled per star by
`fwhm_gaussian_px` (`photometry_core.py` ~11.4-21.6 px at fw = 2.395). **25-45 px is closure
harness only.** Step 1i E4 is evidence about the harness, not the production path, until J3
tests production geometry separately (done below; production J1 slope still -0.266).

---

## J1 -- F(12) internal consistency (decisive)

4083 star-frames, 34 stars with F(12) > 0 at harness annulus 25-45 px, fitted centroids from cache.

| metric | value | source |
|--------|------:|--------|
| Fitted slope d log10(F12) / dG | **-0.285** | J1 harness |
| Expected slope | -0.4 | catalogue flux |
| Scatter RMS about fit [dex] | 0.244 | J1 |
| Same-mag pairs (dG <= 0.05) | 16 | J1 |
| Same-mag F(12) ratio median | 1.008 | J1 |
| Same-mag F(12) ratio p95 | **2.449** | J1 |
| Same-mag F(12) ratio max | **2.605** | J1 |
| Pearson(star median F12 residual, star median sky_ann - global) | 0.217 | J1 |

**Smoking gun (section 0 confirmed):**

| proxy | G | median F(12) [ADU] |
|-------|---:|-------------------:|
| 1497368849430107904 | 11.52 | 43 222 |
| 1497091703781835776 | 11.53 | 16 595 |
| Ratio | dG = **0.006** | **2.605x** |

Expected flux ratio at dG = 0.006: `10^(-0.4 * 0.006) = 0.994` (~1%). Measured F(12) differs
by **2.6x**. Sky_ann - global: +46 ADU/px (G 11.52) vs +79 (G 11.53) in Step 1i E4; Step 1j
median difference **-33 ADU/px** (sign convention: higher annulus sky on G 11.53 side of pair).

**Conclusion:** F(12) is not the star's enclosed flux; it is flux minus a sky estimate that
varies by tens of ADU/px between stars of the same catalogue magnitude. The Step 1i "F(12)
drops 6.6x with G" statement conflates a sky offset with a magnitude trend.

Source: `tmp/closure_step1j_diagnostics.json` J1.

---

## J2 -- annulus examined (25-45 px harness)

591 star-frames on 20 evenly spaced representative frames.

| metric | value |
|--------|------:|
| Pipeline local sky at star position | **not accessible** (no VY_SKYSF / sky-surface coeffs in draft FITS/CSV) |
| Sources in annulus median | 3 |
| Pearson(sky_ann - global, n_sources) | 0.237 |
| Pearson(sky_ann - global, field-centre separation) | **-0.531** |
| Annulus median minus 3-sigma clip mean [ADU] | 0.24 |

Stellar contamination alone (median 3 sources in ~4400 px annulus) is weakly correlated with sky
offset. **Field position** correlates more strongly (-0.53), consistent with large-scale sky
structure across the frame contributing to annulus sky error.

Source: `tmp/closure_step1j_diagnostics.json` J2.

---

## J3 -- annulus geometry control

Three geometries, same centroids, everything else unchanged:

| geometry | r_in / r_out [px] | J1 slope | same-mag ratio med | same-mag ratio max |
|----------|-------------------|--------:|-------------------:|-------------------:|
| harness (current) | 25-45 | -0.285 | 1.008 | 2.605 |
| narrow | 12-20 | -0.266 | 1.041 | 2.605 |
| production-scaled | ~11.4-21.6 at fw=2.395 | -0.266 | 1.033 | 2.605 |

**F(12) catalogue consistency is NOT restored** under any geometry (slopes remain ~-0.27 vs -0.4;
same-mag max ratio unchanged at 2.605 for the G 11.52/11.53 pair).

**EE std per proxy (EE at 1.916 px):**

| proxy G | harness | narrow 12-20 | production-scaled |
|--------:|--------:|-------------:|------------------:|
| 11.52 | 0.0175 | 0.0177 | 0.0171 |
| 11.53 | 0.0320 | **0.0180** | **0.0173** |
| 12.03 | 0.0398 | 0.0662 | 0.0609 |
| 12.59 | 0.0638 | n/a | n/a |
| 12.68 | 0.0750 | 0.0505 | 0.0490 |

Narrow/production annuli **halve EE scatter for G 11.53** but do not fix J1 F(12) vs G. G 12.59
COG fails at smaller annuli (edge/boundary). Fixture 25-45 px is not proven as the sole defect;
sky offset per star persists across geometries.

Source: `tmp/closure_step1j_diagnostics.json` J3.

---

## J4 -- amp/peak threshold

4083 star-frames with Gaussian fit amp/peak.

| percentile | amp/peak |
|------------|--------:|
| p5 | 0.211 |
| p16 | 0.427 |
| median | 0.818 |
| p95 | 0.977 |

**Proposed threshold (distribution shape):** amp/peak < **0.427** (16th percentile) -- lower
envelope of converged fits. Rejects **654/4083** (16%) star-frames.

| threshold | rejected | frac | G 12.59 proxy rejected |
|----------:|---------:|-----:|-----------------------:|
| 0.20 | 188 | 4.6% | 139/139 |
| 0.25 | 266 | 6.5% | 139/139 |
| 0.30 | 365 | 8.9% | 139/139 |
| 0.427 (p16) | 654 | 16.0% | (included in total) |

Threshold 0.25-0.30 catches all G 12.59 proxy frames (amp/peak ~ 0.09 in Step 1i) while leaving
G 11.52 untouched. Cause-based, not EE-band filtering.

Source: `tmp/closure_step1j_diagnostics.json` J4.

---

## Mechanism summary (amends Step 1i)

| Step 1i claim | Step 1j verdict |
|---------------|-----------------|
| F(12) drops 6.6x with G (magnitude trend) | **Partially reframed:** catalogue flux ratio G 11.52->12.68 is 2.91x; measured F(12) 5.83x; at identical G (11.52 vs 11.53) measured 2.6x. Offset dominates. |
| Production annulus same as harness | **Withdrawn** |
| Normalisation family correct | **Confirmed** (E5 placement excluded) |
| Cause: faint-star small F(12) | **Replaced:** per-star sky estimate error (~tens ADU/px) |

---

## What Step 1k must change

1. **Diagnose sky_ann per star:** compare harness annulus sky to independent local estimate (e.g.
   clipped median in outer ring, or production annulus geometry with same comparison).
2. **Do not filter on EE vs Moffat band** (R8).
3. **Evaluate amp/peak gate** (~0.25-0.43) on cause grounds from J4 before re-measurement.
4. **Production-path J1:** run same F(12) vs G test using production photometry sky values from
   `proc_*.csv` column `sky_adu_per_px_annulus` (production geometry) vs harness recomputation.
5. **Fixture:** 25-45 px not changed yet (J3 did not restore F(12) consistency); revisit after
   sky model is understood. No consolidated delta_ap.

---

## Docs impact

| Doc | Action |
|-----|--------|
| `VYVAR_AUDIT_CLOSURE_REGISTER.md` | Mechanism sentence amended (J-a) |
| `VYVAR_AUDIT_FINAL.md` | Step 1i production annulus entry withdrawn; Step 1j added |
| `VYVAR_DECISIONS.md` | no entry |
| `VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1j status |
| `closure_a1_reference_fixture.py` | no change (J3 did not prove 25-45 is sole defect) |

---

## Errors (if any)

None.

---

## Files changed

| File | Role |
|------|------|
| `dev/tools/closure_step1j_test_f12_normalisation.py` | J1-J4 harness |
| `dev/results/CURSOR_RESULT_closure_step1j.md` | this report |
| `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` | J-a mechanism |
| `docs/VYVAR_AUDIT_FINAL.md` | withdraw Step 1i production note; add Step 1j |
| `docs/VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1j status |

## Commands

```bash
python dev/tools/closure_step1j_test_f12_normalisation.py \
  --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \
  --step1b-json tmp/closure_step1b_results.json \
  --step1g-json tmp/closure_step1g_results.json \
  --cache tmp/closure_step1f_ee_cache.npz \
  --out tmp/closure_step1j_diagnostics.json
```
