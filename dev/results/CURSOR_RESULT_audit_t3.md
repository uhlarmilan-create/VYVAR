CURSOR RESULT � Science Audit Tranche 3 (2026-07-30)

Base: `06ed950` on `origin/main`. **Bundled fix is local only � not pushed.**

---

## Tranche 3 accepted � causal chain inverted

July diagnostic arc premise reversed: draft_435 `proc_*` had **doubled gradient** (`2g`), draft_450 plain had natural `g`. Two defects cancelled on 435:
- P-10 sign error ? inflated `bg_std` ? effective DAO threshold **3.81?** (nominal 2.1)
- Clean catalogue (3.7% `DAO_ONLY`) was a side-effect, not correct physics

`283/136.8 ? 2.07` SKYSF forensics ratio matches doubled-gradient prediction.

---

## Implemented locally (single bundle � do not ship P-10 alone)

| Item | Change |
|------|--------|
| **P-10** | `z_s = work - bg_median`; `out = work - surf` (`pipeline.py`) |
| **Estimator** | `_pixel_noise_sigma_pp_adu` + `_dao_noise_sigma_adu`; used in both DAO paths (~7425, ~8171) |
| **Threshold** | `masterstar_dao_threshold_sigma` **2.1 ? 3.8** (`config.json`, `config.py`, registry help) � targets ~175 ADU at ?_pp?46 |
| **Tests** | `test_preprocess_sky_surface.py` (P-10), `test_dao_sigma_pp_estimator.py` (?_pp) |
| **I-12** | (from T2) PM unavailable warning � already local |

**NOT done:** anchor re-run, ?flux/?err/?_bkg_ap/?�/DAO delta table, anchor re-cut, push.

---

## Documentation updated

| File | Change |
|------|--------|
| `docs/VYVAR_DECISIONS.md` | `P-10-SKYSURF-SIGN` + three July preprocess regimes |
| `dev/results/CURSOR_RESULT_draft451_analysis.md` | post-fix expectation table **VOID** |
| `dev/validation/VYVAR_VALIDATION_LEDGER.json` | *(pending � append P-10 note to VL-ANCHOR-WCSINV)* |

---

## Prediction after bundle (to verify on re-run)

| Quantity | P-10 only (wrong) | P-10 + ?_pp + 3.8? (this bundle) |
|----------|-------------------|----------------------------------|
| Pass-1 DAO | **> 8926** (catastrophic-looking) | **~2550 class** (threshold ~175 ADU) |
| `DAO_ONLY` | **> 40%** | **~3.7%** (verify, don't assume) |
| Large-scale residual | removed | removed |

---

## Next steps (Milan)

1. Review local diff; push **one commit** with full bundle.
2. Re-run anchor input; produce delta table.
3. Re-cut anchor + update `VL-ANCHOR-WCSINV` fingerprints.
4. Recalibrate `skysurface_regression` �K4 ratio bounds (were calibrated on doubled-gradient frames).
5. I-11 (Howell on sky-subtracted frames) � separate change after delta measurement.
