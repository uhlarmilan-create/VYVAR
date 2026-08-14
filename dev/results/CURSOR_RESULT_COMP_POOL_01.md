# CURSOR RESULT - COMP-POOL-01 (full)

Date: 2026-08-14
Register ID: COMP-POOL-01

## Verdict

Comparison-star handling is split into **pool** (star-only, uncapped, draft-derived thresholds) and **assignment** (pair properties + recorded relaxation). Stages 1-3 are committed locally; **not pushed**. Anchors invalidate at Stage 2 by design; draft rebuild needed for post-change LC metrics.

Commits: Stage 1 `ffb4402`, Stage 2 `696c849`, Stage 3 (this follow-up).

## Stage summary

| Stage | Content | Selection numbers |
|------:|---------|-------------------|
| 1 | Howell+sys noise curve, NP validation, scintillation compare, derived thresholds | Unchanged (verified) |
| 2 | Admit pool; remove ~150 cap; wire `build_global_comp_pool` | Change: 512 plan 140 -> admitted **187** |
| 3 | Explicit assignment relax order in provenance | Assignment math unchanged |

## Noise model (draft 512)

```
sigma_total^2 = sigma_phot^2(Howell) + sigma_sys^2
```

| Parameter | Value |
|-----------|------:|
| gain | 3.17 e-/ADU (equipment) |
| RN | 7.6 e- |
| sky median | ~1550 ADU |
| aperture area median | ~36.6 px^2 |
| sigma_sys | 0.00974 +/- 0.00024 mag |
| scint (Osborn/Young, D=0.2 m) | 0.00199 mag |
| sys/scint | **4.90** (P-R2; P-02/WIDE-ERR) |
| NP/param median ratio | 1.12 |

Assumption: bulk of field stars are non-variable.

## Derived thresholds (draft 512)

See `CURSOR_RESULT_COMP_POOL_01_S2.md` table. Operative faint limit is where sigma_phot = sigma_sys (G~11.46, SNR~79).

## Pool decisions

Machine-readable: `dev/results/COMP_POOL_01_s2_512_decisions.csv` (+ 435, 510).

BO CVn (`1498613634033133184`): 5 -> 4 eligible; dropped `1497368849430107904` for faint+dilution (P-R3).

## Assignment

Relax order (recorded): colour_tier_widen_T1_to_T4 > adaptive_delta_mag > sparse_fallback_path.

No SysRem/TFA. No per-frame rejection. No colour in pool admission. No pool size cap.

## Validation matrix

| Case | Result |
|------|--------|
| Draft 512 rich | Pool 187; parametric OK; NP usable from ~G8 |
| Draft 510 sparser proxy | Pool 176; parametric determined |
| Draft 435 deeper MS | Pool 115; inv_eta p84 tighter (0.66) |
| Second rig | **Not available** in Archive (named gap) |

## Literature

See S2 memo table (Broeg, CSI 2264, Sokolovsky/VaST, Kjeldsen & Frandsen, Astrokit, Seager/Howell dilution).

## Pre-registered

| Rule | Fired? |
|------|--------|
| P-R0 | Yes - named chosen: lin_frac 0.85, NP min_bin_n 8, dilution p16/p10/p05 ladder |
| P-R1 | Soft - ratio 1.12; reported |
| P-R2 | Yes - sys/scint ~4.9 |
| P-R3 | Yes - one BO pool drop explained |
| P-R4 | Yes - pool size not tuned |

## Register diff

`docs/VYVAR_AUDIT_2026_REGISTER.md` Wave 10: COMP-POOL-01, COMP-POOL-SCINT.

## Not done / needs Milan

- Push authorization
- Draft 512 rebuild for LC metric before/after
- Scatter-vs-magnitude figure packaging
- Newton/C9.25 second-rig noise-curve test when data exist
