# CURSOR RESULT - COMP-POOL-01 Stage 2

Date: 2026-08-14
Commit target: Stage 2 only (pool admission + cap removal). No push.

## What I did

Wired draft-derived comparison-star **pool admission** (star-only criteria) and removed the ~150 spatial pool cap. Assignment (colour / mag / spatial / Broeg) is unchanged in this stage.

## Design split (authorized)

| Stage | Question | This commit |
|-------|----------|-------------|
| 1 | Noise curve + derived thresholds | Done (`ffb4402`) |
| 2 | Pool admission; no size cap | **This** |
| 3 | Per-target assignment / relaxation | Deferred unless needed |

## Code changes

- `src_py/comp_pool_noise.py`: physical faint limit (`sigma_phot == sigma_sys`), dilution attach + stepped percentile, `admit_pool_stars`, `analyze_draft_comp_pool`
- `src_py/photometry_core.py`: `build_global_comp_pool` applies derived admission when `comp_pool_derived_admission`; skips legacy RMS prefilter when admission runs; plan pool uses `comparison_stars_pool_n` (0 = uncapped)
- `src_py/pipeline.py`: `select_comparison_stars_spatial_grid` returns all filtered stars when `n_comp <= 0`
- `src_py/config.py`: `comp_pool_derived_admission=True`, `comparison_stars_pool_n=0`
- `src_py/night_run.py`, `src_py/app.py`: pending default `n_comparison_stars=0`

## Draft 512 derived thresholds (with rules)

| Quantity | Value | Rule |
|----------|------:|------|
| detect_frac_min | 1.000 | p16 of detect_frac among mag_g<=14 |
| faint_limit_g | 11.455 | mag where sigma_phot = sigma_sys |
| faint_limit SNR | ~78.8 | MAG_ERR_SCALE / sigma_total at that mag (community thumb ~100) |
| bright upturn | false | NP bright-end rise vs mid; not seen |
| default_lin_frac | 0.85 | **CHOSEN / D1-2** (linearity knee unmeasured) |
| dilution_threshold | 0.9876 | p16 of D (Seager/Howell); missing D does not reject |
| stability MAD excess | 1.920 | p84 of scatter_mad / sigma_total |
| stability IQR excess | 1.951 | p84 of scatter_iqr / sigma_total |
| inv_eta excess | 0.876 | p84 of 1/vonNeumann |
| sigma_sys | 0.00974 mag | bright G8-10.5 residual |
| scint_predicted | 0.00199 mag | Osborn/Young, D=0.2 m |
| sys/scint | ~4.90 | **P-R2** fired; do not adjust |
| NP/param median ratio | ~1.12 | usable bins; within soft agreement |
| nonparametric_min_bin_n | 8 | **CHOSEN** (NP validation only, not admission) |
| edge margin | existing | kept (chip_interior / safe_bbox) |

Assumption (stated): bulk of field stars are non-variable; median scatter traces noise.

## Pool size (no cap)

| Draft | Stars summarized | Admitted | Prior plan comps |
|------:|-----------------:|---------:|-----------------:|
| 512 | 680 | **187** | 140 |
| 510 (sparser proxy) | 644 | 176 | (same rig) |
| 435 (deeper MS) | 2770 | 115 | (same rig) |

P-R4: pool size was not tuned.

## BO CVn (1498613634033133184) - P-R3

Draft 512 ensemble before: 5 TIER1 comps.

| Comp | After Stage-2 pool |
|------|--------------------|
| 1497771992240531712 | still admitted |
| 1499200223486564608 | still admitted |
| 1497974027502858240 | still admitted |
| 1499053747922698240 | still admitted |
| 1497368849430107904 | **dropped** |

Drop reason (stated criteria, not LC appearance): `fainter_than_11.455` and `dilution<0.9876` (D=0.490). Mag_g=11.52.

`check_scatter` / `ac_scatter` / `lc_rms_ooe` are **unchanged until rebuild** (assignment not re-run in this stage). After rebuild, ensemble size for BO CVn is expected to be 4 unless Stage-3 assignment replaces the dropped star from the larger pool.

## Sparse-field behaviour

No true sparse second-rig draft with proc CSVs is in Archive. Among available wide-rig nights, draft **510** has the fewest stars in-frame (659 in first proc; 644 summarized). Parametric curve remained determined (`sigma_sys=0.00973`, n_fit bright asymptote). Pool admitted 176 with the same code path and no mode switch.

Draft 435 (richer MASTERSTAR, 2770 stars) shows the gate is not a fixed star count: admitted 115, with a tighter inv_eta p84 (0.657) that drops the archived BO comps for **inv_eta>0.657** only. That is a draft-derived threshold difference, not a hard-coded cut.

## Second rig

No Newton / C9.25 draft with proc products was available in Archive. Universality across plate scale is therefore **untested** on a second instrument in this stage; same-code dual-draft (512 vs 435) on the 200 mm wide field shows different fitted floors (0.00974 vs 0.01184) without config changes.

## Literature / cross-tool table

| Practice | Source | VYVAR Stage 2 |
|----------|--------|---------------|
| Comp pool without prior field study; weights then down-weight bad stars | Broeg et al. 2005; Jena AIfA notes | Pool = star-only; Broeg weights stay in assignment |
| Compare scatter to median at same magnitude | CSI 2264 / Stauffer et al.; common survey practice | NP curve validates; parametric is operative |
| Variable if >=~3x median noise | CSI 2264 | Data-derived p84 excess ~1.92 (stricter than 3x cut, different estimator) |
| Variability index ~1.5 | Kjeldsen & Frandsen 1992 | Excess thresholds ~1.9 MAD/IQR |
| MAD/IQR for short LCs | Sokolovsky et al. 2017 (VaST) | Used; no sigma-clip |
| 1/eta for slow drift | von Neumann ratio; used in VS surveys | Used alongside scatter |
| Ensemble ~10; dmag~2; few arcmin | Astrokit | Assignment Stage 3 / existing tiers (not pool) |
| SysRem/TFA not default for unknown variables | VaST optional; literature warning | Excluded (task) |
| Dilution D = F/(F+neighbors) | Seager & Mallen-Ornelas 2003; Howell 2006 | `dilution.py` |

## Pre-registered rules

| Rule | Status |
|------|--------|
| P-R0 | Named chosen: `default_lin_frac=0.85` (D1-2); `nonparametric_min_bin_n=8` (NP usability only). Dilution percentile ladder p16->p10->p05 is a named procedure when D piles at 1.0. |
| P-R1 | NP/param median ratio ~1.12 on 512; no paper-over. |
| P-R2 | sys/scint ~4.9; both reported; not adjusted (bears on P-02 / WIDE-ERR). |
| P-R3 | One BO CVn pool member dropped; reasons stated above. |
| P-R4 | Pool sizes not tuned. |

## Explicitly not derived / not done here

- Full draft rebuild (anchors invalidate at Stage 2 by design)
- Stage 3 assignment / relaxation provenance text
- Second-rig noise-curve test
- Scatter-vs-magnitude figure PNG (CSV curves written; figure deferred to methods paper packaging)
- `masterstar_detection_cap_*` unchanged (detection, not comparison pool)

## Machine-readable outputs

- `dev/results/COMP_POOL_01_s2_512_summary.json`
- `dev/results/COMP_POOL_01_s2_512_decisions.csv`
- `dev/results/COMP_POOL_01_s2_512_np_curve.csv`
- `dev/results/COMP_POOL_01_s2_512_curve_ratio.csv`
- same pattern for drafts 435 and 510

## Measured delta (selection path)

- Plan pool cap 150 -> uncapped (`comparison_stars_pool_n=0`)
- Global pool: derived admission replaces legacy `max_comp_rms` prefilter when enabled
- Draft 512 admitted pool **187** vs prior plan **140**
- BO CVn eligible comps 5 -> 4 (one fails faint+dilution)

## Tests

`pytest dev/tests/test_comp_pool_noise_s1.py dev/tests/test_comp_pool_noise_s2.py` - 7 passed.
