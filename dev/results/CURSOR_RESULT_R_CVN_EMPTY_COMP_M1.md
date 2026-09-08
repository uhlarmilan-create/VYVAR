CURSOR RESULT - 2026-09-08 R-CVN-EMPTY-COMP-M1

Date: 2026-09-08. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 14690ae.
Class: MEASUREMENT ONLY. No production code change.
Live 516/517 read-only. Sandbox C-EXPORT-GAP reused as log evidence
(tmp sandbox already cleaned). STOP with design menu; nothing wired.

## Premise (Rule 0.1)

Compared: the three C-EXPORT-GAP AAVSO empty-point IDs, versus the
star actually named R CVn, versus the three Phase-2A
`phase2a_empty_comp_drop` IDs, versus the production color-cap /
empty-pool code. They are three different sets. The ROADMAP row
name "R-CVN-EMPTY-COMP" and the tasked IDs do not name the same
mechanism. Numbers below are for the rule Milan asked for; they
are not a license to treat the three export failures as empty-pool.

## POST-453 blocker

`docs/VYVAR_PROCESS.md:190-194` (2026-07-27): a sandbox harness
that does not go through the production path cannot characterise
production. That is a process rule, not a science hold. This
measurement is read-only (production functions + existing log +
live 516 files). **No real blocker remains.** The ROADMAP
"blocked-on POST-453" tag is stale.

## Refute (governing, before tables)

1. **R CVn is not among the three tasked IDs.**
   R CVn = `1496795041799526400`, VSX type **M**, G=7.121,
   BP-RP=**5.675**. The three tasked IDs are CSS EW / HAT VAR /
   Gaia RS (see M1). 150041 is VSX type **RS** (an RS CVn), which
   is a naming collision with the row, not the star R CVn.

2. **The three tasked IDs are not the empty-comp set.**
   C-EXPORT-GAP log: they had comps (8 / 3 / 8) and wrote LC CSVs;
   Phase 2A `lc_rms=nan`; export FAILURE
   `no exportable LC points (flags/mag empty)`
   (`export_reports.py:997-1004`).
   Actual `phase2a_empty_comp_drop=3` IDs:
   `1497245497969274240`, `1498425548825498112`,
   `1497227287309482624`.

3. **Empty pool is not the color cap.**
   Those three are pinned ensembles with **n_pin=3**. Phase 1:
   `PinnedEnsembleInsufficientError` after
   `drops=[('1500467303261764096', 'rms_violation')]`,
   `n_survivors=2 < n_min=3`
   (`pinned_ensembles.py:439-441`, `:659-664`;
   `phase01_run.py:772-779` logs it as "neocakavana chyba").
   Color path on the same field would have left 214 / 1356 / 1370
   stars inside the 0.79 cap (M2).

4. **R CVn already has comps -- pinned T4, huge mismatch.**
   Phase 1 used the pin overlay
   (`photometry_comp.py:2221-2256`), 8 comps, DeltaBPRP median
   4.852 / max 5.629 (C-EXPORT-GAP log:70-73). Color cap never
   ran. Phase 2A succeeded (`lc_rms=0.0124`). AAVSO exported.
   Pin members are catalog tier 4; |dBP-RP| 4.14-5.63 (M2 pins).

5. **Architect tier numbers are the code fallbacks, not config.**
   `comp_selection_per_target.py:266-271` fallbacks are
   0.25 / 0.48 / 0.79 / unbounded. Live `config.json`
   `comp_color_tiers` = **0.15 / 0.30 / 0.55 / 1.10**.
   Ladder last rung is `comp_max_delta_bprp` **0.79**
   (`photometry_comp.py:1138-1157`); T4 1.10 is **not** on the
   ladder. Spatial filter does **not** apply the cap
   (`comp_selection_per_target.py:406-426`, COMP-ADMIT-03).
   Ranking uses the cap at `:1691` / `:1750` of
   `comp_selection_per_target.py` (ladder argument, not a spatial
   cut).

6. **If R CVn were not pinned and BP-RP were finite, the color
   path would empty.** Live MS: 0 of 3300 spatial survivors have
   |dBP-RP| <= 0.79. Minimum achievable |dBP-RP| in the field is
   **2.219**. That is the only target here for which Milan's
   red-target fallback is the governing gap.

7. **NaN-BP-RP still bypasses the cap on the color path.**
   `_select_comps_by_rms_then_color` sets `_delta_bprp_abs=0.0`
   when target BP-RP is not finite (`photometry_comp.py:1328-1334`).
   That is how a red target could theoretically pass 0.79 without
   a pin. R CVn's masterstars BP-RP is finite (5.675); the pin is
   why it passed, not NaN.

## Funnel functions (file:line)

Color-path (unpinned targets), `select_comparison_stars_per_target`
`photometry_comp.py:2029` via `phase01_run.py:718`:

| stage | function | lines | hard cut? |
|---|---|---|---|
| field | masterstars input | -- | -- |
| sat / zone / VSX / chip-margin / NSS / QSO / GAL | `_filter_comp_candidates_spatial_static` | `comp_selection_per_target.py:307-522` (gates `:364-404`, `:428-455`) | yes (measurability / known-var / geometry) |
| distance | same; COMP-ADMIT-03 | `:360-363`, `:406-411` | **no** on cand_mask; **yes** on `_base_mask` `:509-510` (min 60") |
| color cap | same | `:406-426` | **no** at spatial |
| D3 | `_build_candidates_pre_adaptive_mag` -> `apply_d3_comparison_candidacy` | `:524-555`; `d3_comparison_candidacy.py:45` | yes if columns present |
| mag | `_adaptive_mag_filter` | `:290-304` | **no-op** (COMP-ADMIT-03) |
| per-frame sat peak | `_apply_comp_metric_hard_filters` | `photometry_comp.py:2462` | sat peak / edge |
| MAD / pool RMS | `_ensemble_mad_filter_rms` | `:2602` | reduces ~530 -> 223 (log) |
| RMS loo ceiling | `_select_comps_by_rms_then_color` | `photometry_comp.py:1159`, ceiling log `:1243` | 223 -> 141 @ 0.080 |
| isolation 3 FWHM | same | after ceiling | 141 -> 95 |
| color ladder T1-T3-cap | `_bprp_tier_ladder_for_selection` + loop | `:1138-1157`, `:1348-1353`; cap passed at `comp_selection_per_target.py:1691,1750` | last rung 0.79 |

Pinned path (R CVn + empty-comp trio + CSS 149884):

| stage | function | lines |
|---|---|---|
| pin hit | `get_pinned_members_for_target` early return | `photometry_comp.py:2221-2256` |
| per-member sat/zone/dist/color-tier/RMS | `validate_pinned_member` | `pinned_ensembles.py:400-441` |
| abort if n_survivors < n_comp_min | `PinnedEnsembleInsufficientError` | `:659-664` |
| Phase 1 catch -> no comps | `phase01_run.py:772-779` | counted later as empty-comp |
| Phase 2A drop | `_phase2a_skip_empty_comps_target` | `photometry_lightcurve.py:77-95`; stub `:53-75` sets `ac_skip_reason=no_comps`, `lc_csv=""` |
| export of empty-mag LC | `_select_export_lc_rows` + `record_export_failure` | `export_reports.py:893`, `:997-1004` |

GAP: live `masterstars_full_match.csv` (41 cols) lacks
`vy_identity_gate` / `gaia_dao_resid_px` / `snr_ap_pixscaled`, so
D3 cannot be re-run on live MS. C-EXPORT-GAP log: D3 n_in=n_out
on this field (no-op). Spatial+color counts below use live MS
`_base_mask | det_mask` (n_field=3610). Later RMS/isolation
counts for unpinned targets are from the C-EXPORT-GAP log (sandbox
selection inputs).

## M1 -- identity and color

Local VSX: `VSX/vyvar_vsx_local_v2.db` table `vsx_data`, nearest
within 30". R CVn is **not** in the tasked trio.

| catalog_id | cohort | VSX name | type | G | BP-RP | RA | Dec | pin n | live n_comp |
|---|---|---|---|---|---|---|---|---|---|
| 1498842882207281152 | task export-fail | CSS_J135929.8+421520 | EW | 13.814 | 0.971 | 209.87456 | +42.25582 | 8 | 8 TIER1 |
| 1499842372636900992 | task export-fail | HAT-188-0003359 | VAR | NaN (mag_t 14.29) | 1.865 | 206.16253 | +39.64141 | 0 | 4 TIER1 |
| 1500410236033012352 | task export-fail | Gaia DR3 1500410236033012352 | **RS** | NaN (mag_t 14.46) | 1.200 | 207.41106 | +41.31337 | 0 | 8 TIER1 |
| **1496795041799526400** | **R CVn** | **R CVn** | **M** | **7.121** | **5.675** | 207.23767 | +39.54253 | 8 | 8 TIER4 |
| 1497245497969274240 | empty-comp drop | HAT-188-0000323 | VAR | 11.192 | 2.249 | 207.26854 | +40.13852 | 3 | 3 TIER3 |
| 1498425548825498112 | empty-comp drop | ASASSN-V J140619.34+422109.5 | SR | 12.537 | 1.727 | 211.58058 | +42.35263 | 3 | 3 TIER1 |
| 1497227287309482624 | empty-comp drop | Gaia DR3 1497227287309482624 | VAR | 13.561 | 1.724 | 207.76571 | +40.28094 | 3 | 3 TIER1 |

VSX seps: 0.06-0.19" for the secure matches; HAT-188-0003359 1.33",
HAT-188-0000323 1.07", Gaia 149722 2.06" (still the nearest VSX
row). CSV: `m1_identity.csv`.

## M2 -- stage-by-stage (live MS spatial + log ranking)

Live MS n_field = **3610**. After sat/var/NSS/min_dist base:
**3300-3302**. Mag filter: no-op (same n). Color cap on that pool:

| catalog_id | n spatial | n T1 0.15 | n T2 0.30 | n T3 0.55 | n cap 0.79 | min \|dBP-RP\| | color-path kill |
|---|---|---|---|---|---|---|---|
| 149884 | 3302 | 1504 | 2738 | 3066 | 3160 | 0.00005 | cap does not empty |
| 149984 | 3302 | 50 | 118 | 302 | 840 | 0.0029 | cap does not empty |
| 150041 | 3301 | 627 | 1363 | 3060 | 3182 | 0.00024 | cap does not empty |
| **R CVn** | **3300** | **0** | **0** | **0** | **0** | **2.219** | **cap empties** |
| 149724 | 3301 | 26 | 50 | 105 | 214 | 0.0012 | cap does not empty |
| 149842 | 3302 | 77 | 167 | 510 | 1356 | 0.0025 | cap does not empty |
| 149722 | 3302 | 81 | 170 | 513 | 1370 | 0.0056 | cap does not empty |

C-EXPORT-GAP color-path ranking (unpinned 149984 / 150041; same
223-pool after MAD):

| id | D3 | sat | MAD n_cand | ceiling 0.08 | iso 3 FWHM | color rung | admitted |
|---|---|---|---|---|---|---|---|
| 149984 | 533->533 | -3 | 223 | 223->141 | 141->95 | 0.300 n=3 (T2) | 3 |
| 150041 | 531->531 | -3 | 223 | 223->141 | 141->95 | 0.150 n=24 (T1) | 8 |

CSS 149884: pinned 8, Phase 1 DeltaBPRP med 0.104 -- **not** empty.
Empty-comp trio: pinned n=3, kill = RMS on shared pin
`1500467303261764096` (G=12.66, BP-RP=1.794), survivors 2.

CSV: `m2_funnel.csv`, `m2_pins.json`, `m2_live_comps.csv`.

## M3 -- what the field offers (all cuts except color cap)

Nearest-in-color among the 3300 spatial survivors. SNR = median
`psf_snr` if present, else `flux / sigma_bkg_ap` from live proc
(134 frames). Negative values = sky-subtracted flux below zero;
those stars are not usable comps. G from `phot_g_mean_mag` else
`mag`.

**R CVn (the rule target). Minimum achievable |dBP-RP| = 2.219.**

| k | id | \|dBP-RP\| | BP-RP | G/mag | SNR |
|---|---|---|---|---|---|
| 3 | 1498145551317166336 | 2.219 | 3.456 | 14.10 | 2.3 |
| 3 | 1499899203644126464 | 2.754 | 2.921 | 14.61 | 1.2 |
| 3 | 1496945331296454144 | 2.758 | 2.917 | 14.87 | -0.7 |
| 5 | +1496036791094488192 | 2.775 | 2.900 | 14.56 | 1.7 |
| 5 | +1498172523712237696 | 2.786 | 2.889 | 14.54 | 0.6 |
| 8 | +1498819758103415680 | 2.872 | 2.803 | 14.51 | 9.2 |
| 8 | +1497245497970801664 | 2.876 | 2.799 | 12.98 | **31.4** |
| 8 | +1486042470915854080 | 2.904 | 2.771 | 14.90 | 1.3 |

Closest usable-SNR star in that eight is G=12.98 at |dBP-RP|=2.876
(SNR 31). The three nearest-in-color are G~14-15 at SNR ~0-2.
Pinned comps actually used: G 8.24-10.29, |dBP-RP| 4.14-5.63
(high SNR; e.g. 1500727513856914944 psf_snr ~271 on frame 001).

Tasked trio (for completeness; cap does not empty):

| target | k=3 nearest \|dBP-RP\| | G/mag | SNR |
|---|---|---|---|
| 149884 CSS | 0.00005 / 0.00011 / 0.00011 | 14.70 / 12.67 / 14.16 | 3.7 / 24.4 / 6.9 |
| 149984 HAT | 0.0029 / 0.0049 / 0.0111 | 13.37 / 14.91 / 14.84 | 8.2 / 0.06 / 1.6 |
| 150041 RS | 0.00024 / 0.00076 / 0.0026 | ~14-15 | 4.3 on the third |

CSV: `m3_nearest_color.csv`.

## M4 -- the three LC CSVs

Live path:
`Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/photometry/lightcurves/`.
C-EXPORT-GAP sandbox CSVs were cleaned; sandbox behaviour is log
only (`lc_rms=nan` for all three; export FAILURE x3).

**No `skip_reason` / `ac_skip_reason` marker in any live file
header or first 8k.** Downstream reader cannot see a classified
skip from the CSV itself.

| id | live rows | live exportable | live flags | skip marker | sandbox (log) |
|---|---|---|---|---|---|
| 149884 | 134 | **132** | 132 normal / 2 no_data | none | lc_rms=nan; export FAIL |
| 149984 | 134 | **71** | 71 normal / 63 no_data | none | lc_rms=nan; export FAIL |
| 150041 | 134 | **0** | 134 no_data; mag/BJD empty | none | lc_rms=nan; export FAIL |
| R CVn | 134 | 134 | all normal | none | lc_rms=0.0124; AAVSO ok |
| empty-comp trio | 134 each | 134 each | all normal | none | **no LC CSV** (`lc_csv=""`); export skip "no LC CSV" |

150041 live is the honest empty-export artifact: 134 rows, flag
`no_data`, empty `bjd`/`mag_calib_final`, `k2_source=none`, no
skip_reason. `_select_export_lc_rows` (`export_reports.py:893`)
returns empty -> batch FAILURE wording
(`:997-1004`).

Live vs sandbox diverge: 149884/149984 are exportable on live,
empty-comp trio have live LCs (older product still has 3 comps)
but sandbox pin-RMS abort wrote no file. Do not treat live 516
hashes as a replay of C-EXPORT-GAP photometry.

## M5 -- k" exposure (no new photometry)

`k2_mode=literature` (`config.json`). Setup `NoFilter_60_2` ->
`band_failsafe_clear` -> literature **NONE**
(`k2_extinction.py:161-162`; `comp_weights.py:107-113`).
Live LC column `k2_source=none`. Term
`|k2| * DeltaX * |dBP-RP|` = **0** on this band.

Airmass from live LCs (same for all four files): min 1.0126,
max 1.2186, **DeltaX = 0.2060**.

Counterfactual only (if this night had been Sloan; NOT the
observed band). Smith 2002 native * Jordi 2010
`SLOPE_GR_PER_BPRP=1.054` (`k2_extinction.py:40,52-62,115-130`):

| band | k2_bprp | term at R CVn min d=2.219 | term at d=2.876 (SNR 31 star) |
|---|---|---|---|
| NoFilter (actual) | NONE | **0** | **0** |
| Sloan g | -0.016864 | 7.7 mmag | 10.0 mmag |
| Sloan r | -0.004216 | 1.9 mmag | 2.5 mmag |

On this night Milan's k" budget for admitting T4 is **0**. The
systematic he must still budget is the existing color-term
extrapolation (Phase 2A warning: target 5.675 vs comps
[0.046, 1.533]), not k". CSV: `m5_k2_term.csv`, `m5_airmass.csv`.

## Gaps (loud)

- Live MS missing D3 columns; D3 counts from log only.
- C-EXPORT-GAP sandbox LC CSVs gone; cannot re-inspect those
  three empty-mag files. Live 150041 is the surviving example.
- SNR for faint red comps is `flux/sigma_bkg_ap`, not a photon
  noise-model column (proc has no `flux_err`; `psf_snr` NaN on
  those rows).
- Why sandbox Phase 2A wrote `lc_rms=nan` for 149884/149984
  despite admitted comps is **not** solved here (separate from
  empty-pool). Live 149884/149984 are exportable.
- Live empty-comp LCs exist; sandbox did not write them. Two
  write policies already coexist across products.

## G4 live 516 (read-only verify)

| product | sha256 prefix | verdict |
|---|---|---|
| masterstars_full_match.csv | bfa24039778f437b... | PASS |
| MASTERSTAR.fits | 13e77cf8a1dcb4e7... | PASS |
| masterstar_epsf.fits | 172f95403beae36d... | PASS |

Expected csv `bfa24039` / fits `13e77cf8` / epsf `172f9540`.
Live draft was not written.

## STOP -- design menu (nothing wired)

Milan picks. Architect drafts decision text after the choice.

**(i) Red-target fallback rule** (Milan's 2026-09-07 direction):
when the cap empties the pool, deterministically admit tier4 by
minimal |dBP-RP| with an SNR floor and a pre-registered
systematics budget (k" term from M5); LC flagged with the
color-mismatch metadata.

Numbers for (i), R CVn only: cap empties (0 within 0.79);
min |dBP-RP| = 2.219 at G=14.10 / SNR~2.3 (not photometrically
useful); first usable-SNR neighbor |dBP-RP|=2.876 / G=12.98 /
SNR~31. On NoFilter, k"=0; color-term extrapolation remains.
Current pin already admits 8 stars at |dBP-RP|~4.1-5.6 with
high SNR. (i) does **not** apply to the three tasked IDs or the
pin-RMS empty-comp trio.

**(ii) Export classification:** reclassify "no exportable points"
caused by a counted upstream `no_comps` drop as a classified skip
(honest wording, 02C precedent), not a batch FAILURE.

Evidence: the three C-EXPORT-GAP FAILURE IDs **had comps**; the
`no_comps` trio was already a "no LC CSV" skip. (ii) as written
fits the empty-comp trio, not the tasked IDs. The tasked IDs are
a different class (LC present, mag empty / `lc_rms=nan`). 150041
live is that class with no skip_reason in the file.

**(iii) Empty LC CSV write policy:** keep writing with an explicit
`skip_reason` marker (artifact honesty) vs skip writing with a
loud log.

Today: empty-comp stub sets `ac_skip_reason=no_comps` and
`lc_csv=""` (no file). Export-empty path writes a 134-row CSV
with `no_data` / empty mag and **no** skip_reason, then marks
batch FAILURE. Milan picks which of those two artifacts is the
rule.

`--fast --clean` OVERALL PASS (1624 passed, 34 skipped; clean-tree
PASS: pytest 32, ruff, pyflakes). db-quick-check WAIVED (malformed
local sqlite; known). Dirty-tree untracked historical session
files not staged.

## Evidence

`dev/results/context/session_20260907_rcvn_m1/`
(`measure_r_cvn_empty_comp_m1.py`, `m1_identity.csv`,
`m2_funnel.csv`, `m2_pins.json`, `m2_live_comps.csv`,
`m3_nearest_color.csv`, `m4_lc_csv.csv`, `m5_k2_term.csv`,
`m5_airmass.csv`, `summary.json`, `probe_ids.py`).

Log reused: `dev/results/context/session_20260907_cexportgap/night_run_photometry.log`.
