# CURSOR RESULT - SAT-LIMIT-01

Date: 2026-08-17
Draft 515 photometry SHA: da9cce4
Push: NOT authorized

Premise: MASTERSTAR catalog saturation tagging compares `peak_max_adu` to
`saturate_limit_adu` / `saturate_limit_adu_85pct`. On draft 515 both limit
columns were NaN, so pandas `peak > limit` was False for every star and
`is_saturated` never fired. SAT-DIAG already knew `sat_adu=65535` DERIVED;
that result was not wired into MASTERSTAR catalog tagging.

## Verdict

INV-SAT-LIMIT closed in code: an unresolved clip no longer silently admits.
Draft 515 catalog reclassified at peak-test = 0.80 x 65535 = 52428 ADU
(WARN-named conservative default). D1-2 remains OPEN (dome-flat ramp).
Production `comparison_stars_per_target.csv` not rewritten; BO ensemble
without C2 is a sidecar + check-MAD measurement for Milan.

## B1 - where NaN enters (3621/3621)

Quantity: catalog rows with non-finite `saturate_limit_adu` and
`saturate_limit_adu_85pct`. Domain: `masterstars_full_match.csv` before
reclassify. SHA da9cce4.

| item | value |
|------|-------|
| n_rows | 3621 |
| saturate_limit_adu NaN | 3621/3621 |
| saturate_limit_adu_85pct NaN | 3621/3621 |
| is_saturated True | 0 |
| likely_saturated True | 5 (plateau path only; limit None) |
| zone | linear 2908, noise 713, saturated 0 |
| C2 `1500748301498613248` | G=7.991, peak_max_adu=64350.48 (98.19% of 65535), zone=linear, is_saturated=False |

Resolver chain (read, not inferred):

1. Header SATURATE/MAXLIN/LINLIMIT/MAXADU: `_saturate_limit_adu_from_header`
   (`pipeline.py`). MASTERSTAR.fits: BITPIX=-32, all those keywords null,
   DATAMAX/MAXPIX null. Data max=68429.12 ADU (float stack).
2. Equipment DB: `EQUIPMENTS.SATURATE_ADU` for ID=1 (QHY294MM / Camera1) is
   NULL. `database.py` `_migrate_qhy294mm_saturate_adu_null` (line 2825)
   wiped the wrong 16384; `get_equipment_saturation_adu` (line 3149) returns
   None. File: `vyvar.sqlite3`.
3. BITPIX guess: `_infer_sat_limit_from_bitpix` returns None for BITPIX -32.
4. `_effective_saturation_limit` previously returned `(None, "none")`.
5. `_annotate_masterstars_flux_zones` previously set `is_saturated=False`
   when `peak_sat_lim is None`. Pandas `peak > NaN` is False.
6. SAT-DIAG (`sat_diag.json`): sat_adu=65535 DERIVED, lin_adu=55704.75
   DEFAULT_FRAC, header_value=null, equipment_value=null. Ran later, on the
   per-frame catalog path, not during MASTERSTAR annotate.
7. Second hole: `_resolve_peak_saturation_limit_adu` vetoed the camera clip
   when `frame_max > cam_lim * 1.02`. 515 stack max 68429 vs frac-scaled
   55705 tripped that veto even if 65535 had been supplied. Interpolation
   overshoot is not a unit change.

`likely_saturated` True on 5 other stars (peaks 65346-67462) via the plateau
path that runs when limit is None. C2 (64350) did not plateau-flag.

## B2 - fire proof

Historical hole (still true in pandas): `pd.Series([65000.0]) > float("nan")`
is False. Test: `test_pandas_nan_limit_is_silently_false`.

After fix: `equipment_saturate_adu=None`, `peak_max_adu=65000` ->
`is_saturated=True`, zone=saturated, finite limits. Tests:
`test_inv_sat_limit_unresolved_clip_does_not_silently_admit`,
`test_inv_sat_limit_nan_equipment_value_does_not_silently_admit`,
`test_effective_saturation_limit_never_none`,
`test_masterstar_stack_few_percent_overshoot_still_flags_peak`.
On-disk C2 row with unresolved clip: is_saturated True, peak-test 52428 ADU.

INV-SAT-LIMIT: missing clip is a WARN-named conservative default
(65535 container, 0.80 peak-test), not a hard abort and not silent admit.
Precalibrated skip kept when a camera clip is supplied and frame_max >
1.20 x raw clip (98232 vs 65535 still skips).

## B3 - knee

Cheap D1-2 check: instrumental minus Gaia G residual vs `peak_max_adu` on
2644 currently-linear stars; ZP from G 10-13, peak 2k-20k (n=746).

Auto-detector flagged 25000 ADU because the 25-30k bin (n=8) median residual
was -76 mmag vs 3 x ref-bin std. The next bin reversed (-40 mmag). That is
not a resolved linearity knee. The 60-65k bin (n=9) is +213 mmag - flux
deficit at the container clip, i.e. saturation, not a sub-clip knee.

Choice (WARN, not silent): peak-test = 0.80 x 65535 = 52428 ADU.
JSON semantics (SAT-RERANK-01 A3): `knee.knee_resolved` is the physics verdict
(false); `auto_detector_resolved` is the detector flag (true).
Source string: `conservative_default_0.80x_container_clip_65535`.
D1-2 remains OPEN (dome-flat ramp per sensor).

## B4 - reclassify 515 and BO without C2

Peak-test 52428 ADU on existing `peak_max_adu` (container domain).

| quantity | before | after |
|----------|-------:|------:|
| is_saturated | 0 | 24 |
| zone -> saturated | 0 | 24 |
| C2 zone | linear | saturated |

20 production ensembles contain at least one newly saturated member.
Two catalog IDs appear: C2 `1500748301498613248` (BO and 12 other targets)
and `1498814260545232384` (7 targets). BO known case confirmed.

BO ensemble (production CSV unchanged; sidecar only):

- old (n=5): C2 + `1497771992240531712`, `1499200223486564608`,
  `1497974027502858240`, `1497368849430107904`
- new (n=4): same minus C2. Still >= n_comp_min=3. No RMS re-rank, no fill.

Fixed-meter check MAD (product mag_calib / pytics ZP; check
`1498020894186918144`; 134 epochs; same formula as 01B):

| ensemble | check_scatter_mad_mmag |
|----------|----------------------:|
| old (with C2) | 7.050 |
| new (without C2) | 8.580 |

Old cell matches D515-ACCEPT-01B 515 BO 7.0498 mmag exactly. Dropping C2
without a replacement **raises** check MAD by 1.53 mmag on this meter.
Architect flux-sum LOO (10.33 -> 7.82 mmag) is a different frame (unweighted
target LOO); pytics already down-weighted C2 (weight 6564 vs 55043). SAT gate
must still exclude C2 (98% of clip). A quieter BO ensemble needs a Phase 1
re-rank replacement, not a 4-star leftover. Milan decision.

D515-ACCEPT-01B 515 BO cell is superseded for ensemble-membership questions;
pointer this file. The 01B 2x2 vs 514 same-meter table is not overwritten.

Other drafts not touched. Production `comparison_stars_per_target.csv`
unchanged. Catalog CSV rewritten with backup
`masterstars_full_match_before_sat_limit_01.csv`.

## B5 - register cross-links

SAT-LIMIT-01 CLOSED (code + 515 catalog). BIN-8-9-REGRESSION-01 stays OPEN:
bright-end LOO excess is now plausibly partly nonlinearity/saturation (24
stars above 0.80 clip, including C2); not closed by this gate alone.
D1-2 stays OPEN/DEFERRED: cheap residual-vs-peak did not resolve a knee;
dome-flat ramp still required.

## Spec defects / physics outranks

1. AIJ Table.tbl missing from the supplied folder (Part A).
2. Auto-detector "knee" at 25000 ADU rejected; 0.80 x clip adopted as specified
   when the knee is not resolved.
3. Check MAD without C2 got worse on the product meter. Spec assumed
   improvement from the flux-sum LOO; meters differ. Reported, not forced.

## Files

- `dev/results/SAT_LIMIT_01_summary.json`
- `dev/tests/test_masterstar_zone_classifier.py`
- `src_py/pipeline.py` (INV-SAT-LIMIT)
- `dev/tools/sat_limit_01_measure.py`, `dev/tools/sat_limit_01_b4.py`
- 515 catalog backup + BO sidecar under Archive/Drafts/draft_000515/

## Errors

None blocking.

## Verify

`session_baseline_check.py --fast`: OVERALL PASS (1434 passed, 28 skipped).
HEAD still a217e1d (uncommitted). Push NOT authorized.
