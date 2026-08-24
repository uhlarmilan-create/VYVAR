CURSOR RESULT - 2026-08-24T18:00:00Z (EPSF-PIN-CENSUS-01)

What I did
Measured why pinned (and resolved) comparison stars fail `psf_fit_ok` on
draft 516, 134 epochs, 60 aperture-LC targets. Read-only: no re-fit, no
proc/LC rewrite. Chi2 and flux reported separately. Not pushed.

Premise (0.1): AC-02 left BO CVn at 23/134 full-membership epochs and
FW CVn at 0/134 under INV-PSF-LC-PIN-01. Those meters used stored
`psf_fit_ok` (converged AND chi2<50, then possibly cleared by quality
fallback). This census classifies every non-ok star-epoch from stored
columns and recomputes the AC-02 meters under admission variants. The
hypothesis under test is that the chi2<50 half of `psf_fit_ok` is a
third brightness-correlated cut (after FD-A sky-only weights and AC
chi2<5). AC-02 numbers being compared: BO 23/134, RMS 38.8 mmag, level
offset +40.5 mmag on the same 134-epoch aperture LC. Positive control
matched (n_full 23, RMS delta -0.000085 mmag).

## Gates

| Gate | Status | Detail |
|------|--------|--------|
| G1 tip | PASS | `876053ab92f42ac2de1a3795ab2ac1e0c81a2de5` is a descendant of itself. Declared-dirty: AC-02 wiring uncommitted (ancestor of this tree). Unrelated DAO/VALID dirt left unstaged. |
| G2 era + ePSF | PASS | Production ePSF SHA256 `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` unchanged (hash guard before=after). Era constants in `session_baseline_check.py` unchanged: core `9902d918` n=121 / ext `472bc9e4` n=179. |
| Hash guard | PASS | Aperture LCs, AAVSO, VarAstro, production ePSF FITS+meta byte-identical. |
| `--fast` | see close | Run at task end. |

Harness elapsed 30.4 s (C2 0.22 s). Rule 0.3: measurement-only; no
science-path timing change.

A row with a finite stored `psf_chi2` implies the PSF fit completed.
That is usable evidence. Stated once.

## How `psf_fit_ok` becomes false (file:line)

Single-star completed fit (`src_py/psf_photometry.py`):

```
3324:            converged = (flags & 8) == 0
3344:            chi2_ok = math.isfinite(chi2) and chi2 < _chi2_limit
3345:            fit_ok = bool(converged and chi2_ok)
3356:                    "psf_fit_ok": fit_ok,
```

Default `_chi2_limit` is 50 (`psf_chi2_threshold`). Photutils `flags`
bit 8 is not written to any proc column (`PROC_STORE_COLS` in
`src_py/proc_frame_store.py` 31-98 has `psf_fit_ok`, `psf_chi2`,
`psf_quality`, `psf_quality_fallback`; no flags field). Convergence is
therefore not recoverable from stored data.

Quality fallback can clear `psf_fit_ok` after that test:

```
3418:        _q = assess_psf_quality(...)
3427:        r["psf_quality"] = _q
3430:        if _q == "bad" and _quality_fallback_on:
3431:            r["psf_quality_fallback"] = True
3432:            r["psf_fit_ok"] = False
```

`assess_psf_quality` (`2791-2838`) marks `bad` for chi2 >= `chi2_bad`
(50), SNR < 5, position shift >= 1 FWHM, close/contaminating neighbour,
or non-finite chi2/shift (`2813-2835`). `psf_quality_fallback` is
persisted (`proc_frame_store.py:88`).

Other paths that stamp `psf_fit_ok` false without a completed-fit
signature (NaN flux and NaN chi2):

- invalid x/y: `3114`
- per-star default before fit: `3136`
- edge cutout skip: `3188-3189`
- bad cutout shape: `3198-3199`
- fit exception: `3370-3371` (appends the NaN/`False` base)
- pipeline missing-column default: `src_py/pipeline.py:522-523` (also
  `631-632`, `641-642`)
- Phase 2A row with no CSV hit: `src_py/photometry_core.py:3396`

Grouped helper (`2760-2768`): flux<=0 returns None (caller falls through
to single-star, `3166-3173`). If the joint fit returns a row:

```
2762:    converged = (flags & 8) == 0
2763:    chi2_ok = (not math.isfinite(chi2)) or (chi2 < float(chi2_limit))
2768:        "psf_fit_ok": bool(converged and chi2_ok),
```

NaN chi2 **passes** grouped `chi2_ok` (opposite of single-star
`3344`). The later quality loop still runs on grouped rows and can
clear `fit_ok` when chi2 is non-finite (`2813` + `3430-3432`). That
path leaves `nonfinite_chi2` in the census taxonomy if flux is finite.

Persistence gap (not a census failure): a false `fit_ok` with finite
flux and finite chi2 < 50 cannot be split into non-convergence vs
SNR/shift/neighbour fallback when `psf_quality_fallback` is also false,
because the flags bit is not stored. Parent class name for that cell is
`inferred_non_converged_or_quality_fallback`. On this draft that class
is empty (0 of 22220 fails).

## C1 - exclusive stored-column classes

Table header: the last class is **inferred**. Convergence flag bits are
not persisted. Classification from stored columns:

| class | definition |
|-------|------------|
| `missing_row` | no catalog row for that (comp, epoch) |
| `nonfinite_psf_flux` | row present, flux not finite or <= 0 |
| `chi2_ge_50` | stored `psf_chi2` finite and >= 50 |
| `nonfinite_chi2` | finite flux, chi2 not finite (reported separately; not inferred) |
| `inferred_non_converged_or_quality_fallback` | `fit_ok` false AND finite flux>0 AND finite chi2 < 50 |
| `fit_ok` | stored `psf_fit_ok` true |

60 targets, 67 unique comps, 56548 membership-weighted star-epochs
(a comp used by N targets is counted N times). Unique (comp, epoch)
cells: 67 x 134 = 8978.

| cause | star-epochs | fraction of fails |
|-------|-------------|-------------------|
| `chi2_ge_50` | 22220 | **1.000** |
| `inferred_non_converged_or_quality_fallback` | 0 | 0 |
| `nonfinite_psf_flux` | 0 | 0 |
| `missing_row` | 0 | 0 |
| `nonfinite_chi2` | 0 | 0 |
| `fit_ok` | 34328 | (not a fail) |

Target-epoch pin drops (first missing pinned/resolved comp): **7453**,
all `chi2_ge_50`. Fraction of pin drops that are the chi2 gate: **1.0**.
Fraction that are genuine non-finite / missing / inferred: **0.0**.

Chi2 histogram of the 22220 `chi2_ge_50` star-epochs:

| bin | n |
|-----|---|
| 50 <= chi2 < 100 | 13963 |
| 100 <= chi2 < 200 | 4330 |
| chi2 >= 200 | 3927 |
| median / p16 / p84 | 78.2 / 56.4 / 212.5 |

Diagnostic sub-split (parent class stays `chi2_ge_50`): callout series
show `psf_quality_fallback` true on 107/107 BO-comp chi2 fails and
130/130 FW-comp chi2 fails. Expected: chi2 >= 50 already grades `bad`.

Brightness: mean fail fraction vs G (unique comps, 134 epochs):
G<9 n=4 mean fail 0.993; 9-9.5 n=10 mean 0.710; 9.5-10 n=13 mean 0.309;
10-10.5 n=9 mean 0.020; 10.5-11 n=9 mean 0.001. Three faint outliers
also fail hard (G=11.36, 12.23, 12.79). The chi2<50 cut is strongly
brightness-correlated on the bright end; it is not exclusively a bright
star cut.

### Per-comp fails (30 of 67; all fails = `chi2_ge_50`)

Full table: `c1_per_comp_cause.csv`. Failing comps by G mag:

| catalog_id | G | n_ok | n_fail | median chi2 of fails |
|------------|---|------|--------|----------------------|
| 1500296402219939584 | 8.243 | 0 | 134 | 376.6 |
| 1497613731286514432 | 8.450 | 0 | 134 | 196.3 |
| 1499906247391001088 | 8.743 | 0 | 134 | 227.7 |
| 1497442379271632384 | 8.851 | 4 | 130 | 97.3 |
| 1498735778606786816 | 9.113 | 49 | 85 | 58.6 |
| 1497119157212720896 | 9.131 | 24 | 110 | 82.0 |
| 1497528072458898432 | 9.218 | 24 | 110 | 66.5 |
| 1500727513856914944 | 9.237 | 0 | 134 | 115.9 |
| 1497674651102612992 | 9.287 | 13 | 121 | 79.0 |
| 1497203132413443328 | 9.307 | 35 | 99 | 62.4 |
| 1497370563121917952 | 9.347 | 110 | 24 | 54.2 |
| 1498326455340079616 | 9.407 | 4 | 130 | 72.0 |
| 1498062332030906880 | 9.451 | 0 | 134 | 76.5 |
| 1497837207025312768 | 9.453 | 129 | 5 | 60.8 |
| 1500460813567859456 | 9.510 | 63 | 71 | 56.7 |
| 1496798993170465536 | 9.521 | 10 | 124 | 65.2 |
| 1496948904709272576 | 9.548 | 130 | 4 | 59.1 |
| 1497617407778562304 | 9.570 | 121 | 13 | 54.7 |
| 1497719902878053888 | 9.598 | 94 | 40 | 58.6 |
| 1501956561697995008 | 9.621 | 78 | 56 | 54.3 |
| 1499200223486564608 | 9.679 | 27 | 107 | 62.0 |
| 1497771992240531712 | 9.752 | 118 | 16 | 59.8 |
| 1496984467038399744 | 9.787 | 133 | 1 | 55.8 |
| 1500664876053883904 | 9.866 | 27 | 107 | 56.5 |
| 1498020894186918144 | 10.285 | 110 | 24 | 60.5 |
| 1500355466608253184 | 10.501 | 133 | 1 | 79.4 |
| 1500576537166572544 | 11.356 | 0 | 134 | 235.9 |
| 1496786967262035712 | 12.230 | 6 | 128 | 72.3 |
| 1500602959805382656 | 12.793 | 0 | 134 | 158.7 |
| 1500579870061241088 | 13.252 | 133 | 1 | 57.2 |

37 comps are 134/134 `fit_ok`.

### Callouts

BO CVn pin set (4 comps): `1497771992240531712`,
`1499200223486564608`, `1497974027502858240` (G=11.17, 134/134 ok),
`1497368849430107904` (G=11.52, 134/134 ok).

**BO dropper `1499200223486564608`**, G = 9.679:
27 `fit_ok`, 107 `chi2_ge_50`. Fail chi2 min/p16/med/p84/max =
50.1 / 53.1 / 62.0 / 72.2 / 90.5. All fails are below 100. Per-epoch
series: `c1_bo_comp_1499200223486564608_chi2_series.csv`. Other BO
dropper `1497771992240531712` G=9.752: 16 fails, median chi2 59.8.

**FW dropper `1497442379271632384`**, G = 8.851:
4 `fit_ok`, 130 `chi2_ge_50`. Fail chi2 min/p16/med/p84/max =
53.2 / 75.7 / 97.3 / 122.3 / 188.9. Series:
`c1_fw_comp_1497442379271632384_chi2_series.csv`. Other FW dropper
`1499906247391001088` G=8.743: 0/134 ok, median chi2 227.7.

## C2 - AC-02 meters under admission variants

Predicate replicates PIN membership (`psf_internal_lc.py:396`:
`psf_fit_ok AND finite psf_flux > 0`) and then relaxes it. No LC
rewrite. RMS is PSF-minus-aperture `delta_mag` on full-membership
epochs only.

Variants:

- `chi2_lt50_current` - stored `fit_ok` (positive control)
- `chi2_lt100` / `chi2_lt200` - raise the gate only; admit
  50 <= chi2 < T; do **not** admit the inferred class
- `admit_chi2_ge50` - admit rows with finite flux and finite chi2 >= 50
- `admit_inferred_fallback` - admit the inferred class only
- `conv_finite_both` - `fit_ok OR (finite flux AND finite chi2)`
  (union of the two admissions)

### BO CVn `1498613634033133184`

| variant | n_full | coverage | RMS mmag | offset mmag | demeaned mmag | RMS ~= \|median\| |
|---------|--------|----------|----------|-------------|---------------|-------------------|
| chi2_lt50_current | 23 | 0.172 | 38.8 | +40.5 | 9.9 | yes |
| admit_inferred_fallback | 23 | 0.172 | 38.8 | +40.5 | 9.9 | yes |
| chi2_lt100 | 134 | 1.000 | 37.3 | +36.9 | 8.5 | yes |
| chi2_lt200 | 134 | 1.000 | 37.3 | +36.9 | 8.5 | yes |
| admit_chi2_ge50 | 134 | 1.000 | 37.3 | +36.9 | 8.5 | yes |
| conv_finite_both | 134 | 1.000 | 37.3 | +36.9 | 8.5 | yes |

Admitting chi2>=50 **holds** (does not degrade) vs aperture: RMS 38.8 ->
37.3 mmag, offset 40.5 -> 36.9 mmag, demeaned 9.9 -> 8.5 mmag, coverage
23/134 -> 134/134. `chi2_lt100` already buys full BO coverage because
every BO pin-drop chi2 is in [50, 100). Admitting inferred buys **zero**
epochs (class empty).

### FW CVn `1497343732462852864`

| variant | n_full | coverage | RMS mmag | offset mmag | demeaned mmag | RMS ~= \|median\| |
|---------|--------|----------|----------|-------------|---------------|-------------------|
| chi2_lt50_current | 0 | 0.000 | n/a | n/a | n/a | n/a |
| admit_inferred_fallback | 0 | 0.000 | n/a | n/a | n/a | n/a |
| chi2_lt100 | 0 | 0.000 | n/a | n/a | n/a | n/a |
| chi2_lt200 | 25 | 0.187 | 49.0 | +48.2 | 4.8 | yes |
| admit_chi2_ge50 | 134 | 1.000 | 48.5 | +48.6 | 5.2 | yes |
| conv_finite_both | 134 | 1.000 | 48.5 | +48.6 | 5.2 | yes |

Admitting chi2>=50 is what buys FW coverage (0 -> 134). Raising the
gate to 100 buys nothing (blocking comps sit above 100). Raising to 200
buys 25/134. Admitting inferred buys nothing. On the 134 admitted
epochs, leftover is a ~48.6 mmag level offset (RMS ~= |median|); demeaned
scatter 5.2 mmag. Quality **holds** in the same sense as BO: the series
is an offset, not jump scatter.

`conv_finite_both` equals `admit_chi2_ge50` on this draft because the
inferred class is empty.

## C3 - STOP for Milan

Quality of PSF-vs-aperture `delta_mag` **holds** when chi2-flagged
bright comps are admitted. The chi2<50 half of `psf_fit_ok` is the
entire pin-drop cause on draft 516 (100% of 7453 pin drops; 100% of
22220 star-epoch fails). It is a completed-fit quality gate, not a
missing-row / non-finite-flux / measured-convergence failure.

Proposal (interim, not wired): introduce `psf_fit_ok_for_zp` =
convergence + finite flux + finite chi2, with chi2 still recorded.
From stored columns that is `fit_ok OR (finite flux AND finite chi2)`.
Use it as INV-PSF-LC-PIN-01 membership. Keep the named default
(`psf_chi2_threshold=50` inside `psf_fit_ok`) until Milan GO.

Do not wait on a different numeric threshold for this coverage hole:
`chi2_lt100` fixes BO and leaves FW at 0/134; only admitting chi2>=50
fixes both.

If Milan does not GO the interim, leave PIN membership on current
`psf_fit_ok` and wait for EPSF-CORE-01 to move bright-star chi2.

Either way, EPSF-CORE-01 acceptance gains a coverage metric: pinned-comp
`psf_fit_ok` fraction on BO CVn and FW CVn (current: 23/134 and 0/134
full-membership epochs; per-comp BO `1499200223486564608` 27/134, FW
`1497442379271632384` 4/134 and `1499906247391001088` 0/134). Rig tag:
WIDE rig draft 516 (Carl-Zeiss 200 mm / QHY294MM, NoFilter_60_2).

Persistence follow-up (recommendation, not this task): store photutils
flags bits and a fallback-reason enum through the F6 merge, F3-style
additive columns under INV-PSF-ADDITIVE-01, so the inferred class can
be measured rather than named.

## Files touched

This task (measurement + docs):
`dev/scripts/epsf_pin_census_01.py`,
`dev/results/CURSOR_RESULT_EPSF_PIN_CENSUS_01.md`,
`docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_JOURNAL.md`.

No production code, no proc CSV, no LC rewrite.

Artifacts: `dev/results/session_20260824_epsf_pin_census_01/` and
context copy `dev/results/context/session_20260824_epsf_pin_census_01/`
(CSV/JSON; `BLOB_SHA_MANIFEST.json`). Rule 0.2: commit/push of the raw
numbers waits on Milan (this task is not pushed).

Unrelated dirty files (DAO CURSOR_RESULT/TASK, VALID_02 md, d10_1
context, AC-02 uncommitted wiring) left unstaged.

## Errors (if any)

None on the science path.

## `--fast` close

PASS: 1524 passed, 32 skipped, OVERALL PASS on HEAD `876053a` (declared-dirty
AC-02 tree + this census/docs). Watch `test_comp` did not appear as a
failure (same 1524/32 as AC-02). git-origin-main WARN: local series ahead
of origin (do not pull/push). db-quick-check WARN via committed waiver.
Elapsed 592 s.
