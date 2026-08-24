CURSOR RESULT - 2026-08-24T15:40:00Z (EPSF-AC-01)

What I did
Measured the F6 aperture-correction policy on production ePSF catalogs
(AC inverted: psf_raw = psf_flux / psf_ac_factor). No production AC wiring
(no Milan GO). Chi2 and flux reported separately. Not pushed; Milan authorizes.

Premise (0.1): SHAPE-01-F compared production iterative PSF/DAO = 0.671 (AC on)
with sandbox AC-off PSF/DAO = 1.218 on the same ePSF. Those numbers differ by
the F6 scalar `psf_ac_factor` (~0.528 on frame 001). This task asks whether
ANY scalar AC can be honest, by measuring uncorrected PSF/DAO vs magnitude on
the full science set (333), not the bright-30 subset.

## Gates

| Gate | Status | Detail |
|------|--------|--------|
| G0 commit hygiene | PASS | F3 code `6fd1452`; SHAPE-01-F results + H5 superseded pointer `cf95c53`. F-work started from that clean tip (unrelated DAO/VALID dirt left unstaged). |
| G1 tip | PASS | `cf95c5309bcc71834b814241671b021dd87872f4` is a descendant of `1f9f921`. |
| G2 `--fast` | PASS | 1520 passed, 32 skipped, OVERALL PASS on G0 tip `cf95c53`. `test_exoplanet_local_match.py::test_comp` did **not** appear (prior G0-era flake does not recur here; no investigation item). git-origin-main WARN is local commits ahead of origin (do not pull/push). |
| G3 era | PASS | checker constants unchanged: core `9902d918` n=121 / ext `472bc9e4` n=179 |
| G4 production ePSF | PASS | SHA256 `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` unchanged |

Hash guard (aperture LCs, AAVSO, VarAstro, production ePSF+meta):
`hashes_before.json` == `hashes_after.json`. Positive control changed.
Run time (A1-A3 catalog path): 4.86 s.

## A1 - uncorrected PSF/DAO vs magnitude

Production ePSF, 20-frame subset, science set n=333, fit_ok rows only
(n=4224). Chi2 recorded, not used as a row filter. Flux finding: PSF/DAO
from inverted AC. Chi2 finding: not mixed into the ratio.

| M1 bin (G mag) | n stars | median PSF/DAO | RMS | RMS ~= |median| |
|----------------|---------|----------------|-----|----------------|
| 5.94-10.55 | 40 | 1.266 | 1.274 | yes (offset) |
| 10.55-12.39 | 50 | 1.564 | 1.558 | yes |
| 12.39-13.25 | 51 | 1.695 | 1.720 | yes |
| 13.25-14.06 | 54 | 2.158 | 2.426 | yes |
| 14.06-15.56 | 74 | 2.207 | 2.233 | yes |

corr(night-median ratio, mag) = 0.694. Slope in the BO CVn pinned-comp span
= 0.128 per mag.

Bright-30 inverted median = 1.261 vs F2 sandbox re-fit 1.218 (delta +0.043).
Catalog inversion is the production grouped path; close enough to trust A1.

Pinned ensemble (BO CVn, n=4, source=pinned): G 9.68-11.52. Target G=9.72
ratio 1.357. Comp ratios 1.357 / 1.417 / 1.466 / 1.533. 44 science-set stars
in that G span: median ratio 1.357, RMS~=|median|, p16-p84 = 173 mmag
equivalent, minmax = 352 mmag.

Verdict: uncorrected PSF/DAO is NOT flat in mag, including inside the
comp-ensemble span that matters for differential work. Tolerance: not within
a few mmag; p16-p84 already 173 mmag among science-set stars in 9.68-11.52,
and 136 mmag between the brightest and faintest of the four pinned comps.
A scalar AC cannot be honest on this model.

Rig tag: mag slope is **rig-amplified** by the too-narrow ePSF (SHAPE-01
core OPEN). The statement "scalar AC requires a flat ratio" is **generic**.

Artifacts: `a1_summary.json`, `a1_ratio_vs_mag_bins.csv`,
`a1_star_night_median.csv`, `a1_star_frame.csv`.

## A2 - AC ensemble census (brightness-cut evidence)

chi2<5 gate, 20 frames, fit_ok science-set stars:

| meter | night median |
|-------|----------------|
| n admitted | 167 |
| frac of fit_ok | 0.774 |
| admitted mag | 13.35 |
| science-set mag | 12.90 |
| frac of brightest-30 admitted | **0.0** |
| brightest-30 ever admitted (any frame) | **0 / 30** |

Induced bias (AC = median DAO/PSF on the named set), night medians:

| ensemble | AC factor |
|----------|-----------|
| chi2<5 (production P0) | 0.500 |
| all fit_ok (P3) | 0.567 |
| M1 bin0 .. bin4 | 0.782 / 0.638 / 0.594 / 0.469 / 0.457 |

Production frame-001 factor 0.528 sits on the chi2<5 column. Bright stars
(chi2 ~20-68) never enter the ensemble, then receive that global factor:
1.26 * 0.50 ~ 0.63, matching the 0.67 droop direction.

Chi2 finding: the gate uses chi2; the membership outcome is a magnitude cut.
Flux finding: the factor trained on faint members over-corrects bright flux.

Rig tag: "cut on a statistic that correlates with brightness" is **generic**
(same class as FD-A sky-only weights). Empty bright-30 membership is
**rig-amplified** by bright chi2 ~68.

Artifacts: `a2_summary.json`, `a2_ensemble_census.csv`, `a2_ac_bias.csv`.

## A3 - candidate policies (sandbox)

P1 K=20 (lowest 20 percent chi2 per M1 mag bin, then one robust median).
P2 mag-binned AC, monotone in the data direction, **constant hold of the
end-bin factor** at both bright and faint ends (no linear extrapolation).
P3 all fit_ok. P4 factor=1. P0 = current chi2<5 (baseline).

| policy | span median ratio | span p16-p84 mmag | BO coverage | BO dmag-ap RMS mmag |
|--------|-------------------|-------------------|-------------|---------------------|
| P0 | 0.680 | 175 | 0.993 | 614.3 |
| P1 | 0.760 | 182 | 0.993 | 614.3 |
| P2 | 0.913 | 171 | 0.993 | 667.7 |
| P3 | 0.764 | 172 | 0.993 | 614.3 |
| P4 | 1.357 | 173 | 0.993 | 614.3 |

Flux: P0/P1/P3/P4 `psf_delta_mag` are identical (scalar ZP cancel). P2 is
the only policy that changes delta_mag (target vs comps get different
factors) and it got slightly worse vs aperture.

P4 invariance: P0 vs P4 max |d(delta_mag)| = 5.3e-15 (n=133);
identical_to_1e12 = true. Reconstruction vs production internal PSF LC
RMS 2.9e-7 (float noise from invert). Header/ratio columns may differ;
delta_mag must not, and does not.

P4 residual mag-slope bias (target G minus ensemble-mean G, times A1
slope): 76 mmag. That is the differential term a scalar-free LC still
leaves when A1 is not flat. It is NOT the 614 mmag RMS: that RMS is the
existing production PSF LC vs aperture, dominated by ensemble fit_ok
drops (e.g. frame 004: PSF delta 0.385 vs aperture 0.866), unchanged by
any scalar AC.

Chi2: unused as a photometry filter in A3 application; P0/P1 only use it
for ensemble membership.

Artifacts: `a3_policy_table.csv`, `a3_P{0,1,2,3,4}_*.csv`,
`a3_p4_invariance.json`.

## A4 - recommendation + STOP

Rank (this rig, this model):

| | honesty | delta_mag vs aperture | abs scale (ratio flat) | simplicity | Newton |
|--|---------|----------------------|------------------------|------------|--------|
| P0 | fail (hidden brightness cut) | same as P4 (614 mmag existing) | worst (span median 0.68) | simple | becomes less of a cut if bright chi2~1 |
| P1 | better membership, still a scalar | same as P4 | still sloped (0.76) | medium | generic percentile idea |
| P2 | no hidden cut | slightly worse | best scalar-family (0.91) but not flat | complex | still useful if A1 sloped |
| P3 | no chi2 brightness cut | same as P4 | still sloped (0.76) | simple | least-wrong scalar |
| P4 | honest for relative product | same as P0 (invariance) | uncorrected 1.36 | simplest | generic ZP cancel |

Recommend:
(a) Production F6 merge: **do not wire a change in this task**. Keep chi2<5
as an explicit named fallback until Milan GO. Preferred GO options: AC-off
(P4; stamp uncorrected; absolute scale stays untrusted under SHAPE-01) or
P2 if a nearer-to-1 PSF/DAO per mag bin is wanted. P3 is the least-wrong
scalar and still cannot flatten A1.
(b) Internal PSF LC: **P4** (no AC). Charter is relative-only; invariance
is measured. Mag-slope residual ~76 mmag on BO CVn is the honest leftover
until EPSF-CORE-01.

STOP for Milan. No config wiring, no provenance stamp, no BO CVn
acceptance-meter recut, no anchor-era recut.

## What changes on a well-sampled rig (Newton 0.65 arcsec/px)

If seeing ~3 arcsec, Newton FWHM ~4.6 px. Bright chi2 should sit nearer 1,
so chi2<5 would admit bright stars and P0 would stop being a disguised
brightness cut. A1 must still be measured: if PSF/DAO is flat, a scalar
AC is legal; if not, P4 remains the internal-LC policy and P2 remains the
only absolute-scale candidate. Do not copy this wide-rig P0 factor (0.50)
onto Newton.

## Files touched

Production code: none (measurement + docs only; no AC wiring).

Sandbox:
- `dev/scripts/epsf_ac_01.py`
- `dev/results/session_20260824_epsf_ac_01/**`
- `dev/results/context/session_20260824_epsf_ac_01/`

Docs:
- `docs/VYVAR_DECISIONS.md` (AC chi2<5 defect + recommendation)
- `docs/VYVAR_STATE.md` (SHAPE-01 core OPEN; AC-01 policy STOP)
- `docs/VYVAR_ROADMAP.md` (EPSF-AC-01 + EPSF-CORE-01)
- `dev/results/CURSOR_RESULT_EPSF_SHAPE_01_M.md` (H5 superseded; in G0 `cf95c53`)

G0 (already committed):
- `6fd1452` SHAPE-01-F F3 persistence
- `cf95c53` SHAPE-01-F results + H5 pointer

Not pushed.

`--fast` OVERALL PASS: 1520 passed, 32 skipped. `test_comp` absent from the fail list.
