DEPS-SCOUT  dependency landscape vs VYVAR (2026-07-18)

Author: Claude (web-research scout). Grounding: photutils changelog (readthedocs, fetched
2026-07-18), astropy 8.0 changelog + PyPI, NumPy release info; VYVAR usage mapped by grep on
origin/main d437bcd. Cursor verified the repo-grounded usage map read-only (see "Cursor
verification" at the end).

## Current state

| Package   | VYVAR pin        | Installed | Latest stable        | Gap        |
|-----------|------------------|-----------|----------------------|------------|
| numpy     | >=2.4,<3         | 2.4.3     | 2.4.4 (2026-03-29)   | patch      |
| astropy   | >=7.2,<8         | 7.2.0     | 8.0.1 (2026-07-05)   | one major  |
| photutils | >=2.3,<3         | 2.3.0     | 3.0.0 (2026-04-17)   | one major  |

Pins correctly hold majors; a fresh install today reproduces our majors. photutils 3.1 is in
development (unreleased) and is on the watchlist below.

## photutils 2.3 -> 3.0 (the one that matters)

VYVAR usage map: 13 DAOStarFinder import sites (QC, alignment, masterstar, SIPS preset);
aperture_photometry + Circular{Aperture,Annulus} (photometry core); detect_sources/SourceCatalog
(2 sites); centroid_sources only in the xval harness; PSF module already on the MODERN API
(ImagePSF, PSFPhotometry, SourceGrouper). Background2D: NOT used (sky-surface is in-house) - the
3.0 Background2D fixes do not affect us.

### Behavior changes that would MOVE OUR NUMBERS (upgrade blockers to handle)

1. Star-finder min_separation default change (#2216): in 3.0 the default is None => an implicit
   2.5 x FWHM minimum separation is APPLIED. In 2.x no such filtering happened by default.
   Upgrading without action changes detection results (close pairs merged/dropped) -> census
   counts, masterstar, comp pools shift -> anchor breaks BY DESIGN, not by bug. Mitigation: pass
   min_separation=0 explicitly at every DAOStarFinder site to freeze 2.x behavior; adopting the
   new default later would be a deliberate science decision (effectively a built-in crowding
   filter - relevant to our dense-field program, but only after validation).
2. Deprecated kwargs we USE (12 hits, e.g. utils.py): sharplo/sharphi -> sharpness_range,
   roundlo/roundhi -> roundness_range, peakmax -> peak_max, brightest -> n_brightest. Still work
   in 3.x with warnings; removed in 4.0. Mechanical migration.
3. Keyword-only deprecation (#2219): optional args passed positionally now warn. One audit pass.
4. Column renames (xcentroid -> x_centroid etc.) deprecated-not-removed; our readers keep working
   through 3.x; migrate once during the upgrade.

### Gains for us in 3.0

- DAOStarFinder significantly faster (vectorized cutouts/moments, #2201): we call DAO everywhere
  - pipeline-wide runtime win.
- 2D-array threshold + scale_threshold (#2202): spatially varying detection threshold - a future
  option for vignetted wide-field frames.
- PSF robustness fixes (non-finite local-bkg flags, #2131): improves the PSF program we plan to
  enable; our module needs at most one check (EPSFBuilder usage) since removed classes
  (FittableImageModel, EPSFModel) are not imported by us.
- Independent confidence: our own cross-validation (draft_310) ALREADY ran against photutils
  3.0.0 with Delta < 0.001 mag agreement.

## photutils 3.1 (unreleased)  watchlist

- Aperture photometry ~2-25x faster + thread-parallelizable (#2292); ApertureStats ~5-15x
  (#2314). Our per-frame aperture loops dominate the ~2000 s full run - material runtime
  reduction expected.
- segmentation_image / mask_method in aperture_photometry (#2309, #2321): neighbor-contamination
  masking/correction INSIDE aperture photometry - directly adjacent to our neighbor-subtraction
  program; evaluate against our approach when released.
- Contract changes: aperture_sum_err and area columns always present; photometry returns
  ApertureResults (tuple-compatible). Our parsers should be defensive about extra columns.

## astropy 7.2 -> 8.0

Changelog highlights: FITS fixes (compressed-section slicing with scaled data; ImageDataDiff on
full arrays), config/cache context managers, frame YAML serialization. Nothing photometry-numeric
flagged for our paths. photutils 3.x requires astropy >= 6.1.x - compatible either way. Verdict:
bundle into a gated cycle; low expected risk but major bump = gate anyway.

## numpy 2.4.3 -> 2.4.4

Patch release. Adopt freely in the next cycle.

## Recommended plan

- CYCLE 1 (cheap, anytime): in-range refresh (numpy 2.4.4 + any astropy 7.2.x patches) -> pytest
  + --full. Expected byte-identical.
- CYCLE 2 (the real one; schedule after the INSTALL/Lenovo arc): photutils 3.0 (+ astropy 8.0
  bundled) migration branch:
  (a) kwarg migration (sharpness_range etc., keyword-only audit),
  (b) explicit min_separation=0 at all DAO sites (freeze 2.x behavior),
  (c) PSF-module EPSFBuilder check,
  (d) full pytest + --full vs anchor 435. Expectation: NOT guaranteed byte-identical (perf
  refactors can reorder float ops). If diff: overlay confirmation, then documented re-anchor per
  policy - that machinery exists precisely for this.
- Later, deliberate science evaluations (separate arcs, not upgrades): new-default min_separation
  as a crowding filter; 3.1 segmentation-based neighbor masking vs our neighbor_sub.

## Process (for the DEPS-POLICY task)

- --fast gains an informational pip list --outdated WARN line.
- docs/DEPS_POLICY.md: quarterly ritual = fresh venv with candidate versions -> pytest -> --full;
  identical = free upgrade + pin move + DECISIONS entry; different = finding (adopt-and-re-anchor,
  or hold and report upstream). No scenario is bad; both produce knowledge.

---

## Cursor verification (read-only, tree d437bcd, 2026-07-18)

Confirmed against the current tree:
- Background2D: 0 hits in src_py/ (sky-surface is in-house). CONFIRMED.
- PSF module on modern API: psf_photometry.py / psf_runner.py import ImagePSF, PSFPhotometry,
  IterativePSFPhotometry, SourceGrouper, EPSFBuilder; no FittableImageModel / EPSFModel.
  CONFIRMED (EPSFBuilder IS used -> CYCLE-2 (c) check is warranted).
- centroid_sources: only in src_py/xval_run.py (offline harness). CONFIRMED.
- detect_sources / SourceCatalog: only in src_py/pipeline.py. CONFIRMED.
- Blocker #1 live: no DAOStarFinder call in src_py/ passes min_separation today, so photutils
  3.0's implicit 2.5xFWHM default would change detection. The min_separation=0 freeze is
  genuinely un-applied. CONFIRMED.

Refinement (honesty): on the src_py/ SCIENCE PATH specifically, the deprecated DAOStarFinder
kwargs actually in use are `brightest=` (-> n_brightest) and `roundlo=`/`roundhi=` (->
roundness_range, in vyvar_platesolver.py). No sharplo/sharphi/peakmax in src_py/ (those live in
dev/scripts/ harnesses). The science-path kwarg migration is therefore narrower than the
repo-wide "12 hits" figure, which correctly includes dev/ scripts.

DAOStarFinder call sites in src_py/: pipeline.py (7145, 7414, 8146, 16382), vyvar_alignment_frame.py
(246, 283), vyvar_platesolver.py (4822, 6174), plus utils.py / ui_dao_stars.py /
wide_slope_noise_core.py.

## CYCLE 1 execution log (2026-07-18)

Executed by Cursor. In-range refresh: numpy 2.4.3 -> 2.4.4 (pip install numpy==2.4.4;
astropy 7.2.0 and photutils 2.3.0 held). Pin unchanged (numpy>=2.4,<3 already permits it).

Gates at HEAD 30c803f:
- pytest: 963 passed, 19 skipped (246.9 s) - identical to pre-upgrade baseline.
- session_baseline_check.py --full: OVERALL PASS (2278 s pipeline).
  - full-science-compare: n_lc=166 failures=0
  - full-snapshot-sha-core:    3d26f4692ac81fc5... n=333
  - full-photometry-sha-core:  3d26f4692ac81fc5... n=333  (matches)
  - full-photometry-sha-extended: 6420f1daa53a0d5d... n=499
  - counters: expected {"phase2a_empty_comp_drop": 1} (structural)
  - Ledger anchor items auto-stamped commit=30c803f, last_verified=2026-07-18.

Verdict: numpy 2.4.4 is BYTE-IDENTICAL vs anchor 435. Validated for production.

Finding: the live pip index now offers numpy 2.5.1 (minor bump, still <3), not 2.4.4 as
the scout table stated. CYCLE 1 was scoped to the 2.4.x patch, so 2.5.1 was NOT adopted
here - it is a candidate for a future in-range cycle (still requires pytest + --full).
astropy 8.0.1 / photutils 3.0.0 remain gated CYCLE 2 cross-major work.

## PUSH (2026-07-18)

Pushed the DEPS arc stack to origin/main (fast-forward d437bcd..488b02b).

Preflight: origin/main had not moved (d437bcd); clean fast-forward. Working tree clean
except 3 known untracked scratch scripts (dy_peg_night_run_bvr.py,
forensic_disc_ui_match2.py, qatar8_night_run_v.py).

Outgoing commits (git log --oneline d437bcd..HEAD):
  488b02b docs(readme): drop concrete anchor SHAs, keep generic SHA-256 baseline wording
  a5a9235 chore(deps): CYCLE 1 numpy 2.4.4 in-range refresh - byte-identical PASS
  30c803f feat(deps-policy): add DEPS_POLICY.md + informational deps-outdated WARN in --fast
  eb63314 docs(scout): archive DEPS-SCOUT dependency landscape + Cursor verification

Gates:
- pytest: 963 passed, 19 skipped.
- session_baseline_check.py --fast: OVERALL PASS. WARNs all expected (pre-existing
  git-untracked scratch + ledger-todo VL-ANCHOR-424/DQ-430, plus the deps-outdated
  informational line this stack introduced by design).
- SCIENCE-PATH RULE: no src_py/ files in the outgoing stack -> --full not required; it
  was nonetheless run byte-identical PASS for the numpy 2.4.4 change (anchor 435,
  photometry-sha-core 3d26f469 n=333, extended 6420f1da n=499).

Post-push: HEAD == origin/main == 488b02b; working tree clean (bar the 3 known scratch
scripts).
