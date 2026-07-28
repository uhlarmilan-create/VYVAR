# VYVAR - Decisions & rationale

Durable design decisions and *why* they hold. This is the reference for "why is it like
this" - it should not be reopened without a new decision recorded here. Per-draft validation
numbers and the day-by-day record live in `VYVAR_JOURNAL.md`; open work in `VYVAR_ROADMAP.md`.

---

## RELEASE-TREE-HYGIENE - post-bundle compiled-artifact cleanup (2026-07-23)

**Problem.** WSL/MSVC Cython builds leave `.pyd`/`.so` beside sources (and occasionally
at repo root when relocation fails). These must never appear in `git status` or confuse
dev interpreted runs.

**Rule.** After every successful `build_bundle.py` run, **`run_clean()` runs automatically**
(removes `src_py/*.pyd|*.so`, repo-root droppings, `build/lib.*`, `build/_cython_out/`).
Opt out for debugging only: `build_bundle.py --no-post-clean`. Manual reset anytime:
`python build/setup_cython.py clean` (Windows and WSL).

**Gitignore.** `*.pyd`, `*.so`, `build/lib.*/`, root `/*.pyd`, `/*.so` are ignored; physical
delete is still required so interpreted dev does not shadow stale binaries.

---

## LATENT-NAMES-COMPILE-GATE - latent NameError sweep before Cython release (2026-07-21)

**Trigger.** Cython compile-as-static-analysis + pyflakes/ruff F821 gate ahead of
CYTHON-RELEASE closed-source bundle.

**Fixed (10 + EXC-0030 accessor).**
1. `_get_lc_psf_strict` deleted as collateral in `0b5eb8b` while consumers remained
   (`photometry_core:7419`, `method_lc_output._build_flux` lazy import) -- restored with
   `psf_ac_applied` guard (T2 never-silent AC rule).
2-9. Eight EXC-batch log templates referenced `exc` while handlers bound `e`, `_e`,
   `_ap_exc`, `_alt_exc`, `_meth_exc`, `_wcs_exc` (`photometry_core` x7, `pipeline` x1).
10. `calibration.py`: missing `import sqlite3` (latent NameError on equipment OSC path).
11. `comp_selection_per_target.py:173`: `sqlite3.Row.get` -- **root cause of recurring
    EXC-0030 noise** in `--full` logs (`'sqlite3.Row' object has no attribute 'get'`).

**Systemic guard.** ruff **F821** enabled repo-wide (`pyproject.toml`); `import *` shims
(`photometry.py`, `photometry_phase2a.py`) remain -- F821 blind in those two files only.

**Recurrence.** New guard: ruff F821 (dangling-name class, 2nd+ occurrence).

---

## CYTHON-RELEASE-2 - bundle + embedded runtime + install docs (2026-07-23)

**Decisions (Milan, R1-R6).** Release ships bundled Python 3.12 on Win (embeddable) and
Linux (python-build-standalone); catalogs never shipped; LICENSE reused; preview tags
`preview-YYYYMMDD` (pre-release); equipment via DB Explorer; install docs CZ+EN.

**Data-dir separation (B2).** `resolve_data_root()` in `config.py`: `VYVAR_DATA_DIR` env
override; git dev checkout keeps install root as data root (dev-neutral); bundled installs
default to `%LOCALAPPDATA%\\VYVAR` / `~/.local/share/vyvar`. First-launch bootstrap in
`vyvar_runtime.py`. **Requires --full anchor** after path-resolution change.

**Bundle tooling.** `dev/tools/cython_release/bundle/` (`build_bundle.py`, runtime pins,
launchers, `--selftest`, `smoke_bundle.py`). Public repo content staged under
`release/public_repo/` (Milan copies to `VYVAR-release`; Cursor never pushes public repo).

**Next.** Milan: WSL Linux bundle + GitHub pre-release upload per `docs/VYVAR_RELEASE_RUNBOOK.md`.

---


**Scope (S1-S4).** Release compilation is a packaging step only: `src_py/` stays
interpreted in git; `.pyd`/`.so` and `build/_cython_out/` are gitignored. Dev/CI
anchors run interpreted; compiled builds must reproduce the **same** anchor SHAs.

**MODULE_LIST rule (S2).** Single source: `dev/tools/cython_release/module_list.py`.
All `src_py/*.py` except `app.py`, `ui_*.py`, plus `EXPLICIT_EXCLUDE` (one-line
reason per entry). OSC modules (`osc_extract`, `osc_align`, `gaia_johnson`) included.
84 science modules compiled at RELEASE-1 (`b4c372a`, 2026-07-23); **85** after RELEASE-2
added `vyvar_runtime.py` for bundled first-launch bootstrap (`3369832`); **86** after
BUNDLE-FIELD-FIXES-2 added `run_preflight_log.py` for pre-infolog RUN error capture.

**Pinned flags (S3).** Plain compile only: `annotation_typing=False`,
`Options.docstrings=False`, `language_level=3`, `embedsignature=False`.
`build_release.py` refuses to run if these drift.

**Release gate (S4).** Compiled `--full` must match interpreted anchor byte-identity:
core `03d8fb64...` n=333, extended `bbfcc92e...` n=499, science compare 0 failures.
P1 golden compiled: 7/7 byte-identical (`VL-P1-GOLD`).

**Conditional-skip policy.** Tests using `inspect.getsource` or `.py` source scans on
compiled modules skip only when the module loads from `.pyd`/`.so`
(`dev/tests/cython_compat.py`). No ad-hoc test weakening.

**Tooling.** `dev/tools/cython_release/` (build, clean, latent sweep, smoke, MP verify);
`build/setup_cython.py` shim. Recurrence: `test_cython_annotation_typing.py` (full list
+ flag-drift guard).

**Next.** RELEASE-2: bundling/installer (mixed compiled science + interpreted UI).

---


**Finding.** Cython 3 pure-Python mode defaults `annotation_typing=True`, turning PEP-484
annotations into C type declarations. Plain compile of `comp_selection_per_target` without
the pin changed P1 science (core SHA `4ecbae9f...` vs VL-P1-GOLD `074ae881...`; up to
167/169 `phase2a_empty_comp_drop`).

**Decision.** All release compiles use `annotation_typing=False` in
`build/setup_cython.py` `COMPILER_DIRECTIVES`. PEP annotations remain metadata only.
Typed speedups later use explicit `cython.*` declarations in dedicated modules, never
via PEP-484 annotations on production paths.

**Verification.** With the pin: `comp_selection_per_target` alone and all three spike
modules (`photometry_core`, `comp_selection_per_target`, `photometry_phase2a`) compile;
P1 golden 7/7 byte-identical on Windows (MSVC 14.51, Cython 3.2.8).

**Recurrence test.** `dev/tests/test_cython_annotation_typing.py` pins the directive.

---

## DEPS-CYCLE-2 - photutils 3.0 + astropy 8.0 (+ numpy 2.4.4+) (2026-07-20)

**Pins.** `photutils>=3.0,<4` (installed 3.0.0); `astropy>=8.0,<9` (8.0.1);
`numpy>=2.4.4,<2.5` (validated on 2.4.4; 2.5.1 rejected 2026-07-21, see below).

**Migration (scout checklist).** Kwarg migration (`sharpness_range`, `roundness_range`,
`n_brightest`); explicit `min_separation=0` at every `DAOStarFinder` site to freeze 2.x
detection (#2216); column readers migrated to `x_centroid`/`y_centroid`; PSF module verified
(EPSFBuilder/ImagePSF/PSFPhotometry unchanged API).

**M2 freeze decision.** Adopting photutils 3.0's implicit 2.5xFWHM crowding filter is a
**future science arc** (dense-field program), never an upgrade side effect. All production
DAO sites pass `min_separation=0` deliberately.

**G3 outcome (path a).** `--full` vs VL-ANCHOR-WCSINV **byte-identical**: core
`03d8fb64...` n=333, extended `bbfcc92e...` n=499; science-compare 166/166 PASS.
No anchor or P1 re-cut. Runtime ~2025 s vs ~2296 s pre-upgrade (~12% faster).
See `dev/results/CURSOR_RESULT_deps_cycle2.md`.

**numpy 2.5.1 validation (2026-07-21).** Fresh-install path tested in
`tmp/deps_cycle2_venv` (numpy 2.5.1 + astropy 8.0.1 + photutils 3.0.0): `--full`
science-compare PASS (166/166) but photometry SHA core/extended **FAIL** (run
`842443c7...` / `9cabd82d...` vs anchor `03d8fb64...` / `bbfcc92e...`). Pin
tightened to `numpy>=2.4.4,<2.5`; 2.5.x is a future gated candidate, not blessed
for the compiled release bundle.

## ANCHOR-RECUT-SIGMA-NOTES - VL-ANCHOR-WCSINV re-cut after ASCII slope notes (2026-07-20)

**Origin.** PUBLICATION-PREP Part B `--full` STOP: 19 `comp_quality_*.json` files differed from
`draft_000435_snapshot_skysurface_20260716` only in slope-exclusion note text: ASCII `sigma`
(run) vs Unicode U+03C3 (snapshot). All 166 `lightcurve_*.csv` byte-identical;
`full-science-compare` PASS. Cause: ENCODING-POLICY commit `ecbae90` transliterated the
comp-stability note template in `photometry_core.py`; policy stands, not reverted.

**Gate-spec miss (plain).** The ENCODING-POLICY gate set did not require `--full` before push;
latent SHA drift shipped until Part B touched the shared Phase 0 path and ran `--full`.
**PROCESS fix:** string-literal changes in `src_py` science modules now require `--full` before
push (comments exempt).

**Coverage hole.** P1 golden mini (16 frames) never emits slope notes, so `VL-P1-GOLD` did not
catch the drift. **Test fix:** `dev/tests/test_recur_shatext_templates.py` pins comp_quality
note template strings.

**Re-cut.** Two independent fresh `--full` runs byte-identical; new anchor SHAs locked via
ledger refresh on `VL-ANCHOR-WCSINV` (superseded core `3d26f469...` / extended `6420f1da...`
retained in notes). See `dev/results/CURSOR_RESULT_anchor_recut.md`.

## EXPORT-HEADER-SLIM - methods matrix + slim AAVSO/VarAstro headers (2026-07-20)

**Decision.** AAVSO and VarAstro text exports drop unconditional citation blocks
(`[CORE]`, `[CATALOGS & TIME]`, `[SOFTWARE]`, `[FIELD ASTROPHYSICS]`). They keep
format/site/observer fields, a machine-generated `# METHODS MATRIX (this run):`
ON/OFF block, `# [METHODS - this run]` one-line citations only for methods that
are ON, and a pointer to the SUMMARY MEASURE REPORT (PDF). The full citation
block plus the same matrix remain in the PDF methods section.

**Single-source rule.** The matrix is built from the same `RunCitationContext`
flag->method mapping that gates conditional citations in `citations.py` (extend
that mapping when a gated method is missing -- one source, no drift).

**Citation corrections (CITATIONS.bib, verified vs FLOW ch 19 / ADS):**
- AstroImageJ: Collins, Kielkopf, Stassun & Hessman (2017) AJ 153, 77
  (bibcode 2017AJ....153...77C) -- was wrongly PASP 129, 144502 / Stelzer.
- Jordi et al. (2010): A&A 523, A48 (bibcode 2010A&A...523A..48J) -- was wrongly
  A&A 515, A16. FLOW ch 19 already had 523/A48.
- Emitting notes are plain-ASCII (strip LaTeX `\_`, `\Delta` / `\ensuremath\Delta`).

**OBSCODE warning:** warn ONLY when observer code is missing/empty in config.
Any explicitly set value (including UMIA) emits no warning.

## INVARIANTS-P3P4-CLOSEOUT - recurrence/forensic rubric + honest scope (2026-07-20)

**Delivered.** Invariants program P1-P4 closed.

- **P3:** PROCESS "Recurrence and forensic discipline" (recurrence rule + RESULT
  `Recurrence:` field; forensic PROMOTE/ARCHIVE/DELETE rubric; weekly
  `dev/tools/invariants_report.py` cadence). Pilot: `forensic_disc_ui_match2.py`
  -> **PROMOTE+ARCHIVE** (cat_match_arc=2.0 parity asserted in
  `test_ui_chain_byte_identity`; script header state ARCHIVED).
- **P4:** STATE pinned "Invariants program -- honest scope" (GUARANTEED vs NOT
  guaranteed; claim-verified). GAPS D2 void addendum (EXCEPT-BULK-2 already
  closed 2026-07-08).
- Result: `dev/results/CURSOR_RESULT_invariants_P3P4.md`.

## K2-NIGHT-FIT-V2-IMPLEMENTED - gated NIGHT_FIT fitter + synthetic recovery (2026-07-20)

**Scope finding.** The v2 fit path was never in production: `k2_extinction.py` had v1
literature machinery only; the four `k2_fit_*` keys existed since v1 for forward
compat. This ships v2 per `dev/results/specs/VYVAR_K2_DESIGN_SPEC.md` v1.1 S5/S6.
Remaining activation blocker is **only** the data night (GAPS B2 / K2-DATA-BLOCKER).
GAPS C1 (prove the fitter) is delivered by the synthetic suite against the real
production fitter.

**Adherence.** Model on Honeycutt residuals uses the S5-identifiable form
`k2*(C-Cref)*dX` (dX = X-mean(X) after frame+star CM removal). Pre-gate: monotonic
airmass refuse; detectability `sigma_k2_pred <= |k2_lit|/k2_fit_min_detectability`;
outer colour/brightness tertile + arc consistency within
`k2_fit_consistency_sigma * max(sigma_boot, 0.10.|k2|)`; plausibility ceiling +
literature factor/sign. Any failure -> LITERATURE_DEFAULT with
`k2_fit_refuse_reason`. Default `k2_fit_enabled=false` is byte-identical.

**Synthetic validation (summary).** Recovery sweep over k2in{0,0.02,0.05,0.08} x
spreadin{0.2,0.5,1.0} x noisein{5,15,30} mmag: accepted fits within
`max(2.sigma_boot, 0.005)`; non-detectable cells refuse with `detectability`. Explicit
REFUSE: monotonic X; absurd k2=0.5; tertile-split injection; zero-signal clean.
Draft-427 fixture: `fixture_source=synthesized_from_decisions` (sandbox JSON gone;
signature from SPEC S1/S6 + recon notes) fails items 3/4.

**Activation.** Still blocked on B2 data night. Do not flip `k2_fit_enabled` here.

---

## PER-FRAME-SAT-GATED - per-frame target saturation behind flag (2026-07-19)

**Context re-scope.** The original M67 evidence (76 Green / 49 Red whole-star drops)
came from an ASTROPHOTO dataset with long exposures - an extreme case, not
representative of photometric nights (Milan, 2026-07-19). Severity re-graded
**HIGH -> MED**. The principle stands: a star's usability is a per-frame fact
(`peak_max_adu` is already measured per frame), while today the TARGET-level drop
comes from the MASTERSTAR zone of one reference exposure (`skip_photometry` via
`zone_flag`). Failure mode both ways: a target saturated on 30% of frames is dropped
wholly (loses 70% good data); a borderline target clean on the master can carry
saturated rows. The M67 dataset is gone; **real validation is DEFERRED** to the
next dataset containing saturated bright stars (ROADMAP revisit trigger).

**Design shipped (default OFF).** New flags `per_frame_saturation_enabled=false`
and `per_frame_sat_min_clean_frac=0.5` (clamp [0.1, 1.0]). When OFF: byte-identical
to today's zone-based `skip_photometry` (comps keep their existing per-frame >10%
rule). When ON, for TARGETS only: master zone `saturated` / `likely_saturated` is
advisory; compute clean fraction from per-frame sat flags; >= threshold -> measure
(saturated rows keep flag `saturated`, mask-first); < threshold -> skip with
`skip_reason=per_frame_saturation`; missing per-frame peak data -> fail-safe fallback
to zone behavior + `per_frame_sat_fallback=true`. Provenance: `sat_clean_frac` on
`photometry_summary` (outside photometry SHA / science comparator file set).
INV-CFG-01 extended: flag OFF => none of the new markers appear.

---

## Dependency policy + CYCLE 1 in-range refresh (numpy 2.4.4) (2026-07-18)

**Context (DEPS-SCOUT / DEPS-POLICY).** Grounded in Claude's dependency-landscape
scout (`dev/results/DEPS_SCOUT.md`), Cursor-verified against tree `d437bcd`.

- **Policy adopted.** Upgrades are a gated ritual, not accidents. `docs/DEPS_POLICY.md`
  is the reference: pin majors in `requirements.txt`; a candidate cycle is fresh env
  -> `pytest` -> `session_baseline_check.py --full` (byte-identical anchor gate).
  Byte-identical = free upgrade + (pin move if needed) + DECISIONS entry; a diff is a
  documented finding (adopt-and-re-anchor, or hold and report upstream). Neither
  outcome is a failure. `--fast` gains an **informational** `deps-outdated` line
  (WARN/PASS/SKIP only, offline-tolerant, never blocks).
- **CYCLE 1 result = byte-identical PASS.** numpy 2.4.3 -> 2.4.4 (in-range, within
  the existing `numpy>=2.4,<3` pin, so no pin move). `pytest` 963 passed / 19 skipped;
  `--full` vs anchor 435 PASS at HEAD `30c803f`: `full-science-compare` n_lc=166
  failures=0, `full-photometry-sha-core` `3d26f4692ac81fc5...` (n=333), extended
  `6420f1daa53a0d5d...` (n=499), pipeline 2278 s. Ledger anchor items auto-stamped
  `commit=30c803f, last_verified=2026-07-18`. numpy 2.4.4 is validated for production.
- **Finding for the next cycle.** The scout's table listed numpy latest as 2.4.4, but
  the live index now offers **numpy 2.5.1** (a minor bump, still within `<3`). CYCLE 1
  was scoped to the 2.4.x patch, so 2.5.1 was deliberately **not** adopted here; it is
  a candidate for a future in-range cycle (minor bump -> still needs pytest + `--full`).
  astropy 8.0.1 and photutils 3.0.0 remain gated cross-major work (CYCLE 2, after the
  INSTALL/Lenovo arc).

---

## License, repo visibility & distribution model (2026-07-18)

**Milan decision (DOCS-FIX-ARC1).**

- **License = proprietary, all rights reserved.** VYVAR ships under a short
  proprietary `LICENSE` at the repo root: Copyright (c) 2026 Milan Uhlar; no use,
  copying, modification, or distribution without prior written permission; no
  warranty. The `docs/README_FULL.md` / `docs/README_CZ.md` license sections match this text.
  Rationale: VYVAR is not (yet) an open-source release; keep all rights with the
  author until a deliberate licensing decision is made.
- **Repository visibility = private** (already applied on the remote). The GitHub
  front-door README is written for eventual public reading, but the repo stays
  private for now.
- **Compiled-library distribution = deferred (recorded, not adopted).** The idea of
  shipping VYVAR as a compiled/obfuscated library instead of source was considered
  and explicitly parked; revisit only if/when a distribution need arises.

---

## WAVE-B-PARAM-REDUCTION - cut/merge/wire the 304-key surface to 269 (2026-07-18)

**Milan sign-off (post PARAM-BUDGET-AUDIT).** Acting on the audit dispositions in
`dev/results/param_budget_audit.csv`, the registered-parameter surface was reduced
from 304 to 269 entries (config.json persists 249). Executed as anchor-gated steps,
one commit each, full pytest green throughout, and closed by the `--full` anchor gate
vs `draft_435` (fallback removals touch the science path; byte-identical was required).

**What was cut and why.**
- **DELETE-DEAD (4):** `aperture_fwhm_factor_medium`, `masterstar_log_astroalign`,
  `phase01_comparison_proximity_tiebreak`, `phase01_comparison_rms_bin_mag` - no live
  readers; removed from AppConfig, registry, generated docs, config.json.
- **MERGE 14 -> 3:** eight `comp_tier{1..4}_bprp_limit/_weight` scalars -> one
  `comp_color_tiers` list-of-dicts; four `phase01_tier{1..4}_mag` -> one `phase01_tiers`
  list; `aperture_fwhm_factor_small/_large` -> one `aperture_snr_sizing` mapping. Loader
  accepts the OLD scalar keys for one transition release (deprecation log) and maps them
  into the structured key; save writes only the new form.
- **DELETE-DB-DUP (9):** `gain`, `read_noise`, `plate_scale_arcsec_per_px`,
  `phase01_plate_scale_arcsec_per_px`, `export_arcsec_per_px`, and the observer mirrors
  (`observer_lat/lon/alt_m/location_name`) removed from config.json persistence in both
  directions. The AppConfig fields remain as run-time hydrated mirrors; the DB/FITS
  resolver is now the ONLY authoritative source. The vestigial DB `SETTINGS` table (no
  production readers) is dropped via idempotent migration; dead `get_setting_int` /
  `set_setting` removed. Masterdark/flat validity days confirmed config-authoritative.
- **HARDCODE (20):** blind/plate-solve solver internals that were never tuned in practice
  moved to module-level constants in `vyvar_blind_solver.py`, `vyvar_platesolver.py`,
  `pipeline.py`; removed from AppConfig, registry, config.json, generated docs.

**WIRE-IN finding (bug fix).** `calibration_master_ccd_temp_tolerance_c` was registered
but never passed to `find_best_calibration_library_path`; the dark-selection tolerance was
silently hardcoded. Now wired from `cfg` at both importer call sites. No-op for current
runs (key absent from Milan's config.json -> effective value stays 0.5).

**INTERNALIZE (2).** `frame_width_px` / `frame_height_px` stay as AppConfig fields (they
still resolve from FITS NAXIS at run time) but leave the USER parameter space:
owner=internal, widget=hidden, tier=expert, and dropped from config.json persistence.

**Focal-length precedence.** The two resolution paths were audited and already resolve
DB-optics-first with FITS-header fallback (`param_resolver.resolve_focal_mm`); no code
change was needed - the audit note was stale.

**The 80-key "never-touched expert" hardcode pool: REJECTED - stays KEEP.** Universality
argument: those keys are legitimate scientific knobs that apply across all sites/targets;
the fact that Milan has not re-tuned them from defaults is evidence they are well-chosen
defaults, not evidence they are dead. They remain user-visible expert parameters.

**sips_dao_fwhm clarification.** The audit's bonus `sips_dao_fwhm` claim is RETRACTED: the
code uses the registered `sips_dao_fwhm_px` everywhere (config.py, pipeline.py,
photometry_core.py, vyvar_platesolver.py, night_run.py, app.py) - there is no bare
`sips_dao_fwhm` key.

---

## COMP-TRUST-MIN-COMPS - 3 vs code default 5 is INTENTIONAL (2026-07-17)

`comp_trust_min_comps=3` (config.json) vs code default 5: INTENTIONAL. Since 1c80219
(2026-06-16, Phase-1 graceful comp degradation) the key is the GREEN trust threshold, not
a hard RED floor; 1-4 good comps yield YELLOW with sigma scaling by N (see
VYVAR_COMP_DEGRADATION_SPEC.md). GREEN at 3 vs spec example 5 validated by matrix 164157.
Config page in SUMMARY MEASURE REPORT will permanently list this as a deviation from code
defaults - that is correct behavior.

---

## INVARIANTS-P1-GOLDEN-MINI (2026-07-19)

**Mini source.** In-Archive `draft_000435` (live draft). Fail-early requires
calibrated lights + detrended_aligned proc products + platesolve
MASTERSTAR/catalogs. Raw lights exist (150) but in-draft `Raw/darks` and
`Raw/flats` are empty (masters come from CalibrationLibrary at import time).

**Scope note.** Mini starts at **photometry-ready** stage: 16 calibrated + matching
`proc_*.fits/csv` + parent platesolve catalogs/MASTERSTAR. Chain coverage matches
`session_baseline_check --full` (`run_full_photometry_pipeline`). Calibrate -> QC ->
align are not re-run on the mini (would require local masters + QC DB state); the
UI-order test still exercises UI path discovery (`_find_phase2a_paths`) vs headless
direct paths for the photometry composition. Dual-entry divergence remains an
F-431-class STOP (report, do not silent-fix).

**Size / selection.** 16 frames, even DATE-OBS stride across the time-sorted list of
frames that have both calibrated and aligned products (QC-rejected frames excluded),
first frame included. Same masters/catalogs/config as parent night.

**Expected at n=16.** Variability detection no-ops (`variability_min_frames=30`);
`lc_quality` may classify short-baseline - both correct, not failures.

**Reproducibility before lock.** Build + full headless run twice from scratch;
science outputs byte-identical; only then register `VL-P1-GOLD`. Confirmed
2026-07-19 (run1 611.6 s, run2 574.7 s; core/ext SHAs identical).

---

## VYVAR-INVARIANTS - machine-enforced contracts (2026-07-16)

**Premise (Milan sign-off by commissioning).** Static audits cannot catch integration-class
defects (ordering, path assumptions, transform relationships, cross-entry divergence,
unprovenanced code). Every defect found in the F-428/F-431 arc was caught by a
**measurement**, not by reading code. This program converts VYVAR's physics/math contracts
into machine-enforced invariants.

**Goal (honest):** not "zero bugs" (unattainable); rather (a) no defect reaches science output
silently, and (b) no defect class recurs once fixed. After P1-P3 the guarantee is: silent-wrong
science requires simultaneously fooling independent guards (flux conservation, WCS identity,
census bands, SHA determinism, golden UI<->night_run equivalence); every ledger defect class has
a recurrence guard. Audits continue for dead code / style / docs / obvious logic only.

**Timing.** Starts **after Anchor #3 closes**. Phases land as separate commits; pytest green
each. Order: P1 golden E2E equivalence -> P2 contract registry / runtime gates -> P3 PROCESS
recurrence discipline -> P4 STATE scope statement.

**Out of scope:** UI Settings block (parallel), new science features.

### INVARIANTS-P2-REGISTRY (2026-07-19)

**Delivered.** Human registry `docs/VYVAR_INVARIANTS.md` + runtime library
`src_py/invariants_runtime.py`. Wired gates (check-only): INV-FLUX-01/02 (FAIL),
INV-FLAT-01 (WARN), INV-WCS-01 (WARN), INV-DAG-01 (FAIL; cold-start / gap allowed for
sparse stamps), INV-RNG-01 (TEST AST; zero hits), INV-PROV-01 + INV-CFG-01 (FAIL end-of-run).
INV-FLUX-03 / INV-SHA-01 / INV-WCS-00 are registry-only pointers.

**Meta / SHA scope.** `pipeline_meta.json` is outside core/extended photometry SHA and the
`--full` science comparator (`dev/tests/photometry_sha.py`). Invariants/stages blocks are
written into `pipeline_meta.json` freely; science outputs remain the byte-identity contract.

**FLOW 4.5 correction (doc-drift evidence).** Builder prose claimed sky-surface preprocess was
"flux-conserving / mean-preserving". Code and T3 decision subtract the **full** fitted surface
including the constant term (pedestal; Delta median ~ -96 ADU on the reference frame). Registry
INV-FLUX-03 now owns that claim; FLOW regenerated accordingly. This is the class of
prose-physics drift the facts guard cannot catch.

---

## T3-PREPROCESS-SKY-SURFACE - order-2 shared preprocess (2026-07-16)

**Milan sign-off (commissioning T3-RESTORE).** Per calibrated light frame, fit a 2D polynomial
surface of order 2 to the background (source-masked + sigma-clipped fit), subtract it in the
shared ``calibrated -> processed`` step used by BOTH chains (UI and headless route through
``preprocess_calibrated_to_processed`` -> ``_preprocess_calibrated_one``). Config:
``preprocess_sky_surface_order`` (int, 0 = off, **default 2**). Subtract the **full** fitted
surface including the constant term (pedestal convention: Delta median ~ -96 ADU on BO CVn
Light_008 vs draft_429).

**Rationale.** DAOStarFinder and Labbe background assume a locally flat field. The lost
429-era preprocess step removed large-scale gradient; without it the deterministic repo path
inflates pass-1 DAO (~8927) and flips automatic ``DENSITY_OVERRIDES``. Restoring the surface in
shared preprocess targets **429-class** census/quality without per-chain drift.

**Provenance.** Per-frame ``VYSKYORD`` / ``VYSKYP2P`` FITS headers; preprocess row stats in QC
dataframe.

**Dirty gate (paired).** ``git_dirty_code`` (import-relevant ``*.py`` only) gates anchor /
FAIL-CLOSED; ``git_dirty`` + scratch lists remain for transparency.

**Empirical gap (honest).** Cal-only surface fit vs draft_429 oracle MS: DAO pass-1 sim **2579**
vs logged **2816** (~8%); smooth-field residual p99 **173 ADU** vs oracle **52 ADU**. Frame
median shift differs from meta ``sky_adu`` (~1478 Labbe annulus). Full MS residual unavailable
at preprocess time; Milan UI run is acceptance gate.

---

## F-431-HEADLESS-DIVERGENCE - CLOSED (2026-07-16 / T3)

**Root cause:** lost unprovenanced order-2 preprocess ADU step (``cal==proc`` on deterministic
path). **Resolution:** shared ``_fit_subtract_preprocess_sky_surface`` in
``preprocess_calibrated_to_processed``. See `CURSOR_RESULT_t3_restore_anchor.md`.

**Recurrence (2026-07-22 / SKIPPROC, second occurrence):** ``013cb0c`` retired the
``processed/lights`` copy tree and removed the in-place order-2 sky-surface subtract from the
mono calibration path. Same defect class as F-431; see **SKY-SURFACE-RESTORE** (2026-07-27).
First closure shipped **without a regression guard**, so the step could return silently until
**INV-PREP-01** (preprocess gradient / large-small ratio on calibrated frames) made a third
occurrence detectable from the saved infolog.

---

## F-428-PASS2-CONTAMINATION - draft_428 census products unreliable (2026-07-16)

**Status (amended 2026-07-16 / F-431):** **DOWNGRADED** as the *primary census driver*.
Draft_433 (`715391b`, FIX 2 / WCS-INV active, clean tree, UI-parity match sep) still reproduces the
**6699-class** inflated census. Primary driver of pass-1 inflation is therefore the **absent
preprocess ADU mutation** that distinguishes 429 (`cal!=proc` Light_008 / MASTERSTAR) from the
deterministic repo path (`cal==proc`). Keep both facts below.

**Original finding (still true, secondary for census).** On draft_428, Gaia-targeted DAO pass-2
ran through a corrupt forward SIP (~12 arcsec / ~5 px bookkeeping offset per v5). Evidence retained:
428 `n_raw_dao=8927` vs 429 `2816`; unmatched 2724 vs 179; 1172 matched catalog_ids only in 428
(median mag 15.1, median sep-to-Gaia 17 arcsec). **Per-target photometry of well-detected stars is
unaffected** (v5 flux/identity correct). The WCS inversion asymmetry on 428 was a **real, fixed**
defect (`F-428-WCS-INV`); it is **not** sufficient to explain the 8927->2816 pass-1 collapse once
FIX 2 is live (see F-431-HEADLESS-DIVERGENCE / `CURSOR_RESULT_headless_forensics.md`).

**Density override coupling.** Inflated matched census still flips automatic `DENSITY_OVERRIDES`
(428/431: `annulus_inner_fwhm=5.75`, tighter comp gates vs 429 defaults). Monitor
`config_snapshot` near the density threshold.

**429 provenance.** draft_429 remains the **census QUALITY target** (3054 / 2875), but is an
**unprovenanced anomaly** (`git_dirty=true`, dirty file list never recorded). Deterministic repo
state today is the **6699-class**. Do not treat 429 as a golden clean-SHA run until the lost ADU
transform is restored in committed shared preprocess.

**Anchor implication.** New anchor requires healthy-census tree under protocol v2 +
`git_dirty=false` hard gate. Anchor arc **BLOCKED** until F-431 root closure.

---

## F-428-WCS-INV - MASTERSTAR WCS invertibility gate + coordinate finalization (2026-07-16)

**Decision.** After F-428 COORD v5 (`RECLASSIFY-PROJECTION`), encode three durable guards:

1. **Round-trip invertibility gate** (`wcs_invertibility.py`): after every plate solve (WARN +
   provenance flag) and after every optimizer SIP refit (FAIL-CLOSED - keep previous WCS). Metric:
   p99\|pix - world2pix(pix2world(pix))\| on a 9x9 grid; threshold 0.2 px. Persist
   `wcs_roundtrip_p99_px` / `wcs_roundtrip_pass` in `pipeline_meta.json`.

2. **SIP inverse regeneration** (FIX 2): forward SIP fit (`A/B`) must ship with fitted `AP/BP` in
   the same step (`ensure_sip_inverse_coefficients`). Optimizer SIP refit pairs DAO `(x,y)` with
   **Gaia catalog sky** (not stale `ra_deg`/`dec_deg` on the row) - root cause of v4/v5 bookkeeping
   offset while `world2pix(Gaia)` agreed with pixels.

3. **Coordinate finalization** (`finalize_masterstar_sky_coords`): matched rows -> Gaia catalog
   `ra_deg`/`dec_deg` + `coord_source=gaia_catalog`; unmatched -> `final_wcs` pix2world. Post-match
   identity sep gate in pixel space (WARN > 1.5xFWHM, drop assignment > 3xFWHM).

**Why.** v5 showed `world2pix(Gaia[cid])` within ~1.3 px of ms `x/y` while `pix2world(x,y)` was
~12 arcsec from Gaia - science flux/identity correct, sky columns wrong. Internal round-trip can still
pass (astropy numerical world2pix); the identity gate + finalization close the bookkeeping chain.
Angle histogram concentration (97/160 in two bins) is SIP-field distortion pattern, not isotropic
confusion. Photometry/LC science columns unchanged by this batch.

**Radius / field DB.** F-428-A3-RADIUS remains OPEN (deeper/wider field DB; GAIA-DR4 ~Dec 2026).

---

## Second-order extinction k'' - band-aware v1 (2026-07-07)

**Decision.** Ship k'' as a **deterministic, provenance-tagged correction** (not a per-night free
parameter in v1). Source hierarchy: `night_fit` (v2 only, pre-gate) -> `literature_default` -> `none`.

**Q1 - default `k2_mode=literature`.** Population-mean Smith/Henden coefficients; explicit
`k2_source`/`k2_value` on every LC row; independent of `apply_color_term` (CT default remains off).

**Q2 - v1 scope:** `band_classify` CT wiring + CV/CR flip + literature k'' path. NIGHT_FIT machinery
stays in sandbox; `k2_fit_enabled=OFF` until a feasible dataset exists (K2-DATA-BLOCKER).

**Q3 - draft_425 validation** under snapshot discipline (manifest + raw-light checksum sample).

**OSC tri-colour (BLUE/GREEN/RED, TG/TB/TR):** `STANDARD_FILTER` for CT eligibility only; **k2=none**
- no citable k'' for Bayer RGB bandpasses (`k2_none_tokens` second check in `k2_extinction.py`).

**Mandatory ordering.** k'' on comp instrumental mags **before** `fit_color_term_c1`; k'' on target
`mag_calib` **before** `apply_color_term` - otherwise fitted c1 absorbs k''xX_bar (CT bias).

**Missing data (fail-soft).** No finite Gaia BP-RP -> no k'' (`k2_source=none`); no finite airmass ->
skip that frame; never invent default colour.

**Converter.** `k2_bprp = k''_native x d(C_native)/d(C_bprp)` (Jordi et al. 2010 slopes in
`k2_extinction.py`; B-band uses cited `d(B-V)/d(BP-RP)`).

Spec: `docs/VYVAR_K2_DESIGN_SPEC.md` v1.1.

---

## Colour: raw Gaia BP-RP only; B-V / Riello citation removed (2026-06-25)

**Decision.** Johnson B-V is **deprecated** for tiering and reporting. Colour comes from raw
Gaia `bp_rp` on matched stars; no BP-RP -> B-V transform is implemented or desired. The PDF
report and citation emitter no longer cite Riello 2021 as a B-V transform. The legacy `b_v`
column plumbing remains (mostly NaN) - removal is a separate bounded task.

**Err model (F-HOWELL-3, Stage C FIXED 2026-06-25).** Proc export writes explicit
`sky_adu_per_px_annulus` (annulus median ADU/px) alongside legacy `noise_floor_adu` (detection
floor on MASTERSTAR). `_photometric_error` prefers the annulus column; falls back to
`noise_floor_adu` on older proc CSVs. Verified on draft_424: canonical LC science byte-identical;
sky-dominated faint targets show ~12-14% err inflation if detection floor were used instead of
annulus. Edge case (`photometry_mode=epsf` without ePSF): aperture enhance skipped - rare;
structural insurance via explicit column + fallback.

**Time provenance (F-BJD-1, Stage D FIXED 2026-06-25).** `_recompute_bjd_hjd_with_status` returns
`(bjd, hjd, time_base)` where `time_base` is `BJD_TDB` on astropy success or `JD_FALLBACK` on
the three documented fallback paths (invalid coords; observer 0/0; astropy batch failure). The
2-tuple wrapper preserves sandbox/test callers. Production Phase 2A writes constant per-frame
`time_base` into `lightcurve_*.csv`; numeric time columns unchanged. `compare_photometry_science_meaningful`
excludes `time_base` from science diffs.

---

**Decision.** VYVAR measures variable targets only when they have a **masterstar (DAO+Gaia) cross-match**.
VSX entries without such a match are **excluded upstream** in Faza 0 (`select_active_targets`), not
measured via forced aperture at VSX catalog coordinates. The former `catalog_only` / `forced_aperture`
Phase 2A branch, `lc_source` provenance column, and `catalog_only_n_comps` config are **removed**.

**Variable-target flux rule (strict).** Active variable targets are measured **only on a direct DAO
`catalog_id` hit** in the per-frame proc CSV (`read_flux_from_csv`). If the target `catalog_id` is
missing from the frame catalog -> **nondetection / NaN** (`no_data`). **No XY fallback** for variable
targets (prevents wrong-neighbor grabs on intermittent DAO frames). Comp stars retain the legacy
`_lookup_star_in_csv` path with the bright-neighbor reject (`fallback_mag > -8.0`).

**Removed knobs.** `phase2a_variable_xy_fallback_mag_tol`, `_phase2a_variable_xy_fallback_expected_mag`.

**Shared helper preserved.** `_catalog_only_fixed_aperture_flux` was **renamed** to
`_annulus_sky_subtracted_flux` (body unchanged) - it remains the annulus sky-subtraction primitive for
DAO PSF local-sky (`psf_photometry.py`) and neighbor-sub residual aperture (`psf_neighbor_sub.py`).

**Unchanged.** Saturated targets still get `skip_photometry=True` / zero-frame skip in Phase 2A.

**Supersedes.** G2-F001 forced-aperture routing (`f42ce89`) and G2-F002 catalog_only WCS placement
(`0dd59d7`) - both paths deleted rather than maintained in parallel.

**Validation (strip-FORCED + Phase-2A vs draft 419, Jirny comp lists).** `mag_inst` byte-identical
**360/360** B stable pool; **346/350** R (4 frames = intermittent-DAO NaN gaps, intended). `mag_calib`
byte-identical **357/360** B and **344/350** R. One B target (`458415401545371264`) shows a **uniform
~30 mmag zeropoint shift** on all 12 frames (`mag_inst` unchanged): comp_qa correctly **excludes**
intermittent forced-only comp `458412790204894208` (10/12 DAO, 2 frames FORCED_APERTURE-only in 419) ->
ensemble 8->7 comps. **Accepted** as intended DAO-only effect (variability shape preserved; `lc_rms`
slightly improved). R outlier `458470858164631936` frame 0041: wrong-neighbor XY closed -> NaN gap.

**Future backlog (not this fix).** Comp selection should exclude intermittently-DAO comps upfront
(e.g. `458412790204894208`, listed `comp_n_frames=12` but DAO-only 10/12) so comp_qa need not drop
them post-hoc.

---

## Photometry: canonical published magnitude `mag_calib_final` (Path A, 2026-06-22)

**Decision.** One canonical calibrated magnitude for all **publication-facing** outputs:
`mag_calib_final = mag_calib + ct_correction (if ct_ok) + delta_m_corr (if ac_ok)`, stored in
each `lightcurve_*.csv` alongside provenance columns (`mag_calib`, `mag_calib_ct`,
`mag_calib_ac`, `mag_calib_raw`).

**Consumers.** AAVSO export, VarAstro MAG column, main per-star PDF LC plots, candidate PDF LC
figures - all read `mag_calib_final` (`export_reports.py`, `photometry_report.py`).
VarAstro `delta_mag` remains the ensemble differential (not CT/AC adjusted).

**Not changed.** `lc_rms`, `lc_rms_ooe`, comp_qa, trust gate continue to use `mag_calib`
(CT/AC are per-target/night constants -> scatter invariant). Variability detection uses
instrumental `dao_flux` from proc CSVs.

**Supersedes.** Split consumer model (export AC-only vs PDF CT-only) flagged as G5-F011;
G5-F003 candidate-figure AC precedence is subsumed by this column.

**Implementation.** `be3e193`; full data-flow doc: `docs/VYVAR_CALIBRATION.md`.

---

## Fix C diagnosed (C1): run-414 bad frames are PSF/FWHM-bloated, not mis-aligned; recovery N/A (2026-06-18)

**Finding (measured, run-414 g, production alignment path; `CURSOR_RESULT_fixC_diag.md`).** The 14
late-night frames Fix B drops are **PSF-degraded**, not "good data that only failed alignment". bad-14
median **FWHM 8.60 px = 1.85x the good baseline 4.64 px**; concentration flux_large/flux **13.1 vs 1.65**
(8x worse); **corr(FWHM, alignment-residual) = 0.95** across 161 frames. The bloated-donut centroid noise
(~2.4 px) is the single root: it breaks astroalign's asterism matching (the **misalignment is a symptom**)
and it is exactly what B.2 (concentration) and Fix-B (alignment residual) measure - two downstream
symptoms of one cause. Most likely cause: **late-night focus drift on the deliberately-defocused rig**
(a pure transparency/flux drop would not bloat FWHM); the post-flip half may not have been refocused
(observer question).

**Decision - Fix C "alignment recovery" is NOT APPLICABLE; close it.** The frames are **not recoverable
to sub-px**: residual geometry on the matched subset is incoherent ~2.4 px scatter (rotation ~0 deg, scale
~1.0, |t| ~0.1 px; a similarity fit does not reduce it; no radial trend) + ~50 % source loss - centroid
noise, not a fixable transform. Their centroid floor (~2.4 px) is **above the 1.37 px gate**, so even a
perfect alignment would still be flagged. Candidate tests confirm it: control-point cap 50 -> 3/14, cap 80
-> 2/14, cap+isolation -> 1/14; per-frame WCS is absent (0/162) so WCS-reproject is unavailable;
translation-refinement is inapplicable to incoherent scatter. Force-aligning bloated PSFs would not yield
science-grade photometry and would risk the 147 working frames / all rigs. **Diagnose-before-fix
prevented shipping a useless, risky alignment change.**

**Decision - Fix B + B.2 are the correct, PERMANENT handling.** They reject genuinely unusable frames;
there is nothing to "recover", so the residual gate is a **permanent quality gate**, not a stop-gap that
self-deactivates once "Fix C" succeeds. (The earlier "self-deactivating" wording is superseded below.)

**Decision - log a SEPARATE perf/robustness ticket (not Fix C recovery).** astroalign at the production
mcp~200 on dense fields is ~654 s/frame (and still fails); cap to ~50 (astroalign's design point) ->
~3-10 s. Two integration shapes - additive recovery rung (keeps the 147 byte-identical, does **not** fix
slowness) vs primary cap (fixes slowness, changes the 147 transforms -> needs a **cross-rig regression,
home + narrow rigs**, before adoption). Deferred until cross-rig data is available. See ROADMAP.

**Watch-item.** Mildly-bloated near-threshold frames kept by the gate (e.g. g_0231, FWHM 5.12 = 1.10x):
differential photometry largely cancels common-mode FWHM changes, so likely benign - watch the LC near
the good->bad transition.

---

## Fix B: reject-on-alignment-residual frame gate is rig-agnostic + cause-correct; default-OFF (2026-06-18)

> **[Updated by C1, 2026-06-18.]** The "Problem"/"always-on"/"Self-deactivating" clauses below assumed
> the frames were *recoverable* once "Fix C" fixed alignment. **C1 refuted that** - the frames are
> PSF/FWHM-bloated (FWHM 1.85x; corr(FWHM,residual)=0.95) and **unrecoverable** (centroid floor ~2.4 px >
> 1.37 px gate). Fix B is therefore a **permanent quality gate**, and the alignment residual is a
> *symptom* of PSF bloat (alongside B.2), not a signal awaiting a future alignment fix. See the C1
> decision above.

**Problem.** run-414 D-A/D-B: the catastrophic V0454 LC outliers are 13 phase_correlation frames
mis-aligned ~2.1 px (translation-only fallback after astroalign failed on the dense field). The frames'
photometry data is fine - only the alignment failed. ~~They are recoverable once alignment is fixed
(Fix C), but until then they must not reach photometry.~~ **[C1: not recoverable - PSF/FWHM bloat;
permanent gate. See C1 decision above.]**

**Decision 1 - record a per-frame alignment residual as always-on QC.** Compute, at the Phase-2A
frame-selection point, a per-frame residual = median (over bright matched sources, 10<=mag<=13) of the
deviation of (x,y) from each source's robust across-night median position, and write it as
`align_residual_px` in `alignment_report.csv`. *Why this metric:* the across-night median reference is
dominated by the well-aligned majority, so a translation-mis-aligned frame's residual ~ its full shift
- it reproduces the diagnostic's clean separation (astroalign ~0.36 px vs phase_corr ~2.13 px) without
needing the alignment transform's internal control points. *Why always-on:* it is additive QC metadata
(does not feed photometry -> baseline byte-identical) and is a direct measure of frame quality.
**[C1: it is a *symptom* of PSF/FWHM bloat (corr=0.95), not a signal a future "Fix C" will lower.]**

**Decision 2 - gate threshold is RELATIVE (rig-agnostic), not a fixed pixel value.** Reject if
residual > `frame_align_residual_max_frac x science-aperture-radius-px`. *Why relative:* a fixed px
threshold would mis-generalize across rigs/plate-scales/focus; expressing it as a fraction of the
aperture radius ties it to where flux physically leaves the aperture. *Default 0.25:* on run-414 the
good/bad gap (1.206->1.450 px) is 0.22-0.27 x the 5.47 px aperture radius, and physically residual
>=~0.2x aperture-radius is where the defocused-donut flux starts leaving the science aperture; 0.25
(->1.37 px) sits in the gap. Safety floor (`min_keep_frames`, default 10) makes the gate a no-op rather
than nuke a marginal night.

**Decision 3 - default-OFF; method-agnostic; keep distinct from B.2.** Default-OFF => byte-identical
(verified: 70 targets 0 diff, V0454 max|diff|=0). ON drops 14 frames = all 13 phase_correlation **+ 1
mis-aligned astroalign** (dr=1.648, itself an LC outlier): the gate keys on *measured residual*, not on
the alignment method, so it correctly catches a bad astroalign frame and would spare a well-aligned
phase_correlation frame - this is desired, not a defect. **Relationship to B.2 (kept separate):** the
residual gate (alignment quality) is the cause-correct signal; B.2 (flux_large/flux concentration) is
the aperture-integrity symptom and also catches genuine transparency collapse the residual gate would
miss. On run-414 the residual gate is a strict superset of B.2 (overlap 13, residual-only the 1
astroalign, B.2-only 0). Consolidation is deferred (a future question), not done here. ~~**Self-
deactivating:** once Fix C makes astroalign succeed on these frames their residual drops below
threshold and the guard stops rejecting them - a safety net, not a permanent exclusion.~~ **[Corrected
by C1, 2026-06-18: the gate is PERMANENT - the frames are PSF/FWHM-bloated (centroid floor ~2.4 px >
1.37 px threshold) and unrecoverable; there is no "Fix C" that lowers their residual. See C1 above.]**

---

## Per-point `err` = photon (+) ensemble-ZP residual SEM (NOT std of comp instrumental mags) (2026-06-18)

**Bug.** The LC `err` term-3 was `ensemble_scatter/sqrtn_ens` with `ensemble_scatter = np.std(comp_vals)`
on the comparison stars' **instrumental** magnitudes (`photometry_core.py:2552,:2567`). The std of
comps' instrumental mags is dominated by their **brightness spread** - a fixed ensemble-composition
number, ~constant frame-to-frame - not by per-point uncertainty. For V0454's 2 comps (instrumental Delta
1.655 mag) it injected a constant **0.585 mag** onto every point, ~23x the empirical 0.025 mag
differential scatter. The LC centres were always correct (flux-sum zeropoint `ens_med`); only the error
bars were wrong. This is a distinct error-propagation bug that **shares the sparse/colour-mismatched-comp
structural root** with trust-RED (a thin, brightness-spread ensemble amplifies it) but is not the same
issue.

**Decision (bug fix; corrected formula is the default, no flag).** Term-3 is now the per-frame
**ensemble zeropoint standard error** from the comps' residuals: for each comp, residual = its
instrumental mag minus its **own across-night median** (`comp_ref_map`), so the comps' brightness and
constant colour offsets cancel and only the genuine per-frame zeropoint scatter remains; the per-point
contribution is `std(residuals, ddof=1)/sqrtn` (Honeycutt 1992 PASP 104:435, `honeycutt1992`). The former
`comp_rms_med/sqrtn_ens` term-2 is **dropped** to avoid double-counting the same ensemble-ZP quantity. The
photon/SNR base term-1 is **kept** (correctly large/NaN when a mis-centred aperture tanks SNR - that is
Fix B's domain). Small-n robustness: a near-zero residual SEM leaves `err` = photon base (the floor).

**Consumers (Step-1 audit).** `err` does **not** feed the trust verdict, `lc_rms` (empirical
`np.std(mags)`), the production Broeg ensemble combine (`1/comp_rms^2` weights), or production
sigma-clipping - so trust and production weights are unaffected. It feeds AAVSO/VarAstro export MAGERR +
the PDF median-error (intended; now correct) and **SysRem's inverse-variance weights**
(`run_sysrem_field`, `W=1/err^2`, Tamuz 2005) - `sysrem_enabled` is **default-OFF**, and the fix
*improves* its weighting (mis-aligned frames down-weighted instead of the old ~uniform weighting). No
SysRem code change.

**Verification (run-414 g, re-run vs committed artifacts).** `mag_calib`/`delta_mag` **byte-identical**
(max|diff|=0); V0454 `err` 0.581->0.013 vs empirical 0.025 (mis-cal 23.5x->0.5x); err now tracks
brightness (corr(err,mag) +0.75), no fixed baseline; the 13 mis-aligned frames still flagged via the
photon term. Evidence: `CURSOR_RESULT_414_diag.md`, `tmp/fixA_verify.py`.

---

## V0454 CrA amplitude = real eclipse egress + a position-dependent meridian-flip step (2026-06-17)

**Finding (diagnose-only, no code change).** V0454 CrA's ~0.45 mag gate-ON rise on draft_413 g
decomposes into a **real eclipse egress that dominates ~4:1** plus a **secondary ~+0.1 mag
post-fainter meridian-flip step**. The egress and the flip coincide in time (egress is pre-flip,
plateau is post-flip) but are cleanly separated by their comp-dependence:

- **Egress is real (comp-invariant).** Differential curves vs comps spanning 141-1840 px overlay in
  the pre-flip branch: pre-flip rise mean -0.288, **std 0.088 mag**; ~0.369 mag of the rise occurs
  *within the single pre-flip orientation* (no flip involved); SIPS independently reproduces the same
  egress->plateau on the same aligned frames.
- **Flip step is instrumental (comp/position-dependent).** Post-flip offsets fan out with comp
  position: near-centre comps -0.25, mid/far -0.38...-0.53 (**std 0.174 mag** across comps). Check stars
  near V0454 show a cross-flip step **median +0.100 mag (post-fainter)**, growing with separation
  (~0 co-located -> +0.2 at ~450 px). Boundary continuity (D3) shows the post-flip plateau sits +0.144
  mag below the extrapolated egress (part natural turnover, part step).

**Consequence.** The VYVAR-0.450 vs SIPS-0.548 amplitude gap (0.098 mag) is a **comp-choice / flip-step
artifact, not a pixel-data disagreement** - SIPS used the identical aligned frames; its comp simply
carried a flip step opposite to VYVAR's near-centre ensemble (measuring V0454 against far comps yields
~0.55 mag, SIPS-like). Root cause = **uncorrected flat-field under the 180 deg p->-p flip mapping**,
exacerbated by non-cal data. This grounds the ROADMAP **flip-aware comp selection** candidate (prefer
near-target comps / per-side normalization when a flip is detected). Evidence: `tmp/v0454_flip_diag.md`
(throwaway), `docs/round2_figs/v0454_flip_diag.png`, JOURNAL 2026-06-17 end-of-day.

---

## Aperture-skirt fix REJECTED; transparency frame-quality gate ADOPTED (default-OFF) (2026-06-17)

**B.1 aperture-skirt - not implemented.** The SNR-optimal science aperture (~5 px) captures only
EE~0.65 of a defocused star (curve-of-growth on bright isolated pre-flip stars), so the premise
"aperture under-captures the skirt" is true. But the decisive test refutes the *fix*: differential
frame-to-frame scatter is **flat** from r=5 px to the EE>=0.99 plateau (~24->27 mmag; minimum ~21 mmag
at r~6 px, within noise), and a per-frame FWHM-adaptive aperture is **worse** (30-32 mmag). The
+-5-7% skirt-fraction swing (~48 mmag nominal) is **common-mode PSF breathing** that differential
photometry already cancels; the bright-star floor is not aperture-limited. Grounded decision: widening
the aperture is not justified (a 5->6 px bump is within noise). Evidence: `CURSOR_RESULT_round2.md`,
`tmp/b1_cog_diag.py`.

**B.2 transparency frame-quality gate - adopted behind a default-OFF flag.** A per-frame
PSF-concentration statistic, median `flux_large / flux` over bright unsaturated sources, cleanly
separates transparency/PSF-collapsed frames (ratio ~11-16, FWHM at the rail, science aperture
catching only noise) from clear-but-faint frames (ratio stays ~2.7 as flux falls equally in both
apertures). Grounded in the curve-of-growth / SNR-optimal aperture framework (Howell 1989). The gate
rejects whole frames whose ratio is a robust outlier (`z>k`, primary) guarded by `FWHM >= factor.median`
(spares sharp frames); safety floor on min kept frames. **Default OFF => baseline byte-identical**
(preserves the UI test). Params `frame_quality_{gate_enabled,ratio_k,fwhm_factor,min_keep_frames}`
(config + ui_settings + VYVAR_PARAMS); `howell1989` gated. Isolated measurement (draft_413 g, ON vs
OFF): LC scatter for bright targets drops **median -257 mmag** (14/15 improved; a flat field star
0.342->0.035 mag); **trust unchanged (RED)** because RED is set by the structural check-star/thin-comp/
colour-term-off gates, not LC scatter - the gate is a precision win, not a trust-flip. Scope: Phase 2A
(target LC/trust); Phase-0+1 comp selection not gated yet. Evidence: `CURSOR_RESULT_round2.md`,
`tmp/b2_measure.py`, `docs/round2_figs/`.

---

## variable_targets selection is spatial-first (frame bbox), and it also purges variables from the comp pool (2026-06-17)

VSX `variable_targets` are selected from the **frame footprint** (frame bbox + the 50 px in-frame
margin) via `_query_vsx_local_frame_bbox`, **not** the 3.5 deg cone box. The cone hit
`catalog_query_max_rows=15000` with **no `ORDER BY`** and silently dropped a contiguous Dec slice
(the northern half of the field, incl. bright named variables - V0454 CrA m9.9, KQ/KM/KT CrA).
Completeness must not depend on row order; the bbox is sub-degree so the SQL result is tiny and the
cap never bites.

**Accepted consequence (Milan, 2026-06-17): this is NOT a pure target-add.** The same
`variable_targets` list drives the **global comparison-pool veto**
(`build_global_comp_pool(..., variable_target_catalog_ids=...)`). The now-complete list correctly
removes newly-recognised variables from the comparison ensemble, which shifts a minority of
previously-measured targets (draft_413 g: 6/19, max |Deltamag|=0.122) - both by dropping a
variable-as-comp directly and by lowering surviving comps' field-wide RMS (the variables had
inflated ensemble scatter) so weights re-rank. This is a comp-purity **improvement**, deterministic,
and accepted as the correct behaviour. Byte-identity vs the capped baseline is therefore *not* a
goal for B. The plan-level `comparison_stars.csv` (the unconditional cone veto in
`write_photometry_plan_files`) is byte-identical old-vs-new; the shift is entirely in the phase-1
global-pool path. Evidence: `CURSOR_RESULT_round1.md`, `tmp/fix2_e2e_oldnew.py`,
`tmp/fix2_mechanism3.py`.

## Completeness gate scores measurable targets, not raw active count (2026-06-17)

`audit_photometry_completeness` (`night_run.py`) is a **false-success / truncation** guard, not a
yield gate. Its verdict is taken against **measurable** targets: a missing target (active, no
summary row) counts as honest *unmeasurable* (does NOT fail) when it is fainter than the achieved
detection depth (faintest measured target's catalog mag); a missing target at-or-above depth is a
*measurable miss* and still fails (the silent-truncation guard - draft_383/385-style cut-short runs
must fail). Depth is derived from the data, so **no new threshold/param** is introduced.
Conservative fallbacks (no mag, nothing measured) treat misses as measurable so truncation can never
masquerade as honest. This unblocks honest RED nights (g 19/22 = 86.4% now PASS) without weakening
the guard. Evidence: `tests/test_completeness_gate_measurable.py` (4/4).

---

## Product scope: light curves in, period science out (2026-06-09)

**VYVAR scope:** produce, validate, and prepare light curves for submission (AAVSO / VarAstro / VSX).
Scientific analysis of those light curves -- period finding, classification -- is **OUT of scope**
and left to downstream tools (Peranso, VStar, Period04).

Internal Lomb-Scargle / BLS use is **not** VYVAR analyzing its own LCs as a science product. It
runs only on:

1. **External TESS cutouts** in `tess_verify.py` -- to confirm a variable-star candidate against an
   independent survey.
2. **Catalog-period display** (VSX / ASAS-SN / ZTF) in the variability UI -- detection/validation
   aids, not folded LC products.

Do **not** expand these into the PDF report as a period product (this descopes TODO-GS9).
Citations `lomb1976` / `scargle1982` / `vanderplas2018` **stay** -- they back the `tess_verify`
TESS cross-check.

## Product scope: single-night is the canonical publishable unit (2026-06-25)

**Decision (Milan).** The unit VYVAR produces, validates, and submits is a **single-night
light curve**. This is the regime cross-validated against SIPS/AIJ (V0612, wide rig) and the
unit a user processes per session. **Multi-night global matching / inter-night zeropoint
(TODO-GS8) is descoped from HIGH to FUTURE / nice-to-have** - built only if a long-baseline
science case requires it. Consistent with the TODO-GS9 (downstream period analysis) scope
boundary: stitching and long-baseline analysis live outside VYVAR's core submission path.

## CAL-DIAG - calibration-time radiometry gate (agreed 2026-07-07)

**Decision (Milan).** Add a camera-agnostic calibration-time diagnostic. Spec:
`docs/VYVAR_CAL_DIAG_SPEC.md` (v1.1, 2026-07-07). VYVAR must verify radiometry from data
when users build masters at arbitrary binning -- not assume per-camera SUM/MEAN conventions.

**Scope (planned):**
- Post-dark-subtraction sky-median sanity (> 0, plausible).
- Pre-subtraction `median(light)` vs `median(resampled dark)` cross-check; loud auto-correction
  (SUM -> MEAN retry) or fail-closed abort.
- Provenance flag `dark_resample` = `SUM` | `MEAN_AUTOCORRECTED` | `PASSTHROUGH`.

**Gap today:** `calibration.py` `get_processed_master` guards geometric resampling only; no
radiometric validation. **F-BINGAIN-1 RN sub-question** (db 7.6 e- scaled x2 -> 15.2 e- if DB
already read-mode-0) resolves here, not via ad-hoc param_resolver exponent change.

**Status:** **IMPLEMENTED (2026-07-07); re-validated 2026-07-14 on HEAD 13341b3.** Spec v1.1
**APPROVED (Milan, 2026-07-14).** Grounding trace `CURSOR_RESULT_caldiag_flow.md`. Result:
`CURSOR_RESULT_cal_diag_impl.md`. **NOT PUSHED** -- pending Milan review of 2026-07-14 closeout.

---

## Calibration masters: manual library build only; no auto-stack on import (2026-07-07)

**Decision (Milan).** When an imported session folder contains raw darks/flats, VYVAR copies
them to `Raw/darks|flats` in the archive but does **NOT** auto-stack them into masters at
import time. Master creation stays **manual via the CalibrationLibrary UI**
(`generate_master_dark_from_source_dir` / `generate_master_flat_from_source_dir`,
importer.py:1483-1608; ui_calibration_library.py:366,398), which registers the result in
`CALIBRATION_LIBRARY` for reuse.

**Why.** The library is the single curated source of validated masters (scoped per
equipment+telescope, validity-windowed). Auto-stacking arbitrary session frames at import
would silently admit unvetted masters (wrong temp match, too few frames, light leaks) into
science calibration -- against the "trust in the numbers" mission. The user builds masters
deliberately, once, and the library serves them everywhere.

**Context.** The product description previously implied auto-build ("folder with
lights/darks/flats creates new masters"); the 2026-07-07 flow trace
(`CURSOR_RESULT_caldiag_flow.md` Q3) showed the code never did this. Milan confirmed the
code behavior as intended -- this entry codifies it so the gap between description and code
does not resurface as a "bug".

## Calibration masters: library precedence over session raw (2026-07-07)

**Decision (Milan).** When a session provides raw darks/flats AND a valid scoped library
master exists for the same obs_group, the **library master wins** for calibration
(importer.py:1438-1439). Session raw frames are archived under `Raw/darks|flats` as
provenance, not stacked or preferred.

**Why.** A registered library master is a known quantity: scoped to the equipment set,
within its validity window (90 d dark / 200 d flat), built from a deliberate stack. A
handful of session raw frames is not automatically better -- and choosing it silently would
make calibration depend on what happened to be in the source folder. Deterministic
precedence beats per-session guessing. The user who wants session-specific masters builds
them via the UI (previous decision), which then wins the scoped lookup as the freshest
valid entry.

## READNOISE_E is per-pixel at bin1; RN_eff = RN_db * bin is correct for software binning (2026-07-07)

**Decision.** The DB `EQUIPMENTS.READNOISE_E` value is defined as the **per-pixel read noise
at bin1** (native read mode). The EQUIP-BINNING scaling `RN_eff = RN_db * binning`
(exponent 1; `_scale_bin1_to_binning`, param_resolver.py:154-159, applied in
`resolve_read_noise` param_resolver.py:493-498) is **physically correct** for
software/digitally binned CMOS: a bin-b superpixel sums b^2 independent pixel reads, so
read noise adds in quadrature, sigma = RN_px * sqrt(b^2) = RN_px * b. For the QHY294MM,
7.6 e- (per-pixel, per spec sheet) -> 15.2 e- effective at bin2. There is **no
double-count** under this semantic; a double-count would require the DB to store an
already-binned effective value, which nothing in code or docs claims.

**Closes the F-BINGAIN-1 RN sub-question directionally.** Final empirical closure = photon
transfer on **bin2 flats** (Milan data item; the 2026-07-07 field-light attempt was
inconclusive, g_eff~0.9). No param_resolver exponent change; the guard is CAL-DIAG's
data-driven radiometry gate, not per-camera constants.

**Implementation note:** semantic comment at param_resolver.py:155 and in the
`set_equipment_cosmic_params` docstring (database.py:2928), landed with the CAL-DIAG PR.

---

## Photometry method & scale

### Plate scale is WCS-derived (~ 9.77 arcsec/px on the wide rig), not 1.3
The project-wide `1.3 arcsec/px` belief was wrong - it was a Newton 300/1200 + C3-26000 (binned 2x)
placeholder leaking onto the wide Carl-Zeiss/QHY294MM field via a global config default. The
resolver is **solved WCS/CD authoritative -> config only as last resort** (sane clamp widened
to `0.1-30.0`). WCS-dependent geometry (ePSF isolation, FOV / `max_dist_deg`, TESS context) is
now correct; pixel-based geometry (aperture / annulus / SNR-optimal table / field density) was
immune. **Status: settled (2026-05-29/30).** Any residual `1.3` in old `pipeline_meta` is stale
run metadata, overwritten on a clean re-run (see ROADMAP: WIDE-RIG-REPROCESS).

### Brno / external data: characterize before PSF or NEIGHBOR-SUB (2026-06-08)

Before relying on PSF or NEIGHBOR-SUB for publishable output on incoming Brno University data (or any
new field), run the standard characterization gate: plate scale + pixel sampling, ePSF-vs-star Moffat
mismatch (decisive), and crowding (`compute_crowding_index`). NEIGHBOR-SUB is **validated at fine
scale** (draft 367: mismatch ~1.0, A9 HV ~83%, FAIL-SILENT 0). If new data is **coarse or
under-sampled** (mismatch > ~3%), it falls back to the **SAFE_LOW_YIELD** regime -- bright-neighbour
blends will correctly **REFUSE**, not be silently deblended. That is the publishable-safe behaviour;
do not force deblending outside the validated regime.

### PSF fit weights: sky + read noise only (2026-06-09)

Mid-mag PSF bias on V3d (+4.5%) was **flux-dependent weighting**: including source Poisson in
photutils fit `error` makes relative pixel weights depend on brightness, so the bright/faint flux
ratio becomes PSF-model-dependent and biases point-source fluxes (Astier et al. 2013; Lacroix /
Regnault 2025). Production fix: `psf_weight_mode=sky_only` -- one estimator for all magnitudes,
uniform per-stamp sigma from sky + read noise only. Accepted small bright-end precision cost vs
object-weighted fits. Forced position (Guy et al. 2010) not required at fine scale after Fix 1.
Residual hardware systematics (brighter-fatter / pocket effect; Lacroix 2025) remain out of scope.

### Sandwich reported PSF uncertainty (2026-06-09)

With sky-only fit weights, reported `psf_flux_err` must propagate **true pixel variance**
(sky + source Poisson + read noise) through the **actual** weights used in the fit
(`psf_err_mode=sandwich_skyonly`). This calibrates error bars (V3d P3 ~1 mag<=17) without
changing fluxes. Stetson 1987 / Mighell 1999 cited for ensemble context; sandwich is the
production implementation for per-star PSF errors.

### EPSF-1 robust FWHM QC (2026-06-08, diagnostic only)

`epsf_fwhm_native` uses an azimuthally-binned radial profile with linear 0.5 crossing (not
first-pixel half-max). QC warning band tightened to **[0.80, 1.25]**. Does not enter flux path
or `assess_psf_quality`; numeric SHA unchanged. Validated via harness V3e.

### NEIGHBOR-SUB shape: PSF subtract contaminant, aperture measure target (2026-06-09)

For blended targets, use ePSF to fit and subtract a bright neighbour, then run the existing
aperture path on the residual stamp. PSF does not replace aperture for science flux; it only
removes contamination. Does not revive grouped PSF / rule 2 (mutually exclusive thresholds).
Gated `psf_neighbor_sub_enabled` OFF. Synthetic validation: **VALIDATED_FINE_SCALE_IDLE**
(A9 HV ~83%, FAIL-SILENT 0 on draft 367). Real-field enablement after Brno characterization.
Full design: `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md`.

### Fail-safety hygiene #4 (2026-06-08, Milan confirmed)

**MASTERSTAR WCS persist (`vyvar_platesolver.py`):** failed `fits.writeto` (or read/update) on
MASTERSTAR is **fail-closed** for that draft -- returns `solved=False` via `_SolveWcsWriteError`,
`LOGGER.error`, pipeline blocks Phase 2A (no silent stale WCS). Other frames in the batch are
unaffected.

**Edge-ok filter (`photometry_core._edge_ok_from_masterstar_pipeline`):** on check failure,
**fail-open** (all stars treated edge-ok so detection is not zeroed) but **loud** --
`edge_filter_failed=True` + `edge_filter_note` on `variability_candidates.csv` only (not on
byte-identity SHA files: `lightcurve_*.csv`, `comp_quality_*.json`,
`comparison_stars_per_target.csv`). Report cover shows edge-filter status when flagged.

**Dead UI modules:** `ui_photometry_results.py` and `ui_suspected_lightcurves.py` deleted;
function covered by `ui_aperture_photometry` + `ui_variability`.

### Crowding + ePSF FWHM context use measured core (VY_FWHM_GAUSS), not DAO search scale (2026-06-09)
`VY_FWHM` on MASTERSTAR is the DAOStarFinder search parameter (~3.4-3.8 px on h & chi Per L);
`VY_FWHM_GAUSS` is the 2D Gaussian core fit (~2.7 px) already used by aperture photometry
(`pipeline.py:9206`). Crowding (`_load_wcs_meta`) and ePSF build context
(`get_epsf_fwhm_from_context`) previously read `VY_FWHM` only, inflating blend disks and
deflating ePSF QC ratios. **Decision:** shared `header_core_fwhm_px` prefers
`VY_FWHM_GAUSS` -> `VY_FWHM_GAUSSIAN` -> `VY_FWHM` at exactly those two sites; display-only
and plate-solve hint readers keep `VY_FWHM`. Validated: numeric SHA `770966c3...` unchanged;
h & chi Per L crowding 77/87 -> 58/53 is_blended.

### Aperture is the validated workhorse; PSF validated-but-gated
At 9.77 arcsec/px the PSF is well-sampled and stable across the field, so a single ePSF already
captures it and aperture wins. Every PSF variant - single ePSF, spatial `GriddedPSFModel`,
`SourceGrouper` joint fit, per-star adaptive selector - was implemented and **lost to aperture**
at this scale (single ePSF ~3x worse comp RMS; grid starves cells; grouper diverges on
sub-resolution blends). **Decision: keep all PSF flags OFF in production on the wide rig.**
On fine-scale synthetic truth (draft-367-like), PSF is now **publication-grade** (accuracy,
precision, sandwich P3) but remains **gated OFF** until real Newton / Brno data passes the
characterization gate. **Status: settled on wide; fine-scale validated-but-gated.**

### Full-frame DAO over fixed-position stamp photometry (overnight-batch model)
Every frame runs full-frame `DAOStarFinder` + match (the "master fast path" skips the per-frame
Gaia cone, not the detection). The SIPS-style speedup considered 2026-04-22 - read flux only at
fixed catalog positions, no full-frame finder - is **deliberately not adopted.** Full-frame
detection buys QC that fixed positions lose: per-frame local centroiding (absorbs WCS / drift /
resampling offsets), shape-roundness rejection of cosmics, hot pixels and CCD columns,
match-count sanity against a bad solve, new-source detection, and completeness diagnostics. The
compute cost is accepted because **VYVAR runs as an overnight batch** after the session ends,
while the observer sleeps - the binding constraint is a *trustworthy* Summary Measure Report by
morning, not wall-clock throughput. The same logic licenses PSF's extra cost when it is enabled
on fine-scale data: accuracy and robustness over speed. **Corollary:** performance work is
welcome only where it does not trade away QC (I/O, parallelism, caching); the per-frame
full-frame detection itself is a feature, not a bottleneck to remove. **Status: settled
2026-06-02.** (See also: SIPS comparison below.)

### Per-frame catalog: drop unmatched DAO before aperture; Moffat gated on ePSF (2026-06-12)
**DAO detection stays full-frame** (QC unchanged). After `detect_stars_match_master_reference`,
rows with empty `catalog_id` are dropped **before** aperture / Moffat / PSF work
(`_proc_drop_unmatched_dao_rows`; key on `catalog_id`, not `source_type`). They were never written
to `proc_*.csv` anyway (final `_proc_catalog_keep_matched_rows_only`); this is pure wasted compute.

**Moffat fit is Step 1 of the two-step ePSF path only** - gate `if _run_epsf:` (not
`_run_aperture`). In aperture-only production (`psf_photometry_enabled=False`), `moffat_*` columns
are omitted; LC / comp-QA readers do not consume them. **LC byte-identical** when `VY_FWHM` /
`VY_FWHM_GAUSS` drives aperture radius (verified `draft_000389` B_60_1). Chi_and_H photometry SHA
unchanged (`proc_*.csv` not in SHA set).

### What VYVAR deliberately does NOT adopt from SIPS (2026-06-02)
Comparison against SIPS (Moravian Instruments; v4.4 manual + Pejcha & Cagas 2022, A&A 667,
A53) confirmed the two tools share a photometric **family** - both do full-frame per-frame star
detection, intensity-weighted sub-pixel centroids, per-star automatic apertures, robust
background, saturation invalidation, and a flux-summed ensemble. SIPS's speed comes from
**native C/C++ + multicore**, not an algorithmic shortcut - which *validates* VYVAR's full-frame
DAO choice rather than contradicting it.

**Deliberately not adopted:**
- **Fixed-position stamp photometry** (read flux at catalog x,y only; no per-frame finder). See
  the full-frame DAO decision above - QC and trust over throughput.
- **Neural-network variable detection (VDI/NN).** Against the trust mission - VYVAR keeps
  explainable statistics (Sokolovsky indices, RMS hockey-stick, independent cross-validation),
  not a black box.
- **UCAC4-based calibration.** A step back from VYVAR's native Gaia DR3 + BP-RP (deeper, modern,
  colour-complete).
- **SPL scripting / REST API.** Different stack; VYVAR is Python/Streamlit.

**Worth borrowing - scoped in ROADMAP, not core changes:**
- **Wide-field WCS distortion check** (MEDIUM): confirm SIP / higher-order terms on the wide
  rig; SIPS uses a 3rd-order 2D polynomial (Monomial/Legendre).
- **Spatial term in ensemble calibration** (LOW): SIPS's `x1.X + y1.Y + ...` field-gradient terms
  - relevant only for a future whole-field absolute mode, not the current per-target differential
  path.

The Pejcha & Cagas paper is a **citation / positioning reference for the GS7 paper**, not a
source to copy. Where VYVAR is already ahead: Gaia-native catalog, independent cross-val +
per-target trust gate, comp_qa, literature-backed comp selection, and reproducibility/citation
discipline. **Status: settled 2026-06-02.**

### What VYVAR does / does not adopt from CoLiTecVS (2026-06-09)
Comparison in the same spirit as the SIPS entry. Sources: Savanevych/Briukhovetskyi/Khlamov/
Kudzej/Dubovsky/Parimucha et al. -- Astron. Nachr. 2019;340:68-70; CAOSP 49,151 (2019);
2022A&C....4000605S; Dubovsky et al. 2017 OEJV 180 (inverse-median-filter detail). CoLiTecVS is
from the same community Milan works in (Kolonica Saddle / Vihorlat Obs. / UPJS Kosice).

**Shared photometric family.** Both are automated differential aperture-photometry pipelines
that take raw frames to AAVSO-style light curves with ensemble comparison stars and minimal
manual step-by-step handling. Both were validated against the C-Munipack / Muniwin class of
tools and reach comparable scatter (CoLiTecVS aperture uncertainty < 0.04 mag; on the MASTER OT
J174305 field CoLiTecVS auto-ensemble SD ~0.0078 vs C-Munipack+MCV ~0.0067, i.e. parity).
Same problem domain, same accuracy class.

**Where VYVAR is ahead:**
- **Gaia DR3-native + colour.** CoLiTecVS selects comparison stars from AAVSO charts (LookSky
  tool). VYVAR is Gaia DR3-native with BP-RP colour, colour-term handling, and colour-aware comp
  selection -- a more modern catalog/colour basis.
- **Independent QA + per-target trust verdict.** CoLiTecVS reports a global aperture uncertainty
  and validates in aggregate (one mean/SD table); it exposes no per-target machine-checkable
  verdict and no independent second-extractor cross-check in-product. VYVAR adds comp_qa
  (Sokolovsky leave-one-out locus), a **production per-target trust gate** (comp health +
  check-star scatter + `lc_quality_flag`; see Trust & validation below), and **offline-only**
  SEP cross-validation (~0.2%/frame via `xval_run.py` - validation study, not a pipeline trust axis).
- **Reproducibility / provenance.** VYVAR has SHA-256 byte-identity on photometry artifacts, a
  citation emitter, and a decision log. CoLiTecVS is a compact closed all-in-one -- by the
  authors' own note you cannot isolate and test a single internal stage.
- **Modularity / auditability.** VYVAR stages are individually inspectable and read-only
  auditable; CoLiTecVS is monolithic (raw -> LC, turnkey).

**Where CoLiTecVS is ahead (VYVAR gap):**
- **Inverse-median-filter brightness equalization** (their signature). Removes large-scale
  illumination non-uniformities from Moonlight / scattered light that dark+flat do NOT correct.
  The authors report it usually beats classical flat-field for background equalization, with no
  measurable photometric non-linearity (Dubovsky et al. 2017). VYVAR is flat-only and has NO
  scattered-light / large-scale gradient equalization stage. Real gap for moonlit nights and
  light-polluted sites.
- **Online / real-time mode (OLDAS-Night):** processes data live off the sensor. VYVAR is offline
  batch only.
- **Maturity / proven scale:** CoLiTec lineage (700k+ observations in the asteroid-detection
  heritage); CoLiTecVS tested on 100+ time series (20-600 frames) and in regular operational use
  at Kolonica. VYVAR is a single-observer pipeline.
- **LookSky one-click AAVSO-chart comp selection + reusable per-target task-file** (slick repeat-
  target UX). Workflow convenience, not necessarily better science.
- **Compact turnkey UX** (minimal interaction). VYVAR needs more setup and understanding.

**Worth borrowing -- scoped in ROADMAP, not core changes:**
- **PRIMARY: optional large-scale illumination-gradient removal** (inverse-median-filter or
  equivalent background equalization), pre-photometry, OFF by default and gated. Directly addresses
  the one real capability gap above; ties to existing items TODO-LC-TREND (differential extinction /
  moonless-night note) and the LOW SIPS "spatial term". VYVAR can adopt it more safely than
  CoLiTecVS validated it: byte-identity SHA, comp_qa locus, SEP cross-val, and check-star scatter
  are the acceptance harness -- enable on a moonlit/gradient draft, confirm locus + check-star
  improve (or are unchanged) and constant-star differential RMS does not degrade, with numeric SHA
  tracked as a separate baseline. **Risks to gate:** median-background subtraction can remove real
  extended flux and perturb faint-star annulus estimation; must stay optional, never silently alter
  the photometry path, and pass the trust-gate acceptance before default-on (mirrors PSF "OFF until
  validated" discipline).
- **SECONDARY (optional, UX):** reusable per-target "task-file" (fixed target + comp set reused
  every reduction). Only if VYVAR lacks an equivalent per-target config; low priority, workflow not
  science.

**Deliberately not adopted:**
- **Monolithic compact architecture** -- VYVAR's modular, auditable design is a deliberate strength.
- **Online OLDAS mode** -- out of scope for VYVAR's offline-rigor model.
- **AAVSO-chart comp selection** -- VYVAR's Gaia DR3 + BP-RP colour basis is more modern;
  switching to chart-based selection would be a step back.

**Net:** borrow ONE idea (optional gated illumination-gradient equalization), validated through
VYVAR's existing trust harness; keep everything else as VYVAR already does it. **Status: settled
2026-06-09.**

### COG (curve-of-growth) aperture correction: implemented, default OFF
Per-frame encircled-energy correction removes the constant target<->comp enclosed-flux bias and
the seeing-correlated systematic from per-star SNR-optimal radii. Byte-identical when OFF.
**Decision: ship gated, leave OFF** until validation on real nights. Mixed-frame guard is
wired (see `APCORR-MIXEDFRAME-ALLORNOTHING` below).

### APCORR-MIXEDFRAME-ALLORNOTHING (2026-07-19)
**Context.** With `cog_aperture_correction_enabled=True`, a night that only partially succeeds
at COG (`cog_ok=False` on some frames) would otherwise route corrected flux on some epochs and
raw `dao_flux` on others -> a step in the light curve that is not astrophysical.

**Decision.** Night-level all-or-nothing gate after per-frame COG AC computation: if **any**
science frame of the night lacks a usable correction (`cog_ok=False` or missing `cog_ok`),
COG application is **disabled for the entire night** - every Phase 2A row takes the standard
Metoda B AC chain. Log:
`[APCORR] COG night fallback: N/M frames without cog_ok -> whole night uses standard AC`.
Provenance: `cog_night_fallback=true` (plus counts) in `photometry/pipeline_meta.json`.
Per-frame `fallback_ee` remains a FUTURE refinement (noted on ROADMAP closure).

**Status:** implemented; COG still default OFF pending enablement validation.

### SKIPPROC-PERMANENT (2026-07-22)
``skip_processed_directory`` removed; in-place QC + ``qc_metrics.csv`` allowlist is the only
preprocess path. The ``processed/lights`` copy tree is retired (it existed only as a check
artifact; allowlist supersedes). Resume/skip-drafts: preprocess complete when
``calibrated/lights/qc_metrics.csv`` exists (or ``pipeline_meta`` preprocess stage stamp).
**Status:** implemented.

### SKY-SURFACE-RESTORE (2026-07-27)
Accidental drop of order-2 sky-surface subtract on mono frames in `013cb0c` (SKIPPROC) is
**reversed**. This is the **second occurrence** of F-431-HEADLESS-DIVERGENCE (lost preprocess
ADU step); the first closure (2026-07-16) did not add a guard, so SKIPPROC could remove the step
again without tripping any gate. T3-PREPROCESS-SKY-SURFACE (2026-07-16) remains authoritative: when
``preprocess_sky_surface_order > 0``, ``_qc_enrich_calibrated_in_place`` calls
``_fit_subtract_preprocess_sky_surface`` for all non-mosaic lights (no ``VY_CHANNEL`` gate);
OSC Bayer mosaics are still skipped to avoid double subtract on channels. Headers:
``VY_SKYSF``, ``VYSKYORD``, ``VYSKYP2P``. Guards: **INV-PREP-01** (large_small_ratio WARN on
calibrated-frame gradient; detects missing or ineffective sky-surface subtract from infolog),
**INV-MS-01** (DAO_ONLY fraction WARN/FAIL). C.4 acceptance on BO CVn raw path pending Milan
raw frames on disk. **Status:** code landed; science validation pending.

### CONFIG-PATH-DATA-ROOT (2026-07-27)

**Problem.** Config path-valued keys stored as relative strings (e.g.
``exoplanets/vyvar_exoplanet_local.db``, ``VSX/vyvar_vsx_local_v2.db``) were resolved with
``Path.resolve()`` against the **process CWD**, not ``data_root``. When night runs or Streamlit
sessions set CWD under ``Archive/Drafts/`` (common since ``c99fcec`` data-dir bootstrap), a
configured file under ``<data_root>/`` was missed; VSX/exoplanet queries returned empty and the
pipeline reported success (draft 450, 2026-07-24).

**Trigger.** Latent CWD-relative resolution predates the regression; first production symptom on
draft 450 coincides with ``c99fcec`` (2026-07-24) separating ``data_root`` from install root while
night-run CWD stayed under ``Archive/Drafts/``. ``3c31bfa`` materialized relative paths in
``config.json`` but did not change resolution semantics.

**Rule.** One resolution function for all config paths: ``resolve_config_path(raw, data_root)`` in
``config.py`` -- absolute paths as-is, relative paths against ``data_root``, never CWD. Catalog DB
open paths use ``database._resolve_catalog_db_path`` + ``require_*_db_path`` helpers; a configured
path that does not exist or has zero rows (VSX) **must fail loud** with the config key name. Guard:
``dev/tests/test_catalog_db_path_resolution.py``. **Status:** implemented (POST-451 closeout).

### PHASE0-ACTIVE-COUNT-VT-CONTEXT (2026-07-27)

**Problem.** POST-451 C.4 acceptance table listed active targets **~160-175** by extrapolating from
the anchor's 165 actives. That band was wrong: anchor actives come from a **frozen**
``variable_targets.csv`` built when ``vsx_variable_targets_mag_limit = 14.5`` still existed (245
VT rows, faintest G=14.50). The live full-path VT has **no mag limit** (873 VSX rows). Comparing
live active counts to anchor without stating the VT context repeats the apples-to-oranges error
made earlier with the ``~283`` masterstars target.

**Verified decomposition (draft_452, live VT):**

- VT rows with G <= 14.5: **243** (vs 245 on frozen anchor VT).
- Active total **201** = Group A **163** (shared with anchor 165, minus 2 zone/skip) + Group B
  **38** (live-path-only actives).
- Group B actives carry ``gaia_match_source`` **masterstars** (37) or **masterstars_exo** (1) -
  not ``gaia_dr3_direct`` (Phase 0 rejects ``gaia_dr3_direct`` as ``not_target_eligible``).

**Rule.** When citing anchor active count (165) as a regression target for full-path runs, always
state the frozen VT / mag-limit context. Do not treat 165 as the expected live-path active count
without that caveat. **Status:** recorded; no code change.

### DETECTION-DEPTH-VS-LC-USABILITY (2026-07-27)

**Principle.** A star detectable on a single stacked MASTERSTAR frame at the DAO threshold is
**not** guaranteed to yield a scientifically usable differential light curve across the series.
Group B on draft_452 (38 actives): median ``lc_rms`` **0.398 mag**, **20** flagged RED - real
variables, genuinely detected, correctly identified, but not submission-grade at this rig and
exposure.

**Rule.** No magnitude cut, no target limit, no new parameter. The existing trust system (RED /
``lc_quality_flag``) is the correct handling. Do not reopen as a pipeline defect when faint
actives appear with poor curves. **Status:** recorded; no code change.

### GUARD-HEADLESS-OBSERVABILITY (2026-07-27)

**Problem.** ``INV-PREP-01`` and ``INV-MS-01`` called ``log_event()`` only, which writes to the
in-memory Infolog ring buffer. Headless ``run_night_pipeline`` did not mirror Part B's
``logging.info`` + ``log_event`` dual path; guard output never reached night-run stdout or
``infolog_*.txt`` on disk.

**Fix.** Guards emit ``LOGGER.info`` + ``log_event`` (same dual path as Part B VSX-GAIA / FAZA 0
funnel). ``log_milestone()`` helper added to ``infolog.py`` for future Cython rebuild. Headless
``run_night_pipeline`` calls ``ensure_infolog_logging()`` at start and ``save_infolog_to_disk()``
on success. Tests: ``test_inv_ms01_milestone_reaches_headless_logger``,
``test_inv_prep01_milestone_reaches_headless_logger``. **Status:** implemented.

### QC-ALLOWLIST-AUTHORITY
``qc_metrics.csv`` under the draft lights root is the **authoritative frame allowlist** for
alignment: only rows with ``status=ok`` (exact match) enter ``astrometry_align_and_build_masterstar``.
Frames on disk but missing from the CSV, or with any non-ok status, are excluded; a missing CSV is
fail-closed. ``VY_QC`` FITS headers remain **diagnostic only** (including prefilter stamps such as
``rejected_prefilter_fwhm``); they are not alignment gates. In-place QC visits **every** light
frame and records segmentation FWHM/elongation as diagnostics; it does **not** self-reject on
segmentation FWHM. Frame selection authority is the DB DAO-FWHM prefilter (Analyze Auto FWHM limit)
feeding ``prefilter_rejected`` status rows. Rationale: ``dev/results/CURSOR_RESULT_skipproc_qc_leak.md``.
Wired gate: **QC-01**. **Status:** unconditional since SKIPPROC-PERMANENT.

### VSX-AUTO-MAGLIM (2026-07-22)
``vsx_variable_targets_mag_limit`` removed. VSX rows enter ``variable_targets.csv`` from the
frame bbox query without a static ``mag_max`` pre-filter; **measurement scope** remains
detection-limited: a VSX star is active only with a DAO detection cross-matched to Gaia on
MASTERSTAR (unchanged Phase 0 criterion). Rationale: catalog ``mag_max`` band heterogeneity
made the static threshold semantically fuzzy; detection limit is the honest gate. Comp-pool
veto coverage grows with the fuller ``variable_targets`` list (correctness). **Status:** implemented.

### VSX-GAIA-MATCHER-TWO-STEP (2026-07-26, PHASE0-IDENTITY-GATE / MATCHER-FIX)

**Decision.** Plan-time cross-match is a **two-step separation** (design A):

1. **VSX -> Gaia DR3** over the frame bbox (deep local catalogue via ``query_local_gaia``;
   same geometry as ``_query_vsx_local_frame_bbox``). Measured ``rho`` sets acceptance radius
   ``r_max = sqrt(0.01 / (pi * rho))`` (1% contamination budget). Mixture fit yields ``Q``, ``w``,
   ``sigma_narrow``, ``sigma_broad`` for **ranking** multi-candidate cases; Sutherland reliability
   tiers are quality indicators only (not accept/reject gates).
2. **Phase 0** promotes only when the accepted Gaia ``source_id`` is present in
   ``masterstars_full_match.csv`` (``gaia_match_source=masterstars``). Matches without DAO
   detection remain in ``variable_targets.csv`` as ``gaia_dr3_direct`` (comparison-pool veto only).

**Why not masterstars as match RHS.** Masterstars are detection-limited (``Q ~ 0.32`` on anchor
night); fitting nearest-neighbour separations against them locks ``sigma`` onto the chance scale
(~100 arcsec), not astrometry. ``Q`` is a **fitted** mixture parameter (Sutherland & Saunders),
not assumed 1.

**True-match separation model (MATCHER-FIX-2).** A single Rayleigh is insufficient for VSX: the
catalogue is heterogeneous (774/873 in-frame entries on the anchor night are Gaia-identified
names with near-zero true separation; classical entries carry mixed epoch/quality astrometry),
and Gaia DR3 (epoch 2016.0) vs VSX (generally J2000) introduces a proper-motion tail to ~1-2 arcsec.
The true-match term is therefore a **two-component Rayleigh mixture** (``w``, ``sigma_narrow``,
``sigma_broad``) plus measured ``rho`` and fitted ``Q``. The <=1% contamination gate
(``mean(rho * pi * (sep/3600)^2)``) was **never relaxed** in this arc.

**Acceptance rule (MATCHER-FIX-3).** Acceptance is defined solely by the pre-registered 1%
contamination budget, not by a reliability threshold. The acceptance radius follows from measured
field density:

``r_max = sqrt(0.01 / (pi * rho))``  [``r`` in deg; ``rho`` in deg^-2]

A Gaia source is a **candidate** when it lies within ``r_max`` of the VSX position. Exactly one
candidate is accepted; when several lie inside ``r_max``, the mixture-based Sutherland likelihood
ratio **ranks** them and the best is taken. Reliability is still computed per row for
``gaia_match_quality`` tiers but **must not reject** a positional candidate. This mirrors the Gaia
DR3 cross-match design: the figure of merit ranks neighbours; positional compatibility gates them.
An arbitrary FoM cutoff was explicitly avoided by the Gaia authors because it would be arbitrary.

**PM path.** When ``pmra``/``pmdec`` columns are present and finite in the local Gaia DB,
positions propagate from ``GAIA_EPOCH=2016.0`` to ``VSX_MATCH_EPOCH=2000.0`` (J2000 assumption).
When absent, ``pm_path=broadened`` with a small quadrature term; separation quantiles before/after
are logged.

**Degeneracy guard.** ``sigma_broad > 0.25/sqrt(rho)`` [deg] plus astrometric-tail checks on the
sub-10 arcsec core (chance-scale p50 lock-on). At plan-time match, degeneracy is **WARN** only
(ranking may degrade; acceptance via ``r_max`` is unaffected). Strict FAIL remains in unit tests
on direct mixture fit.

**Outcome check (G3).** WARN when ``masterstars_accepted / masterstars_eligible < 80%``, where
``masterstars_eligible`` counts VSX rows with any DAO ``source_id`` candidate within ``r_max``.

**Comparison baselines (plan regen).** Withdrawn: frozen 245-row anchor VT histogram
``masterstars=178, gaia_dr3_direct=64, no_match=1, masterstars_exo=2`` (14.5 mag limit era).
Correct like-for-like: ``draft_000450`` regen on same night/field without mag limit:
``873 rows | masterstars 283 | gaia_dr3_direct 443 | no_match 147``.

**Status:** implemented (uncommitted pending Milan STOP clearance).

### OSC-CHANNEL-EXTRACTION (2026-07-22, phase 1/3)
OSC Bayer mosaics: calibrate on CFA (dark subtract + flat divide on raw mosaic; flat via
``normalize_flat_master`` per-tile using ``EQUIPMENTS.BAYERMASK``). **No** demosaic/interpolation.
After calibration, plane-split extraction to four obs-groups (``oneRGGB``, ``R``, ``G``, ``B``) with
AVERAGE semantics; optional ``osc_channel_binning`` NxN average post-extraction. Sky-surface fit runs
**only** on extracted channels, never on the mosaic. ``oneRGGB`` is internal/diagnostics only (not
AAVSO). Effective gain/RN stamped per channel (``VY_EGAIN``, ``VY_RDNOIS``). Wired gate **OSC-01**
(no mosaic without ``VY_CHANNEL`` in alignment). Literature: AAVSO DSLR Observing Manual (channel
separation); Carrasco/Riello Johnson transforms deferred to OSC-3.
**Status:** phase 1 implemented; see OSC-WCS-SOLVE-ONCE for phase 2; OSC-3 (band mapping,
TG/TB/TR exports) queued.

### OSC-WCS-SOLVE-ONCE (2026-07-22, phase 2/3)
OSC multi-band drafts: **unified frame set** across oneRGGB/R/G/B (QC verdict replicated from
oneRGGB with per-channel diagnostics preserved; ``qc_source=oneRGGB``). Plate-solve runs **once**
on the oneRGGB MASTERSTAR; WCS is propagated to R/G/B masterstars and aligned frames. Registration
transforms are computed on oneRGGB only and stored in ``platesolve/<oneRGGB>/osc_registration_handoff.json``
for verbatim reuse on sibling channel frames (D2 artifact). Channel MASTERSTAR photometry (DAO
catalog, comp selection) uses each channel's own pixels; only geometry/WCS is shared (D4). Per-channel
DAO-Gaia match rate logged after propagation; below ``MASTERSTAR_PLATESOLVE_MIN_MATCH_RATE`` (0.60,
same gate as ``generate_masterstar_and_catalog``) -> WARN with channel numbers; FAIL only if
oneRGGB itself failed (B channel fewer stars expected). Band tokens: R->TR, G->TG, B->TB,
oneRGGB->CLEAR (k'' none-token path). Known systematic: Bayer channel
planes sit at sub-cell offsets (<=0.5 raw px = 0.25 superpixel); accepted for catalog crossmatch
tolerances - no per-channel WCS micro-shift. Wired gate **OSC-02** (identical frame ID sets).
**Status:** phase 2 implemented; see OSC-BAND-EXPORT for phase 3.

### OSC-BAND-EXPORT (2026-07-22, phase 3/3)

E1. AAVSO band codes for OSC DSLR tri-color (untransformed): R->TR, G->TG, B->TB per
AAVSO practice. ``oneRGGB`` is **internal-only** (LC + report diagnostics); hard excluded
from AAVSO and VarAstro exports (mixed-channel mags undefined).

E2. Comp/check catalog magnitudes per band: TG ensemble reduces against Johnson V comps,
TB against B, TR against Cousins R_C (AAVSO comp-mag rule).

E3. Gaia DR3 G + BP-RP -> Johnson/Cousins via ``src_py/gaia_johnson.py``. **Coefficient
source:** Gaia DR3 documentation CU5 Table 5.9 (GBP-GRP polynomials for G-V, G-B, G-R;
sigma column 0.03017 / 0.0633 / 0.03167 mag). **Validation reference:** Ruelas-Mayorga
et al. 2025 RASTI 4:37 (doi:10.1093/rasti/rzaf037; Landolt-standard independent fit,
Crossref-verified author list). Validity 8<G<16, BP-RP in pinned range. Out-of-validity
comps excluded with logged reason; Table 5.9 sigma added in quadrature. Riello et al.
2021 EDR3 Table C.2-style fallback in unit tests only.

E4. Jordi B-V legacy path (mono k2 slope) **unchanged**; ``gaia_johnson`` consumed only by
OSC band exports in this phase.

E5. No new config parameters. Wired gate **OSC-03** (no oneRGGB in export files).

**Status:** phase 3 implemented; M71 E2E acceptance pending Milan.

## Comp-star selection & QA

### Selection priority: stability > colour > proximity; proximity is a GATE, not a rank
- **Stability** is enforced as a hard `max_comp_rms` cap + iterative MAD filter, and as the
  Broeg (2005) `1/sigma^2` ensemble weight - which dominates the actual measurement.
- **Colour** (`|DeltaBP-RP|`) is a hard cut (<= 0.79) + tier, and is the **primary pick sort key**.
  For **NoFilter/broadband** this is justified: without a filter the colour term is a
  *first-order* systematic (no bandpass to cancel it), and colour systematics don't average
  down like random scatter. Grounded in Henden & Kaitchuck (1982) and the AAVSO CCD Guide.
- **Proximity** is enforced only as the distance gate (`max_dist_deg` ~ 1.5 deg, `min_dist_arcsec`
  60 arcsec). A proximity ranking tie-break was implemented and **reverted**: because Broeg `1/sigma^2`
  weights are order-independent, any RMS-bin tie-break necessarily trades stability for
  proximity, violating the agreed order. **Status: settled 2026-06-02** (revert restored the
  comp set byte-for-byte).

### comp_qa - Sokolovsky leave-one-out self-consistency QA
Per-comp variability indices on the zero-median LOO differential light curve: `sigma_IQR`
(amplitude), `1/eta` von Neumann (slow drift), `spike` (dropouts), flagged against a
**magnitude-dependent locus** (0.5-mag bins) rather than a flat floor. Grounded in Broeg (2005)
+ Sokolovsky et al. (2017) + Howell (1989). The flat-floor v1 over-flagged faint comps; the
locus version is correct. **Status: productionized 2026-06-02** as a read-only post-Phase-2A
stage (photometry byte-identical); outputs per-comp flags + per-target `n_clean`.

### Crowding classifier: gated infra, NOT enabled on the wide rig
Detection-independent signals (Gaia density, `blend_frac` at depth, comp availability) replace
the erratic stars/Mpx class; LOOSEN on comp scarcity, TIGHTEN on real blend fraction, with a
**sampling gate (FWHM >= 3 px)**. The wide rig is floor-limited (scintillation/undersampling at
FWHM ~ 2.6 px), so tightening there only cuts good comps. **Decision: keep OFF for the wide rig;
enable on the well-sampled Newton cluster.**

## Trust & validation

### Production trust gate - hard/soft model, inform-only (v1)

Per-target GREEN/YELLOW/RED from **comp-set quality** (`n_clean` from comp_qa Sokolovsky LOO),
**check-star scatter**, and **`lc_quality_flag`**. **No SEP / xval axis in production** - SEP vs
DAO cross-validation is an offline harness only (`xval_run.py`, `assign_sep_confidence` in
`xval_harness_core.py`); see Cross-validation CLOSED below.

**Hard** warnings (real red flags: `n_clean == 0`, no check star, bad `lc_quality` e.g. saturated,
check >= 0.05) vs **soft** (thin comps, sparse_fallback, marginal check 0.02-0.05, short_baseline).
RED = any hard OR >= 3 escalating soft; YELLOW = any soft; GREEN = `n_clean >= strong` + check OK +
no warnings. Thresholds derive from `comp_trust_min_comps` / `phase01_comparison_n_comp_max`.
**Unevaluated targets -> RED** (fail-closed). **`lc_quality="noisy"` is informational only** -
variability-driven (counting it as a hard warning wrongly demoted real variables). **v1 is
inform-only** (RED is surfaced, not auto-dropped from exports). **Status: shipped 2026-06-02;
SEP axis removed from production 2026-06-03** (trust gate v2 - supersedes the earlier harness-era
gate that also read `xval_results.csv` / `sep_confidence`).

### Cross-validation CLOSED (aperture path); SEP is the independent witness
draft_000365 triple-validated (photutils + sep + dao): the science number reproduces to ~1 %,
and **sep matches VYVAR extraction to 0.2 %/frame** (a SExtractor mesh-background pipeline ==
VYVAR aperture photometry). photutils-annulus inflates on crowded/faint targets and is NOT a
reliable independent witness; sep is. VYVAR `lc_rms` is consistent with and slightly
conservative vs the raw differential floor (never under-reports noise). **IRAF/PyRAF closed as
unnecessary** (no independent axis; not feasible on Py3.12/Ubuntu24). **PSF cross-val deferred**
to a PSF-heavy/faint draft. The `xval_run.py` harness is validated and reusable.

### In-pipeline `sep_xval` stage retired; trust gate re-anchored on comp-stability (2026-06-03)
The draft-level production stage (`sep_xval_core`, `sep_xval_*` config, per-target
`sep_confidence` in `photometry_summary.csv`) is **removed**. Rationale: the validated
independent witness (SEP via `xval_run.py`) remains available offline; running a second full
extraction pass on every production draft duplicated harness work without adding a distinct
trust axis once comp_qa (Sokolovsky LOO + magnitude locus) is productionized. The **trust gate
v2** uses only: comp health (`n_clean` from comp_qa), check-star scatter, and
`lc_quality_flag`. **`lc_quality="noisy"` stays informational only.** Runtime citations are driven by
`comp_qa` / trust / photometry paths only (no SEP axis). SEP/SExtractor entries remain in
`CITATIONS.bib` for the offline `xval_run.py` harness. Historical cross-val rationale above is
unchanged. Harness helpers live in `xval_harness_core.py` (shared with `xval_run.py`).

## Calibration & parameters

### Single authoritative parameter resolver (provenance)
`param_resolver.py`: equipment-intrinsic (gain/RN/pixel/focal/saturation) = **DB(valid) ->
header(cross-check warn) -> config**; observation-specific = header -> DB -> config; site
(lat/lon/elev) = **per-draft `ID_LOCATION` -> header -> config (flagged, never silent)**. BJD /
airmass are now **config-independent** (derive from the draft's own site). `config.json`
`observer_location` is UI / last-session state only - moot for the science. **Status: settled
2026-05-30.** Closes TODO-GEO (ROADMAP 2026-06-09).

## Catalogs

### Retire APASS/Tycho B-V -> pure Gaia BP-RP
APASS/Tycho B-V is reached only via the last-resort `lookup_bv_from_local_db` fallback; no
production algorithm needs it (colour term + tiers are already BP-RP-native; ~3 % of targets
on 362). **Stage 1 done** (fallback disconnected in 4 callers; determinism verified - the
earlier ~45 % comp diff was run-context drift, not the disconnect). **Recommended scope = A+B**
(also retire all Johnson B-V dual-mode).

**B-V A+B executed (2026-06-03):** Johnson B-V retired; comp tiering and hard colour filter
are **Gaia BP-RP only**. Targets without `bp_rp` -> T4 / magnitude-proxy (accepted minority).
`lookup_bv_from_local_db`, `bp_rp_to_bv`, `teff_to_bv`, dual-mode config (`phase01_use_bprp_primary`,
`*_bv_limit`, `phase01_tier*_bv`) removed from production. `VSX/vsx_make.py` builds VSX-only;
regenerate `vyvar_vsx_local.db` on the catalog machine (see JOURNAL). Stages 2-4 complete.

## Reporting & export

### AAVSO / VarAstro correctness
MTYPE = **STD** with `TRANS=NO` (every prior file was mislabeled DIFF); table-driven FILT map
with `#WARNING` on unknown filters (no silent CV); honest `meth=` label; **KMAG = measured
ensemble-standardized check-star magnitude** (per-row sidecar; check star excluded from its own
ensemble). **Routing: eclipsing -> VarAstro (LC); pulsating/all -> AAVSO.**

### Citations: `CITATIONS.bib` is the single source of truth
One conditional emitter (`citations.py`) shared by the AAVSO export, VarAstro export, and PDF
Methods - cites **only methods that actually ran**. CORE (always) + a gated **DATA-QUALITY
GATE** section (Sokolovsky / von Neumann when comp_qa/trust on). SEP/SExtractor citations are
offline-harness only (see *In-pipeline sep_xval retired*, 2026-06-03). Comp-selection rationale cited via Broeg, Henden & Kaitchuck, AAVSO CCD
Guide. **Status: settled 2026-06-02.** Runtime DATA-QUALITY GATE cites Sokolovsky / von Neumann
when comp_qa/trust run; SEP citations are harness-only after 2026-06-03 (see DECISIONS:
*In-pipeline sep_xval retired*).

### PDF: R1 overflow guarantee
Wrapping `Paragraph` + pagination + layout guard -> **0 overflow violations**. Aperture-only
default output is byte-stable. Trust badge + per-method overlays are additive and must preserve
the 0-overflow guarantee.

## Strategic

### Comet photometry - feasible, but a future parallel phase (do NOT start yet)
Architecture is sound and reuses the front-end (calibrate -> platesolve -> star-stack -> Gaia ZP),
then forks into comet-rate stacking + extended coma photometry + ICQ/COBS export. Mature tools
exist (KOPR, Tycho-Tracker, Comphot); VYVAR's value is workflow integration + a Gaia zeropoint,
not novel science. **Decision: analysis only; start only after the variable-star pipeline is
finished.** The B-V/APASS removal does not block it (Gaia->V Riello gives V-equivalent comps).

### Brand / paper title - locked
The name **VYVAR** is final. Working title: *VYVAR: An Automated Differential Photometry
Pipeline for Amateur Variable Star Observers* (PASP / AN).

### APCORR-COLOR - extrapolation hard-block; NoFilter CT still off (2026-06-03)
**Prototype (draft_000366, NoFilter):** `VYVAR_CT_PROTOTYPE=1` measured would-be CT without
changing production LCs. Findings supersede earlier roadmap estimates (`c1 ~ -1.0`,
cat-inst ~0.12-0.16 mag): median c1 ~ -0.07 (all) / -0.36 (nonzero) / -0.53 (gate-passers);
median |ct_corr| ~ 0.019 mag (p90 ~ 0.5 mag); cat-inst scatter 0.078->0.053 mag; only ~11%
pass numeric gates; worst cases up to ~4.8 mag on extrapolated red targets. **NoFilter CT
enable remains parked.**

**Correctness fix (all filters):** `_check_color_term_extrapolation` now **returns False**
when target BP-RP lies outside the comp BP-RP range (+- `phase01_ct_extrapolation_tol`, default
0). Call site skips `apply_color_term` (`ct_ok=False`, uncorrected fallback) instead of
warn-only. Targets are never dropped or NaN'd. `should_apply_color_term` NoFilter skip unchanged.

## Colour term - validation & production (2026-06-03)

### In-range CT apply path validated (machinery + science-grade)
Machinery-grade on M67 Blue (astro-RGB); science-grade on h & chi Per photometric B/V/Rc.
**Acceptance evidence = comp-scatter reduction scaling with `|c1|` + physical `|c1|` ordering
(Rc<V<B) + `stderr_ratio` <= 0.5** - not `|c1|` magnitude.

### Retraction: "photometric -> small `|c1|<<1`" expectation is wrong
c1 is relative to Gaia G; a large B term (~1) is physical. Validate via fit quality + scatter
reduction + ordering, not absolute c1 size.

### `phase01_ct_min_comp = 7` default retained
Do **not** flip on single-field evidence. M67 Green favoured lower; h & chi Per (n_comp~140) shows
the gate is moot in rich fields. Settle via cross-field experiment. `stderr_ratio <= 0.5` is the
real quality guard.

### Exposure-merge not adopted as a CT fix
Refuted on M67 - degrades c1 stderr. Same-filter/different-exposure stays exposure-aware for the
c1 fit.

### Red CT on M67 is data-limited (saturation)
Shorter Red exposures are the lever, not algorithm changes.

### Colour term decoupled from target selection (production)
Colour term is an applied-correction **toggle** (`apply_color_term`: auto/on/off). Photometry
always runs the full VSX field for every filter. `VYVAR_CT_PROTOTYPE` presel is an opt-in
validation mode ONLY, never the production path.

### Non-cal mode: no `calibrated/` directory
Frames live in `non_calibrated/lights/`; all consumers read via one source root.
`calibration_mode=pre_calibrated` recorded end-to-end.

### Conceptual (science output): G-referenced magnitudes
VYVAR colour terms are relative to Gaia G -> corrected magnitude is G-referenced, not standard
Johnson/Sloan. AAVSO-standard B/V/Rc requires a standard catalog (APASS) or a documented
G->standard transform - to resolve before science submission.

### trust_flag_core Phase E (2026-06-08)

- **Finding A:** summary/export targets absent from `trust_map` default to **RED** with reason
  `not evaluated (no comp QA / missing from trust map)`; `LOGGER.warning` when any summary id
  is missing. Conservative mission-safe default (was GREEN).
- **Finding B:** `classify_warnings` adds soft note `no check-star verification available` when
  `check_scatter` is nan; can shift GREEN->YELLOW when no other warnings. Max 2 soft preserved.
  **2026-06-10 (draft_382 check-star audit):** 15 hard-RED check-star targets on 12-15 frame
  sessions are not genuine variables (8 crowding-blend, 4 short-baseline-outlier, 2 metric-mismatch,
  1 thin-pool). Follow-ups CS-1..4 logged in ROADMAP (frame-blind 0.05 gate, select/gate metric
  mismatch, ensemble-exclusion gap, crowding caveat) - record only; not fixed with #3.
- **Finding C (C1 chosen):** keep `np.nanstd(km)` ddof=0; 0.02/0.05 thresholds calibrated to
  population std. Revisit ddof+threshold co-calibration on ROADMAP (not this pass).
- **Finding D:** `len(soft) >= 3 -> RED` kept as forward guard (today max 2 soft).
- **Finding E:** deferred -- lc_quality-missing soft note (would make D reachable). **2026-06-10
  (rev b):** `short_baseline` is a **non-escalating** soft (excluded from `len(soft)>=3 -> RED`);
  Finding E **stays OPEN** -- not the third escalating soft source. See
  `VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md`.
- **draft_000366 trust re-run:** 10 GREEN->YELLOW, 0 GREEN->RED; numeric LC/comp_quality unchanged.

### B905 zip strict policy (2026-06-08, Phase D)

`strict=True` only where paired iterables are equal-length by construction (parallel
per-frame arrays, pairwise boundaries, same-length Series `.tolist()` pairs).
`strict=False` where ragged length is intentional (`.get(col, pd.Series())` fallbacks,
cross-DataFrame UI zips) or on untested display code. `strict=False` preserves today's
truncate-to-shortest behavior; `strict=True` adds a defensive length assertion only.

### comp_qa_core CQ-C - fix-once magnitude locus (2026-06-09, Phase F)

The comp QA magnitude locus was rebuilt from an accumulating `dropped_global`, coupling per-target
flag thresholds to target processing order (circular: drops shaped the locus that shaped drops).
**Decision:** use the **fix-once** pass-1 locus (`build_locus` over the full pass-1 pool) for all
per-comp `locus_at` / spike / flag evaluation; `dropped_global` remains for survivor bookkeeping only.

**Validation (draft_000366):** order-independence PASS (>=5 shuffled target orders -> byte-identical
QA payload). Bounded diff vs iterative locus: **1** comp flag flip, **1** target `n_clean` +1,
**0** trust-label changes (borderline only). `lightcurve_*.csv`, `comp_quality_*.json`, and
`comparison_stars_per_target.csv` unchanged.

**SHA transition:** core photometry subset (283 files) stays **`770966c3...`**; reference baseline
expanded to include `comp_qa_*.json` sidecars (426 files) -> **`edbd97e7...`** (intentional
CQ-C re-baseline, not a photometry regression). Sibling: ddof+threshold co-calibration (ROADMAP).

### Gaia DR3 catalog ingest -- GAIA-3 Riello G correction (2026-06-10)

DR3 `phot_g_mean_mag` already includes the Riello et al. 2021 milli-mag correction for
6-param and 2-param solutions; **do not** re-apply. Prior "missing correction" concern closed.
See `VYVAR_GAIA_DR3_AUDIT.md` (GAIA-3).

### Gaia audit GAIA-1 / GAIA-2 deferred to DR4 (2026-06-10)

`pmra`/`pmdec` (PM propagation) and `ruwe`/`duplicated_source` (astrometric-quality filter) will
be added in the **Gaia DR4** catalog build (DR4 ~Dec 2026, ref epoch J2017.5), not by restarting
the in-progress DR3 rebuild. The DR3 build completes as-is on the existing schema.

**Rationale.** Gaia DR4 requires a fresh full-sky build regardless; restarting a ~50 h DR3 build
for an interim catalog superseded within ~6 months is not worth the sunk cost.

**Accepted interim risk (until DR4).** Platesolver PM propagation (`_apply_proper_motion`,
`GAIA_EPOCH = 2016.0`) stays a no-op against the DR3 catalog; no `ruwe`-based comp filtering.
Wide rig (~9.77"/px): negligible. Fine rig (~0.65"/px) in dense fields: GAIA-1 mis-association
risk remains **unmitigated** -- treat fine-scale dense-field reference magnitudes with this caveat
until DR4.

**DR4 migration hooks (act at DR4 build time):**
1. Reference epoch J2016.0 (DR3) -> **J2017.5** (DR4): `GAIA_EPOCH` at `vyvar_platesolver.py:63`
   must update; prefer sourcing from catalog metadata.
2. DR4 `build_gaia_catalog.py`: SELECT + `_ROW_COLUMNS` + `init_db` + INSERT must include
   `pmra`, `pmdec`, `ruwe` (+ optional `duplicated_source`); downstream already tolerates them
   (`database.py` :210-212).
3. Re-verify lite-table column availability per DR4 datamodel.
4. DR4 ~2.5B sources with reliability split; G <= 16.5 cut keeps VYVAR in the reliable subset.

(GAIA-3 already closed: G-band correction baked into DR3 values; do not re-apply.)
See `VYVAR_GAIA_DR3_AUDIT.md`.

### Short-baseline LC quality `short_baseline` (#3, 2026-06-10, spec ready)

New terminal `lc_quality` class for `[lc_quality_short_min_frames, lc_quality_min_frames)` with
OK normal fraction. Defaults: short=**3**, min=**20** (LPV/Mira few-frame nights submittable).
Terminal (no noisy/good sub-verdict); YELLOW trust; **exportable** to AAVSO; **excluded** from
`len(soft)>=3` RED escalation. Implementation: `VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md`.
Follow-up: vsx_type-aware frame thresholds (out of scope).

### Comp selection - proximity tie-break reverted (2026-06-08)

`dist_score` removed from `comp_selection_per_target.py`: the proximity tie-break was
deliberately reverted (Broeg 2005; Henden & Kaitchuck 1982; AAVSO CCD Guide) - proximity
belongs as a **gate**, not a ranking criterion - so the orphaned local and its
`optimal_dist_arcsec` helper were removed. Final comp ranking sorts by `comp_rms`
(`out.sort_values(["comp_rms", "catalog_id"], ...)`).

## Blind plate-solve - rig prior (2026-06-04)

Telescope + camera are always known in VYVAR -> **plate scale and FOV are legitimate priors**, not
unknowns to search (ASTAP / astrometry.net `--scale` model). `blind_use_rig_prior=True` (default)
enforces: (1) pre-vote `L3_image/L3_catalog` ratio gate, (2) post-fit WCS scale consistency in verify,
(3) FOV-derived central selection and triangle size caps (not index `log_L3_max`), (4) gnomonic
triangle sides when FOV >= 2 deg. Full scale-blind mode remains via `blind_use_rig_prior=False`. Index
**series** (`fine` / `wide` density tiers, all mag14) selects tier from known scale.

## Trust / comp QA - Chi_and_H diagnostic (2026-06-04)

- **CT result locked:** reproduced on clean fresh run draft_380 (B -1.08, V -0.38, Rc -0.02);
  CT-toggle/decoupling verified end-to-end on all filters (~371 targets, 0 "nan", comps + check-star LCs).
- **n_clean=0 / trust RED on Chi_and_H is draft-specific plumbing, NOT a cleaning-gate regression**
  (draft_366 baseline reproduces original n_clean/trust with current code). Root cause: hardcoded
  `proc_*_Light_*.csv` glob in `load_proc_pivot` - the **pre-cal-naming class** again. The fix belongs in
  a **single canonical pre-cal proc-CSV resolution**, not a one-off per consumer.

## Chi_and_H catalog policy - zaloha-only (2026-06-11)

**Adopted:** `chiandh_night_run_bvr.py` and the anchor recipe use **only** paths from
`config.json` pointing at `GAIA_DR3/zaloha/` (G<=16) + zaloha blind PKLs. **No field DB, no
TAP, no astroquery** in the Chi_and_H night-run path. `build_gaia_catalog.py` adaptive-split
remains DEFERRED until the next full-sky build.

**Retired anchors:** `d246a5be` / `30a2f461` (draft_382 TAP G<=19.5); `f4bcc0ee` / `bd0b1792`
(draft_385 truncated photometry / false success).

## Confirm-reproducibility-before-locking (2026-06-11)

Standing discipline: **two independent fresh runs must be byte-identical** on photometry SHA
before recording a new anchor (`draft_386 == draft_387` for the current cut). Record SHAs,
recipe, and `git rev-parse HEAD` in STATE/JOURNAL. Trust/QA changes must re-verify photometry
SHA unchanged; trust counts may move (intended).

## Night-run completeness gate (2026-06-11)

`night_run.audit_photometry_completeness` fails `night_run_success` when any setup's
`photometry_summary` covers <90% of `active_targets`. Guards the silent-truncation-as-success
class (draft_385, draft_383).

## Trust / check-star correctness (2026-06-11)

**Findings A/B closed** (`VYVAR_TRUST_CHECKSTAR_HARDENING_SPEC.md`): un-evaluated trust -> RED;
`check_star_min_epochs=5`; `check_star_scatter` uses `ddof=1`. **Finding E re-checked:**
`short_baseline` remains non-escalating YELLOW.

**CS-3 (circular check):** Phase-2A check-star selection used column-based ensemble exclusion
that was dead code - on draft_387 ~97% of selected checks were still ensemble members. Fix:
`ensemble_member_ids()` + `select_check_star(..., ensemble_ids=...)`. Spec:
`VYVAR_CHECKSTAR_SELECTION_SPEC.md`.

**CS-2:** `check_select_rms_floor` guards artefact `comp_rms~0` rankings. **CS-4:** drop
candidates with `contamination_idx > aperture_correction_max_contamination` when column present.

**Reserved check-star (hold-one-out by design):** PARKED - would change which stars enter the
ensemble and **move the photometry anchor**; requires explicit re-cut if ever adopted.

## Broad-except regression guard (2026-06-11)

**Enforced:** ruff `BLE001` + `E722` in `pyproject.toml` `select`; `.pre-commit-config.yaml`;
`tests/test_ble001_regression.py`. Existing sites grandfathered with `# noqa: BLE001` (168 added);
4 bare `except:` narrowed to `except Exception:`; 8 `photometry_core` parse paths narrowed.
Critical LC/completeness path reviewed - no silent swallow found.

## Comparison-star trust floor - ADOPTED (2026-06-11, Option B)

**Spec:** `VYVAR_COMP_FLOOR_POLICY_SPEC.md`. **Trust-only split** - byte-identity-neutral w.r.t.
photometry (anchor `203254fd` / `95a5515a` unchanged).

**Adopted:**

- `comp_trust_min_comps = 5` (config + Settings Data quality) - trust RED floor via `n_clean`.
- `phase01_comparison_n_comp_min` stays **3** (Phase-1 selection / ensemble unchanged).
- Trust `strong = min(comp_trust_min_comps + 2, phase01_comparison_n_comp_max)` -> **7** at defaults.
- `n_clean` 5-6 -> thin comp soft YELLOW; 3-4 -> RED.
- **Trust baseline on draft_387:** **1382 YELLOW / 106 RED** (floor-5; 1488 rows). Pre-floor-5
  1400/88 superseded.

**Not adopted (Option A):** raising Phase-1 selection floor to 5 - would move photometry SHA
(draft_387 footprint: 45 per-setup hits at 3-4 comps); anchor re-cut required if ever pursued.

**Literature rationale:** Broeg weighted ensemble; AAVSO ~12-20 good comps; robustness floor ~5+.
See spec for citations.

**Check-star coupling:** CS-3 left 60 independent checks on draft_387; reserved-check design still
parked (ROADMAP).

## Comp-slope stability on common-mode-removed residual (2026-06-11)

**Adopted (science-changing on long baselines):**

- **Common-mode detrend before slope test** (`check_comparison_stability`, default ON): fit a line
  to the per-frame median of active comp LCs, subtract for slope evaluation only - ensemble
  magnitudes unchanged. Formal basis: Honeycutt (1992) ensemble `em(e)` term; Broeg (2005)
  artificial-comparison framing; Sokolovsky (2017) indices judged vs the comp population.
- **BJD sort before `np.interp`** (B2): frame/proc order is not monotonic in BJD; unsorted xp
  corrupted the common-mode estimate (Step A: 97 vs 237 mmag/hr on DY Peg).
- **Significance gate** (B1): exclude on slope only if **both** `|slope| > comp_max_slope_mmag_hr`
  **and** `|slope|/stderr >= comp_slope_significance_k` (default 3.0) on the **post-detrend**
  residual. Large-but-insignificant slopes (noise / imperfect common-mode removal) are kept.
- **`comp_slope_significance_k`** in config + Settings + `VYVAR_PARAMS.md`.
- **Honeycutt 1992** citation emitted only when common-mode stability detrend runs
  (`pipeline_meta.common_mode_stability_detrend`); remains in VarAstro flux-sum line (Collins +
  Honeycutt combination).

**Thin fields unchanged:** when `n_good < n_comp_min`, slope/p2p flags -> `suspect` with
`kept: n_good<min` - ensemble membership unchanged. DY Peg (2 comps) stays RED via comp_qa
`n_clean` skip, not this change.

**Anchor footprint (`draft_000387`):** 12 frames/setup (<20 guard) -> detrend + slope paths never
exercised; **0** historical `slope=` comp notes; **LC byte-identical** on re-run expectation.
Re-baseline (Step D) waits for Milan acceptance of bounded diff on longer-baseline validation
(e.g. DY Peg `draft_000390`: slope notes removed, ensemble unchanged).

### Sparse-only comp fallback (2026-06-12)

**Decision:** Wholesale iterative comp clip rejected on anchor `draft_000387` (marginal churn, not net
precision). Ship the same CM-residual clip machinery as a **per-target sparse-only fallback** behind
`comp_sparse_fallback_enabled` (**default ON** from 2026-06-11 re-baseline lock; alias
`comp_iterative_clip_enabled`).

1. Run default a-priori selection unchanged; if `>= comp_sparse_fallback_min` comps -> stop.
2. Else if flag ON: generous masterstars pool (no global RMS pre-filter / no a-priori comp_rms gate),
   iterative leave-one-out 5sigma-MAD on CM-removed residuals; recover LC if `>= n_comp_min`.
3. Provenance: `comp_path`, funnel columns on `comparison_stars_per_target.csv`.
4. Trust: all `sparse_fallback` targets -> **YELLOW**; default-path targets unchanged.

**Anchor re-baseline (2026-06-11):** Three-way reconcile (`203254fd...` vs current code flag-OFF
class vs flag-ON) on in-SHA artefacts. Rich `draft_000387` has **0 default-starved targets** in
full photometry - fallback **inert** (all `comp_path=default`; 0 recovery LCs). Raw SHA moves to
`3f7c9e7a...` / `d5b72d08...` (two-run repro). Drift vs old cut is benign: `comp_path` provenance,
BJD/HJD ~1.9x10-^9 d, per-frame `err` QC recalc (~2.5x scale, mag/flux byte-identical). Accept via
`compare_photometry_science_meaningful` (PROCESS) - not raw `filecmp` vs `203254fd...`.

**Baseline comparison method:** raw byte SHA for lock/repro; science-meaningful tolerance gate for
regression vs prior anchor (provenance + QC excluded; BJD/HJD <=1e-6 d; mag/flux <=1e-6).

### Plate-solver: scoped robustness + Brno production fix (2026-06-14)

**Decision (locked):** Production defaults in `solve_wcs_with_local_gaia`:

| Flag | Default | Rationale |
|------|---------|-----------|
| `solver_use_cone_for_sip` | ON | SIP pass 2 rematches on **deep Gaia cone** (not triangle slice) |
| `solver_fits_header_hint_sep_escape` | ON | Verified-strong escape only (see below) - not match% alone |
| `solver_apply_roworder_yflip` | **OFF** | **Rejected** - regression gate: ~**320 px** home-rig star displacement (77% LCs broken) |
| `solver_legacy_masterstar_mirror_sweep` | ON | Single orientation resolver (home + Brno); mirror sweep retained |

**ROWORDER `BOTTOM-UP` Y-flip rejected** - anchor regression gate showed it displaces home-rig stars
~320 px. Kept **OFF**; legacy mirror sweep is the orientation resolver.

**Brno 83.1% match retracted as a target** - draft_399 / lower detection count (154 vs 250) artifact;
never production-validated on `generate_masterstar_and_catalog`. **Policy:** do not chase high match%;
pass an **overlay-confirmed** correct-but-distortion-limited solve at lower match when appropriate.

**Stale-hint cone recenter (real Brno blocker):** Gaia cone was built at `VY_TARG` while the linear
WCS center was **0.228 deg** off. When header hint vs solved center offset **>= 0.05 deg**, solver
re-queries Gaia at the **solved WCS center** and re-runs full-pair refit.

**hint_sep escape only on verified-strong solves:** cone recenter applied + **>= 75%** brightest-N
match + RMS **<= 2 px** (+ overlay confirmation for distortion-limited passes) - **never on match%
alone**.

**Anchor gate:** same-harness legacy-vs-scoped re-cut on `draft_000387`: **0 science failures**
(B) vs (A); B WCS **~0.003 px**. Re-cut vs archive alone is **not** a reliable gate (~2.26 mag B
harness drift, internally deterministic) - use **`sandbox/anchor387_legacy_vs_scoped_gate.py`**.

**Anchor re-baseline:** **3f7c9e7a (core) / d5b72d08 (full)** with sparse-only fallback default ON.
**Science-meaningful comparator** adopted (numeric tolerance on BJD/mag/flux; excludes provenance
columns).

**SIP guard:** `force_apply` on MASTERSTAR requires `rms_sip <= rms_linear`. Distortion-limited fields
may remain linear when SIP regresses.

**Equipment:** C5A-150M (id=4, 3.76 um), AZ800 (id=6, F=5480 mm) seeded in `initialize_database()`.

### Per-set astrometry fault isolation (2026-06-14)

**Decision:** In multi-group drafts, a plate-solve / MASTERSTAR failure in one filter/setup must **not**
abort astrometry for sibling sets or block photometry on sets that already produced catalogs.

**Mechanism:** `astrometry_align_and_build_masterstar` loops jobs with try/except; merges survivor
reports via `_merge_astrometry_group_reports`; attaches `skipped_subgroups` for failed setups.
All-fail still raises. **Single-group path unchanged** - one set, nothing to continue to.

**RUN VYVAR:** photometry stage hard-fails only when **no** set completed; partial success logs OK +
skipped/failed sets (including astrometry skips from `skipped_subgroups`).

**Fail-closed on skipped set:** exception before catalog / `per_frame_catalog_index.csv` write - no
half-written MASTERSTAR downstream.

**TASK 2 (shipped 2026-06-14):** catalog-recovery verification gate + hint-as-prior on MASTERSTAR.

**Accept gate (VERIFIED):** `catalog_recovery_tight >= masterstar_catalog_recovery_min` (default **0.65**),
`n_matched_tight >= masterstar_min_matched_floor` (default **40**), and distortion healthy
(`distortion_limited_benign` **or** `centre_rms <= masterstar_centre_rms_max_px`, default **1.20 px**).
Detection-denominated `_match_rate` / brightest-N remain **informational only**.

**hint_sep:** once VERIFIED, stale pointing offset is **`hint_sep_warn`** (non-fatal; PDF cover note via
`VY_HSWN`). Hard reject only when **not VERIFIED** and `hint_sep > max(1.5 deg, fov_diameter_deg)`.
Stacked FITS-header escape blocks (>=85% match + RMS <=2 px) **removed** - superseded by this rule.

**Distortion benign ratio:** edge/centre cap **2.50 -> 3.20** (`masterstar_distortion_benign_ratio_max`;
Brno `r` ratio ~3.0).

**Citations:** Lang et al. 2010 (Astrometry.net) emitted when catalog-recovery verification runs.

**Supersedes** hint_sep escape paragraphs above (>=75% brightest-N + RMS <=2 px widen) and TASK 2 blocked note.

### Plate-solver: scoped robustness lock (2026-06-14, superseded)

### Iterative ensemble-relative comp clip (2026-06-12, superseded by sparse-only fallback)

**Decision:** Retire binding a-priori `comp_rms` cuts (global pool pre-filter + per-target gate)
for sparse-field recovery. Replace with **generous candidate intake + iterative 5sigma-MAD clip on
CM-removed ensemble residuals** (Gilliland & Brown 1988; Broeg 2005; Honeycutt 1992 common-mode
detrend; Burdanov et al. 2014 / epsilon Indi 2020 practice; Everett & Howell 2001).

**Superseded:** wholesale flag - use sparse-only fallback above.

---

## Photometry math / simple differential (2026-06-15)

- ALG-3 comp temporal binning (`temporal_bin_comp_lc`) is incorrect for VYVAR's regime; **default
  OFF**. Proven root cause of non-home-set chaos (mechanism: per-frame common-mode breakage;
  corr(injection, transparency HF)=0.9995).
- Color term (c1) to be **dropped** in favor of color-matched comp selection (min |delta BP-RP|):
  removes the color systematic at source.
- Comp selection criterion = **min |delta(BP-RP)| + min RMS**; plain per-frame ensemble; no
  temporal binning, no color term, no complex weighting.
- Trust RED/YELLOW **temporarily disabled** during photometry tuning; to be re-derived on corrected
  numbers afterward.
- Legacy fields/anchors (h&chi Per, DY Peg, BO CVn) and old-SHA re-cut framing are **retired**; we
  are on the new catalog + new pkl.
- Fix mis-attribution: ALG-3 is **Hartley & Wilson 2023, MNRAS 526, 3482** (not Broeg-Bischoff &
  Dreizler) at docstring + config.py:452 + dev/results/config_schema.md:145 (archived), and the UI caption ("after
  ensemble" -> ALG-3 runs BEFORE ensemble).
- **Supersede** proposed `comp_color_window_bprp` param (PARAMS 2026-06-15): reuse existing
  tier ladder (`comp_tier1_bprp_limit` 0.15 -> tier2 0.30 -> tier3 0.55 -> cap
  `comp_max_delta_bprp` 0.79) in `_select_comps_by_color_then_rms`; no lone 0.2 step, no new key.
- Phase-1 comp rank artefact floor: **`comp_select_rms_floor` = 1e-6** (drop isolated_bin comps
  before RMS ranking; mirrors CS-2 `check_select_rms_floor` pattern at 1e-4).
- **Workstream A landed (2026-06-15):** dataclass + config.json defaults (`temporal_binning_enabled`
  False, `apply_color_term` off); Phase-1 routes through `_select_comps_by_color_then_rms` in
  `_assign_comp_tiers_to_pool`; tier load-clamp fixed (0.15/0.30/0.55 survive JSON). DoD-A PASS
  V0612 `delta_mag` 0.0113 / 0.949 / 7 comps.
- **Gate:** >=1 additional ground-truth field recommended before treating V0612-only as global
  default risk closure (Milan risk call).

---

## Decision-grounding rule (2026-06-15, ADOPTED - Milan)

Any design fork Claude brings to Milan must be grounded in physics/math, peer-reviewed literature,
or documented field practice. Bare engineering preference is not sufficient; no "recommended" label
without a cited basis. Grounding may supersede earlier recommendations. Method citations belong in
`CITATIONS.bib` at call sites when code changes land.

---

## Reporting-column fix - grounded synthesis (2026-06-15, supersedes B1/B2)

**Earlier B1/B2 framing withdrawn:** "guard the airmass detrend" treated a non-physical step as
load-bearing; not grounded in differential-photometry physics.

**Code audit (read-only, 2026-06-15):**

| function | file:line | finding |
|----------|-----------|---------|
| `airmass_detrend_lc` | *(removed)* | Least-squares fit **`mag = a.airmass + b` on the target's own curve**. **Not** a comp-derived extinction coefficient. **[2026-06-19 - fully removed]** wiring helper `_apply_airmass_detrend_helper` (T1-2) and functions `airmass_detrend_lc`/`airmass_detrend_lc_piecewise` (T1-7) deleted; per-target airmass detrend no longer runs (Phase-2A summary reports `am_detrended=False`), per the grounded fix below. |
| `detect_outliers` | `photometry_core.py:3323-3360` | Global median + MAD on all finite mags; **no VSX/feature mask**. Eclipse dimming -> `outlier_lo` (`mag > med + thr`, `:3354-3356`). V0612 DoD-A LC: **2x `outlier_lo`** (ingress). |
| `delta_mag` export | `save_lightcurve_csv` `:7594`, `:3814` | **Unchanged** by outlier/airmass stages; only `mag_calib*` columns are rewritten (`:7486-7496`). |
| Shape preservation | DoD-A LC `tmp/phase10/.../lightcurve_1111749368289526912.csv` | `corr(delta_mag, mag_calib_raw)` **0.998**; historical `corr(delta_mag, mag_calib)` **0.59** after target-fit airmass detrend (slope ~ **0.78** mag/airmass) - detrend path since removed. |

**Grounded fix (three parts):**

1. **Reported mag = validated differential + ensemble zero-point** (`delta_mag + ZP_ensemble` per
   frame from colour-matched comps; Honeycutt 1992 ensemble - already cited). For V0612,
   pre-detrend `mag_calib_raw` already matches `delta_mag` shape (corr 0.998); implementation must
   make that the shipping curve, not hope post-hoc guards salvage a target-fit detrend.
2. **Remove per-target airmass detrend from the variable reporting path** - redundant after
   colour-matched differential (Plavchan et al. arXiv:0704.3584; Dhillon PHY217); signal-absorbing
   when fitted to the target (confirmed above). Any residual extinction -> comp ensemble, not target LSQ.
3. **Mask-first known-variable guard on `detect_outliers`** - clip out-of-eclipse only; extend mask
   around ingress/egress (TESS subdwarf recipe arXiv:2402.16018; democratic detrender clips
   out-of-transit only - arXiv:2411.09753). Required regardless of (1-2).

**DoD-B (2026-06-15): PASS** - ``apply_reporting_postprocess``; V0612 ``mag_calib`` corr **0.958** /
pre **0.011** (was 0.57); ingress 24/24 ``normal``. Harness: ``tmp/phase11/dod_b_workstream_b.json``.

**Tier-2 (PARKED):** comp-ensemble-derived k for wide delta-airmass - ROADMAP.

---

## Canonical ensemble combination - A vs B resolved (2026-06-15)

**Decision-grounding:** Gauss-Markov / Broeg (2005) AN 326:134; SPECULOOS-South arXiv:2005.02423;
Howell (1989) sigma budget. Flux-sum equals inverse-variance weighting only in the photon-limited,
all-constant limit.

**Resolution (conditional, not taste):**

1. **Canonical science product = Broeg inverse-variance estimate** - *when* sigma is complete and
   error bars are validated (chi^2/dof ~ 1 on a constant star).
2. **`delta_mag` (flux-sum) retained as AIJ-validation / diagnostic column** (`tot_C_cnts` parity);
   not the primary science export once sigma is trusted. The ~0.002 corr gap vs ``mag_calib`` on
   V0612 is the expected weighting difference, not a bug.
3. **Load-bearing work = sigma budget** (photon + read + sky + scintillation + Broeg intrinsic
   inflation) - same machinery required for TODO-GS8 / TODO-MULTISET multi-rig combine.

### Read-only audit - current code vs Broeg-canonical (2026-06-15)

| Question | Finding | Anchor |
|----------|---------|--------|
| **1. What sigma feeds `ZP_weighted`?** | **Not** the per-frame Howell CCD ``err``. Weights use **night-level `comp_rms`** = RMS of **detrended relative flux** around 1.0 (dimensionless stability metric from Phase 1 / global pool), mapped into ``w = 1/rms^2 x tier_weight``. **No scintillation**; dark only via read-noise in the separate LC ``err`` column, not in weights. | ``comp_pool_rms.py:356-380``; ``comp_selection_per_target.py:1556``; ``ensemble_normalize`` ``:2437-2446``; ``_photometric_error`` ``:636-656`` (photon+sky+read only) |
| **2. Broeg iteration / variable comp inflation?** | **Partial.** ``pytics_iterative_weights`` (default **on**, ``config.json``) iteratively **inflates `comp_rms`** from per-comp residual scatter vs weighted ZP - Broeg-like, but on **stability RMS**, not per-frame photon sigma. ``check_comparison_stability`` MAD-filters high p2p comps (excludes/suspects), does not iteratively drop variables inside ``ensemble_normalize``. **Ensemble combination itself is flux-sum** (explicitly *not* Broeg-weighted - comment: 1/rms^2 deforms extinction slope). | ``:2409-2418``; ``pytics_iterative_weights`` ``:1821-1906``; ``check_comparison_stability`` ``:1914+`` |
| **3. Error bars validated (chi^2/dof ~ 1)?** | **No production gate.** LC ``err`` = Howell photon+sky+read per frame; **not** propagated into ensemble weights; **no** chi^2/dof check on constant stars in the Phase-2A export path (Mighell chi^2-gamma cited export-only per ``VYVAR_MATH_PHYS_AUDIT.md``). DoD-B constant gate used no-regression + RMS ratio, not chi^2. | ``:1428-1434``; ``VYVAR_MATH_PHYS_AUDIT.md`` Mighell row |

**Outcome:** sigma **incomplete** for Broeg-canonical ensemble combine -> **hold flux-sum for `delta_mag`**
(AI/diagnostic); **reporting `mag_calib` already uses partial Broeg (ZP offset only)**. Do **not**
promote inverse-variance **ensemble combine** until: (a) weights use validated per-frame sigma
(Howell + scintillation + inflation), (b) chi^2/dof ~ 1 on a constant calibrator. That sigma fix
is load-bearing for GS8/MULTISET regardless.

**Citations added (sandbox, 2026-06-15):** `young1967`, `osborn2015`, `dravins1998`,
`murray2020speculoos` in `CITATIONS.bib`. Spec: `docs/VYVAR_SIGMA_BUDGET_SPEC.md`;
sandbox: `tmp/phase12/`.

### Sigma-budget work item - chi^2 audit + sandbox (2026-06-15)

**Read-only chi^2 audit:** Mighell (1999) is **export-only** (`citations.py` PSF block); no
production chi^2/dof on constant stars. PSF `reduced_chi2` and trust `check_star_scatter` are
unrelated. **Verdict:** promote **new** reduced-chi^2/dof gate (not Mighell chi^2-gamma as-is).

**Sandbox shipped:** Osborn eq. (7) scintillation + Howell quadrature + Broeg inflation helpers;
chi^2 gate harness. **Not production** until chi^2/dof ~ 1 on verified-constant calibrator.
`delta_mag` unchanged.

---

## draft_409 trust/consistency cleanup - Fixes 1-3 (2026-06-16)

### Comp stability on ensemble residual (not raw `mag_inst`)

**Problem:** `check_comparison_stability` peak-to-peak on raw instrumental mag included
night-level common-mode drift (~0.35 mag), flagging all comps suspect despite GREEN LOO trust.

**Decision:** Assess stability on **per-frame ensemble residual** (median-subtracted differential
quantity) before optional common-mode detrend. Aligns comp QA labels with trust-line intent.

### Measured aperture + observed-band SNR sizing

**Decision:** PDF card and LC export report **measured** proc `aperture_r_px`, not Phase-2A replan.
SNR-opt sizing prefers observed-band catalog `mag` over Gaia G (`_APERTURE_SIZING_MAG_COLS`).

### `lc_rms (OOE)` for variables

**Decision:** Headline precision on variable target cards = **out-of-eclipse** scatter
(`lc_rms_ooe`, brightest tertile). Full undemeaned `lc_rms` retained but not the headline for
variables (eclipse-dominated otherwise).

**Validation:** draft_409 V0612 cross-validated vs SIPS - eclipse shape + single bright outlier at
~JD 2461200.385 match in both reductions (frame-level artifact, not VYVAR bug).

---

## Development process harness (2026-06-16)

**Decision:** Adopt JSON pass/fail validation ledger + session-start baseline check
(**DEV-PROCESS-A/B**), and load **CLAUDE_OPERATING_PRINCIPLES** at session init. Grounded in
Anthropic context-engineering / long-running-agent harness guidance (get your bearings, verify a
known-good baseline before new work, structured pass/fail ledger, leave a clean state).

**Charter:** `docs/VYVAR_CLAUDE_OPERATING_PRINCIPLES.md` -- session-init required read; governs
pre-claim checklist, diagnose-then-fix, and reproducibility (archive proc before re-proc; exact
19-digit source_id matching).

**Open (spec only, ROADMAP):**
- **DEV-PROCESS-A** -- JSON ledger of end-to-end validation items (`passes: true/false`); agents
  may edit status only; no deleting or weakening tests.
- **DEV-PROCESS-B** -- session-start script reproducing reference draft / regression baseline before
  new work (motivating case: lost V0612 proc).

Definition-of-Done discipline remains in `VYVAR_PROCESS.md`; old **TODO-DEV-PROCESS** folded/replaced
by A/B plus charter session-start checklist.

---

## Phase-1 comp degradation trust bands (2026-06-16)

**Decision:** sparse_fallback + check sidecar currently caps trust at **YELLOW** even when sparse comps
are unusable (degenerate / V0611 sparse `comp_rms` ~1.9-3.3 mag). An **absolute sanity ceiling** on
sparse comps is **deferred to Phase-2** as a conscious decision (ROADMAP item 1 after Phase-1b).

**Correction logged:** prior validation note "expected degenerate -> RED" was wrong for the current
trust model. Locked rule: sparse path + check present -> **YELLOW** (not RED), until Phase-2 ceiling
lands.

**Known issue (b):** per-target `comp_rms` gate not authoritative for N_good; RMS fallback + routing
fix is **Phase-1b** (immediate next), not part of the Phase-1 commit.
