# VYVAR — Decisions & rationale

Durable design decisions and *why* they hold. This is the reference for "why is it like
this" — it should not be reopened without a new decision recorded here. Per-draft validation
numbers and the day-by-day record live in `VYVAR_JOURNAL.md`; open work in `VYVAR_ROADMAP.md`.

---

## Photometry method & scale

### Plate scale is WCS-derived (≈ 9.77″/px on the wide rig), not 1.3
The project-wide `1.3″/px` belief was wrong — it was a Newton 300/1200 + C3-26000 (binned 2×)
placeholder leaking onto the wide Carl-Zeiss/QHY294MM field via a global config default. The
resolver is **solved WCS/CD authoritative → config only as last resort** (sane clamp widened
to `0.1–30.0`). WCS-dependent geometry (ePSF isolation, FOV / `max_dist_deg`, TESS context) is
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
At 9.77″/px the PSF is well-sampled and stable across the field, so a single ePSF already
captures it and aperture wins. Every PSF variant — single ePSF, spatial `GriddedPSFModel`,
`SourceGrouper` joint fit, per-star adaptive selector — was implemented and **lost to aperture**
at this scale (single ePSF ~3x worse comp RMS; grid starves cells; grouper diverges on
sub-resolution blends). **Decision: keep all PSF flags OFF in production on the wide rig.**
On fine-scale synthetic truth (draft-367-like), PSF is now **publication-grade** (accuracy,
precision, sandwich P3) but remains **gated OFF** until real Newton / Brno data passes the
characterization gate. **Status: settled on wide; fine-scale validated-but-gated.**

### Full-frame DAO over fixed-position stamp photometry (overnight-batch model)
Every frame runs full-frame `DAOStarFinder` + match (the "master fast path" skips the per-frame
Gaia cone, not the detection). The SIPS-style speedup considered 2026-04-22 — read flux only at
fixed catalog positions, no full-frame finder — is **deliberately not adopted.** Full-frame
detection buys QC that fixed positions lose: per-frame local centroiding (absorbs WCS / drift /
resampling offsets), shape-roundness rejection of cosmics, hot pixels and CCD columns,
match-count sanity against a bad solve, new-source detection, and completeness diagnostics. The
compute cost is accepted because **VYVAR runs as an overnight batch** after the session ends,
while the observer sleeps — the binding constraint is a *trustworthy* Summary Measure Report by
morning, not wall-clock throughput. The same logic licenses PSF's extra cost when it is enabled
on fine-scale data: accuracy and robustness over speed. **Corollary:** performance work is
welcome only where it does not trade away QC (I/O, parallelism, caching); the per-frame
full-frame detection itself is a feature, not a bottleneck to remove. **Status: settled
2026-06-02.** (See also: SIPS comparison below.)

### What VYVAR deliberately does NOT adopt from SIPS (2026-06-02)
Comparison against SIPS (Moravian Instruments; v4.4 manual + Pejcha & Cagaš 2022, A&A 667,
A53) confirmed the two tools share a photometric **family** — both do full-frame per-frame star
detection, intensity-weighted sub-pixel centroids, per-star automatic apertures, robust
background, saturation invalidation, and a flux-summed ensemble. SIPS's speed comes from
**native C/C++ + multicore**, not an algorithmic shortcut — which *validates* VYVAR's full-frame
DAO choice rather than contradicting it.

**Deliberately not adopted:**
- **Fixed-position stamp photometry** (read flux at catalog x,y only; no per-frame finder). See
  the full-frame DAO decision above — QC and trust over throughput.
- **Neural-network variable detection (VDI/NN).** Against the trust mission — VYVAR keeps
  explainable statistics (Sokolovsky indices, RMS hockey-stick, independent cross-validation),
  not a black box.
- **UCAC4-based calibration.** A step back from VYVAR's native Gaia DR3 + BP-RP (deeper, modern,
  colour-complete).
- **SPL scripting / REST API.** Different stack; VYVAR is Python/Streamlit.

**Worth borrowing — scoped in ROADMAP, not core changes:**
- **Wide-field WCS distortion check** (MEDIUM): confirm SIP / higher-order terms on the wide
  rig; SIPS uses a 3rd-order 2D polynomial (Monomial/Legendre).
- **Spatial term in ensemble calibration** (LOW): SIPS's `x1·X + y1·Y + …` field-gradient terms
  — relevant only for a future whole-field absolute mode, not the current per-target differential
  path.

The Pejcha & Cagaš paper is a **citation / positioning reference for the GS7 paper**, not a
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
  verdict and no independent second-extractor cross-check. VYVAR adds comp_qa (Sokolovsky
  leave-one-out locus), SEP cross-validation (~0.2%/frame), and the three-axis GREEN/YELLOW/RED
  trust gate.
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
Per-frame encircled-energy correction removes the constant target↔comp enclosed-flux bias and
the seeing-correlated systematic from per-star SNR-optimal radii. Byte-identical when OFF.
**Decision: ship gated, leave OFF** until the mixed-frame guard is wired (a partial-success
night could mix corrected + uncorrected frames — see ROADMAP: APCORR-MIXEDFRAME).

### `skip_processed_directory` flow
Raw → Calibrated → QC-in-place (VY_* headers on calibrated) → Aligned → proc CSV → LC, skipping
the `processed/` copy. Saves ~1.5 GB and ~20–30 % per draft. **Status: Phase 1 shipped (gated,
default false); legacy `processed/` removal pending validation.**

## Comp-star selection & QA

### Selection priority: stability > colour > proximity; proximity is a GATE, not a rank
- **Stability** is enforced as a hard `max_comp_rms` cap + iterative MAD filter, and as the
  Broeg (2005) `1/σ²` ensemble weight — which dominates the actual measurement.
- **Colour** (`|ΔBP-RP|`) is a hard cut (≤ 0.79) + tier, and is the **primary pick sort key**.
  For **NoFilter/broadband** this is justified: without a filter the colour term is a
  *first-order* systematic (no bandpass to cancel it), and colour systematics don't average
  down like random scatter. Grounded in Henden & Kaitchuck (1982) and the AAVSO CCD Guide.
- **Proximity** is enforced only as the distance gate (`max_dist_deg` ≈ 1.5°, `min_dist_arcsec`
  60″). A proximity ranking tie-break was implemented and **reverted**: because Broeg `1/σ²`
  weights are order-independent, any RMS-bin tie-break necessarily trades stability for
  proximity, violating the agreed order. **Status: settled 2026-06-02** (revert restored the
  comp set byte-for-byte).

### comp_qa — Sokolovsky leave-one-out self-consistency QA
Per-comp variability indices on the zero-median LOO differential light curve: `σ_IQR`
(amplitude), `1/η` von Neumann (slow drift), `spike` (dropouts), flagged against a
**magnitude-dependent locus** (0.5-mag bins) rather than a flat floor. Grounded in Broeg (2005)
+ Sokolovsky et al. (2017) + Howell (1989). The flat-floor v1 over-flagged faint comps; the
locus version is correct. **Status: productionized 2026-06-02** as a read-only post-Phase-2A
stage (photometry byte-identical); outputs per-comp flags + per-target `n_clean`.

### Crowding classifier: gated infra, NOT enabled on the wide rig
Detection-independent signals (Gaia density, `blend_frac` at depth, comp availability) replace
the erratic stars/Mpx class; LOOSEN on comp scarcity, TIGHTEN on real blend fraction, with a
**sampling gate (FWHM ≥ 3 px)**. The wide rig is floor-limited (scintillation/undersampling at
FWHM ≈ 2.6 px), so tightening there only cuts good comps. **Decision: keep OFF for the wide rig;
enable on the well-sampled Newton cluster.**

## Trust & validation

### 3-axis trust gate — hard/soft model, inform-only (v1)
Per-target GREEN/YELLOW/RED from comp health (`n_clean`), independent SEP cross-val
(`sep_confidence`), check-star scatter, and `lc_quality_flag`. **Hard** warnings (real red
flags: `n_clean < min_comps`, `sep=review`, `saturated`, check ≥ 0.05) vs **soft** (faint/thin:
unverified sep, thin comps, marginal check 0.02–0.05). RED = `n_clean < min` OR any hard OR
≥ 3 soft; YELLOW = 1–2 soft; GREEN = clean. Thresholds derive from the user-configurable
`phase01_comparison_n_comp_min/max`. **`lc_quality="noisy"` is informational only** — it is
variability-driven (e.g. V0349 Dra: lc_rms 0.155 but check-star 0.0054 → GREEN); counting it as
a warning wrongly demoted 48 real variables. **v1 is inform-only** (RED is surfaced, not
auto-dropped from exports). **Status: shipped 2026-06-02 — supersedes the earlier flat
W-count gate (the standalone 81/52/10 → production 69/59/15 with complete check-star data).**

### Cross-validation CLOSED (aperture path); SEP is the independent witness
draft_000365 triple-validated (photutils + sep + dao): the science number reproduces to ~1 %,
and **sep matches VYVAR extraction to 0.2 %/frame** (a SExtractor mesh-background pipeline ≡
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
`param_resolver.py`: equipment-intrinsic (gain/RN/pixel/focal/saturation) = **DB(valid) →
header(cross-check warn) → config**; observation-specific = header → DB → config; site
(lat/lon/elev) = **per-draft `ID_LOCATION` → header → config (flagged, never silent)**. BJD /
airmass are now **config-independent** (derive from the draft's own site). `config.json`
`observer_location` is UI / last-session state only — moot for the science. **Status: settled
2026-05-30.** Closes TODO-GEO (ROADMAP 2026-06-09).

## Catalogs

### Retire APASS/Tycho B-V → pure Gaia BP-RP
APASS/Tycho B-V is reached only via the last-resort `lookup_bv_from_local_db` fallback; no
production algorithm needs it (colour term + tiers are already BP-RP-native; ~3 % of targets
on 362). **Stage 1 done** (fallback disconnected in 4 callers; determinism verified — the
earlier ~45 % comp diff was run-context drift, not the disconnect). **Recommended scope = A+B**
(also retire all Johnson B-V dual-mode).

**B-V A+B executed (2026-06-03):** Johnson B-V retired; comp tiering and hard colour filter
are **Gaia BP-RP only**. Targets without `bp_rp` → T4 / magnitude-proxy (accepted minority).
`lookup_bv_from_local_db`, `bp_rp_to_bv`, `teff_to_bv`, dual-mode config (`phase01_use_bprp_primary`,
`*_bv_limit`, `phase01_tier*_bv`) removed from production. `VSX/vsx_make.py` builds VSX-only;
regenerate `vyvar_vsx_local.db` on the catalog machine (see JOURNAL). Stages 2–4 complete.

## Reporting & export

### AAVSO / VarAstro correctness
MTYPE = **STD** with `TRANS=NO` (every prior file was mislabeled DIFF); table-driven FILT map
with `#WARNING` on unknown filters (no silent CV); honest `meth=` label; **KMAG = measured
ensemble-standardized check-star magnitude** (per-row sidecar; check star excluded from its own
ensemble). **Routing: eclipsing → VarAstro (LC); pulsating/all → AAVSO.**

### Citations: `CITATIONS.bib` is the single source of truth
One conditional emitter (`citations.py`) shared by the AAVSO export, VarAstro export, and PDF
Methods — cites **only methods that actually ran**. CORE (always) + a gated **DATA-QUALITY
GATE** section (Sokolovsky / von Neumann when comp_qa/trust on). SEP/SExtractor citations are
offline-harness only (see *In-pipeline sep_xval retired*, 2026-06-03). Comp-selection rationale cited via Broeg, Henden & Kaitchuck, AAVSO CCD
Guide. **Status: settled 2026-06-02.** Runtime DATA-QUALITY GATE cites Sokolovsky / von Neumann
when comp_qa/trust run; SEP citations are harness-only after 2026-06-03 (see DECISIONS:
*In-pipeline sep_xval retired*).

### PDF: R1 overflow guarantee
Wrapping `Paragraph` + pagination + layout guard → **0 overflow violations**. Aperture-only
default output is byte-stable. Trust badge + per-method overlays are additive and must preserve
the 0-overflow guarantee.

## Strategic

### Comet photometry — feasible, but a future parallel phase (do NOT start yet)
Architecture is sound and reuses the front-end (calibrate → platesolve → star-stack → Gaia ZP),
then forks into comet-rate stacking + extended coma photometry + ICQ/COBS export. Mature tools
exist (KOPR, Tycho-Tracker, Comphot); VYVAR's value is workflow integration + a Gaia zeropoint,
not novel science. **Decision: analysis only; start only after the variable-star pipeline is
finished.** The B-V/APASS removal does not block it (Gaia→V Riello gives V-equivalent comps).

### Brand / paper title — locked
The name **VYVAR** is final. Working title: *VYVAR: An Automated Differential Photometry
Pipeline for Amateur Variable Star Observers* (PASP / AN).

### APCORR-COLOR — extrapolation hard-block; NoFilter CT still off (2026-06-03)
**Prototype (draft_000366, NoFilter):** `VYVAR_CT_PROTOTYPE=1` measured would-be CT without
changing production LCs. Findings supersede earlier roadmap estimates (`c1 ≈ −1.0`,
cat−inst ~0.12–0.16 mag): median c1 ≈ −0.07 (all) / −0.36 (nonzero) / −0.53 (gate-passers);
median |ct_corr| ≈ 0.019 mag (p90 ≈ 0.5 mag); cat−inst scatter 0.078→0.053 mag; only ~11%
pass numeric gates; worst cases up to ~4.8 mag on extrapolated red targets. **NoFilter CT
enable remains parked.**

**Correctness fix (all filters):** `_check_color_term_extrapolation` now **returns False**
when target BP-RP lies outside the comp BP-RP range (± `phase01_ct_extrapolation_tol`, default
0). Call site skips `apply_color_term` (`ct_ok=False`, uncorrected fallback) instead of
warn-only. Targets are never dropped or NaN'd. `should_apply_color_term` NoFilter skip unchanged.

## Colour term — validation & production (2026-06-03)

### In-range CT apply path validated (machinery + science-grade)
Machinery-grade on M67 Blue (astro-RGB); science-grade on h & χ Per photometric B/V/Rc.
**Acceptance evidence = comp-scatter reduction scaling with `|c1|` + physical `|c1|` ordering
(Rc<V<B) + `stderr_ratio` ≤ 0.5** — not `|c1|` magnitude.

### Retraction: "photometric → small `|c1|≪1`" expectation is wrong
c1 is relative to Gaia G; a large B term (~1) is physical. Validate via fit quality + scatter
reduction + ordering, not absolute c1 size.

### `phase01_ct_min_comp = 7` default retained
Do **not** flip on single-field evidence. M67 Green favoured lower; h & χ Per (n_comp~140) shows
the gate is moot in rich fields. Settle via cross-field experiment. `stderr_ratio ≤ 0.5` is the
real quality guard.

### Exposure-merge not adopted as a CT fix
Refuted on M67 — degrades c1 stderr. Same-filter/different-exposure stays exposure-aware for the
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
VYVAR colour terms are relative to Gaia G → corrected magnitude is G-referenced, not standard
Johnson/Sloan. AAVSO-standard B/V/Rc requires a standard catalog (APASS) or a documented
G→standard transform — to resolve before science submission.

### trust_flag_core Phase E (2026-06-08)

- **Finding A:** summary/export targets absent from `trust_map` default to **RED** with reason
  `not evaluated (no comp QA / missing from trust map)`; `LOGGER.warning` when any summary id
  is missing. Conservative mission-safe default (was GREEN).
- **Finding B:** `classify_warnings` adds soft note `no check-star verification available` when
  `check_scatter` is nan; can shift GREEN->YELLOW when no other warnings. Max 2 soft preserved.
- **Finding C (C1 chosen):** keep `np.nanstd(km)` ddof=0; 0.02/0.05 thresholds calibrated to
  population std. Revisit ddof+threshold co-calibration on ROADMAP (not this pass).
- **Finding D:** `len(soft) >= 3 -> RED` kept as forward guard (today max 2 soft).
- **Finding E:** deferred -- lc_quality-missing soft note (would make D reachable).
- **draft_000366 trust re-run:** 10 GREEN->YELLOW, 0 GREEN->RED; numeric LC/comp_quality unchanged.

### B905 zip strict policy (2026-06-08, Phase D)

`strict=True` only where paired iterables are equal-length by construction (parallel
per-frame arrays, pairwise boundaries, same-length Series `.tolist()` pairs).
`strict=False` where ragged length is intentional (`.get(col, pd.Series())` fallbacks,
cross-DataFrame UI zips) or on untested display code. `strict=False` preserves today's
truncate-to-shortest behavior; `strict=True` adds a defensive length assertion only.

### comp_qa_core CQ-C — magnitude locus order coupling (2026-06-08, Phase F)

The comp QA magnitude locus is rebuilt from an accumulating `dropped_global`, so per-target flag
thresholds are coupled to the (deterministic) target processing order. **Decision:** keep current
behavior for now. A fix-once locus (computed once over the full pass-1 pool) is a **separate
methodology change** — validate with a bounded n_clean/trust diff; ROADMAP sibling of ddof+
threshold co-calibration (Finding C1).

### Comp selection — proximity tie-break reverted (2026-06-08)

`dist_score` removed from `comp_selection_per_target.py`: the proximity tie-break was
deliberately reverted (Broeg 2005; Henden & Kaitchuck 1982; AAVSO CCD Guide) — proximity
belongs as a **gate**, not a ranking criterion — so the orphaned local and its
`optimal_dist_arcsec` helper were removed. Final comp ranking sorts by `comp_rms`
(`out.sort_values(["comp_rms", "catalog_id"], …)`).

## Blind plate-solve — rig prior (2026-06-04)

Telescope + camera are always known in VYVAR → **plate scale and FOV are legitimate priors**, not
unknowns to search (ASTAP / astrometry.net `--scale` model). `blind_use_rig_prior=True` (default)
enforces: (1) pre-vote `L3_image/L3_catalog` ratio gate, (2) post-fit WCS scale consistency in verify,
(3) FOV-derived central selection and triangle size caps (not index `log_L3_max`), (4) gnomonic
triangle sides when FOV ≥ 2°. Full scale-blind mode remains via `blind_use_rig_prior=False`. Index
**series** (`fine` / `wide` density tiers, all mag14) selects tier from known scale.

## Trust / comp QA — Chi_and_H diagnostic (2026-06-04)

- **CT result locked:** reproduced on clean fresh run draft_380 (B −1.08, V −0.38, Rc −0.02);
  CT-toggle/decoupling verified end-to-end on all filters (~371 targets, 0 "nan", comps + check-star LCs).
- **n_clean=0 / trust RED on Chi_and_H is draft-specific plumbing, NOT a cleaning-gate regression**
  (draft_366 baseline reproduces original n_clean/trust with current code). Root cause: hardcoded
  `proc_*_Light_*.csv` glob in `load_proc_pivot` — the **pre-cal-naming class** again. The fix belongs in
  a **single canonical pre-cal proc-CSV resolution**, not a one-off per consumer.
