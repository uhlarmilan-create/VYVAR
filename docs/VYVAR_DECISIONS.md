# VYVAR — Decisions & rationale

Durable design decisions and *why* they hold. This is the reference for "why is it like
this" — it should not be reopened without a new decision recorded here. Per-draft validation
numbers and the day-by-day record live in `VYVAR_JOURNAL.md`; open work in `VYVAR_ROADMAP.md`.

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

### Per-frame catalog: drop unmatched DAO before aperture; Moffat gated on ePSF (2026-06-12)
**DAO detection stays full-frame** (QC unchanged). After `detect_stars_match_master_reference`,
rows with empty `catalog_id` are dropped **before** aperture / Moffat / PSF work
(`_proc_drop_unmatched_dao_rows`; key on `catalog_id`, not `source_type`). They were never written
to `proc_*.csv` anyway (final `_proc_catalog_keep_matched_rows_only`); this is pure wasted compute.

**Moffat fit is Step 1 of the two-step ePSF path only** — gate `if _run_epsf:` (not
`_run_aperture`). In aperture-only production (`psf_photometry_enabled=False`), `moffat_*` columns
are omitted; LC / comp-QA readers do not consume them. **LC byte-identical** when `VY_FWHM` /
`VY_FWHM_GAUSS` drives aperture radius (verified `draft_000389` B_60_1). Chi_and_H photometry SHA
unchanged (`proc_*.csv` not in SHA set).

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
  **2026-06-10 (draft_382 check-star audit):** 15 hard-RED check-star targets on 12-15 frame
  sessions are not genuine variables (8 crowding-blend, 4 short-baseline-outlier, 2 metric-mismatch,
  1 thin-pool). Follow-ups CS-1..4 logged in ROADMAP (frame-blind 0.05 gate, select/gate metric
  mismatch, ensemble-exclusion gap, crowding caveat) — record only; not fixed with #3.
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

### comp_qa_core CQ-C — fix-once magnitude locus (2026-06-09, Phase F)

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

## Chi_and_H catalog policy — zaloha-only (2026-06-11)

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

**Findings A/B closed** (`VYVAR_TRUST_CHECKSTAR_HARDENING_SPEC.md`): un-evaluated trust → RED;
`check_star_min_epochs=5`; `check_star_scatter` uses `ddof=1`. **Finding E re-checked:**
`short_baseline` remains non-escalating YELLOW.

**CS-3 (circular check):** Phase-2A check-star selection used column-based ensemble exclusion
that was dead code — on draft_387 ~97% of selected checks were still ensemble members. Fix:
`ensemble_member_ids()` + `select_check_star(..., ensemble_ids=...)`. Spec:
`VYVAR_CHECKSTAR_SELECTION_SPEC.md`.

**CS-2:** `check_select_rms_floor` guards artefact `comp_rms~0` rankings. **CS-4:** drop
candidates with `contamination_idx > aperture_correction_max_contamination` when column present.

**Reserved check-star (hold-one-out by design):** PARKED — would change which stars enter the
ensemble and **move the photometry anchor**; requires explicit re-cut if ever adopted.

## Broad-except regression guard (2026-06-11)

**Enforced:** ruff `BLE001` + `E722` in `pyproject.toml` `select`; `.pre-commit-config.yaml`;
`tests/test_ble001_regression.py`. Existing sites grandfathered with `# noqa: BLE001` (168 added);
4 bare `except:` narrowed to `except Exception:`; 8 `photometry_core` parse paths narrowed.
Critical LC/completeness path reviewed — no silent swallow found.

## Comparison-star trust floor — ADOPTED (2026-06-11, Option B)

**Spec:** `VYVAR_COMP_FLOOR_POLICY_SPEC.md`. **Trust-only split** — byte-identity-neutral w.r.t.
photometry (anchor `203254fd` / `95a5515a` unchanged).

**Adopted:**

- `comp_trust_min_comps = 5` (config + Settings Data quality) — trust RED floor via `n_clean`.
- `phase01_comparison_n_comp_min` stays **3** (Phase-1 selection / ensemble unchanged).
- Trust `strong = min(comp_trust_min_comps + 2, phase01_comparison_n_comp_max)` → **7** at defaults.
- `n_clean` 5-6 → thin comp soft YELLOW; 3-4 → RED.
- **Trust baseline on draft_387:** **1382 YELLOW / 106 RED** (floor-5; 1488 rows). Pre-floor-5
  1400/88 superseded.

**Not adopted (Option A):** raising Phase-1 selection floor to 5 — would move photometry SHA
(draft_387 footprint: 45 per-setup hits at 3-4 comps); anchor re-cut required if ever pursued.

**Literature rationale:** Broeg weighted ensemble; AAVSO ~12-20 good comps; robustness floor ~5+.
See spec for citations.

**Check-star coupling:** CS-3 left 60 independent checks on draft_387; reserved-check design still
parked (ROADMAP).

## Comp-slope stability on common-mode-removed residual (2026-06-11)

**Adopted (science-changing on long baselines):**

- **Common-mode detrend before slope test** (`check_comparison_stability`, default ON): fit a line
  to the per-frame median of active comp LCs, subtract for slope evaluation only — ensemble
  magnitudes unchanged. Formal basis: Honeycutt (1992) ensemble `em(e)` term; Broeg (2005)
  artificial-comparison framing; Sokolovsky (2017) indices judged vs the comp population.
- **BJD sort before `np.interp`** (B2): frame/proc order is not monotonic in BJD; unsorted xp
  corrupted the common-mode estimate (Step A: 97 vs 237 mmag/hr on DY Peg).
- **Significance gate** (B1): exclude on slope only if **both** `|slope| > comp_max_slope_mmag_hr`
  **and** `|slope|/stderr ≥ comp_slope_significance_k` (default 3.0) on the **post-detrend**
  residual. Large-but-insignificant slopes (noise / imperfect common-mode removal) are kept.
- **`comp_slope_significance_k`** in config + Settings + `VYVAR_PARAMS.md`.
- **Honeycutt 1992** citation emitted only when common-mode stability detrend runs
  (`pipeline_meta.common_mode_stability_detrend`); remains in VarAstro flux-sum line (Collins +
  Honeycutt combination).

**Thin fields unchanged:** when `n_good < n_comp_min`, slope/p2p flags → `suspect` with
`kept: n_good<min` — ensemble membership unchanged. DY Peg (2 comps) stays RED via comp_qa
`n_clean` skip, not this change.

**Anchor footprint (`draft_000387`):** 12 frames/setup (<20 guard) → detrend + slope paths never
exercised; **0** historical `slope=` comp notes; **LC byte-identical** on re-run expectation.
Re-baseline (Step D) waits for Milan acceptance of bounded diff on longer-baseline validation
(e.g. DY Peg `draft_000390`: slope notes removed, ensemble unchanged).

### Sparse-only comp fallback (2026-06-12)

**Decision:** Wholesale iterative comp clip rejected on anchor `draft_000387` (marginal churn, not net
precision). Ship the same CM-residual clip machinery as a **per-target sparse-only fallback** behind
`comp_sparse_fallback_enabled` (**default ON** from 2026-06-11 re-baseline lock; alias
`comp_iterative_clip_enabled`).

1. Run default a-priori selection unchanged; if `≥ comp_sparse_fallback_min` comps → stop.
2. Else if flag ON: generous masterstars pool (no global RMS pre-filter / no a-priori comp_rms gate),
   iterative leave-one-out 5σ-MAD on CM-removed residuals; recover LC if `≥ n_comp_min`.
3. Provenance: `comp_path`, funnel columns on `comparison_stars_per_target.csv`.
4. Trust: all `sparse_fallback` targets → **YELLOW**; default-path targets unchanged.

**Anchor re-baseline (2026-06-11):** Three-way reconcile (`203254fd…` vs current code flag-OFF
class vs flag-ON) on in-SHA artefacts. Rich `draft_000387` has **0 default-starved targets** in
full photometry — fallback **inert** (all `comp_path=default`; 0 recovery LCs). Raw SHA moves to
`3f7c9e7a…` / `d5b72d08…` (two-run repro). Drift vs old cut is benign: `comp_path` provenance,
BJD/HJD ~1.9×10⁻⁹ d, per-frame `err` QC recalc (~2.5× scale, mag/flux byte-identical). Accept via
`compare_photometry_science_meaningful` (PROCESS) — not raw `filecmp` vs `203254fd…`.

**Baseline comparison method:** raw byte SHA for lock/repro; science-meaningful tolerance gate for
regression vs prior anchor (provenance + QC excluded; BJD/HJD ≤1e-6 d; mag/flux ≤1e-6).

### Plate-solver: scoped robustness + Brno production fix (2026-06-14)

**Decision (locked):** Production defaults in `solve_wcs_with_local_gaia`:

| Flag | Default | Rationale |
|------|---------|-----------|
| `solver_use_cone_for_sip` | ON | SIP pass 2 rematches on **deep Gaia cone** (not triangle slice) |
| `solver_fits_header_hint_sep_escape` | ON | Verified-strong escape only (see below) — not match% alone |
| `solver_apply_roworder_yflip` | **OFF** | **Rejected** — regression gate: ~**320 px** home-rig star displacement (77% LCs broken) |
| `solver_legacy_masterstar_mirror_sweep` | ON | Single orientation resolver (home + Brno); mirror sweep retained |

**ROWORDER `BOTTOM-UP` Y-flip rejected** — anchor regression gate showed it displaces home-rig stars
~320 px. Kept **OFF**; legacy mirror sweep is the orientation resolver.

**Brno 83.1% match retracted as a target** — draft_399 / lower detection count (154 vs 250) artifact;
never production-validated on `generate_masterstar_and_catalog`. **Policy:** do not chase high match%;
pass an **overlay-confirmed** correct-but-distortion-limited solve at lower match when appropriate.

**Stale-hint cone recenter (real Brno blocker):** Gaia cone was built at `VY_TARG` while the linear
WCS center was **0.228°** off. When header hint vs solved center offset **≥ 0.05°**, solver
re-queries Gaia at the **solved WCS center** and re-runs full-pair refit.

**hint_sep escape only on verified-strong solves:** cone recenter applied + **≥ 75%** brightest-N
match + RMS **≤ 2 px** (+ overlay confirmation for distortion-limited passes) — **never on match%
alone**.

**Anchor gate:** same-harness legacy-vs-scoped re-cut on `draft_000387`: **0 science failures**
(B) vs (A); B WCS **~0.003 px**. Re-cut vs archive alone is **not** a reliable gate (~2.26 mag B
harness drift, internally deterministic) — use **`sandbox/anchor387_legacy_vs_scoped_gate.py`**.

**Anchor re-baseline:** **3f7c9e7a (core) / d5b72d08 (full)** with sparse-only fallback default ON.
**Science-meaningful comparator** adopted (numeric tolerance on BJD/mag/flux; excludes provenance
columns).

**SIP guard:** `force_apply` on MASTERSTAR requires `rms_sip ≤ rms_linear`. Distortion-limited fields
may remain linear when SIP regresses.

**Equipment:** C5A-150M (id=4, 3.76 µm), AZ800 (id=6, F=5480 mm) seeded in `initialize_database()`.

### Per-set astrometry fault isolation (2026-06-14)

**Decision:** In multi-group drafts, a plate-solve / MASTERSTAR failure in one filter/setup must **not**
abort astrometry for sibling sets or block photometry on sets that already produced catalogs.

**Mechanism:** `astrometry_align_and_build_masterstar` loops jobs with try/except; merges survivor
reports via `_merge_astrometry_group_reports`; attaches `skipped_subgroups` for failed setups.
All-fail still raises. **Single-group path unchanged** — one set, nothing to continue to.

**RUN VYVAR:** photometry stage hard-fails only when **no** set completed; partial success logs OK +
skipped/failed sets (including astrometry skips from `skipped_subgroups`).

**Fail-closed on skipped set:** exception before catalog / `per_frame_catalog_index.csv` write — no
half-written MASTERSTAR downstream.

**TASK 2 (shipped 2026-06-14):** catalog-recovery verification gate + hint-as-prior on MASTERSTAR.

**Accept gate (VERIFIED):** `catalog_recovery_tight ≥ masterstar_catalog_recovery_min` (default **0.65**),
`n_matched_tight ≥ masterstar_min_matched_floor` (default **40**), and distortion healthy
(`distortion_limited_benign` **or** `centre_rms ≤ masterstar_centre_rms_max_px`, default **1.20 px**).
Detection-denominated `_match_rate` / brightest-N remain **informational only**.

**hint_sep:** once VERIFIED, stale pointing offset is **`hint_sep_warn`** (non-fatal; PDF cover note via
`VY_HSWN`). Hard reject only when **not VERIFIED** and `hint_sep > max(1.5°, fov_diameter_deg)`.
Stacked FITS-header escape blocks (≥85% match + RMS ≤2 px) **removed** — superseded by this rule.

**Distortion benign ratio:** edge/centre cap **2.50 → 3.20** (`masterstar_distortion_benign_ratio_max`;
Brno `r` ratio ~3.0).

**Citations:** Lang et al. 2010 (Astrometry.net) emitted when catalog-recovery verification runs.

**Supersedes** hint_sep escape paragraphs above (≥75% brightest-N + RMS ≤2 px widen) and TASK 2 blocked note.

### Plate-solver: scoped robustness lock (2026-06-14, superseded)

### Iterative ensemble-relative comp clip (2026-06-12, superseded by sparse-only fallback)

**Decision:** Retire binding a-priori `comp_rms` cuts (global pool pre-filter + per-target gate)
for sparse-field recovery. Replace with **generous candidate intake + iterative 5σ-MAD clip on
CM-removed ensemble residuals** (Gilliland & Brown 1988; Broeg 2005; Honeycutt 1992 common-mode
detrend; Burdanov et al. 2014 / ε Indi 2020 practice; Everett & Howell 2001).

**Superseded:** wholesale flag — use sparse-only fallback above.

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
  Dreizler) at docstring + config.py:452 + config_schema.md:145, and the UI caption ("after
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

## Decision-grounding rule (2026-06-15, ADOPTED — Milan)

Any design fork Claude brings to Milan must be grounded in physics/math, peer-reviewed literature,
or documented field practice. Bare engineering preference is not sufficient; no "recommended" label
without a cited basis. Grounding may supersede earlier recommendations. Method citations belong in
`CITATIONS.bib` at call sites when code changes land.

---

## Reporting-column fix — grounded synthesis (2026-06-15, supersedes B1/B2)

**Earlier B1/B2 framing withdrawn:** "guard the airmass detrend" treated a non-physical step as
load-bearing; not grounded in differential-photometry physics.

**Code audit (read-only, 2026-06-15):**

| function | file:line | finding |
|----------|-----------|---------|
| `airmass_detrend_lc` | `photometry_core.py:3584-3651` | Least-squares fit **`mag = a·airmass + b` on the target's own curve** (`mag_fit = mag_calib[mask]`). **Not** a comp-derived extinction coefficient. Applied via `_apply_airmass_detrend_helper` (`:5732-5754`) to `mag_for_airmass` (= `mag_calib_ct` when CT ok, else pre-CT `mag_calib`; `:7440-7487`). |
| `detect_outliers` | `photometry_core.py:3323-3360` | Global median + MAD on all finite mags; **no VSX/feature mask**. Eclipse dimming → `outlier_lo` (`mag > med + thr`, `:3354-3356`). V0612 DoD-A LC: **2× `outlier_lo`** (ingress). |
| `delta_mag` export | `save_lightcurve_csv` `:7594`, `:3814` | **Unchanged** by outlier/airmass stages; only `mag_calib*` columns are rewritten (`:7486-7496`). |
| Shape preservation | DoD-A LC `tmp/phase10/.../lightcurve_1111749368289526912.csv` | `corr(delta_mag, mag_calib_raw)` **0.998**; `corr(delta_mag, mag_calib)` **0.59** after target-fit airmass detrend (slope ≈ **0.78** mag/airmass on normal frames, within `:3630-3638` guard). |

**Grounded fix (three parts):**

1. **Reported mag = validated differential + ensemble zero-point** (`delta_mag + ZP_ensemble` per
   frame from colour-matched comps; Honeycutt 1992 ensemble — already cited). For V0612,
   pre-detrend `mag_calib_raw` already matches `delta_mag` shape (corr 0.998); implementation must
   make that the shipping curve, not hope post-hoc guards salvage a target-fit detrend.
2. **Remove per-target airmass detrend from the variable reporting path** — redundant after
   colour-matched differential (Plavchan et al. arXiv:0704.3584; Dhillon PHY217); signal-absorbing
   when fitted to the target (confirmed above). Any residual extinction → comp ensemble, not target LSQ.
3. **Mask-first known-variable guard on `detect_outliers`** — clip out-of-eclipse only; extend mask
   around ingress/egress (TESS subdwarf recipe arXiv:2402.16018; democratic detrender clips
   out-of-transit only — arXiv:2411.09753). Required regardless of (1–2).

**DoD-B (2026-06-15): PASS** — ``apply_reporting_postprocess``; V0612 ``mag_calib`` corr **0.958** /
pre **0.011** (was 0.57); ingress 24/24 ``normal``. Harness: ``tmp/phase11/dod_b_workstream_b.json``.

**Tier-2 (PARKED):** comp-ensemble-derived k for wide delta-airmass — ROADMAP.

---

## Canonical ensemble combination — A vs B resolved (2026-06-15)

**Decision-grounding:** Gauss-Markov / Broeg (2005) AN 326:134; SPECULOOS-South arXiv:2005.02423;
Howell (1989) sigma budget. Flux-sum equals inverse-variance weighting only in the photon-limited,
all-constant limit.

**Resolution (conditional, not taste):**

1. **Canonical science product = Broeg inverse-variance estimate** — *when* sigma is complete and
   error bars are validated (chi²/dof ~ 1 on a constant star).
2. **`delta_mag` (flux-sum) retained as AIJ-validation / diagnostic column** (`tot_C_cnts` parity);
   not the primary science export once sigma is trusted. The ~0.002 corr gap vs ``mag_calib`` on
   V0612 is the expected weighting difference, not a bug.
3. **Load-bearing work = sigma budget** (photon + read + sky + scintillation + Broeg intrinsic
   inflation) — same machinery required for TODO-GS8 / TODO-MULTISET multi-rig combine.

### Read-only audit — current code vs Broeg-canonical (2026-06-15)

| Question | Finding | Anchor |
|----------|---------|--------|
| **1. What sigma feeds `ZP_weighted`?** | **Not** the per-frame Howell CCD ``err``. Weights use **night-level `comp_rms`** = RMS of **detrended relative flux** around 1.0 (dimensionless stability metric from Phase 1 / global pool), mapped into ``w = 1/rms² × tier_weight``. **No scintillation**; dark only via read-noise in the separate LC ``err`` column, not in weights. | ``comp_pool_rms.py:356-380``; ``comp_selection_per_target.py:1556``; ``ensemble_normalize`` ``:2437-2446``; ``_photometric_error`` ``:636-656`` (photon+sky+read only) |
| **2. Broeg iteration / variable comp inflation?** | **Partial.** ``pytics_iterative_weights`` (default **on**, ``config.json``) iteratively **inflates `comp_rms`** from per-comp residual scatter vs weighted ZP — Broeg-like, but on **stability RMS**, not per-frame photon sigma. ``check_comparison_stability`` MAD-filters high p2p comps (excludes/suspects), does not iteratively drop variables inside ``ensemble_normalize``. **Ensemble combination itself is flux-sum** (explicitly *not* Broeg-weighted — comment: 1/rms² deforms extinction slope). | ``:2409-2418``; ``pytics_iterative_weights`` ``:1821-1906``; ``check_comparison_stability`` ``:1914+`` |
| **3. Error bars validated (chi²/dof ~ 1)?** | **No production gate.** LC ``err`` = Howell photon+sky+read per frame; **not** propagated into ensemble weights; **no** chi²/dof check on constant stars in the Phase-2A export path (Mighell χ²-gamma cited export-only per ``VYVAR_MATH_PHYS_AUDIT.md``). DoD-B constant gate used no-regression + RMS ratio, not chi². | ``:1428-1434``; ``VYVAR_MATH_PHYS_AUDIT.md`` Mighell row |

**Outcome:** sigma **incomplete** for Broeg-canonical ensemble combine → **hold flux-sum for `delta_mag`**
(AI/diagnostic); **reporting `mag_calib` already uses partial Broeg (ZP offset only)**. Do **not**
promote inverse-variance **ensemble combine** until: (a) weights use validated per-frame sigma
(Howell + scintillation + inflation), (b) chi²/dof ~ 1 on a constant calibrator. That sigma fix
is load-bearing for GS8/MULTISET regardless.

**Citations added (sandbox, 2026-06-15):** `young1967`, `osborn2015`, `dravins1998`,
`murray2020speculoos` in `CITATIONS.bib`. Spec: `docs/VYVAR_SIGMA_BUDGET_SPEC.md`;
sandbox: `tmp/phase12/`.

### Sigma-budget work item — χ² audit + sandbox (2026-06-15)

**Read-only χ² audit:** Mighell (1999) is **export-only** (`citations.py` PSF block); no
production χ²/dof on constant stars. PSF `reduced_chi2` and trust `check_star_scatter` are
unrelated. **Verdict:** promote **new** reduced-χ²/dof gate (not Mighell χ²-gamma as-is).

**Sandbox shipped:** Osborn eq. (7) scintillation + Howell quadrature + Broeg inflation helpers;
chi² gate harness. **Not production** until χ²/dof ~ 1 on verified-constant calibrator.
`delta_mag` unchanged.

---

## draft_409 trust/consistency cleanup — Fixes 1-3 (2026-06-16)

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

**Validation:** draft_409 V0612 cross-validated vs SIPS — eclipse shape + single bright outlier at
~JD 2461200.385 match in both reductions (frame-level artifact, not VYVAR bug).
