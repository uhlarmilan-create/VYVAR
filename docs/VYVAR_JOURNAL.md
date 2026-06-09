Historical session log. Current state → VYVAR_STATE.md; decisions → VYVAR_DECISIONS.md; open work → VYVAR_ROADMAP.md.

---

## Session -- NEIGHBOR-SUB design recorded (2026-06-09)

Read-only design for TODO-PSF-NEIGHBOR-SUB: fit neighbour ePSF, subtract, aperture residual.
Worklist on corrected crowding (375/380 L: 58/53 blended, 39/34 hard). Insertion at per-frame
measurement (not `compute_lc_flux_method` router). `_load_adaptive_blend_map` needs full worklist
row extension. Doc `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md`. No code.

---

## Session -- TODO-FWHM-CONSISTENCY implemented (2026-06-09)

`header_core_fwhm_px` in `masterstar_context.py`; `crowding_index._load_wcs_meta` and
`psf_photometry.get_epsf_fwhm_from_context` now prefer VY_FWHM_GAUSS -> VY_FWHM_GAUSSIAN -> VY_FWHM.
h & chi Per 375/380 L live crowding: 58/53 is_blended, 39/34 hard; ePSF QC ratio ~0.78/0.81.
Numeric SHA 770966c3 unchanged; pytest 183/6. ROADMAP TODO-FWHM-CONSISTENCY closed.

---

## Session -- crowding recompute VY_FWHM_GAUSS 375/380 L (2026-06-09)

Read-only recompute: baseline crowding (VY_FWHM) self-check OK (77/87 is_blended). Corrected with
VY_FWHM_GAUSS (~2.73 px): 58/53 is_blended, 39/34 hard nn<1.0. **PROCEED** for NEIGHBOR-SUB on
h & chi Per. Diagnostic preceded production fix; doc `docs/VYVAR_HCHIPER_CROWDING_RECOMPUTE.md`.

---

## Session -- decisive ePSF FWHM test 375/380 L (2026-06-09)

Independent Moffat/Gaussian/azimuthal FWHM on built ePSF + stellar cutouts. **Dominant: EXPLANATION 3**
(OBS_FILES seeing ~3.84 px vs core ~2.0 px). EXPLANATION 2 rejected (ePSF ~= stars). Secondary:
buggy half-max at 2.236=sqrt(5). Doc `docs/VYVAR_EPSF_FWHM_TEST.md`. No production changes.

---

## Session -- ePSF path audit EPSF-1 (2026-06-08)

Read-only audit at fe8201c: `epsf_fwhm_native` half-max estimator (`psf_photometry.py:500-516`)
biases ratio<1 on h & chi Per -- diagnostic only, not flux/gating. Doc `docs/VYVAR_EPSF_AUDIT.md`;
ROADMAP `TODO-EPSF-1-FWHM-QC`; harness V3e added. No production changes.

---

## Session -- h & chi Per PSF probe drafts 375+380 (2026-06-08)

Read-only crowding + ePSF QC on solved MASTERSTAR data. Scale **~1.30"/px bin2** (not fine).
L-band richest (30 frames). blend_frac_1fwhm ~3.7-4.4%; 77-98 LC stars is_blended on L.
ePSF asymmetry ~0.001 (no smear flag); ePSF/input FWHM ratio 0.59-0.67. Doc:
`docs/VYVAR_HCHIPER_PSF_PROBE.md`. pytest 183/6 unchanged; production untouched.

---

## Overnight session — inject-and-recover validation harness (2026-06-08)

Built `tests/validation/` (gen_frame, gen_series, recover, score) wired to real VYVAR
entry points: crowding blend metrics, Sokolovsky indices, aperture photometry, trust gate,
color-term fit, BJD/airmass, calibration masters. Tier A single-frame + Tier B 60-frame
Gaia-structured series (fallback catalog). First full run: **14 pass / 2 fail / 2 skip**.
FAIL findings (not production fixes): A3 ePSF asymmetry on smeared cutout, A7 photutils vs
SEP ~0.7% flux offset. Docs: `docs/VYVAR_VALIDATION.md`, `tests/validation/README.md`.
Production photometry untouched; pytest 183 passed / 6 skipped.

---

Last updated: 2026-06-03 (session 03.6.2026 — APCORR-COLOR Path B: extrapolation guard block)

---

## APCORR-COLOR — prototype + extrapolation block (2026-06-03)

**Prototype (draft_000366, read-only):** 141 targets in `ct_prototype.csv`. Median c1 −0.07
(−0.36 nonzero); median |ct_corr| 0.019 mag (69 >0.02, 42 >0.05); cat−inst scatter
0.078→0.053 mag; `gate_would_pass` 15/141 (10.6%). Worst |ct_corr| ~4.8 mag on red targets
with BP-RP outside comp range. NoFilter production unchanged.

**Path B fix:** `_check_color_term_extrapolation` → bool; out-of-range blocks CT (warn kept),
`ct_ok=False`, target kept uncorrected. Config `phase01_ct_extrapolation_tol` (default 0).
NoFilter skip in `should_apply_color_term` untouched.

---

Last updated: 2026-06-02 (session 02.6.2026 — cross-validation draft_000365: V842 spot-check + whole-night 143 targets via xval_run.py)

---

## Cross-validation — draft_000365 (V842 Her, EW) — photutils + SExtractor vs VYVAR  [2026-06-02]

Independent end-to-end check of VYVAR aperture photometry against two outside engines.
Each built its OWN Gaia DR3 catalogue from the frame WCS, own detection, own apertures and
background; only the input FITS are shared with VYVAR. Compared via the SAME unweighted
leave-one-out differential method against VYVAR per-frame `dao_flux` (`proc_*_Light_*.csv`)
and reported `lc_rms` / `comp_rms`. 127 aligned NoFilter_60_2 lights, 9.77"/px wide field,
FWHM ~3.0px. Join key = Gaia `catalog_id`. Target V842 Her (EW) 1400549875578714240,
lc_rms 0.1696, 8 TIER1 comps.

| engine              | comp RMS | target RMS | per-frame vs dao_flux |
|---------------------|----------|------------|-----------------------|
| photutils (annulus) | 0.0143   | 0.1709     | 0.0102 mag            |
| sep / SExtractor    | 0.0105   | 0.1706     | 0.0022 mag            |
| VYVAR dao_flux      | 0.0104   | 0.1713     | —                     |
| VYVAR reported      | 0.0117   | 0.1696     | —                     |

Conclusions:
- Science number reproduced by THREE engines to ~1% (0.1696–0.1713). No systematic offset.
- sep matches VYVAR extraction to 0.2%/frame and 0.0105 vs 0.0104 comp RMS → VYVAR aperture
  photometry is equivalent to a SExtractor mesh-background pipeline.
- The 25% photutils comp-RMS gap is fully explained: photutils local-annulus background +
  centroid_com added ~0.010 mag/frame; sep (mesh background) reproduces VYVAR. Budget closes:
  sqrt(0.0104² + 0.0102²) = 0.0146 ≈ photutils 0.0143.
- VYVAR `comp_rms` (0.0117) sits just above the 0.0104–0.0105 floor — slightly conservative (safe).
- Big aperture (2×FWHM) independently confirmed VYVAR's small SNR-optimal aperture: at 9.77"/px a
  bright neighbour falls in the annulus (3/8 comps flagged, ratio up to 2.08); small aperture avoids it.
- Alignment independently confirmed: median centroid drift 0.39px, stable across all 127 frames.

### Whole-night — all 143 targets (harness `xval_run.py`)

Ran the consolidated harness on the full draft (photutils + sep extraction of 1108
sources incl. all 143 targets + their comps; decomposed vs VYVAR `dao_flux` + `lc_rms`).
Regression: V842 Her reproduced exactly (phot 0.1709 / sep 0.1706 / dao 0.1713 / VYVAR 0.1696).

| engine vs VYVAR lc_rms | N | median ratio | median \|Δ\| | within ±15% |
|------------------------|---|--------------|-----------|-------------|
| dao (my diff on VYVAR flux) | 140 | 0.93 | 0.0052 | 70% |
| sep / SExtractor (independent) | 133 | 0.97 | 0.0082 | 60% |
| photutils (annulus) | 133 | 1.22 | 0.0332 | 35% |

`comp_rms` sep/dao ratio = 1.055 (N=142) → independent SExtractor reproduces VYVAR
extraction precision across the whole night, not just one target.

Whole-night conclusions:
- Independent pipeline reproduces VYVAR per-target RMS to ~3% (median) over 143 targets.
- VYVAR `lc_rms` is consistent with, and slightly conservative vs, the raw differential
  floor (dao ratio 0.93) — VYVAR never under-reports its noise (safe for the database).
- photutils-annulus is NOT a reliable independent witness on crowded/faint targets
  (ratio 1.22, inflates); sep is. Independently justifies VYVAR's robust background + small aperture.
- ~8 low-confidence targets (e.g. Gaia 1399954352593040512, V1138 Her, Gaia 1403010994918399232):
  sep's OWN extraction blows up while dao≈VYVAR → VYVAR flux is fine, just not independently
  confirmable with a quick aperture pipeline (faint+blended). Flag for caution pre-AAVSO.

IRAF/PyRAF: CLOSED as unnecessary — two independent engines (one matching VYVAR to 0.2%) already
validate extraction; legacy IRAF adds no independent axis and is not feasible on Py3.12/Ubuntu24.

Harness gotchas (for `xval_run.py`): isolation must be a PHYSICAL bright-neighbour test (within
annulus r_out AND within ~2.5 mag), not raw nearest-neighbour on a deep catalogue; force-include
VYVAR comps+target; Gaia TAP returns SOURCE_ID upper-case (normalise case); join catalog_id as 19-digit string.

Harness `xval_run.py` validated on full draft_000365 (143 targets); reusable for other drafts.

---

## Reporting & export overhaul — COMPLETED (P1–P3.5)

- **P1 data correctness:** AAVSO MTYPE fixed DIFF→STD (TRANS=NO; every prior AAVSO file was mislabeled DIFF); validator no longer warns on STD+NO. Universal table-driven AAVSO FILT map (full code set: U/B/V/RJ/IJ/Rc/Ic/Sloan SU–SZ/CV/CR/TB/TG/TR/J/H/K/Y) + config override (`aavso_filter_map`); unknown filter → FILT=UNKN + `#WARNING` (no silent CV). Honest method label (`meth=` in `#SOFTWARE`/NOTES; was hardcoded `"aperture"`). OBSCODE guard (`#WARNING` when empty/default UMIA).
- **P2 citations:** now sourced from `CITATIONS.bib` (28 entries) via a single **conditional** emitter (`citations.py`) shared by AAVSO export + VarAstro export + PDF Methods. Cites only methods that actually ran in the run (e.g. Anderson&King/Moffat only when PSF on). Added Eastman 2010 (BJD), VSX/Watson, astroquery/numpy/scipy, + conditional refs.
- **P3 separate per-method reports:** aperture (always) + psf/adaptive (when enabled), one method per LC/report, suffixed (`_psf`/`_adaptive`) + labeled. Aperture-only default is **byte-stable** (verified SHA-256 on 362) — no disruption to current workflow.
- **P3.5 KMAG:** now the **measured ensemble-standardized check-star magnitude** (per-row via `check_kmag_{target}.csv` sidecar; check star excluded from its own ensemble); was `na`.

**ROUTING (confirmed intended):** eclipsing → VarAstro (LC); pulsating/all → AAVSO.

**NEW-DRAFT READINESS:** AAVSO Extended + VarAstro auto-produced after Phase 2A; manual upload.

**USER ACTION:** set `aavso_observer_code` (default UMIA placeholder warns).

**OPEN (optional, low priority):** B.R.N.O. minima (Kwee–van Woerden) for eclipsing; AAVSO API auto-upload (TODO-GS10); 4 KMAG=na targets on 362 (insufficient comps).

---

## VYVAR SESSION SUMMARY — 30.5.2026 (parameter provenance + setup UX)

### COMPLETED 2026-05-30

- **[PARAM-PROVENANCE]** Single authoritative parameter resolver across all stages +
  per-draft observer-site fix (kills the config-drift trap), plus the setup UX layer
  (default markers, Scan-Source optics auto-detect, poor-FITS prompt). Two commits:
  `bd4a539` (Phase 1, production correctness) + Phase 2–4 (UX layer).
  - **Resolver (`param_resolver.py`), two classes + site:**
    * equipment-intrinsic (gain, read-noise, pixel, focal, saturation):
      **DB-set(valid) → header(cross-check warn) → config**. The DB cross-check rejects
      *plausible-but-wrong* headers a sanity range would accept — proven by 363
      `XPIXSIZE=10.0` rejected in favour of DB `3.76`.
    * observation-specific (binning, exptime, pointing, time): header → DB → config.
    * site (lat/lon/elev): **per-draft `ID_LOCATION` → header `SITELAT/LONG/ELEV` →
      config (FLAGGED, never silent)**; unresolved → `ok=False` → Phase-4 prompt.
  - **Divergent paths eliminated** (gain Phase-2A-DB-only vs error-map header-first;
    site time_utils-FITS-first vs Phase-2A-config-only) — every consumer routes through
    the one resolver: BJD, airmass, per-frame error-map, lunar context, SNR-table,
    `crowding_index`, QA detector readout, AAVSO `#LATITUDE/#LONGITUDE/#ELEVATION`,
    VAR.ASTRO `#Site`, PDF Observer Location (submitted coords now tied to the BJD site).
    Documented deep fallback: `pipeline.py` worker-meta `cfg.observer_*` (worker already
    resolves draft→header first).
  - **VALIDATED (config stays Jirny throughout):** 360/361/362 → **Jirny**, 363 →
    **Dablice**, each from its own `ID_LOCATION`; forcing `config.json` to a **bogus
    Sydney** location leaves every draft unchanged → BJD/airmass are now
    **config-independent**. 363 BJD shift **0.0402 ms** vs the old config-only path
    (360 = 0.0000 ms, Jirny == config). Pre-commit sanity: gain/RN/pixel **identical**
    across old-error-map / old-Phase-2A / resolver for 360 and 363 (DB was already used
    on this data) — no science movement beyond the negligible site shift.
  - **config.json** `observer_location` is now **UI / last-session state only — moot for
    the science** (BJD/airmass derive from per-draft `ID_LOCATION`). Not changed by this work.
  - **Phase 2 schema (DB):** `IS_DEFAULT` on `EQUIPMENTS`/`TELESCOPE`/`LOCATION` + `ACTIVE`
    on `LOCATION` (idempotent migrations); `set_table_default`/`get_default_id` + editor
    exclusivity (exactly one default each). Seeded **QHY294MM(1) / Carl-Zeiss 200mm(1) /
    Jirny(2)** — explicit user markers, not a silent `id=1` fallback.
  - **Phase 3 (`optics_autodetect.py`):** fingerprint camera (INSTRUME + full-res sensor
    dims + GAIN), telescope (FOCALLEN+APTDIA), site (SITELAT/LONG); `TELESCOP` treated as a
    useless sample string. 360 → QHY294MM 0.90 / Carl-Zeiss 0.90 / Jirny 0.90 (all high);
    363 → C3-26000 0.55 (medium; 6252×4176 dims + GAIN 0.78, **no name**), telescope/site
    unmatched. Confident matches auto-fill the Scan-Source selectors (override default);
    user can still override.
  - **Phase 4 (`assess_unresolved`):** surfaces ONLY the gaps, pre-filled from default.
    363 → prompts Telescope + Observer site + Pointing; 360 → none.
  - **UI:** Scan-Source selectors pre-select the `IS_DEFAULT` rows; library editor exposes
    `IS_DEFAULT` checkbox; poor-FITS prompt panel.
  - **Migration scope:** backfilled ONLY library-table markers — **zero `OBS_DRAFT` rows
    modified** (frozen per-draft `ID_EQUIPMENTS/ID_TELESCOPE/ID_LOCATION` left intact:
    360–362 Jirny, 363 Dablice).
  - Files: `param_resolver.py`, `optics_autodetect.py` (new); `time_utils.py`,
    `photometry_core.py`, `pipeline.py`, `crowding_index.py`, `ui_quality_dashboard.py`,
    `export_reports.py`, `photometry_report.py` (Phase 1); `database.py`, `app.py`,
    `ui_database_explorer.py` (Phase 2–4).

### NEW TODO — parameter provenance

- **[TODO-PIXEL-XCHECK-BINNING]** `param_resolver.resolve_pixel_um` cross-check compares the
  raw header `XPIXSZ` against the DB **native** `PIXELSIZE`, so it warns on a legitimately
  *binned* pixel (360: header `XPIXSZ=9.26` = 2×2 of native `4.63` → spurious ">5% disagrees"
  warning). It still correctly uses the DB native value, so **no LC impact** — purely a noisy
  log. Fix: make the cross-check binning-aware (divide header pixel by `XBINNING`/`YBINNING`
  before comparing like-to-like), mirroring the binning-aware logic already in
  `optics_autodetect.detect_equipment`. LOW / cosmetic.

- **[TODO-CONFIG-CHURN]** The app re-serializes the **tracked** `config.json` every run
  (rewrites last-used session state — site/rig — into the same file as the static
  overrides) → perpetual git diff that has to be `checkout`-discarded each time. **Zero
  functional effect** (the resolver ignores the config site; the UI uses
  `LOCATION.IS_DEFAULT`; the rewritten `observer_location` is vestigial). Durable fix:
  **separate the session-state the app rewrites from the static user overrides** (e.g. a
  distinct `session_state.json` / UI-state store) so `config.json` stops drifting. **Do
  NOT gitignore `config.json`** — it still holds real user overrides. LOW.

- **[TODO-BROAD-EXCEPT-HYGIENE]** Broad `except Exception: pass` / `: continue` wrapping a
  **fallback / protection / degradation path** can swallow **programming** errors
  (`NameError` / `AttributeError` / `TypeError`) → the safety net fails *silently*, so an
  unusual/new set hits the un-protected path with no warning. 4 such instances were already
  found & fixed: `app_config` access, the optics FOV override, header-median pointing, and the
  association-slice rebuild (all were dead-on-arrival behind a bare except). **Likely more
  exist** (~700 bare `except: pass/continue` codebase-wide; the dangerous subset is the ones
  guarding a safety/fallback path in the core runtime: `pipeline.py`, `vyvar_platesolver.py`,
  `photometry_core.py`, `comp_selection_per_target.py`, `psf_photometry.py`, `param_resolver.py`,
  `optics_autodetect.py`, `calibration.py`). Fix pattern: **narrow each protective except to the
  EXPECTED runtime exceptions only** (e.g. `(KeyError, ValueError, TypeError)` for header parses,
  `(OSError, FileNotFoundError)` for IO) so `NameError`/`AttributeError` surface instead of
  disabling the protection invisibly. MEDIUM (do as a data-independent sweep). LOW risk per edit.

- **[APCORR-COG]** Per-frame curve-of-growth (encircled-energy) aperture correction —
  config-gated, **default OFF**. Fixes STEP-3 (audit): per-star SNR-optimal radii feed
  `dao_flux` with **no** enclosed-fraction correction → constant target↔comp differential
  bias + seeing-correlated systematic. New `compute_per_frame_cog_correction` builds a
  per-frame EE(r) from bright/isolated/unsaturated/high-SNR stars (isolation 6×FWHM, SNR≥50,
  peak<0.85·sat, capped 60), `ac_factor = 1/EE(r_star)` puts every star on the common
  `cog_ref_fwhm×FWHM` (4.5) ref-radius scale. Emits `dao_flux_apcorr`/`ac_factor`/`cog_ok`
  (never overwrites `dao_flux`); routed into `ensemble_normalize` via `read_flux_from_csv`
  `use_apcorr_flux` only when enabled + `cog_ok`. New config keys `cog_aperture_correction_*`
  (all gated OFF). Files: `config.py`, `photometry_core.py`, `pipeline.py`.
  - **Validation 360/363 (OFF vs ON):** target↔comp enclosed gap **23%/34% → 0.000%**.
    363 (variable seeing): robust quartile (n=178 vs 89) seeing slope **+0.037 → −0.006
    mag/px (−85%)**, p2p **40.7 → 13.9 mmag (−66%)**, comp LC RMS **19.1 → 17.4 mmag (−9%)**.
    360 (stable seeing): comp RMS unchanged (harmless). Gate byte-identical when OFF
    (proc CSV emits no apcorr cols → reader uses raw `dao_flux`). Forced fallback degrades
    gracefully (`ac_factor=1`, `cog_ok=False`, no crash). NB: stable-comp radius span was
    tiny (0.15px, all near the faint floor) yet still gave OFF corr +0.94 → real benefit is
    **larger** for target/comp pairs with bigger radius gaps (extreme-group test ~768 mmag).

- **[CROWDING-CLASSIFIER]** Detection-independent signal classifier + sampling-gated
  TIGHTEN — config-gated `crowding_classifier_enabled`, **default OFF** (zero production
  change; OFF == legacy stars/Mpx path). Replaces the erratic detection/scale-locked
  **stars/Mpx** density class with detection-independent `crowding_index` signals
  (gaia/arcmin² + `blend_frac` @ measured depth + comp-availability + `frame_limit_mag`).
  Decouples the two concerns the single legacy class conflated: **LOOSEN** keys on comp
  **availability** (few usable catalog comps in FOV), **TIGHTEN** on real **blend_frac**
  (contamination @ depth); both fire independently, shared keys sum additively.
  - **SAMPLING GATE:** TIGHTEN fires only when the PSF is resolved (**FWHM ≥ 3 px**). On
    undersampled fields the comp-RMS 0.08–0.10 tail is the **field floor**
    (scintillation/undersampling), not resolvable contamination, so tightening
    `max_comp_rms` there only cuts good comps. Never fires on the wide rig (FWHM ≈ 2.5–2.6).
  - Removed dead `aperture_fwhm_factor -0.3` from the legacy `dense` override (science
    aperture comes from the SNR-optimal table, which ignores it; verified 0-px effect).
  - **A/B VALIDATED — 361/362 OFF (legacy DENSE→tighten) vs ON (gated→no tighten)** (plate
    scale resolves ~9.77″/px both runs, fixing the stale-1.3 A4 artifact):
    * **360** (FWHM 2.59, legacy `normal`): **neutral by construction** — gate suppresses
      tighten and legacy `normal` doesn't tighten either. OFF==ON.
    * **361** (FWHM 2.47): ON **recovers** — robust LC scatter 0.07316→**0.07179**
      (−1.4 mmag), median lc_rms 0.07817→**0.07498** (−3.2 mmag); keeps the 0.08–0.10
      comp band (19 pairs vs 0).
    * **362** (FWHM 2.65): ON marginally **worse** — robust scatter 0.09876→0.09915
      (+0.4 mmag, sub-mmag), p2p +1.4 mmag, lc_rms +3.0 mmag. Floor-limited,
      comp-geometry-dependent → diminishing returns; not investigated further.
  - **DECISION: committed gated infra, NOT enabled for the wide rig.** The wide rig is
    **floor-limited** (scintillation/undersampling, FWHM 2.6 px); the tighten's payoff is
    **well-sampled crowded data — enable on the Newton cluster**, not here. The gate is
    correct in principle (prevents the demonstrated 360-style over-tightening harm).
  - Files: `config.py` (config keys + `CROWDING_{LOOSEN,TIGHTEN}_OVERRIDES` +
    `apply_crowding_overrides`), `crowding_index.py` (now consumed by the gated classifier),
    `photometry_core.py` (gated block in `run_phase0_and_phase1`; `db`/`draft_id` threaded).

### NEW TODO — aperture correction

- **[TODO-APCORR-MIXEDFRAME]** Before enabling on **sparse/cloudy** nights: wire a nightly-
  median `fallback_ee` (median EE from `cog_ok=True` frames → applied to fallback frames,
  flagged) so a draft never mixes corrected + uncorrected frames. `cog_ok` is per-frame; a
  mix would inject a cross-frame step (~target↔comp bias). Hook (`fallback_ee`) exists but is
  not yet wired in the pipeline. **Cannot occur on star-rich fields** (360/363: 261–320
  eligible stars/frame, 0 fallbacks). Alternative: draft-level gate (require all-frames
  `cog_ok`). MEDIUM.
- **[TODO-APCORR-COLOR]** (audit STEP 7) NoFilter↔Gaia-G color term **c1 ≈ −1.0 mag/(bp_rp)**;
  CT correction (`mag_calib_ct`/`fit_color_term_c1`) frequently inactive (`ct_ok=False`).
  Activate it robustly or tighten comp color-matching so residual color dependence (comp
  cat−inst std ~0.12–0.16 mag) is removed. MEDIUM.
- **[TODO-WIDE-RIG-REPROCESS]** 361/362 **production** `photometry/` still carry
  **stale-1.3-scale** LC products (A4) — the A/B reprocess wrote only isolated dirs (since
  removed). Reprocess 361/362 with the wide rig when convenient to refresh the production
  LCs at the correct ~9.77″/px scale. LOW / housekeeping.

---

## VYVAR SESSION SUMMARY — 29.5.2026 (plate-scale fix + crowding index + PSF grouper)

### COMPLETED 2026-05-29

- **[PLATESCALE-FIX]** Resolver made WCS/CD-authoritative (config last resort); sane clamp
  widened `0.3–5.0` → `0.1–30.0`. Fixed `_resolve_plate_scale_arcsec_per_px`,
  `_read_plate_scale_from_fits_path`, `_get_plate_scale_from_cfg`
  (`photometry_core.py`), `psf_photometry._read_plate_scale_arcsec_px_from_fits`, and
  routed `tess_verify` through the fixed reader. 362 `MASTERSTAR` `VY_PLTS` hygiene
  `1.3 → 9.768` (rewritten from its own CD matrix; WCS/CD keys untouched).
  - **CONFIRMED real scale = 9.77″/px** (200 mm + 9.26 µm binned). `VY_PLTS=1.3` was a
    GLOBAL config placeholder = Newton 300/1200 + C3-26000 binned 2× (~1.29″/px) leaking
    onto the wide-field set.
  - Blast radius (now correct): GS11 context, FOV / `max_dist_deg`, ePSF isolation, TESS.
  - Pixel-based geometry (aperture / annulus / SNR-optimal table / `field_density`) was
    IMMUNE and confirmed unchanged.
  - Validation (360/361/362): all three resolvers return ~9.77 after; ePSF 362 rebuilt
    with corrected isolation (~98″ vs old ~13″), `n_stars_used` 304→296; FOV
    `max_dist_deg` 0.34° → ~2.55°; `field_density`/`density_class` identical pre/post.

- **[CROWDING-INDEX]** Built parallel detection-independent `crowding_index.py` +
  `run_crowding_index.py` (NOT wired into the pipeline). Depth-aware: frame limit
  (SNR=5 Howell), Gaia footprint density/arcmin², blend fraction, miss decomposition
  (below_depth / blend_miss / threshold_miss), per-target blend worklist
  `crowding_targets.csv`.
  - FINDING: old `field_density` (Gaia-matched DAO count) is detection-limited →
    undercounts dense fields (blends merge). `threshold_miss ≫ blend_miss` ⇒ iterative
    DAO is the big completeness lever; PSF deblend is only 1.4–4.1% of the field but
    ~19% of *targets*. Hercules (362) is intrinsically denser than the CVn fields —
    OPPOSITE of what the old metric reported.

- **[PSF-GROUPER]** `SourceGrouper` joint-fit implemented, **DEFAULT OFF** (config
  `psf_grouper_enabled` / `psf_group_sep_fwhm` / `psf_neighbor_include_fwhm`). Offline
  362 ensemble test (corrected ePSF) shows it DEGRADES: blended median ratio
  1.10→1.70, CSS_J161519.8 1.56→20.4. Blends at ~1.5 FWHM (~5 px at 9.77″/px) are
  sub-resolution ⇒ joint fit ill-conditioned (3.38% divergence, fallbacks). Single-star
  PSF on a clean model ≈ aperture here. Re-test on Newton 0.65″/px data where blends are
  resolved.

- **[PSF-SPATIAL]** Spatially-varying ePSF (`GriddedPSFModel`) implemented, **DEFAULT OFF**
  (config `psf_spatial_enabled` / `psf_spatial_grid="3x3"` / `psf_spatial_min_stars_per_cell`).
  New `psf_photometry.build_epsf_grid_model` (full-frame CSV candidates, per-cell EPSFBuilder,
  `grid_from_epsfs`, per-cell fallback flagging) + `interp_gridded_epsf_array` + gated
  `gridded_model=` path in `psf_photometry_stars` (per-(x,y) interpolated ePSF; single path
  untouched when OFF). Refactored shared `_epsf_prepare_stars` / `_epsf_build_imagepsf_from_stars`.
  - VALIDATION (offline ensemble, 360/361/362, 9.77″/px, 3×3, 0 fallback cells, full-frame
    coverage, n≈90–280/cell): **gridded does NOT win**. By field region, median
    RMS(`mag_calib`) ratio **gridded/single = 1.06–1.59 everywhere** (gridded always worse);
    **gridded/aper ≥ 1.0** in essentially all bins. Single ePSF beats aperture only on 362
    (cleanest field): comp/edge single/aper **0.73** (0.0149 vs 0.0203), comp/centre 0.83;
    on 360/361 single/aper > 1. Variable targets ≈ flat (~1.0) for all methods.
  - WHY: at 9.77″/px the PSF is well-sampled and stable across the 5.7° field, so a single
    ePSF already captures it; subdividing into cells just starves each ePSF of stars and adds
    scatter. **Decision: keep `psf_spatial_enabled` OFF.** Revisit only on finer optics where
    field-dependent aberration (coma/curvature) actually varies the PSF across the chip.

- **[EPSF-CENTER-BUGFIX]** Fixed a latent ePSF-build bug uncovered by the spatial work: the
  per-cutout sky-sub *re-extraction* in `_epsf_prepare_stars` (ex `build_epsf_model`) treated
  `EPSFStar.center` as `(y, x)`, but the installed photutils returns `(x, y)`. The transpose
  pushed every star with `x > ~1388` (the chip-height axis = 1397) out of bounds → silently
  dropped, confining all ePSF candidates to a 1388×1388 box (left/upper region only). Now
  `(x, y)` — global ePSF candidate coverage is full-frame; build still produces a sane model
  (362 `n_stars_used`=296, asym=0.004). Affects the production global ePSF build (improvement).

- **[PSF-ISOLATION-FIX]** ePSF candidate isolation now compares each candidate against the
  **FULL Gaia cone catalogue** (`field_catalog_cone.csv`), not just the other candidates.
  New `psf_photometry._load_cone_catalog`; `_epsf_prepare_stars` rejects a candidate if any
  cone source lies within `3×FWHM` (correct 9.77″/px) **and** is within `Δmag ≤ 2.5` (brighter,
  or ≤2.5 mag fainter = contaminating). Candidate `mag` carried into the candidate frame; falls
  back to candidate-vs-candidate only if the cone CSV is missing. **This is a correctness fix to
  the production ePSF build.**
  - VALIDATION (rebuilt 360/361/362): old candidate-vs-candidate barely filtered
    (360 334→330), the cone test correctly drops bright-neighbour candidates:
    **360 334→213 (−35 %), 361 292→148 (−49 %), 362 304→161 (−46 %)**; models stay sane
    (asym 0.003–0.006, nan 0). Cleaner isolation did **NOT** move single-PSF past aperture:
    comp-star median RMS single-PSF/aper = **3.47 / 2.70 / 3.72** (360/361/362) — single ePSF
    is still ~3× worse than aperture at 9.77″/px (confirms the well-sampled-stable-PSF picture;
    aperture wins on coarse optics).

- **[PSF-QUALITY-FALLBACK]** Per-star PSF quality + auto-fallback + residual QA (the RMS-20.4
  safety lesson). `psf_photometry.assess_psf_quality` grades every fit `good/marginal/bad` from
  reduced χ², fit SNR (flux/flux_err), fitted-position shift (in FWHM) and nearest-neighbour
  proximity (`nn_dist_fwhm` + neighbour Δmag; a close *bright* neighbour ⇒ bad). New proc-CSV
  columns from `psf_photometry_stars`: `psf_quality`, `psf_quality_fallback`, `psf_snr`,
  `psf_pos_shift`, `psf_nn_dist_fwhm`. Quality is **always computed**. New config
  `psf_quality_fallback_enabled` (**default TRUE**): a `bad` fit drops `psf_fit_ok` and sets
  `psf_quality_fallback=True` so the caller substitutes aperture — a bad PSF flux can never
  silently become the reported value. **No production wiring yet** (Phase-2A still aperture).
  - VALIDATION (offline ensemble, 127/147/93 frames):
    measurement grades ≈ 28 % good / 43 % marginal / 29 % bad; target stars bad-majority
    66/57/56, marginal-majority 60/43/44 (360/361/362).
    **CSS_J161519.8+491001 (neighbour 3.5 mag brighter at 1.46 FWHM) ⇒ `bad` in 127/127 frames.**
    Fallback cuts PSF-specific blowups: e.g. 362 target `…0227968` single-PSF RMS **0.747 → 0.023**
    (≈ aperture 0.016); 360 max target RMS **1.28 → 0.97**; #targets with PSF RMS >1.5× aper
    **off→on: 360 82→65, 362 35→20**. (Residual-QA PNG + summary per draft in
    `…/psf_robust_qa_d<draft>/`.) Where a star is already bad in *aperture* (intrinsic variable),
    per-epoch fallback ≈ aperture — the safe floor, as intended.

- **[PSF-WIRING-ADAPTIVE]** Fixed the dead `psf_photometry_enabled` toggle + added a gated
  per-star adaptive flux selector. **Both default OFF → production stays pure-aperture.**
  - ROOT CAUSE: Phase-2A's flux reader (`read_flux_from_csv`) + the PERF-8 `_flux_matrix`
    cache carried ONLY `dao_flux`, so `psf_flux` never reached `_get_lc_psf_or_dao` →
    `psf_photometry_enabled` was a no-op. FIX: the reader now also carries per-star/per-frame
    `psf_flux`, `psf_fit_ok`, `psf_quality`, `psf_quality_fallback`, `psf_snr` (b.5 columns;
    default NaN/False when absent → no behaviour change). Confirmed end-to-end: with PSF run,
    31k–42k/frame-rows carry finite `psf_flux` and `_get_lc_psf_or_dao` now uses it.
  - ADAPTIVE SELECTOR (`compute_lc_flux_method` + `_get_lc_adaptive`, config
    `psf_adaptive_enabled` / `psf_adaptive_resolve_fwhm=2.0` / `psf_adaptive_snr_lo=15`):
    per-star/per-frame choice ∈ {aperture, psf}, CONSERVATIVE (default aperture; → PSF only
    with positive evidence + good quality). Rules: (1) bad/`!fit_ok`/no-flux → aperture;
    (2) resolvable blend (`is_blended ∧ nn_dist_fwhm ≥ resolve_fwhm`) → psf; (3) faint
    (`SNR ≤ snr_lo`) ∧ quality good → psf; (4) else aperture. Choice emitted as
    `lc_flux_method`. Blend map loaded best-effort from `crowding_targets.csv`.
  - VALIDATION (offline ensemble, 3 modes, 360/361/362):
    | mode | comp RMS (360/361/362) | targ RMS |
    |---|---|---|
    | aperture-only | 0.0118 / 0.0105 / 0.0105 | 0.0882 / 0.0628 / 0.0847 |
    | PSF-everywhere | 0.0437 / 0.0377 / 0.0318 | 0.1181 / 0.0807 / 0.0926 |
    | **adaptive** | **0.0118 / 0.0105 / 0.0105** | **0.1040 / 0.0628 / 0.0908** |
    Method split → PSF: comp **0.0–0.34 %** (bright flat comps stay aperture), targ 6.8–17.6 %.
    **comp adaptive == aperture exactly** (never hurts the flat stars) and adaptive ≪
    PSF-everywhere in every cell (smarter than blindly forcing PSF). On targets adaptive ≤
    aperture on 361 (==), slightly above on 360/362 — the rule-3 faint picks where single-PSF
    still loses at 9.77″/px (the rule pays off at fine scale, as expected; rule 2 barely
    fires — blends unresolvable here). **CSS_J161519.8 → aperture in 127/127 frames**
    (quality `bad` ⇒ rule 1), RMS == aperture (0.061), never worse.
  - DECISION: keep both flags OFF at 9.77″/px (PSF doesn't beat aperture). The wiring +
    selector are correct and ready to pay off on fine-scale Newton data.
  - CONFIG CORRECTION: `config.json` had `psf_photometry_enabled=true`, which was a SILENT
    no-op only because the reader dropped `psf_flux`. Now that the wiring carries it, leaving
    the flag on would have flipped production to PSF-everywhere (worse). Set
    `psf_photometry_enabled=false` so production genuinely stays pure-aperture.

### CORRECTIONS to prior STATE

- **Plate scale is 9.77″/px, NOT 1.3** (project-wide belief was wrong).
- **"359 rebuilt-1.3 good / 360-361 stale-9.55 bad" was WRONG**: ePSF quality tracks
  CROWDING, not plate scale. 359 is unreliable (ePSF overwritten during debugging) →
  DROPPED. Trust 360/361/362.
- **"No finer set exists" was WRONG**: focal length is in the `TELESCOPES` table; Newton
  300/1200 + C3-26000 = ~0.65″/px (cluster-capable).

### NEW TODO — PSF roadmap (priority order)

- **[TODO-PSF-SPATIAL]** `GriddedPSFModel` implemented, gated default OFF. LOSES to single
  ePSF in all regions on 9.77″/px (marginal sampling → little spatial variation to capture;
  3x3 grid → per-cell star starvation). Low priority: narrow Newton FOV has minimal spatial
  variation either. Kept for completeness.
- **[TODO-PSF-WIRING]** `psf_flux` → Phase 2A wiring + per-star adaptive aperture/PSF
  selection **DONE** (see `[PSF-WIRING-ADAPTIVE]`). Remaining: neighbor-sub branch
  (`[TODO-PSF-NEIGHBOR-SUB]`) + actually running PSF in production (still default OFF).
- **[TODO-PSF-NEIGHBOR-SUB]** Neighbor-subtracted aperture: fit + subtract bright
  neighbour ePSF, aperture the residual (deblend that works at coarse resolution, unlike
  the grouper).
- **[TODO-PSF-MULTIFRAME]** Multi-frame ePSF build (all frames of the night).
  Isolation `candidate-vs-full-Gaia` part **DONE** (see `[PSF-ISOLATION-FIX]`); only the
  multi-frame stacking remains.
- **[TODO-PSF-QUALITY]** **DONE** — see `[PSF-QUALITY-FALLBACK]` below (quality flags +
  auto-fallback + residual QA). Remaining: realistic per-star uncertainties.
- **[TODO-OIS]** Difference imaging (Alard & Lupton OIS) — dense clusters with fine optics.
  Long-term.
- **[TODO-PLATESCALE-PERSET]** Resolver: use `TELESCOPES.focal` × camera pixel as a per-set
  plate-scale fallback (between WCS and config).
- **[WATCH]** comp-selection: validate local vs wide comps before the next production run
  uses the now-correct ~2.55° `max_dist_deg` (wide-field distant-comp systematics risk).

---

## VYVAR SESSION SUMMARY — 28.5.2026 (časť 4 — pipeline + PSF finalizácia)

## skip_processed_directory — Fáza 1 (commit 7db6914)

### Architektúra zmena

**Starý flow:**
Raw → Calibrated → Processed (QC + kópia FITS) → Aligned → proc_*.csv → LC

**Nový flow (skip_processed=true):**
Raw → Calibrated → QC in-place (VY_* headers na calibrated) → Aligned → proc_*.csv → LC

### Čo sa zmenilo

| Zmena | Detail |
|-------|--------|
| `skip_processed_directory` | Nový config bool (default `false`) |
| `_qc_enrich_calibrated_in_place` | QC headers písané in-place na calibrated FITS |
| `_get_vy_qc_status` | Helper pre VY_QC=ok filter pri alignment |
| `_archive_preprocess_lights_root` | Routing calibrated/ keď skip=true |
| `astrometry_align_and_build_masterstar` | Filter VY_QC=ok z calibrated |
| `resolve_masterstar_input_root` | Preferuje calibrated/ keď skip=true |
| `find_qc_metrics_csv` | Hľadá qc_metrics.csv v calibrated/ aj processed/ |
| `qc_fwhm_limit`, `qc_elong_limit` | Nové config polia (8.0, 1.8) |
| Temporal sigma clip | Zámerne vynechané z nového flow |

### Úspora (per draft)

- ~1.5 GB diskového priestoru (processed/ adresár)
- ~20–30% rýchlejší pipeline (vynechanie kopírovania 139 FITS)

### Fáza 2 (budúca)

- Odstrániť starý processed/ kód po validácii nového flow
- Otestovať kompletný run s skip_processed_directory=true

## PSF pipeline — finálny stav (28.5.2026)

### Výsledky validácie draft_359

| Metóda | RMS median | Fit rate | Záver |
|--------|-----------|----------|-------|
| Aperture (DAO) | 0.080 mag | ~100% | ✅ Primárna |
| ePSF | 0.115 mag | 7.1% | ⚠️ Crowded fields |
| Moffat | 0.143 mag | 1.4% | ❌ Asymetrický PSF |

### Kľúčové zistenia

- BO CVn pole má asymetrický PSF (tracking smear) → symetrický Moffat/ePSF nevyhrá
- PSF metódy budú lepšie pre iné noci / crowded fields
- Moffat AC faktor = 0.806 (flux outside cutout ~20%)
- alpha median = 5.67 (fyzikálne rozumné pre seeing-dominated PSF)

### Open TODOs — PSF

| ID | Popis | Priorita |
|----|-------|----------|
| TODO-PSF-ASYMMETRY | Tracking smear diagnostika — elongation angle/ratio per frame | HIGH |
| TODO-PSF-SATURATION | Auto-exclude saturated stars z Moffat fit ✅ DONE | ✅ |
| TODO-PSF-NEIGHBOR-SUB | PSF-cleaned aperture (ALLFRAME štýl) | MEDIUM |
| TODO-PSF-SPATIAL | spatial_order=1 pre Newton (čaká na TODO-MULTISET) | LOW |
| TODO-PSF-MULTIFRAME | Multi-frame ePSF build (všetky framy noci) | MEDIUM |
| TODO-PSF-PHASE2 | Moffat centroidy → ePSF build (TODO na call site) | MEDIUM |

### Nové citácie pridané dnes

| Citácia | Relevancia |
|---------|-----------|
| Vicuña et al. (2025), A&A Oct 2025 | PSF fitting radius — Fisher information |
| Libralato et al. (2016), MNRAS 456 | PSF-based K2 photometry |
| Stetson (1994), PASP 106, 250 | ALLFRAME — neighbor subtraction |

## pytest stav

| Session | Passed | Failed | Skipped |
|---------|--------|--------|---------|
| Ráno (bug hunt) | 103 | 0 | 6 |
| Po PSF zmenách | 103 | 0 | 6 |
| Po skip_processed | 103 | 0 | 6 |

---

## VYVAR SESSION SUMMARY — 28.5.2026 (časť 3 — PSF pipeline)

**Last validated draft: `draft_000359`**

## PSF pipeline v1 — implementácia (28.5.2026)

### Čo sme implementovali

| Blok | Zmena | Súbor |
|------|-------|-------|
| `fit_moffat_psf_stars` | Moffat2D+Const2D fit via LevMarLSQFitter; per-star sky border estimate + residual sky fitting; flux z analytického integrálu π·amp·γ²/(α-1); výstup: gamma, alpha, FWHM, sky, chi2, fit_ok | `psf_photometry.py` |
| `_moffat_fwhm_px` | Helper: FWHM = 2γ√(2^(1/α)-1) | `psf_photometry.py` |
| ePSF sky subtraction | Sky odčítaný pred EPSFBuilder (global median z MASTERSTAR) | `psf_photometry.py` |
| ePSF normalizácia | Native integral = 1.0 (sum/osamp²); `epsf_norm_factor` v meta | `psf_photometry.py` |
| ePSF QC | Radial profile FWHM, quadrant symmetry, NaN fraction; uložené v `epsf_qc` meta JSON | `psf_photometry.py` |
| ePSF build params | `smoothing_kernel=\"quadratic\"` pre osamp≤2; `fit_shape=2×FWHM+1`; `min_stars=30` | `psf_photometry.py` |
| IterativePSFPhotometry | `_epsf_noop_finder`; per-star fallback na PSFPhotometry | `psf_photometry.py` |
| Aperture correction | `_compute_aperture_correction` σ-clip median ratio dao/psf; chi2<5.0; min 5 ref hviezd | `psf_photometry.py`, `pipeline.py` |
| `photometry_mode` | `\"aperture\"` / `\"epsf\"` / `\"both\"` (default `\"both\"`) | `config.py`, `config.json`, `ui_photometry.py`, `app.py` |
| VY_PLTS | Zápis plate scale do MASTERSTAR.fits; safer read order (VY_PLTS > CD matrix > CDELT > SCALE≤5.0) | `psf_photometry.py`, `pipeline.py` |

### Diagnostické nálezy (draft_359, BO CVn)

| Test | Výsledok | Poznámka |
|------|----------|---------|
| ePSF build | 330 hviezd, asymmetry=0.014 ✅ | Dvojnásobok oproti starému buildu (148) |
| ePSF pedestal | Odstránený ✅ | Sky sub pred EPSFBuilder |
| ePSF normalizácia | native integral=1.0 ✅ | |
| Moffat fit (saturované) | fit_ok 2/10, chi2=26–186 | Saturované hviezdy — nie vhodné pre PSF fit |
| Moffat fit (nesaturované) | fit_ok 14/15, chi2 median=1.96 ✅ | Funguje správne |
| Moffat FWHM (nesaturované) | median=1.84px | Podhodnotené — asymetrický PSF profil |
| ePSF photometry chi2 | medián=72 | Stále vysoké — asymetrický PSF v BO CVn |

### Kľúčový nález — asymetrický PSF

Hviezdy v BO CVn poli majú pravostranný chvost (tracking smear alebo kóma).
Symetrický Moffat2D / ePSF to nemôže správne popísať → chi2 systematicky vysoké.
ePSF (empirický) je správna dlhodobá cesta — zachytí asymetriu automaticky.

### Vedecké citácie (PSF pipeline)

| Citácia | Použitie vo VYVAR |
|---------|------------------|
| Moffat (1969), A&A 3, 455 | `fit_moffat_psf_stars` — základ Moffat profilu |
| Anderson & King (2000), PASP 112, 1360 | `EPSFBuilder` — definícia ePSF |
| Stetson (1987), PASP 99, 191 | Hybridný PSF prístup — inšpirácia dvojkrokovej architektúry |
| Stetson (1990), PASP 102, 932 | Aperture correction metodológia |
| Trujillo et al. (2001), MNRAS 328, 977 | β≈4.765 pre atmosferický seeing |
| Anderson (2016), WFC3 ISR 2016-12 | EPSFBuilder vylepšenia |
| Bradley et al. (2024), Zenodo | photutils softvérová citácia |
| Vicuña et al. (2025), A&A (Oct 2025) | PSF fitting radius — Fisher information |

### Open TODOs — nové (28.5.2026 časť 3)

| ID | Popis | Priorita |
|----|-------|----------|
| TODO-PSF-ASYMMETRY | Riešenie asymetrického PSF profilu v BO CVn — tracking smear diagnostika | HIGH |
| TODO-PSF-SATURATION | Automaticky vylúčiť saturované hviezdy z Moffat fit (použiť `likely_saturated` flag) | HIGH |
| TODO-PSF-RATIO | Validácia moffat_flux/dao_flux ratio na nesaturovaných hviezdach (merge fix) | MEDIUM |
| TODO-PSF-FWHM | Moffat FWHM 1.84px vs VY_FWHM 3.52px — prečo? Tracking smear analýza | MEDIUM |
| TODO-PSF-PHASE2 | Fáza 2: Moffat centroidy → ePSF build (presnejší ePSF z Moffat pozícií) | MEDIUM |
| TODO-PSF-PHASE3 | Fáza 3: Hybrid model Moffat + ePSF residual (DAOPHOT filozofia) | LOW |
| TODO-MASTERSTAR-EPSF-ALL | Prebudovať ePSF pre drafty 311, 321, 358 (stará plate_scale 9.55) | MEDIUM |

# VYVAR SESSION SUMMARY — 28.5.2026

**Last validated draft: `draft_000359`** — bug fix session + ePSF rebuild s novými parametrami

## Dokončené dnes (28.5.2026)

### Bug Hunt — systematic static review (37 findings, 27 fixes, 11 safe-pattern comments)

| Batch | Bugy | Typ |
|-------|------|-----|
| HIGH | BUG-004, 014, 015, 016 | Crash / user-facing |
| MEDIUM critical | BUG-001, 002, 017, 019, 024 | Data correctness + WCS |
| MEDIUM perf | BUG-033, 034, 035, 036 | I/O + memory |
| LOW | BUG-003, 005–012, 020–021, 023, 025–026 | Guards + config drift |
| Docs | BUG-013, 018, 022, 027–032, 037–038 | Safe-pattern comments |

#### HIGH fixes
| ID | Súbor | Fix |
|----|-------|-----|
| BUG-004 | `ui_variability.py:932` | `setdefault("tess_results", {})` — KeyError v crossmatch dialógu |
| BUG-014 | `photometry_core.py:3646` | `np.percentile` na prázdnych finite dátach — field map PNG |
| BUG-015 | `photometry_core.py:3587` | `np.percentile` na prázdnom cutout — target PNG |
| BUG-016 | `photometry_core.py:3753` | Rovnaký vzor — `save_target_field_map_png` |

#### MEDIUM critical fixes
| ID | Súbor | Fix |
|----|-------|-----|
| BUG-001 | `photometry_core.py:1825` | `wcs.has_celestial` guard pred `all_world2pix` |
| BUG-002 | `photometry_core.py:9054` | WCS celestial guard — CATALOG_ONLY forced aperture |
| BUG-024 | `photometry_core.py:9085` | `VY_PSOLV == 1` guard — combined s BUG-002 |
| BUG-017 | `variability_detector.py:679` | `np.where(den > 0, ...)` — clip_ratio division by zero |
| BUG-019 | `ui_variability.py:1890` | `sub.empty` check pred `iloc[0]` — VT export |

#### MEDIUM perf fixes
| ID | Súbor | Fix |
|----|-------|-----|
| BUG-033 | `photometry_core.py:7686` | Sidecar CSV dict cache v `stress_test_relative_rms_from_sidecars` |
| BUG-034 | `ui_finalization.py:332` | `usecols=["catalog_id","aperture_mag"]` — 13–20× menej RAM |
| BUG-035 | `ui_aperture_photometry.py:1635` | LC preload cap 200 + `usecols` + `ttl=3600` |
| BUG-036 | `photometry_report.py:1582` | `usecols=_AIRMASS_COLS` — bez `nrows` capu |

#### LOW fixes
| ID | Súbor | Fix |
|----|-------|-----|
| BUG-003 | `photometry_core.py:2062` | PyTICS weight sum guard `not isfinite(s) or s <= 0` |
| BUG-005 | `photometry_core.py:5199` | `is_file` guard pred `active_targets_csv` / `comparison_stars_csv` |
| BUG-006 | `ui_aperture_photometry.py:58` | `is_file` guard v `_cached_read_csv` |
| BUG-007 | `photometry_core.py:1457` | `is_file` guard pred per-frame proc CSV read |
| BUG-008 | `vyvar_platesolver.py:1546,2481` | `app_config=None` + `_cfg_ps` — 6 `AppConfig()` nahradených |
| BUG-009 | `vyvar_blind_solver.py:63` | `app_config=None` + `_cfg` v `find_blind_hint` |
| BUG-010 | `ui_aperture_photometry.py:1416` | `cfg` z render scope namiesto `AppConfig()` |
| BUG-011 | `photometry_core.py:4588,10412` | Hoisted `_cfg_summary`, `_cfg_base` |
| BUG-012 | `pipeline.py:9381` | `_cfg_for_workers` hoisted pred loop |
| BUG-020 | `vyvar_blind_solver.py:170` | Empty/missing x,y guard → `return None` |
| BUG-021 | `ui_variability.py:2034` | Explicit `df.empty` check + warning pred `iloc[0]` |
| BUG-023 | `photometry_core.py:1954` | Komentár — rolling median guard overený ako safe |
| BUG-025 | `variability_detector.py:478` | `abs(mu) > 1e-3` floor pre RMS% |
| BUG-026 | `photometry_core.py:2517` | `log_event` keď `ensemble_normalize` nemá comp hviezdy |

---

## ePSF — kompletná implementácia a robustnosť (28.5.2026, časť 2)

### Čo sme dnes urobili (ePSF)

| Blok | ID | Zmena | Súbor |
|------|----|-------|-------|
| A1 | smoothing_kernel | `"quadratic"` pre oversampling≤2, `"quartic"` pre ≥3; uložené v meta | `psf_photometry.py` |
| A2 | IterativePSFPhotometry | Nahradili `PSFPhotometry`; `_epsf_noop_finder` pre photutils 2.3.0; per-star fallback | `psf_photometry.py` |
| A3 | fit_shape fix | `2×FWHM+1` namiesto `cutout_size-4`; pre FWHM=3.5px → `(9,9)` | `psf_photometry.py` |
| A4 | ePSF QC | Radial profile FWHM, quadrant symmetry, NaN fraction; uložené v `epsf_qc` v meta JSON | `psf_photometry.py` |
| A5 | min_stars | Default 30 (bol 15), config-driven `epsf_min_stars` | `psf_photometry.py`, `config.py`, `config.json` |
| B1 | photometry_mode | `"aperture"` / `"epsf"` / `"both"` (default `"both"`) | `config.py`, `config.json` |
| B2 | Pipeline routing | `_photometry_mode_run_flags()` helper; guards na 6 miestach | `pipeline.py` |
| B3 | UI prepínač | Radio button v `ui_photometry.py`; caption v `app.py` | `ui_photometry.py`, `app.py` |
| C1 | Aperture correction | `_compute_aperture_correction()` — σ-clip median ratio `dao_flux/psf_flux`; chi²<5.0; min 5 ref hviezd | `psf_photometry.py`, `pipeline.py` |
| C2 | spatial_order doc | Komentár k 0/1/2 order; TODO-MULTISET pre per-set config | `config.py` |
| plate_scale | VY_PLTS fix | Safer read order (VY_PLTS > CD matrix > CDELT > SECPIX > SCALE≤5.0); `VY_PLTS` zápis pri MASTERSTAR build | `psf_photometry.py`, `pipeline.py` |

### ePSF rebuild — draft_359 výsledky

| Metrika | Starý build (27.5) | Nový build (28.5) |
|---------|-------------------|------------------|
| `n_stars_used` | 148 | **330** |
| `plate_scale_arcsec_px` | 9.55 (WCS bug) | **1.3** (VY_PLTS) |
| `smoothing_kernel` | chýbalo | **quadratic** |
| `fit_shape` | chýbalo | **[9, 9]** |
| `epsf_qc` | chýbalo | **prítomné** |
| `epsf_fwhm_native_px` | 17.0 (zlá metrika) | **2.236** |
| `epsf_vs_input_fwhm_ratio` | 4.832 | **0.636** ✅ |
| `epsf_asymmetry` | 0.041 | **0.014** ✅ |

### Vedecký kontext

- **Anderson & King (2000), PASP 112, 1360** — základ ePSF algoritmu
- **Anderson (2016), WFC3 ISR 2016-12** — vylepšenia EPSFBuilder
- `IterativePSFPhotometry` — subtrakcia susedných hviezd pred refittingom; kritické pre crowded fields a variabilné hviezdy s blízkymi susedmi
- Aperture correction štandardný krok DAOPHOT pipeline — teraz implementovaný

### Isolation filter (overené 28.5)

- `_isolation_radius_px = 3 × fwhm_px` (čisto pixely)
- Porovnanie susedov: **uhlová** vzdialenosť RA/Dec → `dists_px = dists_deg × 3600 / plate_scale_arcsec_px`
- `plate_scale` mení konverziu arcsec→px, nie samotný izolačný polomer

---

# VYVAR SESSION SUMMARY — 27.5.2026

**Last validated draft: `draft_000359`**

| Metrika | Hodnota |
|---------|---------|
| Light curves | **196** |
| Frames | **139** |
| LC RMS median | **0.0857 mag** |
| BO CVn comps | **4** (RMS 0.007–0.013) |
| masterstar_matched | **186** |
| catalog_only | **10** |
| AAVSO export | **184 súborov** |
| VAR.ASTRO export | **11 súborov** (eclipsing only) |
| Night run elapsed | **~3437 s** (~57 min) |

## Commits dnes (27.5.2026)

| Hash | Popis |
|------|-------|
| `45f7e9f` | fix: field_density uses Gaia-matched star count instead of raw VY_NDAO |
| `bf1adee` | fix: skip catalog_only targets in Phase 2A photometry loop |
| `f80a4e1` | fix: WCS rescale now handles PC+CDELT matrix format (Fix A) |
| `60bbc63` | fix: pixel-distance fallback when WCS plate scale deviates >20% |
| `4e74851` | fix: recompute masterstars ra_deg/dec_deg after WCS rescale (Fix C) |
| `5642da5` | fix: excluded comp stars show w(rel)=0.000 in PDF/UI export |
| `b2ed7f8` | remove: L.A.Cosmic (AstroScrappy) and Background Flattening |
| `139b313` | refactor(ui): move RUN VYVAR next to Scan Source, remove Pre-processing section |
| `4470bc6` | fix: restore _db_for_calibration_tasks removed in L.A.Cosmic cleanup |
| `dc9edca` | perf: ePSF model built only from masterstars + comp stars |
| `fe6bb6e` | fix: aavso/varastro export no longer produces empty directories |
| `84c7876` | fix: include noisy targets in UI LC selector default filter |
| `2a46f79` | fix: PDF report cleanup — catalog_only, field map, Method B text |
| `81c41a4` | feat: per-comp exclusion/suspect reasons persisted and displayed |

## Kľúčové opravy dnes

### WCS cascade bug (root cause BO CVn 0 comps)
- PC-matrix WCS plate scale 9.79″/px namiesto 1.30″/px
- Comp selection aj select_active_targets používali zlé RA/Dec
- Fix: pixel-distance fallback keď WCS scale deviates >20%
- Fix B aplikovaný na oba miesta (Phase 0 + Phase 1)

### Pipeline vylepšenia
- field_density z Gaia-matched count (nie raw VY_NDAO)
- catalog_only targety preskočené v Phase 2A
- ePSF len z masterstars + comp hviezd (nie všetky hviezdy)
- AAVSO/VAR.ASTRO export opravený (prázdne adresáre)

### UI/UX
- RUN VYVAR presunutý vedľa Scan Source
- Pre-processing + MAKE MASTERSTAR sekcia odstránená
- L.A.Cosmic + Background Flattening odstránené
- BO CVn viditeľná v UI (noisy do default filtra)

### PDF/Report kvalita
- excluded comp hviezdy: odstránené z tabuľky
- suspect comp hviezdy: dôvod zobrazený v status stĺpci
- catalog_only targety: žiadne LC stránky v PDF
- Field mapa: len masterstar-matched targety
- Method B: vysvetlenie pridané do PDF

---

## Open TODOs (backlog)

_Merged from repo (27.5–28.5.2026) and local archive (21.5.2026). CLOSED items marked ✅._

### Active (repo — 27.5.2026)

| ID | Popis | Priorita |
|----|-------|----------|
| TODO-MULTISET | Per-telescope-set config architektúra | HIGH |
| TODO-GS9 | Ground LC periodická analýza — LS + BLS + folded LC v PDF | HIGH |
| TODO-GS8 | Multi-Night Global Matching (Phase 3) | HIGH |
| TODO-LC-TREND | PARTIAL — common-mode detrend; re-validácia na moonless night | MEDIUM |
| TODO-FORCED-COMP | forced_aperture catalog_only bez Phase 1 tier selection | MEDIUM |
| TODO-GS10 | AAVSO Direct API Upload | MEDIUM |
| TODO-LIB | Cython .pyd kompilace | MEDIUM |
| TODO-GS7 | Paper draft (PASP/AN) | FUTURE |
| TODO-CACHE-CENTRAL | Centralizovať `csv_cache` na Phase 1 entry (BUG-037) | LOW |
| TODO-EPSF-VALIDATE | Full night run `photometry_mode="both"` na draft_359; porovnať `psf_flux` vs aperture na comp hviezdach; overiť aperture correction faktor | HIGH |
| TODO-MASTERSTAR-EPSF-ALL | Prebudovať ePSF pre drafty 311, 321, 358 (stará plate_scale 9.55) | MEDIUM |
| TODO-EPSF-SPATIAL | `spatial_order=1` pre Newton/Noctutec set — čaká na TODO-MULTISET | LOW |

### Active (local — 21.5.2026)

| ID | Popis | Priorita |
|----|-------|----------|
| TODO-LC-QUALITY | LC classification filter: saturated, noisy2/3, `no_data`; `lc_quality_flag`; LC count semantics | MEDIUM |
| TODO-LC-TREND | Differential extinction + ALG audit (rastúce/klesajúce LC u niektorých ROT) | MEDIUM |
| TODO-GEO | Observer geographic position audit (BJD, airmass, HJD) | MEDIUM |
| TODO-DEV-PROCESS | `scripts/regression_test.py` + smoke checklist pred commitom | HIGH |
| TODO-MASTERSTAR-QA validation | Overiť cyan FORCED_APERTURE overlay v UI na draft_342 | LOW |
| TODO-13 validation | Gaia→DAO completeness v QA stále ~3.5% — overiť po Phase A+B na novom exporte | MEDIUM |

### Background gradient & flux combination (Han & Brandt 2023 TGLC brainstorm)

Brainstorm source: Han & Brandt 2023 (TESS-Gaia Light Curve, AJ 165:71) independently
confirms VYVAR's own finding — aperture wins in most conditions; PSF wins faint/sparse +
deblends near variables; a weighted aperture+PSF combo is best.

- **[TODO-SKY-PLANE]** (priority LOW-MED): replace the constant/median local sky in the
  aperture annulus with a fitted **TILTED PLANE** (2D linear gradient) per star. Removes
  **additive** sky gradients tilting across a star's region (moon, light-dome, wide
  field). Benefit concentrated on **wide rigs** (OAT, planned 8" f/2.72) under gradient
  sky; small on narrow fields and largely cancels in differential photometry for
  smooth gradients. Does **NOT** fix multiplicative flat-field residual. Bounded change,
  fits the per-star paradigm (photutils plane-fit / LocalBackground). **TESTED —
  NEGATIVE** (drafts 361/362, OAT wide, 9.77"/px; standalone read-only): frame-scale
  sky gradients are **real** (~8–12% of sky, steepest on twilight/first frames), **but**
  the decision metric — plane-vs-median sky difference — is dominated by **residual**
  (median-clipping asymmetry + annulus contamination/scatter), **not** the linear tilt.
  The **linear** component (what a plane would fix on a symmetric annulus) is ~0.3–0.6%
  of the plane-median split and ~0.5% of the photometric error (only 3.7–5.4% of stars
  exceed 0.5× phot err on the linear-only part) — negligible. As predicted, a linear
  gradient cancels in the symmetric annulus median (the ring averages it out). No
  field-position correlation (0.07/0.12). Differential target–comp separations are
  ~2.5° (less gradient cancellation differentially), but since the linear part cancels
  in the annulus anyway, this does not rescue SKY-PLANE. **VERDICT: skip** for standard
  symmetric-annulus photometry; revisit only for asymmetric annuli (neighbor masking),
  curvature, or extreme gradients. **Minor future hint** (not actionable now): the
  per-star annulus sky **scatter** from median-clipping/contamination is larger than
  the gradient effect — if sky-noise reduction is ever pursued, a more robust annulus
  estimator (two-sided sigma-clip vs the current upper-only 2σ) would help more than
  any gradient model.

- **[TODO-WEIGHTED-LC]** (priority LOW-MED): produce a **WEIGHTED aperture+PSF light curve**
  (linear combination), per Han & Brandt 2023 (~0.4×PSF + 0.6×aperture, field/
  crowding-dependent; ~10–20% precision gain reported). Continuous-weight alternative
  to the hard adaptive selector (`psf_adaptive`). Needs both aperture+PSF fluxes per
  star; optimal weight is field-dependent. Small build, potential near-term precision
  win on faint/crowded. **TESTED — NEGATIVE** (drafts 361–364, standalone ceiling,
  relaxed PSF quality = upper bound): weighted LC w×PSF + (1−w)×aperture vs
  aperture-alone. The combination helps **only** where aperture and PSF have **comparable
  RMS** (error-decorrelation gain); nothing when one channel dominates.
  361 (9.77"/px): w_opt 0.2, ~2.9% | 362 (9.77"/px): w_opt 0.3, ~2.7% |
  363 (0.65"/px): w_opt 0, 0% | 364 (0.39"/px): w_opt 0, 0%.
  Gain tracks PSF-vs-aperture comparability: undersampled wide rigs (361/362) have large
  arcsec-apertures that are sky-noise-limited, so PSF nearly matches aperture → ~3%
  ceiling; well-sampled rigs (363/364) have PSF 37–68% worse → w→0, zero gain. The ~3%
  ceiling is undersampled-only and shrinks under the strict production quality gate →
  **below the cost/complexity threshold. VERDICT: not worth implementing; deprioritized.**
  **Future hint** (not actionable now): on 362 the G14–15 bin had PSF-alone beat
  aperture by 28% → possible PSF niche on **faint, sky-noise-limited undersampled
  wide-field** data (relevant to OAT / planned 8" f/2.72). Revisit only if wide-field
  faint precision becomes a priority.

- **[TODO-SCENE-FORWARD-MODEL]** (priority HIGH-effort / conditional): TGLC-style **scene
  forward model** for crowded-field decontamination. Fix neighbor positions+fluxes from
  the (deep) Gaia cone, fit a **LOCAL ePSF + a background model** (flat + 2D linear
  gradient) **SIMULTANEOUSLY** as a linear least-squares problem over a region, subtract
  modeled neighbors, photometer on the decontaminated residual. Principled crowded-
  field path; natural home for faint/crowded science (Brno / globular clusters).
  Major architectural addition; the linear trick relies on **FIXED Gaia priors** → fits
  a "fix neighbors, float target" decontamination mode, not standard differential
  photometry. TESS-specific strap/CCD-artifact terms **NOT** needed for ground-based
  (ground background = additive sky/light-pollution/moon + multiplicative flat
  residual). Justified only if faint/crowded becomes a priority.

**CLOSURE — Background gradient & flux combination brainstorm: CLOSED.** Two of three
TODOs tested **NEGATIVE** (TODO-WEIGHTED-LC, TODO-SKY-PLANE). Only
TODO-SCENE-FORWARD-MODEL remains **OPEN** — explicitly **CONDITIONAL** on
crowded-faint science (Brno / globular clusters) and **LOW priority** (further lowered
by the grouper-negative result and the modest realized PSF benefit). No further
background work planned unless that science becomes a priority.

### Reference — CLOSED / completed (from local archive)

| ID | Status | Notes |
|----|--------|-------|
| TODO-GS6b | ✅ CLOSED | AAVSO Extended Format validator (`scripts/validate_aavso_export.py`) — 20.5.2026 |
| CQ-1, CQ-2, CQ-4 | ✅ CLOSED | `run_phase2a`, `render_live_view`, `solve_wcs_with_local_gaia` splits — 20.5.2026 |
| PERF-9 | ✅ CLOSED | Vectorized haversine VSX match — 20.5.2026 |
| TODO-23, TODO-25, TODO-16, TODO-17 | ✅ CLOSED | Adaptive match radius, Gaia completeness UI, crossmatch coords — 20.5.2026 |
| TODO-ALG-2, TODO-ALG-3, TODO-ALG-4, TODO-ALG-5 | ✅ CLOSED | Savitzky-Golay, temporal binning, Democratic Detrender, PyTICS — 20.5.2026 |
| TODO-44, TODO-8 | ✅ CLOSED | Role-aware aperture; ePSF infrastructure — 20.5.2026 (Bootes → TODO-8-BOO) |
| PERF-1 … PERF-10 | ✅ CLOSED | Performance series — 19.5.2026 |
| CQ-3, TODO-35 | ✅ CLOSED | Comp selection split; SysRem MVP — 19.5.2026 |
| TODO-ALG-2 … TODO-ALG-5, TODO-44, TODO-8 | ✅ CLOSED | See 19.5.2026 session backlog table |

---

## Known issues / next session

_Merged from repo (27.5–28.5.2026) and local (21.5.2026)._

### 28.5.2026 (bug fix session — časť 1)

- Všetky HIGH a MEDIUM bugy z Bug Hunt reportu opravené ✅
- LOW docs (BUG-013, 018, 022, 027–032, 037, 038) — safe-pattern komentáre pridané
- Commit `ba7c142`: 27 fixes + `VYVAR_BUG_HUNT_REPORT.md`

### 28.5.2026 (ePSF robustness — časť 2)

- ePSF pipeline kompletný (A1–A5, B1–B3, C1–C2, VY_PLTS) — pozri tabuľku vyššie
- **draft_359** ePSF prebudovaný s `VY_PLTS=1.3`, `n_stars_used=330`, QC ratio **0.636**
- **draft_359 ePSF ratio=0.636** — fyzikálne správne (ePSF užší než DAO FWHM); validovať na fotometrii (**TODO-EPSF-VALIDATE**)
- **Staré ePSF modely** (drafty 311, 321, 358) — postavené so SCALE=9.55 → **TODO-MASTERSTAR-EPSF-ALL**
- **`photometry_mode="epsf"` only** — `mag_calib` routing z PSF flux čaká na validáciu; zatiaľ používať **`"both"`**
- Odporúčané: `pytest tests/` po commite ePSF zmien

### Repo (27.5.2026)

- BO CVn má 4 comp hviezdy (tier 1, dense override max_comp_rms=0.08)
  → riešiť pri TODO-MULTISET (per-set config)
- WCS pixel-distance fallback aktívny pre toto pole (9.79″/px WCS)
  → správne správanie, Fix B robustný
- draft_359 je aktuálny validovaný run
- 103 testov passed, 6 skipped

## Test suite

| Dátum | Výsledok |
|-------|----------|
| 27.5.2026 | pytest tests/ → **103 passed**, **6 skipped**, **0 failed** |
| 28.5.2026 | Odporúčané po commite (bug fixes + ePSF) — overiť 103 passed, 0 failed |

## Known sets

| ID | Telescope | Camera | Scale | Site |
|----|-----------|--------|-------|------|
| 1 | 200mm Carl-Zeiss | QHY294MM | ~1.3″/px | Jirny |
| 2 | 300/1200 Newton | C3-26000 | TBD | Dáblic obs. |
| 3 | Noctutec 206/560 f/2.72 | C3-26000 | TBD | TBD |

### Local (21.5.2026)

- **Staré drafty (≤341):** `FORCED_APERTURE` + `dao_flux` / `photometry_ok` platí až po re-exporte s `3d2363c` (SNR per-star aperture loop).
- **LC trends:** niektoré ROT hviezdy rastúce/klesajúce — čaká na **TODO-LC-TREND**.
- **MASTERSTAR QA:** Gaia→DAO Completeness **~3.5%** na dashboarde — stále nízke; čaká na **TODO-13** validáciu po novom DAO pass 2 + forced rows.
- **2 LC bez súboru:** draft_342 má 134 summary riadkov, 132 `lightcurve_*.csv` — overiť 2 targets bez `lc_csv`.

---

---

# VYVAR SESSION SUMMARY — 21.5.2026

**Last validated draft: `draft_000342`** (`simulate_night_run`, commit `3d2363c`)

| Metrika | draft_342 |
|---------|-----------|
| Light curves | **134** (summary); **132** `lightcurve_*.csv` on disk |
| HIP 67011 `lc_rms` | **0.012** mag (139 frames) |
| `DET_*` names v proc CSV | **0** |
| `FORCED_APERTURE` + `dao_flux` | **100%** measured (per-star aperture fix) |
| Targeted `psf_flux` non-null / frame | ~**183** (variables + top 40 comps) |
| Night run čas | **4345 s** (~72 min) |

## Dokončené dnes (21.5.2026)

| # | Zmena | Súbor(y) / commit |
|---|-------|-------------------|
| TODO-13 Phase A | Pass 2 (catalog-seeded DAO) v MASTERSTAR fast path — `_dao_targeted_pass2_unmatched_gaia` po pass 1 | `pipeline.py` |
| TODO-13 Phase B | Forced aperture rows + catalog-only proc CSV (`GAIA_MATCHED` + `FORCED_APERTURE` only); `_proc_catalog_keep_matched_rows_only` | `pipeline.py` |
| Comp selection | PSF χ² filter disabled pre DAO-era proc CSV (`max_psf_chi2=inf`) | `photometry_core.py`, `comp_pool_rms.py` |
| ISOLATED_BIN | `rms_map.pop(cid)` pre rms &lt; 1e-4 (žiadny floor — ultra-stable comps vylúčené) | `comp_pool_rms.py` |
| TODO-MASTERSTAR-QA | Layer overlay z proc CSV MASTERSTAR frame (green=GAIA_MATCHED, cyan=FORCED_APERTURE) | `ui_masterstar_qa.py`, `masterstar_qa_plot.py` |
| TODO-EPSF-TARGETED | ePSF len variables + top 40 comps (~183/frame vs 3500+) | `pipeline.py`; fix `read_vyvar_csv` import (`0b01955`) |
| DET_* names | Matched DAO → `name=catalog_id` po sky match | `pipeline.py` (`14cf0f9`) |
| photutils 2.3.0 | Per-star `CircularAperture` loop; `FORCED_APERTURE` → `photometry_ok=True` po meraní | `photometry_core.py` (`3d2363c`) |
| `ps` UnboundLocalError | `export_per_frame_catalogs`: `ps` pred `_ap_st` | `pipeline.py` (`0b01955`) |

---

# VYVAR SESSION SUMMARY — 20.5.2026

**17 tasks completed, 0 regressions, E2E validated on draft_321**

| Task | Detail |
|------|--------|
| TODO-GS6b | AAVSO Extended Format validator (`scripts/validate_aavso_export.py`) |
| MTYPE fix | `export_reports.py`: MTYPE=STD → MTYPE=DIFF |
| CQ-1 | `run_phase2a` split: 1515 → 188 lines + 3 helpers |
| CQ-2 | `render_live_view` split: 1395 → 731 lines + 3 helpers |
| CQ-4 | `solve_wcs_with_local_gaia` split: 1843 → 1224 lines + 3 helpers |
| PERF-9 full | Vectorized haversine VSX match in `photometry_core.py` |
| TODO-23 | Adaptive match radius (plate_scale × 3); universal telescope support |
| TODO-25 | UI reads `gaia_dao_completeness_pct` from `pipeline_meta.json`; pipeline writes it |
| TODO-16 | UI crossmatch uses `active_targets.csv` coords (WCS-verified) |
| TODO-17 | `crossmatch_auto_done` gated — only set True when candidates have catalog data |
| TODO-ALG-4 | Democratic Detrender (arXiv:2411.09753v2) — 3 models, `delta_mag_democratic` + `err_inflation` columns |
| TODO-ALG-3 | Temporal Binning adaptive window (MNRAS 2023) — `temporal_bin_comp_lc()` before stability/PyTICS |
| TODO-ALG-5 | PyTICS iterative comp weights (RASTI 2026) — `pytics_iterative_weights()` after stability check |
| TODO-ALG-2 | Savitzky-Golay detrending (default disabled) — `savgol_detrend_lc()` after airmass |
| TODO-ALG-1 | BLS ground LC — parkovaný, väčší scope (TESS-only L-S/BLS today) |
| TODO-44 | ✅ CLOSED — Role-aware aperture (`aperture_variable_factor` / `aperture_comp_factor`); verified draft_321 |
| TODO-8 | ✅ CLOSED — ePSF infrastructure: `build_epsf_model` + per-frame PSF export + Phase 2A flux selector + standalone 🔬 ePSF tab + `run_epsf` job + ⚡ RUN ePSF button; default `psf_photometry_enabled=false`; Bootes validation → **TODO-8-BOO** |
| TODO-8 ePSF dashboard | `load_epsf_metrics_for_draft` + UI overlay (aperture vs PSF) + PDF PSF section; `catalog_id` str/normalize fix; `drafts_before_session` path fix |
| TODO-8 ePSF RUN button | `run_epsf` job + ⚡ RUN ePSF Photometry button v ePSF tab |
| Reporting A | PDF Methods dynamic ALG citations; VAR.ASTRO `# ALG:` lines; UI Settings ALG toggles + TODO-44 sliders |
| FutureWarning | `variability_detector.py` pandas downcasting — zaznamenané ako tech debt |

### Backlog updates
Mark as CLOSED: TODO-GS6b, CQ-1, CQ-2, CQ-4, PERF-9, TODO-23, TODO-25, TODO-16, TODO-17, **TODO-ALG-2, TODO-ALG-3, TODO-ALG-4, TODO-ALG-5, TODO-44, TODO-8**

**TODO-8 note:** Infrastructure complete — `build_epsf_model` + per-frame PSF export + Phase 2A flux selector + standalone ePSF tab + RUN ePSF button. Bootes validation pending (**TODO-8-BOO**).

### Bug — Calibration Library re-register (20.5.2026, koniec dňa)

- `generate_master_dark_from_source_dir()` prepíše existujúci DB záznam (ID_EQUIPMENTS, ID_TELESCOPE) keď súbor s rovnakým názvom už existuje
- Správanie: mal by vytvoriť NOVÝ záznam pre iný set, nie UPDATE existujúci
- Dopad: Dark_60s bol preregistrovaný z QHY294MM+Carl-Zeiss na C3-26000+DDT
- Fix: INSERT nový záznam ak sa líši ID_EQUIPMENTS alebo ID_TELESCOPE (`register_calibration_library_entry` scope guard + scoped filename `_eq{N}_tel{M}`; `importer.py`, `database.py`)
- Priorita: HIGH (blokuje NGC 5466 / Bootes test)

### Brand / paper title — VYVAR (20.5.2026)

**Názov VYVAR je finálny — nemeň (skvelý brand).** V anglickom článku ho stačí „astronomicky“ obhájiť; oficiálny working title:

**VYVAR: A High-Automation Variable Star Photometry Pipeline…**

Do úvodnej sekcie článku (Introduction) alebo pod čiaru — krátka poznámka pre recenzentov:

> The name "VYVAR" is derived from the Slovak words for "Variable" (premenná) and "Archive/Reduction", while colloquially playing on the culinary term "broth/soup" – metaphorically boiling down raw FITS images into a rich, concentrated astrophysical summary report.

Recenzenti takéto jazykové hry zvyčajne vítajú; ukazuje, že softvér má „dušu“ a nepísal ho korporátny robot.

### Open / parked (po ALG series)
| ID | Popis | Priorita |
|----|-------|---------|
| TODO-8-BOO | Bootes globular cluster validation: test ePSF vs aperture on dense field (~2h data) | MEDIUM |
| TODO-LIB | Compile VYVAR modules to `.pyd` (Cython) — hide source, enable C translation | MEDIUM |
| TODO-ALG-1 | BLS period search on ground LC (`delta_mag` + BJD) — parkovaný, väčší scope | LOW |

### Strategic — Gold Standard 2.0

| ID | Popis | Priorita | Úsilie |
|----|-------|---------|--------|
| TODO-GS8 | **Multi-Night Global Matching** — Phase 3 modul: globálny crossmatch comp hviezd cez viacero nocí, medzinocný ZP solver, zlúčenie do jednej LC bez skokov. Závislosť: TODO-GS6 (AAVSO validácia na viacerých nociach). | HIGH | 2–4 dni |
| TODO-GS9 | **Ground LC periodická analýza** — Lomb-Scargle + BLS periodogramy priamo na Phase 2A LC CSV (astropy.timeseries); automatický fázový diagram (folded LC) pre variability kandidátov; výstup v PDF reporte. Súvisí s TODO-ALG-1 (parkovaný). | HIGH | 1–2 dni |
| TODO-GS10 | **AAVSO Direct API Upload** — tlačidlo "Submit to AAVSO" v UI; `AAVSO_USERNAME` / `AAVSO_PASSWORD` v config; odoslanie cez AAVSO WebObs API po validácii (GS6b). Závislosť: TODO-GS6. | MEDIUM | 1 deň |
| TODO-GS11 | **Flux Dilution Factor** — pixel-level blend correction: pre každý target vypočítať D = Flux_target / (Flux_target + ΣFlux_neighbors) z Gaia pozícií + magnitúd v aperture rádiuse; opraviť nameranú amplitúdu; pridať `dilution_factor` stĺpec do LC CSV a PDF. Súvisí s contamination_map (už existuje). | MEDIUM | 1–2 dni |

---

---

# VYVAR SESSION SUMMARY — 19.5.2026

**28 tasks completed, 5 bug fixes, E2E validated on draft_321**

Total night run time improved: **~1383 s → ~1047 s** (draft_321 measured; cieľová trieda ~1124 s, **~−19%** oproti pred-optimalizácii)

| Metrika | Pred | Po | Δ |
|---------|------|-----|---|
| Celkový night run (`simulate_night_run`) | ~1383 s | **1046.5 s** (draft_321) | **−336 s (~−24%)** |
| Photometry step | ~657 s (draft_320) | **524.9 s** | pod cieľom 580 s |
| Phase 1 comp selection | ~530 s (draft_320) | **368.8 s** | −161 s |
| SysRem (82 LC × 139 frames) | — | **5.7% RMS improvement** | BO CVn field |

## Dokončené dnes (19.5.2026)

### Performance (PERF-1 … PERF-10) — všetko ✅
| TODO | Čo |
|------|-----|
| PERF-1 | `csv_cache` v suspected-variables; hit/miss log |
| PERF-2 | MASTERSTAR `ms_data`/`ms_header` cache (Phase 2A + PNG/edge) |
| PERF-3 | Comp Gaia batch prefetch pred Phase 1 |
| PERF-4 | `comp_pool_rms.py` vectorized flux + groupby |
| PERF-4B | Hybrid `_accumulate_per_frame_comp_metrics` (iterrows N&lt;50, groupby N≥50); catalog_only skip Phase 1 |
| PERF-5 | `ProcFrameStore` — jeden disk read na snímok |
| PERF-6 | UI variability + `load_field_flux_matrix` cez ProcFrameStore |
| PERF-7 | UI LC index: cached summary + overlay CSV |
| PERF-8 | Shared flux matrix Phase 2A (~384 hviezd × 139 snímok) |
| PERF-9 | ✅ Vectorized haversine VSX match in `photometry_core.py` (`select_active_targets`) |
| PERF-10 | DAO QC v calibration pass (`dao_qc_in_calibrate`) |

### Code quality + algorithms
| # | Čo | Výsledok |
|---|-----|----------|
| CQ-3 | `select_comparison_stars_per_target` split | 1893 → ~200 riadkov orchestrátor + `comp_selection_per_target.py` (12 helpers) |
| TODO-35 | SysRem MVP (Tamuz et al. 2005) | `delta_mag_sysrem` stĺpec; 3 iter; 5.7% RMS improvement |
| TODO-ALG-3/4/5 | Backlog algoritmov | Pridané do STATE (temporal binning, Democratic Detrender, PyTICS) |
| TODO-GS6b | ✅ AAVSO Extended Format validator (`scripts/validate_aavso_export.py`) |

### Headless pipeline + E2E
| # | Čo | Výsledok |
|---|-----|----------|
| — | `night_run.py` | Headless runner (základ pre TODO-11) |
| — | `simulate_night_run.py` | CLI E2E (`D:\BO_CVn`, eq=1, tel=1) |
| — | **draft_321** | exit 0, **84 LC**, **139 frames**, photometry **524.9 s** |
| — | **draft_342** | exit 0, **134 LC**, **139 frames**, night run **4345 s**, HIP 67011 `lc_rms=0.012` |
| — | BO CVn / FW CVn `lc_rms` | 0.1515 / 0.0153 (within 0.001 mag of reference) |
| — | PDF report | **98 strán**, všetkých **84** hviezd |

### Bug fixes (19.5.2026)
| Bug | Fix |
|-----|-----|
| ProcFrameStore subscriptable | `.get()` + `__getitem__` safety net |
| `WindowsPath` / FIELD JUMP DB | `VyvarDatabase(dbp2)` v `detect_field_jumps` |
| PDF `_norm_cid` | Module-level + staticmethod na `_PhotometryReportBuilder` |
| PDF `_var_results` | `self._var_results` v `__init__` |
| PDF `TITLE_H` | Opravené v report builderi |
| `aperture_px` key | Konzistentný kľúč v summary/LC |
| `contamination_map` init | `contamination_map = {}` v comp selection (draft_320 crash) |

### Otvorený backlog (po 19.5.2026; CQ-1/2/4, PERF-9, GS6b, TODO-16/17/23/25 → CLOSED 20.5.2026)
| ID | Popis |
|----|-------|
| TODO-GS6 | AAVSO submission |
| TODO-GS7 | Paper draft |
| TODO-LIB | Compile VYVAR modules to `.pyd` (Cython) — hide source |
| TODO-ALG-1 | BLS ground LC — parkovaný (väčší scope) |
| TODO-ALG-2 | ✅ CLOSED — Savitzky-Golay detrend (`savgol_detrend_enabled=false` default) |
| TODO-ALG-3 | ✅ CLOSED — Temporal binning + adaptive window cap |
| TODO-ALG-4 | ✅ CLOSED — Democratic Detrender (`delta_mag_democratic`, `err_inflation`) |
| TODO-ALG-5 | ✅ CLOSED — PyTICS iterative comp weights |
| TODO-44 | ✅ CLOSED — Role-aware aperture (`aperture_variable_factor` / `aperture_comp_factor`) |
| TODO-8 | ✅ CLOSED — ePSF tab + RUN ePSF + `run_epsf` job; Bootes validation → **TODO-8-BOO** |
| TODO-8-BOO | Bootes globular cluster validation: ePSF vs aperture on dense field (~2h data) | MEDIUM |
| TODO-11 | Auto-trigger watchdog |
| TODO-LC-QUALITY | LC classification filter (saturated, noisy2/3, `no_data`) |
| TODO-LC-TREND | Differential extinction + ALG audit |
| TODO-GEO | Observer geographic position audit (BJD, airmass, HJD) |
| TODO-DEV-PROCESS | `scripts/regression_test.py` + development checklist |
| TODO-MASTERSTAR-QA validation | Cyan FORCED_APERTURE overlay v UI |
| — | `app.py` → `night_run.run_night_pipeline()` (deferred) |
| PERF-9 cKDTree | Spatial index for comp pool (optional; haversine done) |
| SysRem full-field | Všetky `proc_*.csv` hviezdy (deferred po PERF-5/6) |

---

# VYVAR SESSION SUMMARY — 18.5.2026

## Dokončené dnes (18.5.2026)

### Validácia
| # | Čo | Výsledok |
|---|----|---------|
| 1 | Gold standard roadmap + TODO-GS1–GS7 | Zdokumentované v STATE |
| 2 | TODO-45: RGB kamera (IMX533) | Zdokumentované v STATE |
| 3 | TODO-27: Re-validácia po float64 fix | CLOSED — 17 hviezd = DAO detekčná limita |
| 4 | TODO-31: Growth curve korekcia | CLOSED as NOT APPLICABLE (diferenciálna fotometria) |
| 5 | photutils LC validácia (draft_310) | 67 hviezd, mag 8–13 Δ<0.001 mag ✅ |
| 6 | PyRAF LC validácia | NOT FEASIBLE (IRAF float32 FITS bug) |
| 7 | Muniwin LC validácia | 3 hviezdy, ±5–15% RMS ✅ |
| 8 | AIJ validácia | PLANNED (manuálna, používateľ) |

### Fyzikálny audit + pipeline opravy
| # | TODO | Výsledok |
|---|----|---------|
| 9 | TODO-29: Airmass fit po outlier detect | ✅ |
| 10 | TODO-30: CT → airmass na CT-korigovanom mag | ✅ |
| 11 | TODO-GS1: Citačné komentáre (Howell, Broeg, Stetson) | ✅ |
| 12 | TODO-GS2: Unit test suite 11/11 | ✅ |

### UI + export
| # | Čo | Výsledok |
|---|----|---------|
| 13 | Kompletný preklad UI → angličtina | ~766 stringov, 0 SK/CZ zostáva ✅ |
| 14 | Vedecké citácie v AAVSO + VAR.ASTRO exportoch | ✅ |

### Code quality (6 passes)
| # | Fix | Výsledok |
|---|----|---------|
| 15 | Silent exceptions → logging | 38 blokov, 9 súborov ✅ |
| 16 | Gaia ID normalizácia | 8 duplicátov → canonical ✅ |
| 17 | Dead code označený | 5 UI + 13 legacy helpers ✅ |
| 18 | Draft path resolution | `resolve_draft_dir()` ✅ |
| 19 | CSV dtype konzistentnosť | `read_vyvar_csv()` ✅ |
| 20 | Split dlhých funkcií | `_PhotometryReportBuilder` (3384→63 lines) ✅ |

### CSV schema cleanup
| # | Čo | Výsledok |
|---|----|---------|
| 21 | Phase 1: 15 obsolete stĺpcov odstránených | ✅ |
| 22 | Phase 2: 11 ďalších stĺpcov odstránených | ✅ |
| 23 | Bug fix: variability_candidates.csv TESS stĺpce | ✅ |

### Dokončené dnes (18.5.2026) — doplnok

| # | Čo | Výsledok |
|---|----|---------|
| 24 | TODO-GS3: README.md | ✅ |
| 25 | TODO-GS4: CITATIONS.bib | ✅ |
| 26 | TODO-GS5: CHANGELOG.md | ✅ |
| 27 | TODO-19: Infolog ukladanie na disk | ✅ Auto-save po RUN VYVAR + MAKE MASTERSTAR + manual button |
| 28 | CQ-6: Delete legacy deprecated functions | ✅ 13 funkcií + 2 helpers = 1021 riadkov odstránených |

---

## Otvorené TODO — úplný backlog

### HIGH priority
| TODO | Popis |
|------|-------|
| TODO-GS1 | ✅ Hotovo |
| TODO-GS2 | ✅ Hotovo |
| TODO-GS3 | ✅ Hotovo |
| TODO-GS4 | ✅ Hotovo |
| TODO-GS5 | ✅ Hotovo |
| TODO-GS6 | AAVSO submission + cross-observer validácia |
| TODO-GS6b | AAVSO Extended Format validation (blocker for TODO-GS6) |
| TODO-GS7 | Paper draft (PASP / AN) |
| TODO-35 | ✅ COMPLETED — SysRem MVP (exported LC targets) |

**TODO-GS6b: AAVSO Extended Format validation**
- Before first submission to AAVSO (TODO-GS6), add automated validator that checks VYVAR AAVSO export against official spec: https://www.aavso.org/aavso-extended-file-format
- Required fields: `#TYPE`, `#OBSCODE`, `JD`, `MAGNITUDE`, `MAGNITUDE_ERROR`, `FILTER`, `TRANSFORMED`, `MAG_TYPE`, `COMP_STAR_1`, `COMP_STAR_2`, `CHARTS`, `NOTES`, `NAME`, `AFFILIATION`, `MTYPE`, `GROUP`, `CHART`, `HERALD`
- Validator should: parse export file, check all required columns present, check value ranges (`JD > 2400000`, mag 1–25, filter codes valid), report any violations before upload
- Effort: 2–4h | Priority: **HIGH** (blocker for TODO-GS6) | Dependency: TODO-GS6

### MEDIUM priority
| TODO | Popis |
|------|-------|
| TODO-8 | ✅ CLOSED — ePSF tab + RUN ePSF + `run_epsf` job; Bootes validation → **TODO-8-BOO** |
| TODO-8-BOO | Bootes globular cluster validation: ePSF vs aperture on dense field (~2h data) | MEDIUM |
| TODO-19 | ✅ Hotovo |
| TODO-31 | ✅ CLOSED as NOT APPLICABLE |
| TODO-44 | ✅ CLOSED — Role-aware aperture (SIPS-style; SNR table + role factors) |
| TODO-45 | RGB kamera podpora (IMX533) — de-Bayer → G kanál |

### LOW priority
| TODO | Popis |
|------|-------|
| TODO-7 | Plate solver refactor |
| TODO-9 | Inštalácia VYVAR na Lenovo T460 (crossval env hotové) |
| TODO-10 | Settings záložka refaktor + CONFIG_GUIDE.md |
| TODO-11 | Auto-trigger po konci pozorovania |
| TODO-12 | HRD klasifikácia hviezd (po novej DB) |
| TODO-14 | PDF size optimization (po plate solving refaktore) |
| TODO-16 | ✅ UI crossmatch — `active_targets.csv` coords |
| TODO-17 | ✅ `crossmatch_auto_done` gated (aperture tab) |
| TODO-18 | ✅ Hotovo (UI preklad) |
| TODO-20 | Mean stack MASTERSTAR — **LOW**: zlepší WCS/FWHM na stacku, nie photometry SNR na single-exposure LC |
| TODO-23 | ✅ Adaptive matching radius (plate_scale × 3) |
| TODO-25 | ✅ Gaia→DAO — `pipeline_meta.json` + UI read |
| TODO-LIB | Compile VYVAR to `.pyd` (Cython) |
| TODO-29 | ✅ Hotovo |
| TODO-30 | ✅ Hotovo |

### Code quality — remaining tech debt
| # | Popis | Riziko |
|---|-------|--------|
| CQ-1 | ✅ `run_phase2a()` split (1515 → 188 lines + 3 helpers) | — |
| CQ-2 | ✅ `render_live_view()` split (1395 → 731 lines + 3 helpers) | — |
| CQ-3 | ✅ `select_comparison_stars_per_target()` → `comp_selection_per_target.py` (12 helpers) + ~356-line orchestrator in `photometry_core.py`; timing regression fixed: catalog_only skip + hybrid accumulate (iterrows N&lt;50, groupby N≥50); draft_321 validated | — |
| CQ-4 | ✅ `solve_wcs_with_local_gaia()` split (1843 → 1224 lines + 3 helpers) | — |
| CQ-5 | Wire alebo remove orphaned UI modules | MEDIUM |
| CQ-6 | Delete legacy importer/pipeline helpers | ✅ |
| CQ-7 | Photometry module surface (`photometry` vs `photometry_core` imports) | LOW |
| pandas FutureWarning | `variability_detector.py` downcasting — fix v ďalšej session | LOW |

### Performance optimization TODOs (z auditu 18.5.2026)

| TODO | Popis | Impact | Effort | Priorita |
|------|-------|--------|--------|---------|
| TODO-PERF-1 | ✅ COMPLETED — `csv_cache` už v calleri; hit/miss log v `_write_suspected_variables` | MEDIUM | done | HIGH |
| TODO-PERF-2 | ✅ COMPLETED — `ms_data`/`ms_header` z `run_phase2a` cache do PNG/edge helperov; shared header v `run_full_photometry_pipeline` | LOW | done | HIGH |
| TODO-PERF-3 | ✅ COMPLETED — comp star Gaia batched via `_comp_gaia_prefetch` before Phase 1 loop | MEDIUM | done | HIGH |
| TODO-PERF-4 | ✅ COMPLETED (Option A) — `comp_pool_rms.py` flux vectorized; `select_comparison_stars` → CQ-3 | MEDIUM | done | HIGH |
| TODO-PERF-5 | ✅ COMPLETED (Option B) — `ProcFrameStore` unified in-memory proc CSV cache | HIGH | done | MEDIUM |
| TODO-PERF-6 | ✅ COMPLETED — UI variability uses ProcFrameStore when in session | HIGH | done | MEDIUM |
| TODO-PERF-7 | ✅ COMPLETED — UI LC index: cached summary + overlay CSV cache | MEDIUM | done | MEDIUM |
| TODO-PERF-8 | ✅ COMPLETED — shared flux matrix (all LC stars × frames) before target loop; per-target slice replaces ~11k `read_flux_from_csv` calls; fallback preserved | MEDIUM | done | LOW |
| TODO-PERF-4B | ✅ COMPLETED — `_accumulate_per_frame_comp_metrics` hybrid: vectorized groupby N≥50, iterrows N&lt;50; log `[PERF-4B]`; catalog_only skip in Phase 1; draft_321 validated | MEDIUM | done | HIGH |
| TODO-PERF-9 | ✅ COMPLETED — vectorized haversine in `select_active_targets` + `_filter_comp_candidates_spatial_static`; cKDTree optional deferred | MEDIUM | done | LOW |
| TODO-PERF-10 | ✅ COMPLETED — DAO QC merged into calibration pass (`dao_qc_in_calibrate`) | HIGH | done | LOW |

### Algorithm upgrade TODOs (z Gemini analýzy 18.5.2026)

| TODO | Popis | Impact | Effort | Priorita |
|------|-------|--------|--------|---------|
| TODO-35 | ✅ SysRem MVP — `delta_mag_sysrem` on exported LC; full field deferred | HIGH — sub-mmag RMS | done (MVP) | HIGH |
| TODO-ALG-1 | BLS ground LC — parkovaný (TESS `tess_verify` only today) | — | TBD | LOW |
| TODO-ALG-2 | ✅ CLOSED — Savitzky-Golay detrend after airmass (`savgol_detrend_lc`) | opt-in | done | MEDIUM |
| TODO-ALG-3 | ✅ CLOSED — Temporal binning comp ensemble (MNRAS 2023, adaptive window cap) | BO ↓3% draft_321 | done | MEDIUM |
| TODO-ALG-4 | ✅ CLOSED — Democratic Detrender 3-model ensemble (arXiv 2411.09753v2) | CSV err bars | done | MEDIUM |
| TODO-ALG-5 | ✅ CLOSED — PyTICS iterative comp intercalibration (RASTI 2026) | done | done | MEDIUM |
| TODO-8 | ✅ CLOSED — ePSF infrastructure (build + export + Phase 2A + tab + RUN button) | — | — | — |
| TODO-8-BOO | Bootes globular cluster validation: ePSF vs aperture on dense field (~2h data) | MEDIUM — dense fields | validation ~2h | LOW — test dataset: globular in Bootes |

---

**TODO-ALG-3: Optimized Temporal Binning of comparison ensemble** — ✅ CLOSED (20.5.2026)

Reference: MNRAS (2023) 526, 3482–3489 — *"Optimised temporal binning of comparison star measurements for differential photometry"*

**Problem:** Shot noise + read noise of comp stars adds in quadrature to target noise. High-frequency random noise in comp ensemble artificially degrades LC quality (especially mag > 12, short exposures).

**Solution:** Before ZP subtraction in Phase 2A `ensemble_normalize()`, apply optimal temporal smoothing (rolling window / spline) to comp flux time series. Preserves low-frequency atmospheric trend but removes high-frequency random noise from comp measurements.

**Integration point:** `ensemble_normalize()` in `photometry_core.py` — smooth `comp_mag_inst` per comp star before weighted median. Optimal window size: find mathematically via minimizing target RMS over window sizes [3, 5, 7, 9, 11] frames.

**Expected gain:** 15–30% RMS reduction without hardware changes.

Impact: HIGH | Effort: 4–8h | Priority: MEDIUM

Config: `temporal_binning_enabled` (bool), `temporal_bin_window` (int, 0=auto)

---

**TODO-ALG-4: Democratic Detrender — ensemble multi-model detrending** — ✅ CLOSED (20.5.2026)

Reference: arXiv:2411.09753v2 (February 2026) — *"The democratic detrender: Ensemble-Based Removal of the Nuisance Signal in Stellar Time-Series Photometry"*

**Problem:** Current VYVAR uses linear/polynomial airmass fit — wrong polynomial degree risks underfitting (residual trend) or overfitting (erasing real physical variability like eclipse minimum).

**Solution:** Run 3 independent detrending models in parallel:
- A) Cosine filtering (current `airmass_detrend_lc` — keep as-is)
- B) Low-degree polynomial fit (degree 2–3)
- C) Gaussian Process regression (sklearn/george/celerite2)

Compute marginalized mean of all 3 models. Use MAD between models as adaptive error-bar inflation factor — adds model-selection uncertainty to per-point error bars.

**Integration point:** after `airmass_detrend_lc()` in `run_phase2a()`, new function `democratic_detrend(mag_calib, airmass, bjd, flags)`. Output: `delta_mag_democratic` + `err_democratic` columns in LC CSV.

**Note:** GP model is computationally expensive — consider optional (`democratic_gp_enabled` config flag). Cosine + Poly alone already gives marginalized errors without GP overhead.

**Expected gain:** publication-quality error bars; overfitting immunity.

Impact: HIGH (publication) | Effort: 8–16h | Priority: MEDIUM

Config: `democratic_detrend_enabled` (bool), `democratic_gp_enabled` (bool)

---

**TODO-ALG-5: PyTICS iterative comp star intercalibration** — ✅ CLOSED (20.5.2026)

Reference: RASTI (2026) — *"PyTICS: an iterative method for photometric light-curve intercalibration using comparison stars"*

**Problem:** Some comp stars are micro-variable (low amplitude, unknown to VSX). Current MAD sigma-clip catches gross outliers but misses stars with systematic scatter pattern across the night. These silently inject noise into ZP calibration.

**Solution:** Multi-component noise model in closed loop:
1. Compute preliminary ZP (current Broeg 2005 ensemble)
2. Compute per-comp residuals vs ZP across all frames
3. Assign lower weight to comps with systematically higher scatter
4. Recompute ZP with updated weights
5. Iterate until weights converge (typically 3–5 iterations)

**Integration point:** `check_comparison_stability()` and `ensemble_normalize()` in `photometry_core.py` — replace fixed MAD threshold with iterative weight update.

**Synergy:** combines well with TODO-ALG-3 (temporal binning) and existing Broeg (2005) weighted ensemble.

**Note:** VYVAR already has comp stability check + MAD sigma-clip — PyTICS is an evolutionary improvement, not a replacement. Keep existing logic as fallback / sanity check.

**Expected gain:** full autonomy from catalog quality; detects 0.01 mag micro-variability in comp stars.

Impact: MEDIUM-HIGH | Effort: 6–12h | Priority: MEDIUM

Config: `pytics_enabled` (bool), `pytics_n_iter` (int, default 5)

---

### Competitive position (vs MUNIWIN / AIJ / SIPS / MaxIm DL)

**VYVAR exceluje:**
- Plná autonómnosť (kávový test ☕) — žiadny iný softvér nemá
- SNR-optimálna per-star apertura — unikátne
- TESS + VSX + Gaia ekosystém — unikátne
- MAD sigma-clip ZP per frame — vzácnosť v amatérskom softvéri
- 200-stranový PDF report — bez konkurencie

**VYVAR zaostáva:**
- Interaktívne čistenie dát (AIJ dominuje)
- Exoplanétové tranzitné modely (Mandel & Agol)
- Hardvérová kontrola (MaxIm DL dominuje)
- Rýchlosť na veľkých poliach (C/C++ vs Python)
- ePSF pre preplnené polia — infrastructure done (TODO-8); validation **TODO-8-BOO**

**VYVAR algorithm upgrades** (literature-backed; ✅ implemented 20.5.2026):
- **TODO-ALG-3** ✅ — optimised temporal binning of comparison ensemble (MNRAS 2023)
- **TODO-ALG-4** ✅ — Democratic Detrender multi-model detrending (arXiv 2411.09753v2)
- **TODO-ALG-5** ✅ — PyTICS iterative comp intercalibration (RASTI 2026)
- **TODO-ALG-2** ✅ — Savitzky-Golay detrend (opt-in, default off)
- **TODO-ALG-1** ⏸ — BLS on ground LC (parked; TESS path unchanged)

### Gold standard — zostatok
| Krok | Stav |
|------|------|
| Peer-reviewed algoritmy s citáciami | ✅ |
| Reprodukovateľnosť | ✅ (vizuálne overená) |
| Transparentnosť | ✅ |
| Validácia (photutils, Muniwin, IRAF, SExtractor) | ✅ |
| Dokumentácia | ✅ unit testy |
| AAVSO validácia | ⏳ TODO-GS6b → TODO-GS6 |
| Open source README + CITATIONS.bib + CHANGELOG | ✅ |
| Paper draft | ⏳ TODO-GS7 |

---

*Session 18.5.2026 — 28 úloh dokončených*

# VYVAR STATE — 2026-05-18 (aktualizácia)

## Issues Status

### a) Double MASTERSTAR — COMPLETED ✅

### b) Comp stars = variable targets — COMPLETED ✅
- Fix 1: catalog_id dtype (str)
- Fix 2: Field map dedup in save_field_map_png
- Fix 3: Proximity veto in select_comparison_stars_spatial_grid
- Proximity veto now always logs (even 0 removals)

### c) Border filter — COMPLETED ✅
- Root cause fixed: aligned_files passed after RAM flush
- Verified manually via test_border_bbox.py on draft_000283
- safe_bbox_px = [30.4, 30.4, 2050.6, 1365.6]

### d) Stale x/y — COMPLETED ✅
- x/y refresh po MAKE MASTERSTAR implementovaný (`_refresh_variable_targets_xy` pred `select_active_targets`)
- Eliminuje issue (d) úplne

### e) SUMMARY MEASURE REPORT PDF — COMPLETED ✅
- PDF po fotometrii: `generate_photometry_report` v `photometry_report.py`
- Po RUN VYVAR volanie z `app.py` po `run_full_photometry_pipeline`

---

### TODO-1: Adaptívny config podľa hustoty poľa — COMPLETED ✅
### TODO-2: x/y refresh po MAKE MASTERSTAR — COMPLETED ✅
### TODO-3: Globálny comp pool — COMPLETED ✅
### TODO-4: Summary Measure Report PDF — COMPLETED ✅
### TODO-5: TESS reaktivácia — COMPLETED ✅
### TODO-6: BP-RP slidery v `ui_settings.py` — COMPLETED ✅

### BP-RP UI tabuľka — COMPLETED ✅
### VSX crossmatch bug fix — COMPLETED ✅
### ⚡ RUN VYVAR — COMPLETED ✅
### Draft_287 overenie — COMPLETED ✅
### Export TXT hlavička cleanup — COMPLETED ✅

---

## Dnes implementované (14.5.2026)

### RUN VYVAR — „I/O operation on closed file" — COMPLETED ✅
### Variabilita dashboard zjednodušenie — COMPLETED ✅
### TESS auto-trigger + robustnosť — COMPLETED ✅
### TESS — robustnosť a kvalita periódy — COMPLETED ✅
### TESS blend check alpha fix — COMPLETED ✅
### TESS auto-trigger for all candidates — COMPLETED ✅
### SUMMARY MEASURE REPORT refactor — COMPLETED ✅
### VYVAR_report 5 enhancements — COMPLETED ✅
### VYVAR_report PDF redesign (TODO-15) — COMPLETED ✅

---

## Dnes implementované (15.5.2026)

### MASTERSTAR QA — auto-load draft — COMPLETED ✅
- `render_masterstar_qa` (`ui_masterstar_qa.py`): `default_ap` priority chain
  1. `draft_dir_override`
  2. `vyvar_last_job_output["archive_path"]` / `vyvar_post_cal_archive_path`
  3. `Drafts/draft_{id}` z `vyvar_last_draft_id` alebo `draft_id` arg
  4. `vyvar_last_import_result.archive_path` (fallback)
- Pole sa automaticky predvyplní rovnako ako ostatné dashboardy

### Variabilita — katalógy stale "žiadny záznam" — COMPLETED ✅
**Root cause:** UI crossmatch bežal pred pipeline s mierne zlými koordinátmi
→ zapísal "žiadny záznam" do `bullets_map` a `_crossmatch/*.json`
→ pipeline crossmatch tieto záznamy považoval za "hotové" a preskakoval

**Fix reťazec (3 vrstvy):**
- Fix A: `load_katalogy_map_from_disk` + `_merge_katalogy_maps` v `ui_variability.py`
  → UI tabuľka vždy číta `variability_candidates.csv` z disku (disk = ground truth)
  → Sync späť do `var_catalog_bullets` (export + TESS + PDF vidia rovnaké dáta)
- Fix B: `_has_positive_catalog_match` v CSV skip guard (`crossmatch_runner.py`)
  → Pipeline preskakuje riadok len ak má aspoň jeden POZITÍVNY katalógový match
  → Riadky s iba "žiadny záznam" sa znova crossmatchujú
- Fix C: `_has_positive_catalog_match` v JSON cache guard (`crossmatch_runner.py`)
  → Stale cache súbory (len "žiadny záznam") sa zmažú a API sa zavolá znova
  → Cache write logika zostáva nezmenená

**Otvorené follow-ups (LOW priority):**
- TODO-16: UI crossmatch — použiť `active_targets.csv` coords namiesto flux-matrix
- TODO-17: `crossmatch_auto_done` — re-enable ak disk prázdny

### TESS duplikácia — UI nespúšťa TESS ak result.json existuje — COMPLETED ✅
- `_tess_result_json_on_disk(cid)` check pred UI auto-trigger
- `to_tess` a `_need_tess` vylučujú kandidátov kde `_tess/{cid}/result.json` existuje
- Pipeline TESS volania nedotknuté
- Efekt: RUN VYVAR spustí TESS raz; otvorenie Variabilita tabu nespustí znova

### SUMMARY MEASURE REPORT — kompletný refactor (TODO-14, TODO-15) — COMPLETED ✅

#### Fáza 1: Nová štruktúra stránok
| Str. | Sekcia |
|------|--------|
| 1 | Cover sheet (logo, title, draft/setup/dates) |
| 2 | Observation summary (metrics, conditions, methods, comp pool) |
| 3 | FITS Quality Assessment |
| 4 | Summary of all stars (sorted by vsx_type: EA→EB→EW→ROT→VAR, then lc_rms) |
| 5 | HRD (Hertzsprung-Russell diagram) |
| 6 | Field map (full page, landscape) |
| 7–N | Per-star pages (LC + field + comp table, one page each) |
| N+1 | Variability Analysis — RMS Hockey Stick |
| N+2 | Variability Candidates table |
| N+3+ | TESS Analysis (per candidate, 1–2 sektory/strana podľa výšky) |
| Last | Abbreviations & Notes |

#### Fáza 2: Hockey stick + Candidates vylepšenia
- Hockey stick: farebné odlíšenie (Stable=green, Known VSX=amber×,
  Candidate+match=orange●, Candidate no match=red●); ukladá `hockey_stick_report.png`
- Candidates table: pagination, katalogy=positive lines only (max 4+N more),
  row coloring (green=match, red=no match)

#### Fáza 3: Per-star + TESS layout
- Per-star: max 12 comp rows + "(+N more)", TESS odstránený z per-star strany
- TESS: header + metrics + period table + 1–2 sektory/strana (phased P + 2P + blend)

#### Fáza 4: PDF veľkosť optimalizácia
- `_compress_image_for_pdf`: JPEG rekompresia pred každým embed (typ→max_px+quality)
- Hockey stick DPI: 96
- plt.close(fig) po každom savefig
- Výsledok: 29 MB / 187 strán (z 53 MB / ~200+ strán)
- TODO-14 zostáva PENDING pre ďalšiu optimalizáciu po plate solving refaktore

#### Layout opravy (vizuálna revízia)
- Cover: em-dash fix ("VYVAR — Summary Measure Report"), logo 40% šírky, centrovaný
- Observation summary: Broeg referencia zalomená (textwrap 110 chars, font 7.5)
- FITS QA: odstránená "Top 5 Masterstar candidates" tabuľka; FWHM limit z
  `_qa_fwhm_limit_px` (rovnaká logika ako `ui_quality_dashboard.py`);
  masterstar frame z `_resolve_masterstar_used_frame` (FITS header → CSV → fallback)
- HRD: všetky slovenské labely preložené do angličtiny (`hrd_analysis.py`)
- Field map: duplikát odstránený; len full-page landscape verzia
- Hockey stick: landscape strana, 90%×80% využitie plochy
- Variability candidates: celé catalog_id (bez skracovania), dynamická výška riadku
  z `Paragraph.wrap`, "ďalších"→"more", "žiadny záznam"→"—"
- TESS: dynamický height budget (fit 1 alebo 2 sektory podľa dostupnej výšky),
  period analysis tabuľka (Sector|N pts|P(d)|Method|P_anova|P_consensus) z result.json
- Abbreviations: font 7.5pt, textwrap 55 chars, nové skratky (AAVSO, VAR.ASTRO,
  DAO, SNR, ZP, lc, comp, obs_group, dr)

---

## Dnes implementované (16.5.2026)

### TODO-13: Multi-step iterative matching — COMPLETED ✅
- **Best frame FWHM pre DAO** (nie median `VY_FWHM` v hlavičke); flag `masterstar_use_best_frame_fwhm`
  · `build_masterstar_from_detrended()` → `best_frame_fwhm_px` v `ms_selection_meta`
  · `dao_fwhm_bypass_header` v `detect_stars_and_match_catalog()`
  · draft_000303: median header ≈ 3.52 px → best frame ≈ 3.09 px
- **2-pass iteratívny DAO** — pass 2 na unmatched Gaia pozíciách (`_dao_targeted_pass2_unmatched_gaia`)
  · Pass 2 sigma: **1.9** (`masterstar_dao_pass2_sigma`, min. 1.5 v kóde)
  · Očakávaný match rate: **83% → ~95%+**
  · draft_000303 simulácia: **+889** detekcií (+25.3 %), merge 3515 → 4404
- **Match-rate monitoring** — log warning ak &lt; 88 % po 1. match passe (bez auto-retry)

### TODO-21: SNR-optimal per-star aperture — PARTIALLY COMPLETED ✅
- **Fáza 1–2** v `photometry_core.py`: gain/RN z `EQUIPMENTS` do `_photometric_error()`;
  `compute_snr_optimal_aperture_table()` + per-star `apertures_px` v Phase 2A;
  `draft_dir/aperture_snr_table.json`
- **Fáza 3 ✅ (16.5.2026):** per-frame `dao_flux` cez `enhance_catalog_dataframe_aperture_bpm()`
  + pipeline precompute SNR table pred exportom CSV
- draft_303 (gain 3.17, RN 7.6): mag 8 → 4.53 px; mag 11 → 3.43 px; mag 14 → 2.47 px
  (vs globálna ~3.99 px)
- **Pending:** Fáza 4 LC scatter validácia (e2e draft_305 + `photometry_summary.csv`)

---

## Dnes implementované (17.5.2026)

### Match rate metrika: Gaia→DAO completeness — COMPLETED ✅ (17.5.2026)
- **Pôvodné:** DAO→Gaia = n_matched_dao / n_detected_dao (všade)
- **Nové (paralelné):** Gaia→DAO = unique(catalog_id) / catalog_rows
- **pipeline.py** `detect_stars_and_match_catalog()`:
  · meta keys: `gaia_dao_completeness_pct`, `n_gaia_undetected`
  · log `[DAO] Gaia→DAO completeness: N/M Gaia stars detected (X.X%) | catalog_only: K`
  · warning ak &lt; 80 %
- **pipeline.py** `generate_masterstar_and_catalog()`:
  · raw aj optimized MATCH STATS log rozšírený o Gaia→DAO %
  · `[MASTERSTAR] Gaia→DAO completeness: N/M (X.X%) | catalog_only: K`
- **ui_masterstar_qa.py** `render_masterstar_qa()`:
  · m2: `DAO→Gaia Match (%)` (nezmenené)
  · m3: `Gaia→DAO Completeness (%)` z `field_catalog_cone.csv` / n_ok
  · captions: VYNIKAJÚCA ≥90% / DOBRÁ 80–90% / NÍZKA &lt;80%
- **Known limitation:** UI (n_ok/cone_rows) vs pipeline (unique catalog_id/catalog_rows)
  môžu mierne líšiť ak 1 Gaia hviezda → viac MASTERSTAR riadkov (rare)
- **Aktivuje sa po:** re-run MASTERSTAR / per-frame DAO na draft_305;
  UI potrebuje `field_catalog_cone.csv` v `platesolve/<setup>/`

### TODO-22: Gain/RN Settings UI + DB — COMPLETED ✅ (17.5.2026)
- **database.py**: `set_equipment_cosmic_params(equipment_id, gain, read_noise)`
  → uloží `GAIN_ADU` / `READNOISE_E` (NULL ak ≤ 0); `get_equipment_cosmic_params()` nezmenené
- **ui_settings.py** (tab Fotometria): sekcia „Detektor — fotometrické parametre"
  · všetky EQUIPMENTS riadky (`get_equipments(active_only=False)`)
  · per-kamera expander: Gain [e⁻/ADU] + Read Noise [e⁻] inputs → Uložiť
  · warning ak hodnoty chýbajú (Phase 2A fallback 1.0 / 10.0)
- **ui_quality_dashboard.py**: banner po načítaní light rows
  · info (modrý) ak gain+RN nastavené v DB
  · warning (žltý) s odkazom na Settings → Fotometria → Detektor ak nie
  · caption ak draft nemá equipment_id
- Database Explorer grid: bez zmeny

### VYVAR UI — preklad do angličtiny — COMPLETED ✅ (17.5.2026)
- **ui_variability.py**: ~82 strings (Streamlit UI, Plotly, matplotlib, TESS, crossmatch)
- **ui_masterstar_qa.py**: ~38 strings (metrics, completeness captions, VSX controls)
- **ui_quality_dashboard.py**: ~45 strings (hover templates, FWHM, gain/RN banner)
- **ui_settings.py**: ~95 strings (tab labels, help blocks, detector UI, Phase 0+1)
- **ui_photometry.py**: ~18 strings (subheader, checkboxes, help, save button)
- **photometry_report.py**: 0 changes (PDF strings already in English)
- Nezmenené: LOGGER.*, log_event, column keys, catalog names (VSX/Gaia/TESS),
  config keys, CSV column names, `žiadny záznam` parsing
- py_compile: ✓ všetky súbory

### Fix: float64 catalog_id precision loss → XY fallback → false candidates (17.5.2026)
- **Root cause:** Gaia IDs (int64, ~19 digits) uložené ako float64 v proc_*.csv
  strácajú posledné bity → ID lookup miss → XY fallback → NaN frames
  → zlá LC → záporné airmass slopes → falošní variability kandidáti
- **Reťazec:** float64 precision → ID miss → 70× NaN/frame → LC scatter ↑
  → airmass fit zlyhá (slope < 0) → Hockey Stick červený bod
- **gaia_catalog_id.py:**
  · `normalize_gaia_source_id()`: Decimal pre large/sci-notation IDs
  · `catalog_id_series_for_masterstars_export()`: blank pre missing IDs
  · `GAIA_PROC_CSV_READ_DTYPE` + `catalog_id_series_for_proc_csv_export()`
- **photometry_core.py:**
  · `_build_csv_lookup()`: indexuje pod masterstar_row_gaia_key + name + catalog_id
  · `_lookup_star_in_csv()`: normalizuje cid pred lookup
  · `read_flux_from_csv()`: cid_key = _normalize_gaia_id(cid); KNOWN ISSUE blok odstránený
- **pipeline.py + diagnose_*.py:** `dtype={"catalog_id": str, "name": str}` všade
- **Potrebuje:** re-export proc_*.csv + re-run Phase 2A pre draft_305/307
- **Očakávané:** nula XY fallback warnings, viac am_detrended=True, menej red kandidátov

### Fyzikálny audit — CCD equation opravy (17.5.2026)
Zdroj: Howell (1989 PASP 101:616), DAOPHOT/MUNIWIN/IRAF referenčná implementácia.

**FIX 1 — _photometric_error: sky term /gain (photometry_core.py)**
- Chyba: `sky_pp × area` (ADU²) namiesto `sky_pp/gain × area` (e⁻)
- Oprava: `variance = flux/g + sky_pp/g × area + (RN/g)² × area`
- Dopad: err 2.87% → 1.69% pre sky-dominated hviezdy (mag > 12)
  → ~41% redukcia fotometrických chýb pre slabé hviezdy
- `compute_snr_optimal_aperture_table()`: nezmenená (samostatná TODO)

**FIX 2 — ensemble_normalize(): ZP sigma-clip per frame**
- Chyba: žiadny outlier reject na ΔZP = cat_mag − inst_mag per frame
- Oprava: MAD-based σ-clip (3σ) pred weighted mean ZP
  · len ≥ 4 comps: clip, re-compute weighted ZP
  · fallback nanmedian ak &lt; 2 po clipe
  · log `[ZP] Frame sigma-clip: N/M comps kept` pri outlieroch
- Zdroj: štandard IRAF phot / MUNIWIN (iteratívny ZP clip)
- Dopad: LC spiky z cosmic rays / saturovaných comp frameov

**Zostatok z auditu (ďalšie TODO):**
- SNR table: ADU/e⁻ mix v compute_snr_optimal_aperture_table()
- Airmass fit pred outlier detekciou (swap poradia)
- Color term aplikovaný pred airmass (nie na CT-korigovanom)
- Growth curve korekcia pre mag > 13 (Howell 1989 §3)

---

## TODO — backlog (updated 21.5.2026; draft_342 validated)

### TODO-13: Multi-step iterative matching — COMPLETED ✅ (16.5.2026; Phase A+B 21.5.2026)

**Phase A (21.5.2026):** Pass 2 catalog-seeded DAO v `detect_stars_match_master_reference` (MASTERSTAR fast path) — `_dao_targeted_pass2_unmatched_gaia` po pass 1 + prefilter.

**Phase B (21.5.2026):** `_inject_forced_aperture_rows` pre unmatched master stars; `_proc_catalog_keep_matched_rows_only` — len `GAIA_MATCHED` + `FORCED_APERTURE`; **0 `DET_*`** v proc CSV; `source_type` filter.

**E2E:** draft_342 — 134 LC, DET_*=0; Gaia→DAO completeness v QA dashboard stále nízka (~3.5%) → **TODO-13 validation** otvorené.
### TODO-14: PDF size optimization — PENDING (po plate solving refaktore)
- Aktuálne: 29 MB / 187 strán
- Cieľ: < 10 MB
- Hlavná príčina: LC grafy generované inline (matplotlib → priamo do PDF, bez disk cache)
- Riešenie: po TODO-13 (menej catalog_only hviezd = menej LC strán)

### TODO-15: VYVAR_report PDF redesign — COMPLETED ✅ (15.5.2026)

### TODO-16: UI crossmatch — active_targets.csv coords — COMPLETED ✅ (20.5.2026)
- `_get_candidate_row()` preferuje `active_targets.csv` RA/Dec (WCS-verified; mirrors `crossmatch_runner`)

### TODO-17: crossmatch_auto_done gate — COMPLETED ✅ (20.5.2026)
- Aperture tab: `crossmatch_auto_done=True` len keď `candidates` non-empty (nie pri prázdnom zozname)

### TODO-18: Field map — preložiť titulok do angličtiny — LOW
- Titulok "VYVAR — Field Map (červené=target, cyan=catalog_only, zelené=comp)"
  sa generuje v pipeline (save_field_map_png alebo podobne)
- Preložiť pri celkovom anglickom prepise VYVAR UI

### TODO-20: Mean stack MASTERSTAR — PENDING (priorita **LOW**)
- Zkombinovať 5 najlepších frames (sorted by FWHM) do median stack pred DAO detekciou
- Očakávaný benefit: √5 ≈ 2.2× lepší SNR na **detekciu** (MASTERSTAR / WCS / FWHM odhad) — **nie** na Phase 2A photometry SNR (LC stále z jednotlivých single-exposure frames)
- Poznámka: mean stack zlepší kvalitu WCS/FWHM, ale nezvýši photometry SNR v časovej sérii, ak každý frame zostane single-exposure limited
- Predpoklad: single best frame fix (TODO-13) musí byť overený e2e pred implementáciou stacku
- Implementovať po validácii TODO-13 na reálnom observe

### TODO-21: SNR-optimal per-star aperture — PARTIALLY COMPLETED ✅ (16.5.2026)

Fáza 1 ✅ — gain/RN z DB do Phase 2A `_photometric_error()`
  · gain=3.17 e⁻/ADU, RN=7.6 e⁻ pre QHY294MM (draft_303)
  · variance = flux/gain + sky·area/gain + RN²·area/gain²
  · log: `[PHASE 2A] Photometric errors: gain=X e-/ADU, RN=Y e-`

Fáza 2 ✅ — `compute_snr_optimal_aperture_table()`
  · Gaussian PSF enclosed flux model
  · SNR(r) = F(r) / sqrt(F(r)/g + π·r²·sky/g + π·r²·(RN/g)²)
  · Lookup table mag 7–18, krok 0.5 mag
  · Uložené: `draft_dir/aperture_snr_table.json`

Fáza 2 ✅ — Phase 2A per-star apertures z SNR table
  · `apertures_px[cid] = r_opt(mag_hviezdy)`
  · Clamped: r_min=0.8×FWHM, r_max=2.5×FWHM

Výsledky pre draft_303 (FWHM=3.094, sky=1581.6, gain=3.17, RN=7.6):
  · mag 8 → 4.53 px (+13% vs globálna 3.99)
  · mag 11 → 3.43 px (−14%)
  · mag 14 → 2.47 px (−38%)

Fáza 3 ✅ (16.5.2026) — per-frame CSV + pipeline precompute

`photometry_core.py`:
  · `snr_aperture_table=None` na `enhance_catalog_dataframe_aperture_bpm()`
  · `_get_star_aperture_px()` + `_snr_table_radius_for_mag_bin()`
  · `CircularAperture(pos, r=r_ap_arr)` → per-row `aperture_r_px`
  · `load_snr_aperture_table_from_draft_dir()`
  · `resolve_fwhm_px_for_snr_aperture_table()`
  · `estimate_median_sky_adu_per_px_for_snr_table()`
  · `precompute_and_save_snr_aperture_table_for_draft()`

`pipeline.py`:
  · Pred `export_per_frame_catalogs()`: precompute SNR table (gain/RN z DB, FWHM z MASTERSTAR, sky z aligned frames)
  · `aperture_snr_table.json` do `draft_dir` **pred** Phase 2A
  · `snr_aperture_table` predaná do `enhance_catalog_dataframe_aperture_bpm()`
  · Celé v `try/except` — backward compatible (globálna apertúra pri zlyhaní)

Poradie (jeden pipeline beh):
  MASTERSTAR → SNR table precompute → export CSV (per-star r_opt)
  → Phase 2A (môže JSON aktualizovať s presnejším sky z `proc_*.csv`)

Logy: `[PIPELINE] aperture_snr_table.json uložená pred exportom CSV` ·
  `[FÁZA 2A] SNR per-star apertures: min=… median=… max=…`

**⚠️ Regresia (21.5.2026, draft_341):** photutils **2.3.0** — array `r` na `CircularAperture` → tichý `except` → `dao_flux=NaN`. **FIXED ✅ (`3d2363c`):** `_aperture_flux_sky_per_star` loop; `FORCED_APERTURE` → `photometry_ok=True`; WARNING log namiesto silent swallow. Validované draft_342: HIP 67011 `lc_rms=0.012`, 100% forced flux.

Fáza 4 ✅ PARTIAL PASS (17.5.2026) — validácia LC scatter
  Setup: NoFilter_60_2 · 139 snímok · draft_303 (PRED) → draft_305 (PO)

  Mag bin | RMS PRED | RMS PO  | Párovaný Δ | Verdikt
  8–10    | 0.76 %   | 0.83 %  | −5 %       | ❌ mierne horšie (r_opt ≈ r_global; sky estimate vyšší v 305)
  11–12   | 2.52 %   | 2.14 %  | +7 %       | ✅ nad očakávaním (cieľ 5–10 %)
  13–14   | 4.90 %   | 4.39 %  | +11 %      | ⚠️ pod cieľom 15–25 % (sky 2060 vs 1581 ADU/px agresívny)
  15+     | 3.14 %   | 1.94 %  | +15 %      | ✅ v rozsahu (N=4, orientačné)

  Celkový medián: 2.41 % → 2.20 % (~9 % relatívne zlepšenie)
  Otvorené: aperture_r_px chýba v 305 proc_*.csv — overiť BPM path
  Odporúčanie: per-star aperture benefit pre mag ≳ 11; mag 8–10 flat

Gain/RN zdroj (priorita): EQUIPMENTS DB ✅ pre draft_303 · Settings UI ✅ (TODO-22) · fallback 1.0/10.0

### TODO-22: Gain/RN do Settings UI a DB — COMPLETED ✅ (17.5.2026)

### TODO-24: Vylúčiť catalog_only z photometry pipeline — COMPLETED ✅ (17.5.2026)
- **Flag:** `zone_flag == "catalog_only"` alebo `zone == "catalog_only"`
  (`has_dao_detection` neexistuje v kóde)
- **Helper:** `_is_catalog_only(df)` — lokálne v každom súbore;
  backward compatible (chýbajúci stĺpec → maska all-False)
- **Fix A** `photometry_core.py` — `build_global_comp_pool()` +
  `select_comparison_stars_per_target()` +
  `auto_export_variability_candidates_csv()`;
  log `[COMP] catalog_only excluded: N removed, M remain`
- **Fix B** `photometry_core.py` `run_phase2a` — pred zápisom
  `photometry_summary.csv`; draft_305: 148 → 86 riadkov (−62);
  log `[PHASE 2A] photometry_summary.csv: excluded N catalog_only stars`
- **Fix C1** `ui_variability.py` — Hockey Stick scatter + flux matica +
  comp CSV; log `[HOCKEY STICK UI] Excluding N catalog_only`
- **Fix C2** `photometry_report.py` — po načítaní
  `photometry_summary.csv`; log `[HOCKEY STICK] Excluding N catalog_only stars`
- **Fix D** `photometry_core.py` + `ui_variability.py` — pred
  sigma-clipping; VSX catalog_only zachované s `catalog_only_warning=True`;
  log `[VARIABILITY] Excluding N catalog_only from candidate detection`
- **Ponechané bez zmeny:** `proc_*.csv`, `active_targets.csv`,
  field map (cyan body), `pipeline.py`
- **Future:** proc_*.csv zone=NaN (~197/frame Gaia-fill) — riešiť samostatne
- **Phase 2A LC loop skip** (17.5.2026): `run_phase2a()` preskočí
  `catalog_only` na začiatku per-target loopu (pred flux/LC);
  log `[PHASE 2A] Skipping N catalog_only targets`

### UI counter fixes — konzistentné čísla (17.5.2026)
- **Phase 2A progress log** (`photometry_core.py`):
  `[PHASE 2A] 148 targets (84 active LC + 64 catalog_only skipped)`
  (pred: "148 targetov" bez rozlíšenia)
- **Hockey Stick metric** (`ui_variability.py`):
  "RMS candidates" = `n_combined` (po catalog_only + edge + VSX filter)
  = rovnaká množina ako tabuľka a `variability_candidates.csv`
  (pred: raw `is_variable_candidate_rms` ~193)
- **TESS auto-run log**:
  `[TESS] Auto-run eligible: N candidates (from variability_candidates.csv)`

### Cross-validácia — photutils + SExtractor + IRAF (17.5.2026)
Setup: Lenovo T460, Kubuntu, Python 3.12, photutils 3.0.0
Frame: `proc_BO_CVn_Light_001.fits`, 1750 hviezd, FWHM=3.094px

**Growth curve analýza (photutils vs VYVAR dao_flux):**

| Apertura | ×FWHM | Bright (8-10) | Medium (10-12) | Faint (12-13) |
|----------|-------|---------------|----------------|---------------|
| 2.0 px   | 0.65× | 0.808         | 0.798          | 0.707         |
| 3.0 px   | 0.97× | 1.021         | 1.000 ✓        | 0.933         |
| 3.5 px   | 1.13× | 1.092         | 1.071          | 0.999 ✓       |
| 4.0 px   | 1.29× | 1.155         | 1.122          | 1.035         |

- VYVAR `dao_flux` (r=4.0px nominálne) ≈ photutils r=3.0px (0.97×FWHM)
- Efektívna apertura VYVAR ~0.97×FWHM pre medium hviezdy
- Sky subtrakcia VYVAR odčíta ~15% viac ako photutils globálny median

**SExtractor 2.28.0 cross-validácia (MASTERSTAR.fits):**
- Detekovaných: 1575 hviezd (VYVAR: 1750)
- Matched < 3px: 1439 / FLAGS=0: 1259
- FWHM median SExtractor: 2.25px (VYVAR header: 3.09px)

| Mag bin | N | SEx/VYVAR ratio | Photutils/VYVAR (r=3px) |
|---------|---|-----------------|------------------------|
| 8–10    | 60 | 1.065          | 1.021                  |
| 10–12   | 273| 1.058          | 1.000                  |
| 12–13   | 394| 0.919          | 0.933                  |

**IRAF/PyRAF 2.2.4 cross-validácia:**
- Nástroj: Community IRAF V2.17.1, task `apphot.phot`
- Apertura: r=3.0px, annulus 6–10px, sky=median, gain=3.17 e⁻/ADU
- Hviezdy: 48 (mag 8–12, flag=0/NoError)
- ZP offset: 24.977 mag (očakávané 25.0; Δ=0.023 = EPADU=1.0 vs 3.17)
- Scatter std: 2.38% — VYVAR a IRAF zhodujú sa na 2.2%
- Flux ratio median: 1.0000 (po ZP korekcii)

**Finálna tabuľka trojitej cross-validácie:**

| Nástroj | Zhoda | Poznámka |
|---------|-------|----------|
| photutils 3.0 (r=3px) | 2.0% scatter | Optimálna apertura = 0.97×FWHM |
| SExtractor 2.28 | 6% offset | Growth curve efekt (PSF wings) |
| IRAF apphot (r=3px) | 2.2% scatter | Po ZP korekcii; gain fix needed |

**Záver cross-validácie:**
VYVAR diferenciálna fotometria je fyzikálne správna a konzistentná
s tromi nezávislými profesionálnymi nástrojmi na úrovni 2–6%.
Systematický 6% offset (SExtractor) = growth curve efekt — identický
v IRAF/photutils keď sa použije r=4px vs r=3px.

**Nástroje nainštalované na Lenovo:**
- SExtractor 2.28.0 ✓
- photutils 3.0.0 ✓
- astropy 7.2.0 ✓
- ccdproc 2.5.1 ✓
- IRAF/PyRAF 2.2.4 (Community IRAF V2.17.1) ✓
- `scripts/validate_photometry_crossval.py` ✓
- `scripts/install_vyvar_crossval_lenovo.sh` ✓

### FWHM priority fix — VY_FWHM_GAUSS pre per-frame apertúry (17.5.2026)
- **pipeline.py** `gaussian_fwhm_px_override` (~8088):
  · Pred: PRIORITA 1 = `VY_FWHM × 0.667` (vždy ak `VY_FWHM` existuje)
  · Po: PRIORITA 1 = `VY_FWHM_GAUSS` (priamo, bez škálovania)
           PRIORITA 2 = `VY_FWHM × 0.667` (fallback)
- Dopad: `fwhm_gaussian_px` ~2.25px namiesto ~2.06px (+9%)
  → `r_ap` mierne väčšia → lepší flux capture pre faint hviezdy
- Zjednotenie: per-frame, SNR table aj Phase 2A teraz všetky
  používajú `VY_FWHM_GAUSS` ako primárny zdroj

### TODO-26: Variability threshold auto-calibration — COMPLETED ✅ (17.5.2026)
- **Problém:** 180 false candidates (mag 12-13, RMS 8-24%, Jaccard=0.99
  across runs) — systematický noise floor, nie reálna variabilita
- **Root cause:** field envelope fit (~10% upper @ mag 12) príliš nízky
  pre tento field/night (comp P90 rms_pct @ mag 12 = 8.7%)
- **Fix** `variability_detector.py`:
  · Per mag bin: P90 of comp stars flux matrix rms_pct
  · `upper_envelope = max(field_upper, comp_P90[bin] × factor)`
  · Missing bins: fallback `expected_rms_pct × 2`
  · Log: `[VARIABILITY] Comp noise floor applied: P90@mag12=X% → upper=Y%`
- **Config:** `variability_comp_floor_factor = 1.5` (tunable)
- **Výsledok (draft_309):**
  · Upper envelope @ mag 12: ~13.0% (8.7% × 1.5)
  · RMS candidates: 248 → 47 (všetky mag)
  · Band mag 12-13: 128 → 21
  · VSX known variables: 3 zachované (0 stratených)
- **Tuning:** zvýšiť factor → menej kandidátov, znížiť → viac
- **Ďalší krok:** SysRem MVP hotové (TODO-35); full-field SysRem po PERF-5/PERF-6

### TODO-27: Re-export + re-validate draft_305/307 po float64 fix — CLOSED ✅ (18.5.2026)
- Re-validácia vykonaná na draft_310 (Lenovo, `validate_lc_crossval.py` v4)
- 17 hviezd NOT IN proc CSV = fyzikálna detekčná limita DAO (nie float64 bug)
- Tieto hviezdy nie sú v žiadnom proc frame naprieč draft_303, draft_310, `/media/milan/DISK`
- float64 fix (17.5.2026) je správny a platný; 17 hviezd je legitímne preskočených
- PyRAF LC validácia: NOT FEASIBLE (IRAF float32 FITS bug) — single-frame IRAF
  validácia z 17.5.2026 (2.2% zhoda) zostáva platná

### TODO-28: SNR table units fix — ADU/e⁻ mix — COMPLETED ✅ (17.5.2026)
- `compute_snr_optimal_aperture_table()`: `snr = enclosed(ADU) / noise(e⁻)` → `snr = (enclosed/g) / noise`
- Dopad: absolútna SNR bola chybná o ~√gain; `r_opt` poloha maxima nezmenená

### TODO-29: Airmass fit — swap poradia — COMPLETED ✅ (18.5.2026)
- Nový poriadok: ZP → CT → outlier detect → airmass fit (na čistých dátach)
- Airmass korekcia aplikovaná na všetky frames; fit len na non-outlier frames
- Log: `[PHASE 2A] Airmass fit on N/M frames (after outlier mask)`
- Dopad: airmass slope nie je skreslený oblakmi / bad frames

### TODO-30: Color term → airmass na CT-korigovanom mag — COMPLETED ✅ (18.5.2026)
- `mag_for_airmass = mag_calib_ct` (nie `mag_calib`)
- NoFilter/Clear: nulový dopad (mag_calib_ct == mag_calib)
- Filtered + CT on: log `[PHASE 2A] Airmass detrend applied on CT-corrected mag`

### TODO-GS1: Citačné komentáre — COMPLETED ✅ (18.5.2026)
- Howell (1989) PASP 101:616 — CCD equation, sky subtraction, SNR aperture
- Broeg, Fernandez & Neuhäuser (2005) AN 326:134 — comp weights, ZP ensemble
- Stetson (1987) PASP 99:191 — ZP MAD sigma-clip
- 6 lokácií v photometry_core.py (riadky 814, 837, 2174, 2181, 5732, 8827)

### TODO-GS2: Unit test suite — COMPLETED ✅ (18.5.2026)
- `tests/test_photometry_core.py` — 11 testov, 11/11 passed (~0.9s)
- Pokryté: Howell CCD error, Broeg weights, sky subtraction, ZP sigma-clip, SNR aperture
- Spustenie: `python -m pytest tests/test_photometry_core.py -v`

### TODO-31: Growth curve korekcia pre faint hviezdy — CLOSED as NOT APPLICABLE ✅ (18.5.2026)
- Implementovaná a testovaná na draft_310
- Výsledok: Median lc_rms 0.1114 → 0.1145 (mierne horšie), RMS<0.05 = 23 (bez zmeny)
- Príčina: growth curve faktor sa aplikuje rovnako na target aj comp hviezdy
  → eliminuje sa v diferenciálnom magnitude → nulový efekt na LC scatter
- Korekcia má zmysel len pre absolútnu fotometriu (mag kalibrácia), nie diferenciálnu
- Kód revertovaný do pôvodného stavu (py_compile OK)
- Referencia: Howell (1989) §3 — growth curve correction pre small apertures

### TODO-32: IRAF EPADU fix pre cross-validáciu — LOW
- IRAF varoval "Keyword 3.17 not found" → použil EPADU=1.0
- Fix: v cross-val scripte použiť `iraf.datapars.epadu=3.17`
  namiesto `iraf.datapars.gain="3.17"` (string vs float)
- Dopad na ZP: 0.023 mag = 2.5×log10(3.17) / correction
- Pre VYVAR kód: žiadny dopad (len pre cross-val script)

### TODO-44: Role-aware aperture (variable vs comp) — ✅ CLOSED (20.5.2026)
- `_apply_role_aware_aperture_scaling()` v `photometry_core.py` po SNR tabuľke
- Config: `aperture_variable_factor` (default 1.0), `aperture_comp_factor` (default 1.1)
- E2E draft_321: 260 comps scaled ×1.1; BO/FW `lc_rms` unchanged vs baseline
- Reporting A: PDF Methods (dynamic ALG + aperture factors), VAR.ASTRO `# ALG:` header lines, `ui_settings.py` sliders/toggles
- Referencia: SIPS Photometry tool documentation (Moravian Instruments)

### TODO-45: RGB kamera podpora (de-Bayer → G kanál fotometria) — MEDIUM

**Motivácia:** IMX533 (RGB) užívateľská podpora; G kanál ≈ širší V-filter,
vhodný pre diferenciálnu fotometriu premenných hviezd.

**Fáza 1 — Minimálna verzia (G kanál only):**
- Detekcia Bayer FITS z headera (`BAYERPAT` keyword: RGGB / BGGR / GRBG / GBRG)
- De-Bayer cez `opencv` (`cv2.cvtColor`) → extract G kanál → štandardný 2D FITS
- Zvyšok pipeline (MASTERSTAR, Phase 2A, PDF): bez zmeny
- Flat kalibrácia: použiť G-channel flat (alebo luminance flat ako fallback)

**Fáza 2 — Plná verzia (future):**
- Per-kanál R+G+B fotometria
- Color index R−G, B−G export → HRD klasifikácia bez Gaia BP-RP
- Per-kanál LC export

**Náklad:** Fáza 1 = 3–5 dní · Fáza 2 = 3–4 týždne
**Závisí od:** nič (nezávisí od iných TODO)
**Referencia:** IMX533 Bayer pattern = RGGB

### TODO-25: Gaia→DAO zdroj UI vs pipeline — COMPLETED ✅ (20.5.2026)
- Pipeline: `generate_masterstar_and_catalog()` → `platesolve/<setup>/photometry/pipeline_meta.json`
- UI: `ui_masterstar_qa.py` číta `gaia_dao_completeness_pct`; fallback na CSV row counts ak meta chýba

### TODO-23: Adaptívny matching radius z pixel scale — COMPLETED ✅ (20.5.2026)
- `select_active_targets` / `run_phase0_and_phase1`: `match_radius = plate_scale × 3` (fallback 15″)
- `_read_plate_scale_from_fits_path`: SCALE, VY_PLTS, CDELT1, …

### TODO-19: Infolog — ukladanie na disk — MEDIUM
- Aktuálne logy sú len v pamäti (Infolog dashboard)
- Pridať automatické ukladanie do draft_dir/infolog_<date>.txt po každom RUN VYVAR

### TODO-7: Plate solver refactor — LOW (nízka priorita)
### TODO-8: ePSF napojenie do pipeline — ✅ CLOSED (20.5.2026)
- **Infrastructure:** `build_epsf_model()` after MASTERSTAR; per-frame `psf_photometry_stars()` via `export_per_frame_catalogs`; `_get_lc_psf_or_dao()` in Phase 2A; `psf_chi2_threshold` config (default 50)
- **UI:** standalone 🔬 ePSF tab (`ui_epsf_dashboard.py`); metrics + aperture/PSF overlay; `run_epsf` job + ⚡ RUN ePSF Photometry button (no full RUN VYVAR)
- **Validated draft_321:** 157 PSF stars, 96/139 BO `psf_fit_ok`, `lc_rms` unchanged (differential ZP)
- **Default:** `psf_photometry_enabled=false` in `config.json`
- **Follow-up:** **TODO-8-BOO** — Bootes globular cluster dense-field validation (~2h dataset)
### TODO-9: Inštalácia VYVAR na Lenovo T460 — PENDING
### TODO-10: Settings záložka refaktor + CONFIG_GUIDE.md — PENDING
### TODO-11: Auto-trigger po konci pozorovania — PENDING (nízka priorita)
### TODO-12: HRD klasifikácia hviezd — PENDING (po novej DB)

### TODO-LC-QUALITY: LC Classification Filter — PENDING

**Kontext (draft_342):** Phase 2A hlási **134** LC; HIP 67011 `lc_rms=0.012` po aperture fixe. Stále treba formalizovať saturated / noisy2/3 / `no_data` a zosúladiť počítadlo LC s export kvalitou.

**Úlohy:**

1. **Saturated hviezdy (`zone=saturated`)**
   - Vylúčiť z AAVSO / VAR.ASTRO exportu, **alebo**
   - Exportovať so špeciálnym flagom (dokumentovať v AAVSO komentári / metadata).

2. **Noisy krivky (`noisy2` / `noisy3` v existujúcej klasifikácii)**
   - Definovať RMS threshold pre akceptovateľnú kvalitu (per target / per zone?).
   - Zosúladiť s Variability tab / Hockey Stick / PDF reportom.

3. **`lc_quality_flag` vo výstupe**
   - Hodnoty: `good` | `saturated` | `noisy` | `no_data` (prípadne `catalog_only` / `forced_no_flux`).
   - Stĺpec v `photometry_summary.csv` + propagácia do export reportov.

4. **Prehodnotiť počítadlo „Light curves: N“**
   - Fáza 2A: počítať len `good`? alebo všetky s existujúcim `.csv`?
   - UI / night_run / infolog: jednotná definícia (napr. `n_lc_good` vs `n_lc_total`).

**Súvisiace:** photutils 2.3 fix hotový (`3d2363c`); staré drafty bez re-exportu môžu mať staré proc CSV.

---

### TODO-LC-TREND: Differential extinction + ALG audit — PENDING

- Rastúce/klesajúce LC u niektorých ROT hviezd (vizuálna kontrola + fyzikálna interpretácia).
- Audit existujúcich ALG krokov (airmass, CT, SysRem, Democratic, PyTICS) vs trend artefakty.
- Súvisí s **TODO-GEO** (pozorovateľská poloha) a **TODO-LC-QUALITY**.

---

### TODO-GEO: Observer geographic position audit — PENDING

- Overiť BJD / HJD / airmass pre draft_342 (observer lat/lon/elev v DB + FITS + `time_utils`).
- Impact na airmass detrend a periodické trendy v LC.

---

### TODO-MASTERSTAR-QA validation: FORCED_APERTURE overlay — PENDING

- Kód: cyan = `FORCED_APERTURE`, green = `GAIA_MATCHED` (`masterstar_qa_plot.py`).
- **Úloha:** otvoriť MASTERSTAR QA na draft_342 a potvrdiť cyan body na správnych WCS pozíciách.

---

### TODO-DEV-PROCESS: Improved Development & Testing Process — PENDING

**Motivácia:** Commit `0b01955` (ePSF targeted) išiel do repo bez overenia, že `_epsf_target_catalog_ids` volá `read_vyvar_csv` — v runtime vždy `NameError` → PSF na všetkých riadkoch. Podobne draft_341: proc CSV s `dao_flux=NaN` pre forced rows kvôli tichému pádu SNR per-star apertures (photutils 2.3).

**Povinný postup po každej implementácii (pred commitom):**

1. **Smoke test** na jednom frame / jednom draft (min. 5 min, nie len `py_compile`).
2. **Checklist pre novú feature:**
   - [ ] Unit / smoke test na jednom frame (`proc_BO_CVn_Light_001` alebo referenčný draft)
   - [ ] Porovnanie metrík **pred / po**: LC count, `DET_*` count v proc CSV, `psf_flux` non-null count
   - [ ] Commit až po úspešnom teste (žiadny „fix neskôr v ďalšom commite“ bez dôvodu)

**Automatizovaný regression test — `scripts/regression_test.py` (nový):**

- Spustí pipeline krok(y) na referenčných dátach (draft_341 alebo pinnutý baseline draft).
- Porovná s baseline:
  - LC count ≈ **134** (alebo po TODO-LC-QUALITY: **good** LC count)
  - `DET_*` names v proc CSV = **0**
  - `psf_flux` non-null per frame ≈ **183** (targeted ePSF)
- **FAIL** pri regresii (tolerance konfigurovateľné v JSON / CLI).
- Voliteľne: subprocess na `export_per_frame_catalogs` + jeden frame `enhance_catalog_dataframe_aperture_bpm` s `snr_aperture_table` (chytí photutils array-`r` bug).

**Referenčný baseline (`draft_342`, 21.5.2026):**

| Metrika | Očakávanie | draft_342 |
|---------|------------|-----------|
| `DET_*` v proc CSV | 0 | ✅ 0 |
| `psf_flux` non-null / frame | ~150–200 | ✅ targeted ePSF |
| LC count | ≥ 134 | ✅ 134 |
| HIP 67011 `lc_rms` | finite | ✅ 0.012 |
| `FORCED_APERTURE` `dao_flux` | 100% | ✅ |

**Opravy už v repo (pred regression scriptom):**

- `read_vyvar_csv` import — `0b01955` ✅
- SNR per-star aperture photutils 2.3 loop — `3d2363c` ✅

---

### Dnešný súhrn — 17.5.2026

#### Implementované dnes
| # | Zmena | Súbor(y) |
|---|-------|----------|
| TODO-24 full-stack | catalog_only vylúčenie (comp, summary, Hockey Stick, PDF, variability, Phase 2A LC skip) | photometry_core.py, ui_variability.py, ui_masterstar_qa.py, photometry_report.py |
| TODO-21 Fáza 4 | Partial pass validácia (mag 11+ ✅) | — |
| TODO-22 | Gain/RN Settings UI + FITS QA banner | ui_settings.py, database.py, ui_quality_dashboard.py |
| Match rate Gaia→DAO | Completeness metrika v pipeline + MASTERSTAR QA | pipeline.py, ui_masterstar_qa.py |
| Comp BP-RP fix | Outlier cap [0.1–3.5] + B-V fallback (R CVn fix) | photometry_core.py |
| VYVAR UI preklad | 278+ stringov → anglicky (6+2 súborov) | ui_*.py, catalog_crossmatch.py |
| Auto-export variability_candidates.csv | Odblokuje TESS auto-run | ui_variability.py |
| Suggest from MASTERSTAR | Auto-suggest DAO/aperture parametre | ui_dao_stars.py, ui_settings.py |
| float64 catalog_id fix | Precision loss → XY fallback → false candidates | gaia_catalog_id.py, photometry_core.py, pipeline.py |
| catalog_only Phase 2A skip | 64 targets preskočené v LC loop | photometry_core.py |
| Fyzikálny audit Fix 1 | sky term: sky_pp/g×area (CCD equation) | photometry_core.py |
| Fyzikálny audit Fix 2 | ZP MAD sigma-clip per frame (DAOPHOT štandard) | photometry_core.py |
| UI counters | Phase 2A log + Hockey Stick metric konzistentné | photometry_core.py, ui_variability.py |
| TODO-28 SNR units | `snr = (enclosed/g) / noise` v SNR aperture table | photometry_core.py |
| Cross-validácia Lenovo | photutils + SExtractor + IRAF (BO CVn frame) | scripts/validate_photometry_crossval.py, install_vyvar_crossval_lenovo.sh |
| FWHM VY_FWHM_GAUSS | Per-frame apertúry: GAUSS pred VY_FWHM×0.667 | pipeline.py |
| TODO-26 | Comp P90 noise floor pre variability envelope (248→47 kandidátov) | variability_detector.py, config.py, config.json |

#### Otvorené TODO (backlog)
- TODO-LC-QUALITY: `lc_quality_flag`, saturated/noisy export policy
- TODO-LC-TREND: differential extinction + ALG audit
- TODO-GEO: observer position audit (BJD, airmass, HJD)
- TODO-DEV-PROCESS: `scripts/regression_test.py` (baseline: draft_342)
- TODO-MASTERSTAR-QA validation: cyan FORCED_APERTURE overlay
- TODO-13 validation: Gaia→DAO completeness ~3.5% v QA
- TODO-32: IRAF EPADU fix v cross-val scripte (gain 3.17)
- TODO-44: ✅ CLOSED — Role-aware aperture (SIPS-style SNR + comp_factor)
- TODO-45: RGB kamera — de-Bayer → G kanál fotometria (IMX533 RGGB)
- TODO-25: Gaia→DAO zdroj UI vs pipeline zjednotiť
- TODO-23: Adaptive matching radius z pixel scale
- TODO-20: Mean stack MASTERSTAR (LOW — WCS/FWHM, not LC SNR)
- TODO-9: Lenovo — crossval env hotové; plná VYVAR inštalácia ešte otvorená

#### Ďalší krok
- Gold standard: TODO-GS3–GS7 (README, CITATIONS.bib, CHANGELOG, AAVSO, paper)

### TODO-PERF-7: UI LC index lookup — COMPLETED ✅
- **Main LC path** already efficient (1× `_cached_read_csv` per star via `lightcurve_{catalog_id}.csv`)
- **`_load_summary`:** `@st.cache_data(ttl=300)`, cleared after `run_full_photometry_pipeline`
- **Multi-filter overlay:** uses `_cached_read_csv` instead of raw `pd.read_csv`
- **`ui_suspected_lightcurves`:** `csv_cache` param + PERF-7 warning (module inactive; ready for ProcFrameStore)

### TODO-PERF-6: Variability cache / UI ProcFrameStore — COMPLETED ✅
- **Pipeline:** 0 variability disk reads when `csv_cache` + Phase 2A path (PERF-5 `ProcFrameStore`)
- **UI:** `ui_variability._cached_load_matrix` passes `ProcFrameStore` from `st.session_state` when set after RUN VYVAR
- **`run_full_photometry_pipeline`:** returns `proc_frame_store`; Phase 1 injects store into `st.session_state`
- **`load_field_flux_matrix`:** `[PERF-6]` INFO log for cache type (ProcFrameStore / dict / None)
- **Deferred:** `_flux_matrix_from_pivot` true pivot value reuse (CPU-only optimization)

### TODO-PERF-8: Shared flux matrix (Phase 2A) — COMPLETED ✅
- **`run_phase2a`:** union of all target + comp Gaia IDs (`_all_lc_ids_list`, ~384 stars) → one `read_flux_from_csv` pass per frame (139×) before target loop
- **Per target:** slice `_flux_matrix` by `all_ids` + existing edge/catalog-only post-process per frame (no repeated flux extraction)
- **Fallback:** original per-target × per-frame `read_flux_from_csv` loop if matrix build fails
- **Validated:** `simulate_night_run` draft_318 — 384 IDs, 53376 rows, BO CVn / FW CVn `lc_rms` within 0.001 mag of draft_317; photometry step ~576 s vs ~619 s (draft_317 class)

### TODO-PERF-10: Single-pass preprocess+QC — COMPLETED ✅
- **`dao_qc_in_calibrate`** (default `true` in `config.json`): `_quality_inspection_dao_metrics_array` runs inside `_calibrate_one_light_disk` on the in-memory calibrated array (no second raw read)
- **`apply_perf10_dao_qc_to_obs_files`:** writes FWHM / SKY_LEVEL / STAR_COUNT / roundness / pointing to `OBS_FILES`, auto-reject (median FWHM ×1.5), drift sync — same as former step 5
- **`night_run`:** step 5 (`run_draft_ram_calibration_qc_to_obs_files`) skipped when calibration-time QC succeeded; falls back to RAM QC if no metrics (e.g. passthrough-only)
- **Savings:** ~278 redundant raw array loads eliminated (139 frames × 2 passes); ~40–50 s per full night run (draft_316 class)

### TODO-PERF-5: ProcFrameStore — COMPLETED ✅ (Option B)
- **`proc_frame_store.py`:** unified store, single disk read per `proc_*.csv` frame, dict-compat
  interface (`get` / `items` / `values`), legacy fallback when `proc_frame_store=None`
- **Phase 1 + 2A** share one `ProcFrameStore` built in `run_phase0_and_phase1`, passed to `run_phase2a`
- **Tests:** `tests/test_proc_frame_store.py`
- **Note:** full-field SysRem ready (single union column load per frame)
- **`night_run.py`:** headless pipeline runner extracted from `app._run_vyvar_full_pipeline` (no Streamlit); foundation for TODO-11 auto-trigger
- **`simulate_night_run.py`:** CLI e2e simulation for `D:\BO_CVn` (defaults: equipment ID **1** QHY294MM, telescope ID **1** Carl-Zeiss 200mm)
- **`app.py` refactor** to call `night_run.run_night_pipeline` — **deferred** (UI still uses inline `_run_vyvar_full_pipeline`)

### E2E simulate_night_run bugs — FIXED ✅ (19.5.2026)
| Bug | Symptom | Fix |
|-----|---------|-----|
| ProcFrameStore subscript | `[VARIABILITY] … not subscriptable` | `variability_detector._lookup_cached_frame_df` uses `.get()`; `ProcFrameStore.__getitem__` safety net |
| FIELD JUMP DB | `WindowsPath` has no `fetch_draft_light_rows_for_quality` | `pipeline.py` ~4769: `VyvarDatabase(dbp2)` passed to `detect_field_jumps` |
| PDF `_norm_cid` | `name '_norm_cid' is not defined` | Module-level `_norm_cid()` + `staticmethod` on `_PhotometryReportBuilder`; `self.comp_df` in cover rows |
| PDF `_var_results` | `cannot access local variable '_var_results'` | **FIXED** ✅ — `self._var_results` in `__init__` (140, 210–213) and `_variability_cover_metrics` (690); PDF generates without this error |
| PDF `TITLE_H` | Report builder crash / missing title constant | Fixed in `photometry_report.py` |
| `aperture_px` key | KeyError / inconsistent summary column | Unified key in LC/summary export |
| `contamination_map` | `NameError` on every Phase 1 target (draft_320) | `contamination_map = {}` init in `_compute_comp_contamination_map` |
| PERF-4 log | Marker missing in run log | `comp_pool_rms.py`: `logging.debug` → `logging.info` for `[PERF-4]` |

### E2E validation — draft_321 ✅ (19.5.2026)
- **Command:** `python simulate_night_run.py --source D:\BO_CVn --eq 1 --tel 1`
- **Exit:** 0 | **84** light curves | **139** frames
- **photometry_NoFilter_60_2:** 524.9 s (cieľ &lt;580 s)
- **Phase 1 comp selection:** 368.8 s (vs ~530 s draft_320)
- **BO CVn `lc_rms`:** 0.1515 (ref 0.151502) | **FW CVn:** 0.0153 (ref 0.015296)
- **SysRem:** 82 stars × 139 frames × 3 iter → **5.7%** median RMS improvement
- **PERF-4B paths:** 77 vectorized (N≥50), 6 iterrows (N&lt;50)
- **PDF:** 98 pages, all 84 stars

### TODO-PERF-4: comp_pool_rms vectorized flux — COMPLETED ✅ (Option A)
- **`comp_pool_rms.py`:** iterrows over `N_frames × K_stars` → per-frame vectorized ops + `groupby` over stars only
- **CQ-3 (19.5.2026):** `comp_selection_per_target.py` — 12 helpers; orchestrator v `photometry_core.py`; lazy import v tele funkcie
- **Timing regression fixed (draft_321):** Phase 1 skips `catalog_only` comp selection after bp_rp enrich; `_accumulate_per_frame_comp_metrics` uses iterrows for N&lt;50 candidates, groupby for N≥50

### TODO-PERF-3: Batch Gaia comp lookup — COMPLETED ✅ (19.5.2026)
- **Prefetch:** `_comp_gaia_prefetch` v `run_phase0_and_phase1()` pred Phase 1 loop (IDs z `ms_df` + global comp pool)
- **Infra:** `_batch_enrich_targets_bp_rp_from_gaia_db` + `query_local_gaia_by_source_ids` (chunk 500)
- **Wire:** `gaia_prefetch` → `select_comparison_stars_per_target` → `_enrich_comp_bv` (per-star `source_id=?` fallback zachovaný)
- **Log:** `[PERF-3] Comp Gaia prefetch` + `Selected comp stars covered by prefetch`
- **Deferred:** `_enrich_active_targets_b_v_bp` batch (Step 5) — nižší dopad

### TODO-35: SysRem MVP — COMPLETED ✅ (19.5.2026)
- **Implementácia:** Tamuz, Mazeh & Zucker (2005), MNRAS 356, 1466
- **Funkcia:** `run_sysrem_field()` v `photometry_core.py`; volané z `run_full_photometry_pipeline()` po Fáze 2A
- **Rozsah MVP:** matica exportovaných cieľov (`lightcurve_*.csv`, typicky ~82 × N snímok)
- **Nový stĺpec:** `delta_mag_sysrem` (existujúce stĺpce sa nemenia)
- **Config:** `sysrem_enabled` (default `false`), `sysrem_n_iter` (default `3`)
- **Full field** (všetky hviezdy z `proc_*.csv`): odložené po TODO-PERF-5/PERF-6

---

## Dnes implementované (18.5.2026)

### Kompletný preklad UI do angličtiny — COMPLETED ✅ (18.5.2026)

**Rozsah:** všetky ui_*.py, app.py, export_reports.py, photometry_report.py,
variability_detector.py, importer.py

| Fáza | Súbory | Stringov |
|------|--------|----------|
| Task 1 (17.5.2026) | ui_variability, ui_masterstar_qa, ui_quality_dashboard, ui_settings, ui_photometry, ui_dao_stars | ~278 |
| Task 1 (18.5.2026) | app.py, ui_calibration, ui_database_explorer, variability_detector, importer | ~155 |
| Task 1b (18.5.2026) | ui_aperture_photometry, ui_photometry_results, ui_finalization, ui_select_stars, ui_calibration_library, ui_hrd, ui_photometry_quality, ui_suspected_lightcurves, ui_components | ~333 |

**Celkom:** ~766 user-visible stringov preložených
**Grep overenie:** 0 user-visible Slovak/Czech stringov zostáva
**Zachované:** log_event, LOGGER.*, CSV column keys, katalógy/žiadny záznam sentinels, Python identifiers

### Vedecké citácie v AAVSO a VAR.ASTRO exportoch — COMPLETED ✅ (18.5.2026)

**Súbor:** `export_reports.py` — `_vyvar_export_citation_lines()` (riadky 22–36)

Každý AAVSO a VAR.ASTRO TXT export obsahuje:
- Broeg, Fernandez & Neuhaeuser (2005) AN 326:134
- Howell (1989) PASP 101:616
- Stetson (1987) PASP 99:191
- Gaia Collaboration (2023) A&A 674, A1
- `#SOFTWARE=VYVAR/1.0 (Broeg 2005 differential photometry)` (AAVSO)

### Fáza 2A pipeline + gold standard (TODO-29, 30, GS1, GS2)

- **TODO-29:** ZP → CT → outlier detect → airmass fit; korekcia na všetky frames, fit na čistých
- **TODO-30:** Airmass detrend na `mag_calib_ct`; NoFilter bez zmeny správania
- **TODO-GS1:** Citačné komentáre (Howell 1989, Broeg 2005, Stetson 1987) v `photometry_core.py`
- **TODO-GS2:** `tests/test_photometry_core.py` — 11/11 pytest passed

### Code quality audit + fixes (18.5.2026)

Full audit of 94 .py files. Six fix passes, all with pytest 11/11 green.

**Fix 1 — Silent exceptions:** 38 `except: pass` blocks → `LOGGER.warning/debug`
across `photometry_core.py`, `pipeline.py`, `ui_variability.py`, `app.py`,
`comp_pool_rms.py`, `config.py`, `database.py`, `catalog_crossmatch.py`,
`crossmatch_runner.py`, `astrometry_optimizer.py`

**Fix 2 — Gaia ID normalization:** 8 duplicate `_norm_cid`/`_cid_key` functions
replaced with canonical `normalize_gaia_source_id()` from `gaia_catalog_id.py`.
Deleted unused `catalog_id_series_for_proc_csv_export()`.

**Fix 3 — Dead code marked:** 5 orphaned UI modules (inactive NOTE),
13 legacy pipeline/importer helpers (DEPRECATED comment), 1 VSX duplicate.

**Fix 4 — Draft path resolution:** `resolve_draft_dir()` + `resolve_draft_dir_path()`
added to `utils.py`. Replaced 6+ duplicate resolution chains in
`ui_aperture_photometry`, `ui_variability`, `ui_masterstar_qa`, `ui_quality_dashboard`.

**Fix 5 — CSV dtype:** `VYVAR_CSV_DTYPE` + `read_vyvar_csv()` added to
`gaia_catalog_id.py`. High-risk join locations updated in `ui_variability`,
`ui_masterstar_qa`, `hrd_analysis`, `variability_detector`.

**Fix 6 — Long function split:** `generate_photometry_report()` 3384 → 63 lines
via `_PhotometryReportBuilder` class (13 section methods). `run_phase2a()`
partial extraction. `render_live_view()` deferred — TODO markers added.

**Remaining known tech debt:**
- pandas `FutureWarning` — `variability_detector.py` `.fillna()` downcasting (fix v ďalšej session)
- `run_phase2a()` full extraction (1235 lines, large closure surface)
- `render_live_view()` split (1390 lines, heavy session state)
- ~~`select_comparison_stars_per_target()` split~~ ✅ CQ-3 (19.5.2026)
- `solve_wcs_with_local_gaia()` split (1843 lines)

### CSV schema cleanup — Phase 1 + Phase 2 (18.5.2026)

Full audit of 7 generated CSV files (94 .py files scanned).
26 obsolete columns removed, 4 added/fixed. pytest 11/11 green.

**Phase 1 — High-confidence removals:**

proc_*.csv: `flux_raw`, `fwhm_gaussian_px`, `r_small_px`, `r_large_px`,
  `sky_annulus_r_in_px`, `saturated_from_peak`, `saturated_plateau`,
  `snr10_ok`, `gaia_nss`, `gaia_qso`, `gaia_gal`, `catalog_known_variable`
photometry_summary.csv: `n_outliers`, `am_slope_pre`, `am_slope_post`, `am_piecewise`
comparison_stars_per_target.csv: `color_rms_score`
Added: `PROC_CSV_READ_COLS` canonical list to `gaia_catalog_id.py`

**Phase 2 — Medium-confidence removals:**

proc_*.csv: `is_discovery_candidate`, `is_saturated_flagged`, `is_noisy`, `match_sep_arcsec`
photometry_summary.csv: `skip_photometry`
active_targets.csv: `snr50_ok`, `zone`, `is_usable`, `match_dist_arcsec`
masterstars_full_match.csv: `snr10_ok`, `saturate_limit_per_frame_adu`
Kept: `aperture_px` in summary (active consumers in report/export/UI)

**Bug fix:** variability_candidates.csv — added `vsx_known_variable`, `vsx_match`,
  `gaia_dr3_variable_catalog` to export (TESS loader expected them but they were missing)

**Impact:** Leaner proc CSV per frame → faster I/O, smaller disk footprint.
New schemas active after next pipeline run (existing CSVs on disk unchanged).

### "Gold Standard" status — roadmap a aktuálny stav

#### Čo robí IRAF "gold standard":
1. **Peer-reviewed algoritmy** — každá funkcia má citáciu v literatúre
2. **Reprodukovateľnosť** — rovnaký vstup = rovnaký výstup vždy
3. **Transparentnosť** — užívateľ vie presne čo sa počíta
4. **Validácia** — porovnané s inými nástrojmi a pozorovaním
5. **Dokumentácia** — každý parameter má fyzikálny význam

#### Kde VYVAR už je na úrovni IRAF:
- ✅ Broeg (2005) — citovaný v PDF reporte + inline v kóde (GS1, 18.5.2026)
- ✅ Howell (1989) CCD equation — fixnutá (17.5.2026) + citačné komentáre (GS1)
- ✅ Cross-validácia voči IRAF — urobená (2.2% zhoda!)
- ✅ Fyzikálne správna sky subtrakcia
- ✅ Unit testy fyzikálnej korektnosti (GS2, 11/11 pytest)

#### Výsledky trojitej cross-validácie (17.5.2026) — publikačne hodnotné:
| Nástroj | Zhoda | Poznámka |
|---------|-------|----------|
| photutils 3.0 (r=3px) | 2.0% scatter | Optimálna apertura = 0.97×FWHM |
| SExtractor 2.28 | 6% offset | Growth curve efekt (PSF wings) |
| IRAF apphot (r=3px) | 2.2% scatter | Po ZP korekcii |

Záver: tento výsledok ide priamo do sekcie "Validation" v budúcom paperi.

#### Plán pre "gold standard" status:
Krok 1 (1-2 mesiace): Unit testy + citačný reťazec — **hotové** (GS1, GS2, 18.5.2026)
Krok 2 (2-3 mesiace): AAVSO validácia na 3-5 known variables
Krok 3 (3-6 mesiacov): Paper draft (PASP alebo AN)
Krok 4: Submit + peer review

Navrhovaný názov papera: *"VYVAR: An Automated Differential Photometry Pipeline for Amateur Variable Star Observers"*

#### Nové TODO — gold standard:
- TODO-GS3: `README.md` s inštaláciou na GitHub — MEDIUM
- TODO-GS4: `CITATIONS.bib` — všetky použité algoritmy — MEDIUM
- TODO-GS5: `CHANGELOG.md` — MEDIUM
- TODO-GS6b: AAVSO Extended Format validation — HIGH (blocker for GS6)
- TODO-GS6: AAVSO submission + cross-observer validácia — FUTURE
- TODO-GS7: Paper draft (PASP / AN) — FUTURE

### LC cross-validácia — photutils differential (18.5.2026)

**Setup:** draft_310, BO CVn pole, 84 targets, 139 frames, Lenovo T460
**Nástroj:** `validate_lc_crossval.py` v4 — dao_flux z proc CSV +
rovnaké comp hviezdy ako VYVAR (`comparison_stars_per_target.csv`) +
3σ MAD sigma-clip

**Výsledky (N=67 hviezd s platným dao_flux):**

| Mag bin | N | Median Δ | Std Δ | Verdikt |
|---------|---|----------|-------|---------|
| 8–10    | 4 | -0.0008  | 0.005 | ✅ výborné |
| 10–11   | 2 | -0.009   | 0.007 | ✅ výborné |
| 11–12   | 4 | +0.001   | 0.007 | ✅ výborné |
| 12–13   |24 | +0.0001  | 0.011 | ✅ výborné |
| 13–14   |30 | -0.094   | 0.474 | ⚠️ rozptyl (variabilné + growth curve) |

**Záver:** VYVAR diferenciálna LC fotometria je konzistentná s photutils
na úrovni <0.001 mag (mag 8–13). Rozptyl mag 13–14 = reálne variabilné
hviezdy + growth curve efekt (TODO-31).

**17 hviezd preskočených:** NOT IN proc CSV → XY fallback vo VYVAR →
lc_rms nespoľahlivé → vyriešiť cez TODO-27 (re-export po float64 fix)

**Skript:** `scripts/validate_lc_crossval.py` (Lenovo)

### PyRAF LC validácia — ATTEMPTED, NOT FEASIBLE (18.5.2026)

**Pokus:** LC cross-validácia VYVAR vs IRAF apphot na draft_310 (139 frames, 447 hviezd)

**Root cause:** Community IRAF V2.17.1 / PyRAF 2.2.4 číta nesprávne gross flux
z float32 big-endian FITS (`>f4`):
- photutils gross = 194,690 ADU (správne)
- IRAF gross     = 109,652 ADU (faktor ~1.77× nižší)
- Sky a area sú identické → problém je v pixel readout nie sky subtrakcii
- Konverzia >f4 → int32 nepomohla (rovnaký výsledok)
- Toto je známy bug Community IRAF s moderným Linux/Python

**Záver:** PyRAF LC validácia nie je realizovateľná s proc FITS súbormi z VYVAR.

**Dostupná IRAF validácia:** Single-frame test na MASTERSTAR.fits (17.5.2026)
dáva 2.2% zhodu — MASTERSTAR.fits má iný formát (uint16) ktorý IRAF číta správne.

**Finálna validácia LC (platná):**
- photutils differential LC: mag 8-13 zhoda <0.001 mag (N=67 hviezd) ✅
- IRAF single-frame flux: 2.2% zhoda na MASTERSTAR.fits ✅
- Kombinácia týchto dvoch výsledkov = dostatočná validácia pre publikáciu

### Muniwin (c-munipack) LC validácia — COMPLETED ✅ (18.5.2026)

**Setup:** draft_310, BO CVn pole, detrended_aligned FITS, c-munipack 2.1.36
**Workflow:** `muniphot` → `munimatch` → `munilist --diff-mag`
**Parametre:** gain=3.17, RN=7.6, FWHM=2.3976, apertura=3.318px,
  skyinner=11.39px, skyouter=21.58px (zhodné s VYVAR nastaveniami)

**Výsledky (rovnaké comp hviezdy ako VYVAR):**

| Hviezda | Mag | VYVAR RMS | Muniwin RMS | Ratio | N |
|---------|-----|-----------|-------------|-------|---|
| Gaia 1502044 | 9.51 | 0.00958 | 0.00855 | 0.893 | 96 |
| Gaia 1497070 | 12.54 | 0.03108 | 0.03275 | 1.054 | 135 |
| BO CVn (var) | 9.72 | 0.14851 | 0.17040 | 1.147 | 48 |

**Záver:**
- Mag 9.5–12.5: Muniwin a VYVAR sa zhodujú na ±5–15% ✅
- BO CVn rozdiel (15%) = variabilná hviezda + rôzny počet valid frames (48 vs 139)
- Muniwin používa jednoduchý weighted mean; VYVAR používa Broeg (2005)
  → malý rozdiel v RMS je fyzikálne očakávaný
- FITS float32 big-endian čítané správne (na rozdiel od PyRAF/IRAF)

**Finálna validačná tabuľka VYVAR (draft_310, BO CVn pole):**

| Nástroj | Metóda | Zhoda | N hviezd |
|---------|--------|-------|----------|
| photutils 3.0 | differential LC (dao_flux) | <0.001 mag (mag 8–13) | 67 |
| Muniwin 2.1.36 | differential LC (rovnaké comp) | ±5–15% RMS | 3 |
| IRAF apphot | single-frame flux (17.5.2026) | 2.2% scatter | 48 |
| SExtractor 2.28 | single-frame flux (17.5.2026) | 6% offset | 273 |

**Nástroje nainštalované na Lenovo (Linux):**
- c-munipack 2.1.36 ✅ (sudo apt install c-munipack)

---

## Cieľový workflow (kávový test ☕)

Koniec pozorovania (4:00) → ⚡ RUN VYVAR → Fáza 0+1+2A
→ Variabilita detekcia (sigma=2.3) → Auto crossmatch
→ Auto TESS (len bez VAR match):
   · `period_reliability`: reliable / uncertain / noise
   · blend_check PNG: TESS TPF vs Gaia obloha
→ Summary Measure Report PDF (vrátane TESS sekcie)
→ 8:00 používateľ číta report ☕

Chýbajúce napojenie: Bootes ePSF validation (TODO-8-BOO), Auto-trigger (TODO-11)

---

## Pracovný workflow (Claude ↔ Cursor)

### Roly
- **Claude** = analytik / dizajnér — diagnostikuje problémy, navrhuje riešenia, píše inštrukcie
- **Cursor** = staviteľ — číta kód, implementuje fixy podľa Claudových inštrukcií
- **Používateľ** = schvaľuje — rozhoduje medzi diagnostikou a fixom

### Komunikačný protokol
1. Používateľ popíše problém Claudovi (príp. priloží PNG / log)
2. Claude napíše presnú inštrukciu pre Cursor (diagnostika ALEBO fix)
3. Používateľ skopíruje inštrukciu do Cursoru
4. Cursor odpovie → používateľ skopíruje odpoveď späť ku Claudovi
5. Claude vyhodnotí a buď schváli, alebo napíše ďalšiu inštrukciu

### Pravidlá
- Cursor **nemení kód** bez explicitného pokynu od Clauda
- Každý vyriešený problém sa zapíše do `VYVAR_STATE.md` ako COMPLETED ✅
- Otvorené problémy zostávajú v sekcii `## Otvorené otázky` až do overenia

### Language rules
- Claude → Cursor instructions: **English**
- Cursor → Claude responses: **English**
- Claude ↔ User: Czech / Slovak

---

## Otvorené otázky

### TESS blend check — overenie
- Spustiť TESS re-run pre `1451497396118755584`
- Overiť `sector_23_blend_check.png`

### TESS periódy draft_294
- `1451497396118755584`: uncertain (0.0201 d vs 0.1029 d) — vizuálna verifikácia pending
- `1451924483370649472`: P=0.4375 d — typ TBD

### HRD — po nočnom behu DB
- Overiť že `vyvar_gaia_dr3_v3.db` má Teff/logg vyplnené
- Aktualizovať `gaia_db_path` v `config.json`

---

## 31.5 — ePSF FINE-SCALE VALIDATION + ELONGATION RETRACTION (Telescope Live / Palomar 7)
Tested aperture vs PSF/ePSF on fine-scale well-sampled dense data: Telescope Live
Palomar 7 / IC 1276 GC, Planewave CDK24 + QHY600M, El Sauce Chile, 0.389"/px,
FWHM ~6.2 px (draft 364).

COMPLETED:
- Set #4 registered: CAMERA "QHY 600M" (id3; 9576x6388, 3.76um; GAIN/RN/SAT APPROX),
  TELESCOPE "Planewave CDK24" (id4; 3962mm/610mm), LOCATION "El Sauce (Obstech),
  Rio Hurtado, Chile" (id3; -30.4703,-70.7647,1570m). Pre-calibrated Telescope Live
  import validated (vendor _cal.fits, no calibration; blind solve from OBJCTRA/DEC;
  WCS-authoritative resolver -> 0.389"/px).
- APERTURE-vs-PSF IS SAMPLING- AND SNR-DEPENDENT (corrected): OAT (362, 9.77"/px,
  undersampled) PSF 1.3-1.7x WORSE everywhere. CDK24 (364, 0.39"/px), large-N forced
  photometry (667 isolated G12-16; full population 13k+): aperture WINS on bright
  (G12-16 psf/aper ~2.3; FREE-position run = FIXED = 2.33, so NOT a centroid artifact
  -- PSF is model-limited where photon noise is negligible), ratio converges through
  G16-18, PSF WINS only at the faint photon-limited end (G19-20 isolated 0.79).
  Crossover ~G18-19. The earlier small-N DAO-detected comparison (psf/aper ~0.85 on
  bright) was a selection/noise artifact of the detected subset and is SUPERSEDED.
  Practical implication: validates the ADAPTIVE-selector premise -- aperture as
  robust default, PSF reserved for the faint (and likely crowded) regime; method
  choice depends on plate scale AND per-star SNR/crowding, not a global flag.
- ePSF CANDIDATE-SOURCE FIX (enables ePSF on dense fields): (a) fixed indentation bug
  in _epsf_prepare_stars DB-join loop (star_rows.append was outside the for-loop);
  (b) conditional broad-pool augmentation when safe-comp∩CSV < epsf_min_stars, via
  COG-style isolation (_select_frame_stars_from_proc); sparse-field behavior
  unchanged. Dense GC (only 5 IS_SAFE_COMP rows) now builds with 72-98 candidates.
- DEEP GAIA CONE capability: astroquery Gaia DR3 -> field SQLite
  (vyvar_gaia_dr3_pal7_field.db, 34,376 rows to G~20) in gaia_dr3 schema, drop-in via
  gaia_db_path. Matched stars 717 -> 11,189 (faintest matched G~18).

RETRACTED:
- "Tracking smear" / asymmetric-PSF hypothesis (28.5-31.5) WITHDRAWN. The ~1.18-1.19
  per-frame elongation was a MEASUREMENT ARTIFACT of the fixed 9x9 Gaussian-fit window
  (too small vs FWHM, esp. at 6.2 px). Model-free second moments (PSF-scaled window)
  give axis ratio ~1.08-1.09 on BOTH 362 and 364 -> stars essentially round; no on-sky
  smear. NOTE: Gaussian-fit elongation metric biased high ~10pp; use moments where
  elongation drives decisions. qc_elong_limit=1.8 is loose -> no operational impact.

OPEN:
- Faint isolated: PSF wins (G19-20 forced, psf/aper 0.79) -- RESOLVED.
- Crowded-deblending PSF advantage UNCONFIRMED: forced PSF (fixed or free, single-
  source) does not cleanly win on crowded (G19-20 crowded ~1.04). A proper test needs
  free-position GROUPED fitting (gated grouper) -- deferred.
- Bright-end aperture advantage is large (~2.3x); ePSF could possibly be tightened
  (FWHM ratio 0.80, but consistent with the known DAO-FWHM overestimate, cf. 362
  ratio 0.636). Does not change the adaptive-selector conclusion.

### PSF VALIDATION — CLOSURE (gated components, 2026-06-01)

Standalone forced-photometry validation on draft 364 (CDK24, 0.39"/px, dense GC):

- **GROUPER (joint deblending): NEGATIVE for precision.** grouped/single ~0.94 (~6%),
  but grouped/aperture stays **1.05–1.22** on crowded (G17–20) — aperture still wins
  crowded; at G19–20 crowded grouped (1.05) is worse than single (0.98). At 0.39"/px
  blends are largely sub-resolution; joint fitting is a small refinement, not a crowded
  PSF win. **NICHE retained:** deblending protects **AMPLITUDE fidelity** for a target
  with a **VARIABLE close neighbor** (Han & Brandt 2023) — qualitative, not precision.
  Grouper wiring (`neighbor_catalog`) → **LOW priority**.

- **ADAPTIVE SELECTOR (`psf_adaptive`): VALIDATED = the production way to use PSF.**
  Aggregate median RMS aperture-always **0.111**, PSF-always **0.111**, **ADAPTIVE 0.100**
  (~10%, near-oracle within ~4%). Routes faint-isolated → PSF (G19–20 isolated 51%
  routed, **0.148 vs aperture 0.194** ≈ 24% there), keeps crowded/bright on aperture
  (3–13% PSF on crowded). Gain concentrated at faint-isolated end; ≈aperture elsewhere
  (free upside). NOT a strict aperture floor (1.3% slightly worse).

- **SPATIAL / GRIDDED ePSF:** builds cleanly (3×3, all cells ≥ min stars); benefit
  **UNTESTED, DEFERRED** to a field-variable-PSF rig (planned 8" f/2.72 Newton, corner
  coma). Expected neutral on well-corrected fine optics (364 field-elongation flat
  with radius).

**WIRING STATUS:** adaptive is wired into Phase-2A but **STARVED** — PSF computed for
only ~1.4% of rows (target subset) and `crowding_targets.csv` absent → routes 0 PSF.
To deliver the validated ~10% gain, two gaps must close: (1) PSF for the full LC star
set (targets+comps) when PSF enabled, (2) generate the blend map. [Wiring in progress.]

## PSF wiring — VERIFIED (adaptive selector, draft 364)

Wiring done (gated): PSF computed for the full LC star set when `psf_photometry_enabled`;
`crowding_targets.csv` generated when `psf_adaptive_enabled`. Runtime negligible.

**CORRECTIONS from verification:**

- "~1.4% PSF coverage" was an ALL-DETECTION denominator artifact. The LC star set (161
  on 364) was already ~80% covered; wiring → ~84%. LC stars were never starved.
- Adaptive routes 0 PSF on 364's LC ensemble CORRECTLY — that ensemble is BRIGHT
  (comps G~13–16, SNR≫15). The standalone ~10% gain was on a FAINT/isolated population
  (G17–20), NOT the bright comp ensemble. Gain is FAINT-TARGET-SPECIFIC (G≥17): on
  bright targets/comps adaptive == aperture (no harm/no gain); on faint targets it
  routes to PSF and gains (~10–24%). Earlier "~10% in production" corrected to
  faint-target-specific.
- **BUG:** rule 2 (resolvable-blend → PSF) can NEVER fire — `is_blended` is
  `nn_dist_fwhm ≤ 1.5` but rule 2 needs `≥ 2.0` (mutually exclusive). Plus grouper was
  NEGATIVE (deblending no precision gain at 0.39"/px) → blend route unjustified →
  **DROP rule 2**, keep rule 3 (faint isolated → PSF). Variable-neighbor amplitude
  niche handled case-by-case.

**OPEN:** demonstrate the faint-target gain end-to-end (G≥17 targets, re-export PSF,
confirm rule 3 fires and adaptive beats aperture).

## Adaptive selector — FINAL verified result (faint targets, draft 364)

- Rule 2 removed (dead + grouper-negative); rule 3 (faint + good PSF → PSF) is now the
  only PSF route. Rule 1 and aperture default unchanged.
- Faint-target verification (G≥17 isolated as targets, strict production
  `assess_psf_quality`): rule 3 DOES fire, but only on the faintest subset with SNR≤15
  AND good PSF quality (4/35 candidates). Realized gain MODEST: ensemble
  adaptive/aperture ~0.98 (~2%); PSF-routed targets raw RMS ratio ~0.93 (~7%); vs
  standalone faint-isolated ~0.79 (~21%).
- Gap vs standalone = the **QUALITY GATE**: standalone used a relaxed proxy (good if
  finite flux); production uses strict `assess_psf_quality`, blocking most ultra-faint
  (G~20) frames. Strictness is mostly protective but imperfect — one PSF-routed target got
  WORSE (bad PSF frames passing the gate on a few nights).

**FINAL CALIBRATED CONCLUSION:** adaptive selector validated and conservative. Bright
targets/comps == aperture (no change). Faint targets (G≥17): SMALL quality-gated
average gain (~2–7% realized, not the ~21% standalone ideal), with occasional per-target
noise. Recommended use: opt-in for faint deep-field programs; aperture remains the
published default.

**LOW-PRIORITY OPENS** (only if faint precision becomes critical): tune faint-end
`assess_psf_quality` acceptance; and/or re-export a FULL night (vs 10 frames) so more
G≥19 frames reach good quality and the realized gain is measured more robustly.

---

## Reporting PDF revision — R1+R2 done, R3 pending

- **R1 overflow robustness:** `drawString` → wrapping `Paragraph` + pagination + layout guard;
  0 overflow violations on 362 (page 2 Methods/citations paginates; long Gaia IDs wrap).
- **R2 completeness (self-contained night summary):** cover += observer/OBSCODE, plate scale,
  equipment (tel+cam), variability counts (wired `_variability_cover_metrics`); obs-summary
  KV += frames used/rejected, session BJD span, FWHM min/med/max; summary table +=
  n_points/MERR/median/quality; per-star += check-star KNAME/KMAG/scatter (P3.5 sidecar) +
  ground period. Additive; overflow=0.
- **R3 PENDING:** aperture-vs-PSF/adaptive overlay on per-target LC (primary report only;
  per-method PDFs stay single-method). Instruction written, not run.

---

## B-V legacy removal (APASS/Tycho) — audit + Stage 1 done; Stages 2–4 pending

- **Audit:** APASS/Tycho B-V reached ONLY via `lookup_bv_from_local_db` (last fallback: Gaia
  bp_rp → teff → APASS → Tycho → unknown), in `vyvar_vsx_local.db` (apass ~24M / tycho2
  ~1.1M rows). No production algorithm requires it; ~3% targets / ~7% masterstars on 362;
  color term + tiers already BP-RP-native.
- **Stage 1 DONE (reversible):** disconnected the APASS/Tycho fallback in 4 callers;
  `lookup_bv_from_local_db` left DEFINED but unused. The 5 APASS/Tycho targets on 362 fall
  back to unknown/NaN gracefully (bp_rp unchanged); selected comps 100% gaia_bprp.
- **Determinism VERIFIED:** comp loop fully deterministic (R1==R2, 100% overlap; mergesort,
  no RNG). The ~45% before-vs-after comp diff was run-context drift (active-set incl. a
  duplicate; full Phase 0+1 vs frozen; adaptive density overrides), NOT the disconnect.
- **PENDING:** scope A (APASS/Tycho only) vs A+B (also retire all Johnson B-V → pure BP-RP,
  recommended; kills the legacy |dB-V| dual-mode); then Stage 2 (delete dead APASS/Tycho
  code + UI/export AP/TY provenance), 3 (retire Johnson B-V + legacy mode; missing-bp_rp
  → mag-proxy), 4 (DROP apass_data/tycho2_data + update VSX/vsx_make.py + regenerate).

---

## Session 2026-06-03 — B-V A+B Stages 2–4 (Johnson B-V retired)

- **Scope:** A+B — pure Gaia BP-RP tiering; `bp_rp`-less targets → T4 / mag-proxy.
- **Stage 2** (`3480bd0`): removed `lookup_bv_from_local_db`; `VSX/vsx_make.py` VSX-only
  (no `apass_data` / `tycho2_data`). Byte-identity gate on `draft_000366`: **not run here**
  (no Archive draft on dev PC).
- **Stage 3** (`8945bbf`): removed `bp_rp_to_bv`, `teff_to_bv`, `bv_to_bprp_linear`,
  `_enrich_active_targets_b_v_bp` → `_enrich_active_targets_bp_rp`; comp selection BP-RP-only;
  config/UI/export/report columns (`b_v`, `bv_source`, `phase01_use_bprp_primary`, `*_bv_*`
  limits) retired from production paths. Tier-change report on `draft_000366`: **not run here**.
- **Stage 4** (this commit): docs parity (DECISIONS/ROADMAP/JOURNAL/PARAMS); DB regen command
  documented below. **Tests:** 111 passed / 6 skipped.
- **DB regen (operator, catalog machine):** from repo root, with Vizier/network:
  `python VSX/vsx_make.py --db path/to/vyvar_vsx_local.db` — builds VSX table only (Gaia
  photometry stays in separate `gaia_dr3` DB). Confirm `apass_data` / `tycho2_data` absent:
  `sqlite3 vyvar_vsx_local.db ".tables"`.
- **APCORR-COLOR watch-point:** `fit_color_term_c1` / `apply_color_term` already use
  `comp_bp_rp` dict — no production colour-term path reads `b_v` after Stage 3.

---

## Session 2026-05-31 — comp QA (Sokolovsky) + 3-axis trust flag (draft_000365)

### Report label fix (commit `8e8cf29`)

Per-star comp table column was labeled **"p2p RMS"** but printed **comp_rms** (Phase-1 flux
scatter). Relabeled to **comp_rms**; added real **rms_p2p** from `comp_quality_{target}.json`
(matches exclusion footnote). Re-render draft_000365: **0 overflow**, **160 pages**. Report
bytes intentionally change; no numeric pipeline output changed.

### comp_qa (standalone `comp_qa.py` + `scripts/comp_qa_flagged_lcs.py`; `xval_out/` not in git)

Grounded in Broeg 2005 (already cited) + Sokolovsky et al. 2017 (MNRAS 464, 274). LOO
differential mags (zero-median), time-ordered:

| Index | Definition | Role |
|-------|------------|------|
| σ_IQR | (P75−P25)/1.349 | Amplitude (robust) |
| 1/η | s²/δ² von Neumann | Slow drift (white ≈ 0.5) |
| spike | std(m)/σ_IQR | Dropout frames |

**Flag:** σ_IQR above magnitude-dependent locus (0.5-mag bins, median+4·MAD spread, rebuilt
each iterative LOO pass) **OR** 1/η peer-outlier (median+4·MAD of target pool) and >1.0 **OR**
spike>3; iterative drop-worst, min 3 comps.

**Evolution:** flat floor+peer v1 (47→67, over-flagged faint comps) → Sokolovsky **27**
flagged (25 amplitude, 1 spike, 1 amp+invNV). Faint comps …190720/…050880 sit on locus (clean);
dropout caught (NSV 20420 …439552, spike≈8.48); tight-pool FPs clean (V0348 Dra, NSVS
J1618591+485752). **n_clean buckets:** ≥5: **133** / 3–4: **8** / <3: **2**.

### Trust flag (`trust_flag.py` → `xval_out/trust_per_target.csv`, 143 rows)

3-axis per-target gate for non-experts. Inputs: comp-health (`comp_qa_targets.csv` n_clean),
sep cross-val (`xval_results.csv`: `confirmed` / `vyvar_ok_indep_failed` / `review` /
`no_independent` / `no_vyvar_rms`), VYVAR check-star scatter + `lc_quality_flag`.

**Warnings (W):** sep≠confirmed; n_clean 3–4; check_star_scatter≥0.02 mag;
`lc_quality`∉{good,noisy} (e.g. **saturated** — genuine data-quality demotion).

**Levels:** RED if n_clean<3 or W≥2; YELLOW if W==1; GREEN if W==0 (confirmed + n_clean≥5 +
check<0.02 + no hard quality flag).

**Key decision:** `lc_quality="noisy"` is variability-driven (informational only, not a
warning). Counting it as a warning wrongly demoted **48** real variables (e.g. V0349 Dra:
lc_rms 0.155, check-star 0.0054 → GREEN after fix). Adding `lc_quality∉{good,noisy}` catches
**saturated** (HD 143776 YELLOW→RED).

**Final counts (draft_000365, xval_results present):** GREEN **81** / YELLOW **52** / RED **10**.

### Cross-validation

**CLOSED** for aperture path — triple-validated photutils+sep+dao on draft_000365 (`xval_run.py`
harness). PSF cross-val **deferred** to a PSF-heavy/faint draft with per-frame ePSF.

### Parked next steps

(a) Wire trust flag into AAVSO/VarAstro export + PDF (GREEN/YELLOW/RED + reason at submit);
(b) PSF cross-val on a faint draft; (c) optional gate tuning — grade check-star or let strong
comp-health (n_clean≥6) absorb one mild warning (borderline faint REDs e.g. 1399187099635410432).

---

## Deferred to next session

- Formal cross-validation: DONE for draft_000365 — V842 spot-check + whole night 143 targets (sep/dao ~3% vs lc_rms); IRAF closed; `xval_run.py` harness validated. Muniwin still optional.
- Reporting R3 (aperture-vs-PSF overlay).
- ~~B-V removal Stages 2–4~~ — done 2026-06-03 (see session above).

---

## Future phases / backlog

### FUTURE BIG PHASE (not started) — Comet photometry mode

**STATUS:** analysis only (2026-06-01). Do NOT start until the variable-star pipeline is
finished. Comets are the NEXT MAJOR PHASE after stellar; current priority stays
variable-star VYVAR.

**VERDICT (feasibility analysis):** feasible; architecture sound; HIGH front-end reuse; but a
SIGNIFICANT new module (weeks, not days), and mature validated tools already exist — KOPR
(Czech, kopr.astro.cz; TYC2/APASS comps; click-to-measure coma aperture), Tycho-Tracker v12
(all-in-one: calibration/platesolve/align/photometry/report, star-removal filter, growth
curve, ICQ output), Comphot (BAA, two-image method). VYVAR's value = workflow integration
(own rigs, one tool) + Gaia zeropoint (vs APASS/Tycho), NOT novel science.

**WHY THE PROPOSED ARCHITECTURE IS CORRECT:** it matches the professional standard. Comphot
uses exactly two stacks — a STAR-stack ("fixed") for the photometric zeropoint and a
COMET-stack ("offset", shift-and-add on the comet's motion) for the coma flux, via a
median-annulus aperture that ignores stars. Comets are EXTENDED (nucleus+coma+tail) and
MOVING (different reference stars each night); the useful CCD measure is the total coma
magnitude m1, reported in ICQ format to COBS.

**REUSE (front-end, shared):** calibration; plate solving; star-align stacking (= the "fixed"
star-stack); Gaia comp ensemble + zeropoint (+ Gaia→V Riello transform for V-equivalent
comps); aperture machinery; the overhauled export/reporting layer.

**NEW (comet module, after the calibrate/platesolve/star-ZP front-end):**

- **C1.** Comet ephemeris / apparent motion (JPL Horizons or MPC) → comet-rate stacking
  (track-and-stack: shift-and-add along the motion vector; coma sharp, stars trailed;
  median + masking to suppress trailed stars).
- **C2.** Zeropoint transfer star-stack → comet-stack (comet stack has no usable comps).
- **C3.** Extended-source coma photometry: growth curve (flux vs radius), detectable coma
  diameter, total magnitude m1; optionally Afrho (dust parameter). THIS is the crux
  (low surface brightness, flat-sky sensitivity, trailed-star contamination).
- **C4.** ICQ-format export + COBS; comet night-summary report (reuse reporting layer).

**TOGGLE REALITY:** not a small switch — a PARALLEL photometry mode sharing the front-end
(calibrate → platesolve → star-stack → Gaia ZP), then forking into comet stacking +
extended photometry + ICQ export. Front-end shared, back-end forked. User wants a
stellar/comet mode selector.

**B-V CONNECTION:** KOPR/Comphot use APASS/Tycho V comps — the catalogs being removed from
VYVAR's stellar pipeline. VYVAR's Gaia→V (Riello) gives V-equivalent comps from Gaia, so
the B-V/APASS/Tycho removal does NOT block comet work; Gaia ZP is a cleaner, more modern
base.

**FIRST STEP IF PURSUED:** read-only audit of VYVAR stacking / aperture / zeropoint internals to
confirm the reuse points, before any C1–C4 design.

---

## Session 2026-06-03 — Remove in-pipeline `sep_xval`; trust gate v2

**Removed (production):**
- `sep_xval_core.py` (deleted); harness helpers moved to `xval_harness_core.py` for `xval_run.py`.
- `sep_xval_*` config keys and the post-comp_qa stage call in `photometry_core.py`.
- Runtime SEP citation emission in `citations.py` (`barbary2016` / `bertin1996` no longer in
  export/PDF DATA-QUALITY GATE; `.bib` entries kept for harness).

**Trust gate v2 semantics** (`trust_flag_core.py`):
- Axes: `n_clean` (comp_qa), check-star scatter, `lc_quality_flag` only — no `sep_confidence`.
- RED: `n_clean < min_comps` OR any hard (bad lc_quality, check ≥ 0.05).
- YELLOW: any soft (thin comps, check 0.02–0.05).
- GREEN: `n_clean ≥ strong`, check < 0.02, lc_quality ∈ {good, noisy}, no warnings.
- Targets previously YELLOW/RED only for `sep≠confirmed` promote (expected distribution shift
  vs 69/59/15).

**Validation:**
- `pytest tests/`: 111 passed / 0 failed / 6 skipped.
- Photometry byte-identity: stage was read-only — Phase-2A LC unchanged by design.
- Trust re-run on draft_000366: **not executed on this dev PC** (no draft tree in workspace);
  re-run `trust_flag.py --photometry-dir …` where comp_qa columns exist and record counts in
  STATE.

---

## Session 2026-06-04 — Blind index series (mag14 density tiers + solve-rate)

**Problem:** Single prem-density mag14 index failed on dense Newton fields (draft_380); wide-rig
regression untested when `draft_361` MASTERSTAR was absent.

**Solution:**
- Density-matched **fine** tier (reuse `gaia_triangles_mag14.pkl` → `gaia_triangles_fine.pkl`).
- **Wide** tier build: mag14, cell 2°, 16 stars/cell → 224k stars, 1.43M triangles, log_L3 med 3.28 dex.
- Manifest `GAIA_DR3/blind_index_series.json`; orchestrator `vyvar_blind_series.py` integrated in
  `vyvar_platesolver.py` (`auto` / `series_all` / `single`).
- Solve-rate battery `validation/blind_solve_battery.json` + `scripts/blind_solve_rate.py`.
- Wide MASTERSTAR staged: `draft_000365/platesolve/NoFilter_60_2/MASTERSTAR.fits`.

**Config:** `blind_index_series`, `blind_index_select_mode` in `config.json` / `VYVAR_PARAMS.md`.

**Metrics:** `validation/blind_solve_rate.csv` — **9/10 HIT (90.0 %)** on battery v2 (8× Newton
Chi_and_H + M67 + 1× wide MISS on draft_365). Median sep 0.07° on hits. Wide tier: verify fails,
nearest votes ~11.5° — open tuning item.

**Rig-prior (same day, wide closure attempt):** `blind_use_rig_prior`, `blind_scale_tol_frac` (0.10);
A1 pre-vote L3 ratio gate; A2 verify WCS scale gate; A3 FOV bounds; gnomonic sides FOV≥2°; wide vote
fallback re-enabled under gates. Wide diag (`scripts/diagnose_blind_solver_wide.py`): **0 votes &lt;2°**,
nearest ~11–20°, flat≈gnomonic at edge sample — distortion not dominant; **wide index/triangle
correspondence** still the blocker. Newton spot-check OK post-change.

---

## Session 2026-06-08 — audit campaign + trust_flag_core Phase 2

- F841 triage finished: all flagged locals (`dist_score`, `rms_f2`, `c1_stderr@7141`, `lc_df`)
  are dead/redundant, none are real bugs; ranking sort confirmed correct (sorts by `comp_rms`).
  Automated lint layer essentially exhausted/cosmetic.
- Phase 2 manual audit of `trust_flag_core.py`: findings A–F (see `docs/VYVAR_AUDIT_FINDINGS.md`).
  Headline: un-evaluated target defaults to GREEN (A); missing check-star = no penalty (B);
  check-star scatter uses ddof=0 (C).
- Recorded language rule in PROCESS (Cursor↔Claude English; Milan↔Claude SK/CZ).
- Open items moved to ROADMAP (`NEXT SESSION` section).
- F841 batch 3: production 18 -> 1 pending (`n_rms_candidates` Cat 3). Removed dead locals
  across `comp_qa`, `photometry_core` (`ra_ms`/`de_ms`, `gaia_teff`), `pipeline` (`n0`×4, `cfg`),
  `psf_photometry` (`fit_shape`), `vyvar_alignment_frame` (`max_detected_stars`),
  `vyvar_platesolver` (`center`), and harness modules. Audit notes: `cfg`/`max_detected_stars`/
  `fit_shape` confirmed vestigial; `n_rms_candidates` awaits Milan (wire m2 RMS count vs remove).
- Phase A1: `VYVAR_AUDIT_FINDINGS.md` re-encoded UTF-8 ASCII; `_gen_audit_findings.py` ASCII emitter.
- Verify: pytest 174/6 skip; photometry SHA unchanged; PDF overflow 0.
- Phase C: ruff safe auto-fix batch (production) -- 37 fixes (SIM114 17, RUF010 8, B009 4,
  B010 4, SIM300 3, SIM910 1) across 16 modules; parenthesized dense SIM114 and/or merges.
  Verify: 0 remaining in production scope; pytest 174/6 skip; photometry SHA unchanged; PDF 0.
- Phase D: bug-class lint sweep (B023/B905/B904/RUF012/B007) -- 51 ruff instances cleared.
  B905 policy recorded in DECISIONS. `_norm_med_for_bin` duplication flagged for Phase F;
  B023 fix required `frame_med = nan` init before mag-bin branch (conditional binding).
  Verify: pytest 174/6 skip; photometry SHA unchanged.
- Phase E (trust_flag_core A+B): un-evaluated default RED + warn; nan check-star soft note;
  forward guard documented; C1 ddof=0; E deferred. `tests/test_trust_flag.py` (9 tests).
  draft_000366 trust re-run: 10 GREEN->YELLOW, 0 GREEN->RED; numeric LC/comp_quality unchanged.
  pytest 183/6 skip.
- Phase F manual audit: comp_qa_core.py done (CQ-A..E in AUDIT_FINDINGS). Headline: proc-CSV
  HIGH likely resolved via `PROC_CSV_GLOB`; CQ-C locus order-coupling needs conscious decision;
  CQ-B useless ternary; norm_id + `_norm_med_for_bin` duplication flagged for shared-core sweep.
  Next: calibration.py, database.py, vyvar_platesolver.py.
- Phase F: calibration.py done (CAL-A..D). Clean module; only LOW/future notes (passthrough
  caller logging, RGGB assumption for TODO-45, Bayer global rescale doc). Next: database.py,
  vyvar_platesolver.py.
- Phase F COMPLETE: database.py (DB-A..D, mostly sound), vyvar_platesolver.py (PS-A..C, well-gated
  blind solver). No new correctness bugs beyond CQ-B dead ternary. Headline: CQ-A proc-CSV HIGH
  likely resolved -- verify pre-cal draft and close. PS-B -> Phase G priority. Actionable list in
  AUDIT_FINDINGS Phase F COMPLETE section.
- Phase F follow-ups (consolidated): CQ-B/E, shared `norm_med_for_bin`, DB-A allowlist, m2
  RMS-only UI; CQ-A verified (proc glob tests, HIGH closed); CAL-A (no passthrough callers);
  DB-B closed (pipeline per-rerun). CQ-C documented + ROADMAP fix-once locus. pytest 183/6 skip;
  on-disk 283-file SHA stable across diff; PDF 0.
- Phase G batch 1: confirmed COMP_QA/TRUST stage wrappers log; 8 platesolver solve-result-path
  excepts now LOGGER.debug (logging-only). pytest 183/6 skip; numeric SHA 770966c3 unchanged.
- Phase G batch 2: 6 platesolver pass-excepts now log (1 warning MASTERSTAR WCS persist, 5
  debug); ~25 skip-OK reviewed. OPEN QUESTION: fatal MASTERSTAR write? pytest 183/6; SHA unchanged.
- Phase G batch 3: 7 photometry_core excepts logged (edge-ok fail-open, variability export x2,
  color-term x2, pipeline_meta); remainder skip-OK. OPEN QUESTION: edge-ok fail-closed? pytest
  183/6; SHA unchanged.
- Phase G batch 4: 3 pipeline.py excepts logged; worker error-surfacing reviewed; critical path
  COMPLETE (platesolver + photometry_core + pipeline). pytest 183/6; SHA unchanged.
- Phase H: cosmetic lint value-filtered (SIM118 x11, RUF022 x2, RUF007 x2, RUF034 x3); ProcFrameStore
  `.keys()` x2 kept; ~89 style accepted (PROCESS). Clean-code campaign A-H COMPLETE. pytest 183/6;
  SHA unchanged.
