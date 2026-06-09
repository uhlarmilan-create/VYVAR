# VYVAR — Roadmap (open work)

Single source of truth for **open** tasks. Closed work lives in `VYVAR_JOURNAL.md`;
durable rationale in `VYVAR_DECISIONS.md`; current architecture in `VYVAR_STATE.md`.

Reconciled against the full development log on **2026-06-04** — de-duplicated, and
stale-closed items removed (e.g. GS6b had been listed as open in a side register but was
closed 2026-05-20).

Priority legend: **HIGH / MEDIUM / LOW / FUTURE**. Each item is a short status, not a history.

---

## NEXT SESSION — open items

1. **trust_flag_core Finding E (deferred)** — lc_quality-missing soft note; revisit together with
   Finding D `len(soft)>=3` guard when a third soft source is added.
2. **trust_flag_core ddof+threshold co-calibration (C1 follow-up)** — if switching to ddof=1,
   re-tune 0.02/0.05 gates; currently ddof=0 locked (DECISIONS). Sibling:
   **comp_qa fix-once magnitude locus** (CQ-C; see DECISIONS).
3. **F841 production** — cleared (`n_rms_candidates` wired to m2, 2026-06-08 Phase F).
4. **Phase C — catalog rebuild mechanics:** deepening mag (full-sky 16.5) requires clearing
   `strip_progress` OR building to a new DB then swapping `GAIA_DB_PATH`; PKL rebuild only for
   DEC (southern) expansion. ("k tomuto sa vrátime")
5. **INSTALL-MANUAL** — install manual + installer for a new user (incl. catalog build via the
   2 GAIA scripts), tied to TODO-9 (Lenovo T460) + TODO-LIB (Cython `.pyd` package).
6. **Open question** — `ui_photometry_results.py` + `ui_suspected_lightcurves.py` are disconnected
   from `app.py`: intent vs regression? Decide.
7. *(optional)* **"new selection philosophy" ranking review** — sort by `comp_rms` @2089
   confirmed correct; broader read optional.

---

## HIGH

- **TODO-MULTISET — per-telescope-set config architecture.** One config per rig (wide
  Carl-Zeiss 200 mm + QHY294MM ≈ 9.77″/px vs Newton 300/1200 + C3-26000 ≈ 0.65″/px).
  Underpins per-set plate scale, aperture, and crowding gating; blocks clean multi-rig
  production.
- **TODO-GS8 — Multi-Night Global Matching + global ZP solver (Phase 3).** Cross-night comp
  matching + inter-night zeropoint → one long-baseline LC with no vertical jumps.
  Dep: AAVSO validation (GS6b ✓). ~2–4 days.
- **TODO-GS9 — Ground-LC period analysis in the PDF.** Lomb-Scargle + BLS on the Phase-2A LC
  CSV + folded/phase diagram for candidates, rendered into the report. LS/BLS citations are
  already present — **verify whether period analysis is actually wired into the PDF or only
  cited.** ~1–2 days.
- **APCORR-MIXEDFRAME — latent (COG default OFF); blocks enabling COG in production.** With
  `cog_aperture_correction_enabled=True`, `cog_ok=False` frames keep *uncorrected* flux
  alongside corrected ones → cross-frame step in the LC. The QA dashboard `IS_REJECTED` path
  is separate and clean (rejected frames are dropped before calibration, never enter the LC).
  Fix = drop `cog_ok=False` frames / wire the nightly `fallback_ee` / all-or-nothing per
  night. (Rationale in DECISIONS.)
- **PSF on fine-scale (Newton ≈ 0.65″/px) data.** Infrastructure is DONE and default **OFF**
  (wiring `psf_flux`→Phase 2A, adaptive selector, per-star quality + auto-fallback, spatial
  grid, grouper — all lose to aperture at 9.77″/px, correctly kept off). OPEN:
  - **Crowding rule-2 bug** — blend condition `nn_dist_fwhm ≤ 1.5` vs rule-2 requiring
    `≥ 2.0` is contradictory, so the resolvable-blend→PSF rule can never fire. Fix the
    thresholds.
  - **TODO-PSF-NEIGHBOR-SUB** — joint-fit + subtract neighbour, aperture residual (gated OFF).
    **Steps 1-2a DONE** (`055595d`): A9 envelope, `psf_neighbor_sub.py`, fail-safe guards;
    realistic mismatch **FAIL-SILENT 0**, **SAFE_LOW_YIELD** (~18% HV recover at coarse bin2).
    **Step 2b blocked** (pipeline wire): re-test fine-scale A9 (draft 367) and/or ePSF first.
    Design: `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md`; validation: `docs/VYVAR_VALIDATION.md` §A9.
  - **TODO-PSF-MULTIFRAME** — multi-frame ePSF stacking (isolation part done).
  - **TODO-PSF-ASYMMETRY** — tracking-smear diagnostics (BO CVn right-tail PSF).
  - ~~**TODO-FWHM-CONSISTENCY**~~ **DONE (2026-06-09)** — `header_core_fwhm_px` in
    `masterstar_context.py`; `crowding_index._load_wcs_meta` + `psf_photometry.get_epsf_fwhm_from_context`
    prefer `VY_FWHM_GAUSS` -> `VY_FWHM_GAUSSIAN` -> `VY_FWHM`, matching aperture path
    (`pipeline.py:9206`). h & chi Per L: is_blended 77/87 -> 58/53; numeric SHA 770966c3 unchanged.
    See `docs/VYVAR_HCHIPER_CROWDING_RECOMPUTE.md`.
  - **TODO-EPSF-1-FWHM-QC** — fix `epsf_fwhm_native` half-max estimator in
    `psf_photometry.py:500-516` (azimuthally-binned radial profile + recalibrated 1343
    warning thresholds). Audit finding EPSF-1: ratio<1 on h & chi Per (0.59-0.67) is likely a
    **diagnostic artifact**, not a narrow ePSF — does not affect flux or `assess_psf_quality`.
    Validate via harness **V3e** (synthetic known-FWHM -> ratio in [0.85,1.15]). See
    `docs/VYVAR_EPSF_AUDIT.md`. Milan decision before implementation.
  - Realistic per-star PSF uncertainties; then validate + enable on real Newton data.
  - Blocked on having a Newton / dense-field draft.
- **Per-frame saturation (not whole-star skip).** Whole-star `zone_flag=saturated` from the
  longest-exposure masterstar drops comps/targets with viable unsaturated frames (76 Green/49 Red
  on M67). Same "silent wrong drop" class as the cross-group MASTERSTAR bug.

## MEDIUM

- **Expose comp_qa / trust as Settings toggles** + write their defaults to
  `config.json`. These are **user-facing** QA features (the trust badge is observer-facing),
  currently config-only and UI-hidden. Add a "Data quality & validation" Settings section for
  the two. Keep the experimental `phase01_comparison_proximity_tiebreak` / `rms_bin_mag` and
  the PSF/COG/crowding flags hidden.
- **Persist offline cross-val results to a tracked ledger.** Append one summary row per
  `xval_run.py` harness run to `validation/xval_ledger.csv` (in git):
  `date, draft_id, engine, n_targets, median_ratio, n_confirmed, n_review, n_indep_failed,
  commit, notes`. The offline `xval_out/` stays scratch; per-draft photometry tree is unchanged
  (no in-pipeline `sep_xval_*` artifacts). Gives a longitudinal validation record (mission +
  paper Validation section). *(Re-scoped 2026-06-03 after in-pipeline sep_xval retirement.)*
- **TODO-APCORR-COLOR — Extrapolation guard warn→block: DONE** (2026-06-03). Target BP-RP
  outside comp range → CT skipped, target kept uncorrected (`phase01_ct_extrapolation_tol`,
  default strict 0). **NoFilter CT enable: PARKED** — prototype on draft_000366 showed modest
  effect (c1 ~ −0.1…−0.5 for gate-passers, cat−inst scatter 0.078→0.053 mag, only ~11% pass
  the numeric gate; colour cut ≤0.79 already suppresses most of it). Revisit when filtered /
  Newton fine-scale data exists.
- **Verify wide-field WCS distortion model (from SIPS comparison).** Confirm whether the
  plate solve fits higher-order distortion (SIP-style) or only a linear CD matrix. On the wide
  rig (~5.6°×3.8°, with a corrector) residual field curvature can reach several pixels at the
  edges → edge astrometry/photometry offsets. SIPS models this with a 3rd-order 2D polynomial
  (Monomial/Legendre). **Verify first:** if SIP is already fitted, close this; if linear-only,
  add higher-order distortion terms. (See DECISIONS: *What VYVAR deliberately does NOT adopt from SIPS* — WCS distortion scoped here.)
- **TODO-GS10 — AAVSO Direct API upload** ("Submit to AAVSO" button; WebObs API after the
  GS6b validation). Dep GS6b ✓.
- **TODO-45 — RGB camera support** (IMX533 RGGB → de-Bayer → G-channel photometry).
- **TODO-8-BOO — Bootes globular-cluster validation** (ePSF vs aperture on a dense ~2 h
  field). Pairs with the PSF/Newton work.
- **TODO-FORCED-COMP — forced-aperture `catalog_only` without Phase-1 tier selection.**
- **TODO-LC-TREND — differential extinction + ALG audit** (partial; re-validate on a
  moonless night).
- **TODO-LIB — Cython `.pyd` compilation** (hide source, enable C translation).
- **TODO-CONFIG-CHURN — dedicated day.** The app rewrites session/UI state into the tracked
  `config.json` each run → perpetual git diff. **Zero functional effect** (the resolver
  ignores the config site; UI uses `LOCATION.IS_DEFAULT`). Fix = split a session-state store
  from the static overrides. **Do NOT gitignore `config.json`** — it holds real overrides.
- **TODO-BROAD-EXCEPT-HYGIENE — dedicated day(s).** ~700 bare `except: pass/continue`; the
  dangerous subset is those guarding safety/fallback paths in the core runtime. **Phase G batch 1
  (2026-06-08):** comp_qa/trust stage wrappers confirmed logging (`[COMP_QA]`/`[TRUST]` warning,
  closed); 8 platesolver solve-result-path excepts now `LOGGER.debug` before None/False fallback
  (no control-flow change). **Batch 2 (2026-06-08):** 6 of ~31 `pass`-style excepts in
  `vyvar_platesolver.py` now log (1 `LOGGER.warning` for failed MASTERSTAR WCS persist, 5
  `LOGGER.debug` for skipped refinements/header writes); ~25 reviewed skip-OK (RANSAC inner-loop,
  diag-log guards, optional VY_* headers, fallback refinements). **Batch 3 (2026-06-08):**
  7 `photometry_core` high-risk excepts now log (3 warning: edge-ok fail-open, variability export;
  4 debug: color-term fit x2, pipeline_meta write); ~223 remainder reviewed skip-OK.
  **Batch 4 (2026-06-08):** 3 `pipeline.py` excepts logged (comparison-star sync skip ->
  warning; cone/variables + prefetch CSV writes -> debug); worker error-surfacing via status
  dicts + graceful fallbacks reviewed. **Critical path DONE** (platesolver / photometry_core /
  pipeline). Remaining ~700 repo-wide count is lower-risk modules (UI, importer, tess_verify,
  etc.) -- opportunistic only. **OPEN QUESTIONS (Milan):** failed MASTERSTAR `fits.writeto`
  fatal? edge-ok check fail-open vs fail-closed (logging only so far).
- **Phase H cosmetic lint (2026-06-08, DONE):** value-filtered subset applied (SIM118 x11,
  RUF022 x2, RUF007 x2, RUF034 x3 dead-ternary); ProcFrameStore SIM118 x2 kept; ~89 style
  findings accepted per PROCESS. **Clean-code campaign Phases A–H COMPLETE.**
- **TODO-GEO — observer geographic position audit (BJD/airmass/HJD).** **Likely superseded
  by PARAM-PROVENANCE** (per-draft `ID_LOCATION` site resolution, 2026-05-30) — verify and
  close if so.
- **Cross-field min_comp experiment (7 vs 5).** Across several archived fields (single-exposure
  groups, healthy stderr). Data points: h & χ Per (n_comp~140), M67 Green (clean but
  count-blocked), Pal7.
- **`classify_lc_quality` min_frames for short-baseline sessions.** Default `min_frames=20` marks
  ~12-frame sessions `no_data` → hard trust fail. Decide a lower floor or a session-length-aware
  policy (separate from the pre-cal proc-CSV glob bug; see STATE 2026-06-04).
- **AAVSO-standard output.** G→standard B/V/Rc transform or standard catalog (APASS) so
  CT-corrected mags sit on a standard system (see DECISIONS conceptual note).

## LOW

- **comp_qa fix-once magnitude locus (CQ-C methodology).** Per-target flag thresholds are coupled
  to deterministic target processing order via accumulating `dropped_global`. Keep current behavior;
  validate a fix-once locus (computed once over full pass-1 pool) with bounded n_clean/trust diff —
  sibling of ddof+threshold co-calibration (NEXT SESSION #2).
- **FITS-side proc glob consistency.** `pipeline.py` uses inline `aligned_dir.glob("proc_*.fits")`
  (~5578, 12604, 12683) rather than a shared helper; functionally correct (`proc_*` matches both
  naming styles). Optional consistency cleanup only.
- **Spatial term in calibration (from SIPS comparison) — only for a future whole-field absolute
  mode.** SIPS's ensemble adds x,y polynomial terms (`x1·X + y1·Y + x2·X² + y2·Y² + xy·XY`) for
  field gradient / vignetting. **Not a gap in VYVAR's current per-target differential path** —
  the local-comp ensemble already cancels most spatial systematics. Relevant only if/when VYVAR
  does whole-field absolute photometry (all targets against one frame-wide solution). Sibling of
  APCORR-COLOR (the colour term). (See DECISIONS: *What VYVAR deliberately does NOT adopt from SIPS* — spatial term scoped here.)
- **TODO-WIDE-RIG-REPROCESS — clean re-run of 361/362.** Not a code bug: MASTERSTAR CD/WCS +
  ePSF are already ≈ 9.77; the stale `pipeline_meta` carries `1.3` from an old run. Re-run
  with the current WCS-first resolvers to refresh comp geometry + meta. (DECISIONS.)
- ~~**B-V legacy removal — Stages 2–4.**~~ **Closed 2026-06-03** (scope A+B; commits in
  JOURNAL). Regenerate `vyvar_vsx_local.db` on the catalog machine after pulling.
- **TODO-10 — Settings-tab refactor + `CONFIG_GUIDE.md`.** Ties to `VYVAR_PARAMS.md` /
  config↔UI parity.
- **TODO-13 validation — Gaia→DAO completeness ~3.5%** still low in the QA dashboard;
  validate after the DAO pass-2 + forced rows.
- **TODO-LC-QUALITY — LC classification filter.** `lc_quality_flag` exists and is consumed by
  the trust gate; verify the saturated/noisy export policy is complete.
- **TODO-14 — PDF size optimization** (29 MB → < 10 MB).
- **TODO-MASTERSTAR-QA — FORCED_APERTURE cyan overlay** in the QA UI.
- **Misc LOW:** TODO-7 plate-solver refactor · TODO-11 auto-trigger watchdog (`night_run`
  foundation exists) · TODO-12 HRD classification (after new DB) · TODO-20 mean-stack
  MASTERSTAR (improves WCS/FWHM only, not LC SNR) · TODO-CACHE-CENTRAL centralize `csv_cache`
  · TODO-PIXEL-XCHECK-BINNING binning-aware pixel cross-check (cosmetic log) ·
  **TODO-INSTALL-MANUAL — inštalačný manuál + inštalátor pre nového užívateľa (vrátane
  katalógov)** · TODO-PLATESCALE-PERSET focal×pixel per-set plate-scale fallback.

### TODO-INSTALL-MANUAL (naviazané: **TODO-9**, **TODO-LIB**)

**Cieľ:** nový užívateľ (bez znalosti vnútra) nainštaluje celý VYVAR + sprevádzkuje katalógy
podľa manuálu; na referenčnom stroji **LENOVO T460** (Linux — potvrdiť presnú distro) to spraví
**jedným inštalátorom**. Inštalátor = realizácia; manuál = dokumentácia toho istého cieľa.
Prepojiť s Cython balíkom (**TODO-LIB**). Nahrádza/rozširuje starý one-liner **TODO-9**.

**Manuál (dokumentácia):**
1. Prostredie — Python, venv, `requirements.txt`, spustenie Streamlit dashboardu.
2. Aplikácia — získanie repa/balíka, prvé spustenie.
3. **Katalógy (kľúčové)** — 3 gitignored súbory: `vyvar_gaia_dr3.db` +
   `gaia_triangles_fine.pkl` + `gaia_triangles_wide.pkl`. Dve cesty: **(a) build**
   (`build_gaia_catalog.py` → ESA TAP, hodiny; potom `build_blind_index.py`) alebo **(b) stiahnuť
   hotové** (ak hostované). Na T460 je full-sky build @ G≤16.5 nepraktický → uprednostniť hotové
   (príp. slim podmnožina).
4. Konfigurácia — Settings: `GAIA_DB_PATH`, `BLIND_INDEX_FINE_PATH`, `BLIND_INDEX_WIDE_PATH`,
   `archive_root`, `calibration_library_root`, `database_path`; Test connection / Skontrolovať index.
5. Prvý beh — verifikačný checklist (test solve / krátky pipeline, trust gate GREEN).

**Inštalátor (T460):** skript/balík pripraví prostredie + app + nasmeruje na katalógy (download
hotových alebo build). Zohľadniť RAM (fine PKL ~1.3 GB + in-memory verify katalóg), disk (~10 GB DB),
CPU. Jedna zdrojová pravda s manuálom.

**Otvorené (rozhodnúť pri implementácii):** distribúcia katalógov (build vs download vs slim);
presná OS/distribúcia T460; forma inštalátora (shell / conda / pip / Cython wheel); min. RAM/disk v
manuáli.

**Definition of Done:** manuál → čistá inštalácia od nuly po GREEN trust na testovacom poli;
inštalátor → rovnaký výsledok na T460; manuál a inštalátor konzistentné.

## FUTURE

- **Blind index — 3rd rig tier (Noctutec 206/560).** When a validated draft exists, add a
  third PKL tier + config path; architecture is ready (`blind_index_fine_path` /
  `blind_index_wide_path` + `blind_index_select_mode=auto`).
- **TODO-GS7 — paper draft** (PASP / AN). Working title locked: *VYVAR: An Automated
  Differential Photometry Pipeline for Amateur Variable Star Observers*.
- **Comet photometry mode** — major parallel phase **after** the variable-star pipeline is
  finished (shared front-end calibrate→platesolve→star-stack→Gaia ZP; forked back-end:
  comet-rate stacking + extended coma photometry + ICQ/COBS export). Analysis only — do NOT
  start yet. (DECISIONS.)
- **TODO-SCENE-FORWARD-MODEL** — conditional on crowded-faint science (Brno / globular
  clusters); priority lowered after the grouper-negative result.

---

## Parked (round 2 — refinements, not blocking)

- **Magnitude-aware check-star threshold for the trust gate.** The flat `0.02` / `0.05`
  cutoffs carry the same magnitude-dependence the comp_qa locus fixed; extend the
  Sokolovsky/locus treatment to the check-star axis.
- **PSF cross-validation** — needs a PSF-heavy/faint draft + per-frame ePSF (the aperture
  cross-val is CLOSED).

## Dropped / resolved (do not re-open)

- **Blind solver in dense fields + index series + rig-prior** — **RESOLVED 2026-06-04** (Newton):
  mag14 tiers, `vyvar_blind_series`, solve-rate harness, scale/FOV hard gates (`blind_use_rig_prior`,
  `blind_scale_tol_frac`). **Wide-rig blind HIT** — still **OPEN** (`draft_365`: 0 votes &lt;2°;
  tune `wide` tier / quads — see wide diag report). **PS-A note:** `_verify_blind_candidates`
  relaxes `min_matches` 12→8 when plate scale ≥ 5″/px; when working this item, decide
  fraction/scale_tol compensation for the wide rig.
- **V/R re-run (draft_375)** — **RESOLVED 2026-06-04** via draft_380 clean full run (all filters).
- **Trust `n_clean=0` diagnosis** — **RESOLVED 2026-06-04:** root cause = pre-cal proc-CSV glob in
  `load_proc_pivot` (draft-specific, not a cleaning regression); folded into canonical pre-cal
  proc-CSV resolution (HIGH).
- **Canonical pre-cal proc-CSV resolution** — **CLOSED 2026-06-08:** `load_proc_pivot` uses
  `list_proc_csvs` / `PROC_CSV_GLOB="proc_*.csv"`; verified `tests/test_proc_csv_glob.py` +
  calibrated draft_000366 n_clean populated.
- **ProcFrameStore pre-cal naming** — folded into canonical pre-cal proc-CSV resolution (HIGH,
  2026-06-04); do not fix per-consumer.
- **MASTERSTAR-EPSF-ALL** — dropped 2026-06-02: plate scale is WCS-derived; affected drafts
  311/321/358 are deleted; 361/362 ePSF already ≈ 9.77. No recurrence risk.
- **GS6b** — DONE 2026-05-20 (`scripts/validate_aavso_export.py`). Residual delta only: add a
  headroom check so the new `trust=` AAVSO NOTES field stays within Extended-Format limits.
- **IRAF / PyRAF cross-val** (and TODO-32 EPADU) — closed as unnecessary; two independent
  engines (sep matches VYVAR to 0.2 %) already validate extraction; not feasible on
  Py3.12/Ubuntu24.
- **TODO-WEIGHTED-LC, TODO-SKY-PLANE** — tested negative, closed.
- **TODO-DEV-PROCESS** — folded into `VYVAR_PROCESS.md` (the Definition-of-Done discipline).
- The full **TODO-1…45 / PERF-1…10 / ALG-1…5 / CQ-1…7 / GS1–GS5** series — closed; see
  `VYVAR_JOURNAL.md`.
