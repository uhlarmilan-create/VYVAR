# VYVAR — Roadmap (open work)

Single source of truth for **open** tasks. Closed work lives in `VYVAR_JOURNAL.md`;
durable rationale in `VYVAR_DECISIONS.md`; current architecture in `VYVAR_STATE.md`.

Reconciled against the full development log on **2026-06-04** — de-duplicated, and
stale-closed items removed (e.g. GS6b had been listed as open in a side register but was
closed 2026-05-20).

Priority legend: **HIGH / MEDIUM / LOW / FUTURE**. Each item is a short status, not a history.

---

## NEXT SESSION — open items

1. **trust_flag_core fixes** — Finding A (un-evaluated → RED/UNKNOWN + warn on missing id) and
   Finding B (soft note when check-star scatter is nan). Needs spec + unit tests; guard via
   trust-output baseline, not photometry SHA.
2. **trust_flag_core Finding C** — decide ddof=0 vs ddof=1 for check-star scatter; record in
   DECISIONS.
3. **Phase 2 manual audit continues:** `comp_qa_core.py` → `calibration.py` → `database.py` →
   `vyvar_platesolver.py`.
4. **F841 `g_teff` / `gaia_teff`** — deferred (benign); optional later cleanup.
5. **Phase C — catalog rebuild mechanics:** deepening mag (full-sky 16.5) requires clearing
   `strip_progress` OR building to a new DB then swapping `GAIA_DB_PATH`; PKL rebuild only for
   DEC (southern) expansion. ("k tomuto sa vrátime")
6. **INSTALL-MANUAL** — install manual + installer for a new user (incl. catalog build via the
   2 GAIA scripts), tied to TODO-9 (Lenovo T460) + TODO-LIB (Cython `.pyd` package).
7. **Open question** — `ui_photometry_results.py` + `ui_suspected_lightcurves.py` are disconnected
   from `app.py`: intent vs regression? Decide.
8. *(optional)* **"new selection philosophy" ranking review** — sort by `comp_rms` @2089
   confirmed correct; broader read optional.

---

## HIGH

- **Canonical pre-cal proc-CSV resolution.** One source-of-truth pattern for per-frame `proc_*`
  CSVs used by **all** consumers — alignment source root, `ProcFrameStore`, and
  `comp_qa_core.load_proc_pivot` (currently `proc_*_Light_*.csv`). Pre-cal native basenames
  (`proc_<obj>_*.csv`, no `_Light_`) must match everywhere. **Third consumer hit by this mismatch**
  → stop patching per-site; fix once. It silently zeroes comp QA / `n_clean` / trust on every
  pre-cal run. *(Merged from ProcFrameStore pre-cal naming + Chi_and_H n_clean diagnostic,
  2026-06-04.)*
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
  - **TODO-PSF-NEIGHBOR-SUB** — fit + subtract bright-neighbour ePSF, aperture the residual
    (deblend that works at coarse resolution, unlike the grouper).
  - **TODO-PSF-MULTIFRAME** — multi-frame ePSF stacking (isolation part done).
  - **TODO-PSF-ASYMMETRY** — tracking-smear diagnostics (BO CVn right-tail PSF).
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
  dangerous subset is those guarding safety/fallback paths in the core runtime. Narrow each
  to the expected exceptions only. Includes: verify `comp_qa` / `trust` stage wrappers **log**
  on failure (not silent-pass); and tighten the few adhoc/script callers that use config `1.3` directly instead of the WCS-first plate-scale resolver.
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
  tune `wide` tier / quads — see wide diag report).
- **V/R re-run (draft_375)** — **RESOLVED 2026-06-04** via draft_380 clean full run (all filters).
- **Trust `n_clean=0` diagnosis** — **RESOLVED 2026-06-04:** root cause = pre-cal proc-CSV glob in
  `load_proc_pivot` (draft-specific, not a cleaning regression); folded into canonical pre-cal
  proc-CSV resolution (HIGH).
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
