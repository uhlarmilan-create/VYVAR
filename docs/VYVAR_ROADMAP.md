# VYVAR — Roadmap (open work)

Single source of truth for **open** tasks. Closed work lives in `VYVAR_JOURNAL.md`;
durable rationale in `VYVAR_DECISIONS.md`; current architecture in `VYVAR_STATE.md`.

Reconciled against the full development log on **2026-06-09** — de-duplicated, and
stale-closed items removed (e.g. GS6b had been listed as open in a side register but was
closed 2026-05-20).

Priority legend: **HIGH / MEDIUM / LOW / FUTURE**. Each item is a short status, not a history.

---

## NEXT SESSION — open items

1. **Brno / Newton characterization gate (finish)** — `g_60_4` solves on **production path**
   (draft_400: 75.5% brightest-N, WCS persists). **Open:** Milan **draft_401 UI sign-off + overlay**;
   Brno **r/i/z** end-to-end; gate `_brno_check` tail fixed (draft_400 / skip).
2. **TODO-MULTISET** — per-telescope-set config (wide vs fine optics); blocks clean multi-rig
   production and crowding gating per rig.
3. ~~**Short-baseline LC quality (#3)**~~ **DONE (2026-06-10)** — `short_baseline` terminal class,
   config keys, trust YELLOW non-escalating, exportable. ~~Finding E~~ **re-checked (2026-06-11)**.
4. **Phase C catalog rebuild** — DR3 full-sky build completes on existing schema (G<=17.5).
   GAIA-1/GAIA-2 columns deferred to DR4 (~Dec 2026) -- see DECISIONS.
5. ~~**Chi_and_H baseline re-cut**~~ **DONE (2026-06-11)** — full zaloha anchor
   (core `3f7c9e7a...`, full `d5b72d08...`; draft_000387 re-cut ×2); completeness
   gate in `night_run.py`. See STATE + RUNBOOK.
6. ~~**Trust Findings A/B + CS-1..4 + comp trust floor (Option B)**~~ **DONE (2026-06-11)** —
   specs under `docs/`; trust baseline 1382/106 on draft_387 at `comp_trust_min_comps=5`.
7. ~~**Broad-except hygiene (BLE001/E722 regression guard)**~~ **DONE (2026-06-11)** — see
   DECISIONS + `pyproject.toml` / pre-commit / `tests/test_ble001_regression.py`.
8. **INSTALL-MANUAL** — user install + catalog build guide; Lenovo T460 + TODO-LIB.
9. ~~**Per-frame proc export perf (DAO pre-filter + Moffat gate)**~~ **DONE (2026-06-12)** —
   `_proc_drop_unmatched_dao_rows` before aperture/PSF; Moffat gated on `_run_epsf` only; ~4.8× on
   `draft_000389` B_60_1 (171 → 36 s/frame). See JOURNAL + DECISIONS.
10. ~~**Sparse-only comp fallback**~~ **DONE (2026-06-11)** — default **ON**; anchor
    `3f7c9e7a` / `d5b72d08`; science-meaningful comparator for regression vs prior cut.
11. ~~**Comp-slope stability (B2+B1)**~~ **DONE (2026-06-11)** — common-mode detrend BJD sort +
    significance gate (`comp_slope_significance_k`); Honeycutt 1992 conditional. Anchor footprint
    LC-neutral on `draft_000387`.

---

## HIGH

- **TODO-MULTISET — per-telescope-set config architecture.** One config per rig (wide
  Carl-Zeiss 200 mm + QHY294MM ≈ 9.77″/px vs Newton 300/1200 + C3-26000 ≈ 0.65″/px).
  Underpins per-set plate scale, aperture, and crowding gating; blocks clean multi-rig
  production.
- **TODO-GS8 — Multi-Night Global Matching + global ZP solver (Phase 3).** Cross-night comp
  matching + inter-night zeropoint → one long-baseline LC with no vertical jumps.
  Dep: AAVSO validation (GS6b ✓). ~2–4 days.
- **APCORR-MIXEDFRAME — latent (COG default OFF); blocks enabling COG in production.** With
  `cog_aperture_correction_enabled=True`, `cog_ok=False` frames keep *uncorrected* flux
  alongside corrected ones → cross-frame step in the LC. The QA dashboard `IS_REJECTED` path
  is separate and clean (rejected frames are dropped before calibration, never enter the LC).
  Fix = drop `cog_ok=False` frames / wire the nightly `fallback_ee` / all-or-nothing per
  night. (Rationale in DECISIONS.)
- **PSF on fine-scale (Newton ≈ 0.65″/px) data.** Infrastructure is DONE and default **OFF**
  (wiring `psf_flux`→Phase 2A, adaptive selector, per-star quality + auto-fallback, spatial
  grid, grouper — all lose to aperture at 9.77″/px, correctly kept off). OPEN:
  - ~~**Crowding rule-2 bug**~~ **CLOSED (2026-06-09):** resolvable-blend->PSF (rule 2) was
    **removed** from adaptive routing (`photometry_core.py` ~5484); `is_blended` at 1.5 FWHM
    remains for crowding metrics only. Adaptive PSF uses faint-isolated rule only. NEIGHBOR-SUB
    is the blend path (not grouped PSF).
  - **TODO-PSF-NEIGHBOR-SUB** — subtract neighbour + aperture residual (gated OFF).
    **Steps 1-2a + pre-2b DONE**: A9 envelope, `psf_neighbor_sub.py`, fail-safe guards +
    `bright_close_regime` edge guard. Fine-scale draft 367: mismatch **~1.0**, HV **~83%**,
    FAIL-SILENT **0**; real crowding **sparse** (9 blended) -> **VALIDATED_FINE_SCALE_IDLE**;
    **2b deferred** until blended fine-scale field. Coarse bin2 remains **SAFE_LOW_YIELD**.
    Design: `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md`; crowding: `docs/VYVAR_DRAFT367_CROWDING.md`.
  - **TODO-PSF-V3d-FINE-SCALE** — **DONE (2026-06-08)** harness `tests/validation/v3d_fine_scale.py`.
    Inject-and-recover at draft-367-like scale; PASS on accuracy/precision/calibration pillars.
    Report: `tier_v3d/v3d_fine_scale.md`. Production PSF still OFF; real-field enablement separate.
  - ~~**TODO-PSF-V3d-MIDMAG-BIAS**~~ **DONE (2026-06-09)** sky-only PSF fit weights
    (`psf_weight_mode=sky_only`; Astier 2013 / Lacroix 2025). V3d mid-mag <2%, drift sub-%.
    Report: `tier_v3d/v3d_weight_proof.md`. PSF arc ready to batch (pre-2b + V3d + sky + weights).
  - **TODO-PSF-MULTIFRAME** — multi-frame ePSF stacking (isolation part done).
  - **TODO-PSF-ASYMMETRY** — tracking-smear diagnostics (BO CVn right-tail PSF).
  - ~~**TODO-FWHM-CONSISTENCY**~~ **DONE (2026-06-09)** — `header_core_fwhm_px` in
    `masterstar_context.py`; `crowding_index._load_wcs_meta` + `psf_photometry.get_epsf_fwhm_from_context`
    prefer `VY_FWHM_GAUSS` -> `VY_FWHM_GAUSSIAN` -> `VY_FWHM`, matching aperture path
    (`pipeline.py:9206`). h & chi Per L: is_blended 77/87 -> 58/53; numeric SHA 770966c3 unchanged.
    See `docs/VYVAR_HCHIPER_CROWDING_RECOMPUTE.md`.
  - ~~**TODO-EPSF-1-FWHM-QC**~~ **DONE (2026-06-08)** — `_epsf_fwhm_native_from_profile`
    (azimuthally-binned radial profile); QC warning band [0.80, 1.25]. V3e PASS (NEW ratios
    1.038-1.049 on synthetic Moffat). Diagnostic only; numeric SHA 770966c3 unchanged. See
    `docs/VYVAR_EPSF_AUDIT.md`, `tier_v3e/v3e_epsf_fwhm.md`.
  - ~~**Realistic per-star PSF uncertainties**~~ **DONE (2026-06-09)** via sandwich variance
    (`psf_err_mode=sandwich_skyonly`; V3d P3 ~1 mag<=17). Real-field enablement still blocked
    on a Newton / dense-field draft (Brno characterization gate).
- **Per-frame saturation (not whole-star skip).** Whole-star `zone_flag=saturated` from the
  longest-exposure masterstar drops comps/targets with viable unsaturated frames (76 Green/49 Red
  on M67). Same "silent wrong drop" class as the cross-group MASTERSTAR bug.

## MEDIUM

- **TODO-COMP-P2P-RESIDUAL — evaluate p2p outlier on common-mode-removed residual.** The p2p
  stability check (`check_comparison_stability`) still fires on the raw comp LC; on nights with a
  shared airmass/transparency ramp, `rms_p2p` ≈ the ramp range (same methodological class as the
  pre-B2 slope bug). **Separate science-changing task:** mirror the Honeycutt common-mode removal
  before p2p scoring; own footprint + optional anchor re-cut. Comp-slope fix (B2+B1) is **DONE**
  pending Milan acceptance; p2p deferred deliberately.
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
- ~~**TODO-BROAD-EXCEPT-HYGIENE**~~ **DONE (2026-06-11).** Phase G critical-path logging
  (2026-06-08) + **BLE001/E722 regression guard** (pyproject, pre-commit, pytest); 168 sites
  grandfathered `# noqa: BLE001`; 4 bare excepts fixed; 8 `photometry_core` narrowings. ~1200
  pre-existing noqa sites remain opportunistic. MASTERSTAR writeto fail-closed + edge-ok
  fail-open (#4) unchanged.
- **Phase H cosmetic lint (2026-06-08, DONE):** value-filtered subset applied (SIM118 x11,
  RUF022 x2, RUF007 x2, RUF034 x3 dead-ternary); ProcFrameStore SIM118 x2 kept; ~89 style
  findings accepted per PROCESS. **Clean-code campaign Phases A–H COMPLETE.**
- ~~**TODO-GEO**~~ **CLOSED (2026-06-09)** — superseded by PARAM-PROVENANCE (`param_resolver.py`:
  per-draft `ID_LOCATION` -> header -> config; BJD/airmass config-independent). See DECISIONS.
- ~~**Comparison-star floor policy**~~ **DONE (2026-06-11, Option B).** Trust-only
  `comp_trust_min_comps=5`; Phase-1 selection min stays 3. Spec:
  `VYVAR_COMP_FLOOR_POLICY_SPEC.md`. Option A (selection floor 5) parked — moves anchor.
- **`classify_lc_quality` short-baseline (#3).** **DONE (2026-06-10)** — see
  `VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md`. Follow-up: vsx_type-aware thresholds.
- ~~**Night-run false success (draft_383)**~~ **DONE (2026-06-11)** — completeness gate in
  `night_run.py` (>=90% summary/active per setup).
- ~~**Check-star trust audit CS-1..4**~~ **DONE (2026-06-11)** — specs +
  `VYVAR_CHECKSTAR_SELECTION_SPEC.md`; CS-3 ensemble exclusion via `ensemble_ids`.
- **Reserved check-star (hold-one-out).** PARKED — moves photometry anchor; see DECISIONS.
- **AAVSO-standard output (#4).** PARKED — G→standard B/V/Rc (Broeg band/colour point).

## LOW

- **GAIA-1 / GAIA-2 (pmra/pmdec, ruwe)** -- **DEFERRED to Gaia DR4 build** (~Dec 2026). Not
  restarting the DR3 rebuild. See DECISIONS + `VYVAR_GAIA_DR3_AUDIT.md`. DR4 migration hooks
  (epoch J2017.5, build columns, lite-table check) recorded in DECISIONS.
- ~~**comp_qa fix-once magnitude locus (CQ-C)**~~ **DONE (2026-06-09)** — fix-once pass-1 locus;
  order-independent flagging; bounded diff 1 flag / 1 n_clean / 0 trust on draft_000366; SHA
  `edbd97e7...` (426 files incl. comp_qa). Sibling ddof+threshold co-calibration remains open.
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
  **TODO-RECUT-HARNESS-FIDELITY** — re-cut harness vs frozen anchor (`draft_000387` /
  `3f7c9e7a…`) does not reproduce science-meaningfully (legacy arm alone: 1087 failures, B
  max \|Δmag\| ≈ 2.26); use legacy-vs-scoped same-harness control for solver gates until fixed
  (`sandbox/anchor387_legacy_vs_scoped_gate.py`) ·
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

## Parked (next — Milan chooses, 2026-06-14)

| Item | Notes |
|------|-------|
| **CM-detrend differential** | ~10× lever; opt-in; needs transit injection-recovery test before opt-in |
| **Exoplanet / TOI catalog** | NASA Exoplanet Archive TAP integration |
| **Newton-V colour-term** | Per-rig c1 from field BP-RP |
| **Meridian-flip handling** | Qatar-8 class |
| **`[TODO-RECUT-HARNESS-FIDELITY]`** | Re-cut vs archive gate unreliable (~2.26 mag B drift) |

**Backlog (unchanged):** broad-except Tier-1 (~25); B-V legacy removal Stages 2–4; TODO-46 (skip
airmass detrend for known VSX variables); TODO-LC-QUALITY; TODO-LC-TREND; TODO-GEO; GS8–GS11.

---

## Parked (round 2 — refinements, not blocking)

- **Magnitude-aware check-star threshold for the trust gate.** The flat `0.02` / `0.05`
  cutoffs carry the same magnitude-dependence the comp_qa locus fixed; extend the
  Sokolovsky/locus treatment to the check-star axis.
- **PSF cross-validation** — needs a PSF-heavy/faint draft + per-frame ePSF (the aperture
  cross-val is CLOSED).

## Dropped / resolved (do not re-open)

- **TODO-GS9 — Ground-LC period analysis in the PDF** — **closed: descoped 2026-06-09.**
  Lomb-Scargle/BLS + folded diagram on VYVAR's own Phase-2A LC as a PDF science product is
  out of scope; period finding/classification is downstream (Peranso, VStar, Period04). See
  DECISIONS (product scope boundary). LS/BLS citations remain for `tess_verify` TESS cross-check.
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

---

## Parked for next session (2026-06-11)

| Item | Notes |
|------|-------|
| Reserved check-star | Hold-one-out by design — **moves photometry anchor** |
| AAVSO-standard output #4 | G→B/V/Rc (Broeg band/colour) |
| TODO-MULTISET | Per-rig config (wide vs fine) |
| TODO-GS8 | Phase-3 global ZP |
| DR4 build | ~Dec 2026; J2017.5 epoch hook `vyvar_platesolver.py:63` |
| PSF / NEIGHBOR-SUB | Needs bin1 ~0.65"/px data (Brno gate) |
| `build_gaia_catalog.py` adaptive-split | Next full-sky build only (not this commit) |
| **D1-combination (Broeg-weighted vs flux-sum)** | Re-test weighted `ens_med` after colour/extinction — **moves anchor**; blocked on D3 |
| **D3 extinction/colour physics** | Second-order extinction + standard system → ties **AAVSO #4** |
| **C second pass (Howell + aperture corr.)** | CCD error budget, curve-of-growth / APCORR; citation-integrity follow-up |
