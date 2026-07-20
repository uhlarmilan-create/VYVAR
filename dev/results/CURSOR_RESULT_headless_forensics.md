# CURSOR RESULT - FORENSIC-HEADLESS (431/432 census regression)

**Date:** 2026-07-16. **Commit under test:** `715391b`. **Read-only + one discriminator.** No production fixes.

Bundle: `tmp/headless_forensics/`  
(`f1_config_diff.json`, `f2_chain_and_fixes.json`, `f3_discriminator_census.json`,
`f4_seed_audit.json`, `f5_export_soft_fails.json`, `discriminator_ui_match2.log`)

---

## VERDICT

### Root cause of the census regression (428-class)

**Deterministic MASTERSTAR pixel recipe under current headless/`night_run` produces the contaminated
image** (pass-1 DAO ~8927 -> MS 6699 / matched 3993). This is **not** explained by:

- worktree vs main `config.json` (paths resolve to main Archive/Gaia; config load is
  `Path(__file__).parent/config.json`, not cwd),
- commit delta `695348b`->`715391b` (only `fc177be` stamp/AC/TAP + `a3536a0` identity QA - neither
  changes stack pixels),
- `NightRunParams.catalog_match_max_sep_arcsec=25` vs UI `2.0` (**ruled out by F3**).

**Hard evidence:**

| Draft | Entry | MASTERSTAR == Light_008? | cal == processed (Light_008) | n_ms / matched / vsx |
|-------|-------|-------------------------|-----------------------------|----------------------|
| 428 | UI (pre-WCSINV dirty) | yes (pixels == 431) | **yes** | 6699 / 3975 / 0->46-class |
| 429 | UI `695348b` dirty | yes (!=431 pixels) | **NO** (maxabs~1.7e4 ADU) | **3054 / 2875 / 87** |
| 431/432 | headless pair | yes ==428 | **yes** | 6699 / 3993 / 46 |
| **433** | F3 `night_run` UI-parity (match 2.0, SysRem False) | yes ==431 | **yes** | **6699 / 3993 / 46** |

Healthy 429's MASTERSTAR is **Light_008 after an ADU-mutating step that current
`_preprocess_calibrated_one` does not perform** (pixel copy + headers only). Contaminated
drafts use `processed == calibrated`. Census class tracks that image split, not SIP finalize
bookkeeping.

F-428 identity/finalize **did execute** on 431 (coord_source populated; identity p95=6.24). They
cannot un-inflate a pass-1 that already ran on the contaminated MASTERSTAR.

### Root cause of the SHA mismatch (431!=432)

**`err`:** `measure_empty_aperture_sigma_bkg` uses `np.random.default_rng()` **unseeded** when
`rng is None` -> OS entropy (`photometry_core.py` ~757).

**`delta_mag_sysrem`:** SysRem itself is deterministic given weights, but weights are `1/err^2`.
`scripts/anchor_pair_run.py` **forces** `sysrem_enabled=True` while `config.json` / UI-429 have
`False`. Pair protocol = two full imports (!= RUNBOOK draft_387 same-draft re-run).

Science mag columns remain byte-stable (benign science compare).

### stamp=46

**Not a third legacy stamp branch unique to headless.** Live True count is **match-time** VSX
flags. C1 (`pipeline.py` ~12167) runs **before** `write_photometry_plan_files` (~12386) creates
`variable_targets.csv` -> stamp skipped on both UI and headless. Post-hoc restamp: 429 id_join=180,
431 id_join=199. Match-time 87 vs 46 follows census health, not entry-point stamp wiring.

---

## F1 - Config snapshot

**429 vs 431 - 5 keys only:**

| Key | 429 | 431 |
|-----|-----|-----|
| `annulus_inner_fwhm` | 4.75 | 5.75 |
| `comp_max_delta_bprp` | 0.79 | 0.64 |
| `phase01_comparison_max_comp_rms` | 0.1 | 0.08 |
| `phase01_comparison_min_dist_arcsec` | 60 | 90 |
| `sysrem_enabled` | False | **True** |

First four = `DENSITY_OVERRIDES["dense"]` on inflated census (429 `density=normal`). **Not**
user `config.json` drift. SysRem True = orchestrator force. **431 == 432** (0 config diffs).
All path keys point at main repo (`C:\ASTRO\python\VYVAR\...`), none into `tmp/anchor_run_wt`.

---

## F2 - Call chains + fix sites

`app.py` **does not call** `night_run` (docstring: UI wrapper deferred). Parallel wrappers share
`astrometry_align_and_build_masterstar` / `generate_masterstar_and_catalog` /
`run_full_photometry_pipeline`.

| Stage | UI | Orchestrator / pair |
|-------|----|---------------------|
| Entry | `_run_vyvar_full_pipeline` | `anchor_pair_run` -> `run_night_pipeline` |
| Preprocess | `_vyvar_execute_preprocess_pending` | `_night_run_preprocess` |
| Platesolve+MS | `_vyvar_execute_platesolve_pending` | `_night_run_platesolve` |
| Photometry | `run_full_photometry_pipeline` | same |

| Fix | Both chains |
|-----|-------------|
| FIX 1 round-trip gate | **ON** (shared) |
| FIX 2 Gaia-sky SIP | **ON** (optimizer ~625-656) |
| FIX 3 finalize + coord_source | **ON** |
| FIX 4 identity gate | **ON** (431 p95=6.24) |
| C1 stamp post-finalize | **BYPASSED** chronologically (VT after stamp) |

Param note: UI hardcodes `cat_match_arc=2.0` -> floor `max(10,sep)=10"`; pair default 25".
**F3 shows this is not decisive.**

---

## F3 - Discriminator

`scripts/forensic_disc_ui_match2.py` on **main** `715391b`, `D:\BO_CVn`,
`catalog_match_max_sep_arcsec=2.0`, `sysrem_enabled=False` -> **draft_000433**.

**Sick class** (identical MASTERSTAR pixels + identity QA to 431). Divergence is **not**
restored by UI match radius / SysRem off.  

**Not** a Streamlit `_run_vyvar_full_pipeline` call (no headless Streamlit session). `night_run`
is the intended extract. Next isolation (check with Milan): clean UI RUN VYVAR and verify whether
`processed==calibrated` and census class. **Do not bisect** `fc177be`/`a3536a0` for census -
those commits do not alter MASTERSTAR pixels; F3 already rejects 'match-sep / SysRem' and the
intervening commits as causal.

---

## F4 - Seed audit + pair protocol v2 (proposal only)

- Labbe empty-aperture: **unseeded** `default_rng()`.
- SysRem: deterministic given `err`; inherits Labbe noise; order = sorted `lightcurve_*.csv`.
- Seeds are **not** draft-id/path/time derived.
- **Protocol v2 (do not implement):** import once -> run photometry stages twice on same draft ->
  byte-compare; later harden Labbe with content-derived seed.

---

## F5 - Export soft failures (431/432)

| Target | Format | Root cause |
|--------|--------|------------|
| `1497007144465726080` | aperture LC export | LC CSV missing |
| `1497121459315202560` | aperture LC export | no exportable points (empty flags/mag); CSV exists |
| `1498278351706325248` | aperture LC export | LC CSV missing |

None are programme LCs / export fails on **429**. **Symptom of sick census**, not independent
export bug.

---

## Recommended fix plan (architect -> fix task)

1. **Reproduce healthy MASTERSTAR on clean tree:** Milan UI RUN VYVAR on `715391b` + `D:\BO_CVn`;
   verify `cal==proc` and census. If UI also sick -> 429 health was dirty-tree / lost ADU step.
   If UI healthy with `cal!=proc` -> find the missing ADU transform and restore it in shared preprocess
   (or document intentional sky model).
2. **Until census healthy:** keep anchor **blocked**; no `--finalize`.
3. **Stamp order:** move C1 after `write_photometry_plan_files` (or re-stamp once VT exists).
4. **Pair SHA:** stop forcing SysRem in `anchor_pair_run` (honor config); seed Labbe RNG;
   migrate pair protocol toward same-draft double photometry.
5. **Align defaults:** set `NightRunParams.catalog_match_max_sep_arcsec` to UI effective intent
   (document floor `max(10,sep)`); still secondary after (1).

---

## Files changed (this task)

- `CURSOR_RESULT_headless_forensics.md` (this file)
- `tmp/headless_forensics/*` (bundle)
- `scripts/forensic_disc_ui_match2.py` (discriminator launcher)
- `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_JOURNAL.md` (blocked + HIGH ledger)

Draft **433** left on disk (discriminator). Photometry may still be finishing in background -
MASTERSTAR/census already definitive.
