# VYVAR — Development State

Last updated: **2026-06-04**.

This is the **entry point**: a snapshot of what is true *now* + an index. It deliberately holds
no history and no open-task detail — those live in the linked files.

| File | Holds |
|------|-------|
| `docs/VYVAR_ROADMAP.md` | Open work (the only place to look for "what's next"). |
| `docs/VYVAR_DECISIONS.md` | Durable design decisions + *why* they hold. |
| `docs/VYVAR_JOURNAL.md` | Chronological session log (history, append-only). |
| `docs/VYVAR_PROCESS.md` | How we work: Definition of Done, byte-identity discipline, config↔UI parity, tests. |
| `docs/VYVAR_PARAMS.md` | Config-key ↔ default ↔ clamp ↔ UI-location registry. |
| `CITATIONS.bib` | Single source of truth for all algorithm/software citations. |

---

## Mission

A high-automation differential-photometry pipeline that lets amateur astronomers contribute
science with confidence: **trust in the numbers** (comp-stability QA via comp_qa + per-target
trust gate) and **guardrails for non-experts**. Independent extraction cross-validation lives
in the offline `xval_run.py` harness (not in-pipeline). Aperture photometry is the validated
workhorse; PSF is a conservative, opt-in path for fine-scale optics.

## Pipeline (current)

```
Raw → Calibrate → QC (in-place) → Align → MASTERSTAR + Gaia DR3 catalog
   → Phase 0+1: comp selection per target (colour tier + stability gate, distance gate)
   → Phase 2A: differential photometry (Broeg 1/σ² ensemble; SNR-optimal per-star aperture)
   → comp_qa (Sokolovsky LOO QA, read-only)
   → trust gate (GREEN/YELLOW/RED per target; comp-health + check-star + lc_quality)
   → reports/exports (PDF SUMMARY MEASURE REPORT, AAVSO, VarAstro)
```

Plate scale is **WCS-derived** (≈ 9.77″/px on the wide rig). Differential weighting is **Broeg
(2005) 1/σ²**, order-independent. Comp selection ranks colour tier first, stability second
(both gated), proximity as a distance gate only. (See DECISIONS for the rationale.)

## Production defaults (feature flags)

| Area | Flag | Default |
|------|------|---------|
| Comp QA | `comp_qa_enabled` | **ON** |
| Trust gate | `trust_flag_enabled` | **ON** (inform-only) |
| Proximity tie-break | `phase01_comparison_proximity_tiebreak` | OFF (reverted) |
| PSF (all) | `psf_photometry_enabled`, `psf_adaptive_enabled`, `psf_grouper_enabled`, `psf_spatial_enabled` | OFF |
| COG aperture corr. | `cog_aperture_correction_enabled` | OFF |
| Crowding classifier | `crowding_classifier_enabled` | OFF (wide rig) |
| Detrend | `sysrem_enabled`, `savgol_detrend_enabled` | OFF |
| Skip processed/ | `skip_processed_directory` | OFF |

Comp bounds (user-configurable): `phase01_comparison_n_comp_min/max` = **3 / 8** (the trust gate
and comp_qa derive their thresholds from these). `max_comp_rms` = 0.1; colour cut ≤ 0.79.

## Rigs (known sets)

| ID | Telescope | Camera | Scale | Site |
|----|-----------|--------|-------|------|
| 1 | Carl-Zeiss 200 mm | QHY294MM | ≈ 9.77″/px (wide) | Jirny |
| 2 | Newton 300/1200 | C3-26000 | ≈ 0.65″/px (fine) | Dáblice |
| 3 | Noctutec 206/560 f/2.72 | C3-26000 | TBD | TBD |

Per-set config architecture is still pending (ROADMAP: TODO-MULTISET).

## Status

- **Last validated draft:** `draft_000365` (V842 Her field, 127 NoFilter_60_2 frames, wide
  rig, 143 targets).
- **Cross-validation:** CLOSED for the aperture path (offline `xval_run.py`: sep reproduces
  VYVAR to 0.2 %/frame); in-pipeline `sep_xval` stage **retired 2026-06-03**; PSF cross-val
  deferred.
- **Trust distribution (draft_000365, pre-retirement baseline):** GREEN 69 / YELLOW 59 / RED 15.
  Post-retirement counts: re-run trust gate on draft with comp_qa columns (see JOURNAL 2026-06-03).
- **Tests:** 147 passed / 0 failed / 6 skipped (last full run).
- **Reporting:** R1 overflow guarantee holds (0 violations); R3 (aperture-vs-PSF overlay) pending.
- **CT validation (draft_000380):** science-grade B/V/Rc c1 locked on clean full re-run; production
  architecture decouples CT from target selection (see session blocks below + DECISIONS).

---

### 2026-06-03 — Colour-term validation campaign (M67 machinery → h & χ Per science-grade) + CT production architecture + non-cal session mode

**CT machinery validated — M67 LRGB (draft_000368, Astrodon astro-RGB, SPA-1-CMOS, pre-calibrated).**
In-range apply path exercised on Blue: 11 `ct_ok=True`, `|ct_corr|` 0.06–0.21 mag, cat−inst scatter
dropped in 83 % of targets (median 0.097→0.040). Path-B blocks red giants. c1 are astro-RGB
filter-mismatch artifacts (~−1) → machinery-grade only, not science.
- Green: **count-gate-limited** — clean 240s-only fits (`stderr` 0.065–0.107) blocked at `n_comp` 5–6 < 7;
  recovered at min5-split (scatter ↓ 88 %).
- Red: **data-limited** — 240s saturates bright red comps → narrow comp baseline.
- **Exposure-merge hypothesis REFUTED (explicit retraction):** merging 60s+240s lifts `n_comp` ≥7 but
  **degrades** the c1 fit (`stderr` 0.065→0.76) — mixing heterogeneous exposures inflates per-comp
  scatter. Merge is not the Green fix; the count gate is the lever.
- Found: **whole-star saturation skip** wrongly drops comps/targets that retain usable unsaturated frames
  (76 Green / 49 Red comps) → per-frame saturation needed.

**Science-grade CT — h & χ Persei (NGC 869/884), draft_000375, photometric Johnson-Cousins B/V/Rc.**
User's Newton DDT 300/1200 + C3-26000, bin2 ≈ 1.3″/px, pre-calibrated, 20 s subs, Dablice (set #2).
Fitted on the ~150-comp pool (BP−RP 0.22–2.63):
- **c1: B −1.09 ± 0.013, V −0.40 ± 0.010, Rc −0.026 ± 0.010** (`stderr_ratio` 0.012 / 0.025 / 0.36).
- Comp scatter pre→post: **B 0.38→0.05, V 0.21→0.06, Rc 0.066→0.064** — reduction scales with `|c1|`.
- `|c1|` ordering **Rc < V < B** — physical. 23–24/24 in-range pass; Path-B blocks 5 red giants/filter.
- `n_comp` ~140 → **count gate moot in a rich field** (cross-field min_comp data point).
- Reproduced on full production photometry: B −1.05 (n=142), V −0.39, Rc −0.027.

**Correction of a prior framing (retraction).** The expectation "photometric filters → small `|c1|≪1`"
was WRONG. c1 is computed **relative to Gaia G**, so `|c1|` tracks the filter's spectral distance from G:
Rc≈G→~0, V moderate, B far→large (−1.09) — and that is **physical, not an artifact** (it is also why
astro-B was ~−1.4). Photometric B/V/Rc do show smaller `|c1|` than Pal7 astro-RGB (−1.44/−0.93/−0.61)
per band, but that comparison is **confounded** (different sensor + field). Conceptual: these are
(filter − G) terms → corrected magnitude is **G-referenced**, not standard Johnson/Sloan.

**CT production architecture fixed — colour term decoupled from target selection.**
Root cause: `*_ct_target_presel` overwrote `variable_targets.csv` while `VYVAR_CT_PROTOTYPE=1` was forced
in production scripts → B/V/R limited to ~30 presel targets, `"nan"` display names, starved comp LCs.
Fix: photometry always runs the **full VSX field** for every filter; CT is an applied-correction
**toggle** (`apply_color_term` = auto/on/off; auto on for B/V/Rc, off for L/Clear). Presel is opt-in only
(writes `variable_targets.presel.csv`; overwrites production CSV only under `VYVAR_CT_PROTOTYPE=1`);
`run_full_photometry_pipeline` auto-restores the full VSX cone if a presel stub is detected. Display
names: VSX else Gaia `catalog_id` (never `"nan"`). B re-run (draft_375): 372 targets, 0 `"nan"`, 364 LCs,
294/379 `ct_ok` (Path-B blocked the rest). 147 tests pass.

**Non-cal session mode (pre-calibrated import) — shipped.** UI "RUN VYVAR (non-cal)" skips bias/dark/flat,
treats source as calibrated lights, records `calibration_mode=pre_calibrated` end-to-end
(manifest/log/PDF/pipeline_meta). Per requirement: **no `calibrated/` dir** — frames stay in
`non_calibrated/lights/` and all consumers (alignment, MASTERSTAR candidate discovery, ProcFrameStore)
read from there via a single lights-source root. Path fixes landed in sequence: passthrough removed →
MASTERSTAR candidate discovery repointed → pre-cal CSV `proc_*` alias.

**Plate-solve (draft_375).** Pre-cal FITS lack RA/Dec → blind solver mis-landed at RA 196.5/Dec +38.4
(field is ~35/+57), 6 % match. Resolved by injecting a coordinate hint (RA 35.175/Dec +57.133) into
MASTERSTARs + source frames and resuming with the **standard** `vyvar_gaia_dr3.db` (G≤16). All four
groups solved 97–100 % at ~35.03/+57.14. DB coverage guard: **10,854 stars** at the field → coverage is
NOT the blind-solver cause.

**Blind plate-solve (2026-06-04, index series + rig-prior).** Production uses `blind_index_select_mode=auto` +
`GAIA_DR3/blind_index_series.json`: **fine** (mag14, cell 1°, 95/cell) for Newton; **wide** (mag14,
cell 2°, 16/cell) for Carl-Zeiss ~9.5″/px. **Rig-prior** (`blind_use_rig_prior=True`): pre-vote
L3 scale-ratio gate, verify WCS scale check, FOV-based cone/selection bounds, gnomonic triangle sides
when FOV ≥ 2°. Orchestrator: `vyvar_blind_series.solve_blind_with_series`. Harness:
`scripts/blind_solve_rate.py`; wide diag: `scripts/diagnose_blind_solver_wide.py`. **Wide HIT on
draft_365 still open** (nearest vote ~11–20°; 0 votes &lt;2°) — see `wide_diagnostic_report.json`.

**Closed 2026-06-04:** V/R re-run and `n_clean=0` / trust RED diagnosis — see session block below
(draft_380 full run + root cause: pre-cal proc-CSV glob in comp QA; not a cleaning regression).

---

### 2026-06-04 — Chi_and_H clean full re-run (draft_000380) + n_clean root cause

**draft_000380 — clean fresh full run, all 4 filters (B/V/Rc/L), coordinate hint.** Plate-solve all
anchored ~35.03°/+57.14° (98.5–100 %, no 196/38). **CT reproduced on the clean run:** B −1.084, V −0.383,
Rc −0.023 (vs validated −1.09 / −0.40 / −0.026); `|c1|` ordered Rc<V<B; `stderr_ratio` 0.012/0.023/0.386;
comp scatter pre→post B 0.376→0.053, V 0.188→0.047, Rc 0.061→0.060; all in-range `ct_ok` (363/362/364),
Path-B blocks 8 red giants/filter; L CT off (toggle). **CT science locked.**

**Decoupling/CT-toggle verified end-to-end** on all filters: ~371 targets/filter, **0 "nan"** names,
comps + check-star LCs present (331–364). Production CT path correct.

**n_clean=0 / trust RED — root cause (diagnostic; draft-specific, NOT a regression).**
`comp_qa_core.load_proc_pivot()` hardcodes `glob("proc_*_Light_*.csv")`. Pre-calibrated imports write
native basenames `proc_Chi_H_*.csv` (no `_Light_`) → 0 glob hits → comp QA evaluates 0 targets →
`n_clean` NaN on all 371 → trust maps NaN→0 → RED on every target/filter incl L. Regression baseline
**draft_000366** (Jirny V842 Her, calibrated, `proc_V842_Her_Light_*`, 127 frames) re-run with current
code **reproduces the original exactly** (n_clean median 8, trust G=91/Y=46/R=6) → cleaning algorithm
healthy. **Same pre-cal-naming class as the ProcFrameStore `proc_*`/`*_cal` issue — third consumer hit**
(after alignment/masterstar source root and ProcFrameStore).

**Secondary (separate, draft-specific):** `classify_lc_quality` `min_frames=20` > Chi_and_H's 12 frames →
`lc_quality=no_data` → an independent hard trust fail on 365 targets even if the glob were fixed.

CT result stands independent of the trust plumbing (c1 fit + scatter don't route through comp QA).
