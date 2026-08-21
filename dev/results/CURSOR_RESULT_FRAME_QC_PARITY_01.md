CURSOR RESULT - 2026-08-21 (FRAME-QC-PARITY-01 phase 1)

What I did
Read-only investigation of frame QC divergence class (DRAFT-517-REVIEW trigger).
No `src_py/` changes. Measured draft 516 vs 517 artifacts; traced gate authority
in code. Sandbox: `dev/sandbox/frame_qc_parity_01_measure.py` ->
`dev/results/context/session_20260821_frame_qc_parity/measurements.json`.

## Named mechanism (one paragraph)

VYVAR exposes **two independent frame-QC layers** on the full UI / `night_run`
pipeline path, and **neither layer runs** on the headless `--full` anchor
photometry path (`session_baseline_check.py` copies frozen aligned inputs and
calls `run_full_photometry_pipeline` only). **Layer A (calibration HFR):**
during `calibrate_lights`, `_post_calibration_qc_eval` (`pipeline.py:16328-16404`)
measures median half-flux radius on a central 1000 px DAO crop, compares to
`cfg.qc_max_hfr` (5.0 px; `qc_max_hfr_fwhm_ratio=None` -> legacy px via
`unit_resolver.resolve_hfr_limit_px`), stamps `VYQCPASS` / `VY_QCHFR`, logs
`Frame ... REJECTED (HFR: ...)`, but **still writes the calibrated FITS** --
it does not remove frames from downstream. **Layer B (preprocess FWHM
prefilter):** after Analyze, UI/`night_run` compute an auto FWHM limit
(`app.py:462-480`, draft 517 infolog: **5.362 px**, k=1.5), select passing
frames via DB `calibrated_paths_for_draft_apply_filters`
(`pipeline.py:2368-2428`), mark failures `rejected_prefilter_fwhm` in
`qc_metrics.csv` through `build_prefilter_rejected_map` + in-place QC
(`app.py:817-853`, `night_run.py:241-277`). Alignment and photometry consume
only `status == ok` rows (`filter_files_by_qc_metrics_allowlist`,
`pipeline.py:1248-1264`, gate at `15983`). **Measured on disk: draft 516 and
517 have byte-identical QC artifacts** (0 status/header diffs across 150
frames; both 134 proc CSVs). The DRAFT-517-REVIEW claim that "517 UI rejected
16/150 frames the headless anchor accepted" is **rejected for live 516 vs 517
comparison** -- both drafts share the same 16-frame FWHM prefilter exclusion.
Anchor SHA divergence vs 517 is not explained by a 516-vs-517 QC path split.

## B1 answers (evidence)

### 1. Gate location and authority chain

| Gate | Decision function | Authority value (517) | Ratio mode |
|------|-------------------|----------------------|------------|
| Calibration HFR | `_post_calibration_qc_eval` `pipeline.py:16376-16381` | `config.json` `qc_max_hfr=5.0` (`config.py:1122`, load `1508-1510`); packed in `_qc_pack_from_config` `16213-16214` | `qc_max_hfr_fwhm_ratio=null` in `config.json`; resolver uses legacy px `unit_resolver.py:145-147`. **Note:** `_calibrate_one_light_apply_masters_in_ram` builds `limits` without passing ratio key (`16788-16792`) -- ratio inactive even if set. |
| Preprocess FWHM drop | `calibrated_paths_for_draft_apply_filters` + `qc_metrics.csv` status | Auto FWHM **5.362 px** from infolog line 2371 (`auto_fwhm_enabled`, k=1.5); DB manifest `FWHM` column from Analyze QC | N/A (segmentation FWHM diagnostic in `_qc_enrich_calibrated_in_place`, `18471-18474`: prefilter drives status, not header FWHM reject params) |

No `ui_settings` runtime override observed for `qc_max_hfr` on draft 517.
`draft_provenance.py:437` records `QC_HFR` from manifest for provenance display
only; not the gate authority.

### 2. Path coverage

| Path | Layer A (cal HFR) | Layer B (FWHM prefilter -> qc_metrics) | Rejection removes from photometry? |
|------|-------------------|----------------------------------------|-------------------------------------|
| **(a) UI RUN VYVAR** | Yes -- all 150 calibrations (`infolog` 21 HFR REJECTED logs) | Yes -- preprocess after Auto FWHM (`infolog` 2371, 2381: 134/150 selected) | **Layer B only** -- alignment allowlist 134/150 (`pipeline.py:15983`) |
| **(b) Headless `--full` anchor** | **No** -- photometry-only on frozen snapshot (`session_baseline_check.py:560-639`) | **No** -- snapshot already contains 134 aligned frames + proc sidecars | N/A (inputs pre-filtered at snapshot time) |
| **(c) `night_run` / orchestrator batch** | Yes (shared calibrate path) | Yes -- `_night_run_preprocess_pending` mirrors `app._vyvar_execute_preprocess_pending` (`night_run.py:241-277`) | Layer B via same qc_metrics allowlist |

Layer A failures annotate headers only; Layer B failures set
`rejected_prefilter_fwhm` and exclude from alignment input.

### 3. Why 516 vs 517 diverge (measured)

| Hypothesis | Verdict |
|------------|---------|
| Gate did not run on headless path | **Partially true** for `--full` anchor (photometry-only). **False** for live draft 516 -- it has full qc_metrics + headers. |
| Different limit | **False** between 516 and 517 (identical artifacts). |
| Same limit, different HFR measurement | **False** between 516 and 517 (0 header diffs). |
| Upstream frame set differed | **False** -- both 150 calibrated, 134 proc. |

**Product SHA 517 vs anchor 9902d918** is not explained by 516-vs-517 QC
divergence. Residual drivers (from DRAFT-517-REVIEW, still valid): fresh
MASTERSTAR / census, comp selection (55 vs 60 LCs), era commit stamp
(`b8d5c74` vs `8dea595`).

### 4. The 16 prefilter-rejected frames

Authority for exclusion: FWHM **> 5.362 px** (auto limit). HFR column is
calibration-layer measurement (`VY_QCHFR`); headless `--full` does not
remeasure (frames absent from snapshot aligned/proc tree).

| Frame | qc_metrics FWHM px | VY_QCHFR px | VYQCPASS | infolog HFR reject |
|-------|-------------------|-------------|----------|-------------------|
| BO_CVn_Light_002.fits | 5.255 | 1.915 | true | -- |
| BO_CVn_Light_007.fits | 5.236 | 9.465 | **false** | 9.47 |
| BO_CVn_Light_009.fits | 6.077 | 3.471 | true | -- |
| BO_CVn_Light_049.fits | 5.209 | 1.925 | true | -- |
| BO_CVn_Light_056.fits | 5.292 | 2.107 | true | -- |
| BO_CVn_Light_058.fits | 5.410 | 9.905 | **false** | 9.91 |
| BO_CVn_Light_066.fits | 5.279 | 2.055 | true | -- |
| BO_CVn_Light_074.fits | 5.256 | 2.052 | true | -- |
| BO_CVn_Light_111.fits | 5.254 | 2.040 | true | -- |
| BO_CVn_Light_122.fits | 5.288 | 9.965 | **false** | 9.96 |
| BO_CVn_Light_131.fits | 5.246 | 2.100 | true | -- |
| BO_CVn_Light_141.fits | 5.255 | 2.055 | true | -- |
| BO_CVn_Light_142.fits | 5.242 | 2.118 | true | -- |
| BO_CVn_Light_147.fits | 5.239 | 2.019 | true | -- |
| BO_CVn_Light_149.fits | 5.250 | 2.094 | true | -- |
| BO_CVn_Light_150.fits | 5.233 | 1.960 | true | -- |

**Correction vs DRAFT-517-REVIEW:** review text says "16/150 HFR>5". Measurement:
**16 FWHM prefilter** rejections; **21** distinct HFR calibration failures
(`VYQCPASS=false`); only **3** frames in both sets (007, 058, 122). HFR gate
does not drive the 134-frame photometry subset.

### 5. Blast radius (other QC limits, two-path exposure)

| Limit | Gate location | Runs UI/night_run? | Runs `--full`? | Drops frames? |
|-------|---------------|-------------------|----------------|---------------|
| `qc_min_stars` (10) | `_post_calibration_qc_eval` `16382-16384` | Yes | No | Header only (Layer A) |
| `qc_max_background_rms` (null) | same | If set | No | Header only |
| Auto FWHM prefilter | preprocess / qc_metrics | Yes | No (snapshot) | **Yes** (Layer B) |
| `frame_align_residual_gate` (default OFF) | alignment phase | If enabled | No | Yes when ON |
| Segmentation FWHM/elong in preprocess | `_qc_enrich_calibrated_in_place` | Diagnostic only `18471-18474` | No | No |
| `dao_qc_in_calibrate` PERF-10 | calibrate | Yes | No | No (metrics only) |

Same defect class as 516-02 UI-vs-headless ERR loader split: **two execution
depths** (full pipeline vs photometry-only replay) share one product name.

## Path x gate table

| Path | Gate runs? | Effective limit | HFR / FWHM source | Removes from photometry? |
|------|------------|-----------------|-------------------|--------------------------|
| UI RUN VYVAR | Layer A + B | HFR 5.0 px; FWHM 5.362 px auto | DAO HFR on cal crop; seg FWHM from Analyze DB | Layer B only (134/150) |
| night_run batch | Layer A + B | same as UI | same | Layer B only |
| `--full` anchor | Neither | n/a | Frozen snapshot (134 frames) | Pre-baked at snapshot |

## Phase 2 authority options (no recommendation)

1. **Single shared gate module** both full pipeline and replay call before
   photometry. Consequence: anchor snapshots must record gate version + limits
   or `--full` must re-run gates on raw/calibrated inputs.

2. **Layer B (FWHM prefilter) as sole drop authority**; demote Layer A HFR to
   diagnostic-only (stop logging "REJECTED" unless dropping). Consequence: 21
   HFR-fail frames currently kept in calibrated tree become explicit passes;
   operator confusion reduced.

3. **Layer A HFR as drop authority** wired into qc_metrics / allowlist (align
   with log message semantics). Consequence: drop set changes (21 frames not 16);
   auto FWHM limit becomes secondary.

4. **Freeze gate outputs in snapshot** for anchor replay (`qc_metrics.csv` +
   limits in provenance). Consequence: `--full` compares apples-to-apples but UI
   runs remain non-comparable until replay policy defined (**MS-POOL-POLICY-01**
   adjacent).

5. **Operator-visible single limit** in UI (one slider drives both layers).
   Consequence: requires metric harmonization (HFR vs segmentation FWHM differ
   numerically on same frame -- see table B1.4).

## Rejected premise (standing authority)

**REJECTED:** "Draft 517 UI QC-rejected 16/150 frames that draft 516 headless
accepted." On-disk measurement 2026-08-21: `qc_metrics.csv`, `VYQCPASS`,
`VY_QCHFR`, and proc CSV counts are **identical** between `draft_000516` and
`draft_000517` (134 accepted, 16 `rejected_prefilter_fwhm`). The architectural
split is **full pipeline vs photometry-only anchor**, not 516 vs 517.

## Docs impact (DOCS-SYNC)

| File | Change |
|------|--------|
| `docs/VYVAR_ROADMAP.md` | NEXT SESSION 2026-08-21: FRAME-QC-PARITY-01, MS-POOL-POLICY rescope, MS-QA-DISPLAY, CV-CVN-SKIP, COMP-HISTORY-DB; EMPTY-DAO-01 closed |
| `docs/VYVAR_DECISIONS.md` | 2026-08-20 product model + pool policy; EMPTY-DAO-01 closed |
| `dev/results/CURSOR_RESULT_DRAFT_517_REVIEW.md` | Committed as-is (Part B frame note superseded by this report for mechanism) |

## Errors (if any)

None.

## Files changed

- `dev/results/CURSOR_RESULT_FRAME_QC_PARITY_01.md` (this file)
- `dev/sandbox/frame_qc_parity_01_measure.py`
- `dev/results/context/session_20260821_frame_qc_parity/measurements.json`

STOP -- Phase 2 awaits Milan decision. No code changes authorized.
