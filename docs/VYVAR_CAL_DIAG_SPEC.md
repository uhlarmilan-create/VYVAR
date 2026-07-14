# VYVAR -- CAL-DIAG: Calibration-time radiometry gate (spec)

Status: **v1.1 -- APPROVED (Milan, 2026-07-14).** Implementation on main; pending Milan review push.
Date: 2026-07-07. Workstream agreed 2026-07-07 (ROADMAP "IN-FLIGHT -- CAL-DIAG";
DECISIONS "CAL-DIAG" entry).
Grounding: `CURSOR_RESULT_caldiag_flow.md` (read-only live-tree trace) + Cursor spec review
(2026-07-07). All file:line references come from those verified sources, not from memory.

v1.1 changes vs v1.0: gate coverage extended to ALL calibrate paths (review #1);
`cal_diag.json` -> `pipeline_meta.json` handoff specified (#2); fail-closed UX defined (#3);
`bf` sourced from `ProcessedMasterResult.block_factor` (#4); shape-alignment rule (#5);
saturation resolver named (#6); UX channel reuse (#7); deterministic representative frame
(#8); `VY_CDSTAT=WARN` on auto-correct; byte-identity test exclusions; bf==1 log wording.

---

## 1. Purpose

VYVAR lets the user build masters at any binning and calibrate lights at any (higher or
equal) binning. Today only **geometric** adaptation is guarded (shape match, binning ratio,
`MasterResamplingError`; calibration.py:222-229,473-483). **No radiometric check exists**
(confirmed: CURSOR_RESULT_caldiag_flow.md Q6) -- nothing verifies that the resampled dark is
physically consistent with the light before subtraction, and nothing verifies the subtracted
sky is plausible. A camera/driver that AVERAGES on binning (instead of summing charge) would
today produce silently negative calibrated skies and garbage photometry.

CAL-DIAG verifies radiometry **from data, not camera conventions** -- camera-agnostic, no
per-camera hardcoding.

## 2. Scope / non-goals

**In scope (this spec):**
- Check A -- pre-subtraction convention cross-check (SUM vs MEAN dark resample mismatch).
- Check B -- post-dark-subtraction sky-median sanity.
- Provenance flag `dark_resample = SUM | MEAN_AUTOCORRECTED | PASSTHROUGH`.
- Recording of decisions D1-D3 (Section 3) into `VYVAR_DECISIONS.md` (drafted separately,
  same PR as implementation).

**Out of scope (separate ledger items, Section 10):**
- Flat-norm sanity at calibrate (optional Stage 2; hook identified -- first flat cache miss
  in `_calibrate_one_light_apply_masters_in_ram`, pipeline.py:14329-14339 -- but deferred).
- Validity age-clock unification (mtime vs header date).
- RN header=None inconsistency in the SNR helper.
- Any change to master build flow or library precedence (D1/D2 keep current behavior).

## 3. Decisions recorded (Milan, 2026-07-07)

- **D1 -- Master auto-build on import: NOT adopted.** Session `darks/flats` are copied to
  `Raw/darks|flats` only; master stacking stays **manual via CalibrationLibrary UI**
  (`generate_master_*_from_source_dir`, importer.py:1483-1608; ui_calibration_library.py:366,398).
  Documented in DECISIONS as intended behavior.
- **D2 -- Precedence: library wins.** When session raw darks/flats AND a valid scoped library
  master both exist, the **library master** is used for calibration (importer.py:1438-1439);
  session raw is archived, not stacked. Intended behavior, keep.
- **D3 -- DB `READNOISE_E` semantics: bin1 per-pixel.** `RN_eff = RN_db * bin` (exponent 1;
  param_resolver.py:154-159,493-498) is **physically correct for software/digitally binned
  CMOS**: a bin-b superpixel sums b^2 independent reads, so read noise adds in quadrature,
  sigma_sum = RN_px * sqrt(b^2) = RN_px * b. No double-count **provided** the DB stores the
  per-pixel bin1 value (QHY294MM spec RN is per pixel). Empirical confirmation = photon
  transfer, which needs **bin2 flats** (Milan data item, carried from F-BINGAIN-1).
  Implementation: one-line semantic comment at param_resolver.py:155 ("READNOISE_E is
  per-pixel at bin1; scaled *bin for software-summed binning") + mirror in the
  `set_equipment_cosmic_params` docstring (database.py:2928) + DECISIONS entry.

## 4. Physics grounding

Let `L` be a raw light and `D_b` the master dark resampled to the light binning with
effective block factor `bf` (see 5.4 -- taken from `ProcessedMasterResult.block_factor`).

1. **Charge additivity.** A binned light superpixel integrates the charge of bf^2 physical
   pixels; its dark signal (bias + dark current) is the SUM of the per-pixel dark signal.
   Hence dark resample = block SUM (calibration.py:250-251) is correct **when the light
   itself was binned by summing** (standard CMOS software binning).
2. **Convention mismatch signature.** If the acquisition driver AVERAGES on binning, the
   light's dark content is the per-pixel MEAN, while VYVAR subtracts the SUM => the SUM
   overshoots by exactly **bf^2** (>= 4). Since physically
   `median(L) >= median(dark content of L)` (light = dark + sky + stars, sky >= 0),
   observing `median(D_b) > median(L)` is impossible under a matched convention and is the
   detection signature. The bf^2 factor makes the two hypotheses cleanly separable.
3. **Post-subtraction sky.** `median(L - D_b)` estimates the sky level and must be >= 0 up
   to noise. A robust per-pixel noise scale sigma_r (MAD-based) bounds how negative the
   median can legitimately be; a median below `-k * sigma_r` (conservative practical floor;
   the formal noise-of-the-median is far smaller on full frames) indicates broken
   radiometry (wrong master, wrong convention, wrong scaling).

## 5. Gate design

### 5.1 Coverage: ALL calibrate paths (review #1)

The gate logic lives in **one shared helper**, e.g. `_cal_diag_gate_for_obs_group(...)`,
consulted by every loop that applies masters:

| Path | Location | Notes |
|------|----------|-------|
| `calibrate_lights_to_calibrated` -> `_one_sequential` | pipeline.py:14903-14942 | primary; gate before `_calibrate_one_light_disk` |
| `run_draft_ram_calibration_qc_to_obs_files` (RAM QC) | pipeline.py:1925-1978 | must use the same convention/results as disk calibrate -- no divergence |
| `calibrate_batch` / MP workers (`_calibrate_batch_process_one`) | pipeline.py:17009+ | MP MUST NOT bypass the gate |

**MP handling (pick one, both acceptable; no third option):**
(a) run the gate **in the parent before dispatch** for every `(obs_group, dark_path,
light_binning)` key and pass the results/convention dict to workers, or
(b) `cal_diag_gate_enabled` forces `nw = 1`, mirroring the existing
`master_dark_by_obs_key` fallback (pipeline.py:14764-14765).
(a) is preferred (keeps MP speed); (b) is an acceptable v1 simplification.

The decided convention and gate result are stored once per key in / alongside the
`_dark_np_for_calibration_path` cache (pipeline.py:671-690; key already encodes
`path|light_binning|master_binning`) -- the **single source of truth**. Every subsequent
dark load for that key uses the stored convention; no per-frame re-decision, in any path.

### 5.2 Cadence and representative frame (review #8)

Run **once per key** `(obs_group_key, resolved_dark_path, light_binning)` -- not per frame.
Representative frame = **first light path in sorted order** within the obs_group
(deterministic across runs and iteration orders; do not rely on `_iter_fits_recursive`
encounter order). Full-frame medians (cheap, star-robust).

### 5.3 Geometry (review #5)

Gate medians MUST use the same array geometry as the actual calibration: load the
representative light, call `get_processed_master(..., light_shape=data.shape)`, and apply
the same pre-subtraction crop/match step calibrate uses (`_match_and_crop_pair` if it runs
before subtraction, pipeline.py:14307-14308). Comparing an uncropped master against a
cropped light (or vice versa) would false-trigger.

### 5.4 Check A -- pre-subtraction convention cross-check

Inputs: `m_L = median(L_repr)`, `m_S = median(D_b_SUM)` (current convention), and
`bf = ProcessedMasterResult.block_factor` (calibration.py:502-506) -- NOT
`light_bx / master_binning` from headers, because `infer_spatial_block_factor` can raise the
effective binning above header `XBINNING` (calibration.py:115-135,464-467). The gate cache
key/result records this effective `bf` (review #4).

1. If `m_S <= m_L * (1 + cal_diag_rel_tol)` -> **PASS**, convention = `SUM`.
2. Else (impossible under matched convention): compute the MEAN-equivalent
   `m_M = m_S / bf^2`.
   - If `bf > 1` and `m_M <= m_L * (1 + cal_diag_rel_tol)` and
     `cal_diag_autocorrect_enabled` -> **AUTO-CORRECT**: recompute the dark with block MEAN
     for this key, log **ERROR-level, loud**, convention = `MEAN_AUTOCORRECTED`,
     `VY_CDSTAT = WARN` (auto-correct is an anomaly, not a clean PASS), then re-run Check A
     on the corrected dark (must now PASS, else fall through to 3).
   - Else -> **FAIL-CLOSED** for this obs_group (see 5.6).
3. `bf == 1`: the mean-retry is unavailable (no resample happened); `m_S > m_L` then means
   **wrong master pairing, hot dark, or a non-binning scaling error** -> FAIL-CLOSED
   directly. Log wording must reflect these causes -- NOT "binning convention mismatch"
   (review minor note).

Implementation note: MEAN recompute must NOT duplicate resample logic -- route a dark-mode
override (e.g. `dark_resample_mode`) into `resample_master_to_light_binning`
(calibration.py:199), which already implements block MEAN for flats. One shared core, per
PROCESS.

### 5.5 Check B -- post-subtraction sky-median sanity

On the representative frame, after subtracting the (possibly auto-corrected) dark:

- `s = median(L_repr - D_b)`; `sigma_r = 1.4826 * MAD(L_repr - D_b)`.
- **HARD FAIL (fail-closed, this obs_group):** `s < -cal_diag_hard_sigma * sigma_r`
  (default 5.0). Radiometry broken beyond any noise explanation.
- **WARN (continue):** `-cal_diag_hard_sigma * sigma_r <= s < 0` -- legitimate near-zero sky
  (short exposure / dark site / narrowband) can dip below zero within noise; log WARNING
  with `s`, `sigma_r`.
- **WARN (continue):** `s > cal_diag_sat_warn_frac * saturation_adu` (default 0.90).
  Saturation resolved via the existing resolver chain --
  `_effective_saturation_limit` / `resolve_saturation` (pipeline.py:5128+,
  param_resolver.py:571+) with the `equipment_id` from the calibrate/QC pack. **If
  saturation is unresolvable for the equipment, skip this warn entirely** (review #6).
- Else **PASS**.

### 5.6 FAIL-CLOSED observable behavior (review #3)

When a key fails Check A (no valid hypothesis) or Check B HARD FAIL:

- **skip `_calibrate_one_light_disk` for ALL frames of that obs_group**, including the
  representative frame;
- **no output** under `calibrated/lights/<obs_group>/` -- delete any partial output if the
  failure is detected after frames were written (should not happen with the pre-loop gate,
  but the invariant must hold);
- increment `stats["cal_diag_aborted_groups"]` (surfaced in the calibrate summary /
  progress reporting);
- exactly **one ERROR per key**, labeled with the obs_group, carrying `m_L`, `m_S`, `m_M`,
  `bf`, `s`, `sigma_r` as applicable;
- **surviving obs_groups continue** (per-set fault isolation, same discipline as the
  2026-06-14 solver gate).

### 5.7 UX channel (review #7)

ERROR/WARN surfacing reuses the existing pipeline UI patterns -- `_pipeline_ui_error` and
the job footer state (pipeline.py:757-768) + infolog. No new ad-hoc channel.

### 5.8 PASSTHROUGH

The existing no-dark path (pipeline.py:14786-14798, `VY_CALIB=PASSTHROUGH`) skips both
checks; provenance = `PASSTHROUGH`.

## 6. Provenance outputs

Written on **every calibrated light** of the obs_group (values decided once per key):

| Header | Value | Meaning |
|--------|-------|---------|
| `VY_DKRSMP` | `SUM` / `MEAN_AUTOCORRECTED` / `PASSTHROUGH` | dark resample convention actually applied |
| `VY_CDSKY` | float (ADU) | gate-time post-subtraction sky median (representative frame) |
| `VY_CDSTAT` | `PASS` / `WARN` / (absent on abort) | gate outcome; `WARN` also on AUTO-CORRECT |

**Meta handoff (review #2):** `pipeline_meta.json` is merged at photometry time
(photometry_core.py:5318), not at calibrate. Therefore: write
`archive/<draft>/cal_diag.json` (per-key results: convention, status, medians, bf,
aborted flag) at the end of `quick_calibrate_last_import` /
`calibrate_lights_to_calibrated`; Phase 2A merges it as an additive `cal_diag` block when it
builds `pipeline_meta.json`. Missing `cal_diag.json` (old drafts, gate OFF) => block simply
absent -- no error.

Plus one structured infolog line per key (PASS at INFO; WARN/AUTO-CORRECT/FAIL at ERROR
level with all numbers).

## 7. Config keys (register in `VYVAR_PARAMS.md`; config<->UI parity per PROCESS)

| Key | Default | UI | Notes |
|-----|---------|----|-------|
| `cal_diag_gate_enabled` | **ON** | Settings (exposed) | Master switch. ON is the point of the gate; on matched-convention rigs it is a pure no-op on science outputs. |
| `cal_diag_autocorrect_enabled` | ON | hidden (config-only) | OFF = mismatch always fail-closed. |
| `cal_diag_rel_tol` | 0.02 | hidden | Relative slack on `m_S <= m_L` (median noise + rounding). |
| `cal_diag_hard_sigma` | 5.0 | hidden | Check B hard floor in sigma_r units. |
| `cal_diag_sat_warn_frac` | 0.90 | hidden | Check B saturation-proximity warn; skipped when saturation unresolvable. |

Clamps: `rel_tol` [0, 0.2]; `hard_sigma` [3, 10]; `sat_warn_frac` [0.5, 1.0].

## 8. Do-no-harm and validation (Definition of Done)

1. **Unit tests (synthetic):**
   - matched SUM convention (bin1 master, bin2 summed light): PASS, `VY_DKRSMP=SUM`;
     calibrated **image arrays and pre-existing headers** byte-identical to gate-OFF --
     the comparison **excludes** the new `VY_DKRSMP`/`VY_CDSKY`/`VY_CDSTAT` keys and
     `cal_diag.json` (review minor note);
   - averaged-driver light (dark content = per-pixel mean): Check A triggers,
     AUTO-CORRECT to MEAN, `VY_CDSTAT=WARN`, corrected output matches analytic expectation;
   - garbage dark (wrong camera; m_S >> m_L, mean-retry also fails): FAIL-CLOSED, only that
     obs_group aborted, siblings calibrate, `cal_diag_aborted_groups` incremented, no files
     under the aborted group's `calibrated/lights/<obs_group>/`;
   - bin1 pairing error: FAIL-CLOSED without mean-retry, log names pairing/scaling causes;
   - near-zero sky with noise: WARN, not fail;
   - PASSTHROUGH path: checks skipped, provenance written;
   - **path-coverage test:** RAM-QC path and (if variant (a)) MP path consult the same gate
     result for the same key -- no divergence, no bypass.
2. **Real regression (draft_424, `run_full_photometry_pipeline`):** gate ON --
   photometry outputs **byte-identical** to current baseline (headers/`cal_diag.json`
   additive only); `VY_DKRSMP=SUM` on all frames; 0 WARN/FAIL expected.
3. Gate OFF -> bit-for-bit current behavior; **zero** CAL-DIAG headers and no
   `cal_diag.json` written.
4. `pytest tests/` green; `ruff` BLE001/E722 clean; PARAMS registry updated; 0 PDF overflow
   (no PDF change expected).

## 9. Explicitly resolves / advances

- **F-BINGAIN-1 RN sub-question** -- direction resolved by D3 (no double-count under the
  bin1-per-pixel DB semantic); final empirical closure = photon transfer on **bin2 flats**
  (Milan data blocker, unchanged).
- The ROADMAP CAL-DIAG entry (spec was pending from Claude) -- this document.

## 10. New ledger items discovered by the flow trace (NOT part of this gate)

| ID | Sev | Finding |
|----|-----|---------|
| CAL-AGE-CLOCK | MED | Two validity age clocks: import scan uses filesystem **mtime** (importer.py:873-879) while the library UI uses header capture date (`get_master_age_days`, calibration.py:79-98). Copying a library to another machine resets mtime and revives expired masters. Proposal: unify on header `VY_CDATE`/`DATE-OBS` with mtime fallback. |
| RN-HEADER-NONE | LOW/MED | `photometry_core.py:1211` (SNR aperture table helper) calls `resolve_read_noise(header=None)` -> DB RN unscaled by binning, inconsistent with Phase 2A (header passed, photometry_core.py:6664-6665). Small fix: pass the light/masterstar header. Affects SNR-optimal aperture planning on binned data, not the LC error model. |
| CAL-PASSTHRU-DEAD | LOW | `allow_passthrough` synthetic master in `get_processed_master` (calibration.py:441-452) has no production caller -- remove or mark test-only. |

---

*Discipline reminders for implementation (PROCESS): shared core (Check A's MEAN retry goes
through `resample_master_to_light_binning`; the gate itself is ONE helper for all three
calibrate paths); config<->UI parity; byte-identity check on photometry outputs; Cursor
task text in English; commit + JOURNAL entry at session close.*
