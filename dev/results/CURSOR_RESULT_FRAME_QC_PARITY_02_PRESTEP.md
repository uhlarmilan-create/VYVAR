CURSOR RESULT - 2026-08-21 (FRAME-QC-PARITY-02 pre-step)

What I did
Named the actual Layer B prefilter decision inputs and closed the 134/16 replay
on draft 517 (and 516). Measurement only; no `src_py/` changes.

## Part 0 - PUSH status

**Not executed.** After `git fetch`, `origin/main` = **8dea595**; local tip =
**9494c44** (12 commits ahead). Part D/E blocked on Milan push + architect
review of this report.

---

## Named mechanism (one paragraph)

Layer B prefilter exclusion uses **manifest `inspection.fwhm`** from Analyze
(DAO moment-median FWHM on a central crop, `pipeline.py:1493-1548`,
stored via manifest `files[].inspection.fwhm`, read as column `FWHM` in
`fetch_draft_light_rows_for_quality` / `draft_provenance.py:492`). The auto
limit **5.362 px** is `median(manifest_fwhm) + k * sigma_MAD` with k=1.5
(`compute_auto_fwhm_limit`, `photometry_core.py:13640-13677`; computed in
`app.py:462-480` on Analyze rows). `calibrated_paths_for_draft_apply_filters`
(`pipeline.py:2408-2413`) keeps rows with **`FWHM <= limit`** (strict, all 150
non-null). **`qc_metrics.csv` column `fwhm_px` is a different measurement**
(preprocess in-place segmentation FWHM written during `qc_enrich_calibrated_lights_in_place`;
infolog lines like `QC in-place ... FWHM=5.25` refer to this, not the filter
authority). Phase-1 table used `qc_metrics` FWHM against the manifest-derived
limit -- hence frames with qc FWHM 5.21-5.29 px but manifest FWHM 5.36-5.48 px
appeared "below limit" while correctly rejected.

---

## B1 answers

### 1. Which FWHM value the DB filter compares

| Source | Column | Typical value (frame 002) | Used in filter? |
|--------|--------|---------------------------|-----------------|
| Manifest / DB | `inspection.fwhm` -> `FWHM` | **5.408 px** | **YES** |
| qc_metrics.csv | `fwhm_px` | 5.255 px | No (diagnostic / status label only) |

Authority chain: `app.py:465-476` -> `compute_auto_fwhm_limit(manifest FWHM)` ->
`calibrated_paths_for_draft_apply_filters(..., fwhm_max_px=5.362)` `817-822`.

Full dump: `dev/results/context/session_20260821_frame_qc_parity_02/draft_000517_fwhm_dump.csv`

### 2. Limit value in force (draft 517 infolog)

Infolog `2371`: `Auto FWHM limit=5.362 px (k=1.50)`.

Reconstruction from manifest FWHM (all 150 lights):

| Statistic | Value |
|-----------|-------|
| median_fwhm | **5.311 px** |
| MAD | 0.0231 |
| sigma_MAD (MAD * 1.4826) | 0.0342 |
| auto_limit = median + k * sigma_MAD | **5.311 + 1.5 * 0.0342 = 5.362 px** |
| n_kept (compute_auto_fwhm_limit internal) | 134 |
| n_cut | 16 |

Population: all 150 calibrated lights at Analyze time (before Layer B filter).

### 3. Decision replay (closed)

Replay rule: `manifest_fwhm_px <= 5.362` -> accept.

| Draft | Replay ok | Actual ok | Mismatch |
|-------|-----------|-----------|----------|
| draft_000517 | **134** | **134** | **0** |
| draft_000516 | **134** | **134** | **0** |

Sandbox: `dev/sandbox/frame_qc_parity_02_prestep.py` ->
`dev/results/context/session_20260821_frame_qc_parity_02/summary.json`

### 4. The discrepancy (plain)

Phase-1 reported `qc_metrics.fwhm_px` against limit 5.362. The filter never
reads that column; it reads **manifest FWHM** which is ~0.10-0.20 px **higher**
than qc_metrics FWHM on the same frames (different algorithm: Analyze DAO
moment median vs preprocess segmentation). Example **BO_CVn_Light_002**: manifest
**5.408** > 5.362 -> rejected; qc_metrics **5.255** < 5.362 -> looked wrongly
accepted if qc column were the authority.

---

## 16 rejected frames (manifest FWHM vs qc_metrics FWHM)

Authority: manifest FWHM **> 5.362 px**.

| Frame | manifest FWHM | qc_metrics FWHM | status |
|-------|---------------|-----------------|--------|
| BO_CVn_Light_002.fits | 5.408 | 5.255 | rejected_prefilter_fwhm |
| BO_CVn_Light_007.fits | 5.384 | 5.236 | rejected_prefilter_fwhm |
| BO_CVn_Light_009.fits | 6.185 | 6.077 | rejected_prefilter_fwhm |
| BO_CVn_Light_049.fits | 5.373 | 5.209 | rejected_prefilter_fwhm |
| BO_CVn_Light_056.fits | 5.476 | 5.292 | rejected_prefilter_fwhm |
| BO_CVn_Light_058.fits | 5.484 | 5.410 | rejected_prefilter_fwhm |
| BO_CVn_Light_066.fits | 5.380 | 5.279 | rejected_prefilter_fwhm |
| BO_CVn_Light_074.fits | 5.376 | 5.256 | rejected_prefilter_fwhm |
| BO_CVn_Light_111.fits | 5.368 | 5.254 | rejected_prefilter_fwhm |
| BO_CVn_Light_122.fits | 5.448 | 5.288 | rejected_prefilter_fwhm |
| BO_CVn_Light_131.fits | 5.377 | 5.246 | rejected_prefilter_fwhm |
| BO_CVn_Light_141.fits | 5.367 | 5.255 | rejected_prefilter_fwhm |
| BO_CVn_Light_142.fits | 5.419 | 5.242 | rejected_prefilter_fwhm |
| BO_CVn_Light_147.fits | 5.394 | 5.239 | rejected_prefilter_fwhm |
| BO_CVn_Light_149.fits | 5.364 | 5.250 | rejected_prefilter_fwhm |
| BO_CVn_Light_150.fits | 5.367 | 5.233 | rejected_prefilter_fwhm |

Accepted sample (manifest <= limit, qc may differ):

| Frame | manifest FWHM | qc_metrics FWHM |
|-------|---------------|-----------------|
| BO_CVn_Light_001.fits | 5.226 | 5.305 |
| BO_CVn_Light_008.fits | 5.250 | 5.250 |
| BO_CVn_Light_010.fits | 5.255 | 5.255 |

---

## Part C acceptance

- Replay closes exactly **134/16** on draft 517 (and 516).
- No code changes under `src_py/`.

## Docs impact

None yet (measurement only). Part D provenance stamp must record:
**compared quantity = manifest inspection.fwhm (Analyze DAO moment median)**,
**limit = 5.362 px (median + 1.5 * sigma_MAD, k=1.5)**, not qc_metrics fwhm_px.

## Errors

None.

## Files changed

- `dev/results/CURSOR_RESULT_FRAME_QC_PARITY_02_PRESTEP.md` (this file)
- `dev/sandbox/frame_qc_parity_02_prestep.py`
- `dev/results/context/session_20260821_frame_qc_parity_02/summary.json`
- `dev/results/context/session_20260821_frame_qc_parity_02/draft_000516_fwhm_dump.csv`
- `dev/results/context/session_20260821_frame_qc_parity_02/draft_000517_fwhm_dump.csv`

**STOP** -- await Milan push (Part 0) and architect review before Part D.
