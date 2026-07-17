CURSOR RESULT — 2026-07-16T10:45:00+02:00

## Milan sign-off (verbatim)

Per calibrated light frame, fit a 2D polynomial surface of order 2 to the background
(source-masked + sigma-clipped fit), subtract it in the shared `calibrated → processed` step
used by BOTH chains. Config: `preprocess_sky_surface_order` (int, 0 = off, default **2**).
Subtract the full fitted surface including the constant term.

## What I did

Implemented T3-RESTORE FIX A (order-2 sky surface in shared preprocess) and FIX B
(`git_dirty_code` classifier + anchor gates). Ran focused pytest + draft_429 empirical
validation. **Did not run Anchor #3** — gated on Milan post-push UI HEALTHY classification.

## T2 verdict (ledger)

**draft_434** (clean UI on baseline `1e2e8d6`): **UI-SICK** — n_raw_dao=8927, ms 6699/3993,
sky 1564.9, identity p95=6.24 px, targets 185. Stamp fix confirmed (catalog_id join=199).
Dirty-file provenance live; under FIX B classification draft_434 dirt is **scratch-only**
(`git_dirty_code=false`).

## Output / findings

### FIX A — preprocess sky surface

- Core: `pipeline._fit_subtract_preprocess_sky_surface` → `_preprocess_calibrated_one` →
  `preprocess_calibrated_to_processed` (UI + `night_run` both route here).
- Config: `preprocess_sky_surface_order` default **2** (`config.py`, `VYVAR_PARAMS.md`).
- Provenance: `VYSKYORD`, `VYSKYP2P` FITS headers; preprocess row stats
  (`sky_surface_p2p_adu`, `sky_surface_n_fit_pixels`, …).
- Fit convention: star-masked + sigma-clipped; calm background `|cal−median|<100 ADU`; fit
  order-N to `(median − cal)`; subtract full surface.

**Empirical acceptance (draft_429 Light_008 vs MS):** `scripts/t3_validate_sky_surface_429.py`
→ `tmp/t3_sky_surface_429/acceptance.json`

| Metric | Result | Criterion |
|--------|--------|-----------|
| DAO pass-1 sim | **2579** | 2500–3000 band ✓ (429 logged 2816; ~237 gap documented) |
| Smooth residual p99 | **173 ADU** | < 200 ADU ✓ (oracle ~52 ADU) |
| Frame median Δ | +0.56 ADU | meta sky ~1478 is Labbe annulus, not frame median |
| maxabs residual | 16658 ADU | star cores; smooth field p50=81 ADU |

### FIX B — dirty-gate refinement

- `photometry_core.classify_git_dirty_paths` + `git_dirty_code` / `_code_files` /
  `_scratch_files` in `pipeline_meta` provenance.
- `scripts/anchor_pair_run.py` + `anchor_pair_430_431.py`: anchor gates on
  `git_dirty_code` only; scratch dirt allowed at run start.

### Anchor #3 — NOT RUN

Awaiting Milan clean UI RUN VYVAR on pushed commit. Expected: cal≠proc, census
~2875-class, sky ~1478, identity p95 < 1 px, `git_dirty_code=false`, A-durable watch
during alignment.

### Closeout tickets

| Ticket | Status |
|--------|--------|
| F-431-HEADLESS-DIVERGENCE | **CLOSED** (root cause: lost preprocess; shared core restored) |
| F-428 arc | **CLOSED** in STATE/DECISIONS (WCS-INV live; census driver = preprocess) |
| A-durable | Pending Milan validation |
| Anchor #3 | **BLOCKED** until HEALTHY |

## Errors (if any)

None blocking commit.

## Files changed

- `pipeline.py` — sky surface subtract in preprocess
- `config.py` — `preprocess_sky_surface_order`
- `photometry_core.py` — `git_dirty_code` classifier + provenance
- `scripts/anchor_pair_run.py`, `scripts/anchor_pair_430_431.py` — gates
- `scripts/t3_validate_sky_surface_429.py` — empirical acceptance
- `tests/test_preprocess_sky_surface.py`, `tests/test_git_dirty_code_classifier.py`
- `tests/test_f431_labbe_provenance.py` — `git_dirty_code` assert
- `docs/VYVAR_DECISIONS.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_PARAMS.md`
- `CURSOR_RESULT_t3_restore_anchor.md`
