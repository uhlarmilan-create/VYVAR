CURSOR RESULT - 2026-08-20T07:45:00Z

What I did
Implemented production pinned-ensemble mechanism (Part A), generated
`dev/validation/pinned_ensembles.csv` (48 targets / 333 comp rows, SHA
8431934e0c30eee9), wired Phase-1 pin path + INV-PIN-03 provenance, and ran
ERA-03 full rebuild harness (MS + Phase 0/1/2A on live draft_000516).
Fixed three pin-path bugs found during execution (RMS API, index order,
anchor comp_pt overlay for AC). Acceptance panel A-T0..A-CEN evaluated STOP;
Parts C/D (anchor recut, exports, docs commit series, push) not executed per
pre-registered STOP rule.

## Output / findings

### Part A — Pin mechanism (production)
- `src_py/pinned_ensembles.py` — load/validate/select, INV-PIN-01/02/03 helpers
- `dev/validation/pinned_ensembles.csv` — 477dc8cf ensembles for 48 LC targets
- `src_py/photometry_core.py` — Phase-1 pin dispatch, meta provenance, comp_provenance sidecar, pinned `ct_n_comp` overlay
- `dev/tests/test_pinned_ensembles.py` — 7 tests PASS

### Part B — Rebuild + acceptance panel (STOP)

| Tier | Result | Notes |
|------|--------|-------|
| A-T0 | PASS | Cert 2.5/2.5, sigma 4.5/4.0; census 4990 @ 100%; INV-MS-EXPAND-01 PASS |
| A-T1 | DEVIATE | BO XVAL 4.858 mmag (matrix 4.86) PASS; FW 1.524 (1.52) PASS; FW check MAD 6.22 mmag below band 6.97–9.43; GH XVAL harness has no AIJ reference |
| A-T2 | DEVIATE | 44/48 shape pass (?10 mmag residual); 4 missing LC |
| A-MAG | DEVIATE | 0/48 full-file byte identity (after fixes: BO/FW mag_calib_final median ?=0.0 mmag; residual diff is metadata e.g. ct_n_comp before last fix) |
| A-L1 | DEVIATE | 4 baseline targets missing LC (see below) |
| A-CEN | PASS | Census/cert mtime post-run |
| INV-PIN-03 | PASS | Pin SHA in pipeline_meta |

**Product SHA (live draft_000516, final retry):** `0c4dadbf16f2bee4` (n=29 core photometry hash scope)
**Supersede intent:** 477dc8cf ? 0c4dadbf (not ratified — STOP)

**MS rebuild (new era):** 2643 detection + 967 catalog ? 3610 MS rows; census 4990; cert PASS

**Pin execution (final retry):** 317 pinned + 196 default comp_pt rows; 58 LCs total

### STOP mechanisms (named)

1. **Four pinned targets — full ensemble color re-validation fail**
   - `1496795041799526400`: all 8 pinned comps dropped `color_ceiling_violation` ? no Phase-1 comps ? no LC
   - `1497181966814590848`: got 3 comps but Phase-2A skip `zone_noise`
   - `1497350638770267520`, `1498064771572297856`: missing LC (see panel JSON)
   - Rule behavior is correct (fail-loud); MS-POOL-POLICY-01 refill deferred per task

2. **A-T1 FW check MAD** below anchor band (6.22 vs 6.97–9.43 mmag) despite XVAL matrix PASS

3. **A-MAG** full-file byte identity blocked primarily by field-wide `ct_n_comp` metadata (2345 vs 2229) on otherwise identical BO mag columns; overlay fix added for next run

4. **Initial pin bugs (fixed):** wrong `_detrend_and_compute_comp_rms_map` kwargs; `_dist_deg` computed after `set_index`; missing anchor `contamination_idx` broke AC (`insufficient_ref_stars`)

### ERA-02 alignment
Pinned Phase-1 + anchor comp_pt overlay reproduces BO AC (`delta_m_corr=-0.129885`) and BO/FW XVAL at matrix after fixes — confirms selection-only + metadata path diagnosis from ERA-02.

### Artifacts
- Panel JSON: `dev/results/context/session_20260819_era03/era03_acceptance_panel.json`
- Logs: `tmp/dao_gaia_era_03_close.log`, `tmp/dao_gaia_era_03_retry.log`
- Harness: `tmp/dao_gaia_era_03_close.py`, `tmp/dao_gaia_era_03_retry.py`

### DELETE-OK candidate (do not delete until Milan confirms after green anchor)
- `Archive/Drafts/draft_000516_era_candidate`

## Errors (if any)
Pre-registered STOP — no silent failure. Live draft_000516 remains on new-era MS + partial pinned product (not restored to 477dc8cf per era-cut instruction).

## Files changed
- `src_py/pinned_ensembles.py` (new)
- `src_py/photometry_core.py` (pin hook, provenance, ct_n_comp overlay)
- `dev/validation/pinned_ensembles.csv` (new)
- `dev/tests/test_pinned_ensembles.py` (new)
- `dev/results/CURSOR_TASK_DAO_GAIA_ERA_03_CLOSE.md` (new)
- `dev/results/context/session_20260819_era03/era03_acceptance_panel.json` (harness output)

## Not done (STOP gate)
- Part C: anchor/P1 golden recut, session_baseline_check --fast/--full
- Part D: AAVSO/VarAstro exports, docs series, git separable commits, push authorization

## Recommended next steps for Milan/architect
1. **Policy call on 4 color-fail targets:** accept DROP + report vs widen immutable color ceiling for pinned re-validation only
2. **Re-run** `tmp/dao_gaia_era_03_retry.py` after latest ct_n_comp overlay (may clear A-MAG for ~44 targets)
3. **FW MAD band:** confirm whether below-band improvement is acceptable under ERA-ACCEPT T1
4. If panel green ? proceed Part C/D per task

## Runtime (Part B final retry, seconds)
- Phase 0+1: ~1510
- Phase 2A: ~335
- Total harness: ~2330

Push: **not authorized** (STOP; single push request withheld pending panel green)
