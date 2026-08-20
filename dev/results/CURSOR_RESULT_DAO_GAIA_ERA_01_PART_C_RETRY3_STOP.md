CURSOR RESULT - 2026-08-19T21:05:00Z

What I did
Implemented DAO-GAIA-ERA-01 Part C retry #3 M-amendments (M1-amend membership
G<=15.0 / census 17.5, M2-amend source_state provenance-only, M-guard
INV-MS-EXPAND-01 fail-loud), updated DECISIONS + harness L1-final/L2 baseline-48.
Restored draft_000516 from snapshot (477dc8cf verified), ran full harness
(6342 s wall). **Overall STOP** — L2 red per directive; live draft restored.

## M-amendments (implemented)

| Item | Change |
|------|--------|
| M1-amend | Expand at G<=15.0; enrich/certificate at G<=17.5 |
| M2-amend | Removed `source_state` gate from `force_eligible_masterstar_mask()` |
| M-guard | Expand fail-loud; `verify_ms_expand_guard()` after cert write |
| L1-final | Baseline 48 subset + additive extras; missing baseline = DEVIATE |
| DECISIONS | M1-M4 block updated with amendments + INV-MS-EXPAND-01 + L1-final |

## MS rebuild (retry #3)

| Metric | Value |
|--------|-------|
| n_detection_in | 2643 |
| n_catalog_added | **+967** (G<=15.0) |
| n_ms_out | **3610** |
| census n | **4990** (17.5 depth) |
| certificate | **PASS** 2.5/2.5, sigma 4.5/4.0 |
| INV-MS-EXPAND-01 | **PASS** |
| comp pool | 2240 (unchanged count) |

## Photometry runtime

| Phase | Time |
|-------|------|
| MS rebuild | 164 s |
| Phase 0 | 0.6 s |
| Phase 1 | 5305 s |
| Phase 2A | 597 s |
| **Total** | **6342 s (~106 min)** |

Product SHA after run: **75f9220e** (n_core=121). Live draft **restored** to 477dc8cf.

## L-table

| Limit | Result | Evidence |
|-------|--------|----------|
| **L1** | **DEVIATE** | 60 LCs; baseline 48 **not subset** — 2 missing (`1497181966814590848`, `1498064771572297856`); 14 additive |
| **L2** | **DEVIATE** | **2/48** pass vs 477dc8cf (median <=2 mmag, max epoch <=10 mmag) |
| **L3** | **DEVIATE** | BO MAD 5.15 mmag (below 0.85 band 6.08); FW 8.75 PASS |
| **L4** | **DEVIATE** | BO offset-XVAL RMS **228.5** mmag (matrix 4.86); FW 2.64 mmag (matrix 1.52) |
| **L5** | **PASS** | Census accounting **100%** (4990/4990); empty-sky inv PASS |
| **L6** | **DEVIATE** | sem/scint/sys non-zero vs baseline (46 LCs compared) |

**Overall: STOP** (L2 red — per directive, no anchor/push).

## L2 per-target ensemble diff (architect review)

Only **2/48** baseline targets within tolerance:

| Target | Name | median d (mmag) | max |epoch| d (mmag) | pass |
|--------|------|-----------------|-------------------|------|
| 1497227287309482624 | Gaia DR3 …2624 | 0.0 | 0.0 | **yes** |
| 1498425548825498112 | ASASSN-V J140619.34+422109.5 | 0.0 | 0.0 | **yes** |

Worst deltas (sample):

| Target | Name | median d (mmag) | max |epoch| d (mmag) |
|--------|------|-----------------|-------------------|
| 1496998382733052928 | ASASSN-V J135313.56+394107.4 | +282.6 | +292.5 |
| 1485534187306501376 | Gaia DR3 …1376 | +193.2 | +199.6 |
| 1496293286541396480 | FZ CVn | +191.9 | +198.0 |
| 1502012464992313088 | Gaia DR3 …3088 | -136.5 | -172.5 |

Full per-target table: `dev/results/context/session_20260819_era01_part_c/part_c_rebuild_l1_l6.json` ? `L2.per_target`.

## L1 detail

- Missing from LC set (no lightcurve file): `1497181966814590848`, `1498064771572297856`
- Additive LCs (14): listed in JSON `L1.targets_additive`
- CV CVn skip: `per_frame_saturation` (OK)

## Interpretation (for architect)

M-guard and census path are **green** (L5 PASS, expand provenance bound). Ensemble
continuity (L2/L3/L4/L6) still fails with large per-target mmag shifts despite
G<=15.0 membership (+967 rows) and unchanged comp-pool **count** (2240). Hypothesis
unchanged: comp **selection** / pool-edge sensitivity or detection-era overlay path
differs from 477dc8cf baseline — not membership row count alone. Next: architect
review with per-target ensemble diff in hand; selection-algorithm sensitivity at
pool edges.

## Errors

None (harness completed; exit code 2 = STOP by design).

## Files changed

- `src_py/masterstar_gaia_accounting.py` — sat handling, `verify_ms_expand_guard()`
- `src_py/pipeline.py` — fail-loud expand, depths 15/17.5, INV-MS-EXPAND-01
- `src_py/forced_photometry.py` — M2-amend: removed source_state force gate
- `docs/VYVAR_DECISIONS.md` — M-amendments + M-guard + L1-final
- `tmp/dao_gaia_era_01_part_c_rebuild.py` — restore, L1-final, L2/L6 baseline-48, expand verify

Raw harness log: `tmp/dao_gaia_era_01_part_c.log`
