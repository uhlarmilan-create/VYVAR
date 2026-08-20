CURSOR RESULT - 2026-08-19 (DAO-GAIA-ERA-01 Part C STOP)

What I did
Executed Part C full clean rebuild on tip `b8d5c74c3a2d`: MS catalog
redetection (certificate **PASS** 2.5/2.5, sigma 4.5/4.0 on production
path) -> zone annotation -> comp pool regen -> Phase 0+1+2A (PFS ON,
export_err_mode=calibrated). Evaluated L1-L6 vs 477dc8cf baseline.
**Overall STOP** (L2-L6 DEVIATE). Live `draft_000516` restored from
`draft_000516_snapshot_cleanrebuild_20260818` (SHA guard 477dc8cf).

## Rebuild (executed)

| Phase | Runtime |
|-------|---------|
| MS build + certificate | 34 s |
| Phase 0 | 0.26 s |
| Phase 1 | 2499 s |
| Phase 2A | 451 s |
| **Total** | **3209 s** |

Harness: `tmp/dao_gaia_era_01_part_c_rebuild.py`
Raw: `dev/results/context/session_20260819_era01_part_c/part_c_rebuild_l1_l6.json`

### MS + certificate (production path)

| Metric | Before | After rebuild |
|--------|--------|---------------|
| masterstars rows | 2612 | **2644** |
| comparison_stars | 2241 | 2241 |
| census rows | — | **4131** |
| Certificate | — | **PASS** match/centroid **2.5/2.5 px**, sigma **4.5/4.0** |

Census states: DETECTED_P1 2157, DETECTED_P2 367, FORCED_SEED 123,
SEED_REJECTED 672, BLENDED 221, TOO_FAINT 511, EDGE 80.

### Product SHA (rebuild; superseded by restore)

| | SHA prefix | n |
|---|------------|---|
| Before (477dc8cf era) | 477dc8cf | 97 |
| After rebuild | **541b3f57** | 97 core / 145 ext |

## L1-L6 evaluation

| Limit | Result | Key evidence |
|-------|--------|--------------|
| **L1** | **PASS** | 48/48 LCs; CV CVn `per_frame_saturation` |
| **L2** | **DEVIATE** | 2/48 targets pass; median \|delta\| <= 2 mmag AND max epoch <= 10 mmag vs 477dc8cf |
| **L3** | **DEVIATE** | BO check MAD **8.74 mmag** (anchor 7.15, band 6.08-8.22); FW **7.58** (PASS) |
| **L4** | **DEVIATE** | BO XVAL RMS mis-merge (8484 mmag — eval bug); FW tbl path missing in running harness |
| **L5** | **DEVIATE** | Census completeness above depth **75.7%** (gate >= 99%); empty-sky audits PASS |
| **L6** | **DEVIATE** | `err_sem_rel` max abs vs baseline **0.052 mag** (sem/scint/sys must be 0) |

Provenance: 2 stale mtimes (`dao_gaia_calibration.json`, `gaia_source_state_census.csv`
not rewritten on this run — pre-run A-fix 4 artifacts).

### L2 binding failures (sample)

| Target | Name | \|median delta\| mmag | max epoch mmag |
|--------|------|------------------------:|---------------:|
| 1498613634033133184 | BO CVn | 62.5 | 77.4 |
| 1497343732462852864 | FW CVn | 243.1 | 249.8 |
| 1496293286541396480 | FZ CVn | 177.9 | 183.2 |
| 1497169940906156032 | CSS_J134925.3+393524 | 81.7 | 89.9 |

4 targets missing LC vs baseline (`1496733984545821696`, `1496037650087948160`,
`1485560025830226432`, `1497491273179203456`).

L2 pass count: **2/48** (NSV 19982, ZTF J140300.57+413339.3 only).

## Recovery

Live `draft_000516` **restored** from snapshot (477dc8cf n=97 verified).
Rebuild product SHA `541b3f57` exists only in context JSON / log — not live.

## Not executed (per STOP spec)

- Supersede chain / new anchor SHA registration
- Anchor + P1 golden recut (516-04)
- `--fast` / `--full` session baseline
- BO/FW exports + SUBMIT-01
- Docs (STATE/ROADMAP/DECISIONS/JOURNAL/PARAMS/ledger/CHANGELOG)
- Push

## Architect review needed

Full MS redetection with certified 2.5/2.5 tolerances **passes the
certificate** but **fails bounded-physics L2** (systematic ZP shifts
50-250 mmag on named CVs) and **L5 completeness** (75.7% vs 99% gate).
L6 sem term also moved.

Possible paths (Milan decision):

1. **Overlay-only MS path** (keep detection table; apply Gaia census /
   accounting without row loss) then photometry-only rebuild — retest L2.
2. **Revise pre-registered L2/L5 gates** if membership change is accepted
   as new-era physics (explicit authorization; new anchor SHAs).
3. **Investigate L5** — census completeness drop driven by G 14-15 bins
   (34% at G 14.5-15.0); may couple to seed/pass2 policy vs depth gate.

Push: **not authorized** (STOP; awaiting Milan review).

## Files changed

- `tmp/dao_gaia_era_01_part_c_rebuild.py` (harness, new)
- `dev/results/context/session_20260819_era01_part_c/part_c_rebuild_l1_l6.json`
- This file

Live Archive restored; no git commits.
