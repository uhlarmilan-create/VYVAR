CURSOR RESULT - 2026-08-25 (ANCHOR-DRIFT-01 STOP C1)

What I did
Stage-isolated R1 vs R0 (frozen live 516) without writing live
Drafts. Bisect-style MASTERSTAR reruns at era03 freeze and R1 (and
two in-between science suspects). Named causes. No science-path fix
commit: nothing in-range classified as an unintended defect that
must be patched before C2. C6 re-cut waits Milan GO in chat.

HEAD after C0: `78b3495` on local main, remote `sel-ghost-01`
`78b349598b6fc2fc56ee9c8380fbd0728e614831`. origin/main remains
`b1f5b8c`. Live 516/520 SHA unchanged. Session JSON:
`dev/results/session_20260825_closeout/` (Rule 0.2 copy under
`dev/results/context/session_20260825_closeout/`).

## Premise (Rule 0.1)

**What is compared:** frozen live draft_000516 MASTERSTAR + photometry
products (R0, era03 freeze) versus the B3 T3 R1 full chain at
`c592ecf` (pre-B1). C1c additionally reruns MASTERSTAR at freeze SHA
`ad19e14`, `6fd1452`, `e5a6149`, and `c592ecf` in worktrees.

**How they differ:** R0 is a historical freeze; `--full` never rebuilds
MASTERSTAR (INV-ANCHOR-00 copies the frozen CSV). R1 is a fresh
full chain. B3 T3-P1 reported 0/60 LC SHA equal, 8 ensembles differ,
max |dmag| 44.46 mmag, dRMS -156.8 mmag. This task isolates which
stage moved. C1c worktree expand is not the T3 R1 3606-row census
(see C1c).

## C1a - stage isolation (frozen vs T3 R1)

R0 MS n=3610. R1 MS n=3606. Common catalog_id=3580.

| Stage | n rows differing | Example IDs | Candidate cause |
|-------|------------------|-------------|-----------------|
| catalog_id set | 4 only R0 + 1 only R1 | R0: 1496997386298488832, 1497063283984301696, 1498903316693061248, 1500387696044768384; R1: 1504304603139151872 | census expand membership, not DAO detection set |
| x/y >0.05 px | 12 / 14 | 1485558234828117248 <-> 1485558239123302272 (close-pair ID swap) | greedy match / later D1 (authorized-unrecut) |
| flux / peak_dao | 14 | same swap set | photometry follows the swapped detection |
| mag / ra_deg / dec_deg | 0 | -- | catalog sky coords stable on common IDs |
| phot_g_mean_mag | 14 | same swap set | Gaia G follows the swapped ID |
| source_state | 7 | 1496268959846620544 P1<->P2 | pass label, not a new photometry |
| vy_dao_pass | 14 | same xy set | nan <-> 1/2 with the swap set |
| name | 91 | 1485338783474459648: R0=cid, R1=DET_1394 | pre-B1 F3 DET_ fallback (B1 later on HEAD) |
| vy_identity_gate | missing both | -- | R0 and R1 predate D3 stamps |
| comparison ensembles | 10 (B3 said 8) | 1497169940906156032 8->0; 1500387696044768384 8->0 | MS identity / census / name=DET |
| draft_manifest per-frame QC | 0 listed | -- | keys are draft metadata only, not skip reasons |

Empty-ensemble targets:

- `1497169940906156032`: present in both MS, same x/y/peak; R1
  `name=DET_0784`. Ensemble 8->0. Pre-B1 name vs cid in the
  photometry join (authorized-unrecut; B1 later).
- `1500387696044768384`: only in R0 MS (one of the 4 census-only-R0
  IDs). R1 has no row, so 0 comps.

One-for-one ensemble swaps (same n_comp, different IDs) on six
targets plus 1497284015237511808 (5->6, added 1500418379291011840)
and 1500410236033012352.

## C1b - headline LCs (same ensemble)

BJD nearest-neighbour at 1e-6 d is poisoned by a NaN BJD
(`argmin` on the whole array). Use row index (as B3 T3) or
nearest-finite. `c1_b.json` paired counts are therefore not the
science compare; index-aligned and flag rows are.

### 1498000793739050368 (dRMS -156.8 mmag, dmag 0, ensemble identical)

LC n=134 both. Index-aligned mag_calib identical on overlapping
finites. R1 row 25 `flag=no_data` (BJD and mag NaN). R0 same index
`flag=normal`, mag_calib=10.734, -2635 mmag vs the R0 median. That
one frozen outlier drives lc_rms 0.242 -> R1 0.085. Input: per-frame
target flux / no_data (forced phot kept a bogus epoch in the freeze).
Not an ensemble change.

### 1485987151737107200 (dmag +44.46 mmag, ensemble identical)

Index-aligned mag_inst median abs = 0. mag_calib median abs =
44.46 mmag (= B3 dmag). n finite mag: R0 122, R1 105. R0 has 25
epochs with mag_calib>15 vs R1 17. Input: ZP / combine of the same
comps on a different finite-epoch set (R0 15.x outliers), not a
comp-list swap and not mag_inst.

## C1c - bisect

Era03 freeze git SHA (ledger VL-ANCHOR-WCSINV):
`ad19e14ddb45fdf99a242f6eac97ea39859584e5`. R1 tree: `c592ecf`.

C1c worktrees at `ad19e14`, `6fd1452` (psf x_fit/y_fit), `e5a6149`
(STAGE-01 hand params), and `c592ecf` all wrote the same MASTERSTAR
CSV SHA `2eae423d8a6345b175706161cf25298b31871c0db0e8aa7f4b974750cd7d1216`,
n=2643. Expand raised `No module named 'dao_gaia_stage_01_iter4'`
(that file is not in `c592ecf`; it landed later, B-STOP-1b). The
2643-row file is the pre-expand detection table. No commit in
`ad19e14..c592ecf` moves that table under this harness.

T3 R1 n=3606 because census expand ran in that harness (AppConfig
`max_catalog_rows=100000`). Frozen 3610 vs R1 3606 is census
membership (967 vs 963 added), not a DAO/WCS/pass2 bisect hit.
Live SHA unchanged (`live_unchanged` true).

Suspects from ROADMAP (DAO-GAIA-REGRESS-01, FRAME-QC-PARITY,
masterstar_dao_pass2) did not move the pre-expand CSV. They were
authorized number-changing work that never got a re-cut of the
frozen 3610-row snapshot; they are not an in-range side effect on
the detection table C1c could see.

## C1d - classification (every R1-vs-R0 difference)

| Difference | Named cause | Class |
|------------|-------------|-------|
| name DET_ vs cid (91) | F3 export; B1 later on HEAD | authorized-unrecut |
| x/y/ID swaps (14) | greedy closest-first match; D1 later | authorized-unrecut |
| flux/peak/phot_g on the swap set (14) | follows the swapped detection | authorized-unrecut |
| source_state P1<->P2 (7) | pass label on the swap set | authorized-unrecut |
| vy_dao_pass nan<->1/2 (14) | same | authorized-unrecut |
| catalog_id set 4+1 | census expand membership | unexplained at ID level (mechanism named; C1c did not replay expand) |
| ensembles 10 (not 8) | name=DET join, census-missing target, greedy ID swaps | authorized-unrecut |
| 149800079 dRMS -157 mmag | R1 drops a bogus epoch frozen kept | authorized-unrecut (R1 more correct) |
| 148598715 dmag +44 mmag | ZP on identical mag_inst; different finite epochs | authorized-unrecut |
| draft_manifest frame QC | not present as per-frame skip fields | n/a (nothing to classify) |

Nothing classified **defect-fixed-here**. One row remains
**unexplained** at the ID-membership grain (which 4 R0-only and 1
R1-only census IDs, and why 3610 vs 3606). Architect + Milan decide
whether the re-cut waits on a census-expand replay.

No C1 fix commit. No C6.

## Run time (Rule 0.3)

C1a CSV diffs: seconds. C1b LC: seconds. C1c four skip-expand
MASTERSTAR runs ~60-81 s each; two full-copy attempts ~52-54 s
each (both still failed expand). T3 R1 itself is the B3 chain, not
re-run here.

## Gates

`--fast --clean` OVERALL PASS at `78b3495` (C0c) before this STOP
docs commit. Live 516 CSV
`bfa24039778f437b2bf7ed37056b6b507e068d52d2c7b4a222a73002125b250a`
unchanged. No writes under Archive/Drafts.

## Errors

C1c expand failed in every worktree (`dao_gaia_stage_01_iter4`
absent at freeze and at R1). Partial 2643-row CSV still written.
BJD pairing in `measure_c1b.py` is not usable when any BJD is NaN.

## Docs impact

DECISIONS ANCHOR-DRIFT-01; ROADMAP re-cut waits C6 GO; PROCESS:
full-chain required before any future re-cut. STATE/JOURNAL.

## Recurrence

`--full` cannot catch MASTERSTAR/census drift (INV-ANCHOR-00 copies
the frozen MS). A fresh full-chain run, not photometry-only, is
required before any anchor re-cut. NaN BJD must not go through
`argmin` pairing.

## Files changed

- `dev/results/CURSOR_RESULT_ANCHOR_DRIFT_01.md` (this STOP)
- `dev/results/context/session_20260825_closeout/c1_ab.json`
- `dev/results/context/session_20260825_closeout/c1_b.json`
- `dev/results/context/session_20260825_closeout/c1_c.json`
- `dev/results/context/session_20260825_closeout/c1_c_full.json`
- measure scripts under `dev/results/session_20260825_closeout/`
  (untracked session scratch; JSON copies above are the Rule 0.2 set)
