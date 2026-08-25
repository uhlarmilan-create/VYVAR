CURSOR RESULT - 2026-08-25 (SEL-GHOST-01 Part B stage 2 / B-STOP-2)

What I did
Wired D1-D4 (S1-S5), retried 520 g_60_4 MASTERSTAR with the AZ800 DB
scale (S6), hardened `--fast --clean` (S7), then sandbox photometry
and `--full`. Push: NO. Live 520/516 read-only. No anchor re-cut.

HEAD after this docs commit is on `main`, ahead of `origin/main`
`b1f5b8c`. Session: `dev/results/session_20260825_sel_ghost_01_b2/`
plus Rule 0.2 copy `dev/results/context/session_20260825_sel_ghost_01_b2/`.

## Premise (Rule 0.1)

**What is compared:** (S1-P1) skip-solve 516 catalog_id set vs the
`c592ecf` control (3581 IDs at
`dev/results/session_20260825_sel_ghost_01_b1b/sandbox_516_c592ecf/`).
(S3-P1) the same set after D1 one-pass radius vs S1. (S5-P1) live 516
comps for the 60 `skip_photometry=False` LC targets, joined to sandbox
MASTERSTAR D3 columns (not the 75-row ensemble dump). (S6) a re-solve
of 520 `g_60_4` using Equipment+Telescope 0.5661 "/px, not the 15.511
FITS/UI Zeiss-wide default. (P-516-2) sandbox photometry from the S3
MASTERSTAR vs live 516 aperture LC / AAVSO / VarAstro. (P-516-3)
`--full` on frozen snapshot 9902d918 / 472bc9e4 (INV-ANCHOR-00: frozen
MS CSV, not the S3 sandbox). Those two 516 photometry comparisons are
not the same experiment.

**How they differ:** D1 radius on 516 is 152.32" (3 x 5.195 px x
9.774 "/px), not the 12" floor. Frozen `--full` MASTERSTAR has no
`vy_identity_gate` / `gaia_dao_resid_px` / `snr`. D3 raises on missing
columns (no silent default).

## Commit table S1-S7

| Commit | Item | What |
|--------|------|------|
| `dc48ece` | S1 D4 | lock reject on 3xFWHM identity threshold |
| `4fc6b8f` | S2 | stamp effective vs `*_config_default` DAO-Gaia tols |
| `b13d393` | S3 D1 | one-pass catalog match radius; 0.70/x1.5 and 0.95/x1.12 loops removed |
| `5d209d7` | S4 D2 | optimizer refit guard + `VY_W0_*` / `.entry.wcs` backup |
| `1cd404a` | S5 D3 | comparison candidacy before RMS ceiling |
| `9550e4f` | S7 | `--fast --clean` worktree gate; `project_root` = `(git toplevel)` |
| `e0f918a` | S6 | pass `telescope_id` into MASTERSTAR plate-scale resolve |
| `24855e0` | S5 | stamp MASTERSTAR `snr` = peak / sky sigma |
| `1ec035f` | S6/S7 | keep DB plate scale over FITS/UI; restore `_LC_OVERVIEW_COLS`; ruff exclude `dev/results` |
| `936512f` | S5 | synthetic fixtures gain D3 columns |

## Predictions

| ID | Verdict | Evidence |
|----|---------|----------|
| S1-P1 | **TRUE** | skip-solve 516 catalog_id set == c592ecf 3581; empty diffs. `s1_516.json` |
| S3-P1 | **FALSE** | `match_sep_effective=152.32"` (not 12.0); cid n=3583 vs 3581. only_s3=`1485911972629595392`,`1497137402233837952`,`1497195367112531712`,`1502034042907960704`; only_c592=`1485912110068525824`,`1504304603139151872`. Tighten-to-4.5" did not stick. `s3_516.json` |
| S5-P1 | **FALSE** | 60 LC targets; 2 ensemble rows fail D3 SNR. Comp `1500579870061241088` snr=6.48 (peak_dao=157.7), DETECTED_P1, gate=ok, resid=0.30 px, on targets `1496998382733052928` and `1497683722074089728`. `s5_p1.json` |
| S6-P1 | **TRUE** | identity gate ok=52 warn=9 = 61 >= 50. Solve rms=1.442 px, n_cat_tri=113, n_det cap=33, accepted (`masterstar_verified=true`). |
| S6-P2 | **FALSE** | 1/8 ghosts carry catalog_id: `1111922300852743808` is `catalog_membership` (not a DET false lock). Other 7 have no cid. |
| S6-P3 | **TRUE** | optimizer n=61 == gate-out; refit rejected (`p95_candidate=2.381 > p95_entry=2.343`, rms_sip=1.33, n=61); first entry wrote `.entry.wcs` + `VY_W0_*` (idempotent later). |
| S6-P4 | **TRUE** | honest d<=3 FWHM = 61 of 61 gate-out. |
| P-516-1 | **FALSE** | same cid delta as S3-P1 (3583 vs 3581). |
| P-516-2 | **FALSE** | 0/60 aperture LC SHA match (positive control: self-hash TRUE, two live files differ TRUE). 7/60 ensembles changed membership; swapped stars all pass D3 (not the SNR=6.48 star). 53 identical ensembles still differ in mag_calib (sandbox MS != live). AAVSO/VarAstro also mismatch. INV-CAL-01 raised after export (`cal_diag` missing in sandbox). `bstop2_516_hash.json`, `p516_2_ensemble_delta.json` |
| P-516-3 | **FALSE** | `--full` OVERALL FAIL at `936512f`. Frozen MS lacks `vy_identity_gate`; D3 raises; `phase2a_empty_comp_drop=14`; core SHA n=93 vs snap n=121 (`6a7bf1fb...` vs `9902d918...`). No re-cut. |
| P-520-1 | **TRUE** | unique selected comp `1111737033143440768`: G=13.870 < 13.9, resid=0.443 px <= 4, snr=51.7 >= 10. Pool after D3 is 7 stars (state drop 373/380). |
| P-520-2 | **FALSE** | lc_rms=0.123 (not 0.05-0.10); lc_rms_ooe=0.069 (not <=0.03). One unique comp. 25 frames loaded. `bstop2_520_phot.json` |
| P-520-3 | **TRUE** | LC `time_base=BJD_TDB`; `_LC_OVERVIEW_COLS` is module-scope again (`1ec035f`). |
| P-520-4 | **FALSE** | Part A 2.5: nonempty cid 111/742=0.150 vs honest 61/742=0.082, abs diff 0.068 >= 0.05. Restricting to DETECTED rows both rates are 1.0. |

## Hash tables (live unchanged)

| Product | SHA256 |
|---------|--------|
| 516 `masterstars_full_match.csv` | `bfa24039778f437b2bf7ed37056b6b507e068d52d2c7b4a222a73002125b250a` |
| 516 `MASTERSTAR.fits` | `13e77cf8a1dcb4e73fae0558437d7234feeb70a5ae4aa85064a8316812b01345` |
| 516 ePSF | `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` |
| 520 `masterstars_full_match.csv` | `5ce9b07fe0490103b2e16f6fbe3b18ffc7cd987fbee8a334722cc2fd46c6a683` |

Positive control for P-516-2: hashing a live LC twice matches; hashing two different live LCs does not.

## `--fast --clean`

At `936512f`: **OVERALL PASS** (1552 passed, 32 skipped; clean-tree
worktree `b1b_clean_648d5248` pytest 31, ruff PASS, pyflakes PASS).
572 s. git-origin-main WARN (ahead of `b1f5b8c`, expected). db-quick-check
WARN waived.

Earlier `--fast --clean` at `24855e0` FAIL: BLE001 on session harnesses
+ F821 `_LC_OVERVIEW_COLS` indented under `_lc_time_axis_title`. Fixed
in `1ec035f`. Then pytest FAIL on synthetic fixtures missing D3
columns. Fixed in `936512f`.

## S6 solver log (first fail, then retry)

First attempt (`s6_520.json` 14:44Z): DB scale 0.5661 then FITS/UI
15.511 overwrote the triangle filter. `n_cat_tri=113`, adaptive cap 33,
`nenasiel som zhodny trojuholnik`. Live SHA unchanged.

Retry after `1ec035f`: WARNING kept DB 0.5661; Starting solve
Scale=0.566 (F=5480 mm, Px=15.04 um, Bin=4). rms=1.44 px, accepted.
Infolog `s6_retry/infolog_20260825_165305.txt`. 111 s.

## P-516-2 / P-516-3 delta (STOP, no re-cut)

D3 predicate on the 7 sandbox-vs-live ensemble swaps: all
`source_state=DETECTED_P1`, `vy_identity_gate=ok`, resid 0.09-1.08 px,
snr 18-352. Membership change is adaptive re-selection on the S3
MASTERSTAR, not D3 rejecting the live comps.

`--full` 14 empty-comp targets (frozen CSV missing D3 columns; D3
raises `vy_identity_gate`). `full-phase0-funnel`
`skip_reason_histogram` includes `no_comps=14`. Core file count 93 vs
snapshot 121.

## Run time (Rule 0.3)

| Step | s |
|------|--:|
| S6 retry solve+MS | 111 |
| `--fast --clean` | 572 |
| 516 sandbox photometry | 1253 |
| 520 V0612 photometry | 37 |
| `--full` | 1576 |

## Errors

- 516/520 sandbox photometry: `INV-CAL-01: cal_diag block missing after dark calibration` after LC export. Products were written; hashes/metrics above use those files.
- `--full` FAIL as P-516-3.
- `vyvar.sqlite3` MASTER_SOURCES still malformed (waived). Do not write live DB.

## Docs impact

DECISIONS SEL-GHOST-01 (F1-F5, D1-D4, retraction). ROADMAP: REG-520-S2
closed; INPUT-PATH-ARCH-01 OPEN; MULTIFILTER-WCS-01 notes the 15.511
overwrite and the successful g_60_4 re-solve; ZONE-SAT-01 MED.
PROCESS: push gate is `--fast --clean`. STATE/JOURNAL one-liners.
INVARIANTS: INV-WCS-01 write guard; INV-MATCH-IDENTITY-01 D4 note.

## Recurrence

existing `dev/tests/test_s3_match_radius_d1.py`, `test_s4_optimizer_d2.py`,
`test_s5_d3_candidacy.py`, `test_s7_clean_tree_name.py`,
`test_inv_match_identity_01.py`. New: `--full` vs D3 missing-column
raise is a documented STOP, not a silent default.

Milan authorizes push after review (architect will request a
SESSION-CLOSE inventory then).
