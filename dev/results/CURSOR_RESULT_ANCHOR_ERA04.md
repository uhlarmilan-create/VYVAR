CURSOR RESULT - 2026-08-26T13:50:00Z

What I did
C6 ANCHOR RE-CUT era04 (Milan GO 2026-08-26), C6-0 final through C6-3.
R1'' informational (declared shim); C6-1 full-chain products written
into a new era04 snapshot; C6-3 STOP (unnamed mag_calib motion vs
era03). era04 is NOT locked. era03 untouched. origin/main stays
7c086e8. No PUSH_AUTH SHA.

## Premise (Rule 0.1)
Compared: (C6-0) isolated c592ecf + A files + one declared 3-tuple
shim versus T3 R2. (C6-1/C6-3) HEAD full-chain 516 products in
draft_000516_snapshot_era04_20260826 versus frozen era03
draft_000516_snapshot_era03_20260820 (ad19e14; core 9902d918 n=121;
ext 472bc9e4 n=179). Differ: era04 is HEAD MASTERSTAR + photometry
on the live 134-frame QC allowlist; era03 is the 2026-08-20 freeze.

## C6-0 R1'' (informational; not a gate)

Label: "pre-B1 control, reconstructed with declared shim"

Shim, only in worktree copy of dao_gaia_stage_01_iter4.py
(.worktrees/c6_r1p_c592ecf), 24571 -> 24590 bytes:

old:
        det_to_g, gaia_owner_p1, _, _ = lock_existing_and_leftover_assign(
            dx, dy, gaia_g18, locked_pairs=None, leftover_radius_px=match_radius_px
        )

new:
        res = lock_existing_and_leftover_assign(
            dx, dy, gaia_g18, locked_pairs=None, leftover_radius_px=match_radius_px
        )
        det_to_g, gaia_owner_p1, _ = res[:3]

JSON: dev/results/context/session_20260826_c6/c6_0_r1pp_shim.json

R1'' then FAIL: KeyError 'frame' in
dao_gaia_stage_01.g2_empty_false_accept (empty_df has no frame
column; worktree lacks empty_positions_main.csv layout). No 60-row
table. C6-0-P1 untested. Skipped; did not STOP on C6-0.
JSON: c6_0_r1pp_skip.json. elapsed_s 49.9.

## C6-1 Full chain into era04

Path: Archive/Drafts/draft_000516_snapshot_era04_20260826
era03 still present. Live 516 SHA unchanged
(csv bfa24039 / fits 13e77cf8 / epsf 172f9540).

Calibration: production dark+flat, draft_id=None (live OBS_FILES
not mutated). TOP1 MASTERSTAR BO_CVn_Light_109.fits (same as live
manifest). Certificate passed.

First photometry FAIL INV-CAL-01 (cal_diag only at draft root).
Copied cal_diag/sat_diag/draft_manifest to platesolve/ and
platesolve/NoFilter_60_2/.

Preprocess with draft_id=None left qc_metrics status=ok for all 150
(fwhm_limit_px=0). Live/era03 admit 134 via rejected_prefilter_fwhm.
Stamped the same 16 live basenames onto era04 qc_metrics and dropped
those aligned products. MASTERSTAR kept. Live DB not mutated.
JSON: c6_1_qc_allowlist.json. Photometry then loaded 134 frames.

Phot-only: photometry 1446.2 s; PSF LC 19.8 s; 56 aperture LCs + 56
internal PSF LCs. INV-CAL-01 PASS after sidecar copy.

sha_core 961d590fd957a39ab4837ba82a2160a036d231dc5a288eefaf37bd3b2ac3cd32 n=169
sha_ext  59206a24319a796f5501de2eda14d8977981bf18046f53e821cd364fe296a5f1 n=222

INV counters:
- identity gate CSV vy_identity_gate: n=3600 ok=2623 warn=7 fail=0
- identity gate MS pipeline_meta (before phot-only overwrite):
  passes=2 ok=5238 warn=14 fail=0 n_lock_geometry_reject=0
- optimizer refit: rejected=false reason=ok n=2630
  p95_entry=1.3066 p95_candidate=1.2277
- D3 global pool drops: state=766 gate=0 resid=0 snr_ap_pixscaled=1094
  (n_in=2927 n_out=1067)
- zone_saturated_n=24
- suspected_variables_n=163

JSON: context/session_20260826_c6/c6_1_summary.json

## C6-3 Delta ledger era03 -> era04  STOP (no lock)

n_targets_union=60. n_unchanged_all_files=0. n_changed=60.

Named (closed list):
  [FRAME-29] 1
  [D3-D5]    2  (1498964240802993408 / 1500579870061241088 path)
  [C3-K5]    1  (1496315070616056064)
  [NAME-FIX] 1  (CSS_J134925 1497169940906156032)
  [DEPTH]    0  (1500387696044768384 absent on both freezes)
  [ZONE]     0  (zone_new_saturated empty)
  [ZP-OK]    0  (aperture LCs also changed)
  [D1-AC]    0  (not a constant 2.4 mmag overlay)

UNNAMED: 55 targets. Sanity FAIL: same-ensemble per-epoch mag_calib
is not 0 and not the AC overlay constant.

BO CVn 1498613634033133184 (AAVSO candidate):
  files aperture_lc;psf_lc; n_comps 4/4; ids_swapped empty;
  source files identical (134 frames, Light_002 etc. excluded both);
  mag_calib median +2.787 mmag std 1.555 mmag range -2.822..+6.887;
  mag_calib_final median +59.389 mmag; ac_correction delta 0.
  No closed-list tag.

FW CVn 1497343732462852864: 8/8 comps; median_dmag 10.836 mmag UNNAMED
GH CVn 1498804639818507904: 8/8 comps; median_dmag 6.814 mmag UNNAMED

CSV: c63_era03_era04_ledger.csv
JSON: c6_3_stop.json

Did not invent a tag. era04 not locked.

## C6-2 Determinism  PASS (does not lock without C6-3)

Two consecutive `--full` OVERALL PASS on HEAD d75440f against the
era04 tree (temporary SNAPSHOT_NAME point; reverted after C6-3 STOP
so the committed gate stays era03).

Run1b: full-pipeline 1352s; PSF 56 files 21s; core 961d590f n=169;
ext 59206a24 n=222; science-compare n_lc=56 failures=0.
Run2:  full-pipeline 1355s; PSF 56 files 21s; same SHAs.

First --full attempt crashed: science comparator read
lightcurve_*_psf.csv comment headers (ParserError). Comparator now
skips _psf/_adaptive (photometry_sha._lc_map). Not a science-file
diff.

C6-2 PASS. C6-3 STOP. Both gates required. era04 not locked.

Summaries: c62_full_run1b_summary.txt, c62_full_run2_summary.txt

## C6-4 / C6-5
Lock skipped (C6-3 STOP). SNAPSHOT_NAME remains era03
(9902d918 / 472bc9e4). INV-ANCHOR-00 pointer unchanged. No
PUSH_AUTH SHA. `--fast --clean` OVERALL PASS (1575 passed, 32
skipped; clean-tree pytest/ruff/pyflakes PASS). Push sel-ghost-01
by name.

## Errors
C6-0 R1'' skipped (KeyError frame). C6-3 STOP unnamed + sanity.
C6-1 first photometry INV-CAL-01 (fixed by copying cal_diag).

## Files changed
- dev/results/CURSOR_RESULT_ANCHOR_ERA04.md
- dev/results/context/session_20260826_c6/c6_0_r1pp_shim.json
- dev/results/context/session_20260826_c6/c6_0_r1pp_skip.json
- dev/results/context/session_20260826_c6/c6_1_qc_allowlist.json
- dev/results/context/session_20260826_c6/c6_1_summary.json
- dev/results/context/session_20260826_c6/c6_3_stop.json
- dev/results/context/session_20260826_c6/c63_era03_era04.json
- dev/results/context/session_20260826_c6/c63_era03_era04_ledger.csv
- dev/results/session_20260826_c6/run_c61.py (harness)
- dev/results/session_20260826_c6/compare_era034.py (harness)
- `dev/tests/photometry_sha.py` (_lc_map skips `_psf` / `_adaptive`)
- docs STATE / ROADMAP / JOURNAL / DECISIONS
- `dev/results/CURSOR_RESULT_SESSION_CLOSE_20260826.md`
- Archive era04 snapshot (not git-tracked)
