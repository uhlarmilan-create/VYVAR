CURSOR RESULT - SESSION CLOSE 2026-08-26

Architect handoff file. C6-0 STOP blocked era04 lock.
No SHA for main fast-forward in this file.

Date: 2026-08-26. Branch: sel-ghost-01. Tip at this write: aa90d7f
plus the C6-0 STOP commit that follows. origin/main stays 7c086e8.
era03 freeze: ad19e14; core 9902d918 n=121; ext 472bc9e4 n=179.
era04: NOT LOCKED. era03 not overwritten.

## Commits b1f5b8c..tip grouped by task

### SEL-GHOST-01 A
- c592ecf docs: SEL-GHOST-01 A measured

### SEL-GHOST-01 B (identity / provenance / WCS skip-solve)
- d8c18a7 fix(identity): INV-MATCH-IDENTITY-01 one gate, no name rehydration
- e2a0a84 fix(provenance): log widen loop and stamp match/optimizer meta
- 01f6f77 fix(wcs): skip-solve path must not shadow module WCS

### SEL-GHOST-01 B1b
- 0684ba9 B1 follow-up: lock 4-tuple callers
  (first-tracked dao_gaia_stage_01_iter2/3/4.py AND dao_gaia_stage_01.py)
- 58a2187 INV-SOURCE-STATE-01: detected means detected
- b39982c B1 follow-up: track empty-sky CSV for --fast
- 6dad937 B1 follow-up: STAGE-01 np.load kwargs for PP-KWARG-01
- e59f2a2 docs: SEL-GHOST-01 B-STOP-1b measured
- 4fc6b8f B2 addendum: stamp effective vs config DAO-Gaia tols

### SEL-GHOST-01 B2 (D1-D4, S5-S7, 520 re-solve)
- dc48ece D4: lock reject on 3xFWHM identity threshold
- b13d393 D1: one-pass catalog match radius, no widen loops
- 5d209d7 D2: optimizer refit guard and entry WCS backup
- 1cd404a D3: comparison candidacy before RMS ceiling
- 9550e4f S7: --fast --clean worktree gate
- e0f918a S6: pass telescope_id into MASTERSTAR plate-scale resolve
- 24855e0 S5 follow-up: stamp MASTERSTAR snr from peak over sky sigma
- 1ec035f S6: keep DB plate scale over FITS/UI
- 936512f S5 tests: give synthetic comp fixtures D3 columns
- a0768ef docs: SEL-GHOST-01 B-STOP-2

### SEL-GHOST-01 B3 (T1/T2, INV-CAL, production-path evidence)
- e410130 T1: D1 catalog radius uses 3xFWHM only
- 6e0fd5c T2: MASTERSTAR snr is aperture SNR
- 6950495 INV-CAL-01: pre_calibrated drafts skip cal_diag requirement
- 7f80c28 docs: SEL-GHOST-01 B-STOP-3 (no re-cut, no push)

### CLOSE-OUT C0
- c929c0b C0a: one haversine for comparison-star _dist_deg; persist 1e-9 deg
- 78b3495 C0b: D3 gates on snr_ap_pixscaled

### CLOSE-OUT C1-C2
- e2fc6d0 C1: ANCHOR-DRIFT-01 STOP (R1 vs R0 freeze lag; census 4+1)
- bf8eb42 C2: COMP-RMS-DEF-01-A STOP
- e73fd29 docs: CLOSE-OUT C0-C2/C4; full-chain before recut

### CLOSE-OUT C3 / C7
- 2ac6580 C3: wire LOO mag RMS and ZONE-SAT-01 peak test
- 24fe281 C3 STOP: COMP-RMS-DEF-01-B k=5, P-C3-1 HIT, one 516 miss r=5.10
- 3e085bd C7 STOP: R1 harness contamination; lost VSX is depth+DAO
- 50379f9 docs: CLOSE-OUT C3/C7 STOP
- ba9fef3 docs: INV-COMP-RMS-01 is a code contract
- 7c086e8 C3: register comp_rms_loo_photon_k as auto widget
  (this SHA is origin/main after the 2026-08-25 push incident)

### P0 push guard + CLOSE-OUT C8
- 987dd9a P0: refuse unauthorized git push to origin/main
- ebe03e8 C8 STOP: R1 prime blocked (one-file); frame 29 QC admitted; DEPTH-AUTH-01

### ZP-OK v2 (C4)
- b79429f C4 W1+W3: PSF ZP membership fit_ok_for_zp on rig 1:1
- 6924998 C4 W4: ZP-OK v2 STOP, wide-rig only, XRIG-01 parked
- aa90d7f C4: record --fast --clean PASS and W3b live BO PSF SHA

### era04
- not locked (C6-0 STOP)

## era04 SHAs
None. Do not write PUSH_AUTH for a recut that did not happen.

## Open ROADMAP ids
- EPSF-SHAPE-01 HIGH (narrow ePSF core; routed to EPSF-CORE-01)
- EXPORT-PARITY-01 HIGH
- EPSF-ZP-OK-XRIG-01 MED (extend fit_ok_for_zp past wide 1:1)
- MULTIFILTER-WCS-01 MED (sibling-seed for 520 z_90_4; g_60_4 re-solve note)
- FRAME-QC-PARITY phase 2 MED (n_stars item: 516 frame 29 n=263 vs ~100)
- DEPTH-AUTH-01 LOW (G=15.56 VSX stays absent at recut)
- EPSF-CORE-01 FUTURE
- INPUT-PATH-ARCH-01 OPEN
- DAO-TOL-FLOOR-01 (carry; see ROADMAP body)
- COMP-RMS-DEF-01 CLOSED (k=5 rule)
- ZONE-SAT-01 CLOSED
C6 full-chain era04 remains the next GO after this STOP is resolved.

## Push-incident note
2026-08-25 `git push origin HEAD` advanced origin/main from b1f5b8c
to 7c086e8. No force-push. P0 guard: `dev/scripts/push_guard.py`;
`session_baseline_check` installs `.git/hooks/pre-push`. Updating
main requires gitignored `dev/PUSH_AUTH_main_<YYYYMMDD>.txt` with the
target SHA. Pushes name the ref. Banned: `git push origin HEAD` and
bare `git push`. origin/main stays 7c086e8 until Milan writes
PUSH_AUTH after era04 is locked (not this STOP).

## Known issues carried
- sqlite threading flake on malformed-DB WARN (C4 --fast: 13/13 in isolation)
- 520 selection wiring lives on the branch; 520 has no era snapshot
- Newton 518 unchanged (gated pool 26; ZP-OK does not apply)
- C6-0 R1' blocked: three-file copy still missing dao_gaia_stage_01
- T3 R1 remains contaminated; C6-0-P1 untested
- `--full` on frozen era03 MS still raises until a full-chain recut
  (D3/C3 columns absent on freeze)

## SHA for PUSH_AUTH_main_<date>.txt
Do not write one. era04 is not locked. After a future C6 recut that
passes C6-2 and C6-3, the SHA is the sel-ghost-01 tip of that lock
commit, not aa90d7f and not this STOP commit unless that commit is
the lock.
