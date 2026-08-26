CURSOR RESULT - SESSION CLOSE 2026-08-26

Architect handoff file. APERTURE-01c STOP blocked era04 lock
(6 UNNAMED in ledger v5). Independent AIJ gate PASS (2.7833 mmag).
No SHA for main fast-forward in this file.

Date: 2026-08-26. Branch: sel-ghost-01. origin/main stays 7c086e8.
era03 freeze: ad19e14; core 9902d918 n=121; ext 472bc9e4 n=179.
era04 products (f=1.35 recut): 988a2e13 n=160 / e8fed401 n=210;
NOT LOCKED. candidate2 (f=0.385) kept. era03 not overwritten.
Live 516 SHA unchanged.

APERTURE-01c: config f=1.35 mode (a). AIJ comps from Table.tbl.
Recut r_ap=7.0088 all 134 frames. C6-2 skipped. C6-5 = this file.

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

### era04 / APERTURE-01c
- not locked (ledger v5: 6 UNNAMED). C6-2 skipped.
- f=1.35 recut products 988a2e13 n=160 / e8fed401 n=210
- candidate2 (f=0.385) kept; candidate1 kept; era03 untouched
- APERTURE-01 STOP a23ee3d; APERTURE-01b STOP 2509f02
- this STOP commit: f=1.35 + gate + ledger v5 + docs

## era04 SHAs (products only; not the `--full` gate)
APERTURE-01c f=1.35 recut (not locked):
core 988a2e135b37fcb0d88c7d08c98cf270428e4ce314cd29c8281665cbd3d54106 n=160
ext  e8fed401667c6947dc0cf40f0050fe79a6f466d8cf45994edbfcc54b4f06998b n=210
Prior C6-1 product (SNR-table / later overwritten): 961d590f n=169 /
59206a24 n=222. candidate2 = APERTURE-01 f=0.385 recut, kept.
Do not write PUSH_AUTH for origin/main. Gate remains era03
9902d918 / 472bc9e4.

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
C6 full-chain era04 remains blocked until Milan names the
same-ensemble mag_calib motion or holds the recut.

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
- C6-0 R1'' skipped (KeyError frame); C6-3 STOP unnamed mag_calib
- APERTURE-01c ledger v5: 6 UNNAMED (lost LC at r=7 / Phase1 starve)
- T3 R1 remains contaminated; C6-0-P1 untested
- `--full` on frozen era03 MS still raises until a locked recut
  (D3/C3 columns absent on freeze). era04 `--full` PASS is not the gate.

## SHA for PUSH_AUTH_main_<date>.txt
Do not write one. era04 is not locked. sel-ghost-01 STOP tip
`57d9567daa0233162e499f8bf607ea05bfe67330` (`57d9567`) is the
APERTURE-01c record, not a main fast-forward. After a future recut
that names the 6 UNNAMED and passes C6-2, the SHA is that lock
commit.
