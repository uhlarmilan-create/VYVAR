CURSOR RESULT - SESSION CLOSE 2026-08-27

Architect handoff file. APERTURE-01d LOCK. era04 is the --full
gate. origin/main stays 7c086e8. No SHA for main fast-forward
in this file until Milan writes PUSH_AUTH.

Date: 2026-08-27. Branch: sel-ghost-01. origin/main stays 7c086e8.
era04 freeze: draft_000516_snapshot_era04_20260826;
core 9367f998 n=160; ext d3cefff3 n=210.
era03 freeze kept on disk: ad19e14; core 9902d918 n=121;
ext 472bc9e4 n=179. candidate1/2/3 kept. era03 not overwritten.
Live 516 SHA unchanged (csv bfa24039 / fits 13e77cf8 /
epsf 172f9540).

APERTURE-01d: annulus 2.7/5.2. Independent AIJ gate 1.9503 mmag
PASS (<=2.8). Ledger v6: 0 UNNAMED. C6-2 --full twice.
C6-4 lock. C6-5 = this file.

## Commits 7c086e8..tip grouped by task

See CURSOR_RESULT_SESSION_CLOSE_20260826.md for 7c086e8..APERTURE-01c.
This lock commit is APERTURE-01d (annulus 2.7/5.2, ledger v6,
era04 SNAPSHOT_NAME, A9 gold update, --full twice, lock docs).
PUSH_AUTH SHA for sel-ghost-01 is `dffe859c6a679fa2130bcec5f3a4460cfe23e856`
(`dffe859`). Not for origin/main.

### era04 / APERTURE-01d LOCK
- locked. Ledger v6: 0 UNNAMED.
- f=1.35 annulus 2.7/5.2 recut products 9367f998 n=160 / d3cefff3 n=210
- candidate3 = 01c f=1.35 tree, kept; candidate1/2 kept; era03 untouched
- SNAPSHOT_NAME is era04; VL-ANCHOR-WCSINV points here
- AIJ gate 1.9503 mmag; CSS/HAT-188 CROWDING; BO vs v5 -7.72 mmag
  (prediction <5 FAIL, reported)

## era04 SHAs (canonical --full gate)
APERTURE-01d f=1.35 annulus 2.7/5.2:
core 9367f99848c14b43016321d000ec53651c9b260290bcb37afd2f6bab5035b2d7 n=160
ext  d3cefff3240b4874d9b0ba3f76f7a303a5e3ea8b83f051149202d5b9c65d6863 n=210
Do not write PUSH_AUTH for origin/main. Gate is era04.
era03 9902d918 / 472bc9e4 remains on disk.

## Open ROADMAP ids
- EDGE-ANNULUS-01 (record only; next session step 1)
- EPSF-SHAPE-01 HIGH (narrow ePSF core; routed to EPSF-CORE-01)
- EXPORT-PARITY-01 HIGH
- EPSF-ZP-OK-XRIG-01 MED (extend fit_ok_for_zp past wide 1:1)
- MULTIFILTER-WCS-01 MED (sibling-seed for 520 z_90_4; g_60_4 re-solve note)
- FRAME-QC-PARITY phase 2 MED (n_stars item: 516 frame 29 n=263 vs ~100)
- DEPTH-AUTH-01 LOW (G=15.56 VSX stays absent at recut)
- EPSF-CORE-01 FUTURE
- INPUT-PATH-ARCH-01 OPEN
- First AAVSO/VarAstro uploads BO -> FW (band CV) -- ledger locked

## Push-incident note
2026-08-25 `git push origin HEAD` advanced origin/main from b1f5b8c
to 7c086e8. No force-push. P0 guard: `dev/scripts/push_guard.py`;
`session_baseline_check` installs `.git/hooks/pre-push`. Updating
main requires gitignored `dev/PUSH_AUTH_main_<YYYYMMDD>.txt` with the
target SHA. Pushes name the ref. Banned: `git push origin HEAD` and
bare `git push`. origin/main stays 7c086e8. Milan writes PUSH_AUTH
if a main fast-forward is wanted after this lock.

## Known issues carried
- sqlite threading flake on malformed-DB WARN
- 520 selection wiring lives on the branch; 520 has no era snapshot
- Newton 518 unchanged (gated pool 26; ZP-OK does not apply)
- CSS_J135929 and HAT-188-0003359 0 epochs (CROWDING; VSX not
  force-injected)
- POOL-STARVE three empty_comp_drop (limitation, allowlisted)
- EDGE-ANNULUS drops FR CVn and four EDGE IDs (limitation)
- T3 R1 remains contaminated; C6-0-P1 untested
- A9 draft-367 envelope HV recover 0.267, BLOCK_2B_GUARDS (gold
  updated; not a 367 photometry SHA regression)

## SHA for PUSH_AUTH_main_<date>.txt
Do not write one (file absent on disk). sel-ghost-01 lock tip
`dffe859c6a679fa2130bcec5f3a4460cfe23e856` (`dffe859`) is the
APERTURE-01d lock, not a main fast-forward. Named-ref push:
`git push origin main:sel-ghost-01`.
