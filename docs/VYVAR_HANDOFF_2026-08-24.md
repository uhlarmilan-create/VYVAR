# VYVAR handoff - 2026-08-24

Start here in a new session. Current snapshot: `docs/VYVAR_STATE.md`.
Open work: `docs/VYVAR_ROADMAP.md`. Decisions: `docs/VYVAR_DECISIONS.md`.

Provenance: the architect asked Milan to paste this file verbatim into
SESSION-CLOSE-20260824. The paste was not in the Cursor user_query.
This file is assembled in ASCII from STATE / ROADMAP / DECISIONS plus
the known-issues list that WAS in the close task (quoted below). It is
the next-session start file, not a substitute for an unread architect
memo.

---

## Repository

- **Branch:** `main`
- **Close stack on top of** `92361a3` (CAL-520-01): AC-02 `6fbfbea`,
  REG-520-01+518 `db628be`, this handoff (see `git log -1` after push)
- **Local series before this close (origin/main b1af049):**
  dbb6967, 1f9f921, 6fd1452, cf95c53, d206b43, 876053a, 2926a95,
  d66613c, e5a6149, 505fa13, 92361a3
- **Production 516 ePSF SHA:**
  `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20`
  (unchanged this session)
- **Anchors:** core `9902d918` n=121; extended `472bc9e4` n=179;
  P1 golden `6af4539c` n=115

Session init: read STATE, ROADMAP, latest JOURNAL, PROCESS, and
`docs/VYVAR_CLAUDE_OPERATING_PRINCIPLES.md`.

---

## What closed 2026-08-24

- **EPSF-AC-02** (Milan GO): `psf_ac_policy=p4_none`; INV-PSF-LC-PIN-01.
  Evidence: `CURSOR_RESULT_EPSF_AC_02_WIRE.md`.
- **EPSF-PIN-CENSUS-01:** 100% of 516 pin drops are stored chi2>=50.
- **EPSF-NEWTON-518-01 STOP N2:** gated pool 26 < 30; ZP-OK parked.
- **DAO-GAIA-XFER-01:** STAGE-01 sandbox gate pinned to hand params.
- **CAL-520-01:** library facts (no AZ800 masters; INV-PREP-01 0.02x).
- **REG-520-01:** same non-cal button, June 0.0622 vs today 0.3949 is
  S2 selection (rms ceiling 0.1 + 59-px false locks), not S1 pairing
  starvation and not H-CAL-MISCLASS. Evidence:
  `CURSOR_RESULT_REG_520_01.md`.

Non-cal is a first-class route. PRECAL radiometric metric informs, does
not block.

---

## Next session - act on these (Milan GO)

1. **REG-520-S2** selection-input: comps must sit on the Gaia star
   (residual ~ solve rms / FWHM); rms ceiling 0.1 currently prefers
   59-px offset G~15-17 IDs. Not wired.
2. **DAO-TOL-FLOOR-01** (optional later): pass2/seed floor = f(solve
   rms, FWHM). Cite REG-520-01 M2 curve. Worst live gap: i_70_4
   (tol 1.0 vs rms 2.98). Not wired.
3. **`non_cal_declared`** banner + cautious LC class + submit lock.
4. Add `time_base` to `_LC_OVERVIEW_COLS` (UI "time (unknown)"; on-disk
   LC is BJD_TDB).
5. Newton 518 ePSF: hold for gated pool >=30, or wide-only ZP-OK without
   claiming Newton.
6. Interim `psf_fit_ok_for_zp`: GO or hold (wide-rig CENSUS-01 stands).
7. **`--full` recut** 9902d918 / 472bc9e4.
8. **EXPORT-PARITY-01** (HIGH). First AAVSO/VarAstro BO -> FW (band CV).

z_90_4 solve reject remains parked (MULTIFILTER-WCS-01). Do not lower
`masterstar_catalog_recovery_min`.

---

## Known issues - do not fix from this close

Quoted from SESSION-CLOSE-20260824 (architect, 2026-08-24):

- T1 test still rewrites the live 516 BO CVn PSF LC on every `--fast`
  (fix parked in EPSF-ZP-OK-01-WIRE v2 W3, which has not run).
- ZP-OK v2 GO undecided; DAO-GAIA production tolerance floor unwired;
  520 selection-input defect unwired - all parked in the handoff.

---

## Iron rules (do not violate)

1. Science pixels: no in-frame cosmic-ray cleaning.
2. Gates: do not remove a verification gate on byte-identity evidence
   alone (`INV-GATE-REMOVAL`).
3. Comparison membership: once per draft after Phase 0+1.
4. Measurement over plausibility.
5. Anchor re-cut: Milan authorization before changing P1 golden SHA
   or draft archive products.
6. Push requires Milan authorization in chat.

---

## Three-role model

| Role | Responsibility |
|------|----------------|
| Milan | Observing, telescope decisions, authorizes re-cuts and pushes |
| Claude (architect) | STATE/ROADMAP/JOURNAL/DECISIONS, task specs, review |
| Cursor (implementer) | Code, measurements, CURSOR_RESULT_*.md, tests |
