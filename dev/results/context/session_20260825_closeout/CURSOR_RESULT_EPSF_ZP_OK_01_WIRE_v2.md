CURSOR RESULT - 2026-08-25 (EPSF-ZP-OK-01-WIRE v2)

What I did
STOP at locate. Did not wire W1-W3. Did not reconstruct the parked
v2 task from v1 or from memory. Push: NO. Live 516 untouched.

HEAD `b1f5b8cd3ab58e27b86000720b85dd09aaa7ea25` == `origin/main`.

## Premise (Rule 0.1)

**What is compared:** the parked EPSF-ZP-OK-01-WIRE **v2** task file
(written 2026-08-24, "wide-validated-only scope after Newton 518 could
not build") versus what is on disk / in Cursor context at HEAD
`b1f5b8c` after SEL-GHOST-01 A.

**How they differ:** this amendment names W1-W3 as already specified
in v2 and forbids changing them. v2 is not in the repo. The only
full ZP-OK task on record is **v1** (2026-08-24 18:03, GO-gated,
REJECTED for missing GO). v1 W3 is docs. This amendment's W3 is the
T1 `--fast` live-516 PSF LC rewrite. Those are not the same work
item. Using v1 as v2 would violate the locate rule.

## Locate search (negative)

| Place | Result |
|-------|--------|
| `dev/tasks/` | empty (0 files) |
| `dev/results/CURSOR_TASK*ZP*` | none |
| `dev/results/context/session_20260824_*` | no ZP-OK task file |
| working tree name match `*ZP*OK*` / `*zp_ok*` | none (unrelated zp_clip results only) |
| `tmp/` | none |
| Desktop / Documents / Downloads | none |
| `~/.cursor` filename match | none |
| agent transcripts | **v2 task body never pasted as a user_query** |
| git `b1f5b8c` | confirmed absent (task said so) |

Pointers that *name* v2 but do not contain it:

- `docs/VYVAR_HANDOFF_2026-08-24.md` line 79: T1 rewrite "parked in
  EPSF-ZP-OK-01-WIRE v2 W3, which has not run"
- `dev/results/CURSOR_RESULT_SESSION_CLOSE_20260824.md` same sentence
- ROADMAP: "EPSF-ZP-OK-01-WIRE stays parked" (no v2 file)

## What is on record (not v2)

v1 exists in [EPSF light-curve product](19f2248a-a4bc-4efd-b2d4-a239c12683cc)
transcript line 413 (2026-08-24 18:03). Cursor REJECTED it for missing
Milan GO. v1 headings only (not executed):

- W1 membership predicate (`psf_zp_membership` fit_ok_strict /
  fit_ok_for_zp in INV-PSF-LC-PIN-01 only)
- W2 regenerate 60 internal PSF LCs + all-60 meters
- W3 **docs** (DECISIONS / ROADMAP / STATE)

This amendment's W3 is T1 test rewriting live 516 BO CVn PSF LC on
`--fast`. That item is **not** in v1. Scope A/B (rig allow-list,
EPSF-ZP-OK-XRIG-01) are also not in v1.

## Gates

Not run for wiring (wiring not started). Locate-fail `--fast` after
docs one-liners: **OVERALL PASS** (1530 passed, 32 skipped) at HEAD
`b1f5b8c`.

## Errors

Parked v2 task file not found. Architect re-issues v2 (or pastes it
into chat). Cursor will then apply amendment A-C on top of W1-W3
without reconstructing them.

## Files changed

- `dev/results/CURSOR_RESULT_EPSF_ZP_OK_01_WIRE_v2.md` (this STOP)
- `docs/VYVAR_STATE.md` / `docs/VYVAR_JOURNAL.md` (one-liners)
- no commit, no push, no science-path edit
