CURSOR RESULT — 2026-07-13 (SESSION-CLOSE-0713)

What I did
Closed color-WB continuation session: dirty-hash caption suffix, verification sweep, STATE/JOURNAL/
ROADMAP updates, push.

## Part 1 — dirty-hash suffix
`_repo_short_git_hash()` now uses `git describe --always --dirty --abbrev=7` ? e.g. `c5685c6-dirty`
when tree unclean. Test `test_caption_render_stamp_dirty_hash` added.

## Part 2 — verification

**git log `c5685c6..HEAD`:**
```
2aaf858 Append -dirty suffix to HRD colorfield caption git hash when tree is unclean.
0608739 Set HRD chroma boost default to 2.2 and document 12g6 close-out.
16e26c2 Add HRD colorfield caption provenance stamps (UTC + git hash).
```

**pytest:** 760 passed, 15 skipped.

**session_baseline_check --fast:** OVERALL PASS (HEAD 2aaf858 pre-close push).

**git status note:** working tree has unstaged deletions of legacy `CURSOR_RESULT_*.md` files (not
committed); untracked: `.worktrees/`, `docs/VYVAR_CODE_AUDIT.md`, `scripts/dy_peg_night_run_bvr.py`,
`scripts/qatar8_night_run_v.py`, `docs/round2_figs/v0454_lc_vyvar.png`,
`CURSOR_RESULT_except_retriage3.md`. No sandbox leftovers beyond gitignored tmp artifacts.

## Part 3 — documentation
- `docs/VYVAR_STATE.md` — 0713 snapshot, color-WB arc CLOSED, final defaults, canonical/archive paths.
- `docs/VYVAR_JOURNAL.md` — SESSION-CLOSE-0713 entry (Czech).
- `docs/VYVAR_ROADMAP.md` — NEXT SESSION restated (SIGMA-NEWTON + Milan data tasks); colorfield PDF parked.

## Part 4 — push
Close commit + push hash recorded below after push.

## Color-WB arc summary (CLOSED)
| Stage | Commit area | Outcome |
|-------|-------------|---------|
| 12g | catalog-color field | Gaia BP-RP tint × mono L |
| 12g2 | polish | SNR gate, field_median WP, hue highlights |
| 12g4 | boost | distance-from-white expansion |
| 12g5 | blotch fix | local-bg gate, tapered stamp, hardened G2 |
| 12g6 | hygiene | caption stamps, boost default 2.2, `tmp/colorfield_final/` |

**Canonical outputs:** `tmp/colorfield_final/manifest.json` — 4 renders @ b2.2, G2 worst 0.005–0.027.
**Archive:** `tmp/todo12_hrd_archive_0711/`.

## Errors (if any)
None.

## Files changed (close commit)
- `hrd_colorfield.py` (Part 1, prior commit `2aaf858`)
- `tests/test_hrd_colorfield.py` (Part 1)
- `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_ROADMAP.md`
- `CURSOR_RESULT_close_0713.md`
