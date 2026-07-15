# VYVAR — Runbook (Chi_and_H zaloha-only)

**Canonical procedure:** `docs/VYVAR_CHIANDH_BASELINE_RUNBOOK.md` (full checklist).

## Session baseline check (every session)

Run at session start **before any new work** (see `docs/VYVAR_CLAUDE_OPERATING_PRINCIPLES.md`):

```text
git pull
# read docs/VYVAR_STATE.md + latest JOURNAL
python scripts/session_baseline_check.py          # --fast (default, ~3 min)
python scripts/session_baseline_check.py --full   # after science-touching changes (~25 min)
```

| Tier | What it checks |
|------|----------------|
| **--fast** | Git tree (FAIL on staged changes); config paths; full `pytest -q`; validation ledger + TODO hint for `passes=false` items |
| **--full** | Everything in --fast, plus headless **draft_424** via `run_full_photometry_pipeline` into `tmp/session_baseline/<timestamp>/` (archive inputs read-only); **content anchor gate**: science-meaningful compare vs `draft_000424_snapshot_20260708_full` + photometry SHA (core `92939fab…` / extended `76642318…`); `except_fix_counters` all-zero; updates ledger `VL-ANCHOR-424` + `VL-COUNTERS-ZERO` on PASS. **Provenance block hash** printed informational only (git-bound; not a cross-commit gate). |

Exit **0** = PASS, **1** = FAIL. Concise summary table printed at end.

### Anchor cuts and entry paths (2026-07-08)

| Path | Use |
|------|-----|
| `run_full_photometry_pipeline` | **Production path** and **exclusive anchor-cut recipe** from 2026-07-08. Builds `ProcFrameStore`, Phase 0+1 comp selection, Phase 2A LCs — single coherent artifact tree. |
| Phase-2A-only rerun | Legacy validation shortcut (QUICKWINS-0708 Item 4). Skips `ProcFrameStore`; must **not** be used for anchor cuts. Produced the retired hybrid snapshot (`draft_000424_snapshot_20260708_hybrid_deprecated`). |

**Two-fresh-runs rule:** before locking any photometry anchor, run the full production path
**twice** and both must match (science comparator + photometry SHA).

### Pair protocol v2 (F-431 — preferred for SHA gates)

Prefer the **draft_387 same-draft** precedent over two independent imports:

1. **Import once** (one `draft_id`, one Raw/calibrated tree).
2. Run photometry stages **twice** over that same draft (or wipe only photometry/LC outputs
   between passes — keep processed + MASTERSTAR if testing photometry-only SHA).
3. Byte-compare core photometry SHA (incl. seeded Labbe `err`).
4. Cut snapshot from the first photometry pass only after SHA match.

Two-import pair runs remain allowed for end-to-end smoke, but SHA mismatches that only touch
`err` / SysRem were historically contaminated by unseeded Labbe + forced SysRem — see
`CURSOR_RESULT_headless_forensics.md`. `anchor_pair_run` now honors `config.sysrem_enabled`.

**Retired anchor:** `Archive/Drafts/draft_000424_snapshot_20260708_hybrid_deprecated` — hybrid artifact (2026-06-24 comp CSV + 2026-07-07 Phase-2A-only LCs); not a valid anchor.

---

Quick reference for the **confirmed-reproducible zaloha-only anchor** (`3f7c9e7a` / `d5b72d08`):

| Step | Action |
|------|--------|
| Catalog | `config.json` → `GAIA_DR3/zaloha/vyvar_gaia_dr3.db` (G<=16) + zaloha blind PKLs. **No field DB, no TAP, no astroquery.** |
| Run | `python scripts/chiandh_night_run_bvr.py` (#3 code; Newton bin2 ~1.30"/px) |
| Completeness | `night_run.audit_photometry_completeness` — every setup >=90% summary/active |
| SHA gate | core `3f7c9e7a...` (2806) + full `d5b72d08...` (4285); two fresh runs must match; historical `203254fd...` / `95a5515a...` |
| Expect | ~1401 LCs; trust **1382 YELLOW / 106 RED** at `comp_trust_min_comps=5` |
| PDF | 0 overflow (R1) on all four setups B/V/R/L |

Retired anchors: `d246a5be` / `30a2f461` (TAP field-DB draft_382); `f4bcc0ee` / `bd0b1792`
(truncated draft_385).
