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
| **--full** | Everything in --fast, plus headless **draft_424** via `run_full_photometry_pipeline` into `tmp/session_baseline/<timestamp>/` (archive inputs read-only); science-meaningful compare vs `draft_000424_snapshot_20260708`; `except_fix_counters` all-zero; updates ledger `VL-ANCHOR-424` + `VL-COUNTERS-ZERO` on PASS |

Exit **0** = PASS, **1** = FAIL. Concise summary table printed at end.

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
