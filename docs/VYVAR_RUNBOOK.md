# VYVAR — Runbook (Chi_and_H zaloha-only)

**Canonical procedure:** `docs/VYVAR_CHIANDH_BASELINE_RUNBOOK.md` (full checklist).

Quick reference for the **confirmed-reproducible zaloha-only anchor** (`203254fd` / `95a5515a`):

| Step | Action |
|------|--------|
| Catalog | `config.json` → `GAIA_DR3/zaloha/vyvar_gaia_dr3.db` (G<=16) + zaloha blind PKLs. **No field DB, no TAP, no astroquery.** |
| Run | `python scripts/chiandh_night_run_bvr.py` (#3 code; Newton bin2 ~1.30"/px) |
| Completeness | `night_run.audit_photometry_completeness` — every setup >=90% summary/active |
| SHA gate | core `203254fd...` (2806) + full `95a5515a...` (4285); two fresh runs must match (386==387) |
| Expect | ~1401 LCs; trust **1382 YELLOW / 106 RED** at `comp_trust_min_comps=5` |
| PDF | 0 overflow (R1) on all four setups B/V/R/L |

Retired anchors: `d246a5be` / `30a2f461` (TAP field-DB draft_382); `f4bcc0ee` / `bd0b1792`
(truncated draft_385).
