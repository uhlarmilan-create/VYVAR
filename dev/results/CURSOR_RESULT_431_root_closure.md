# CURSOR RESULT - F-431 root closure

**Date:** 2026-07-16. **Baseline:** `715391b` (+ this fix tree). **Anchor:** still **BLOCKED**.

---

## Ledger corrections

1. **DECISIONS `F-428-PASS2-CONTAMINATION`:** **DOWNGRADED** as primary census driver.
   Draft_433 with FIX 2 live still = 6699-class -> driver = absent preprocess ADU surface.
   WCS-INV asymmetry on 428 remains real/fixed (v5), secondary for census.
2. **STATE 429 narrative:** "healthy 429" = **unprovenanced anomaly** (`git_dirty=true`);
   deterministic repo state = 6699-class; 429 remains census **quality target**.

---

## T1 - Lost transform (done)

Artifact: `tmp/f431_lost_transform.md` + PNGs under `tmp/f431_lost_transform/`.

| Check | Result |
|-------|--------|
| R = MS(429) - cal Light_008 | mean/median ~ -91 / -98 ADU; maxabs ~1.7e4 at sparse pathological pixels |
| Spatial | order-2 surface var_explained **0.49**; phase-corr shift ~0; **not** hot-pixel clip |
| Sky | Delta median -95.7 ADU <-> meta ~1565->1478 |
| DAO | cal **9115** -> MS **2816**; constant pedestald alone **no**; **cal+order2 -> 2626** |

**Class:** `SMOOTH_ORDER2_BACKGROUND_SURFACE`. Re-implement as shared preprocess (gated on T2/Milan).

---

## T2 - Milan clean UI (awaiting)

No draft beyond 433 on disk. **Branch verdict: PENDING.**

When draft arrives, validate: `cal==proc?`, census class, A-durable, `git_dirty` (+ dirty list if true).

---

## T3 - Restore transform (**GATED**)

Not implemented - needs T2 (UI-HEALTHY vs UI-SICK) and Milan sign-off for science default-ON.

---

## T4 - Fix bundle (landed without T3)

| Item | Status |
|------|--------|
| C1 stamp after `write_photometry_plan_files` | **DONE** (`pipeline.py`) |
| `anchor_pair_run` honor `sysrem_enabled` | **DONE** (`None` -> config) |
| Labbe content seed + provenance policy | **DONE** (`photometry_core.py` + tests) |
| Pair protocol v2 in RUNBOOK | **DONE** (`docs/VYVAR_RUNBOOK.md`) |
| NightRun `catalog_match_max_sep_arcsec=2.0` | **DONE** (+ PARAMS note) |
| `git_dirty_files` list when dirty | **DONE** (provenance block) |

Tests: `tests/test_f431_labbe_provenance.py` + related (**21** passed in focused run).

---

## T5 - Closeout

- **pytest (focused):** green for F-431 + stamp + Labbe suites.
- **Anchor attempt #3:** **NOT RUN** - still no healthy-census tree (T3 pending).
- **Commit/push:** prepared for Milan (this tree).

---

## Files touched

- `docs/VYVAR_DECISIONS.md`, `STATE.md`, `ROADMAP.md`, `JOURNAL.md`, `RUNBOOK.md`, `PARAMS.md`
- `pipeline.py` (stamp order)
- `photometry_core.py` (Labbe seed, provenance dirty list)
- `night_run.py`, `scripts/anchor_pair_run.py`
- `scripts/f431_characterize_lost_transform.py`
- `tests/test_f431_labbe_provenance.py`
- `tmp/f431_lost_transform*`

## Next for Milan

1. Clean UI RUN VYVAR on `715391b` + `D:\BO_CVn` (A-durable watch). Cursor will T2 classify.
2. Sign-off on order-2 sky surface as intentional preprocess -> T3.
3. Then protocol-v2 anchor pair on healthy census.
