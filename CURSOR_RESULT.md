CURSOR RESULT — 2026-06-22 (G3-F002 query_local_gaia mag_limit=None)

What I did
Path A: `mag_limit=None` ⇒ no g_mag SQL cap in `query_local_gaia`; MASTER_SOURCES explicit `mag_limit=None`; updated stale comment at `_query_gaia_local`. Tests + ledger. No push.

## Step 1 — SQL construction (before fix)

`database.py:150-152` silently set `mag_limit = 11.5` when omitted.

SQL built at `:238-247` with always `AND g_mag <= {mag_limit}`; optional `ORDER BY g_mag ASC LIMIT` when `max_rows` set.

Clamp/log at `:159-167` when explicit cap > DB MAX(g_mag).

## Step 2 — Fix

- `mag_cap: float | None` — set only when `mag_limit` is explicit finite > 0; invalid values log and apply **no cap** (no silent 11.5).
- `mag_clause` empty when `mag_cap is None`.
- Clamp/"orezávam" skipped when no cap.

## Step 3 — pipeline.py:11404 (MASTER_SOURCES)

- `mag_limit=None` explicit (full depth for faint detections).
- **max_rows:** left unset — bbox is detection extent ±0.01° (tiny); `ORDER BY g_mag LIMIT N` would bias to bright stars and re-cut faint matches; row count not pathological.

## Step 4 — pipeline.py:4405

- Comment updated: `None ⇒ no mag cap` (was "defaults to 11.5").
- When `max_mag is None`, `_ql_kw` omits `mag_limit` → full depth (matches faintest_mag_limit intent).

## Step 5 — Tests

`tests/test_query_local_gaia_g3_f002.py` — **5 passed**
- Explicit `mag_limit=11.5` excludes g>11.5
- `mag_limit=None` includes g=12, 14.5
- Explicit `mag_limit=20` unchanged row set
- MASTER_SOURCES bbox simulation: faint match within 2″ with None; excluded with 11.5

Full suite: **415 passed, 15 skipped**

## Scripts note

`scripts/forced_photometry_pal7.py` and `scripts/diagnose_wide_true_triangle_shape.py` pass explicit `mag_limit` — unchanged. Any script omitting `mag_limit` now gets full depth (acceptable for dev tools).

## Ledger

G3-F002 → **FIXED** (Path A)

## Files changed

- `database.py`, `pipeline.py`
- `tests/test_query_local_gaia_g3_f002.py`
- `docs/VYVAR_FULL_AUDIT_LEDGER.md`

**Not pushed** — stop for Claude review.
