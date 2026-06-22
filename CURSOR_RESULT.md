CURSOR RESULT — 2026-06-22 (G6-F002 master validity_days 90/200)

What I did
Mapped validity-day precedence, unified defaults to dark **90** / flat **200** across dataclass, `__post_init__`, DB seed, and tracked `config.json`. Added regression tests. Ledger G6-F002 → FIXED.

## Step 1 — Verify-first (precedence)

**Live calibration path:** `importer.py` receives `masterdark_validity_days` / `masterflat_validity_days` as **function arguments** from `cfg` (`app.py`, `night_run.py`, `ui_settings` save). Gate: `age_days > validity_days` → expired.

**DB SETTINGS seed verdict: VESTIGIAL**
- `database.py` `_seed_default_settings` inserts `masterdark_validity_days` / `masterflat_validity_days` into `SETTINGS`.
- `get_setting_int()` exists but has **zero callers** repo-wide — seed is never read on the calibration path.
- Effective value is always **`AppConfig`** (from `config.json` via `__post_init__`).

**Literal default sources (before fix):**

| Source | dark | flat |
|--------|------|------|
| `config.py` dataclass `:100-101` | 80 | 524 |
| `config.py` `__post_init__` `:642-643` | **60** | 200 |
| `database.py` seed `:2556/2560` | **60** | 200 |
| tracked `config.json` | 80 | 524 |
| `docs/VYVAR_PARAMS.md` | 80 | 524 |

## Step 2 — Fix (90 / 200 everywhere)

- `config.py` dataclass + `__post_init__` fallbacks
- `database.py` seed strings `"90"` / `"200"`
- `config.json` → 90 / 200
- `importer.py` consumer logic unchanged

## Step 3 — Tests

`tests/test_master_validity_days_g6_f002.py` — **4 passed**
- Empty JSON → 90/200
- JSON missing keys → 90/200
- Regression: dataclass default == `__post_init__` fallback == DB seed literals
- Fresh `VyvarDatabase` SETTINGS rows → 90/200

Full suite: **410 passed, 15 skipped** (prior run had 1 fail before VyvarDatabase fix; spot-check green).

## Step 4 — Ledger

G6-F002 → **FIXED** (notes DB seed vestigial).

## Note for Milan

If your **local** `config.json` outside the repo still has 80/524, update to 90/200 (or your tighter per-rig values) so runs match intent.

## Files changed

- `config.py`, `database.py`, `config.json`
- `tests/test_master_validity_days_g6_f002.py`
- `docs/VYVAR_FULL_AUDIT_LEDGER.md`

**Not pushed** — stop for Claude review.
