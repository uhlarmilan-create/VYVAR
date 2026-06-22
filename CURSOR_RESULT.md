CURSOR RESULT — 2026-06-22 (G5-F002 close + G5-F007 fix)

What I did
Marked **G5-F002 RESOLVED (non-issue)** in ledger. Diagnosed G5-F007 call chain; implemented derive-or-None plate scale + canonical `#SOFTWARE` version; added tests; regression on AAVSO data rows; isolated commit (no push).

## G5-F002 — RESOLVED (non-issue)

**Evidence (one line):** AC applies constant `delta_m_corr` → `mag_calib_ac`; per-point `err` (photon + ensemble SEM) is invariant under constant mag shift; no `err_ac` in pipeline; folding `ac_scatter` per-point would misrepresent a correlated systematic as random.

Ledger: **G5-F002** → **RESOLVED (non-issue)**; fix-log step 7.

---

## G5-F007 — diagnosis (call chain)

### Plate scale `1.3`

| Caller | Passes `arcsec_per_px`? |
|--------|-------------------------|
| `photometry_core.export_all_method_lightcurve_reports` (~8003) | **Before fix:** always `float(_cfg.export_arcsec_per_px)` (config default **1.3**) |
| `export_lightcurve_reports` signature | **Before fix:** default `arcsec_per_px=1.3` |
| Scripts (`reexport_draft_aavso.py`, `verify_method_report_separation.py`) | Explicit literals in some paths |

**Production path:** Phase 2A → `export_all_method_lightcurve_reports` → `export_lightcurve_reports`. Config default **1.3** was threaded on every export unless operator overrode `export_arcsec_per_px` in config — **not** draft WCS/meta.

**Real plate scale at export site:** `photometry/pipeline_meta.json` → `plate_scale_arcsec_px` (Phase 2A); sibling `../MASTERSTAR.fits` WCS/CD via `_resolve_plate_scale_arcsec_per_px`. Reachable from `out_base.parent` (`_phot_dir`).

**Verdict:** **1.3 did reach** VarAstro aperture-arcsec commentary whenever config default was used and meta/WCS were not consulted for export headers. Severity was real, not latent-only.

### `#SOFTWARE=VYVAR/1.0`

AAVSO header used a **literal** `#SOFTWARE=VYVAR/1.0` while `software_version` param only appeared in VarAstro body. No separate config key; canonical string is module constant `VYVAR_SOFTWARE_VERSION = "VYVAR 1.0"`.

---

## G5-F007 — fix (minimal)

**`export_reports.py`**
- `VYVAR_SOFTWARE_VERSION` + `_aavso_software_header_line()` — AAVSO `#SOFTWARE` uses version param/constant (space → `/`).
- `_resolve_export_arcsec_per_px(photometry_dir, cfg)` — `pipeline_meta.plate_scale_arcsec_px` → MASTERSTAR WCS; **derive-or-None** (no 1.3).
- `export_lightcurve_reports` resolves scale internally; VarAstro `#   Aperture: …arcsec` only when scale derivable.

**`photometry_core.py`**
- Removed `arcsec_per_px=float(_cfg.export_arcsec_per_px)` and hardcoded `software_version="VYVAR 1.0"` from export call (~8013).

**`tests/test_export_g5_f007.py`** — 8 tests (positive home/fine rig, derive-or-None omit, version header, regression data row).

---

## Tests

| Case | Result |
|------|--------|
| Meta 9.77″/px (home rig) | VarAstro shows `48.85arcsec` (5×9.77) |
| Meta 0.65″/px (fine rig) | `_resolve_export_arcsec_per_px` → 0.65 |
| No derivable scale | Aperture arcsec line omitted; **never 1.3** |
| `#SOFTWARE` | `#SOFTWARE=VYVAR/1.0` from `VYVAR_SOFTWARE_VERSION` |
| Regression AAVSO body | Data row `REG_STAR,2460000.500000,12.500,0.010,…` unchanged |
| Full suite | **381 passed**, 15 skipped |
| Ruff | `export_reports.py`, `photometry_core.py` clean |

---

## Ledger

| Finding | Status |
|---------|--------|
| **G5-F002** | **RESOLVED (non-issue)** — fix-log step 7 |
| **G5-F007** | **FIXED** — fix-log step 8, commit **`770f062`** |

## Commit

`fix(export): derive AAVSO plate scale from WCS (no hardcoded 1.3); real software version (G5-F007)` — **`770f062`**

**Not pushed** — stop for Claude review.

## Files changed

- `export_reports.py`, `photometry_core.py`, `tests/test_export_g5_f007.py`
- `docs/VYVAR_FULL_AUDIT_LEDGER.md`, `CURSOR_RESULT.md`
