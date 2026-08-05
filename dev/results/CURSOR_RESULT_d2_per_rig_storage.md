CURSOR RESULT - 2026-08-05 (Task D2)

What I did
Design-only deliverable for per-rig storage (group (a)). **No implementation** — Milan approval for storage option (A vs B) was not recorded in the task input, per task §2.

## 1. Premise check — decomposition of group (a)

After C'' scope triage and D1 reclassification of `crowding_tighten_min_fwhm_px` from group (b) ? (a), the registry holds **19** `scope_group=a` keys (was 18 before crowding move). Breakdown with evidence:

### Already resolved at runtime — nothing to store (3)

| key | evidence |
|-----|----------|
| `gain` | `param_resolver.resolve_gain` — FITS header first, then DB `GAIN_ADU` (`database.py` equipment join) |
| `read_noise` | `param_resolver.resolve_read_noise` — DB `READNOISE_E` first |
| `plate_solve_fov_deg` | `param_resolver` / header WCS + optics; `kind=resolved`, `owner=fits_dynamic` in registry |

### Per-rig dict exists — key or shape defect (2)

| key | defect | evidence |
|-----|--------|----------|
| `sigma_sys_mag` | Key is `equipment_id` only; should be `(equipment_id, telescope_id)` pair | `sigma_floor_core.py:129-157` — `dict[str, float]` lookup by `str(equipment_id)` |
| `k2_defaults_bprp` | Flat `{band: value}` — no rig dimension | `config.py:894`, `k2_extinction.py:168-177` band-only override loop |

### Detector / calibration facts — need new per-rig storage (10)

| key | why rig-specific |
|-----|------------------|
| `admission_sat_peak_frac` | Full-well / headroom vs rig SATURATE |
| `saturate_limit_fraction` | Same |
| `cal_diag_sat_warn_frac` | Cal-library saturation diagnostics |
| `bpm_dark_mad_sigma` | Dark BPM sensitivity vs read noise/gain |
| `calibration_master_ccd_temp_tolerance_c` | CCD set-point habit per camera |
| `calibration_library_native_binning` | Native binning of sensor |
| `osc_channel_binning` | OSC rig extraction policy |
| `err_background_mode` | Empirical vs model background error path may differ by rig depth |
| `blind_use_rig_prior` | Blind solver FOV/plate-scale prior |
| `apply_color_term` | Filter/detector system may not support CT on all rigs |

### Legitimately px-native / sampling (1, from D1)

| key | note |
|-----|------|
| `crowding_tighten_min_fwhm_px` | Undersampling gate; px-native by physics (D1 verdict) |

### Deferred — low confidence DAO thresholds (3, out of scope per task §4)

| key | note |
|-----|------|
| `masterstar_dao_threshold_sigma` | draft_501 shows DAO_ONLY *fraction* differs, not proven optimal *threshold* |
| `sips_dao_threshold_sigma` | same |
| `qc_dao_detection_sigma` | same |

**Corrected count:** 3 + 2 + 10 + 1 + 3 = **19** keys in group (a) today.

## 2. Storage design — (A) config nested dict vs (B) DB table

### (A) Nested dict in `config.json`, keyed by rig slug

| pros | cons |
|------|------|
| Matches `sigma_sys_mag` precedent | Config grows with every rig; hard to keep in sync with equipment records |
| Diffable in git | config?UI parity harder (287+ registry keys) |
| No schema migration | Values not tied to `ID_EQUIPMENTS` / `ID_TELESCOPE` lifecycle |
| | k'' provenance columns awkward in flat JSON |

### (B) DB table on `(ID_EQUIPMENTS, ID_TELESCOPE, …)` — **recommended**

| pros | cons |
|------|------|
| Sits beside `GAIN_ADU`, `READNOISE_E`, `SATURATE_ADU`, `FOCAL`, `PIXELSIZE` (`database.py:1153-1161`) | Schema migration + UI surface |
| Values travel with hardware record | Second lookup path (mitigated by single resolver) |
| Natural provenance columns (measured_at, draft_id, source) | |
| Fixes k'' shape: `{band: value}` ? `(rig, band) ? (value, source)` | |
| Same camera / two OTAs ? two rows (plate scale, k'', sigma floor differ) | |

**Recommendation:** **Option B (DB table)**. Per-rig values are equipment facts or measured calibrations, not user tuning knobs. Provenance for k'' is a science deliverable requirement (F-B01 class). Config should hold policy/mode keys only; magnitudes belong with the rig record Milan already maintains in `EQUIPMENTS`/`TELESCOPE`.

**Implementation blocked** pending Milan's explicit A/B choice.

## 3. Resolver design (for whichever storage wins)

Template: `resolve_sigma_sys_mag(equipment_id, cfg)` ? extend to:

```python
def resolve_rig_param(
    key: str,
    *,
    equipment_id: int | None,
    telescope_id: int | None,
    band: str | None,  # k'' only
    cfg: AppConfig,
) -> tuple[float | None, str]:  # value, provenance token
```

**Rules (Milan 2026-08-05):**
- Rig identity = `(OBS_DRAFT.ID_EQUIPMENTS, OBS_DRAFT.ID_TELESCOPE)` (`database.py:1205-1206`).
- k'' keys on `(rig, band)`; band from obs_group filter token or `VY_CHANNEL` (OSC).
- **Fail-safe, never fail-closed:** missing per-rig row ? current global fallback + one INFO log naming key and fallback source.
- No silent `except: pass`.

**Call sites (prototype scope):**
- `sigma_sys_mag` — all LC error assembly using `resolve_sigma_sys_mag` (`sigma_floor_core.py`, photometry error path).
- `k2_defaults_bprp` — `_literature_k2_bprp_for_obs_group` / night-fit branch (`k2_extinction.py:152-183`).

## 4. k'' provenance in report/PDF (four cases)

| case | report should say | `K2Source` token |
|------|-------------------|------------------|
| No override, band in `JOHNSON_K2_ZERO_TOKENS` | k'' = 0 (AAVSO practice) | `LITERATURE_DEFAULT` (computed zero) |
| No override, band with literature k'' | k'' = `<value>` (Smith/Henden converter) | `LITERATURE_DEFAULT` |
| Per-rig measured override | k'' = `<value>` measured on `<rig label>`, `<date/draft if known>` | **`RIG_MEASURED`** (new token — not `LITERATURE_DEFAULT`) |
| Night fit (v2) | k'' = `<value>` fitted this night | `NIGHT_FIT` (existing) |

V/R/I per-rig measured value is a **deliberate override** of the zero default; report must name `RIG_MEASURED`, not literature.

## 5. P1 golden result

Not re-run for D2 (design-only, no code). D1 bisect shows ledger `VL-P1-GOLD` core SHA `e7976de1…` (commit `a9d7eb0`) ? current HEAD `7620077` output `aa72e979…` even without D2/D1 code — see `CURSOR_RESULT_d1_unit_normalisation.md` §5.

## 6. Migration note — remaining 9 group (a) keys

After prototype on `sigma_sys_mag` + `k2_defaults_bprp`:

| batch | keys | pattern |
|-------|------|---------|
| Saturation trio | `admission_sat_peak_frac`, `saturate_limit_fraction`, `cal_diag_sat_warn_frac` | Single float per rig; optional `SATURATE_ADU` derivation |
| Calibration habits | `bpm_dark_mad_sigma`, `calibration_master_ccd_temp_tolerance_c`, `calibration_library_native_binning`, `osc_channel_binning` | Float/int per rig |
| Policy flags | `err_background_mode`, `blind_use_rig_prior`, `apply_color_term` | String enum per rig |
| Px-native | `crowding_tighten_min_fwhm_px` | Float per rig; no unit normalisation |

DAO threshold trio stays deferred pending empirical sweep.

## Files changed

- `dev/results/CURSOR_RESULT_d2_per_rig_storage.md` (this file only)

**No code commits for D2 implementation.**
