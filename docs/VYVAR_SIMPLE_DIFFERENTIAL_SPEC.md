# VYVAR — Simple Differential Photometry: PRODUCTION SPEC

Date: 2026-06-15  
Status: **Workstream A landed** (DoD-A PASS). **Workstream B** — grounded fix audited, implementation pending.

---

## Validated (V0612, draft_407 g, full production path)

- `delta_mag`: **0.0113** pre-eclipse RMS / **0.949** shape corr vs AIJ (~0.011), **7 comps**.
- Mechanism: `temporal_binning_enabled=False` (ALG-3 fix) + tier-ladder selection +
  `comp_select_rms_floor`=1e-6 dropping C3 `1112109595786459136` (isolated_bin artefact).

## NOT yet validated (Workstream B)

1. **Reporting columns.** AAVSO: `mag_calib`/`mag_calib_ac`. PDF LC: `mag_calib_ct`/`mag_calib`.
   V0612 post-finalize: corr **~0.37** on shipped `mag_calib` (target-fit airmass detrend).
2. **Generalization.** One ground-truth field validates mechanism; gate recommends >=1 more field
   before blind global flip.

---

## Workstream A — promote differential (DONE)

1. **Binning OFF default** — `config.py`, `config.json`, `ui_settings.py`, `config_schema.md`.
2. **Phase-1 selector** — `_assign_comp_tiers_to_pool` ? `_select_comps_by_color_then_rms` +
   `comp_select_rms_floor` (retired `_select_comps_tiered` on BP-RP path).
3. **Defaults:** `apply_color_term` off; tier ladder [0.15, 0.30, 0.55] ? cap 0.79; n_comp 3/8;
   `comp_select_rms_floor` 1e-6 (hidden, config-only).

**DoD-A:** `tmp/phase10/dod_a_production_defaults.py` — PASS (2026-06-15).

---

## Workstream B — reporting / export integrity (GROUNDED, not yet coded)

**Problem:** validated science in `delta_mag`; user-facing surfaces read post-processed `mag_calib*`.

**Supersedes earlier B1/B2 fork** (B1 “guard airmass detrend” was not physically grounded).

### Code audit (2026-06-15, read-only)

| Item | Evidence |
|------|----------|
| Target-fit airmass | `airmass_detrend_lc` (`photometry_core.py:3594-3644`) fits **`mag_calib = a·airmass + b` on the target curve**, not comp-derived k. Wired at `:7483-7487` on `mag_for_airmass`. |
| Outliers unguarded | `detect_outliers` (`:3323-3360`): global median+MAD, no eclipse mask. V0612: 2× `outlier_lo` on ingress. |
| `delta_mag` untouched | Saved at `:7594`; outlier/airmass rewrite only `mag_calib*` (`:7486-7496`). |
| Where shape breaks | DoD-A LC: `corr(delta_mag, mag_calib_raw)=0.998`; `corr(delta_mag, mag_calib)=0.59` after detrend (slope ~0.78 mag/am). |

### Grounded fix (three parts)

1. **Reported mag = `delta_mag` + ensemble ZP** (Honeycutt 1992; per-frame ZP from constant comps).
2. **Drop per-target airmass detrend on the reporting path** for colour-matched differential
   (Plavchan et al. arXiv:0704.3584; Dhillon PHY217). Residual extinction from comp ensemble only.
3. **Mask-first known-variable guard on `detect_outliers`** (out-of-eclipse clip; extended eclipse
   mask — arXiv:2402.16018; consistent with democratic detrender arXiv:2411.09753).

**DoD-B:** shipping columns corr >= ~0.94 vs AIJ on V0612 (near-equality with `delta_mag` once wired).

See `docs/VYVAR_DECISIONS.md` (Reporting-column fix) and decision-grounding rule.

---

## Gate — global default risk

Validate on >=1 additional ground-truth field (flat star + different night/rig ideal). Risk decision,
not hard block.

---

## Discipline

Sandbox ? measure ? review ? commit. **No commit until DoD-B passes on shipping columns.**
