CURSOR RESULT — 2026-06-12

What I did
Ran **TOI-1131.01.b** V-band (78 pre-cal FITS) end-to-end → **`draft_000391`** (`V_60_2`). Validated **B2+B1** common-mode comp-stability fix on a **78-frame, ≥3-comp** baseline. **Committed** the fix (no anchor re-baseline — short Chi_and_H cut unchanged).

Scripts: `scripts/toi1131_night_run_v.py`, analysis `tmp/toi1131_analysis.json`, run log `tmp/toi1131_night_run.log`.

---

## Preconditions

| Check | Result |
|-------|--------|
| **Gaia catalog** | Field RA/Dec **248.83°, +61.61°** (FITS WCS, PLTSOLVD). **`GAIA_DR3/zaloha/vyvar_gaia_dr3.db`** (G≤16, **40.8M** stars — not Chi_and_H-only): **283** stars in 0.5° cone (G 7.7–16.0). **No TAP field DB needed.** |
| **Rig / plate scale** | Equipment set **#2**: C3-26000 + DDT 300/1200, Dablice. MASTERSTAR median **1.301″/px** (bin2); FWHM ~2.5 px. `draft_manifest`: **`pre_calibrated`**. |
| **Calibrated path** | `pre_calibrated_mode=True` → `non_calibrated/lights/V_60_2`. |
| **PSF / detrend** | PSF OFF; sysrem/savgol/democratic OFF. |

---

## Run summary (`draft_000391`)

| Item | Value |
|------|-------|
| **draft_id** | 391 |
| **Setup** | `V_60_2` (78 frames, ~1.40 h baseline) |
| **night_run_success** | **false** — completeness **7/8 (87.5%)** |
| **Failure** | Target **`1625467932661420928`** (Gaia DR3 — **TOI host from FITS naming**) skipped: Phase-1 `unhashable type: 'dict'` → Phase-2A **no comp stars** |
| **Other targets** | 7 LCs + trust + PDF |
| **PDF overflow** | **0** (`verify_pdf_overflow.py --draft 391 --obs-group V_60_2`) |
| **Gaia→DAO completeness** | **4.3%** (low — fine-scale + G≤16 depth; flag for follow-up) |
| **Crowding** | `density_class: sparse`; field not blend-dominated at 1.3″/px |

---

## Science — TOI-1131.01.b host

**Primary science target did not get an LC** (see failure above). Proc CSVs show the star measured on frames; pipeline bug blocked comp selection.

**Secondary field targets (7 LCs):**

| Target | n_frames | baseline (h) | lc_rms (mmag) | lc_quality | trust | max dip (mmag)* |
|--------|----------|--------------|---------------|------------|-------|-----------------|
| ASASSN-V J163431.81+613840.6 | 78 | 1.40 | 47.8 | good | YELLOW | −131 |
| ZTF J163831.02+614701.9 | 78 | 1.40 | 49.2 | good | YELLOW | −97 |
| GH Dra | 78 | 1.40 | 36.4 | good | RED (4 clean) | −46 |
| CzeV4348 | 78 | 1.40 | 38.6 | good | RED (3 clean) | −67 |
| others (Gaia IDs) | 78 | 1.40 | 40–56 | good | YELLOW/RED | −73 to −122 |

\*Dip = min(mag_calib) − median — intrinsic variability / sampling, **not** confirmed exoplanet transit for TOI host.

**Transit verdict:** **No LC for TOI host** → transit assessment **not possible** this run. Variable-star neighbours show **50–130 mmag** excursions over 1.4 h (expected for VSX-selected field).

---

## Common-mode fix validation (78 frames, ≥3 comps)

Replay `check_comparison_stability` on draft proc LCs (**k=3.0** vs **k=0.0** magnitude-only):

### Common-mode detrend (example: 8-comp target ZTF J163831…)

```
[STABILITY] Common-mode detrend: 28.67 mmag/hr removed from 7 comps
```

Sorted BJD → stable shared trend removal (not the ~97 mmag/hr under-removal from Step A).

### Significance gate (same target)

| k | slope-excluded comps | ensemble good comps |
|---|----------------------|---------------------|
| **0.0** (old) | **6** (slopes 6–18 mmag/hr, **σ ≈ 0.4–0.7**) | 2 |
| **3.0** (new) | **0** (slopes insignificant) | 7 (+1 p2p-excluded) |

p2p outlier **`39668480`** stays **excluded** at both k (`p2p=0.105 > thr=0.0955`) — genuine outlier path intact.

### Field-wide A/B (6 targets with ≥3 comps)

| Metric | Value |
|--------|-------|
| **Comp disposition flips** (k0↔k3) | **19** |
| **max \|Δmag\|** (ensemble re-norm replay) | **105 mmag** (target ZTF J163831…, ens 2→7) |
| **max \|Δlc_rms\|** | **5.7 mmag** |

Interpretation: magnitude-only k=0 collapsed ensemble to **2 comps** on several targets (wrong); k=3 restores **7–8** comps. Peak frame Δmag ~**2× lc_rms** when ensemble size jumps 2→7 — **bounded, explainable** (zeropoint/ensemble composition), not a per-frame noise blow-up. **Transit-shaped signal not detrended** (target LC path untouched).

**Note:** `pipeline_meta.common_mode_stability_detrend` was **false** on disk (meta records last-target flag only) — replay confirms detrend **did run** per target with ≥20 frames.

---

## Lock decision

Validation **passes**: correct common-mode removal, insignificant slopes kept, p2p/significant exclusions preserved, bounded LC effect. **Fix committed**; anchor **`203254fd…` unchanged** (no re-baseline).

---

## Errors / flags

1. **TOI host LC missing** — Phase-1 `unhashable type: 'dict'` on `1625467932661420928` (separate bug; not introduced by B2+B1).
2. **Completeness gate** 87.5% — same root cause.
3. **Gaia→DAO 4.3%** — fine-scale Newton; consider G≤17 field DB if comp pool thin (zaloha was sufficient here for 7/8 targets).
4. **`pipeline.py` proc perf changes** left **uncommitted** (separate session).

---

## Files changed (commit)

- `photometry_core.py`, `config.py`, `config.json`, `ui_settings.py`
- `citations.py`, `CITATIONS.bib`, `method_lc_output.py`
- `tests/test_comp_stability.py`, `tests/test_export_citations.py`
- `docs/VYVAR_*`, `scripts/toi1131_night_run_v.py`, `CURSOR_RESULT.md`
