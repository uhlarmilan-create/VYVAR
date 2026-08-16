# CURSOR RESULT - WIDE-ERR-02

Date: 2026-08-16
Baseline draft: 515 (run SHA `da9cce4`)
Task: calibrated exported errors + weighted SEM
Push: NOT authorized
Outcome: **STOPPED at W1c** (W3 not implemented; W2 deferred)

Machine: `dev/results/WIDE_ERR_02_prod_components.json`,
`dev/results/WIDE_ERR_02_summary.json`
Harness: `dev/tools/wide_err_02_w1_dump.py`

---

## Verdict

W1b clears (sys+scint ~4.0 mmag at G8-9). **W1c STOPs**: production gain used
on NoFilter_60_2 is **3.17 e-/ADU** vs architect fitted effective **0.24-0.32**
(ratio to mid ~11x, far beyond the 2x gate). Honouring the task: do **not**
implement the W3 calibrated form or the weighted SEM in this wave - a white
scale + floor on top of a suspect photon term would mask the disagreement.

WIDE-ERR and SEM remain OPEN. WIDE-ERR-CROSSRIG remains OPEN.

---

## W1c - gain (STOP)

| quantity | value | source |
|---|---:|---|
| equipment_id | 1 | `draft_manifest.rig.equipment_id` |
| camera | QHY294MM / IMX492 | `EQUIPMENTS` ID=1 |
| EQUIPMENTS.GAIN_ADU | 3.17 | DB row |
| header GAIN | 0.0 (setting index) | FITS `BO_CVn_Light_001.fits` |
| resolved gain | 3.17 e-/ADU | `header_index_mapped` via `GAIN_SETTING_INDEX_MAP[1][0]` |
| pipeline_meta.dynamic_params.gain | 3.17 | stamped run |
| architect fitted effective | 0.24-0.32 e-/ADU | task ground |
| ratio model / fitted mid | **11.32** | STOP (>2x) |

Read noise used in the stamped run: **15.2 e-** (`dynamic_params`; binning 2x2).
EQUIPMENTS.READNOISE_E for ID=1 is 7.6.

---

## W1b - sys + scint at G 8-9 (OK to proceed; overridden by W1c)

Frame: production LC `err_*` columns, rel-flux -> mmag via `MAG_ERR_SCALE`.

G8-9 half-bin union has n=1 LC target at G=8.605; G8-10 context n=3
(includes FW G=9.18, BO G=9.72).

| quantity | G8-9 (n=1) | G8-10 context (n=3) |
|---|---:|---:|
| median sys mmag | 0.0 | 0.0 |
| median scint mmag | 4.00 | 4.00 |
| median sys+scint hypot mmag | **4.00** | **4.00** |
| median err_total mmag | 7.91 | 7.91 |
| median err_photon mmag | 2.31 | 3.52 |

One line: **sys+scint ~4.0 mmag** today where LC-frame truth is ~6-8 mmag -
the ~2x bright deficit vs that floor is reproduced; W1b does **not** STOP.
`sigma_sys_mag` for equipment 1 is unresolved (config map has key `"4"` only)
so sys is identically zero; scint alone is the floor term.

---

## W1a - component dump

- Per-star LC-target medians: `WIDE_ERR_02_prod_components.json` -> `per_star_lc`
- Per G bin (LC targets): `by_G_bin_lc_targets`
- Clean-star Howell recompute (production gain/RN/aperture on empirical F,sbk;
  SEM omitted): `by_G_bin_clean_recompute`

### Clean-star global-ZP scat / production photon (selected bins)

Frame: global-ZP scatter from `wide_err_515_empirical.csv` vs Howell with
g=3.17, RN=15.2, r_ap=3.999 px. **Not** the architect mid-range ratio~1 table
(that used fitted g~0.28).

| G bin | n | med scat mmag | med photon_prod mmag | med scat/photon |
|---|---:|---:|---:|---:|
| (8.0, 8.5] | 13 | 14.2 | 1.04 | **14.2** |
| (8.5, 9.0] | 12 | 14.5 | 1.32 | **11.0** |
| (11.5, 12.0] | 134 | 27.4 | 8.18 | **3.38** |
| (12.0, 12.5] | 192 | 36.6 | 11.7 | **3.12** |
| (14.0, 14.5] | 486 | 149 | 61.0 | **2.49** |

Mid-range underquote ~3x with production gain confirms W1c: the photon term
that matches the empirical CSV mid-range is **not** the production term.

---

## W2 / W3 / W3e

**Not run / not implemented.** Gate `implement_w3=false` after W1c STOP.
Tables W2b and W3e are empty by design in this result.

SEM weighted fix (ratio 0.677) remains parked with WIDE-ERR per standing
decision - must not ship alone; blocked with this STOP.

---

## Spec defects (named)

1. `equipment_id` lives under `draft_manifest.rig`, not top-level - first dump
   fell back to `cfg.gain=1.0` and false-fired W1c for the wrong reason.
2. Empirical clean set (2589) and LC targets (49) have **zero** `catalog_id`
   overlap on draft 515 - published `err_*` cannot answer W1a for the clean-star
   concept without recompute (SEM still missing for non-targets).
3. Architect "photon correct mid-range" refers to the fitted-gain empirical
   model; production Howell at 3.17 e-/ADU does **not** sit at ratio~1 mid-range
   in the global-ZP frame (~3x).
4. `sigma_sys_mag` config has equipment `"4"` only; wide rig equipment 1 -> 0.
5. W1b G8-9 on LC targets is n=1 in the strict (8,9] half-bin union; G8-10
   context used for sanity (still sys+scint=4.0).

---

## Register / docs

- WIDE-ERR: remains **OPEN** (not CLOSED)
- SEM item: remains **OPEN** (not closed with this task)
- WIDE-ERR-CROSSRIG: stays **OPEN**
- DECISIONS: WIDE-ERR-02 STOP entry recorded

---

## Files

- `dev/tools/wide_err_02_w1_dump.py`
- `dev/results/WIDE_ERR_02_prod_components.json`
- `dev/results/WIDE_ERR_02_summary.json`
- `dev/results/CURSOR_TASK_WIDE_ERR_02.md` (copied from drop)
- `dev/results/wide_err_515_empirical.csv` (copied from drop)
- `docs/VYVAR_DECISIONS.md` (WIDE-ERR-02 STOP)

## Errors

W1 dump exits 2 on STOP (intentional gate). No runtime failures after the
equipment_id / ASCII fixes.

## Baseline

`session_baseline_check.py --fast`: **OVERALL PASS** (1422 passed, 28 skipped;
tip `2396949` at check time).
