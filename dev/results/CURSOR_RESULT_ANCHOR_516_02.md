# CURSOR RESULT - ANCHOR-516-02

Date: 2026-08-18
Task: isolate 515-vs-516 ERR drift by measurement; one-variable PFS-ON
Phase 2A; one-page anchor decision input. No re-cut.
Push: NOT authorized. 515 product SHA **de6f7c8** remains the reference.

Premise (0.1): 516-01 compared UI 516 (d5ef039, PFS OFF) to harness 515
(de6f7c8, PFS ON). MAG matched 48/48; ERR / comp_qa / skip_reason /
comparison_stars did not. Two variables were confounded (PFS, git tip
including 057ecdc). This task separates them. Outcome: **ERR identity
after the 515-style harness re-run, SHA still not de6f7c8**. PFS is
**not** the ERR mechanism.

## Part A - ERR decomposition (515 vs original 516 UI / PFS OFF)

Picks (max |delta err| + BO CVn): `1498321301379345408`,
`1497154753901690624`, `1498064771572297856`, plus BO
`1498613634033133184`.

Carrying term: **`err_photon` only**. `err_sem_rel`, `err_scint_rel`,
`err_sigma_sys_rel` max |delta| = **0** on all 48 LCs.

| term | max |delta| (all 48 LCs) |
|------|----------------:|
| err / err_photon | 6.589 **mag** on faint `1498321301379345408` |
| err_sem_rel | 0 |
| err_scint_rel | 0 |
| err_sigma_sys_rel | 0 |

516-01's "6.6 mmag" is a **unit defect**: the faint-star max is 6.589 mag
(err tens of mag). BO CVn max |delta err| on frame 014 is 6.628 mmag
(0.015493 vs 0.008865). BO median err: 8.945 mmag (515) vs 8.532 mmag
(516 UI).

BO frame 014 (`dao_flux=121563`, `sigma_bkg_ap=1664`, `g_pt=0.637`):
`sqrt(F/g + sigma^2)/F = 0.014156` matches **516 UI photon**, not 515's
0.006241.

### comparison_stars_per_target.csv

687/687 keys identical. BO 4-comp membership identical. **`comp_weight`,
`comp_rms`, `sigma_eff_mag` byte-equal** (weights did not differ).

Columns that differ:

- `saturate_limit_adu`: string only (`65535.0` vs `65535`)
- `saturate_limit_adu_85pct`: **52428.0** (515 = INV-SAT-LIMIT 0.80 x 65535)
  vs **55704.75** (516 = 0.85 x 65535)
- `_dist_deg`: float noise ~1e-14

MAG can stay identical while sat-limit columns differ because `delta_mag`
is an AIJ flux-sum (`photometry_core.py` ensemble path ~4322-4373).
`comp_weight` is ZP / SEM and did not move.

### comp_qa 96 vs 48

Writer: `write_comp_qa_artifacts` in `src_py/comp_qa_core.py` (~577-614).
One `comp_qa_{tid}.json` per `compute_comp_qa` target. **Does not delete
stale files.** 515 harness LC_TRIM only unlinks `lightcurve_*.csv`. 515
keeps 48 leftover `comp_qa_*.json` with no LC (8f107cf 96-LC era). 516
has 48/48.

## Part B - 516 Phase 2A PFS ON at d5ef039

Harness: `tmp/anchor_516_02_phase2a_pfs_on.py` (same override as 515
GAIN-PT: `cfg.per_frame_saturation_enabled = True`, `run_phase2a`, plus
`export_err_mode=calibrated`). Original UI PFS-OFF snapshot:
`dev/results/context/session_20260818_anchor_516_02/516_pfs_off_snapshot/`.
On-disk 516 is now the PFS-ON re-run. Nothing deleted.

Runtime: wall **840.0 s**, Phase 2A **583.3 s**, 48 LCs.

| SHA (core, n=97) | value |
|------------------|-------|
| 515 | `de6f7c8…` |
| 516 before (PFS OFF UI) | `6dc6ef2e…` |
| 516 after (PFS ON harness) | `d5f71ab1…` **? de6f7c8** |

LC science vs 515 after PFS ON: **MAG and ERR and err_photon max |delta|
= 0.0** on all 48 LCs (BO median err 8.945 mmag both). vs PFS-OFF
snapshot: MAG still 0; ERR still the Part A photon drift.

Skip after PFS ON: `per_frame_saturation`: 1 (CV CVn `1497007144465726080`
matches 515). `comp_qa` still 48 (516 never had 515 leftovers).

Task outcomes a/b/c as written: **none**. Closest label: SHA is (b)-shaped;
LC MAG+ERR are (a)-shaped. MAG did not differ (not c).

### Why SHA still differs after ERR match

Core byte diffs = 48 files:

- `comparison_stars_per_target.csv` (sat-limit columns, Phase 1)
- 47 LCs: **only `ct_n_comp`** (2363 on 515 vs 2346 on 516; max |delta|=17).
  `ct_c1` and MAG unchanged. 516-OFF already had 2346; PFS did not move it.

Extended SHA: 515 has **48 extra** `comp_qa_*.json` leftovers.

### 057ecdc

Not implicated for ERR. Same tip **d5ef039** (includes 057ecdc) + 515-style
headless `run_phase2a` reproduced 515 MAG+ERR. 057ecdc only pins PT
aperture 4.0 px in `gain_photon_transfer.py`; it does not enter the LC
photon formula.

### Named ERR mechanism (code + data; PFS is not it)

`_photometric_error_with_bkg_mode` has **no PFS branch**. `sat_limit`
in `read_flux_from_csv` only sets `flag`.

Two Phase 2A loaders:

1. **UI full pipeline** (`run_full_photometry_pipeline` ~18178-18190)
   passes `proc_frame_store` (full proc CSV columns, including
   `sigma_bkg_ap`). Empirical path:
   `var = F/g + sigma_bkg_ap^2` ? BO 014 photon **0.014156**.
2. **Headless `run_phase2a`** (515 GAIN-PT and this task) builds a
   `usecols` cache. `_needed_cols_2a` (~9352-9402) **omits
   `sigma_bkg_ap`**. Empirical sees NaN sigma and **falls through to
   Howell** (`_photometric_error`, 1989 eq. 2) ? BO 014 photon
   **0.00604** (LC stores 0.006241; same order, sky/RN rounding).

516 UI = (1). 515 product and 516 PFS-ON harness = (2). The Part B
"PFS ON" knob was confounded with switching loader (1)?(2). Prediction
(not re-run here): headless PFS **OFF** would still match 515 **ERR**
and would **not** match 515 skip_reason.

## Part C - One-page verdict (no re-cut)

| Observed delta | Named mechanism |
|----------------|-----------------|
| ERR / err_photon | UI ProcFrameStore (empirical Labbe) vs headless usecols cache that drops `sigma_bkg_ap` (Howell fallback). Not PFS. Not 057ecdc. |
| skip_reason CV CVn | PFS ON: `apply_per_frame_saturation_to_active_targets` then Phase 2A does not re-force `zone_flag==saturated` (~9529-1053, ~10396-10417). PFS OFF keeps `zone_flag`. |
| comp_qa 96 vs 48 | Writer does not unlink stale JSON; 515 leftovers from 96-LC era. 516 is 48/48. |
| comparison_stars sat-limit columns | Phase 1 tagging: 0.80 x 65535 (INV-SAT-LIMIT unresolved) vs 0.85 x 65535. Membership and weights identical. |
| SHA after PFS ON still ? de6f7c8 | sat-limit metadata + `ct_n_comp` (Phase 1 CT pool 2363 vs 2346) + 515 leftover QA files. Not MAG/ERR. |
| 516-01 "6.6 mmag" | Unit defect: 6.589 mag on a pathological star; BO is ~6.6 mmag. |

UI-default and harness-override runs on **one tip are not expected to
produce identical products**:

- `config.json` has `per_frame_saturation_enabled: false`; 515 used a
  per-instance override `True`.
- UI Phase 0+1+2A feeds ProcFrameStore (empirical ERR). Headless
  Phase 2A-only reads a column-subset cache (Howell ERR).

**Anchor run mode (decision input for 516-03, not executed):**

- To **carry de6f7c8 LC MAG+ERR numbers**: canonical is **headless
  `run_phase2a` without ProcFrameStore** (the 515 GAIN-PT path). PFS ON
  is still required for the CV CVn skip_reason label. Byte SHA still
  needs Phase 1 sat-limit/`ct_n_comp` identity and leftover-QA cleanup;
  PFS-ON Phase 2A alone cannot re-cut de6f7c8.
- de6f7c8 ERR is **Howell-by-omission**, not the configured
  `err_background_mode=empirical` Labbe path. The UI 516 product is the
  one that actually used measured `sigma_bkg_ap`.
- Architect + Milan choose in 516-03 whether the frozen ERR is (i) 515
  Howell-via-usecols, (ii) UI empirical, or (iii) a code fix to add
  `sigma_bkg_ap` to `_needed_cols_2a` then a new SHA.

On-disk note: `draft_000516` lightcurves are the PFS-ON harness product.
PFS-OFF UI snapshot is under `session_20260818_anchor_516_02/516_pfs_off_snapshot/`.
515 untouched.

## Runtime (Rule 0.3)

| part | runtime |
|------|---------|
| A (CSV/JSON measurement) | seconds (no pipeline) |
| B (Phase 2A PFS ON) | 840.0 s wall / 583.3 s Phase 2A |
| C (verdict + docs) | documentation only |

## Spec defects

1. 516-01 reported max ERR delta as 6.6 mmag; it is 6.589 **mag** on
   `1498321301379345408`. BO is ~6.6 **mmag**.
2. Task outcomes a/b/c do not cover "MAG+ERR match, SHA still differs".
3. One-variable PFS re-run was still confounded with UI ProcFrameStore vs
   headless usecols; code+data separate them (PFS does not enter photon).
4. `_needed_cols_2a` omits `sigma_bkg_ap`, so headless Phase 2A cannot
   honour `err_background_mode=empirical`. Not fixed in this task.

## Docs impact

None (measurement). STATE/ROADMAP/DECISIONS/ledger wait for 516-03.

Recurrence: n/a (measurement / decision input). Production defect
`_needed_cols_2a` vs empirical is named for 516-03, not patched here.

## Files

- `dev/results/CURSOR_TASK_ANCHOR_516_02.md`
- `dev/results/CURSOR_RESULT_ANCHOR_516_02.md` (this file)
- `dev/results/context/session_20260818_anchor_516_02/` (Rule 0.2:
  `part_a_summary.json`, `err_term_max_delta_all_lcs.csv`,
  `comparison_stars_diff.json`, `skip_reason.json`,
  `part_b_pfs_on.json`, `part_b_remaining_sha.json`,
  `photon_mechanism.json`, `516_pfs_off_snapshot/`)

## Errors

None blocking. No git push. No draft deleted. No code change.
