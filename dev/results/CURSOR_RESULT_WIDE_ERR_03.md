# CURSOR RESULT - WIDE-ERR-03

Date: 2026-08-16
Baseline draft: 515 (run SHA `da9cce4`); tip at task start `2396949`+
Supersedes: WIDE-ERR-02 W2/W3 (stopped at W1c)
Push: NOT authorized
Outcome: **S5e PASS** (harness); production path wired for container gain,
weighted SEM, and calibrated export.

---

## Verdict

S1a grid-of-4 PASS. Production had been applying native DB gain 3.17 e-/ADU
on container ADU (14-bit left-shifted into 16-bit). Photon-transfer gives
**g_pt = 0.637** e-/ADU_container, CI [0.443, 1.094], matching the architect
0.635 [0.44, 1.09]. S4b gate: LC-frame bins still outside [0.9, 1.1] after the
domain fix alone -> full Stage 5 `s`/`sigma_r` layer. After calibration,
median(scatter/err_exported) is in window for all evaluated G bins including
G8-9; G8-9 exported err stays above the 2.2 mmag scintillation floor.

Retraction (on record): the earlier fitted effective gain 0.24-0.32 is
superseded (g-k_sky degeneracy); the photon-transfer CI is the standing number.

---

## Stage 1

### S1a (gate PASS)

Artifact: `dev/results/WIDE_ERR_03_S1a_grid.json`

Raw bin2 light: **100%** of pixels on residue 0 (`rint(ADU) mod 4`).
Calibrated and detrended/aligned frames redistribute residues ~uniformly
(float dark/flat and resampling break the integer lattice).

### S1b

See `dev/results/WIDE_ERR_03_S1.json` (`s1b_gain_call_sites`). Science sites
that consumed `resolve_gain` / Howell assumed e-/ADU matching the flux ADU
domain but received native 3.17 on container ADU. RN is always in e-;
`(RN/g)^2*npix` mis-scales when g is in the wrong ADU domain.

### S1c (unit chain - also in DECISIONS)

QHY294MM digitizes a 14-bit native sample and stores it in a 16-bit FITS
container by a left shift of two bits, so container ADU = 4 * native ADU
(raw lights sit on a residue class mod 4; S1a). The equipment DB gain 3.17
e-/ADU is the native-domain conversion; the matching container-domain gain is
therefore 3.17/4 = 0.7925 e-/ADU_container (photon-transfer on draft 515 gives
g_pt ~ 0.635 e-/ADU_container inside CI [0.44, 1.09], which excludes bare
3.17). Hardware/software 2x2 binning on this CMOS path sums container-domain
electrons and ADU together, leaving e-/ADU_container unchanged to first order;
bias/dark/flat calibration and later resampling mix the integer grid into
floats but do not change the ADU scale away from the container domain.
Production photon and SNR terms that consume flux and sky in container ADU
must therefore use g_container (g_pt when available, else DB/4), never bare
DB gain; read noise remains in electrons and enters only as (RN/g)^2*npix in
ADU^2.

---

## Stage 2

Artifact: `dev/results/WIDE_ERR_03_S2.json`

| quantity | value | domain |
|---|---:|---|
| g_pt | 0.63707 | e-/ADU_container |
| CI (Theil) | [0.443, 1.094] | e-/ADU_container |
| n_frames | 134 | - |
| aperture_r_px | 3.999 | px |
| authority | g_pt | - |
| S2c bare DB vs g_pt ratio | 4.99 | fires WARN (>2x) |
| S2d | PASS | inside architect CI |

Module: `src_py/gain_photon_transfer.py`. Sidecar:
`.../photometry/gain_photon_transfer.json`.

---

## Stage 3

`resolve_sigma_sys_mag`: missing equipment key -> **explicit 0.0 with INFO log**
(no silent zero). Equipment 1 today: sys=0 with that log; Stage 5 owns the
residual floor. Registry unit = mag.

---

## Stage 4 (gate)

Artifact: `dev/results/WIDE_ERR_03_S4.json`

| frame | G bin | med ratio old (g=3.17) | med ratio new (g_pt) |
|---|---|---:|---:|
| global-ZP | (12.0, 12.5] | 2.84 | **1.076** |
| LC comps | (12.0, 12.5] | 1.26 | 1.133 |
| LC comps | G8-9 | - | scatter 9.24 / err 6.78 = **1.38** |

S4b: **full_s_sigma_r_calibration** (bins outside [0.9, 1.1]).
Architect prediction: G8-9 err~6.5 vs truth 6.7-8.2 - measured err 6.78 mmag,
scatter 9.24 (slightly above the 6.7-8.2 band; thin calib still justified).

Variable guard: fires (synthetic drop + suspected_variables excluded).

---

## Stage 5

Artifact: `dev/results/WIDE_ERR_03_S5.json`
Sidecar: `.../photometry/err_calibration.json`

- Form: `err_exported^2 = (s * err_model)^2 + sigma_r^2` per G bin
- Weighted SEM: `ensemble_sem_mag_from_residuals_weighted` in production
  `ensemble_normalize` when `comp_weight_map` present
- `export_err_mode`: calibrated (default) | model
- AAVSO `#ERR_MODEL=...` comment line; VarAstro parity line
- Fire proofs: chi2 pre G8-9 ratio 1.34 fires; post 1.000; var guard; equal-w
  SEM identity; unequal-w differs; empty calib = identity

### S5e acceptance

**PASS.** All evaluated bins median(scatter/err_exported) in [0.9, 1.1];
G8-9 exported err >= 2.2 mmag.

Note: acceptance is harness-applied on the corrected model for draft 515.
On-disk LC `err` columns from run `da9cce4` are still pre-fix until a
Phase 2A re-export with the new code + sidecars.

---

## Stage 6 / register

- DECISIONS: WIDE-ERR-03 entry (S1c paragraph, authority rule, retraction)
- GAIN-DOMAIN-01: opened and CLOSED in this task
- WIDE-ERR + SEM: CLOSED if S5e harness PASS (this report)
- WIDE-ERR-CROSSRIG: stays OPEN
- Citations: Tamuz+2005, Pont+2006, Kovacs+2005, Gillon 2009, Winn 2008
- SysRem remains out of v1 scope

### Impact statement

Exported **MAGERR / `err` / `err_*` budget columns** change (container-domain
gain, weighted SEM, optional s/sigma_r). Photometric **magnitudes**
(`mag_calib*`, `delta_mag`) are unchanged by this task. Prior err-dependent
diagnostics (chi2-per-rig tables, P-02 bright-floor ratios, WIDE-ERR-02 dumps)
are stale by intent; re-read against this task / `err_calibration.json` +
`gain_photon_transfer.json`.

---

## Spec defects named

1. LC-frame meter approximates mag_calib with LOO flux-sum delta_mag, not full
   pytics-weighted mag_calib (XVAL-BO-01 lesson: close, not identical).
2. S5e PASS is harness-side; published draft 515 LC files need a re-export to
   carry new err columns on disk.
3. Weighted SEM separable commit not split here (single task wave; push N/A).

---

## Files (code)

- `src_py/gain_photon_transfer.py` (new)
- `src_py/err_calibration.py` (new)
- `src_py/sigma_floor_core.py` (weighted SEM; sigma_sys log)
- `src_py/photometry_core.py` (container gain authority; SEM; calib apply)
- `src_py/export_reports.py` (`#ERR_MODEL` comment)
- `src_py/config.py` / `params_registry.py` / `dev/validation/params_registry.json`
- `docs/VYVAR_PARAMS.md` (regenerated)
- `CITATIONS.bib` (Kovacs, Gillon, Winn)
- `dev/tools/wide_err_03_*.py`
- `dev/tests/test_wide_err_03_gain.py`

## Baseline

`session_baseline_check.py --fast`: **OVERALL PASS** (1429 passed, 28 skipped).
