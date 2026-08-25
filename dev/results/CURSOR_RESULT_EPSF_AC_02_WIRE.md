CURSOR RESULT - 2026-08-24T16:30:00Z (EPSF-AC-02-WIRE)

What I did
Wired Milan GO (2026-08-24): production F6 AC = P4 (`psf_ac_policy=p4_none`),
internal PSF LC = P4, INV-PSF-LC-PIN-01 same-membership-or-NaN, EPSF-CORE-01
literature parameters on the ROADMAP row. Stamped live draft 516 sidecars
(invert stored AC; ADDITIVE-01 per file) and regenerated 60 internal PSF LCs.
Chi2 and flux reported separately. Not pushed; Milan authorizes.

Premise (0.1): AC-01 A3 614 mmag PSF-vs-aperture RMS was ensemble membership
drift, not AC. This task wires P4 (scalar off) and the pin rule so a missing
comp cannot silently renormalize the ZP.

## Gates

| Gate | Status | Detail |
|------|--------|--------|
| G0 AC-01 commit | PASS | `d206b43` docs (STATE/ROADMAP/DECISIONS AC finding); `876053a` AC-01 result/harness/artifacts. Unrelated DAO/VALID dirt left unstaged. |
| G1 tip | PASS | `876053ab92f42ac2de1a3795ab2ac1e0c81a2de5` is a descendant of `cf95c53`. |
| G2 `--fast` | PASS | 1524 passed, 32 skipped, OVERALL PASS. `test_exoplanet_local_match.py::test_comp` did **not** appear (watch item; no flake this run). git-origin-main WARN is local commits ahead of origin (do not pull/push). db-quick-check WARN via committed waiver. |
| G3 era | PASS | checker constants unchanged: core `9902d918` n=121 / ext `472bc9e4` n=179 |
| G4 production ePSF | PASS | SHA256 `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` unchanged |

## W1 - AC policy config + P4 F6 merge

New key `psf_ac_policy` (`p4_none` default, `chi2_lt5_legacy` named fallback).
Registry + `config.json` + AppConfig load/to_dict. ePSF dashboard read-only
caption. F6 merge stamps policy into `epsf_ac_merge_meta.json`, job summary,
`pipeline_meta.json`, per-star `psf_ac_policy` column, and PSF LC header.

Live draft 516: `stamp_p4_none_science_sidecars` on 134 `proc_*.csv` (no re-fit).
INV-PSF-ADDITIVE-01 asserted per file. `psf_ac_factor=1`, uncorrected flux.

Legacy formula `_compute_aperture_correction` unchanged (unit test: ratio 2.0
on a 6-star chi2<5 set). Explicit `p4_none` wins over `apply_aperture_correction=True`.

## W2 - INV-PSF-LC-PIN-01

Writer: full pinned set or NaN. Reason column `psf_epoch_drop_reason=comp_psf_fail:<id>`.
Aperture columns stay filled. Header diagnostics measured:
`# psf_lc_n_epochs_full`, `# psf_lc_n_epochs_dropped_pin`,
`# psf_ap_level_offset_mag`. Wired in `VYVAR_INVARIANTS.md` + `WIRED_INV_IDS`.
Synthetic test: one failed comp -> NaN + reason, not a renormalized ZP.

## W3 - regenerate + meters

n_written = 60. Stamp 134 sidecars. Elapsed 37.5 s. Hash guard OK
(128 must-not-change files identical). 66 expected-changed files differed
(60 PSF LCs + 6 sampled proc CSVs).

### Hash tables (split)

Must not change (sample; full set in `hashes_must_not_change_*.json`):

| product | SHA256 (before = after) |
|---------|-------------------------|
| production ePSF FITS | `172f95403beae36d...` |
| BO CVn aperture LC | `a5ea3980cc7c9785...` |
| BO CVn AAVSO | `2f593db24855c7bf...` |
| BO CVn VarAstro | `8eb26770bb9e5843...` |

Expected changed (wiring evidence):

| product | before | after |
|---------|--------|-------|
| BO CVn PSF LC | `d07d8f0ea4f505dd...` | `b6380bf5aea3df91...` |
| proc_BO_CVn_Light_001.csv | `ace9e77f80806912...` | `d14a6e76251c55e4...` |

### BO CVn meters (AC-01 A3 vs after)

| meter | before (A3, all finite epochs) | after (full-membership only) |
|-------|-------------------------------|------------------------------|
| PSF-vs-aperture delta_mag RMS | 614.3 mmag | **38.8 mmag** |
| level offset (psf - ap) | (mixed into RMS) | **+40.5 mmag** |
| RMS ~= \|median\| | n/a (jumps) | **yes** (offset, not scatter) |
| demeaned RMS | n/a | 9.9 mmag |
| coverage | 133/134 finite (partial ZP) | **23/134** (0.172) pin-survive |

Architect prediction: order-of-magnitude RMS drop. Measured 614 -> 39 mmag
(16x). RMS ~= \|median\| is true: leftover is a constant +40.5 mmag level
offset (A1 mag-slope class was ~76 mmag; measured offset is the same order,
smaller). Demeaned scatter 9.9 mmag. Frame 004 (was 0.385 vs 0.866): now
NaN PSF + aperture 0.866 still filled; reason
`comp_psf_fail:1499200223486564608`. Drop census: that comp 95 epochs, other
pinned `1497771992240531712` 16 epochs.

### FW CVn (different ensemble)

| meter | after |
|-------|-------|
| n_full / n_dropped | **0 / 134** |
| RMS / offset | undefined (no full-membership epoch) |
| drop reasons | `1497442379271632384` 130; `1499906247391001088` 4 |

The rule is not tuned to BO: a second ensemble hits the same NaN path, and
this one has zero joint-PSF coverage. Honest: no partial ZP.

### P4 invariance

Unit test `test_p4_invariance_scalar_ac_cancels_in_writer`: scale all
`psf_flux` by 0.528, rewrite; max \|d(delta_mag)\| < 1e-12 on finite
(full-membership) epochs. Matches AC-01 A3 (5e-15) on the new PIN writer.

## W4 - docs

- DECISIONS: Milan GO (a)-(d) + canonical-AC literature note (Stetson 1990
  DAOGROW / DOLPHOT; never chi2-gated DAO ratio).
- ROADMAP EPSF-CORE-01 rewritten with literature parameters (Godden &
  Blundell 2025 sample/osamp/smoothing/gridpoint/upstream watch). Newton
  desired, not the sole unlock. AC-01/AC-02 CLOSED rows.
- STATE: AC-01/AC-02 close; SHAPE-01 root OPEN, routed to EPSF-CORE-01.
- CITATIONS.bib: `stetson1990` added (code comment in
  `_compute_aperture_correction`). Godden & Blundell 2025 is docs-only; not
  added to the bib.

## Files touched

Code / config / tests (this task):
`src_py/config.py`, `config.json`, `src_py/params_registry.py`,
`dev/validation/params_registry.json`, `src_py/psf_photometry.py`,
`src_py/pipeline.py`, `src_py/epsf_psf_merge.py`, `src_py/photometry_core.py`,
`src_py/proc_frame_store.py`, `src_py/psf_internal_lc.py`,
`src_py/invariants_runtime.py`, `src_py/ui_epsf_dashboard.py`,
`dev/tests/test_psf_internal_lc.py`, `dev/tests/test_epsf_psf_merge.py`,
`dev/tests/test_epsf_dashboard_pct.py`, `dev/tests/test_ui_params_dashboard.py`,
`dev/scripts/epsf_ac_02_wire.py`,
`dev/scripts/_build_vyvar_params.py`.

Docs:
`docs/VYVAR_INVARIANTS.md`, `docs/VYVAR_DECISIONS.md`, `docs/VYVAR_ROADMAP.md`,
`docs/VYVAR_STATE.md`, `docs/VYVAR_PARAMS.md`, `docs/VYVAR_CONFIG_GUIDE_EN.md`,
`docs/VYVAR_CONFIG_GUIDE_CZ.md`, `CITATIONS.bib`,
`dev/results/CURSOR_RESULT_EPSF_AC_02_WIRE.md`.

Live draft 516 (PSF columns / PSF LCs only; aperture products hash-identical):
134 `proc_*.csv` under `detrended_aligned/lights/NoFilter_60_2/`,
60 `lightcurve_*_psf.csv`, `epsf_ac_merge_meta.json`, `pipeline_meta.json`
keys `psf_ac_policy` / `psf_ac_params`.

Artifacts: `dev/results/session_20260824_epsf_ac_02_wire/` and context copy
`dev/results/context/session_20260824_epsf_ac_02_wire/`.

## Errors (if any)

None on the science path. An earlier `--fast` started mid-edit failed
`test_apply_install_config.py::test_end_to_end_write_validates_and_drops_author_paths`
because `config.json` already had `psf_ac_policy` while AppConfig did not yet
(unknown-key ERROR). Re-run of that test after wiring: PASS.

## `--fast` close

G2 PASS: 1524 passed, 32 skipped, OVERALL PASS on HEAD `876053a` (dirty AC-02
tree). Watch `test_comp` did not appear. git-origin-main WARN: local series
ahead of origin (do not pull/push). T1 live-516 BO CVn test rewrote that
target's PSF LC via the same writer (expected).
