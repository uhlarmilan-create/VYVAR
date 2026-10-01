CURSOR RESULT - 2026-10-01 LC-OUTLIER-01

What I did
Implemented LC epoch flagging with image evidence. Photometry values are
never altered; only flag / flag_reason change. Isolated spikes need stamp
evidence for artifact; without evidence they become spike_unconfirmed
(always shown, never export-excluded). Flare/eclipse runs (>=2 same-sign
elevated residuals) stay normal. Frame QC uses night MAD-sigma on
FWHM/elongation/sky from draft_manifest inspection.

## Why flagging stopped (git)
Commit 2a8355b800287edb67e84f15485bee620c45ed50
(2026-06-15, "Ship simple differential photometry (Workstreams A+B) for
V0612 closure") introduced skip_sigma_clip=_vsx_known in
apply_reporting_postprocess / detect_outliers. VSX-known targets such as
V1023 Her therefore kept flag=normal for all epochs. Global night MAD
clip (outlier_hi/outlier_lo) was also the wrong product rule (no image
evidence, no flare protection). Both are retired by this task.

Saturation flagging today: phase2a_target.py base_flags from
sat_flags / target_frames flag; photometry_phase2a read_flux_from_csv
sets flag="saturated" on is_sat; preserved via _PRESERVE_FLAGS.

## Fix summary
- NEW src_py/lc_outlier.py: isolated_spike_mask (cadence window =
  clamp(round(30min/median_dt), 3, 12)), image evidence (centroid,
  FWHM, elongation, peak/flux, annulus sky sigma, optional
  psf_chi2_locus_resid_sigma), frame_qc MAD-sigma, assign_lc_flags,
  export_keep_mask, EvidenceCache.
- src_py/photometry_phase2a.py: apply_reporting_postprocess returns
  (flags, flag_reasons); wires EvidenceCache + frame_qc; VSX skip gone.
- src_py/phase2a_target.py / method_lc_output.py: pass err/bjd/source/
  evidence; write flag_reason.
- src_py/photometry_lightcurve.py: LC CSV column flag_reason.
- src_py/export_reports.py: exclude frame_qc+artifact (+hard bad);
  keep spike_unconfirmed; log counts.
- src_py/ui_aperture_photometry.py: toggle hides frame_qc/artifact/
  saturated; always draws spike_unconfirmed (diamond); legend counts.
  Regression: toggle filtered to flag==normal only, so with all-normal
  CSVs it had nothing to hide (symptom of VSX skip, not a broken
  Streamlit rerun).
- src_py/photometry_report.py: PDF LC count line includes epoch flag
  class totals.
- Config keys (config.json + AppConfig + ui_settings + registry):
  lc_outlier_enabled (True),
  lc_outlier_n_sigma (5),
  lc_outlier_adjacent_sigma (3),
  lc_outlier_frame_qc_n_sigma (5),
  lc_outlier_evidence_n_sigma (5).
  Thresholds are MAD-sigma statistical conventions.

## Tests
dev/tests/test_lc_outlier_01.py T1-T6 + VSX regression: 7 PASSED.

## Verification draft 521
Artifacts: dev/results/context/session_20261001_lc_outlier_01/
- verify_draft_521.json
- stamp_v1023_frame_{001,036,110,137}.png (star + annulus, same stretch)

Class totals (all targets, post-reflag; photometry columns hash-identical):
  normal=6492, no_data=1961, artifact=14, spike_unconfirmed=48
  frame_qc night frames flagged=2 (inspection MAD); no LC epochs on
  those keys in this reflag pass (keys matched; frames may lack LC rows).

V1023 Her (1403049512185012992) epochs:
  036 artifact  artifact:annulus_sky_sigma_z=6.85
  110 artifact  artifact:annulus_sky_sigma_z=11.57
  137 artifact  artifact:centroid_offset_px_z=6.07,elongation_z=9.19,
                annulus_sky_sigma_z=14.80
  (+ 1 spike_unconfirmed elsewhere on the LC)

Elevated runs >=2 same-sign (|r|>3): 9 listed; 0 incorrectly flagged
as artifact/spike_unconfirmed.

Byte check: photometry column hashes identical pre/post (0 mismatches).
Only flag / flag_reason (and export consumers) differ. LC CSV content
hashes change; era re-cut is Milan's LOCK.

## Debt (out of scope; registered)
- HFR-UNITS-01: Frame QC HFR limit 5.00 fixed number; 26/140 frames
  "fail" at HFR ~9.4 while FWHM ~5.6 px -- likely unit/definition
  mismatch. Do not touch here.
- D1-1: Cosmic-ray rejection in calibration (still open).

## Gates
- --fast --clean: OVERALL PASS
  (1669+ passed after PARAMS regen + owner-count bump; log:
  session_20261001_lc_outlier_01/fast_clean.txt).
- Photometry values unchanged by design (unit + 521 byte check).

## Files changed
- src_py/lc_outlier.py (new)
- src_py/photometry_phase2a.py, phase2a_target.py, method_lc_output.py
- src_py/photometry_lightcurve.py, export_reports.py
- src_py/ui_aperture_photometry.py, ui_settings.py, photometry_report.py
- src_py/config.py, config.json
- dev/validation/params_registry.json
- docs/VYVAR_PARAMS.md, VYVAR_CONFIG_GUIDE_EN.md / _CZ.md
- dev/tests/test_lc_outlier_01.py, test_ui_params_dashboard.py
- Commit fix+tests: 3129af7
- RESULT commit: dd85094

STOP for Milan. Branch alpha-fixes-01 only; main untouched.
