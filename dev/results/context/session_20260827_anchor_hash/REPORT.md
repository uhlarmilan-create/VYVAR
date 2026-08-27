# ANCHOR-HASH-01 REPORT

Date: 2026-08-27. Architect: Claude. Implementer: Cursor.
Base: origin/consolidate-01 @ d6c84e0. Branch: consolidate-01. ASCII.

Premise: era04 core SHA 9367f998 hashed raw bytes of lightcurve_*.csv
including `# git_hash=<HEAD>` in PSF LCs, so the gate measured the
commit. Compared: the same freeze files under a content hash that
drops provenance header lines. Frozen bytes are unchanged.

## Header-key census (era04 snapshot, read-only)

Aperture LCs (n=53): no `# ` header lines.

PSF LCs (n=53) keys present on every file:

- git_hash, git_dirty
- epsf_model_file, epsf_model_sha256, epsf_n_stars, epsf_build_timestamp,
  epsf_oversampling, epsf_smoothing_kernel, epsf_cutout_size
- psf_weight_mode, psf_err_mode, psf_ac_policy, psf_ap_level_offset_mag,
  psf_lc_n_epochs_full, psf_lc_n_epochs_dropped_pin,
  psf_zp_membership_effective, psf_zp_membership_rig_validated
- ensemble_n_comp, ensemble_pinned_ids, ensemble_source
- gain_authority, product
- bare lines: INTERNAL DIAGNOSTIC banner; PSF absolute-scale notes

Not present on era04 LCs: files, generated, timestamp, vyvar_version.

Census JSON: header_census.json.

## Exclusion list (final)

Dropped before hashing (regex
`^# (git_hash|git_dirty|files|generated|timestamp|vyvar_version)=`):

- git_hash
- git_dirty
- files
- generated
- timestamp
- vyvar_version

Everything else (column data and all other header lines, including
epsf_build_timestamp) stays in the hash.

## v1 / v2 on freeze (read-only)

| | sha | n |
|--|-----|---|
| core v1 (raw, history) | 9367f99848c14b43016321d000ec53651c9b260290bcb37afd2f6bab5035b2d7 | 160 |
| ext v1 (raw, history) | d3cefff3240b4874d9b0ba3f76f7a303a5e3ea8b83f051149202d5b9c65d6863 | 210 |
| core v2 (content, gate) | af218acd32a4892cc4f0030168829852ced5c5140f83575301c1a39869437e66 | 160 |
| ext v2 (content, gate) | ada5caff61692ff0489631e6278efedd8c92cb9bd26d05fcb67f2fb3729b1676 | 210 |

Recorded in dev/validation/anchor_manifest.json as sha_core / sha_ext
(v1) and sha_core_v2 / sha_ext_v2. Ledger VL-ANCHOR-WCSINV keeps v1
and adds v2. DECISIONS: "era04 hash v2 (content), v1 9367f998/d3cefff3
kept as history".

## Commits

| SHA | What |
|-----|------|
| ba59864 | content hash; v2 constants; --full v1 identity + provenance DRIFT + G3 meters; G7 writes PSF LCs |

## Gates

| Gate | Status | Numbers |
|------|--------|---------|
| G1 before | PASS | d6c84e0 --fast --clean 1597 passed, 32 skipped; clean-tree PASS. Log: g1_before.txt |
| G2 --full | PASS | ba59864. pytest 1601 passed, 32 skipped. pipeline 1402s. v2 core af218acd n=160 / ext ada5caff n=210. v1 raw identity: 107 identical, 53 PSF differ (all _psf.csv). Funnel 253/197. counters phase2a_empty_comp_drop=3. science-compare 53/53. provenance DRIFT git_hash ba59864 vs freeze 9d47e2b (informational). Log: g2_full.txt. Work: tmp/session_baseline/20260827T133728Z |
| G3 | measured | From 1d --full PSF LCs, same demeaned RMS as EPSF-ZP-OK-01 (sqrt(mean((x-median)^2))*1000 on finite psf_delta_mag). BO 1498613634033133184: 0/134 nan mmag (ref 8.495). FW 1497343732462852864: 0/134 nan mmag (ref 5.218). Cause: INV-PSF-LC-PIN-01 all-epoch drop (comp_psf_fail) on freeze/--full PSF products. 8.495/5.218 were live 516 W2 regenerate, not these files. JSON: g3_zp_ok_meters.json |
| G4 live 516 | PASS | csv bfa24039778f437b / fits 13e77cf8a1dcb4e7 / epsf 172f95403beae36d |
| G7 --parity | PASS | ba59864 sequential one-copy. W2 1320s + 53 PSF/21s; W1 1365s + 53 PSF/27s. v2 core=af218acd n=160 ext=ada5caff n=210. Log: g7_parity.txt |
| G1 after | PASS | ba59864 --fast --clean 1601 passed, 32 skipped; clean-tree PASS. Log: g1_after.txt |

## STOPs

None vs the 53-PSF-only v1 identity rule.

G3 is a measurement, not a miss vs 8.495: the --full PSF LCs are the
freeze pin-drop product (0 finite), not the live-516 ZP-OK W2 product.

## Files changed

- dev/tests/photometry_sha.py
- dev/tests/test_photometry_sha_content.py
- dev/tests/test_photometry_sha_baseline.py (v1 raw flag)
- dev/tests/test_invariants_p1_seed.py (v1 raw flag)
- dev/scripts/session_baseline_check.py
- dev/validation/anchor_manifest.json
- dev/validation/VYVAR_VALIDATION_LEDGER.json
- docs/VYVAR_DECISIONS.md
- this session folder

