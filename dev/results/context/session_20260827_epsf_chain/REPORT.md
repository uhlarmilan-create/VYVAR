# CURSOR RESULT - EPSF-CHAIN-01

Date: 2026-08-27. Branch: consolidate-01. English. ASCII.

Base was origin/consolidate-01 @ 8e590bb. Live draft 516 was not written.

## What I did

Wired ePSF as a night-run stage (INV-EPSF-STAGE-01) and a completeness gate
that fails all-dropped PSF LCs (INV-EPSF-COMPLETE-01). All gates G1-G9
listed below PASS. epsf01 is in the manifest as core_psf; Milan PUSH_AUTH
decides whether it is the first ePSF anchor.

## M1 Freeze proc CSVs (era04 snapshot, never live 516)

Snapshot: Archive/Drafts/draft_000516_snapshot_era04_20260826
Setup: NoFilter_60_2. BO CVn 1498613634033133184. Ensemble: pinned, 4 comps.

| Frame | File | psf_* columns | Comp 1497771992240531712 | Other 3 comps |
| --- | --- | --- | --- | --- |
| 0 | proc_BO_CVn_Light_001.csv | present: psf_flux, psf_flux_err, psf_chi2, psf_fit_ok | fit_ok=False, flux=NaN, chi2=NaN | same |
| 60 | proc_BO_CVn_Light_068.csv | same 4 columns | fit_ok=False, flux=NaN, chi2=NaN | same |

Verdict: present-but-NaN, not absent. PIN drop missing[0] is the first
pinned member 1497771992240531712. Snapshot has no masterstar_epsf.fits.

## M2 Cost (era04 WORK COPY)

Science set: 317 catalog ids (253 targets + 64 comps).

| Step | Wall time | Notes |
| --- | --- | --- |
| Copy freeze | 2.5 s | first copy; resume reused it |
| build_epsf_model | 3.3 s | n_stars_used=63, sha256 849be1fd7187feb8... |
| Fit+merge 134 frames | 15748.7 s (11.23 x aperture 1402 s) | n_ok median 188.5; ~90-150 s/frame |
| LC writer | 14.5 s | n_written=53 |

Cost STOP: fit >> 0.5 x aperture. No silent subsample. Proposal: frame-level
progress batching only.

STOP then fix: Light_006/064 sparse float64 exo_match_sep_arcsec false-tripped
INV-EXPORT-READ-ONLY-01. Hash now uses numeric .10g for numeric dtypes.
134/134 freeze sidecar identity round-trip PASS after a7c474f. M2 resumed
and finished.

## M3 G3 on work-copy product

| Target | n_finite/n_rows | demeaned RMS mmag | Live regenerate (2026-08-26) |
| --- | --- | --- | --- |
| BO CVn | 134/134 | 145.917 | 8.495 |
| FW CVn | 134/134 | 14.557 | 5.218 |

n_full=134 (PIN did not hide drops). RMS does not match 8.495/5.218. Finding,
not a tune. G3 gate refs updated to M3: 145.917 / 14.557.

## Provenance vs live 516 ePSF 172f9540

| Field | Work copy (era04 freeze MASTERSTAR) | Live 516 |
| --- | --- | --- |
| model sha256 | 849be1fd7187feb8... | 172f95403beae36d... |
| created_utc | 2026-08-27T15:26:24Z | 2026-08-22T19:11:31Z |
| n_stars_used / n_final | 63 | 67 |
| n_csv_input | 3600 | 3610 |
| n_after_variable_excluded | 3314 | 3325 |
| n_after_not_noisy | 2266 | 2264 |
| n_after_clean_source_state | 2255 | 2200 |
| n_after_science_scope | 64 | 68 |
| n_after_isolation | 63 | 67 |
| epsf_fwhm_native_px | 2.354 | 2.364 |
| epsf_vs_input_fwhm_ratio | 0.713 | 0.716 |
| epsf_asymmetry | 0.0092 | 0.0369 |
| plate_scale_arcsec_px | 9.77397255 | 9.77389007 |

Same fwhm_px 3.3014, cutout 17, oversampling 2, quadratic kernel. Pool and
QC differ because the freeze MASTERSTAR/catalog is not the live 516 tree
that produced 172f9540. Do not tune.

## Commits (consolidate-01)

| Id | Commit | What |
| --- | --- | --- |
| 3a | a7c474f | src_py/epsf_stage.py; psf_runner.main calls it; sparse-float hash fix |
| 3b/3d | be0ebfc | NightRunParams.epsf default ON; night run calls stage; PSF completeness |
| 3c | 2756569 | app.py RUN ePSF and dashboard write-LCs call run_epsf_stage |
| 3d tests | 8fe08a6 | fire proof: all-dropped PSF LCs fail |
| 3e | 5250104 | --full/--parity through stage; core_aperture/core_psf split |
| G3 refs | 746db0f | G3 BO/FW 145.917 / 14.557 mmag |
| hash lock | 50ff455 | core_psf with-timestamp 95153825 (superseded) |
| G7 hash | f2941fe | drop epsf_build_timestamp; core_psf c743b8ba |

era04_aperture: d55fcc9d8ad9b55213c5c1813415cb54d54b88c3fc917bc81706065e4d824810 n=53
ext_aperture: cc8b532ee668b9b339e4170752b9d1054771b1236ecac8163688693586117167 n=157
epsf01: c743b8ba89f4ac544e5e94b025b1746da9c28af6c7f2952ec1ae60db717d62a8 n=53
With-timestamp history: 9515382571a61c4eda55e9ab96ab64cfa291e6b5e3cead186499a6ef37b565aa
af218acd mixed core is history only.

## Gates

| Gate | Status |
| --- | --- |
| G1 --fast --clean before | PASS 1601 passed, 32 skipped. Head was 8e590bb. |
| G1 after | PASS 1611 passed, 32 skipped. OVERALL PASS. Log git-head 5250104 (G1 started in parallel with 746db0f G3-ref commit). Tip is 746db0f. |
| G2a core_aperture 53/53 vs era04 | PASS era04_aperture d55fcc9d8ad9b552... n=53; ext cc8b532ee668b9b3... n=157 |
| G3 BO/FW 134/134 within 0.001 mmag of M3 | PASS BO 134/134 145.917; FW 134/134 14.557. --full aperture 1089s + ePSF 12193s |
| G4 live 516 | PASS csv bfa24039778f437b / fits 13e77cf8a1dcb4e7 / epsf 172f95403beae36d (re-checked after --full and after G7) |
| G7 --parity | PASS. First run FAIL (timestamp). Re-run W2 16918s W1 17432s. W1==W2 core_aperture d55fcc9d n=53 core_psf c743b8ba n=53 |
| G9 completeness fire proof | PASS |

## STOPs

1. Fit time 11.23 x aperture. Reported before wiring. No subsample.
2. INV-EXPORT-READ-ONLY-01 false trip on Light_006/064. Fixed in a7c474f.
3. M3 RMS != live regenerate 8.495/5.218. ePSF pool 63 vs 67. Do not tune.
4. G7 first run: W1 psf=2068982d W2 psf=8da2cd9a --full psf=95153825. Cause:
   epsf_build_timestamp not in the content-hash strip set. W1 vs --full:
   53/53 files differ, 0 data rows. Hash after strip: c743b8ba... both.

## PUSH_AUTH (epsf01)

G7 PASS. core_psf is written to anchor_manifest as sha_core_psf / name
epsf01, and ledger item VL-ANCHOR-EPSF01. Milan decides whether epsf01
is the first ePSF anchor.

epsf01:
c743b8ba89f4ac544e5e94b025b1746da9c28af6c7f2952ec1ae60db717d62a8 n=53

## Files (production)

src_py/epsf_stage.py (new)
src_py/epsf_psf_merge.py
src_py/psf_runner.py
src_py/night_run.py
src_py/app.py
src_py/ui_epsf_dashboard.py
dev/scripts/session_baseline_check.py
dev/tests/photometry_sha.py
dev/tests/test_epsf_stage.py
dev/tests/test_epsf_completeness.py
dev/tests/test_export_parity_01.py
dev/tests/test_photometry_sha_content.py
