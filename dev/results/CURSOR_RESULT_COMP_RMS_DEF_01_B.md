CURSOR RESULT - 2026-08-25 (COMP-RMS-DEF-01-B STOP C3)

What I did
Wired COMP-RMS-DEF-01-B + ZONE-SAT-01 per C2f / Milan C3-GO.
C3-0 measured k on T3 R2 516 (read-only). Production default
`comp_rms_loo_photon_k=5`. Selector and suspected_variables share
`compute_loo_mag_rms_map`. Live 516/520 not written. Push after
`--fast --clean` OVERALL PASS (this STOP). C4 ZP-OK v2 is next.

## Premise (Rule 0.1)

**What is compared:** selector `comp_rms` after this wire (LOO mag
MAD vs the candidate pool, ceiling min(0.1, k x photon)) versus C2
fractional mag-bin relative-flux MAD gated as mag. Zone classifier
peak test versus C2 skip-when-85pct-NaN.

**How they differ:** gated column is `comp_rms_loo_mag`; old
statistic kept as `comp_relflux_mad`. No clipping. No QC-admit list
on the draft manifest (`comp_rms_frames_basis=all_loadable`).

## C3-0 k (pre-registered rule, no retune)

T3 R2 516: 70 unique live comps (task said 68), 134 proc frames.
iv = LOO MAD*1.4826 vs own ensembles; photon = 1.0857/snr (MS `snr`
when `snr_ap_pixscaled` absent on that product).

| | value |
|---|---|
| p50(r) | 2.514 |
| p90(r) | 3.672 |
| max(r) | 5.915 |
| rule | p90 round UP 1 sig; if 3<=p90<=5 then k=5 |
| **k** | **5.0** |

520 seven: photon NaN on live 520 MS (no snr column); iv matches C2.
JSON: `dev/results/session_20260825_closeout/c30_k.json`.

## Wires

C3-1  `src_py/comp_rms_loo.py`; attach in `comp_pool_rms.py`;
      pool call site `photometry_core.py`.
C3-2  `_write_suspected_variables` flags on LOO mag; CSV keeps
      `comp_rms_loo_mag`, `comp_relflux_mad`, `comp_rms`.
C3-3  Ceiling `min(phase01_comparison_max_comp_rms, k x photon)`.
      Missing `snr_ap_pixscaled` raises. Config
      `comp_rms_loo_photon_k` default 5.
C3-4  Peak: `peak_max_adu` else `peak_dao` else `flux`. If 85pct
      unresolved, test still runs against `saturate_limit_adu`.
      Stamps `zone_peak_column`, `zone_sat_limit_used`.

## C3-5 tests

`dev/tests/test_c3_comp_rms_loo.py`: old-stat 4.3 / LOO 0.016 passes;
LOO 0.187 vs photon 0.025 fails; clipped peak -> saturated.

## P-C3 STOP predictions (sandbox, no live writes)

P-C3-1 520 V0612 (T4 proc, seven honest pool): **HIT**.
n_pass=5 (G=7.63 and G=13.87 out). G=13.87 not selected.
Ensemble LC RMS vs those 5 = 0.0546 (<=0.06). Zone G=7.63:
on-disk `linear` (85pct NaN); new classifier `saturated`
(`peak_max_adu` 88781.5 vs limit 52428).

P-C3-2 516: unique live comps = 70. R2 MS has no
`snr_ap_pixscaled` (pre-C0b product). Using comparison_stars
max `snr` >=10: 42 D5-like IDs, **all 42 pass** when LOO is vs
that 42-set. Using C3-0 70-pool LOO + photon: **one miss**
`1496315070616056064` loo=0.0446 ceil=0.0437 **r=5.10**
(pinned member on TOI-3919 / FY CVn / CSS_J134925 / GH CVn /
RX CVn). Aperture LCs not regenerated; any ensemble that drops
this star will change. Full photometry not run (live read-only).

P-C3-3 suspected_variables: R2 file n=298 (old relflux). New LOO
on the 70 live comps: **0 flagged** (no live comp becomes
flagged). 520 seven: new flags = G=7.63 and G=13.87 (n=2);
T4 old file n=5.

516 on-disk zone=saturated rows: **24** (unchanged; 85pct already
populated on that product).

## Gates

`--fast --clean` OVERALL PASS recorded with this STOP commit SHA.
INV-COMP-RMS-01 documented. `--full` on frozen era03 MS will
**raise** until C6 recut stamps `snr_ap_pixscaled` (same class as
B-STOP-2 D3).

## Errors

None in unit tests. P-C3-2 "65 all pass" is a 1-star MISS on the
70-pool + photon reading (r=5.10); 42-set reading is HIT.

## Files changed

- `src_py/comp_rms_loo.py` (new)
- `src_py/comp_pool_rms.py`, `src_py/photometry_core.py`,
  `src_py/pipeline.py`, `src_py/config.py`
- `dev/validation/params_registry.json`, `docs/VYVAR_PARAMS.md`
- `dev/tests/test_c3_comp_rms_loo.py`, selector fixtures
- `docs/VYVAR_INVARIANTS.md` (INV-COMP-RMS-01)
- `dev/results/CURSOR_RESULT_COMP_RMS_DEF_01_B.md`
- `dev/results/session_20260825_closeout/c30_k.json`
- `dev/results/session_20260825_closeout/c3_pred.json`

Docs impact: INVARIANTS, PARAMS, STATE/JOURNAL/ROADMAP/DECISIONS.
Recurrence: photon ceiling needs `snr_ap_pixscaled` on MASTERSTAR
before `--full` can pass (C6 recut).
