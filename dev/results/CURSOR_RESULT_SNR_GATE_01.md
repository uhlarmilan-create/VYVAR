CURSOR RESULT - SNR-GATE-01 - 2026-08-14

Register ID: SNR-GATE-01
Follows: DAO-DEPTH-01 (localization accepted)
Status: COMMITTED at 956770c; `--fast` OVERALL PASS on content tip; push await Milan authorization.

Artifacts:
- `dev/results/SNR_GATE_01_f1_per_frame.csv`
- `dev/results/SNR_GATE_01_f1_summary.json`
- `dev/results/SNR_GATE_01_f2_discarded_phot.csv`
- `dev/results/SNR_GATE_01_f2_kept_phot.csv`
- `dev/results/SNR_GATE_01_f2_summary.json`
- `dev/results/SNR_GATE_01_f6_summary.json`
- `dev/results/SNR_GATE_01_f6_masterstars_A.csv`
- `dev/results/SNR_GATE_01_f6_masterstars_AB.csv`
- `dev/tests/test_snr_gate_01_sky_mad.py`

================================================================
1. F0 -- purpose and history
================================================================

Purpose (recoverable from code + `docs/VYVAR_DAO_DETECTION.md`):
before Gaia matching, drop weak DAO peaks under a median + k*sigma floor on peak
ADU so spurious faint detections do not inflate DAO_ONLY / match noise.
Config: `masterstar_prematch_peak_sigma_floor` (default k=1.8).

DAO-PHYS-2 already showed retuning k alone cannot remove DAO_ONLY excess without
depth loss; that is orthogonal to the estimator defect found here.

Estimator history:
- Call sites historically passed `plain_mean_med_std(..., sigma=3.0, maxiters=3)`,
  implying a clipped background scale.
- Commit `c9e1f8f` (2026-08-12, "remove all science-path sigma-clip") introduced
  `plain_mean_med_std` that **ignores** `sigma`/`maxiters` and returns full-frame
  sample std. Prematch kept the same call shape but suddenly measured scene
  variance (stars + large-scale structure). Same class as audit T3-1.

Purpose is recoverable => INV-GATE-REMOVAL does not authorize quiet removal.
The gate stays; the noise scale and pass-2 applicability change.

S1-S3 re-verified on draft 435/512 MASTERSTAR (F2/F6): cliff is current-code;
historical inverted sigma ~83 ADU vs today's plain std ~570; plain std flat vs
40% sky change.

================================================================
2. F1 -- sigma^2 vs sky (G-R0)
================================================================

Physical model (stated before fitting):

  sigma^2 = sky_ADU / g + RN_ADU^2

Pre-registered tolerances (`SNR_GATE_01_f1_summary.json`):
- T_SLOPE_NSIG = 3.0 (slope > 0 at >=3 sigma)
- T_GAIN_LO/HI = 0.05 .. 50 e-/ADU
- T_R2_MIN = 0.25
- Equipment reference: QHY294MM GAIN_ADU=3.17, READNOISE_E=7.6 (binning caveat)

Draft 512, 134 frames, sky ~1329-2413 ADU (ratio 1.81).

| Estimator | qualify | slope a | R^2 | g_imp | notes |
|-----------|---------|---------|-----|-------|-------|
| plain_full (current) | FAIL | -4.93 | 0.055 | n/a | scene variance; G-R0 DISQUALIFY |
| mad_sky_le_median | PASS | 0.341 | 0.700 | 2.94 | preferred; near 3.17 |
| pct5_40_sky | PASS | 0.134 | 0.698 | 7.49 | ok but gain far |
| photutils_bkg2d_rms | PASS | 0.651 | 0.524 | 1.54 | ok |
| dao_convolved_rms | FAIL | ~0 nsig | 0.001 | - | scene-scale |

G-R0 fires on plain_full and dao_convolved_rms.

Production choice: `sky_mad_sigma_adu` = median + 1.4826*MAD on pixels <= median
(`src_py/plain_stats.py`). Implied gain 2.94 vs equipment 3.17 (~7%).

================================================================
3. F2 -- are discarded recoveries real stars (G-R1)
================================================================

Draft 512 replay (broken gate): 3614 prematch in, keep 735, floor~2427.
All matched discards are pass-2 (`n_discard_pass1=0`).

Photometry vs Gaia G:
- Discarded G12-G15 share ZP ~-22 with kept stars; flux_floor_frac=0 (no noise pile-up).
- Scatter rises with G (0.14 -> 0.60) as expected for real faint stars.

Cross-frame (134 frames):
- Discarded median detect fraction = 1.0; >=50% of frames for 90%.
- G12-G14 nearly perfect; G15 median ~0.51 still mostly star-like.
- Bright control G<11: fraction 1.0.

G-R1: discarded population behaves as real stars at least through G14-15 on this
night. Discarding them was loss of real data. Depth limit is not set by this
broken gate; the magnitude where behaviour degrades is ~G15 (repeatability).

================================================================
4. F3 -- should a global gate apply to pass-2
================================================================

Decision (implemented):
1. Delta A -- replace prematch sigma with sky MAD (keep gate for pass-1).
2. Delta B -- exempt pass-2 from the global peak gate (local annulus test +
   catalogue seed already applied).

Arguments:
- For A: G-R0 requires a noise estimator; plain_full fails; MAD passes.
- For B: S4 conflict (local admit, global overruling) is indefensible once F2
  shows the discarded set is real. Classical forced photometry measures and
  flags; it does not drop catalogue-seeded recoveries on a second global scene
  statistic.
- Against removing pass-1 gate entirely: purpose recovered (F0) -- still useful
  against blank-sky pass-1 noise peaks before match.
- Against a single merged criterion only: pass-1 and pass-2 already use different
  local/global detection physics; unifying them is a larger redesign, not required
  once A+B restore consistency.

================================================================
5. F4 -- literature / other packages
================================================================

| Tool | Below-threshold catalogue source | Noise for detection |
|------|----------------------------------|---------------------|
| DAOPHOT / IRAF daofind | Detection catalog only; no ASSOC. Forced photometry is a separate aperture/PSF measure at fixed coords; non-detections retained with large errors / flags, not omitted as a silent gate after a local find. | Local / sky-annulus style background; not full-frame scene std. |
| IRAF apphot | Measures at given coords; reports mag/err even when faint. | Annulus sky. |
| SExtractor | DETECT_THRESH (global or local via WEIGHT); ASSOC matches detections to an input list -- unmatched ASSOC positions are absent unless a separate forced step is used (Bertin & Arnouts 1996; SExtractor manual ASSOC). | BACKGROUND / BACK_FILTER; RMS from background map. |
| photutils | DAOStarFinder uses threshold * background RMS (userable); ForcedPhotometry / aperture photometry at fixed positions always returns a row with flux+/-err. | Background2D / MAD / sigma_clipped_stats on sky; not unclipped full-frame sample std for science noise. |
| sep | extract() threshold relative to noise array; no built-in ASSOC drop gate. | User-supplied ERR / global RMS. |
| AstroImageJ | Multi-aperture photometry at placed apertures; keeps rows, reports SNR. | Annulus. |
| VaST | Detection + association; faint non-detections typically missing from per-frame detection lists unless forced. | Local sky. |
| LSST forced photometry | Measures flux at fixed coordinates regardless of SNR>5 detection; non-detections retained (can be negative flux) with errors/flags (LSST Science Pipelines; DP1 forced-photometry tutorial). | Per-visit PSF / variance plane. |

VYVAR difference (pre-fix): pass-2 is targeted re-detection (not classical forced
aperture), then a second global scene-std gate **omitted** real catalogue-seeded
recoveries. That omission pattern matches none of the forced-photometry packages
above (they measure and flag). After SNR-GATE-01: pass-2 kept; pass-1 still has a
sky-noise peak floor.

================================================================
6. F5 -- why INV-NOCLIP-01 did not fire
================================================================

Verdict: **pattern mismatch by design**, not a scope miss and not a deliberate
acceptance of scene-std as noise.

INV-NOCLIP-01 patterns target:
- astropy `sigma_clip` / `SigmaClip` / `sigma_clipped_stats`
- one-sided annulus sky **value** clip (`sky_pixels < sky_med + ...`)
- named iterative ensemble clip helpers

They do **not** match `peak > median + k*sigma` detection floors. Broadening
NOCLIP to ban all median+k*sigma forms would false-positive on DAOStarFinder
itself and on every legitimate significance cut.

The real defect was the silent estimator regression in `plain_mean_med_std`
(`c9e1f8f`): kwargs still looked like clipped stats; values became scene std.

Scanner change in scope: document this boundary in `dev/tools/iron_gates_scan.py`
(comment on NOCLIP_PATTERNS). Fire-proof extended via
`dev/tests/test_snr_gate_01_sky_mad.py` (sky MAD responds to sky; plain full std
does not; NOCLIP does not flag detection-floor form). No new regex that would
ban detection thresholds.

================================================================
7. F6 -- implementation and measurement
================================================================

Code:
- `src_py/plain_stats.py`: `sky_mad_sigma_adu`
- `src_py/pipeline.py`: prematch uses sky MAD; `vy_dao_pass` tagging; pass-2 exempt
  by default (`prematch_exempt_pass2=True`); meta records estimator + flag
- `src_py/photometry_core.py`: `_noise_floor_adu_from_image_array` **pinned** to
  legacy `plain_mean_med_std` (SNR-GATE-02). Prematch does **not** go through this
  helper; see Part 1/2 below.

Draft 512 MASTERSTAR replay (k=1.8, dao_threshold_sigma=3.8):

| Stage | n masterstars | noise_floor ADU | bg_sigma |
|-------|---------------|-----------------|----------|
| Baseline archived (broken) | 735 | ~2427 | ~570 plain |
| Delta A (sky MAD, pass2 gated) | 3614 | ~1445 | ~24.3 MAD |
| Delta A+B (sky MAD, pass2 exempt) | 3614 | ~1445 | ~24.3 MAD |

Separable deltas (G-R4):
- A_minus_baseline_n = +2879
- AB_minus_A_n = 0 on this draft (correct sky floor already keeps all pass-2 peaks)
- B remains architecturally required when peaks sit near a correct floor

By Gaia G (matched counts, archived vs A/AB):

| G bin | baseline | after A/AB |
|-------|----------|------------|
| <10 | 113 | 113 |
| 10 | 102 | 103 |
| 11 | 245 | 246 |
| 12 | 225 | 503 |
| 13 | 11 | 860 |
| 14 | 15 | 1271 |
| 15 | 7 | 355 |
| nan (DAO_ONLY etc.) | 15 | 163 |

Comparison-star pool proxy (non-variable matched): 666 -> 3341.

**BO CVn baseline (corrected SNR-GATE-02):** catalog_id `1498613634033133184`
- trust GREEN; reason: 5 clean comps, noisy LC (informational)
- check_scatter (S1) = 0.009300
- ac_scatter = 0.013283
- lc_rms_ooe (S7) = 0.046659
- comps (5x TIER1): 1499053747922698240, 1497771992240531712, 1497368849430107904,
  1499200223486564608, 1497974027502858240
- SPARSE FIELD warning for this target: **no**

**Second target (NOT BO CVn):** `1496795041799526400` = R CVn -- previously mislabelled
as the baseline. Informative SPARSE FIELD case: trust RED, check_scatter 0.008297,
lc_rms_ooe 0.008444, ac_scatter NaN, 5x TIER4, SPARSE warning **yes**.

AFTER LC metrics: **not re-measured** (no Phase-1 reprocess in this task).

Binning note (record): 2x2 block-sum leaves fitted slope 1/g unchanged
(`var(S)=4 var(s)=S/g`), so F1 gain 2.94 vs equipment 3.17 is a direct comparison.

================================================================
8. Pre-registered rules that fired
================================================================

Quoted:

- G-R0 (estimator): "Any candidate that fails the sigma^2 versus sky linearity
  test in F1 is not a noise estimator and is disqualified" -- fired on
  `plain_full` and `dao_convolved_rms`.

- G-R1 (reality): "If F2 shows the discarded population behaving as real stars
  ... discarding them is a loss of real data" -- fired; depth ~G15.

- G-R3 (no target count): obeyed; 3614 is the measured answer, not a target.

- G-R4 (separability): A and B implemented/measurable separately; on 512, A
  carries the numerical delta, B is zero but retained.

- G-R2 (purpose): purpose recovered; gate not quietly removed.

================================================================
9. G-R5 -- affected existing results
================================================================

All drafts / MASTERSTAR catalogs / completeness curves / comparison pools built
under code after `c9e1f8f` (2026-08-12) that used the prematch peak gate with
`plain_mean_med_std` full-frame std are affected. Do not re-derive conclusions
here; list only:

- Draft 512 MASTERSTAR depth, Gaia completeness (`g_lim_*`, completeness curve),
  comparison-star tiers / SPARSE FIELD warnings, BO CVn and other target LCs
  locked to that master list.
- Draft 435 (and peers) when reprocessed or interpreted under current code:
  replay keeps ~755 vs historical ~2951.
- Any post-c9e1f8f anchor checksum / star-count baselines tied to MASTERSTAR
  length or G12-15 membership.
- DAO-DEPTH-01 localization remains valid; its deferred items DAO-SNR-SIGMA-01
  and DAO-PASS2-vs-PREMATCH-01 are addressed by this task.
- UI DAO-STARS suggestions that assumed scene-std noise floors.
- MASTERSTAR zone / `noise_floor_adu` / `bg_sigma_adu` written from det_meta on the
  next MASTERSTAR rebuild (coupled to the gate meta; not an aperture-photometry path).

Unaffected in this commit (pinned or not on the path): SNR aperture radii,
proc-CSV annulus sky / Howell errors, trust gates, comparison selection criteria,
DAO detection threshold (convolved RMS / sigma_pp).

Unaffected in kind: calibration physics, alignment, Howell annulus sky after
SKY-CLIP-01, color-term math -- except insofar as denser catalogues change
ensemble membership after an authorized rebuild.

================================================================
10. Register diff for authorization
================================================================

| ID | disposition | status |
|----|-------------|--------|
| SNR-GATE-01 | FIX: sky MAD prematch sigma; pass-2 exempt; tests | COMMITTED (see Part 6 SHA) |
| SNR-GATE-02 | Surgical land + pin non-prematch consumers; verify | DONE with SNR-GATE-01 commit |
| DAO-DEPTH-01 | successor SNR-GATE-01 | SUPERSEDED-FIX |
| DAO-SNR-SIGMA-01 | closed by Delta A | CLOSED |
| DAO-PASS2-vs-PREMATCH-01 | closed by Delta B | CLOSED |
| SNR-DEPTH-01 | F2 G15 repeatability depth; no cut in this fix | DEFERRED |

This commit invalidates the `--full` anchor and the P1 golden ledger SHA again.
Draft 512 rebuild is the follow-up that produces "after" light-curve numbers.
No push without Milan.

Files in the SNR-GATE-01 commit:
- `src_py/plain_stats.py`, `src_py/pipeline.py`
- `src_py/photometry_core.py` (pin docstring / restore plain helper for SNR-table path)
- `dev/tools/iron_gates_scan.py`, `dev/tests/test_snr_gate_01_sky_mad.py`
- `dev/results/SNR_GATE_01_*`, `dev/results/SNR_GATE_02_part5_verify.json`, this memo
- `docs/VYVAR_AUDIT_2026_REGISTER.md`, `docs/VYVAR_DECISIONS.md`, `docs/VYVAR_DAO_DETECTION.md`

================================================================
Could not measure
================================================================

- Full draft-512 Phase-1 reprocess after MASTERSTAR rebuild (LC after metrics).
- Implied RN from MAD fit intercept (negative; NaN in F1 JSON).

================================================================
11. SNR-GATE-02 -- Part 1 impact inventory
================================================================

| what | where | what it feeds | value changes? | intended here? |
|------|-------|---------------|----------------|----------------|
| `sky_mad_sigma_adu` | `plain_stats.py`; used in `pipeline.detect_stars_and_match_catalog` prematch | prematch peak floor + det_meta bg_sigma/noise_floor | yes | YES |
| `plain_mean_med_std` | many modules (DAO fallback, platesolve, alignment, PSF, ...) | various local med/std | no (API unchanged) | n/a |
| `_noise_floor_adu_from_image_array` | `photometry_core.py` | `estimate_median_sky_adu_per_px_for_snr_table` -> precompute SNR table sky input | would change if sky_mad; **pinned** to plain | NO -- pinned |
| `estimate_median_sky...` / `precompute_and_save_snr_aperture_table_for_draft` | photometry_core | early `aperture_snr_table.json` before Phase 2A overwrite | unchanged (pinned helper) | NO |
| Phase 2A `_median_sky_from_phase2a_csv_cache` | photometry_core | final `aperture_snr_table.json` sky from proc `noise_floor_adu` | no (reads annulus sky written as that column) | NO |
| proc CSV `noise_floor_adu` | written = annulus `sky_pp_arr` | Howell legacy fallback; Phase 2A sky median | no | NO |
| `sigma_bkg_ap` / Howell err | photometry_core | exported uncertainties | no | NO |
| trust / check_scatter / comp selection | trust_flag_core, comp_selection | trust flags, tiers | no until rebuild changes master list | NO in this commit |
| DAO detection threshold | convolved RMS / `_pixel_noise_sigma_pp_adu` | DAOStarFinder threshold | no | NO |
| MASTERSTAR zone annotation | `_annotate_masterstars_flux_zones` via det_meta | zone labels on rebuild | yes on rebuild only | coupled to gate meta; not aperture radii |

Explicit answers:
1. Noise floor helper feeds SNR-table **precompute** sky estimate. Final draft table is
   Phase 2A overwrite from annulus sky. With pin: radii do not change. Measured
   Phase 2A rebuild vs archived: **max abs delta = 0.0 px** per mag bin.
2. Photometric error model uses annulus sky / `sigma_bkg_ap` / pre-sub sky surface;
   `noise_floor_adu` in proc is annulus sky. Not fed by prematch sky MAD. Unchanged.
3. Trust / QC / comp selection do not read the prematch estimator. Unchanged until
   an authorized MASTERSTAR rebuild changes the candidate pool.
4. DAO detection threshold is separate (convolved RMS / sigma_pp). Unchanged.

================================================================
12. SNR-GATE-02 -- Part 2 scope decision
================================================================

| Consumer | Decision | Why |
|----------|----------|-----|
| Prematch peak gate (`pipeline`) | CHANGE (sky MAD + pass-2 exempt) | Authorized scientific fix |
| `_noise_floor_adu_from_image_array` | PIN to `plain_mean_med_std` | Would move SNR-table precompute sky (~2922 vs archived path); inseparable third delta |
| SNR aperture table / radii | leave untouched | Part 5 shows zero radius delta |
| Proc photometry / dao_flux / errors | leave untouched | Annulus path; Part 5 dao_flux max |rel|=0 |
| Trust / comps | leave untouched | No Phase-1 reprocess |
| DAO threshold | leave untouched | Different estimator |

Finding for later authorization (not in this commit): replace the SNR-table sky
estimate (misnamed noise-floor `med+k*std`) with a real sky or measured bkg_var-only
path -- own measured delta on radii.

================================================================
13. SNR-GATE-02 -- Part 4 what the fix does not do
================================================================

4.1 Gate inert on draft 512: n_dao_after_spatial_cap=3614 in, n_rows=3614 out.
Floor 1444.93 vs sky median 1401.12; no detection below floor. Rejected ~80% before,
nothing after. Delta B is zero because Delta A left nothing to act on. A gate that
currently rejects nothing is not a calibrated gate; whether it should ever reject
is OPEN.

4.2 DAO_ONLY: 12/735 (1.6%) -> 163/3614 (4.5%). Natural false-positive indicator.
TODO-13 drops DAO_ONLY before photometry, so photometry star counts are unaffected
by these rows; they still matter for MASTERSTAR QA / completeness diagnostics.

4.3 F2 depth: G14 frac_median=0.955; G15 frac_median=0.507, frac_p16=0.170, n=355
in bin. G-R1 says depth is set by measurement. No depth limit implemented.
Deferred as **SNR-DEPTH-01** (decide after rebuild, not before).

================================================================
14. SNR-GATE-02 -- Part 5 verification
================================================================

Artifact: `dev/results/SNR_GATE_02_part5_verify.json`

- Aperture radii (Phase 2A rebuild from 134 proc CSVs vs archived table):
  sky match 1553.3518371582031; FWHM match 3.389154740967996;
  **max_abs_delta_phase2a_rebuild = 0.0**
- dao_flux on `BO_CVn_Light_004.fits` for BO CVn + 5 TIER1 comps vs proc CSV:
  **max abs relative difference = 0.0**
- Iron-gate fire proofs: `dev/tests/test_iron_gates.py` -- 8 passed
- SNR-GATE unit tests: 3 passed
- `dev/tools/ascii_migrate.py --check`: migrated_or_would=0, stop=0

================================================================
15. SNR-GATE-02 -- Part 6 commit
================================================================

- Commit tip: **956770c** (authoritative: `git rev-parse HEAD`)
- `--fast`: **OVERALL PASS** (1341 passed, 27 skipped; measured on `4af0aa7` before memo SHA self-edits; tree identical aside from this memo)
- Message: SNR-GATE-01: sky-MAD prematch noise; pass-2 exempt from global peak gate.
- Not pushed. Milan authorizes.

This commit invalidates the `--full` anchor and the P1 golden ledger SHA again.
Draft 512 rebuild produces the "after" light-curve numbers.
