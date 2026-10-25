# VYVAR injection-recovery test protocol

Test record of the injection-recovery suite (INJECTION-2026-10). Each chapter is one run; the method
section applies to all of them. Tool: `dev/tools/injection_recovery.py`. Data of each run:
`dev/validation/injection_recovery/<night>/`.

## 1. Purpose

Signals of known shape and amplitude are injected into the real frames of a night; the night goes
through the normal VYVAR night pipeline (the same path as RUN VYVAR in the GUI); what VYVAR
measures is compared with what was injected. The test answers:

1. **Fidelity.** Is the amplitude of a variation preserved (no suppression by the aperture
   photometry, the comparison-star ensemble or the light-curve post-processing)?
2. **Timing.** Is the time of minimum of an eclipse preserved?
3. **Flags.** Does the light-curve flagging leave a real variation alone (no `high_err`,
   `spike_unconfirmed` or `artifact` on real signal)?
4. **Noise.** How large is the run-to-run measurement difference, compared with the error VYVAR
   reports?

Background: artificial-star tests are the standard validation of stellar photometry - Stetson &
Harris 1988 (AJ 96, 909), Anderson et al. 2008 (AJ 135, 2055), Bellini et al. 2017, DOLPHOT fake
stars. They inject one star at a time into copies of the frames and measure completeness, bias and
errors. This suite adapts the idea to time series: the injected object is a light curve, and the
result is a recovered light curve.

## 2. Method (step A: signal on a host star)

1. **Control night.** A finished night is the starting point. Its raw LIGHTs are copied; nothing in
   the archive is touched.
2. **Hosts** (`plan`). Gaia stars of the night's MASTERSTAR that are:
   - DAO detections in the linear zone, aperture S/N >= 10, peak below 0.6 x saturation;
   - isolated: no catalogue star brighter than host + 3 mag within 6.5 FWHM;
   - not Gaia variables (`phot_variable_flag`), not non-single stars, not VSX targets of the night;
   - at least 60 px from the frame edge.
   They are sorted by magnitude into 4 bins; in each bin one host gets each signal of the set.
3. **Signals** (dm in mag, positive = fainter, on UTC JD at mid-exposure):
   - `null`: no change (checks that the procedure adds nothing);
   - `sine`: dm = A sin(2 pi (t - t0) / P), A = 0.01, 0.05, 0.20 mag, P = 0.6 x night span;
   - `eclipse`: dm = D exp(-((t - tmin) / w)^2 / 2), D = 0.05, 0.30 mag, w = span / 12.
   Random phases and minima come from a fixed seed (`--seed`, default 20261023): a run is
   repeatable.
4. **Injection** (`inject`). In every raw LIGHT the host's own image is multiplied by
   (1 + k), 1 + k = 10^(-0.4 dm(t)):
   - position: the host's MASTERSTAR position mapped into the raw frame with the inverse of the
     frame's alignment matrix (`VY_AM0..8` of the control night's aligned frame). Checked with a
     2-D Gaussian centroid on night 537: mean offset 0.01 / 0.03 px, scatter 0.06 px;
   - star image = raw - master dark - local sky (median of an annulus 2..3 x the stamp radius),
     inside a circle of 6.5 FWHM; k x star is added. The star image carries the frame's real PSF,
     flat and position, so no PSF model is involved: the test cannot measure a model with itself
     (the condition set by the ePSF literature review, VYVAR_EPSF_LITERATURE_20261022 section 7);
   - the stamp radius must exceed the photometry sky annulus (`annulus_outer_fwhm` = 5.2): with a
     4 FWHM stamp the faint hosts lost about 5 % of the amplitude (an error of the test, found in
     the first run and corrected).
   Limitation: the star's own pixel noise is scaled with it ((1 + k)^2 instead of (1 + k) in
   variance). For |k| <= 0.3 this changes the noise by at most 15 %; it does not bias the signal.
5. **Two night runs.** The control raw frames and the injected raw frames each run through the
   normal night pipeline (`night_run`, camera / telescope / site as for the original night).
6. **Hosts as targets** (`targets`). The hosts are added to `variable_targets.csv` of both drafts
   (catalogue `INJECT`) and the photometry is re-run (`night_run.run_night_photometry` with
   `existing_draft=True`, the photometry part of the night run). As targets they are vetoed as comparison stars; `compare` refuses a host that is a
   comparison star or not a target in either run.
7. **Comparison** (`compare`). For each host the two light curves are merged on the source frame:
   - recovered signal d = mag_calib(injected) - mag_calib(control) (full chain, with the
     comparison ensemble); and d_inst on instrumental magnitudes (measurement only, no ensemble);
   - amplitude ratio = least-squares scale of the injected shape fitted to d (offset free);
   - time-of-minimum shift = shift of the eclipse template that fits d best (0.1 min steps);
   - residual rms of d - dm, mean offset (zero point), flags in each run.
   The noise of the two runs is the same realisation, so d is far more sensitive than a single
   light curve.

## 3. Run 537 (2026-10-23)

- Night: draft 537 (V842 Her field), QHY294M Pro + Carl Zeiss 200 mm, Jirny, NoFilter, 60 s,
  bin 2, 40 frames over 1.3 h, FWHM 2.36 px (9.8"/px).
- Code: commit `70fac91` (g9) / delivered as s34. Plan: `plan.json` (seed 20261023), 24 hosts
  G 9.1-12.3 (4 bins x 6 signals).
- Runtime: two night runs of ~7 min and two photometry re-runs of ~1.5 min (2 workers).

### Results

`err` = median error VYVAR reports for the host; `ratio inst` / `ratio` = amplitude ratio on
instrumental / calibrated magnitudes; `rms` = residual rms of d - dm; `zp` = mean offset of d - dm
(zero point); `tmin` = time-of-minimum shift; flags = flagged epochs control / injected.

| signal | amp (mag) | G | err (mmag) | ratio inst | ratio | rms (mmag) | zp (mmag) | tmin (min) | flags |
|---|---|---|---|---|---|---|---|---|---|
| eclipse | 0.05 | 9.06 | 7.6 | 1.003 | 0.988 | 1.0 | -3.7 | 0.0 | 0 / 0 |
| eclipse | 0.05 | 10.87 | 18.8 | 1.001 | 0.994 | 1.0 | -0.9 | 0.0 | 0 / 0 |
| eclipse | 0.05 | 11.03 | 21.4 | 0.997 | 0.991 | 1.0 | -0.9 | 0.0 | 0 / 0 |
| eclipse | 0.05 | 11.82 | 47.8 | 0.989 | 0.973 | 1.5 | -3.9 | 0.0 | 0 / 0 |
| eclipse | 0.30 | 9.83 | 10.2 | 0.996 | 0.998 | 1.2 | -1.6 | 0.0 | 0 / 0 |
| eclipse | 0.30 | 10.91 | 18.0 | 0.998 | 1.001 | 5.8 | -8.7 | 0.1 | 0 / 0 |
| eclipse | 0.30 | 11.57 | 34.0 | 1.014 | 1.012 | 10.8 | -9.7 | -0.1 | 0 / 0 |
| eclipse | 0.30 | 12.10 | 48.0 | 0.993 | 0.992 | 2.2 | -0.9 | 0.0 | 0 / 0 |
| sine | 0.01 | 9.91 | 9.8 | 1.071 | 1.037 | 2.7 | -23.0 | - | 0 / 0 |
| sine | 0.01 | 10.56 | 13.7 | 0.990 | 1.004 | 1.4 | 0.9 | - | 0 / 0 |
| sine | 0.01 | 11.22 | 25.1 | 1.003 | 1.003 | 6.7 | -3.3 | - | 0 / 0 |
| sine | 0.01 | 11.96 | 43.3 | 1.001 | 0.997 | 1.4 | 0.8 | - | 0 / 0 |
| sine | 0.05 | 9.30 | 7.7 | 0.996 | 0.990 | 1.3 | 3.6 | - | 0 / 0 |
| sine | 0.05 | 10.26 | 8.8 | 1.008 | 0.998 | 2.0 | 5.4 | - | 0 / 0 |
| sine | 0.05 | 11.23 | 27.4 | 0.977 | 0.986 | 7.2 | -17.3 | - | 0 / 0 |
| sine | 0.05 | 11.65 | 29.3 | 1.009 | 1.013 | 1.2 | -4.3 | - | 0 / 0 |
| sine | 0.20 | 9.55 | 8.9 | 1.001 | 0.999 | 2.5 | -8.1 | - | 0 / 1 |
| sine | 0.20 | 10.86 | 19.4 | 1.000 | 0.999 | 6.0 | -5.1 | - | 0 / 0 |
| sine | 0.20 | 11.45 | 24.7 | 0.999 | 1.000 | 1.6 | -4.5 | - | 0 / 1 |
| sine | 0.20 | 12.27 | 65.0 | 1.006 | 1.007 | 3.3 | -1.4 | - | 0 / 1 |
| null | 0 | 9.47 | 8.7 | - | - | 4.4 | -10.4 | - | 0 / 0 |
| null | 0 | 10.84 | 17.1 | - | - | 1.4 | 3.8 | - | 0 / 0 |
| null | 0 | 11.19 | 20.6 | - | - | 7.1 | 9.5 | - | 1 / 1 |
| null | 0 | 11.96 | 40.6 | - | - | 16.3 | 16.5 | - | 0 / 0 |

Figure: `dev/validation/injection_recovery/537/recovered_vs_injected.png` (recovered d against the
injected dm for every host).

### Assessment

1. **Fidelity: PASS.** Amplitude ratio over the 20 hosts with a signal: 0.999 +- 0.013 (calibrated),
   1.003 +- 0.018 (instrumental), from G 9 to G 12.3 and down to A = 0.01 mag. Neither the aperture
   photometry nor the comparison ensemble nor the post-processing suppresses a variation. The
   largest deviation (1.037 / 1.071 at A = 0.01 mag, G 9.9) is within the run-to-run noise of
   that host (rms 2.7 mmag over 40 points against a 10 mmag amplitude).
2. **Timing: PASS.** All 8 eclipses: time of minimum within 0.1 min (the grid step).
3. **Flags: one defect found and fixed, one open.**
   - *Fixed (s34):* `high_err` flagged the bottom of the 0.3 mag eclipses (3-6 epochs each). The
     rule compared each error with the light curve's median error; a fainter star has a larger
     error by itself (15 % at 0.3 mag for a source-limited star). Now a point fainter than the
     star's median brightness is tested with the error it would have at the median (scaled by
     F/F_med, the lenient sky-limited case). After the fix: 0 flagged epochs on all eclipses;
     flagged epochs on the injected night 19 -> 4. A bad frame (inflated error at normal
     brightness) is still flagged (`test_inflated_error_at_normal_brightness_is_still_high_err`).
   - *Open:* 3 of the 4 sines of 0.2 mag carry one flagged epoch: two at the ends of the night
     (index 0 and 39 of 40; one became `artifact` because that star's surroundings fire the image
     evidence in every frame of both runs), one at index 21. The isolated-spike test compares a
     point with the running median of its neighbours; at the end of a steep trend and near a
     sharp extreme the running median lags the curve. Flags are not deletions:
     `spike_unconfirmed` and `artifact` stay in the AAVSO export, named in NOTES. A fix (a local
     trend instead of a running median at the ends) is proposed, not yet made.
4. **Noise.** The recovered signal of a host scatters by 1-16 mmag between the two runs, also for
   the `null` hosts. This is not injection noise: VYVAR chooses each target's aperture and
   comparison ensemble from the data, and between the two runs the choice flipped for many hosts
   (aperture 3.0 instead of 3.6 px; one of the eight comparison stars different). The flip changes
   the zero point by up to ~20 mmag and the per-frame values by a few mmag; it does not change the
   amplitude. It shows how sensitive the data-driven choices are to a small change of the input;
   worth a separate look (stability of the per-target aperture choice).

### Reproduce

```
T=dev/tools/injection_recovery.py
python $T plan    --control-draft <control draft> --raw-dir <raw LIGHTs> --out plan.json
python $T inject  --plan plan.json --raw-dir <raw LIGHTs> --dark <master dark of the night> \
                  --out-dir <injected copy>/lights/<setup>
# night run of the control raw LIGHTs and of the injected copy (same camera / telescope / site)
python $T targets --plan plan.json --draft <control draft>     # then re-run its photometry
python $T targets --plan plan.json --draft <injected draft>    # then re-run its photometry
python $T compare --plan plan.json --control-draft <control draft> \
                  --injected-draft <injected draft> --out-dir <result dir>
```

Before a photometry re-run the draft's `pipeline_meta.json` stage list is trimmed to `masterstar`
(the stage-order invariant INV-DAG-01 refuses a re-run behind a later stamped stage). A script
that starts worker processes must guard its main code with `if __name__ == "__main__":` (the
per-target aperture search uses spawned workers; without the guard it fails and the photometry
falls back to one aperture).

## 4. Method (step B: new stars)

Step A changes stars that exist. Step B adds stars that do not, and so measures what step A cannot:
the error bars against an independent noise realisation, the detection limit, the bias of a star
next to a brighter one, and the leakage of a variable neighbour into a target - for the aperture
path and for the ePSF path.

1. **Plan** (`plan-new`, on the control night's MASTERSTAR):
   - **templates:** the 12 brightest isolated, unsaturated stars (selection as for the step-A
     hosts), spread over the chip. A synthetic star is a copy of the nearest template, so it
     carries the real PSF of its part of the chip;
   - **flux scale of the field:** zp = median of (MASTERSTAR aperture flux x 10^(0.4 G)) over the
     isolated stars. A synthetic star of magnitude G gets the flux zp x 10^(-0.4 G) whatever its
     template. Scaling by the template's own G carries the template's colour term into the new
     star: in the first run on night 537 two G 11 stars came out 0.3 mag apart (an error of the
     test, corrected);
   - **empty-sky stars:** 3 per magnitude on a grid (default G 10, 11, 12, 13, 13.5, 14, 14.5,
     15), at least 2 FWHM from any catalogue star and 6.5 FWHM from any star brighter than
     G + 3. Two of each three are constant; the third carries a sine (A = 0.05 mag, P = 0.6 x span)
     for G <= 13;
   - **neighbour stars:** 8 isolated real stars (hosts, G 9-11) each get three synthetic stars
     3 mag fainter at 1.5, 2.5 and 4.0 FWHM. Every other host gets a sine of 0.2 mag (step-A
     injection on its own image): the leakage source;
   - every synthetic star gets an ID from 9100000000000000001 and a catalogue row (BP-RP 0.9) in
     `inject_catalog.csv`.
2. **Injection** (`inject-new`). In each raw LIGHT: first the host sines (as in step A), then each
   synthetic star - the template stamp (raw - master dark - local sky, 6.5 FWHM) is cut from the
   untouched frame, shifted to the new position with a cubic spline (`scipy.ndimage.shift`, order
   3, sub-pixel), scaled to its flux (x its own signal) and given Gaussian photon noise with the
   gain of the control run (sqrt(signal / gain) per pixel). Positions use the inverse of the
   frame's alignment matrix, as in step A.
3. **Night run with the test catalogue.** VYVAR measures catalogue stars only (the target join is by
   Gaia source_id), so the synthetic stars must be in the catalogue. The night run reads the CSV
   through the test hook `VYVAR_INJECT_CATALOG` (`pipeline_catalog._append_injected_catalogue`:
   rows inside the cone are appended to the local Gaia cone, before the magnitude cut; a WARNING
   is logged on every use). Without the environment variable nothing changes. ePSF is switched on
   (`epsf_auto_run`).
4. **Synthetic stars as targets** (`targets --as-vsx`) and a photometry re-run: they are added as
   VSX-typed targets, so they are vetoed as comparison stars and measured by both paths.
5. **Comparison** (`compare-new`), per star and path (aperture `mag_calib`; ePSF `psf_delta_mag`):
   - status: measured, `below_snr_min` (too faint for a target light curve), `no_dao_detection`;
   - rms of the light curve around the injected shape, median reported error, pull rms
     (rms of residual / reported error; 1 = honest error bars);
   - amplitude ratio for the sine stars (offset free);
   - bias: mean offset minus the median offset of the constant empty-sky stars (the zero point);
   - leakage: least-squares scale of the host's sine fitted to the neighbour star's light curve.

What a single night does not give: the noise of the comparison ensemble is common to all stars and
enters the pull; a 40-frame night gives an amplitude ratio to +-0.05 at G 10 and +-0.2 at G 12
(for A = 0.05 mag).

## 5. Run 537, step B (2026-10-23)

- Night 537 as in chapter 3. Code: s36 (g9 `9041ea3`). Plan: `537/stepB/plan.json` (seed
  20261023, defaults: 48 stars - 24 empty-sky, 24 neighbours of 8 hosts). Gain 0.7925 e-/ADU.
- Data: `dev/validation/injection_recovery/537/stepB/` (`summary_new.md`, `results_new.csv`,
  `inject_catalog.csv`).

### Results

Empty sky (constant and sine stars, 40 frames):

| G | path | measured | median err (mmag) | rms (mmag) | pull rms | amp ratio (sine) | bias (mmag) |
|---|---|---|---|---|---|---|---|
| 10 | aperture | 3 / 3 | 10.6 | 10.7-11.3 | 1.03-1.07 | 0.93 | -21 .. +30 |
| 10 | ePSF | 3 / 3 | 9.6 | 10.8-12.6 | 1.14-1.32 | 0.96 | -8 .. +8 |
| 11 | aperture | 3 / 3 | 19.4 | 17.0-18.7 | 0.87-0.97 | 1.19 | -48 .. -8 |
| 11 | ePSF | 3 / 3 | 16.6 | 13.4-17.6 | 0.81-1.05 | 1.13 | -68 .. +22 |
| 12 | aperture | 3 / 3 | 47 | 34-42 | 0.73-0.91 | 0.73 | +36 .. +42 |
| 12 | ePSF | 3 / 3 | 36 | 33-36 | 0.92-1.00 | 0.66 | -19 .. +49 |
| 13-13.5 | - | 0 / 6 | | | | | below_snr_min |
| 14-15 | - | 0 / 9 | | | | | 1 below_snr_min, 8 not detected |

Neighbours 3 mag fainter than an isolated host (G 12.6-13.9; the night is too shallow for most of
them - 22 of 24 are below the target S/N or not detected):

| sep (FWHM) | host | path | bias (mmag) | rms (mmag) | leak of the host's sine |
|---|---|---|---|---|---|
| 1.5 | constant | aperture | **-1813** | 64 | |
| 1.5 | constant | ePSF | -51 | 96 | |
| 2.5 | sine 0.2 mag | aperture | **-1059** | 104 | **59 %** |
| 2.5 | sine 0.2 mag | ePSF | -114 | 42 | 7 % |

### Assessment

1. **Error bars.** Pull rms 0.73-1.32 over G 10-12 on both paths: the reported errors match the
   scatter to about 20 %. ePSF at G 10 is slightly optimistic (1.14-1.32), the aperture at G 12
   slightly pessimistic (0.73-0.91).
2. **ePSF is less noisy for faint stars:** at G 12 the reported error is 36 vs 47 mmag and the
   rms 33-36 vs 34-42 mmag.
3. **Amplitude.** 0.93 / 0.96 (G 10), 1.19 / 1.13 (G 11), 0.73 / 0.66 (G 12) - all within 1-2 sigma
   of 1 for a 0.05 mag sine on 40 frames. Step A (0.999 +- 0.013) is the precise fidelity test;
   step B shows nothing that contradicts it.
4. **Absolute level.** The bias scatter of the constant stars is +-20-70 mmag. It is the
   calibrated magnitude of a star with no real colour (all synthetic stars carry BP-RP 0.9 and the
   flux of a field-median star) against the G-band zero point; it does not enter differential
   light curves.
5. **Close neighbours: ePSF is the right path.** At 1.5 FWHM from a star 3 mag brighter the
   aperture measures the neighbour's light (1.8 mag too bright), the ePSF fit separates it
   (-0.05 mag). At 2.5 FWHM 59 % of the host's variation leaks into the aperture light curve,
   7 % into the ePSF one. A blended target should be reported from the ePSF light curve; the
   blend columns (`blend_k*`) already warn about such targets.
6. **Detection limit** of this night (60 s, 9.8"/px): target light curves to G 12; G 13 is below the
   target S/N gate, G 14 and fainter are not detected.
7. **Found and fixed through this test:**
   - ePSF measured VSX targets only, so exoplanet hosts, manual targets and the synthetic stars had
     no PSF light curve and the completeness gate failed. ePSF now measures every target with
     aperture photometry (EPSF-ALL-TARGETS-2026-10, decision of 2026-10-23, s36);
   - the template colour term (point 1 of the method, an error of the test itself).

### Reproduce

```
T=dev/tools/injection_recovery.py
python $T plan-new   --control-draft <control draft> --raw-dir <raw LIGHTs> --out plan.json
python $T inject-new --plan plan.json --raw-dir <raw LIGHTs> --dark <master dark> \
                     --out-dir <injected copy>/lights/<setup>
export VYVAR_INJECT_CATALOG=<dir of plan.json>/inject_catalog.csv
# night run of the injected copy with epsf_auto_run = true (same camera / telescope / site)
python $T targets --as-vsx --plan plan.json --draft <injected draft>   # then re-run its photometry
python $T compare-new --plan plan.json --injected-draft <injected draft> --out-dir <result dir>
```

## 6. Run 534, step B (2026-10-23)

- Night: draft 534 (Cygnus field, RA 320.6 Dec +51.9), C5A-150M + AZ 800 (5480 mm), Zdanice,
  Sloan g, 120 s, bin 4 (0.57"/px), 98 frames over 6.5 h (97 aligned). Seeing changes a lot:
  FWHM 4.2 px early, 8.5 px late. Frame depth G 20.1; catalogue depth G 17.5.
- Gain 12.48 e-/ADU (header GAIN = 16 x 0.78: the camera averages the binned pixels; the sky
  noise of the raw frames, 2.1 ADU at a sky of 29 ADU, confirms it).
- Code: s36 + EXPORT-RAM-PEAK-2026-10 + the pandas-3 fix of `select_active_targets` (s37).
  Plan: `534/stepB/plan.json` (seed 20261023; hosts G 12-14, neighbours 3 mag fainter at 1.5 /
  2.5 / 4.0 FWHM; empty sky G 13, 14, 15, 16, 16.5, 17, 17.5, sine to G 16.5). The 8 hosts were
  added as targets as well (`plan_with_hosts.json`, 53 targets; see point 7).
- Runtime (container, 2 cores, 7 GB): night run 42 min (1 worker), photometry re-run with ePSF
  20 min (2 workers).

### Results

Empty sky, 97 frames:

| G | path | measured | median err (mmag) | rms (mmag) | pull rms | amp ratio (sine) |
|---|---|---|---|---|---|---|
| 13 | aperture | 3 / 3 | 3.6 | 3.3-8.2 | 0.83-2.01 | 1.02 |
| 13 | ePSF | 3 / 3 | 3.7 | 3.8-10.9 | 0.97-2.70 | 1.00 |
| 14 | aperture | 3 / 3 | 5.0 | 6.7-9.4 | 1.26-1.67 | 1.04 |
| 14 | ePSF | 3 / 3 | 4.8 | 6.2-13.2 | 1.15-2.57 | 0.97 |
| 15 | aperture | 3 / 3 | 8.8 | 11.3-23.0 | 1.04-2.15 | 1.03 |
| 15 | ePSF | 3 / 3 | 7.6 | 8.9-13.4 | 1.11-1.63 | 0.97 |
| 16 | aperture | 3 / 3 | 19.5 | 38.9-94.5 | 1.44-4.13 | **1.79** |
| 16 | ePSF | 3 / 3 | 15.0 | 17.8-23.8 | 1.00-1.37 | 0.83 |
| 16.5 | aperture | 3 / 3 | 29.7 | 41.6-91.3 | 1.21-2.51 | **1.69** |
| 16.5 | ePSF | 3 / 3 | 21.9 | 31.0-33.0 | 1.17-1.46 | 0.82 |
| 17 | aperture | 2 / 3 | 46 | 78-116 | 1.18-1.67 | |
| 17 | ePSF | 2 / 3 | 34 | 47-49 | 1.14-1.30 | |
| 17.5 | - | 0 / 3 | | | | below_snr_min |

Neighbours 3 mag fainter than a host (G 15.3-16.9):

| sep (FWHM) | path | measured | bias (mmag) | rms (mmag) | leak of the host's 0.2 mag sine |
|---|---|---|---|---|---|
| 1.5 | both | 0 / 8 | | | not detected as a separate star |
| 2.5 | aperture | 7 / 8 | -650 .. -2710 | 390-910 | +0.08 .. +0.85 |
| 2.5 | ePSF | 7 / 8 | -36 .. +173 | 47-192 | -0.68 .. +0.27 |
| 4.0 | aperture | 8 / 8 | -90 .. -630 | 230-900 | **-0.91 .. -1.03** (3 of 4) |
| 4.0 | ePSF | 8 / 8 | -220 .. +61 | 30-201 | -0.02 .. -0.16 |

### Assessment

1. **Error bars.** On the bright end (G 13-15) the scatter of the constant stars is 1.0-2.7 x the
   reported error on both paths, with a floor of 6-11 mmag where the error says 3.6-5 mmag; the
   sine star of the same magnitude can sit at pull 0.8-1.0, so the excess depends on the position
   on the chip, not on the magnitude. The same test on 537 (stable seeing, 40 frames) gave pull
   0.8-1.3. A systematic term that the error model does not contain (flat-field, the change of the
   PSF over the night) is the likely cause; it is not resolved by this test.
2. **ePSF follows a 0.05 mag variation to G 15** (ratio 0.97-1.00) and keeps the scatter close to
   the error to G 17 (pull 1.0-1.5). At G 16-16.5 the amplitude comes out 0.82-0.83 (about 4 sigma
   below 1 with 97 points): the ePSF fit of faint stars suppresses about 17 % of the variation.
   Open point.
3. **Aperture photometry of faint stars in bad seeing is biased.** The aperture radius follows
   the frame's FWHM (4.3 px to 12.7 px over the night). In the last 15 frames 5 of the 8 G 16-17
   stars come out 0.05-0.24 mag too faint (median, after removing the injected signal) while ePSF
   stays within -0.05..+0.02 mag; on a sine whose phase puts a maximum near the end this trend is
   fitted as amplitude 1.7-1.8. The large annulus in a crowded field (Cygnus, frame
   depth G 20) picks up the wings of neighbours and faint uncatalogued stars: the sky is
   overestimated and the star appears too faint. This is the known positive skew of the annulus
   sky in crowded fields (the sky estimate is a median, robust to outliers but not to a smooth
   excess). For faint stars in crowded fields the ePSF light curve is the one to use.
4. **Neighbours.** At 1.5 FWHM neither path separates the pair (no DAO detection). At 2.5 FWHM the
   aperture measures the host's light (-0.65 to -2.7 mag), the ePSF fit keeps the neighbour within
   -0.04..+0.17 mag. At 4.0 FWHM the host lies in the aperture's sky annulus: the aperture light
   curve of the neighbour carries the host's variation with the **opposite sign and full amplitude**
   (leak -0.9 to -1.0; sky follows the host), ePSF -0.02 to -0.16. Same direction as on 537,
   larger because of the seeing.
5. **Hosts** (real stars with three synthetic neighbours each, about 19 % of their flux): host sine
   0.87-0.93 aperture, 0.90-0.96 ePSF - consistent with the dilution by the added neighbours.
6. **Detection and measurement limit** of this night: target light curves to G 17 (2 of 3), G 17.5
   below the target S/N gate.
7. **Found through this test (VYVAR):**
   - EXPORT-RAM-PEAK-2026-10 (fixed, s37): the per-frame export packed every aligned RAM frame up
     front, a second copy of the whole night; 534 was killed out of memory on 7 GB. A/B on 537:
     outputs identical;
   - pandas 3 (fixed, s37): `select_active_targets` crashed on a VSX row without a type
     (`astype(str)` keeps NaN under pandas 3);
   - the "CT presel stub" rule regenerates `variable_targets.csv` from the VSX cone whenever it has
     fewer than 50 rows and MASTERSTAR more than 200 stars; it removed the 45 synthetic targets
     (and would remove hand-added targets). Removed on 2026-10-24 (CT-STUB-REMOVED-2026-10); the 8 hosts were added so
     that the run has 53 targets;
   - the stale `proc_*.fits` glob of the border filter (fixed 2026-10-24, BORDER-FOOTPRINT-2026-10).
8. **Errors of the test itself, corrected:** the first run injected the photon noise with the
   database gain 0.78 (4 x too much noise; pulls 2-3.6); `inject-new` now takes the gain from the
   raw header as VYVAR does. A second run injected nothing because the control's aligned frames had
   been deleted for disk space (they carry the alignment matrices); `inject-new` now stops when
   they are missing.

## 7. Run 534, step D: faint-star amplitude (2026-10-24)

Repeat of the open point 2 of chapter 6 (ePSF amplitude 0.82-0.83 at G 16-16.5, one sine star per
magnitude) with enough stars to measure it.

- Same night, raw frames and gain as chapter 6; code s41 (AP-NIGHT-K). Plan `534/stepD/plan.json`
  (seed 20261024): no hosts, empty sky only, G 14, 15, 15.5, 16, 16.5, 17, ten stars per magnitude,
  six of them with a sine (A = 0.05 mag, P = 0.6 x span, random phase; `plan-new --n-sine 6`).
- Analysis: `compare-new` and `dev/tools/injection_faint_seeing.py` (amplitude with and without a
  term linear in the frame FWHM).

### Results

All 60 stars measured on both paths (97 frames). Amplitude ratio, mean +- standard error over the
sine stars:

| G | aperture | ePSF | ePSF, spread per star | white-noise spread expected |
|---|---|---|---|---|
| 14-15 (12 stars) | 0.984 +- 0.028 | 0.971 +- 0.018 | 0.06 | 0.04 |
| 15.5-16 (12) | 0.999 +- 0.058 | 0.990 +- 0.050 | 0.17 | 0.07 |
| 16.5-17 (12) | 1.154 +- 0.141 | 1.045 +- 0.075 | 0.26 | 0.17 |
| 16 (6) | 0.973 +- 0.108 | 0.960 +- 0.092 | 0.23 | |
| 16.5 (6) | 1.020 +- 0.114 | 0.994 +- 0.073 | 0.18 | |

Seeing dependence (slope of the residual against the frame FWHM, per doubling of the FWHM): median
over stars near zero on both paths, absolute value 13-51 mmag (ePSF) and 21-71 mmag (aperture) at
G 14-16.5, sign different from star to star; G 17: aperture +248 mmag (too faint in bad seeing),
ePSF +86. A FWHM term in the fit narrows the spread of the amplitudes (ePSF G 16: 0.23 -> 0.16,
aperture 0.26 -> 0.17) and does not move their mean.

### Assessment

1. **No amplitude suppression by the ePSF fit.** At G 16-16.5 the ePSF amplitude is 0.96 +- 0.09 and
   0.99 +- 0.07; the 0.82-0.83 of chapter 6 was one star per magnitude and lies inside the spread of
   single stars (0.59-1.23 here). EPSF-FAINT-AMP-01 is closed. Aperture and ePSF agree within their
   errors to G 16.5; at G 17 both scatter, the aperture more (seeing bias of the annulus, chapter 6
   point 3).
2. **Faint light curves carry correlated (red) noise.** The amplitude of a single faint star scatters
   2-2.5 x more than the white noise of its light curve allows (ePSF G 15.5-16: 0.17 vs 0.07). Part of
   it follows the seeing with a sign that depends on the star (its surroundings: uncatalogued faint
   stars of a G 20-deep field in the wings of the PSF / in the annulus); on this night the FWHM
   doubles. The reported error bars do not contain it: one night of a faint star in a crowded field
   gives its amplitude to about 20 % (G 16), not 7 %. FAINT-RED-NOISE-01, closed 2026-10-24 with no
   change (Milan): the red-noise factor beta (time-averaging, Winn et al. 2008) computed from the
   residuals of the star's own model predicts the amplitude scatter (0.9-1.3 x), but VYVAR has no
   model of a target - beta of the raw light curve counts the variability (MN Boo 538: MN Boo, AC Boo
   and two EA stars 3.2-3.8, check stars median 1.12), beta of other constant stars is not the star's
   own (0.7-1.7 x), and a FWHM detrend before the fit removes 16 % of an injected sine. Whoever fits a
   model to a light curve (amplitude, time of minimum) computes beta from its residuals.
3. A fitted FWHM term is not proposed for the light curves: a real variation that runs with the
   seeing over the night (here a sine of 0.6 x the span against a FWHM trend) would be removed with
   it.

## 8. Not covered yet

- The bright-end error floor (chapter 6 point 1).
- **Step C:** times of minimum of realistic eclipse shapes on longer nights; error-bar calibration
  (pulls) from independent noise realisations.
- Other rigs and nights: the result above is one night of one rig.
