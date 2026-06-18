CURSOR RESULT � run-414 V0454 CrA diagnostic � 2026-06-18

# run-414 V0454 CrA: outliers, inflated errors, phase_correlation frames (DIAGNOSE only)

**Mode:** diagnose-only. No code/config/science change, no commits, no UI. Post-hoc from run-414
artifacts. Throwaway script: `tmp/diag414.py`; figure `tmp/diag414_v0454.png`.

**Draft confirmed:** `Archive/Drafts/draft_000414` (fresh overnight run 2026-06-18 02:15, same Boyden
V0454 CrA g data). Artifacts used: `platesolve/g_60_2/alignment_report.csv`,
`.../photometry/lightcurves/lightcurve_4035720806645181440.csv`, `.../comparison_stars_per_target.csv`,
`.../photometry_summary.csv`, and the 161 per-frame `detrended_aligned/lights/g_60_2/proc_*.csv`.

**Run state.** g LC: 59 pre-flip + 102 post-flip = 161 points. Alignment: 148 astroalign + 13
phase_correlation (all post-flip, all `rotation_angle_deg` blank, detected_stars 465�500, clustered
JD .536�.610 / 00:52�02:38). V0454 `lc_rms=0.405`, `lc_rms_ooe=0.286`, trust **RED** (0 clean comps,
sparse_fallback, check-star scatter 0.498). Comps: 2, both "suspect (kept: n_good<min)" � C01
`4035718465808516096` (g 10.86, **?BP-RP 1.030**, comp_rms 0.006), C02 `4035679742383369216`
(g 10.40, ?BP-RP 0.284, comp_rms 0.013). B.2 gate is **OFF** (default) so the 13 bad frames are in.

---

## D-A � Outlier classification by population

Robust outlier flag = |residual from rolling-median(11) trend| > 4�(1.4826�MAD). 16/161 flagged.
Cross-tab (alignment method � B.2 `flux_large/flux` ratio, gate replicated at k=5 + FWHM guard):

| | outlier=False | outlier=True |
|---|---|---|
| astroalign / normal-ratio | 139 | 9 |
| phase_corr / B2-ratio-flagged | 6 | 7 |

**The catastrophic outliers are 100% phase_correlation frames.** All 6 points with |resid| > 0.3 mag
are phase_correlation. Of the 13 phase_corr frames: **7 are >4? outliers** (resid up to **+3.7 mag**,
err up to 16 mag), **6 are NaN** (V0454 unmeasurable), **n_clean = 0** � every phase_corr frame is
unusable. The "9 astroalign outliers" are **spurious flags**: |resid| ? 0.036 mag (median 0.023) �
they're the *cleanest* points, tripped only because the astroalign MAD is ~0.005 mag; none exceed
0.1 mag and none are visible in the LC.

**Key cross-population fact:** in run-414 the phase_correlation set and the high-`flux_large/flux`
set are the **same 13 frames** (phase_corr median ratio_V 11.8 vs 1.5 for astroalign). But the cause
is **not** transparency collapse � these frames have **465�500 detected stars** (dense), opposite of
yesterday's few-star dawn B.2 frames. The ratio spikes because a **mis-centred aperture** (D-B) pushes
flux out of the small science aperture, mimicking PSF collapse. So the B.2 ratio metric *would* catch
them (gate-ON removes them), but they are a distinct **dense-frame astroalign-failure** population, not
the transparency population.

? **Outlier source = the 13 phase_correlation frames. astroalign frames are clean (med|resid| 0.003 mag).**

---

## D-B � phase_correlation alignment quality + root cause

**B.1 � Is the path translation-only? YES (confirmed in code, not a hypothesis).**
`vyvar_alignment_frame.py:566�619`: the phase_correlation branch is a *fallback* taken only when
astroalign (and WCS-reproject) fail. It calls `skimage.registration.phase_cross_correlation(ref0,
src0, upsample_factor=10)` ? a pure `(dy, dx)` shift, applied with `scipy.ndimage.shift`. **No
rotation and no scale are solved or applied** � hence `rotation_angle_deg` is blank for all 13.

**B.2 � Are the 13 frames actually mis-aligned? YES.**
Per-frame matched-source position residual vs each source's across-night median position
(bright stars, aligned grid):

| method | n frames | median resid | p90 | max |
|---|---|---|---|---|
| phase_correlation | 13 | **2.13 px** | 2.25 | 2.33 |
| astroalign | 148 | **0.36 px** | 0.46 | 1.65 |

phase_correlation frames are **~6� worse** (2.1 px vs 0.36 px). On the ~5 px science aperture with a
defocused donut PSF, a 2 px centroid error throws flux out of the aperture ? the ratio_V spike
(6.7�29), the 7 mag outliers and 6 NaNs. **No rotation signature:** residual-vs-radius correlation on
the worst frame = ?0.02 (a rotation error would give a strong positive radial gradient). The residual
is a roughly uniform **translation mis-registration**. Because the residual is only ~2 px (not the
~thousands a 180� mis-rotation would give), the gross meridian flip **is** handled upstream � astroalign
itself derotates 89 of the 102 post-flip frames fine; the failure is fine-registration on these 13.

**B.3 � Root cause (HYPOTHESIS, code-consistent; to confirm with an alignment log/repro).**
astroalign's asterism (triangle) matching fails on these dense post-flip frames, forcing the
translation-only fallback. The attempt ladder (`vyvar_alignment_frame.py:393�396`) relaxes to
`dao_sigma=1.5, max_stars=500`, admitting many spurious/noise detections, and the control-point cap
is `mcp = max(12, min(max_stars, n_fit))` (`:512`) ? **up to 500 control points**. A dense
Galactic-plane field at the 500-detection cap gives a combinatorially ambiguous set of similar
asterisms; astroalign's RANSAC fails to find a consensus transform and raises, so the pipeline drops to
phase_correlation. The 13 failures are all post-flip and all at the 465�500-star cap (the densest /
most-defocused window of the night), consistent with this mechanism rather than with the flip per se.

**Recoverable vs exclude � these frames are RECOVERABLE (data is good); exclude is a correct stop-gap.**
Evidence the data is sound: 465�500 real stars, only a ~2 px residual translation, and the *same field
in the same orientation* aligns cleanly on 89 other post-flip astroalign frames. The failure is the
solver/fallback, not the pixels. **Recommendation:** Milan's exclude is the right *short-term* safety
move (translation-only fallback is unreliable and currently silently admitted to the LC), but the
durable fix is **dense-field alignment robustness** � e.g. cap/clean control points (lower max_stars or
brightest-N for asterism matching), prefer WCS-reproject when a per-frame solve exists, or add a
rotation-aware refinement / reject-on-residual guard so a >~1 px fallback alignment is flagged rather
than fed to photometry. (All code changes are out of scope here.)

---

## D-C � The inflated per-point error model

**C.1 � What `err` is (confirmed in code).** The LC `err` column (`photometry_core.py:4111`) is built
(`:7840�7858`) as three terms in quadrature:
1. **base** = aperture-SNR photon error `? 1.0857 / SNR` (`:5878`) � small for bright V0454, but
   explodes (or ? NaN) when mis-centred-aperture flux loss tanks the SNR;
2. ? `comp_rms_med / ?n_ens` � here ? 0.0095/?2 ? **0.007** (negligible; the comps' own RMS is tiny);
3. ? `ensemble_scatter / ?n_ens`, where `ensemble_scatter = np.std(comp_vals)` and `comp_vals` are the
   **comparison stars' instrumental magnitudes** (`:2552, :2567`).

**C.2 � It is dominated by the 2 sparse comps (confirmed numerically).** Term 3 is the whole story.
The two comps' instrumental mags differ by **1.655 mag** (C02 ?12.92 vs C01 ?11.23), so
`std = 0.827`, and this is **near-constant frame-to-frame** (its own std across frames = 0.020).
`0.827/?2 = 0.585` � exactly the observed baseline `err` (astroalign median 0.580, range 0.547�0.623).
So the "per-point error" on every clean point is really **half the two comps' brightness difference** �
a fixed ensemble-composition number, not a measure of that point's precision.

Decomposition:
- **Baseline inflation (all 161 points):** ~**0.58 mag**, from `std(comp instrumental mags)`.
- **Catastrophic (the 13 phase_corr frames):** **2.1�16 mag**, as term-1 (SNR) blows up on the
  mis-centred apertures (and term-3 worsens as the comps' fluxes also degrade).

**C.3 � Is it mis-calibrated, and where does the inflation enter? YES, ~23�.**
Empirical out-of-eclipse plateau scatter (post-flip, non-outlier, n=92) = **0.025 mag**; formal median
`err` = 0.581 ? **23� over-estimated**. The inflation enters at `ensemble_scatter = np.std(comp_vals)`
on **instrumental** magnitudes: for a small ensemble this is dominated by the comps' brightness
*spread*, which does **not** represent per-point uncertainty (and is largely cancelled by the flux-sum
differential). Note the **LC centres are fine** � `mag_calib`/`delta_mag` use the correct flux-sum
ensemble zeropoint (`ens_med`, `:2561�2563`), which is why the points trace the clean egress; only the
error bars are wrong. This is **both**: it shares the thin/sparse-comp structural root with trust-RED
(the 2-comp colour-mismatched ensemble amplifies it � a tighter comp brightness/colour window would
shrink term 3), **and** it is a distinct error-propagation issue (the `std(instrumental mag)` formula
conflates comp brightness spread with point noise regardless of comp count). Diagnose-only � the fix
(use a per-point/photon + verified comp-residual error, not the comp brightness spread) is out of scope.

---

## Synthesis / answers

- **The outliers are the 13 phase_correlation frames** (7 mag-outliers + 6 NaN, n_clean 0); astroalign
  is clean. Gate-OFF in this run lets them inflate `lc_rms` to 0.405. They coincide with the high
  `flux_large/flux` ratio, but via mis-centred-aperture flux loss, **not** transparency collapse.
- **phase_correlation is translation-only and the 13 frames are mis-aligned by ~2 px** (no rotation
  error). Root-cause hypothesis: astroalign asterism matching degenerates on the dense (?500-star,
  post-flip) frames ? translation-only fallback. **The frames are recoverable** with better alignment;
  exclude is a sound stop-gap but not the root fix.
- **The �0.5 mag error bars are mis-calibrated by ~23�.** Baseline 0.58 mag = `std(2 comps'
  instrumental mags)/?2` (a constant comp-brightness term), with catastrophic 2�16 mag on the
  mis-aligned frames. The LC centres are correct; only the formal error is inflated.

## Files
- `tmp/diag414.py` (throwaway), `tmp/diag414_v0454.png` (LC coloured by alignment method + per-frame
  position-residual panel). No code/config changed; no commits.
