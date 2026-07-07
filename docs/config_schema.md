# VYVAR Config Schema

Parameters defined in config.json. Defaults from AppConfig in config.py (dataclass field defaults, synced with JSON).

## Calibration

| parameter_name | type | default | description |
|----------------|------|---------|-------------|
| masterdark_validity_days | int | 80 |  |
| masterflat_validity_days | int | 524 |  |
| calibration_library_native_binning | int | 1 | When applying masters from CalibrationLibrary, assume this **sensor** binning in the stored FITS (typically ``1`` = full resolution). Lights with ``XBINNING`` 2×2 are matched in RAM (temporary resample). JSON ``null``: read ``XBINNING`` from each master FITS (e.g. Bin2 library files with 2×2 lights). |
| qc_after_calibrate_enabled | bool | true | Post-calibration QC on each calibrated light (metrics + pass/fail vs limits). |
| dao_qc_in_calibrate | bool | true | PERF-10: DAO QC (FWHM/sky/star_count) during calibration; skips RAM QC pass when True. |
| qc_max_hfr | float | 5.0 |  |
| qc_min_stars | int | 10 |  |
| qc_max_background_rms | null | null | If set, fail when sigma-clipped sky RMS exceeds this (same units as calibrated image). |
| auto_fwhm_enabled | bool | true | FITS QA dashboard: odvodzovať predvolený FWHM limit z MAD (median + k×σ_MAD). |
| auto_fwhm_k_factor | float | 1.5 |  |
| auto_fwhm_k_min | float | 1.0 |  |
| auto_fwhm_k_max | float | 4.0 |  |
| bpm_dark_mad_sigma | float | 5.0 | Master-dark column BPM: MAD multiplier for ``*_dark_bpm.json`` (see ``importer``). |
| gain | float | 1.0 | CCD gain (e-/ADU) — used in noise model / SNR estimates. |
| read_noise | float | 10.0 | CCD read noise (e-) — used in noise model. |
| sky_adu_fallback | float | 1581.6 | Fallback sky background level (ADU) when no sky estimate is available. |

## Observer / Location

| parameter_name | type | default | description |
|----------------|------|---------|-------------|
| observer_name | str | "Unknown Observer" |  |
| observer_code | str | "" |  |
| aavso_observer_code | str | "UMIA" | Legacy mirror — synced from ``observer_code`` in ``__post_init__`` (kept for older callers). |
| observer_location_id | int | 2 |  |
| observer_lat | float | 50.1121658 |  |
| observer_lon | float | 14.6982547 |  |
| observer_alt_m | float | 275.0 |  |
| observer_location_name | str | "" |  |
| export_arcsec_per_px | float | 1.3 | Fallback pixel scale (arcsec/px) for export headers if FITS/WCS is unavailable. |

## MASTERSTAR / DAO Detection

| parameter_name | type | default | description |
|----------------|------|---------|-------------|
| masterstar_solver_use_draft_median_if_hint_sep_deg | float | 1.0 | If plate-solve hint RA/Dec vs draft median separation exceeds this (deg), use draft median for solver. |
| masterstar_log_astroalign | bool | true | Log zarovnanie (astroalign): referenčný rámec a počty kontrolných bodov. |
| masterstar_optimizer_mirror_extra_log | bool | true | After astrometry optimizer mirror-orientation warning, log an extra hint line. |
| masterstar_platesolve_sip_max_order | int | 4 | VYVAR plate-solve na MASTERSTAR: max. SIP stupeň (2–5). Solver skúša **nadol** po ``masterstar_platesolve_sip_min_order`` (napr. 5→4→3). |
| masterstar_platesolve_sip_min_order | int | 3 | Najnižší SIP stupeň pri páde vyšších (typicky 3; nie menej ako 2). |
| masterstar_dao_threshold_sigma | float | 2.1 | DAOStarFinder threshold = σ×RMS len pre MASTERSTAR katalóg (hlbšia detekcia; cieľ viac tisíc hviezd). |
| masterstar_prematch_peak_sigma_floor | float | 1.8 | Pred matchom s Gaia: ponechať detekcie s peakom aspoň ``median + k×σ`` (nižšie = viac slabých hviezd). |
| masterstar_platesolve_prewrite_rms_max_px | float | 30.0 | MASTERSTAR: horná hranica px RMS pred zápisom WCS (pred relaxáciou). ``None`` = predvolené 14 px. |
| masterstar_platesolve_prewrite_relaxed_rms_max_px | float | 35.0 | MASTERSTAR: pri dobrom match_rate akceptovať RMS až do tejto hodnoty [px]. ``None`` = 22 px. |
| masterstar_platesolve_nn_refine_max_rms_px | null | null | MASTERSTAR: NN WCS refine sa aplikuje len ak RMS ≤ tejto hodnote [px]. ``None`` = 7.5 px. |
| masterstar_sip_force_rms_guard_ratio | float | 1.15 | Pri ``force_apply`` SIP: zamietnuť ak ``rms_sip > rms_linear * ratio``. ``None`` = bez stráže (pôvodné správanie). |
| masterstar_best_of_n | int | 10 | MASTERSTAR: stack N frames and pick best N for ePSF/catalog build. |
| sips_dao_fwhm_px | float | 2.5 | DAOStarFinder FWHM (pixels) tuned for SIPS-like centroid search (aperture ~13 → ~4–5 px FWHM). |
| sips_dao_threshold_sigma | float | 3.5 | DAOStarFinder threshold = this × background RMS (SIPS “standard deviation count” ≈ 2.5). Pre hlboký MASTERSTAR / široké pole niekedy **0.25–1.0** (viac špičiek); používa sa aj pri VYVAR plate solve, ak volanie neprebije ``dao_threshold_sigma``. |

## Plate Solving

| parameter_name | type | default | description |
|----------------|------|---------|-------------|
| platesolve_anisotropy_threshold | float | 1.3 | Pomer sx/sy (arcsec/px) — nad týmto sa považuje WCS za príliš anizotropný (VYVAR retry / diagnostika). |
| phase01_plate_scale_arcsec_per_px | float | 1.3 |  |
| plate_scale_arcsec_per_px | float | 1.3 | Plate scale (arcsec/px) for Phase 2A metadata, GS11, dilution; Set 1 default 1.3. |
| phase01_match_radius_arcsec | float | 10.0 | Fáza 0: minimum cross-match radius VSX → masterstars (arcsec); used with max(5× plate_scale, this). |

## Alignment

| parameter_name | type | default | description |
|----------------|------|---------|-------------|
| alignment_max_stars | int | 160 | Frame alignment (``astroalign`` + DAO positions): max brightest sources offered as control points per frame. |
| alignment_detection_sigma | float | 5.0 | DAOStarFinder threshold multiplier vs sigma-clipped background RMS (higher = fewer, more significant peaks). |
| qc_dao_detection_sigma | float | 5.0 | Same recipe as QC HFR star detection (``_mean_hfr_bright_stars_dao`` first pass: ``threshold = qc_dao_detection_sigma × std``). Used for frame alignment DAO so it tracks QC-style sensitivity. |
| per_frame_mp_reserve_ram_gb | float | 1.5 | Reserve this much RAM (GB) when capping paralelného exportu katalógov cez ``psutil`` (nad rámec jednotného ``_pw``). |

## Phase 0+1 Comparison Selection

| parameter_name | type | default | description |
|----------------|------|---------|-------------|
| phase01_comparison_max_dist_deg | float | 1.5 | Fáza 0+1 — výber porovnávacích hviezd (``photometry_core.select_comparison_stars_per_target``). Pri **riedkom poli** zväčši ``phase01_comparison_max_mag_diff`` / ``phase01_comparison_max_dist_deg``, prípadne zníž ``phase01_comparison_min_frames_frac`` alebo zvýš ``phase01_comparison_max_comp_rms`` (slabší filter stability). Pri **jasných cieľoch** (``mag`` < ``phase01_comparison_mag_bright_threshold``) sa použije aspoň ``phase01_comparison_max_mag_diff_bright_floor`` ako minimálny /Δmag/ pás (``0`` = vypnuté). |
| phase01_comparison_max_mag_diff | float | 1.5 |  |
| phase01_comparison_mag_bright_threshold | float | 12.75 |  |
| phase01_comparison_max_mag_diff_bright_floor | float | 1.5 |  |
| phase01_comparison_max_mag_diff_absolute | float | 3.0 | Absolútny strop pre adaptívne uvoľňovanie /Δmag/ pri výbere porovnávačiek. Nikdy nejdeme vyššie (ochrana pred miešaním úplne iných jasností). |
| phase01_comparison_max_bv_diff | float | 0.24 |  |
| comp_max_delta_bprp | float | 0.79 | Max /ΔBP-RP/ v efektívnom farebnom priestore (hard filter pri ``phase01_use_bprp_primary``). |
| comp_tier1_bprp_limit | float | 0.15 | Tier limity /ΔBP-RP/ (Gaia BP-RP ako primárny farebný filter pri výbere comp). |
| comp_tier2_bprp_limit | float | 0.3 |  |
| comp_tier3_bprp_limit | float | 0.55 |  |
| comp_tier4_bprp_limit | float | 1.1 |  |
| phase01_use_bprp_primary | bool | true | ``True`` = BP-RP tier + colour hard filter (Riello linear B-V fallback when needed); ``False`` = legacy /ΔB-V/ tiers via ``comp_tier*_bv_limit``. **Persisted** in ``config.json``; **intentionally-hidden** (no Settings toggle — edit json); default ``true``. Consumed by Phase 2A LC viewer (`ui_aperture_photometry.py`). |
| comp_tier1_bv_limit | float | 0.15 | Legacy /ΔB-V/ tier limity — report / export; pri ``phase01_use_bprp_primary`` sa nepoužívajú na výber. |
| comp_tier2_bv_limit | float | 0.3 |  |
| comp_tier3_bv_limit | float | 0.5 |  |
| comp_tier1_weight | float | 1.0 | Tier váhy pre ensemble/AC (multiplikátor k Broeg 1/σ²). |
| comp_tier2_weight | float | 0.85 |  |
| comp_tier3_weight | float | 0.5 |  |
| comp_tier4_weight | float | 0.25 |  |
| comp_contamination_penalty_k | float | 3.0 | Exponential contamination penalty in comp score: score *= exp(-k * contamination_idx). |
| phase01_comparison_n_comp_min | int | 3 |  |
| phase01_comparison_n_comp_max | int | 8 |  |
| phase01_comparison_max_comp_rms | float | 0.1 |  |
| phase01_comparison_min_dist_arcsec | float | 60.0 |  |
| phase01_comparison_min_frames_frac | float | 0.2 |  |
| phase01_comparison_exclude_gaia_nss | bool | true |  |
| phase01_comparison_exclude_gaia_extobj | bool | true |  |
| phase01_ct_min_comp | int | 7 | Fáza 2A: minimum number of comps used in color-term fit before applying CT (``should_apply_color_term``). |
| apply_color_term | str | "off" | Fáza 2A: apply BP-RP colour-term correction (``auto`` = on for B/V/Rc broadband, off for L/Clear). |
| k2_mode | str | "literature" | Second-order extinction: ``off``, ``literature``, or ``fit_else_literature`` (v2). |
| k2_defaults_bprp | object | {} | Optional per-band k'' overrides (mag/airmass/BP-RP). |
| k2_ceiling | float | 0.1 | Hard plausibility bound for fitted k'' (v2). |
| k2_fit_enabled | bool | false | Enable per-night k'' fit (v2). |
| k2_fit_min_detectability | float | 3.0 | Pre-gate: sigma_k2 vs literature detectability. |
| k2_fit_consistency_sigma | float | 2.0 | Pre-gate tertile/arc consistency threshold. |
| k2_fit_lit_factor | float | 4.0 | Pre-gate literature plausibility factor. |
| phase01_ct_extrapolation_tol | float | 0.0 | Fáza 2A: BP-RP tolerance (mag) when testing target vs comp range before applying CT; 0 = strict block on extrapolation. |
| phase01_flux_col | str | "dao_flux" | Column name used for flux in Phase 1 comp selection (dao_flux = aperture DAO; psf_flux = ePSF). |
| phase01_chip_interior_margin_px | int | 50 | Jednotný vnútorný okraj čipu (px) pre **celú Fázu 0+1**: aktívne premenné, porovnávacie hviezdy aj suspected. Hviezdy s ``x,y`` bližšie ako tento počet pixelov od okraja referenčného poľa sa neberú (zmierňuje artefakty pri zarovnaní / posune poľa / okrajoch). ``0`` = vypnuté (celý čip). Predvolene 50 px. |
| phase01_comparison_max_psf_chi2 | float | 50.0 | Aperture correction: reject comp stars with scatter above this (mag). Phase 1 comp gate: max reduced chi2 for PSF fit acceptance in comp selection. |
| phase01_comparison_max_fwhm_factor | float | 1.5 | Phase 1 comp gate: reject comps with FWHM > this factor × field median FWHM. |
| phase01_comparison_isolation_radius_px | float | 25.0 | Phase 1 comp gate: minimum isolation radius (px) — reject comps with neighbour closer than this. |
| phase01_comparison_rms_outlier_sigma | float | 3.0 | Phase 1 comp stability: sigma for outlier rejection in RMS stability check. |
| frame_width_px | int | 2082 | Sensor frame dimensions in pixels (used when FITS NAXIS1/2 unavailable). |
| frame_height_px | int | 1397 |  |
| field_density_sparse_threshold | float | 300.0 | Hustota poľa (hviezd/Mpx z DAO na MASTERSTAR): prahy a adaptívne úpravy Fázy 0+1 / apertúry (baseline = JSON). |
| field_density_dense_threshold | float | 1000.0 |  |
| field_density_adaptive_enabled | bool | true |  |
| global_comp_pool_enabled | bool | true | ``True`` = jeden globálny comp pool (safe_bbox + RMS) pred per-target výberom; ``False`` = legacy. |

## Phase 2A Photometry

| parameter_name | type | default | description |
|----------------|------|---------|-------------|
| aperture_photometry_enabled | bool | true | Use ``photutils`` circular aperture + annulus sky (replaces DAO ``flux`` in sidecar CSV when enabled). |
| save_lightcurve_png | bool | false | Fáza 2A: ukladať PNG (lightcurve, cutout, field map). ``False`` = len CSV + summary; UI používa Plotly z CSV. |
| phase2a_airmass_before_outlier | bool | false | Diagnostic only: ``True`` = pre-TODO-29 order (airmass fit → outlier detect). Default ``False`` keeps outlier → airmass. |
| aperture_fwhm_factor | float | 1.9 | Legacy single aperture factor — used where multi-aperture (B+C) is not active. |
| aperture_fwhm_factor_small | float | 1.5 | Multi-aperture (Method B+C foundation): small / medium / large radii as FWHM multiples. |
| aperture_fwhm_factor_medium | float | 2.5 |  |
| aperture_fwhm_factor_large | float | 4.0 |  |
| aperture_variable_factor | float | 1.0 | TODO-44: Role-aware scale on SNR-optimal radius (SIPS-style); 1.0 = no change. |
| aperture_comp_factor | float | 1.1 |  |
| annulus_inner_fwhm | float | 4.75 |  |
| annulus_outer_fwhm | float | 9.0 |  |
| nonlinearity_peak_percentile | float | 20.0 | Top ``p`` %% brightest by ``peak_max_adu`` checked for FWHM non-linearity vs field median. |
| nonlinearity_fwhm_ratio | float | 1.25 |  |
| aperture_correction_enabled | bool | true | Aperture correction (Method B): reserved for future pipeline; off by default. |
| aperture_correction_min_ref_stars | int | 3 |  |
| aperture_correction_max_contamination | float | 0.15 |  |
| aperture_correction_max_scatter_mag | float | 0.03 |  |
| comp_select_rms_floor | float | 1e-6 | Drop comps with comp_rms below floor before tier-ladder rank (isolated_bin artefact guard). |
| temporal_binning_enabled | bool | false | ALG-3: Temporal binning of comp ensemble before stability/PyTICS (Hartley & Wilson 2023 MNRAS). Default OFF — per-frame ensemble preserves common-mode cancellation. |
| temporal_bin_window | int | 0 |  |
| savgol_detrend_enabled | bool | false | ALG-2: Savitzky-Golay detrend after airmass (opt-in; Aigrain & Irwin 2004 MNRAS). |
| savgol_window_frac | float | 0.5 |  |
| savgol_polyorder | int | 2 |  |
| democratic_detrend_enabled | bool | false | ALG-4: Democratic Detrender ensemble detrend (Caballero-Nieves et al. 2026 arXiv:2411.09753v2). |
| democratic_sg_window_frac | float | 0.5 |  |
| pytics_enabled | bool | true | ALG-5: PyTICS iterative comp intercalibration after stability check (Marconi et al. 2026 RASTI). |
| pytics_n_iter | int | 5 |  |
| comp_max_slope_mmag_hr | float | 5.0 | Fáza 2A: exclude comparison stars with /linear slope/ above this (mmag/hr) in stability check. |
| psf_photometry_enabled | bool | false | Opt-in ePSF fitting on per-frame catalogs (adds ``psf_*`` columns; requires ``masterstar_epsf.fits``). |
| psf_spatial_order | int | 0 | ePSF spatial variation order (0=global; 1=linear) when photutils EPSFBuilder supports it. |
| psf_chi2_threshold | float | 50.0 | Reduced χ² cutoff for PSF fit acceptance (``psf_fit_ok``). |

## GS11 / Dilution

| parameter_name | type | default | description |
|----------------|------|---------|-------------|
| gs11_dilution_enabled | bool | false |  |

## Variability

| parameter_name | type | default | description |
|----------------|------|---------|-------------|
| variability_min_frames | int | 30 |  |
| variability_min_frames_frac | float | 0.5 |  |
| variability_sigma_clip | float | 5.0 |  |
| variability_p85_filter | int | 85 |  |
| variability_slope_floor | float | 0.02 |  |
| variability_sigma_threshold | float | 2.3 |  |
| variability_comp_floor_factor | float | 1.5 | Upper envelope floor = comp P90 rms_pct per mag bin × this factor (TODO-26). |
| variability_smoothness_max | float | 0.8 |  |
| variability_mag_limit | float | 14.5 |  |
| variability_min_rms_pct | float | 1.5 |  |
| variability_min_amplitude_mag | float | 0.01 |  |
| variability_clip_ratio_min | float | 0.8 |  |
| variability_vdi_z_threshold | float | 3.0 |  |
| variability_min_points_rms | int | 20 |  |
| tess_enabled | bool | false | ``True`` = sťahovanie/analýza TESS FFI cez lightkurve (TessCut), UI + ``tess_runner`` + pipeline hook. ``False`` = vypnuté — žiadne sťahovanie; log ``[TESS] preskočené``. Zapnúť: ``"tess_enabled": true`` v ``config.json``. |
| vsx_variable_targets_mag_limit | float | 14.5 | VSX export for variable_targets.csv: keep stars with ``mag_max`` <= limit (or unknown ``mag_max``). Set to ``<= 0`` to disable this cutoff (export all VSX rows in the field cone). |

## System / Paths

| parameter_name | type | default | description |
|----------------|------|---------|-------------|
| archive_root | str | "C:\\ASTRO\\python\\VYVAR\\Archive" |  |
| calibration_library_root | str | "C:\\ASTRO\\python\\VYVAR\\CalibrationLibrary" |  |
| database_path | str | "C:\\ASTRO\\python\\VYVAR\\vyvar.sqlite3" |  |
| gaia_db_path | str | "" | Path to local Gaia DR3 SQLite database (must contain table ``gaia_dr3`` with indexes on ra/dec). |
| blind_index_fine_path | str | "…/GAIA_DR3/gaia_triangles_fine.pkl" | Fine blind triangle index (Newton-scale rigs). |
| blind_index_wide_path | str | "…/GAIA_DR3/gaia_triangles_wide.pkl" | Wide blind triangle index (Carl-Zeiss-scale rigs). |
| blind_index_path | str | *(deprecated)* | Alias of ``blind_index_fine_path`` after load; not serialized. |
| vsx_local_db_path | str | "" | Path to local VSX subset SQLite (table ``vsx_data``: oid, ra_deg, dec_deg, …) for variable-star flags. |
| exoplanet_local_db_path | str | ``exoplanets/vyvar_exoplanet_local.db`` | Path to local exoplanet host SQLite (``exoplanet_data``); informational annotation only. |
| exoplanet_match_max_sep_arcsec | float | 3.0 | Per-detection exoplanet host match tolerance (arcsec). |
| catalog_query_max_rows | int | 15000 | After a cone query, keep at most this many catalog rows (brightest by ``mag``) to avoid RAM/CPU freeze. |
| sysrem_enabled | bool | false | TODO-35: SysRem (Tamuz et al. 2005) on exported ``lightcurve_*.csv`` after Phase 2A. |
| sysrem_n_iter | int | 3 |  |
