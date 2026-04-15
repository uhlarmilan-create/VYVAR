"""Project configuration for the variable-star processing system."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


def config_json_path(project_root: Path) -> Path:
    return project_root / "config.json"


def load_config_json(project_root: Path) -> dict[str, Any]:
    path = config_json_path(project_root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_config_json(project_root: Path, data: dict[str, Any]) -> None:
    path = config_json_path(project_root)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass(slots=True)
class AppConfig:
    """Central application config.

    SQLite schema is intentionally not defined yet.
    """

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # Calibration validity defaults (days)
    masterdark_validity_days: int = 60
    masterflat_validity_days: int = 200

    # Processing defaults
    sigma_clip_sigma: float = 3.0
    sigma_clip_maxiters: int = 5

    #: **Deprecated (unused):** CalibrationLibrary masters in ``importer`` use plain **mean** (dark) and
    #: **median** (flat) over frames — no sigma-clipping. Keys kept for older ``config.json`` files only.
    stacking_sigma: float = 3.0
    stacking_iters: int = 5

    #: After (light−dark)/flat: optional L.A.Cosmic (``astroscrappy``) in calibration step.
    cosmic_clean_enabled: bool = True
    cosmic_sigclip: float = 4.5
    cosmic_objlim: float = 5.0
    background_box_size: tuple[int, int] = (64, 64)
    background_filter_size: tuple[int, int] = (3, 3)
    max_live_targets: int = 10

    #: Estimated field diameter in degrees for VYVAR Gaia plate solve (used as **minimum** hint only).
    #: The actual Gaia kužeľ pre celý čip sa odvodzuje z FOCALLEN+PIXSIZE+NAXIS (pozri ``catalog_cone_radius_deg_from_optics``
    #: v ``utils.py`` a ``solve_wcs_with_local_gaia`` v ``vyvar_platesolver.py``); táto hodnota len zaručí spodnú hranicu.
    plate_solve_fov_deg: float = 1.0

    #: When applying masters from CalibrationLibrary, assume this **sensor** binning in the stored FITS
    #: (typically ``1`` = full resolution). Lights with ``XBINNING`` 2×2 are matched in RAM (temporary resample).
    #: JSON ``null``: read ``XBINNING`` from each master FITS (e.g. Bin2 library files with 2×2 lights).
    calibration_library_native_binning: int | None = 1

    #: Path to local Gaia DR3 SQLite database (must contain table ``gaia_dr3`` with indexes on ra/dec).
    gaia_db_path: str = ""

    #: Path to local VSX subset SQLite (table ``vsx_data``: oid, ra_deg, dec_deg, …) for variable-star flags.
    vsx_local_db_path: str = ""
    #: VSX export for variable_targets.csv: keep stars with mag_max <= limit (or unknown mag_max).
    vsx_variable_targets_mag_limit: float = 13.0

    #: After a cone query, keep at most this many catalog rows (brightest by ``mag``) to avoid RAM/CPU freeze.
    catalog_query_max_rows: int = 15_000

    #: If FITS has no ``SATURATE`` / ``MAXLIN`` / …, use this ceiling (ADU-like units) for ``likely_saturated``.
    #: Typical binned 16-bit OSC: 65535. Set to 0 or omit in JSON to disable fallback (float stacks in e⁻ may need a different value).
    photometry_fallback_saturate_adu: float | None = 65535.0

    #: Use ``photutils`` circular aperture + annulus sky (replaces DAO ``flux`` in sidecar CSV when enabled).
    aperture_photometry_enabled: bool = True
    #: Opt-in ePSF fitting on per-frame catalogs (adds ``psf_*`` columns; requires ``masterstar_epsf.fits``).
    psf_photometry_enabled: bool = False
    aperture_fwhm_factor: float = 1.7
    annulus_inner_fwhm: float = 4.0
    annulus_outer_fwhm: float = 6.0
    #: Top ``p`` %% brightest by ``peak_max_adu`` checked for FWHM non-linearity vs field median.
    nonlinearity_peak_percentile: float = 20.0
    nonlinearity_fwhm_ratio: float = 1.25
    #: Master-dark column BPM: MAD multiplier for ``*_dark_bpm.json`` (see ``importer``).
    bpm_dark_mad_sigma: float = 5.0

    #: If plate-solve hint RA/Dec vs draft median separation exceeds this (deg), use draft median for solver.
    masterstar_solver_use_draft_median_if_hint_sep_deg: float = 1.0
    #: Saturation safety fraction applied to equipment_saturate_adu before classifying MASTERSTAR zones.
    saturate_limit_fraction: float = 0.85
    #: Log astroalign stack summary (frames / reference index).
    masterstar_log_astroalign: bool = True
    #: After astrometry optimizer mirror-orientation warning, log an extra hint line.
    masterstar_optimizer_mirror_extra_log: bool = True
    #: VYVAR plate-solve na MASTERSTAR: max. SIP stupeň (2–5). Solver skúša **nadol** po ``masterstar_platesolve_sip_min_order`` (napr. 5→4→3).
    masterstar_platesolve_sip_max_order: int = 5
    #: Najnižší SIP stupeň pri páde vyšších (typicky 3; nie menej ako 2).
    masterstar_platesolve_sip_min_order: int = 3
    #: DAOStarFinder threshold = σ×RMS len pre MASTERSTAR katalóg (hlbšia detekcia; cieľ viac tisíc hviezd).
    masterstar_dao_threshold_sigma: float = 1.8
    #: Pred matchom s Gaia: ponechať detekcie s peakom aspoň ``median + k×σ`` (nižšie = viac slabých hviezd).
    masterstar_prematch_peak_sigma_floor: float = 3.2
    #: Ak zadané (> 0), MASTERSTAR plate-solve použije túto mierku [arcsec/pix] namiesto odvodzovania len z FOCALLEN×PIXSIZE v hlavičke (filter trojuholníkov / hint).
    masterstar_platesolve_expected_arcsec_per_px: float | None = None
    #: MASTERSTAR: horná hranica px RMS pred zápisom WCS (pred relaxáciou). ``None`` = predvolené 14 px.
    masterstar_platesolve_prewrite_rms_max_px: float | None = None
    #: MASTERSTAR: pri dobrom match_rate akceptovať RMS až do tejto hodnoty [px]. ``None`` = 22 px.
    masterstar_platesolve_prewrite_relaxed_rms_max_px: float | None = None
    #: MASTERSTAR: NN WCS refine sa aplikuje len ak RMS ≤ tejto hodnote [px]. ``None`` = 7.5 px.
    masterstar_platesolve_nn_refine_max_rms_px: float | None = None
    #: Pri ``force_apply`` SIP: zamietnuť ak ``rms_sip > rms_linear * ratio``. ``None`` = bez stráže (pôvodné správanie).
    masterstar_sip_force_rms_guard_ratio: float | None = 1.15

    #: Ak je True a WCS po VYVAR solve zlý (anizotropia / scale), skús ``solve-field`` v WSL (astrometry.net).
    astrometry_net_enabled: bool = True
    #: Cesta k ``astrometry.cfg`` vo WSL (Ubuntu), napr. ``/home/user/.config/astrometry.cfg``.
    astrometry_net_wsl_cfg: str = "/home/uhlar/.config/astrometry.cfg"
    #: Timeout [s] pre ``wsl solve-field`` pri MASTERSTAR fallback.
    astrometry_net_timeout_sec: int = 180
    #: Rozsah mierky pre ``solve-field --scale-low/--scale-high`` (± % okolo očakávanej mierky).
    astrometry_net_scale_err_pct: float = 15.0
    #: Polomer hintu pre ``solve-field --radius`` [deg].
    astrometry_net_radius_deg: float = 3.0
    #: Pomer sx/sy (arcsec/px) — nad týmto sa považuje WCS za príliš anizotropný (VYVAR retry / diagnostika).
    platesolve_anisotropy_threshold: float = 1.3

    #: If set (> 0), overrides physical pixel pitch [µm] read from FITS (native 1×1 sensor pixel, before binning).
    #: Effective pitch for plate solve / WCS is still ``force_pixel_size * binning`` in :func:`extract_fits_metadata`.
    #: JSON key: ``force_pixel_size``; omit or null to use header only.
    force_pixel_size_um: float | None = None

    #: Parallel workers for calibrated QC analyze / detrend preprocess / combined step (1–32). Persisted in ``config.json``.
    #: Environment variable ``VYVAR_QC_PREPROCESS_WORKERS`` overrides this when set.
    qc_preprocess_workers: int = 1

    #: Process workers for per-frame catalog CSV during plate-solve step 3 (DAO + match per FITS). Env ``VYVAR_PER_FRAME_CSV_WORKERS`` overrides.
    per_frame_csv_workers: int = 1
    #: Per-frame catalog matching: max sep [arcsec] for matching detections to the fixed master reference / cone catalog.
    #: Default is intentionally looser (20") so it works at coarse plate scales (~10"/px) and small residuals.
    per_frame_catalog_match_sep_arcsec: float = 20.0

    #: Reserve this much RAM (GB) when capping ``per_frame_csv_workers`` via ``psutil`` (parallel catalog export).
    per_frame_mp_reserve_ram_gb: float = 1.5

    #: Frame alignment (``astroalign`` + DAO positions): max brightest sources offered as control points per frame.
    alignment_max_stars: int = 200
    #: DAOStarFinder threshold multiplier vs sigma-clipped background RMS (higher = fewer, more significant peaks).
    alignment_detection_sigma: float = 5.0
    #: Same recipe as QC HFR star detection (``_mean_hfr_bright_stars_dao`` first pass: ``threshold = qc_dao_detection_sigma × std``).
    #: Used for frame alignment DAO so it tracks QC-style sensitivity.
    qc_dao_detection_sigma: float = 5.0

    #: DAOStarFinder FWHM (pixels) tuned for SIPS-like centroid search (aperture ~13 → ~4–5 px FWHM).
    sips_dao_fwhm_px: float = 2.5
    #: DAOStarFinder threshold = this × background RMS (SIPS “standard deviation count” ≈ 2.5).
    #: Pre hlboký MASTERSTAR / široké pole niekedy **0.25–1.0** (viac špičiek); používa sa aj pri VYVAR plate solve, ak volanie neprebije ``dao_threshold_sigma``.
    sips_dao_threshold_sigma: float = 3.5

    #: Post-calibration QC on each calibrated light (metrics + pass/fail vs limits).
    qc_after_calibrate_enabled: bool = True
    qc_max_hfr: float = 5.0
    qc_min_stars: int = 10
    #: If set, fail when sigma-clipped sky RMS exceeds this (same units as calibrated image).
    qc_max_background_rms: float | None = None

    # Paths derived from config.json (must stay after all init=True fields for dataclass(slots=True)).
    archive_root: Path = field(init=False)
    calibration_library_root: Path = field(init=False)
    database_path: Path = field(init=False)

    def __post_init__(self) -> None:
        data = load_config_json(self.project_root)

        self.archive_root = Path(data.get("archive_root", str(self.project_root / "Archive")))
        self.calibration_library_root = Path(
            data.get("calibration_library_root", str(self.project_root / "CalibrationLibrary"))
        )
        self.database_path = Path(data.get("database_path", str(self.project_root / "vyvar.sqlite3")))

        self.masterdark_validity_days = int(data.get("masterdark_validity_days", 60))
        self.masterflat_validity_days = int(data.get("masterflat_validity_days", 200))

        _fov = data.get("plate_solve_fov_deg", 1.0)
        try:
            self.plate_solve_fov_deg = float(_fov)
            if not math.isfinite(self.plate_solve_fov_deg) or self.plate_solve_fov_deg <= 0:
                self.plate_solve_fov_deg = 1.0
        except (TypeError, ValueError):
            self.plate_solve_fov_deg = 1.0
        # Migration: GAIA_DB_PATH supersedes legacy catalog settings.
        _cln_raw = data.get("calibration_library_native_binning", self.calibration_library_native_binning)
        if _cln_raw is None:
            self.calibration_library_native_binning = None
        else:
            try:
                _cln = int(_cln_raw)
                self.calibration_library_native_binning = max(1, min(16, _cln))
            except (TypeError, ValueError):
                self.calibration_library_native_binning = 1

        self.gaia_db_path = str(data.get("gaia_db_path", data.get("GAIA_DB_PATH", "")) or "").strip()

        self.vsx_local_db_path = str(
            data.get("vsx_local_db_path", data.get("VSX_LOCAL_DB_PATH", "")) or ""
        ).strip()
        _vml = data.get("vsx_variable_targets_mag_limit", self.vsx_variable_targets_mag_limit)
        try:
            self.vsx_variable_targets_mag_limit = float(_vml)
            if not math.isfinite(self.vsx_variable_targets_mag_limit) or self.vsx_variable_targets_mag_limit <= 0:
                self.vsx_variable_targets_mag_limit = 13.0
        except (TypeError, ValueError):
            self.vsx_variable_targets_mag_limit = 13.0
        try:
            self.catalog_query_max_rows = max(
                1000, min(500_000, int(data.get("catalog_query_max_rows", self.catalog_query_max_rows)))
            )
        except (TypeError, ValueError):
            self.catalog_query_max_rows = 15_000
        _pfs = data.get("photometry_fallback_saturate_adu", 65535.0)
        if _pfs is None or _pfs == "":
            self.photometry_fallback_saturate_adu = None
        else:
            _pv = float(_pfs)
            self.photometry_fallback_saturate_adu = _pv if _pv > 0 else None

        _fps = data.get("force_pixel_size", self.force_pixel_size_um)
        if _fps is None or _fps == "":
            self.force_pixel_size_um = None
        else:
            try:
                _fv = float(_fps)
                self.force_pixel_size_um = _fv if _fv > 0 and math.isfinite(_fv) else None
            except (TypeError, ValueError):
                self.force_pixel_size_um = None

        try:
            self.qc_preprocess_workers = max(
                1, min(32, int(data.get("qc_preprocess_workers", self.qc_preprocess_workers)))
            )
        except (TypeError, ValueError):
            self.qc_preprocess_workers = 1
        try:
            self.per_frame_csv_workers = max(
                1, min(32, int(data.get("per_frame_csv_workers", self.per_frame_csv_workers)))
            )
        except (TypeError, ValueError):
            self.per_frame_csv_workers = 1
        _pfm = data.get(
            "per_frame_catalog_match_sep_arcsec",
            self.per_frame_catalog_match_sep_arcsec,
        )
        try:
            self.per_frame_catalog_match_sep_arcsec = float(_pfm)
            if (
                not math.isfinite(self.per_frame_catalog_match_sep_arcsec)
                or self.per_frame_catalog_match_sep_arcsec <= 0
            ):
                self.per_frame_catalog_match_sep_arcsec = 20.0
        except (TypeError, ValueError):
            self.per_frame_catalog_match_sep_arcsec = 20.0
        try:
            self.per_frame_mp_reserve_ram_gb = float(
                data.get("per_frame_mp_reserve_ram_gb", self.per_frame_mp_reserve_ram_gb)
            )
            if not math.isfinite(self.per_frame_mp_reserve_ram_gb) or self.per_frame_mp_reserve_ram_gb < 0:
                self.per_frame_mp_reserve_ram_gb = 1.5
        except (TypeError, ValueError):
            self.per_frame_mp_reserve_ram_gb = 1.5

        try:
            self.alignment_max_stars = max(
                10, min(5000, int(data.get("alignment_max_stars", self.alignment_max_stars)))
            )
        except (TypeError, ValueError):
            self.alignment_max_stars = 200
        try:
            self.alignment_detection_sigma = float(
                data.get("alignment_detection_sigma", self.alignment_detection_sigma)
            )
            if not math.isfinite(self.alignment_detection_sigma) or self.alignment_detection_sigma <= 0:
                self.alignment_detection_sigma = 5.0
        except (TypeError, ValueError):
            self.alignment_detection_sigma = 5.0
        try:
            self.qc_dao_detection_sigma = float(data.get("qc_dao_detection_sigma", self.qc_dao_detection_sigma))
            if not math.isfinite(self.qc_dao_detection_sigma) or self.qc_dao_detection_sigma <= 0:
                self.qc_dao_detection_sigma = 5.0
        except (TypeError, ValueError):
            self.qc_dao_detection_sigma = 5.0
        try:
            self.sips_dao_fwhm_px = float(data.get("sips_dao_fwhm_px", self.sips_dao_fwhm_px))
            if not math.isfinite(self.sips_dao_fwhm_px) or self.sips_dao_fwhm_px <= 0:
                self.sips_dao_fwhm_px = 2.5
        except (TypeError, ValueError):
            self.sips_dao_fwhm_px = 2.5
        self.sips_dao_fwhm_px = max(1.0, min(8.0, float(self.sips_dao_fwhm_px)))
        try:
            self.sips_dao_threshold_sigma = float(
                data.get("sips_dao_threshold_sigma", self.sips_dao_threshold_sigma)
            )
            if not math.isfinite(self.sips_dao_threshold_sigma) or self.sips_dao_threshold_sigma <= 0:
                self.sips_dao_threshold_sigma = 3.5
        except (TypeError, ValueError):
            self.sips_dao_threshold_sigma = 3.5

        try:
            self.stacking_sigma = float(data.get("stacking_sigma", self.stacking_sigma))
        except (TypeError, ValueError):
            self.stacking_sigma = 3.0
        try:
            self.stacking_iters = max(1, int(data.get("stacking_iters", self.stacking_iters)))
        except (TypeError, ValueError):
            self.stacking_iters = 5

        self.cosmic_clean_enabled = bool(data.get("cosmic_clean_enabled", self.cosmic_clean_enabled))
        try:
            self.cosmic_sigclip = float(data.get("cosmic_sigclip", self.cosmic_sigclip))
        except (TypeError, ValueError):
            self.cosmic_sigclip = 4.5
        try:
            self.cosmic_objlim = float(data.get("cosmic_objlim", self.cosmic_objlim))
        except (TypeError, ValueError):
            self.cosmic_objlim = 5.0

        self.qc_after_calibrate_enabled = bool(
            data.get("qc_after_calibrate_enabled", self.qc_after_calibrate_enabled)
        )
        try:
            self.qc_max_hfr = float(data.get("qc_max_hfr", self.qc_max_hfr))
        except (TypeError, ValueError):
            self.qc_max_hfr = 5.0
        try:
            self.qc_min_stars = max(0, int(data.get("qc_min_stars", self.qc_min_stars)))
        except (TypeError, ValueError):
            self.qc_min_stars = 10
        _qmr = data.get("qc_max_background_rms", self.qc_max_background_rms)
        if _qmr is None or _qmr == "":
            self.qc_max_background_rms = None
        else:
            try:
                v = float(_qmr)
                self.qc_max_background_rms = v if v > 0 and math.isfinite(v) else None
            except (TypeError, ValueError):
                self.qc_max_background_rms = None

        self.aperture_photometry_enabled = bool(data.get("aperture_photometry_enabled", self.aperture_photometry_enabled))
        self.psf_photometry_enabled = bool(data.get("psf_photometry_enabled", self.psf_photometry_enabled))
        try:
            self.aperture_fwhm_factor = float(data.get("aperture_fwhm_factor", self.aperture_fwhm_factor))
            if not math.isfinite(self.aperture_fwhm_factor) or self.aperture_fwhm_factor <= 0:
                self.aperture_fwhm_factor = 1.7
        except (TypeError, ValueError):
            self.aperture_fwhm_factor = 1.7
        self.aperture_fwhm_factor = max(0.5, min(6.0, float(self.aperture_fwhm_factor)))
        try:
            self.annulus_inner_fwhm = float(data.get("annulus_inner_fwhm", self.annulus_inner_fwhm))
            self.annulus_outer_fwhm = float(data.get("annulus_outer_fwhm", self.annulus_outer_fwhm))
        except (TypeError, ValueError):
            self.annulus_inner_fwhm = 4.0
            self.annulus_outer_fwhm = 6.0
        if self.annulus_outer_fwhm <= self.annulus_inner_fwhm:
            self.annulus_outer_fwhm = self.annulus_inner_fwhm + 1.0
        try:
            self.nonlinearity_peak_percentile = float(
                data.get("nonlinearity_peak_percentile", self.nonlinearity_peak_percentile)
            )
        except (TypeError, ValueError):
            self.nonlinearity_peak_percentile = 20.0
        self.nonlinearity_peak_percentile = max(0.0, min(50.0, float(self.nonlinearity_peak_percentile)))
        try:
            self.nonlinearity_fwhm_ratio = float(data.get("nonlinearity_fwhm_ratio", self.nonlinearity_fwhm_ratio))
        except (TypeError, ValueError):
            self.nonlinearity_fwhm_ratio = 1.25
        self.nonlinearity_fwhm_ratio = max(1.01, min(3.0, float(self.nonlinearity_fwhm_ratio)))
        try:
            self.bpm_dark_mad_sigma = float(data.get("bpm_dark_mad_sigma", self.bpm_dark_mad_sigma))
        except (TypeError, ValueError):
            self.bpm_dark_mad_sigma = 5.0
        self.bpm_dark_mad_sigma = max(2.0, min(12.0, float(self.bpm_dark_mad_sigma)))

        try:
            self.masterstar_solver_use_draft_median_if_hint_sep_deg = float(
                data.get(
                    "masterstar_solver_use_draft_median_if_hint_sep_deg",
                    self.masterstar_solver_use_draft_median_if_hint_sep_deg,
                )
            )
            if not math.isfinite(self.masterstar_solver_use_draft_median_if_hint_sep_deg):
                self.masterstar_solver_use_draft_median_if_hint_sep_deg = 1.0
        except (TypeError, ValueError):
            self.masterstar_solver_use_draft_median_if_hint_sep_deg = 1.0
        self.masterstar_solver_use_draft_median_if_hint_sep_deg = max(0.0, min(180.0, float(self.masterstar_solver_use_draft_median_if_hint_sep_deg)))
        self.masterstar_log_astroalign = bool(data.get("masterstar_log_astroalign", self.masterstar_log_astroalign))
        self.masterstar_optimizer_mirror_extra_log = bool(
            data.get("masterstar_optimizer_mirror_extra_log", self.masterstar_optimizer_mirror_extra_log)
        )
        try:
            self.masterstar_platesolve_sip_max_order = int(
                data.get("masterstar_platesolve_sip_max_order", self.masterstar_platesolve_sip_max_order)
            )
        except (TypeError, ValueError):
            self.masterstar_platesolve_sip_max_order = 5
        self.masterstar_platesolve_sip_max_order = max(2, min(5, int(self.masterstar_platesolve_sip_max_order)))
        try:
            self.masterstar_platesolve_sip_min_order = int(
                data.get("masterstar_platesolve_sip_min_order", self.masterstar_platesolve_sip_min_order)
            )
        except (TypeError, ValueError):
            self.masterstar_platesolve_sip_min_order = 3
        self.masterstar_platesolve_sip_min_order = max(2, min(5, int(self.masterstar_platesolve_sip_min_order)))
        if self.masterstar_platesolve_sip_min_order > self.masterstar_platesolve_sip_max_order:
            self.masterstar_platesolve_sip_min_order = int(self.masterstar_platesolve_sip_max_order)
        try:
            self.masterstar_dao_threshold_sigma = float(
                data.get("masterstar_dao_threshold_sigma", self.masterstar_dao_threshold_sigma)
            )
        except (TypeError, ValueError):
            self.masterstar_dao_threshold_sigma = 1.8
        self.masterstar_dao_threshold_sigma = max(0.1, min(6.0, float(self.masterstar_dao_threshold_sigma)))
        try:
            self.masterstar_prematch_peak_sigma_floor = float(
                data.get("masterstar_prematch_peak_sigma_floor", self.masterstar_prematch_peak_sigma_floor)
            )
        except (TypeError, ValueError):
            self.masterstar_prematch_peak_sigma_floor = 3.2
        self.masterstar_prematch_peak_sigma_floor = max(0.5, min(6.0, float(self.masterstar_prematch_peak_sigma_floor)))

        def _opt_pos_float(key: str, lo: float, hi: float) -> float | None:
            raw = data.get(key)
            if raw is None or raw == "":
                return None
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(v) or v <= 0:
                return None
            return max(lo, min(hi, v))

        self.masterstar_platesolve_expected_arcsec_per_px = _opt_pos_float(
            "masterstar_platesolve_expected_arcsec_per_px", 0.01, 120.0
        )
        self.masterstar_platesolve_prewrite_rms_max_px = _opt_pos_float(
            "masterstar_platesolve_prewrite_rms_max_px", 1.0, 80.0
        )
        self.masterstar_platesolve_prewrite_relaxed_rms_max_px = _opt_pos_float(
            "masterstar_platesolve_prewrite_relaxed_rms_max_px", 1.0, 120.0
        )
        self.masterstar_platesolve_nn_refine_max_rms_px = _opt_pos_float(
            "masterstar_platesolve_nn_refine_max_rms_px", 0.5, 50.0
        )

        _msiprg = data.get("masterstar_sip_force_rms_guard_ratio", self.masterstar_sip_force_rms_guard_ratio)
        if _msiprg is None or _msiprg == "":
            self.masterstar_sip_force_rms_guard_ratio = None
        else:
            try:
                v = float(_msiprg)
                self.masterstar_sip_force_rms_guard_ratio = v if math.isfinite(v) and v > 0 else 1.15
            except (TypeError, ValueError):
                self.masterstar_sip_force_rms_guard_ratio = 1.15

        self.astrometry_net_enabled = bool(data.get("astrometry_net_enabled", self.astrometry_net_enabled))
        _acfg = data.get("astrometry_net_wsl_cfg", self.astrometry_net_wsl_cfg)
        self.astrometry_net_wsl_cfg = str(_acfg or "").strip() or "/home/uhlar/.config/astrometry.cfg"
        try:
            self.astrometry_net_timeout_sec = max(
                30, min(3600, int(data.get("astrometry_net_timeout_sec", self.astrometry_net_timeout_sec)))
            )
        except (TypeError, ValueError):
            self.astrometry_net_timeout_sec = 180
        try:
            self.astrometry_net_scale_err_pct = float(
                data.get("astrometry_net_scale_err_pct", self.astrometry_net_scale_err_pct)
            )
            if not math.isfinite(self.astrometry_net_scale_err_pct):
                self.astrometry_net_scale_err_pct = 15.0
        except (TypeError, ValueError):
            self.astrometry_net_scale_err_pct = 15.0
        self.astrometry_net_scale_err_pct = max(1.0, min(80.0, float(self.astrometry_net_scale_err_pct)))
        try:
            self.astrometry_net_radius_deg = float(
                data.get("astrometry_net_radius_deg", self.astrometry_net_radius_deg)
            )
            if not math.isfinite(self.astrometry_net_radius_deg):
                self.astrometry_net_radius_deg = 3.0
        except (TypeError, ValueError):
            self.astrometry_net_radius_deg = 3.0
        self.astrometry_net_radius_deg = max(0.1, min(30.0, float(self.astrometry_net_radius_deg)))
        try:
            self.platesolve_anisotropy_threshold = float(
                data.get("platesolve_anisotropy_threshold", self.platesolve_anisotropy_threshold)
            )
            if not math.isfinite(self.platesolve_anisotropy_threshold):
                self.platesolve_anisotropy_threshold = 1.3
        except (TypeError, ValueError):
            self.platesolve_anisotropy_threshold = 1.3
        self.platesolve_anisotropy_threshold = max(1.01, min(5.0, float(self.platesolve_anisotropy_threshold)))

    def to_json(self) -> dict[str, Any]:
        return {
            "archive_root": str(self.archive_root),
            "calibration_library_root": str(self.calibration_library_root),
            "database_path": str(self.database_path),
            "masterdark_validity_days": int(self.masterdark_validity_days),
            "masterflat_validity_days": int(self.masterflat_validity_days),
            "plate_solve_fov_deg": float(self.plate_solve_fov_deg),
            "calibration_library_native_binning": (
                None
                if self.calibration_library_native_binning is None
                else int(self.calibration_library_native_binning)
            ),
            "gaia_db_path": str(self.gaia_db_path or ""),
            "vsx_local_db_path": str(self.vsx_local_db_path or ""),
            "vsx_variable_targets_mag_limit": float(self.vsx_variable_targets_mag_limit),
            "catalog_query_max_rows": int(self.catalog_query_max_rows),
            "photometry_fallback_saturate_adu": (
                float(self.photometry_fallback_saturate_adu)
                if self.photometry_fallback_saturate_adu is not None
                else None
            ),
            "force_pixel_size": (
                float(self.force_pixel_size_um) if self.force_pixel_size_um is not None else None
            ),
            "qc_preprocess_workers": int(self.qc_preprocess_workers),
            "per_frame_csv_workers": int(self.per_frame_csv_workers),
            "per_frame_catalog_match_sep_arcsec": float(self.per_frame_catalog_match_sep_arcsec),
            "per_frame_mp_reserve_ram_gb": float(self.per_frame_mp_reserve_ram_gb),
            "alignment_max_stars": int(self.alignment_max_stars),
            "alignment_detection_sigma": float(self.alignment_detection_sigma),
            "qc_dao_detection_sigma": float(self.qc_dao_detection_sigma),
            "sips_dao_fwhm_px": float(self.sips_dao_fwhm_px),
            "sips_dao_threshold_sigma": float(self.sips_dao_threshold_sigma),
            "stacking_sigma": float(self.stacking_sigma),
            "stacking_iters": int(self.stacking_iters),
            "cosmic_clean_enabled": bool(self.cosmic_clean_enabled),
            "cosmic_sigclip": float(self.cosmic_sigclip),
            "cosmic_objlim": float(self.cosmic_objlim),
            "qc_after_calibrate_enabled": bool(self.qc_after_calibrate_enabled),
            "qc_max_hfr": float(self.qc_max_hfr),
            "qc_min_stars": int(self.qc_min_stars),
            "qc_max_background_rms": (
                float(self.qc_max_background_rms)
                if self.qc_max_background_rms is not None
                else None
            ),
            "aperture_photometry_enabled": bool(self.aperture_photometry_enabled),
            "psf_photometry_enabled": bool(self.psf_photometry_enabled),
            "aperture_fwhm_factor": float(self.aperture_fwhm_factor),
            "annulus_inner_fwhm": float(self.annulus_inner_fwhm),
            "annulus_outer_fwhm": float(self.annulus_outer_fwhm),
            "nonlinearity_peak_percentile": float(self.nonlinearity_peak_percentile),
            "nonlinearity_fwhm_ratio": float(self.nonlinearity_fwhm_ratio),
            "bpm_dark_mad_sigma": float(self.bpm_dark_mad_sigma),
            "masterstar_solver_use_draft_median_if_hint_sep_deg": float(
                self.masterstar_solver_use_draft_median_if_hint_sep_deg
            ),
            "masterstar_log_astroalign": bool(self.masterstar_log_astroalign),
            "masterstar_optimizer_mirror_extra_log": bool(self.masterstar_optimizer_mirror_extra_log),
            "masterstar_platesolve_sip_max_order": int(self.masterstar_platesolve_sip_max_order),
            "masterstar_platesolve_sip_min_order": int(self.masterstar_platesolve_sip_min_order),
            "masterstar_dao_threshold_sigma": float(self.masterstar_dao_threshold_sigma),
            "masterstar_prematch_peak_sigma_floor": float(self.masterstar_prematch_peak_sigma_floor),
            "masterstar_platesolve_expected_arcsec_per_px": (
                float(self.masterstar_platesolve_expected_arcsec_per_px)
                if self.masterstar_platesolve_expected_arcsec_per_px is not None
                else None
            ),
            "masterstar_platesolve_prewrite_rms_max_px": (
                float(self.masterstar_platesolve_prewrite_rms_max_px)
                if self.masterstar_platesolve_prewrite_rms_max_px is not None
                else None
            ),
            "masterstar_platesolve_prewrite_relaxed_rms_max_px": (
                float(self.masterstar_platesolve_prewrite_relaxed_rms_max_px)
                if self.masterstar_platesolve_prewrite_relaxed_rms_max_px is not None
                else None
            ),
            "masterstar_platesolve_nn_refine_max_rms_px": (
                float(self.masterstar_platesolve_nn_refine_max_rms_px)
                if self.masterstar_platesolve_nn_refine_max_rms_px is not None
                else None
            ),
            "masterstar_sip_force_rms_guard_ratio": (
                float(self.masterstar_sip_force_rms_guard_ratio)
                if self.masterstar_sip_force_rms_guard_ratio is not None
                else None
            ),
            "astrometry_net_enabled": bool(self.astrometry_net_enabled),
            "astrometry_net_wsl_cfg": str(self.astrometry_net_wsl_cfg or ""),
            "astrometry_net_timeout_sec": int(self.astrometry_net_timeout_sec),
            "astrometry_net_scale_err_pct": float(self.astrometry_net_scale_err_pct),
            "astrometry_net_radius_deg": float(self.astrometry_net_radius_deg),
            "platesolve_anisotropy_threshold": float(self.platesolve_anisotropy_threshold),
        }

    # Backward-compatible alias (some callers expect to_dict()).
    def to_dict(self) -> dict[str, Any]:
        return self.to_json()

    def ensure_base_dirs(self) -> None:
        """Create base directories required by file-first workflow."""
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.calibration_library_root.mkdir(parents=True, exist_ok=True)

