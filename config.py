"""Project configuration for the variable-star processing system."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
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


def recommended_vyvar_parallel_workers(*, reserve_ram_gb: float = 1.5) -> int:
    """Jednotný počet workerov pre QC, preprocess, combined, per-frame CSV (základ pred RAM stropom), alignment, calibrate MP.

    Berie minimum z CPU-heuristiky a odhadu podľa voľnej RAM (rovnaký odhad pamäte ako pri per-frame exporte),
    aby jedna hodnota bola bezpečná v celom workflow.
    """
    n = os.cpu_count()
    if n is None or n < 1:
        n = 4
    if n <= 1:
        cpu_cap = 1
    else:
        cpu_cap = max(1, min(32, min(n - 1, 16, max(1, n // 2))))
    h, wpx = 2048, 2048
    per_worker = max(int(h * wpx * 4 * 3), 1)
    try:
        import psutil

        reserve = int(max(0.0, float(reserve_ram_gb)) * (1024**3))
        avail = int(psutil.virtual_memory().available) - reserve
        if avail <= 0:
            ram_cap = 1
        else:
            ram_cap = max(1, min(32, avail // per_worker))
    except Exception:
        ram_cap = 32
    return max(1, min(32, min(cpu_cap, ram_cap)))


@dataclass(slots=True)
class AppConfig:
    """Central application config.

    SQLite schema is intentionally not defined yet.
    """

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # Calibration validity defaults (days)
    masterdark_validity_days: int = 60
    masterflat_validity_days: int = 200

    #: After (light−dark)/flat: optional L.A.Cosmic (``astroscrappy``) in calibration step.
    cosmic_clean_enabled: bool = True
    cosmic_sigclip: float = 4.5
    cosmic_objlim: float = 5.0

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

    #: Path to blind plate-solve triangle hash index (``gaia_triangles.pkl``, from ``build_gaia_blind_index.py``).
    blind_index_path: str = r"C:\ASTRO\python\VYVAR\GAIA_DR3\gaia_triangles.pkl"

    #: Path to local VSX subset SQLite (table ``vsx_data``: oid, ra_deg, dec_deg, …) for variable-star flags.
    vsx_local_db_path: str = ""
    #: VSX export for variable_targets.csv: keep stars with ``mag_max`` <= limit (or unknown ``mag_max``).
    #: Set to ``<= 0`` to disable this cutoff (export all VSX rows in the field cone).
    vsx_variable_targets_mag_limit: float = 13.0

    #: After a cone query, keep at most this many catalog rows (brightest by ``mag``) to avoid RAM/CPU freeze.
    catalog_query_max_rows: int = 15_000

    #: Use ``photutils`` circular aperture + annulus sky (replaces DAO ``flux`` in sidecar CSV when enabled).
    aperture_photometry_enabled: bool = True
    #: Fáza 2A: ukladať PNG (lightcurve, cutout, field map). ``False`` = len CSV + summary; UI používa Plotly z CSV.
    save_lightcurve_png: bool = False
    #: Opt-in ePSF fitting on per-frame catalogs (adds ``psf_*`` columns; requires ``masterstar_epsf.fits``).
    psf_photometry_enabled: bool = False
    # NOTE: These are in units of **Gaussian FWHM** (not moment-FWHM).
    # Aperture/annulus radii are computed as factor × fwhm_gaussian_px.
    aperture_fwhm_factor: float = 2.75
    annulus_inner_fwhm: float = 5.5
    annulus_outer_fwhm: float = 10.5
    #: Top ``p`` %% brightest by ``peak_max_adu`` checked for FWHM non-linearity vs field median.
    nonlinearity_peak_percentile: float = 20.0
    nonlinearity_fwhm_ratio: float = 1.25
    #: Master-dark column BPM: MAD multiplier for ``*_dark_bpm.json`` (see ``importer``).
    bpm_dark_mad_sigma: float = 5.0

    #: If plate-solve hint RA/Dec vs draft median separation exceeds this (deg), use draft median for solver.
    masterstar_solver_use_draft_median_if_hint_sep_deg: float = 1.0
    #: Saturation safety fraction applied to equipment_saturate_adu before classifying MASTERSTAR zones.
    saturate_limit_fraction: float = 0.85
    #: Log zarovnanie (astroalign): referenčný rámec a počty kontrolných bodov.
    masterstar_log_astroalign: bool = True
    #: After astrometry optimizer mirror-orientation warning, log an extra hint line.
    masterstar_optimizer_mirror_extra_log: bool = True
    #: Enable verbose debug logs for plate solving / blind solver / hint plumbing.
    debug_platesolver: bool = False
    #: VYVAR plate-solve na MASTERSTAR: max. SIP stupeň (2–5). Solver skúša **nadol** po ``masterstar_platesolve_sip_min_order`` (napr. 5→4→3).
    masterstar_platesolve_sip_max_order: int = 5
    #: Najnižší SIP stupeň pri páde vyšších (typicky 3; nie menej ako 2).
    masterstar_platesolve_sip_min_order: int = 3
    #: DAOStarFinder threshold = σ×RMS len pre MASTERSTAR katalóg (hlbšia detekcia; cieľ viac tisíc hviezd).
    masterstar_dao_threshold_sigma: float = 1.8
    #: Pred matchom s Gaia: ponechať detekcie s peakom aspoň ``median + k×σ`` (nižšie = viac slabých hviezd).
    masterstar_prematch_peak_sigma_floor: float = 3.2
    #: MASTERSTAR: horná hranica px RMS pred zápisom WCS (pred relaxáciou). ``None`` = predvolené 14 px.
    masterstar_platesolve_prewrite_rms_max_px: float | None = None
    #: MASTERSTAR: pri dobrom match_rate akceptovať RMS až do tejto hodnoty [px]. ``None`` = 22 px.
    masterstar_platesolve_prewrite_relaxed_rms_max_px: float | None = None
    #: MASTERSTAR: NN WCS refine sa aplikuje len ak RMS ≤ tejto hodnote [px]. ``None`` = 7.5 px.
    masterstar_platesolve_nn_refine_max_rms_px: float | None = None
    #: Pri ``force_apply`` SIP: zamietnuť ak ``rms_sip > rms_linear * ratio``. ``None`` = bez stráže (pôvodné správanie).
    masterstar_sip_force_rms_guard_ratio: float | None = 1.15

    #: Pomer sx/sy (arcsec/px) — nad týmto sa považuje WCS za príliš anizotropný (VYVAR retry / diagnostika).
    platesolve_anisotropy_threshold: float = 1.3

    #: Paralelizmus (QC, preprocess, combined, per-frame CSV, alignment, calibrate MP): jedna hodnota
    #: počítaná v ``__post_init__``; nie v ``config.json``. Runtime override: ``VYVAR_PARALLEL_WORKERS`` alebo legacy env v pipeline.
    qc_preprocess_workers: int = 1
    #: Reserve this much RAM (GB) when capping paralelného exportu katalógov cez ``psutil`` (nad rámec jednotného ``_pw``).
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

    #: Fáza 0+1 — výber porovnávacích hviezd (``photometry_core.select_comparison_stars_per_target``).
    #: Pri **riedkom poli** zväčši ``phase01_comparison_max_mag_diff`` / ``phase01_comparison_max_dist_deg``,
    #: prípadne zníž ``phase01_comparison_min_frames_frac`` alebo zvýš ``phase01_comparison_max_comp_rms`` (slabší filter stability).
    #: Pri **jasných cieľoch** (``mag`` < ``phase01_comparison_mag_bright_threshold``) sa použije aspoň
    #: ``phase01_comparison_max_mag_diff_bright_floor`` ako minimálny |Δmag| pás (``0`` = vypnuté).
    phase01_comparison_max_dist_deg: float = 1.0
    phase01_comparison_max_mag_diff: float = 0.25
    phase01_comparison_mag_bright_threshold: float = 12.0
    phase01_comparison_max_mag_diff_bright_floor: float = 1.25
    phase01_comparison_max_bv_diff: float = 0.15
    phase01_comparison_n_comp_min: int = 3
    phase01_comparison_n_comp_max: int = 7
    phase01_comparison_max_comp_rms: float = 0.05
    phase01_comparison_min_dist_arcsec: float = 60.0
    phase01_comparison_min_frames_frac: float = 0.3
    phase01_comparison_exclude_gaia_nss: bool = True
    phase01_comparison_exclude_gaia_extobj: bool = True

    #: Jednotný vnútorný okraj čipu (px) pre **celú Fázu 0+1**: aktívne premenné, porovnávacie hviezdy aj suspected.
    #: Hviezdy s ``x,y`` bližšie ako tento počet pixelov od okraja referenčného poľa sa neberú (zmierňuje artefakty
    #: pri zarovnaní / posune poľa / okrajoch). ``0`` = vypnuté (celý čip). Predvolene 100 px.
    phase01_chip_interior_margin_px: int = 100

    # Variability Detection
    variability_min_frames: int = 30
    variability_min_frames_frac: float = 0.50
    variability_sigma_clip: float = 5.0
    variability_p85_filter: int = 85
    variability_slope_floor: float = 0.02
    variability_sigma_threshold: float = 3.0
    variability_smoothness_max: float = 0.80
    variability_mag_limit: float = 14.5
    variability_min_rms_pct: float = 1.5
    variability_min_amplitude_mag: float = 0.01
    variability_clip_ratio_min: float = 0.80
    variability_vdi_z_threshold: float = 3.0
    variability_min_points_rms: int = 20

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

        # ``plate_solve_fov_deg`` is no longer read from JSON — resolved from FITS + DB (see ``resolve_plate_solve_fov_deg_hint``).
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

        _blind_default = self.blind_index_path
        self.blind_index_path = str(
            data.get("blind_index_path", data.get("BLIND_INDEX_PATH", "")) or ""
        ).strip()
        if not self.blind_index_path:
            self.blind_index_path = str(_blind_default or "").strip()

        self.vsx_local_db_path = str(
            data.get("vsx_local_db_path", data.get("VSX_LOCAL_DB_PATH", "")) or ""
        ).strip()
        _vml = data.get("vsx_variable_targets_mag_limit", self.vsx_variable_targets_mag_limit)
        try:
            self.vsx_variable_targets_mag_limit = float(_vml)
            if not math.isfinite(self.vsx_variable_targets_mag_limit):
                self.vsx_variable_targets_mag_limit = 13.0
            # ``<= 0`` = žiadny mag. rez VSX (export všetkých v kuželi); ``> 0`` = max ``mag_max`` z VSX.
        except (TypeError, ValueError):
            self.vsx_variable_targets_mag_limit = 13.0
        try:
            self.catalog_query_max_rows = max(
                1000, min(500_000, int(data.get("catalog_query_max_rows", self.catalog_query_max_rows)))
            )
        except (TypeError, ValueError):
            self.catalog_query_max_rows = 15_000

        try:
            self.per_frame_mp_reserve_ram_gb = float(
                data.get("per_frame_mp_reserve_ram_gb", self.per_frame_mp_reserve_ram_gb)
            )
            if not math.isfinite(self.per_frame_mp_reserve_ram_gb) or self.per_frame_mp_reserve_ram_gb < 0:
                self.per_frame_mp_reserve_ram_gb = 1.5
        except (TypeError, ValueError):
            self.per_frame_mp_reserve_ram_gb = 1.5

        _pw = int(
            recommended_vyvar_parallel_workers(reserve_ram_gb=float(self.per_frame_mp_reserve_ram_gb))
        )
        self.qc_preprocess_workers = _pw

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
        self.save_lightcurve_png = bool(data.get("save_lightcurve_png", self.save_lightcurve_png))
        self.psf_photometry_enabled = bool(data.get("psf_photometry_enabled", self.psf_photometry_enabled))
        try:
            self.aperture_fwhm_factor = float(data.get("aperture_fwhm_factor", self.aperture_fwhm_factor))
            if not math.isfinite(self.aperture_fwhm_factor) or self.aperture_fwhm_factor <= 0:
                self.aperture_fwhm_factor = 2.75
        except (TypeError, ValueError):
            self.aperture_fwhm_factor = 2.75
        self.aperture_fwhm_factor = max(0.5, min(6.0, float(self.aperture_fwhm_factor)))
        try:
            self.annulus_inner_fwhm = float(data.get("annulus_inner_fwhm", self.annulus_inner_fwhm))
            self.annulus_outer_fwhm = float(data.get("annulus_outer_fwhm", self.annulus_outer_fwhm))
        except (TypeError, ValueError):
            self.annulus_inner_fwhm = 5.5
            self.annulus_outer_fwhm = 10.5
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
        self.debug_platesolver = bool(data.get("debug_platesolver", self.debug_platesolver))
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

        try:
            self.platesolve_anisotropy_threshold = float(
                data.get("platesolve_anisotropy_threshold", self.platesolve_anisotropy_threshold)
            )
            if not math.isfinite(self.platesolve_anisotropy_threshold):
                self.platesolve_anisotropy_threshold = 1.3
        except (TypeError, ValueError):
            self.platesolve_anisotropy_threshold = 1.3
        self.platesolve_anisotropy_threshold = max(1.01, min(5.0, float(self.platesolve_anisotropy_threshold)))

        def _f01(key: str, default: float, lo: float, hi: float) -> None:
            try:
                v = float(data.get(key, getattr(self, key)))
                if not math.isfinite(v):
                    raise ValueError
                setattr(self, key, max(lo, min(hi, v)))
            except (TypeError, ValueError, AttributeError):
                setattr(self, key, float(default))

        def _i01(key: str, default: int, lo: int, hi: int) -> None:
            try:
                v = int(data.get(key, getattr(self, key)))
                setattr(self, key, max(lo, min(hi, v)))
            except (TypeError, ValueError, AttributeError):
                setattr(self, key, int(default))

        _f01("phase01_comparison_max_dist_deg", 1.0, 0.05, 10.0)
        _f01("phase01_comparison_max_mag_diff", 0.25, 0.05, 5.0)
        _f01("phase01_comparison_mag_bright_threshold", 12.0, 6.0, 18.0)
        _f01("phase01_comparison_max_mag_diff_bright_floor", 1.25, 0.0, 4.0)
        _f01("phase01_comparison_max_bv_diff", 0.15, 0.02, 3.0)
        _i01("phase01_comparison_n_comp_min", 3, 2, 12)
        _i01("phase01_comparison_n_comp_max", 7, 3, 20)
        if int(self.phase01_comparison_n_comp_max) < int(self.phase01_comparison_n_comp_min):
            self.phase01_comparison_n_comp_max = int(self.phase01_comparison_n_comp_min)
        _f01("phase01_comparison_max_comp_rms", 0.05, 0.01, 0.5)
        _f01("phase01_comparison_min_dist_arcsec", 60.0, 0.0, 600.0)
        _f01("phase01_comparison_min_frames_frac", 0.3, 0.05, 0.95)
        self.phase01_comparison_exclude_gaia_nss = bool(
            data.get("phase01_comparison_exclude_gaia_nss", self.phase01_comparison_exclude_gaia_nss)
        )
        self.phase01_comparison_exclude_gaia_extobj = bool(
            data.get("phase01_comparison_exclude_gaia_extobj", self.phase01_comparison_exclude_gaia_extobj)
        )

        _chip_m = data.get("phase01_chip_interior_margin_px")
        if _chip_m is None and "phase01_suspected_interior_margin_px" in data:
            _chip_m = data.get("phase01_suspected_interior_margin_px")
        if _chip_m is not None and _chip_m != "":
            try:
                self.phase01_chip_interior_margin_px = max(0, min(2000, int(_chip_m)))
            except (TypeError, ValueError):
                self.phase01_chip_interior_margin_px = 100

        # Variability Detection
        try:
            self.variability_min_frames = max(
                1, int(data.get("variability_min_frames", self.variability_min_frames))
            )
        except (TypeError, ValueError):
            self.variability_min_frames = 30
        try:
            self.variability_min_frames_frac = float(
                data.get("variability_min_frames_frac", self.variability_min_frames_frac)
            )
        except (TypeError, ValueError):
            self.variability_min_frames_frac = 0.50
        self.variability_min_frames_frac = max(0.05, min(0.99, float(self.variability_min_frames_frac)))

        def _vfloat(key: str, default: float, lo: float, hi: float) -> None:
            try:
                v = float(data.get(key, getattr(self, key)))
                if not math.isfinite(v):
                    raise ValueError
                setattr(self, key, max(lo, min(hi, v)))
            except (TypeError, ValueError, AttributeError):
                setattr(self, key, float(default))

        def _vint(key: str, default: int, lo: int, hi: int) -> None:
            try:
                v = int(data.get(key, getattr(self, key)))
                setattr(self, key, max(lo, min(hi, v)))
            except (TypeError, ValueError, AttributeError):
                setattr(self, key, int(default))

        _vfloat("variability_sigma_clip", 5.0, 1.0, 20.0)
        _vint("variability_p85_filter", 85, 50, 99)
        _vfloat("variability_slope_floor", 0.02, 0.0, 1.0)
        _vfloat("variability_sigma_threshold", 3.0, 0.5, 20.0)
        _vfloat("variability_smoothness_max", 0.80, 0.05, 1.0)
        _vfloat("variability_mag_limit", 14.5, 0.0, 30.0)
        _vfloat("variability_min_rms_pct", 1.5, 0.0, 100.0)
        _vfloat("variability_min_amplitude_mag", 0.01, 0.0, 10.0)
        _vfloat("variability_clip_ratio_min", 0.80, 0.0, 1.0)
        _vfloat("variability_vdi_z_threshold", 3.0, 0.0, 50.0)
        _vint("variability_min_points_rms", 20, 5, 10_000)

    def to_json(self) -> dict[str, Any]:
        return {
            "archive_root": str(self.archive_root),
            "calibration_library_root": str(self.calibration_library_root),
            "database_path": str(self.database_path),
            "masterdark_validity_days": int(self.masterdark_validity_days),
            "masterflat_validity_days": int(self.masterflat_validity_days),
            "calibration_library_native_binning": (
                None
                if self.calibration_library_native_binning is None
                else int(self.calibration_library_native_binning)
            ),
            "gaia_db_path": str(self.gaia_db_path or ""),
            "blind_index_path": str(self.blind_index_path or ""),
            "vsx_local_db_path": str(self.vsx_local_db_path or ""),
            "vsx_variable_targets_mag_limit": float(self.vsx_variable_targets_mag_limit),
            "catalog_query_max_rows": int(self.catalog_query_max_rows),
            "per_frame_mp_reserve_ram_gb": float(self.per_frame_mp_reserve_ram_gb),
            "alignment_max_stars": int(self.alignment_max_stars),
            "alignment_detection_sigma": float(self.alignment_detection_sigma),
            "qc_dao_detection_sigma": float(self.qc_dao_detection_sigma),
            "sips_dao_fwhm_px": float(self.sips_dao_fwhm_px),
            "sips_dao_threshold_sigma": float(self.sips_dao_threshold_sigma),
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
            "save_lightcurve_png": bool(self.save_lightcurve_png),
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
            "platesolve_anisotropy_threshold": float(self.platesolve_anisotropy_threshold),
            "phase01_comparison_max_dist_deg": float(self.phase01_comparison_max_dist_deg),
            "phase01_comparison_max_mag_diff": float(self.phase01_comparison_max_mag_diff),
            "phase01_comparison_mag_bright_threshold": float(self.phase01_comparison_mag_bright_threshold),
            "phase01_comparison_max_mag_diff_bright_floor": float(
                self.phase01_comparison_max_mag_diff_bright_floor
            ),
            "phase01_comparison_max_bv_diff": float(self.phase01_comparison_max_bv_diff),
            "phase01_comparison_n_comp_min": int(self.phase01_comparison_n_comp_min),
            "phase01_comparison_n_comp_max": int(self.phase01_comparison_n_comp_max),
            "phase01_comparison_max_comp_rms": float(self.phase01_comparison_max_comp_rms),
            "phase01_comparison_min_dist_arcsec": float(self.phase01_comparison_min_dist_arcsec),
            "phase01_comparison_min_frames_frac": float(self.phase01_comparison_min_frames_frac),
            "phase01_comparison_exclude_gaia_nss": bool(self.phase01_comparison_exclude_gaia_nss),
            "phase01_comparison_exclude_gaia_extobj": bool(self.phase01_comparison_exclude_gaia_extobj),
            "phase01_chip_interior_margin_px": int(self.phase01_chip_interior_margin_px),
            "variability_min_frames": int(self.variability_min_frames),
            "variability_min_frames_frac": float(self.variability_min_frames_frac),
            "variability_sigma_clip": float(self.variability_sigma_clip),
            "variability_p85_filter": int(self.variability_p85_filter),
            "variability_slope_floor": float(self.variability_slope_floor),
            "variability_sigma_threshold": float(self.variability_sigma_threshold),
            "variability_smoothness_max": float(self.variability_smoothness_max),
            "variability_mag_limit": float(self.variability_mag_limit),
            "variability_min_rms_pct": float(self.variability_min_rms_pct),
            "variability_min_amplitude_mag": float(self.variability_min_amplitude_mag),
            "variability_clip_ratio_min": float(self.variability_clip_ratio_min),
            "variability_vdi_z_threshold": float(self.variability_vdi_z_threshold),
            "variability_min_points_rms": int(self.variability_min_points_rms),
        }

    # Backward-compatible alias (some callers expect to_dict()).
    def to_dict(self) -> dict[str, Any]:
        return self.to_json()

    def ensure_base_dirs(self) -> None:
        """Create base directories required by file-first workflow."""
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.calibration_library_root.mkdir(parents=True, exist_ok=True)

