"""VYVAR Blind Plate Solver — Triangle Hash Matching.

Nájde aproximatívne RA/Dec stredu snímky bez akéhokoľvek hintu
z FITS hlavičky. Používa predgenerovaný index trojuholníkov z Gaia DR3
(``gaia_triangles_fine.pkl`` / ``gaia_triangles_wide.pkl``, generované skriptom ``GAIA_DR3/build_blind_index.py``).

3D hash (L1/L3, L2/L3, normalizovaný log10 L3 v ″) + hlasovanie podľa centroidu;
metadata môže obsahovať aj RA/Dec vrcholov (8 stĺpcov) pre rozšírenia / diagnostiku.

Výstup: (ra_deg, dec_deg) alebo None ak sa zhoda nenašla.
"""

from __future__ import annotations

import itertools
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy.spatial import KDTree, cKDTree

from infolog import log_event
from config import AppConfig

LOGGER = logging.getLogger(__name__)

HASH_QUERY_K = 8  # index hash-tree matches per image triangle (separate from triangle kNN)
_GNOMONIC_FOV_DEG_THRESHOLD = 2.0


def _blind_img_star_budget(app_config: Any | None, n_top: int) -> int:
    cfg = app_config or AppConfig()
    try:
        budget = int(getattr(cfg, "blind_img_star_budget", 80))
    except (TypeError, ValueError):
        budget = 80
    if budget <= 0:
        budget = int(n_top)
    return max(3, budget)


def _index_k_neighbors(idx: dict) -> int:
    try:
        k = int(idx.get("k_neighbors", 8))
    except (TypeError, ValueError):
        k = 8
    return max(3, min(32, k))


def _blind_img_select_mode(app_config: Any | None) -> str:
    mode = str(getattr(app_config or AppConfig(), "blind_img_select_mode", "per_cell")).strip().lower()
    if mode in ("central", "legacy", "rig_prior"):
        return "central"
    return "per_cell"


def _index_per_cell_meta(idx: dict) -> tuple[float, int]:
    try:
        cell_deg = float(idx.get("cell_deg", 1.0))
    except (TypeError, ValueError):
        cell_deg = 1.0
    if not math.isfinite(cell_deg) or cell_deg <= 0:
        cell_deg = 1.0
    try:
        spc = int(idx.get("stars_per_cell", 80))
    except (TypeError, ValueError):
        spc = 80
    return cell_deg, max(1, spc)


def _dao_flux_column(dao_stars: pd.DataFrame) -> str:
    if "flux" in dao_stars.columns:
        return "flux"
    if "peak" in dao_stars.columns:
        return "peak"
    return "flux"


def cap_brightest_per_pixel_cell(
    dao_stars: pd.DataFrame,
    *,
    cell_px: float,
    stars_per_cell: int,
) -> pd.DataFrame:
    """Brightest-N per square pixel cell (mirrors index per-cell cap on the image plane)."""
    if dao_stars.empty:
        return dao_stars.copy()
    cell = max(1.0, float(cell_px))
    n = max(1, int(stars_per_cell))
    flux_col = _dao_flux_column(dao_stars)
    work = dao_stars.copy()
    if flux_col not in work.columns:
        work[flux_col] = 1.0
    x = work["x"].to_numpy(dtype=np.float64)
    y = work["y"].to_numpy(dtype=np.float64)
    work["_xb"] = np.floor(x / cell).astype(np.int64)
    work["_yb"] = np.floor(y / cell).astype(np.int64)
    capped = (
        work.sort_values(flux_col, ascending=False)
        .groupby(["_xb", "_yb"], sort=False)
        .head(n)
        .drop(columns=["_xb", "_yb"])
    )
    return capped.reset_index(drop=True)


def _select_blind_image_stars(
    dao_stars: pd.DataFrame,
    idx: dict,
    *,
    plate_scale_arcsec_per_px: float,
    app_config: Any | None,
    n_top: int,
    fov_deg: float | None,
    use_rig_prior: bool,
    log_L3_max: float,
    tri_k: int,
) -> tuple[np.ndarray, str]:
    """Return (N,2) star coords and an INFO log line describing the selection."""
    _cfg = app_config or AppConfig()
    mode = _blind_img_select_mode(_cfg)
    cell_deg, spc = _index_per_cell_meta(idx)
    _ps = float(plate_scale_arcsec_per_px)
    cell_px = cell_deg * 3600.0 / _ps

    if mode == "per_cell":
        x_max = float(dao_stars["x"].max())
        y_max = float(dao_stars["y"].max())
        span = max(x_max, y_max, 1.0)
        if cell_px >= span:
            flux_col = _dao_flux_column(dao_stars)
            work = dao_stars.copy()
            if flux_col not in work.columns:
                work[flux_col] = 1.0
            picked = work.sort_values(flux_col, ascending=False).head(spc)
            n_cells = 1
        else:
            picked = cap_brightest_per_pixel_cell(
                dao_stars, cell_px=cell_px, stars_per_cell=spc
            )
            xb = np.floor(picked["x"].to_numpy(dtype=np.float64) / cell_px).astype(np.int64)
            yb = np.floor(picked["y"].to_numpy(dtype=np.float64) / cell_px).astype(np.int64)
            n_cells = len({(int(a), int(b)) for a, b in zip(xb, yb, strict=True)})
        stars = picked[["x", "y"]].to_numpy(dtype=np.float64)
        return (
            stars,
            f"INFO: Blind solver: per-cell image pick — {len(stars)} hviezd z {n_cells} buniek "
            f"(cell={cell_deg}° → {cell_px:.0f}px, SPC={spc}, kNN k={tri_k})",
        )

    x_max = float(dao_stars["x"].max())
    y_max = float(dao_stars["y"].max())
    x_cen = x_max / 2.0
    y_cen = y_max / 2.0
    budget = _blind_img_star_budget(_cfg, n_top)
    _fov_use: float | None = None
    if fov_deg is not None:
        try:
            _fv = float(fov_deg)
            if math.isfinite(_fv) and _fv > 0:
                _fov_use = _fv
        except (TypeError, ValueError):
            pass
    if use_rig_prior and _fov_use is not None:
        L3_max_arcsec = _fov_use * 3600.0 * 0.9
        R_px = (_fov_use * 0.5 * 0.95) * 3600.0 / _ps
    else:
        L3_max_arcsec = 10 ** float(log_L3_max)
        R_px = (L3_max_arcsec / _ps) / 2.0
    dist_from_center = np.sqrt((dao_stars["x"] - x_cen) ** 2 + (dao_stars["y"] - y_cen) ** 2)
    central_stars = dao_stars[dist_from_center <= R_px]
    if len(central_stars) >= 6:
        stars = central_stars.head(budget)[["x", "y"]].to_numpy(dtype=np.float64)
        return (
            stars,
            f"INFO: Blind solver: {len(central_stars)} hviezd v R={R_px:.0f}px od stredu, "
            f"použitých {len(stars)} (budget={budget}, kNN k={tri_k})",
        )
    stars = dao_stars.head(budget)[["x", "y"]].to_numpy(dtype=np.float64)
    return (
        stars,
        f"INFO: Blind solver: fallback {len(stars)} hviezd (budget={budget}), kNN k={tri_k}",
    )


def _rig_prior_enabled(cfg: Any | None) -> bool:
    return bool(getattr(cfg or AppConfig(), "blind_use_rig_prior", True))


def _scale_tol_frac(cfg: Any | None) -> float:
    try:
        v = float(getattr(cfg or AppConfig(), "blind_scale_tol_frac", 0.10))
    except (TypeError, ValueError):
        v = 0.10
    return max(0.02, min(0.50, v))


def _use_gnomonic_triangles(fov_deg: float | None, *, use_rig_prior: bool) -> bool:
    if not use_rig_prior or fov_deg is None:
        return False
    try:
        fov = float(fov_deg)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fov) and fov >= _GNOMONIC_FOV_DEG_THRESHOLD


def _side_arcsec_flat(
    p0: np.ndarray, p1: np.ndarray, *, plate_scale_arcsec_per_px: float
) -> float:
    return math.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1])) * plate_scale_arcsec_per_px


def _side_arcsec_gnomonic(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    x_cen: float,
    y_cen: float,
    plate_scale_arcsec_per_px: float,
) -> float:
    """Tangent-plane (gnomonic at field center) side length in arcsec."""
    u0 = (float(p0[0]) - x_cen) * plate_scale_arcsec_per_px
    v0 = (float(p0[1]) - y_cen) * plate_scale_arcsec_per_px
    u1 = (float(p1[0]) - x_cen) * plate_scale_arcsec_per_px
    v1 = (float(p1[1]) - y_cen) * plate_scale_arcsec_per_px
    return math.hypot(u1 - u0, v1 - v0)


def _triangle_sides_arcsec(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    x_cen: float,
    y_cen: float,
    plate_scale_arcsec_per_px: float,
    use_gnomonic: bool,
) -> tuple[float, float, float]:
    if use_gnomonic:
        fn = lambda a, b: _side_arcsec_gnomonic(  # noqa: E731
            a, b, x_cen=x_cen, y_cen=y_cen, plate_scale_arcsec_per_px=plate_scale_arcsec_per_px
        )
    else:
        fn = lambda a, b: _side_arcsec_flat(a, b, plate_scale_arcsec_per_px=plate_scale_arcsec_per_px)  # noqa: E731
    sides = sorted([fn(p0, p1), fn(p1, p2), fn(p0, p2)])
    return float(sides[0]), float(sides[1]), float(sides[2])


def _catalog_l3_arcsec_from_tree(
    match_idx: int,
    *,
    hash_tree: KDTree,
    log_L3_min: float,
    log_L3_max: float,
) -> float:
    tree_data = getattr(hash_tree, "data", None)
    if tree_data is None or match_idx < 0 or match_idx >= len(tree_data):
        return float("nan")
    log_l3_norm = float(tree_data[int(match_idx), 2])
    log_l3 = log_l3_norm * max(log_L3_max - log_L3_min, 1e-6) + log_L3_min
    return float(10.0 ** log_l3)


def _scale_ratio_accepts(
    l3_img_arcsec: float,
    l3_cat_arcsec: float,
    *,
    scale_tol_frac: float,
) -> bool:
    if not (math.isfinite(l3_img_arcsec) and math.isfinite(l3_cat_arcsec)):
        return False
    if l3_cat_arcsec < 1e-6:
        return False
    ratio = l3_img_arcsec / l3_cat_arcsec
    return abs(ratio - 1.0) <= scale_tol_frac


def _knn_search_coords_from_pixels(
    stars: np.ndarray,
    *,
    x_cen: float,
    y_cen: float,
    plate_scale_arcsec_per_px: float,
    use_sphere_knn: bool,
) -> np.ndarray:
    """Coords for local kNN (unit-sphere gnomonic at field center when wide; else pixels)."""
    if not use_sphere_knn:
        return np.asarray(stars, dtype=np.float64)
    ps = float(plate_scale_arcsec_per_px)
    u_arcsec = (stars[:, 0] - float(x_cen)) * ps
    v_arcsec = (stars[:, 1] - float(y_cen)) * ps
    ang = math.pi / 180.0 / 3600.0
    nx = u_arcsec * ang
    ny = v_arcsec * ang
    nz = np.sqrt(np.maximum(1.0 - nx * nx - ny * ny, 1e-12))
    return np.column_stack([nx, ny, nz]).astype(np.float64)


def iter_local_knn_triangle_indices(
    stars: np.ndarray,
    *,
    k_neighbors: int,
    search_coords: np.ndarray | None = None,
) -> Iterator[tuple[int, int, int]]:
    """Local kNN triangles with dedup ``i == min(i_tri)`` (mirrors index builder)."""
    n = len(stars)
    if n < 3:
        return
    k = min(int(k_neighbors), n)
    if k < 3:
        return
    coords = np.asarray(search_coords if search_coords is not None else stars, dtype=np.float64)
    tree = KDTree(coords)
    _, all_idx = tree.query(coords, k=k)
    all_idx = np.atleast_2d(all_idx)
    combos = list(itertools.combinations(range(k), 3))
    for i in range(n):
        neighbor_idx = all_idx[i]
        for c in combos:
            i_tri = neighbor_idx[list(c)]
            if i != int(np.min(i_tri)):
                continue
            yield int(i_tri[0]), int(i_tri[1]), int(i_tri[2])

_CACHED_INDEX: dict = {}  # module-level cache: path → {tree, metadata}

CLUSTER_RADIUS_DEG = 1.0  # default DBSCAN eps / legacy greedy cluster radius
_CAND_DEDUP_DEG = 0.3


def _sky_cell_vote_winner(
    hits: list[_MatchHit],
    *,
    cell_deg: float,
) -> tuple[float, float, int] | None:
    """Pick RA/Dec from index-style sky cell; rank by vote count then mean hash quality."""
    if not hits:
        return None
    cell = float(cell_deg)
    ra = np.array([h.center_ra for h in hits], dtype=np.float64)
    dec = np.array([h.center_dec for h in hits], dtype=np.float64)
    hd = np.array([h.hash_dist for h in hits], dtype=np.float64)
    dec_bin = np.floor((dec + 90.0) / cell).astype(np.int64)
    cos_dec = np.cos(np.radians(dec))
    ra_width = cell / np.maximum(cos_dec, 0.1)
    ra_bin = np.floor(ra / ra_width).astype(np.int64)
    keys = dec_bin.astype(np.int64) * 10_000_000 + ra_bin.astype(np.int64)
    best_key: int | None = None
    best_score = -1.0
    best_n = 0
    for key in np.unique(keys):
        mask = keys == key
        n = int(np.count_nonzero(mask))
        if n < 2:
            continue
        mean_hd = float(hd[mask].mean())
        score = float(n) / (1.0 + mean_hd)
        if score > best_score:
            best_score = score
            best_n = n
            best_key = int(key)
    if best_key is None:
        return None
    mask = keys == best_key
    return (
        float(np.median(ra[mask])),
        float(np.median(dec[mask])),
        best_n,
    )


@dataclass
class BlindCandidate:
    """Hypothesis from triangle-hash voting + vertex correspondence."""

    center_ra: float
    center_dec: float
    img_px: np.ndarray
    cat_sky: np.ndarray
    hash_dist: float
    vote_count: int
    #: DBSCAN cluster members for pooled RANSAC verify (list of ``_MatchHit``).
    cluster_members: list[Any] | None = None


@dataclass
class _MatchHit:
    center_ra: float
    center_dec: float
    img_px: np.ndarray
    cat_sky: np.ndarray | None
    hash_dist: float


@dataclass
class _BlindPassResult:
    dub: float
    is_first_pass: bool
    n_tried: int
    n_passed: int
    n_below_min: int
    n_above_max: int
    n_in_range: int
    logl3_samples: list[float]
    match_mult_mean: float
    vote_centers: list[np.ndarray]
    match_hits: list[_MatchHit]


def _load_index(index_path: str | Path) -> dict | None:
    """Načíta PKL index do module-level cache (načíta sa len raz za beh)."""
    key = str(Path(index_path).resolve())
    if key in _CACHED_INDEX:
        return _CACHED_INDEX[key]
    try:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        if "tree" not in data or "metadata" not in data:
            log_event("WARNING: Blind index: neplatný formát PKL (chýba tree alebo metadata).")
            return None
        _CACHED_INDEX[key] = data
        log_event(f"INFO: Blind index načítaný: {len(data['metadata'])} trojuholníkov")
        return data
    except Exception as exc:  # noqa: BLE001
        # EXC-0589: T4 -- blind index load fail already surfaced as WARNING; blind tier degrades loudly downstream (EXCEPT-BULK 2026-07-08)
        log_event(f"WARNING: Blind index: načítanie zlyhalo: {exc}")
        return None


def _prepare_blind_context(
    dao_stars: pd.DataFrame,
    index_path: str | Path,
    *,
    n_top: int,
    plate_scale_arcsec_per_px: float | None,
    fov_deg: float | None = None,
    app_config: Any | None = None,
    debug_sink: dict | None = None,
) -> (
    tuple[
        dict,
        np.ndarray,
        bool,
        float,
        float,
        float,
        float,
        list[float],
        int,
        bool,
        bool,
        float,
        float,
        bool,
        bool,
        float,
    ]
    | None
):
    """Shared setup for blind solver passes. Returns None on fatal error."""
    idx = _load_index(index_path)
    if idx is None:
        return None

    hash_dim = int(idx.get("hash_dim", 2))
    if hash_dim == 4:
        log_event(
            "WARNING: Blind solver: stary 4D index — vygeneruj znova (build_blind_index.py)."
        )
        return None
    if hash_dim == 2:
        log_event(
            "WARNING: Blind solver: stary 2D index bez mierky — vygeneruj znova (build_blind_index.py)."
        )
        return None
    if hash_dim != 3:
        log_event(
            f"WARNING: Blind solver: nepodporovaný hash_dim={hash_dim} — znova vygeneruj index."
        )
        return None

    hash_tree: KDTree = idx["tree"]
    tree_data = getattr(hash_tree, "data", None)
    _tree_cols = (
        int(tree_data.shape[1])
        if tree_data is not None and getattr(tree_data, "ndim", 0) == 2
        else 0
    )
    if _tree_cols != 3:
        log_event(
            "WARNING: Blind solver: očakávaný 3D hash-tree; "
            f"strom má {_tree_cols}D — znova vygeneruj index."
        )
        return None

    try:
        log_L3_min = float(idx["log_L3_min"])
        log_L3_max = float(idx["log_L3_max"])
    except (KeyError, TypeError, ValueError):
        log_event("WARNING: Blind solver: v PKL chýba log_L3_min / log_L3_max — znova vygeneruj index.")
        return None

    _cfg = app_config or AppConfig()
    _dbg = bool(getattr(_cfg, "debug_platesolver", False))
    _collect_diag = debug_sink is not None
    if _collect_diag:
        debug_sink.clear()
        debug_sink.setdefault("passes", [])

    if plate_scale_arcsec_per_px is None:
        log_event(
            "WARNING: Blind solver: 3D index vyžaduje plate_scale_arcsec_per_px (mierku) — hint sa nepočíta."
        )
        return None
    try:
        _ps = float(plate_scale_arcsec_per_px)
    except (TypeError, ValueError):
        log_event("WARNING: Blind solver: neplatná mierka (plate_scale_arcsec_per_px).")
        return None
    if not math.isfinite(_ps) or _ps <= 0:
        log_event("WARNING: Blind solver: neplatná mierka (plate_scale_arcsec_per_px).")
        return None

    metadata: np.ndarray = idx["metadata"]
    has_vertices = metadata.ndim == 2 and metadata.shape[1] == 8

    try:
        _idx_tol = float(idx.get("tolerance", 0.02))
    except (TypeError, ValueError):
        _idx_tol = 0.02
    if not math.isfinite(_idx_tol) or _idx_tol <= 0:
        _idx_tol = 0.02

    dub_candidates = sorted(
        {
            float(d)
            for d in (
                min(_idx_tol, 0.002),
                min(_idx_tol, 0.003),
                min(_idx_tol, 0.005),
                min(_idx_tol, 0.01),
                _idx_tol,
            )
            if math.isfinite(d) and d > 0
        }
    )
    if not dub_candidates:
        dub_candidates = [0.002]

    if _dbg:
        log_event(f"DEBUG: Index log_L3 rozsah: min={log_L3_min:.3f} max={log_L3_max:.3f}")

    log_event(
        f"INFO: Blind index: {len(metadata)} trojuholníkov, 3D hash (normalizovaný log L3), "
        f"vertices={'áno' if has_vertices else 'nie (legacy)'}"
    )

    if dao_stars.empty or not {"x", "y"}.issubset(dao_stars.columns):
        return None

    x_max = float(dao_stars["x"].max())
    y_max = float(dao_stars["y"].max())
    x_cen = x_max / 2.0
    y_cen = y_max / 2.0
    use_rig_prior = _rig_prior_enabled(_cfg)
    scale_tol = _scale_tol_frac(_cfg)
    use_gnomonic = _use_gnomonic_triangles(fov_deg, use_rig_prior=use_rig_prior)
    tri_k = _index_k_neighbors(idx)
    stars, pick_msg = _select_blind_image_stars(
        dao_stars,
        idx,
        plate_scale_arcsec_per_px=_ps,
        app_config=_cfg,
        n_top=n_top,
        fov_deg=fov_deg,
        use_rig_prior=use_rig_prior,
        log_L3_max=log_L3_max,
        tri_k=tri_k,
    )
    log_event(pick_msg)

    if len(stars) < 3:
        log_event(f"WARNING: Blind solver: príliš málo hviezd ({len(stars)} < 3).")
        return None

    return (
        idx,
        stars,
        has_vertices,
        log_L3_min,
        log_L3_max,
        _ps,
        float(dub_candidates[0]),
        dub_candidates,
        tri_k,
        HASH_QUERY_K,
        _dbg,
        _collect_diag,
        x_cen,
        y_cen,
        use_gnomonic,
        use_rig_prior,
        scale_tol,
    )


def _iter_blind_pass_results(
    *,
    stars: np.ndarray,
    hash_tree: KDTree,
    metadata: np.ndarray,
    has_vertices: bool,
    log_L3_min: float,
    log_L3_max: float,
    plate_scale: float,
    fov_deg: float | None,
    dub_candidates: list[float],
    first_dub: float,
    tri_k_neighbors: int,
    hash_query_k: int,
    collect_diag: bool,
    collect_logl3: bool,
    x_cen: float,
    y_cen: float,
    use_gnomonic: bool,
    use_rig_prior: bool,
    scale_tol: float,
) -> Iterator[_BlindPassResult]:
    log_L3_range = max(log_L3_max - log_L3_min, 1e-6)

    for dub in dub_candidates:
        is_first = float(dub) == float(first_dub)
        vote_centers: list[np.ndarray] = []
        match_hits: list[_MatchHit] = []
        n_tried = 0
        n_passed = 0
        n_below_min = 0
        n_above_max = 0
        logl3_samples: list[float] = []
        n_queries = 0
        n_finite_matches = 0

        knn_coords = _knn_search_coords_from_pixels(
            stars,
            x_cen=x_cen,
            y_cen=y_cen,
            plate_scale_arcsec_per_px=plate_scale,
            use_sphere_knn=use_gnomonic,
        )
        for i0, i1, i2 in iter_local_knn_triangle_indices(
            stars, k_neighbors=tri_k_neighbors, search_coords=knn_coords
        ):
            n_tried += 1
            p0, p1, p2 = stars[i0], stars[i1], stars[i2]
            L1, L2, L3 = _triangle_sides_arcsec(
                p0,
                p1,
                p2,
                x_cen=x_cen,
                y_cen=y_cen,
                plate_scale_arcsec_per_px=plate_scale,
                use_gnomonic=use_gnomonic,
            )
            if max(0.1, 2.0 * plate_scale) > L3 or L1 / L3 < 0.15:
                continue

            r1, r2 = L1 / L3, L2 / L3
            L3_arcsec = L3
            if L3_arcsec < 0.1:
                continue
            if fov_deg is not None:
                try:
                    _fov = float(fov_deg)
                except (TypeError, ValueError):
                    _fov = 0.0
                if math.isfinite(_fov) and _fov > 0 and L3_arcsec > _fov * 3600.0 * 0.9:
                    continue

            n_passed += 1
            log_L3 = math.log10(L3_arcsec)
            log_L3_norm = (log_L3 - log_L3_min) / log_L3_range
            if collect_logl3:
                logl3_samples.append(log_L3)
            if log_L3_norm < 0.0:
                if collect_logl3:
                    n_below_min += 1
                continue
            if log_L3_norm > 1.0:
                if collect_logl3:
                    n_above_max += 1
                continue

            dists, match_idxs = hash_tree.query(
                [float(r1), float(r2), float(log_L3_norm)],
                k=hash_query_k,
                distance_upper_bound=dub,
            )
            if collect_diag:
                n_queries += 1

            dists_arr = np.atleast_1d(dists)
            idxs_arr = np.atleast_1d(match_idxs)
            img_tri = np.asarray([p0, p1, p2], dtype=np.float64)
            for dist, match_idx in zip(dists_arr, idxs_arr, strict=False):
                if not np.isfinite(dist) or float(dist) >= dub:
                    continue
                if collect_diag:
                    n_finite_matches += 1
                mi = int(match_idx)
                if not (0 <= mi < len(metadata)):
                    continue
                if use_rig_prior:
                    l3_cat = _catalog_l3_arcsec_from_tree(
                        mi,
                        hash_tree=hash_tree,
                        log_L3_min=log_L3_min,
                        log_L3_max=log_L3_max,
                    )
                    if not _scale_ratio_accepts(
                        L3_arcsec, l3_cat, scale_tol_frac=scale_tol
                    ):
                        continue
                center = np.asarray(metadata[mi, :2], dtype=np.float64)
                vote_centers.append(center)
                cat_sky = None
                if has_vertices:
                    row = metadata[mi]
                    cat_sky = np.asarray(
                        [
                            [float(row[2]), float(row[3])],
                            [float(row[4]), float(row[5])],
                            [float(row[6]), float(row[7])],
                        ],
                        dtype=np.float64,
                    )
                match_hits.append(
                    _MatchHit(
                        center_ra=float(center[0]),
                        center_dec=float(center[1]),
                        img_px=img_tri.copy(),
                        cat_sky=cat_sky,
                        hash_dist=float(dist),
                    )
                )

        n_in_range = n_passed - n_below_min - n_above_max
        yield _BlindPassResult(
            dub=float(dub),
            is_first_pass=is_first,
            n_tried=n_tried,
            n_passed=n_passed,
            n_below_min=n_below_min,
            n_above_max=n_above_max,
            n_in_range=n_in_range,
            logl3_samples=logl3_samples,
            match_mult_mean=n_finite_matches / max(n_queries, 1),
            vote_centers=vote_centers,
            match_hits=match_hits,
        )


def _cluster_centroid_votes(
    vote_centers: list[np.ndarray],
    *,
    radius_deg: float = CLUSTER_RADIUS_DEG,
) -> list[tuple[float, float, int]]:
    if len(vote_centers) < 2:
        return []
    votes_arr = np.array(vote_centers, dtype=np.float64)
    r_deg = float(radius_deg)
    clusters: list[tuple[float, float, int]] = []
    used = np.zeros(len(votes_arr), dtype=bool)
    for i in range(len(votes_arr)):
        if used[i]:
            continue
        ra_i, dec_i = float(votes_arr[i, 0]), float(votes_arr[i, 1])
        dra = (votes_arr[:, 0] - ra_i) * math.cos(math.radians(dec_i))
        ddec = votes_arr[:, 1] - dec_i
        sep = np.sqrt(dra * dra + ddec * ddec)
        in_cluster = sep < r_deg
        used |= in_cluster
        count = int(np.count_nonzero(in_cluster))
        clusters.append(
            (
                float(np.median(votes_arr[in_cluster, 0])),
                float(np.median(votes_arr[in_cluster, 1])),
                count,
            )
        )
    clusters.sort(key=lambda t: t[2], reverse=True)
    return clusters


def _cluster_match_hits_weighted(
    hits: list[_MatchHit],
    *,
    radius_deg: float,
) -> list[tuple[float, float, float, int]]:
    """Cluster by sky position; rank by sum 1/(1+hash_dist)."""
    if len(hits) < 2:
        return []
    r_deg = float(radius_deg)
    pos = np.array([[h.center_ra, h.center_dec] for h in hits], dtype=np.float64)
    wt = np.array([1.0 / (1.0 + h.hash_dist) for h in hits], dtype=np.float64)
    clusters: list[tuple[float, float, float, int]] = []
    used = np.zeros(len(pos), dtype=bool)
    for i in range(len(pos)):
        if used[i]:
            continue
        ra_i, dec_i = float(pos[i, 0]), float(pos[i, 1])
        dra = (pos[:, 0] - ra_i) * math.cos(math.radians(dec_i))
        ddec = pos[:, 1] - dec_i
        sep = np.sqrt(dra * dra + ddec * ddec)
        in_cluster = sep < r_deg
        used |= in_cluster
        w_sum = float(wt[in_cluster].sum())
        count = int(np.count_nonzero(in_cluster))
        clusters.append(
            (
                float(np.median(pos[in_cluster, 0])),
                float(np.median(pos[in_cluster, 1])),
                w_sum,
                count,
            )
        )
    clusters.sort(key=lambda t: t[2], reverse=True)
    return clusters


def _pick_cluster_representative(
    cluster_center: tuple[float, float],
    hits: list[_MatchHit],
    *,
    radius_deg: float = CLUSTER_RADIUS_DEG,
) -> _MatchHit | None:
    cra, cdec = cluster_center
    r_deg = float(radius_deg)
    in_cluster = []
    for h in hits:
        if h.cat_sky is None:
            continue
        dra = (h.center_ra - cra) * math.cos(math.radians(cdec))
        ddec = h.center_dec - cdec
        if math.hypot(dra, ddec) < r_deg:
            in_cluster.append(h)
    if not in_cluster:
        return None
    return min(in_cluster, key=lambda h: h.hash_dist)


def _blind_cluster_params(app_config: Any | None) -> tuple[float, int, int, int, int]:
    cfg = app_config or AppConfig()
    try:
        eps = float(getattr(cfg, "blind_cluster_eps_deg", CLUSTER_RADIUS_DEG))
    except (TypeError, ValueError):
        eps = CLUSTER_RADIUS_DEG
    if not math.isfinite(eps) or eps <= 0:
        eps = CLUSTER_RADIUS_DEG
    try:
        min_votes = int(getattr(cfg, "blind_cluster_min_votes", 4))
    except (TypeError, ValueError):
        min_votes = 4
    try:
        min_samples = int(getattr(cfg, "blind_cluster_min_samples", 3))
    except (TypeError, ValueError):
        min_samples = 3
    try:
        vote_span = int(getattr(cfg, "blind_cluster_vote_span", 12))
    except (TypeError, ValueError):
        vote_span = 12
    try:
        coh_cap = int(getattr(cfg, "blind_cluster_coherence_cap", 50))
    except (TypeError, ValueError):
        coh_cap = 50
    return eps, max(2, min_votes), max(2, min_samples), max(0, vote_span), max(5, coh_cap)


def _hits_unit_sphere_xyz(hits: list[_MatchHit]) -> np.ndarray:
    ra = np.array([h.center_ra for h in hits], dtype=np.float64)
    dec = np.array([h.center_dec for h in hits], dtype=np.float64)
    rr = np.radians(ra)
    dd = np.radians(dec)
    return np.column_stack(
        [np.cos(dd) * np.cos(rr), np.cos(dd) * np.sin(rr), np.sin(dd)]
    ).astype(np.float64)


def _dbscan_vote_labels(
    hits: list[_MatchHit],
    *,
    eps_deg: float,
    min_samples: int,
) -> np.ndarray:
    """DBSCAN on vote centers (unit-sphere chord ≈ haversine). Returns label per hit (-1 = noise)."""
    n = len(hits)
    labels = np.full(n, -1, dtype=np.int64)
    if n < min_samples:
        return labels
    xyz = _hits_unit_sphere_xyz(hits)
    eps_rad = math.radians(float(eps_deg))
    chord_r = 2.0 * math.sin(max(eps_rad, 1e-12) / 2.0)
    tree = cKDTree(xyz)
    neighbors = tree.query_ball_point(xyz, r=chord_r)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    for seed in range(n):
        if visited[seed] or len(neighbors[seed]) < min_samples:
            continue
        visited[seed] = True
        frontier = list(neighbors[seed])
        cluster_members = [seed]
        while frontier:
            q = frontier.pop()
            if visited[q]:
                continue
            visited[q] = True
            cluster_members.append(q)
            if len(neighbors[q]) >= min_samples:
                for nb in neighbors[q]:
                    if not visited[nb]:
                        frontier.append(nb)
        if len(cluster_members) < min_samples:
            for m in cluster_members:
                visited[m] = False
            continue
        for m in cluster_members:
            labels[m] = cluster_id
        cluster_id += 1
    return labels



def _hits_to_candidates_dbscan(
    hits: list[_MatchHit],
    *,
    app_config: Any | None,
    top_n: int,
) -> list[BlindCandidate]:
    """Coherent vote clusters → verify candidates (rep = best hash_dist per cluster)."""
    verifiable = [h for h in hits if h.cat_sky is not None]
    if not verifiable:
        return []
    eps_deg, min_votes, min_samples, vote_span, coh_cap = _blind_cluster_params(app_config)
    labels = _dbscan_vote_labels(verifiable, eps_deg=eps_deg, min_samples=min_samples)
    order = np.argsort([h.hash_dist for h in verifiable])
    best_rank: dict[int, int] = {}
    for rank, i in enumerate(order):
        lab = int(labels[i])
        if lab >= 0 and lab not in best_rank:
            best_rank[lab] = rank
    lab_members: dict[int, list[_MatchHit]] = {}
    for i, lab in enumerate(labels):
        li = int(lab)
        if li >= 0:
            lab_members.setdefault(li, []).append(verifiable[i])
    span_hi = min_votes + vote_span
    high_vote_floor = min(min_votes + 6, span_hi)
    verify_cap = max(int(coh_cap) + int(top_n), 100)
    cap_dom = max(1, int(top_n) // 3)
    coh_entries: list[tuple[int, float, float, int, list[_MatchHit]]] = []
    large_entries: list[tuple[int, float, list[_MatchHit]]] = []
    for lab, members in lab_members.items():
        n = len(members)
        if n < min_votes:
            continue
        med_hd = float(np.median([h.hash_dist for h in members]))
        br = best_rank.get(lab, 10**9)
        if n <= span_hi:
            tier = 0 if n >= high_vote_floor else 1
            pri = float(br) / max(float(n), 1.0)
            coh_entries.append((tier, pri, med_hd, n, members))
        else:
            large_entries.append((n, med_hd, members))
    coh_entries.sort(key=lambda t: (t[0], t[1], t[2]))
    selected: list[tuple[int, float, list[_MatchHit]]] = [
        (n, med_hd, members) for _tier, _pri, med_hd, n, members in coh_entries[:verify_cap]
    ]
    for n, med_hd, members in sorted(large_entries, key=lambda t: t[0], reverse=True):
        if len(selected) >= verify_cap + cap_dom:
            break
        entry = (n, med_hd, members)
        if entry not in selected:
            selected.append(entry)
    all_clusters = [(n, med_hd, members) for _t, _p, med_hd, n, members in coh_entries]
    all_clusters.extend(large_entries)
    out: list[BlindCandidate] = []

    def _add_cluster(members: list[_MatchHit]) -> None:
        rep = min(members, key=lambda h: h.hash_dist)
        if rep.cat_sky is None:
            return
        cand = BlindCandidate(
            center_ra=float(rep.center_ra),
            center_dec=float(rep.center_dec),
            img_px=rep.img_px.copy(),
            cat_sky=rep.cat_sky.copy(),
            hash_dist=float(rep.hash_dist),
            vote_count=len(members),
            cluster_members=list(members),
        )
        if not _candidate_near_existing(cand, out):
            out.append(cand)

    for _count, _med_hd, members in selected:
        _add_cluster(members)
    log_event(
        f"INFO: Blind DBSCAN: {len(all_clusters)} klastrov (>={min_votes}), "
        f"{len(out)} verify kandidátov "
        f"(eps={eps_deg:.2f}°, span<={span_hi}, verify_cap={verify_cap})"
    )
    out.sort(
        key=lambda c: (
            0 if int(c.vote_count) >= int(high_vote_floor) else 1,
            float(c.hash_dist),
            -int(c.vote_count),
        )
    )
    return out


def _candidate_near_existing(cand: BlindCandidate, existing: list[BlindCandidate]) -> bool:
    for e in existing:
        dra = (cand.center_ra - e.center_ra) * math.cos(math.radians(cand.center_dec))
        ddec = cand.center_dec - e.center_dec
        if math.hypot(dra, ddec) < _CAND_DEDUP_DEG:
            return True
    return False


def _hits_to_candidates_legacy(
    hits: list[_MatchHit],
    *,
    top_n: int,
    fov_deg: float | None = None,
    index_meta: dict | None = None,
    img_select_per_cell: bool = False,
) -> list[BlindCandidate]:
    verifiable = [h for h in hits if h.cat_sky is not None]
    if not verifiable:
        return []

    clusters = _cluster_match_hits_weighted(verifiable, radius_deg=CLUSTER_RADIUS_DEG)
    out: list[BlindCandidate] = []

    def _add(hit: _MatchHit, vote_count: int) -> None:
        cand = BlindCandidate(
            center_ra=hit.center_ra,
            center_dec=hit.center_dec,
            img_px=hit.img_px.copy(),
            cat_sky=hit.cat_sky.copy(),
            hash_dist=hit.hash_dist,
            vote_count=vote_count,
        )
        if not _candidate_near_existing(cand, out):
            out.append(cand)

    if img_select_per_cell and index_meta is not None:
        cell_deg, _ = _index_per_cell_meta(index_meta)
        sky_pick = _sky_cell_vote_winner(verifiable, cell_deg=cell_deg)
        if sky_pick is not None:
            rep = _pick_cluster_representative(
                (sky_pick[0], sky_pick[1]), verifiable, radius_deg=CLUSTER_RADIUS_DEG
            )
            if rep is not None:
                _add(rep, int(sky_pick[2]))

    by_hash = sorted(verifiable, key=lambda h: h.hash_dist)
    if by_hash:
        _add(by_hash[0], 1)
        q25 = float(np.percentile([h.hash_dist for h in by_hash], 25))
        for hit in by_hash:
            if hit.hash_dist <= q25:
                _add(hit, 1)
    for hit in by_hash[: max(top_n, 10)]:
        _add(hit, 1)

    for cra, cdec, _wsum, count in clusters[:top_n]:
        rep = _pick_cluster_representative((cra, cdec), verifiable, radius_deg=CLUSTER_RADIUS_DEG)
        if rep is not None:
            _add(rep, count)

    out.sort(key=lambda c: (-c.vote_count, c.hash_dist))
    return out[: max(top_n, top_n + 5)]


def find_blind_candidates(
    dao_stars: pd.DataFrame,
    index_path: str | Path,
    *,
    n_top: int = 30,
    top_n: int = 15,
    plate_scale_arcsec_per_px: float | None = None,
    fov_deg: float | None = None,
    app_config: Any | None = None,
    debug_truth_radec: tuple[float, float] | None = None,
    debug_sink: dict | None = None,
) -> list[BlindCandidate]:
    """Return top-N blind-solver hypotheses with vertex correspondence for verification."""
    ctx = _prepare_blind_context(
        dao_stars,
        index_path,
        n_top=n_top,
        plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
        fov_deg=fov_deg,
        app_config=app_config,
        debug_sink=debug_sink,
    )
    if ctx is None:
        return []

    (
        idx,
        stars,
        has_vertices,
        log_L3_min,
        log_L3_max,
        _ps,
        first_dub,
        dub_candidates,
        tri_k,
        hash_k,
        _dbg,
        _collect,
        x_cen,
        y_cen,
        use_gnomonic,
        use_rig_prior,
        scale_tol,
    ) = ctx
    _cfg = app_config or AppConfig()
    if not has_vertices:
        log_event("WARNING: Blind candidates: index bez vrcholov (legacy metadata) — vracia [].")
        return []

    hash_tree: KDTree = idx["tree"]
    metadata: np.ndarray = idx["metadata"]
    best_hits: list[_MatchHit] = []
    best_pass: _BlindPassResult | None = None

    for pres in _iter_blind_pass_results(
        stars=stars,
        hash_tree=hash_tree,
        metadata=metadata,
        has_vertices=has_vertices,
        log_L3_min=log_L3_min,
        log_L3_max=log_L3_max,
        plate_scale=_ps,
        fov_deg=fov_deg,
        dub_candidates=dub_candidates,
        first_dub=first_dub,
        tri_k_neighbors=tri_k,
        hash_query_k=hash_k,
        collect_diag=_collect or _dbg,
        collect_logl3=_collect or _dbg,
        x_cen=x_cen,
        y_cen=y_cen,
        use_gnomonic=use_gnomonic,
        use_rig_prior=use_rig_prior,
        scale_tol=scale_tol,
    ):
        log_event(
            f"INFO: Blind solver(pass dub={pres.dub:.4g}): {pres.n_passed}/{pres.n_tried} "
            f"trojuholníkov prešlo filter, {len(pres.vote_centers)} hlasov"
        )
        if _dbg and pres.logl3_samples:
            arr = np.asarray(pres.logl3_samples)
            log_event(
                f"DEBUG: Blind log_L3(img) n={len(arr)} "
                f"min={arr.min():.3f} p10={np.percentile(arr, 10):.3f} "
                f"med={np.median(arr):.3f} p90={np.percentile(arr, 90):.3f} max={arr.max():.3f} "
                f"| index[{log_L3_min:.3f},{log_L3_max:.3f}] "
                f"| below_min={pres.n_below_min} above_max={pres.n_above_max} "
                f"in_range={pres.n_in_range} "
                f"({100.0 * pres.n_in_range / max(pres.n_passed, 1):.1f}% in range)"
            )
        if len(pres.match_hits) > len(best_hits):
            best_hits = pres.match_hits
            best_pass = pres

    if bool(getattr(_cfg, "blind_verify_enabled", True)):
        candidates = _hits_to_candidates_dbscan(
            best_hits,
            app_config=_cfg,
            top_n=int(top_n),
        )
    else:
        candidates = _hits_to_candidates_legacy(
            best_hits,
            top_n=int(top_n),
            fov_deg=fov_deg,
            index_meta=idx,
            img_select_per_cell=_blind_img_select_mode(_cfg) == "per_cell",
        )
    if _collect and debug_sink is not None:
        debug_sink["candidates"] = [
            {
                "center_ra": c.center_ra,
                "center_dec": c.center_dec,
                "hash_dist": c.hash_dist,
                "vote_count": c.vote_count,
            }
            for c in candidates
        ]
        if best_pass is not None:
            debug_sink.setdefault("passes", []).append(
                {
                    "dub": best_pass.dub,
                    "n_votes": len(best_pass.vote_centers),
                    "match_mult_mean": best_pass.match_mult_mean,
                    "n_candidates": len(candidates),
                }
            )
        if debug_truth_radec is not None and best_hits:
            t_ra, t_dec = float(debug_truth_radec[0]), float(debug_truth_radec[1])
            centers = np.array([[h.center_ra, h.center_dec] for h in best_hits], dtype=np.float64)
            dra_t = (centers[:, 0] - t_ra) * np.cos(math.radians(t_dec))
            ddec_t = centers[:, 1] - t_dec
            sep_t = np.sqrt(dra_t**2 + ddec_t**2)
            debug_sink["votes_near_truth_2deg"] = int((sep_t < 2.0).sum())
            debug_sink["votes_near_truth_5deg"] = int((sep_t < 5.0).sum())
            if _dbg:
                log_event(
                    f"DEBUG: Blind votes near truth: <2°={debug_sink['votes_near_truth_2deg']} "
                    f"<5°={debug_sink['votes_near_truth_5deg']} / {len(centers)}"
                )
    log_event(f"INFO: Blind solver: {len(candidates)} kandidátov pre geometrickú verifikáciu.")
    return candidates


def find_blind_hint(
    dao_stars: pd.DataFrame,
    index_path: str | Path,
    *,
    n_top: int = 30,
    min_votes: int = 3,
    plate_scale_arcsec_per_px: float | None = None,
    fov_deg: float | None = None,
    app_config: Any | None = None,
    debug_truth_radec: tuple[float, float] | None = None,
    debug_sink: dict | None = None,
) -> tuple[float, float] | None:
    """Legacy vote-only blind hint (significance gate)."""
    ctx = _prepare_blind_context(
        dao_stars,
        index_path,
        n_top=n_top,
        plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
        fov_deg=fov_deg,
        app_config=app_config,
        debug_sink=debug_sink,
    )
    if ctx is None:
        return None

    (
        idx,
        stars,
        has_vertices,
        log_L3_min,
        log_L3_max,
        _ps,
        first_dub,
        dub_candidates,
        tri_k,
        hash_k,
        _dbg,
        _collect,
        x_cen,
        y_cen,
        use_gnomonic,
        use_rig_prior,
        scale_tol,
    ) = ctx
    _cfg = app_config or AppConfig()
    hash_tree: KDTree = idx["tree"]
    metadata: np.ndarray = idx["metadata"]
    _ = has_vertices

    for pres in _iter_blind_pass_results(
        stars=stars,
        hash_tree=hash_tree,
        metadata=metadata,
        has_vertices=metadata.ndim == 2 and metadata.shape[1] == 8,
        log_L3_min=log_L3_min,
        log_L3_max=log_L3_max,
        plate_scale=_ps,
        fov_deg=fov_deg,
        dub_candidates=dub_candidates,
        first_dub=first_dub,
        tri_k_neighbors=tri_k,
        hash_query_k=hash_k,
        collect_diag=_collect,
        collect_logl3=_collect or _dbg,
        x_cen=x_cen,
        y_cen=y_cen,
        use_gnomonic=use_gnomonic,
        use_rig_prior=use_rig_prior,
        scale_tol=scale_tol,
    ):
        if _dbg and pres.is_first_pass and pres.n_passed <= 10:
            pass  # query_vec debug omitted in refactor — low priority for legacy path

        log_event(
            f"INFO: Blind solver(pass dub={pres.dub:.4g}): {pres.n_passed}/{pres.n_tried} "
            f"trojuholníkov prešlo filter, {len(pres.vote_centers)} hlasov"
        )
        if _dbg and pres.logl3_samples:
            arr = np.asarray(pres.logl3_samples)
            log_event(
                f"DEBUG: Blind log_L3(img) n={len(arr)} "
                f"min={arr.min():.3f} p10={np.percentile(arr, 10):.3f} "
                f"med={np.median(arr):.3f} p90={np.percentile(arr, 90):.3f} max={arr.max():.3f} "
                f"| index[{log_L3_min:.3f},{log_L3_max:.3f}] "
                f"| below_min={pres.n_below_min} above_max={pres.n_above_max} "
                f"in_range={pres.n_in_range} "
                f"({100.0 * pres.n_in_range / max(pres.n_passed, 1):.1f}% in range)"
            )

        if len(pres.vote_centers) < 2:
            if _collect:
                _append_pass_diag(
                    debug_sink,
                    dub=pres.dub,
                    n_tried=pres.n_tried,
                    n_passed=pres.n_passed,
                    n_below_min=pres.n_below_min,
                    n_above_max=pres.n_above_max,
                    n_in_range=pres.n_in_range,
                    logl3_samples=pres.logl3_samples,
                    n_votes=0,
                    match_mult_mean=pres.match_mult_mean,
                    best_count=0,
                    best_ra=None,
                    best_dec=None,
                    significance=0.0,
                    votes_arr=None,
                    debug_truth_radec=debug_truth_radec,
                )
            continue

        votes_arr = np.array(pres.vote_centers, dtype=np.float64)
        votes_near_truth_2deg = 0
        votes_near_truth_5deg = 0
        best_count_at_truth = 0
        if debug_truth_radec is not None:
            t_ra, t_dec = float(debug_truth_radec[0]), float(debug_truth_radec[1])
            dra_t = (votes_arr[:, 0] - t_ra) * math.cos(math.radians(t_dec))
            ddec_t = votes_arr[:, 1] - t_dec
            sep_t = np.sqrt(dra_t**2 + ddec_t**2)
            votes_near_truth_2deg = int((sep_t < 2.0).sum())
            votes_near_truth_5deg = int((sep_t < 5.0).sum())
            best_count_at_truth = int((sep_t < CLUSTER_RADIUS_DEG).sum())
            if _dbg:
                log_event(
                    f"DEBUG: Blind votes near truth: <2°={votes_near_truth_2deg} "
                    f"<5°={votes_near_truth_5deg} / {len(votes_arr)}"
                )

        sky_area_deg2 = 41253.0
        cluster_area = math.pi * CLUSTER_RADIUS_DEG**2
        expected_random = max(len(pres.vote_centers) * cluster_area / sky_area_deg2, 1e-9)
        w_clusters = _cluster_match_hits_weighted(
            pres.match_hits, radius_deg=CLUSTER_RADIUS_DEG
        )
        best_ra, best_dec, best_count = (None, None, 0)
        best_weight = 0.0
        if w_clusters:
            best_ra, best_dec, best_weight, best_count = w_clusters[0]
        significance = best_weight / max(expected_random, 1e-9)
        log_event(
            f"INFO: Blind solver(pass dub={pres.dub:.4g}): klaster={best_count}, "
            f"signifikantnosť={significance:.0f}x, expected_random={expected_random:.3f}"
        )

        if _collect:
            _append_pass_diag(
                debug_sink,
                dub=pres.dub,
                n_tried=pres.n_tried,
                n_passed=pres.n_passed,
                n_below_min=pres.n_below_min,
                n_above_max=pres.n_above_max,
                n_in_range=pres.n_in_range,
                logl3_samples=pres.logl3_samples,
                n_votes=len(votes_arr),
                match_mult_mean=pres.match_mult_mean,
                best_count=best_count,
                best_ra=best_ra,
                best_dec=best_dec,
                significance=significance,
                votes_arr=votes_arr,
                debug_truth_radec=debug_truth_radec,
                votes_near_truth_2deg=votes_near_truth_2deg,
                votes_near_truth_5deg=votes_near_truth_5deg,
                best_count_at_truth=best_count_at_truth,
            )

        if best_count >= min_votes and significance >= 5.0:
            if _collect:
                debug_sink["deciding_pass_idx"] = len(debug_sink["passes"]) - 1
                debug_sink["hint"] = (best_ra, best_dec)
            log_event(
                f"INFO: Blind solver hint: RA={best_ra:.4f} Dec={best_dec:.4f} "
                f"({best_count} hlasov, {significance:.0f}x nad náhodou)"
            )
            return best_ra, best_dec

        log_event(
            f"INFO: Blind solver: klaster zamietnutý (dub={pres.dub:.4g}, count={best_count}, sig={significance:.1f}x)"
        )

    log_event("INFO: Blind solver: žiadny pass neprešiel prahmi (min_votes/significance).")
    if _collect and debug_sink.get("passes"):
        best_idx = max(
            range(len(debug_sink["passes"])),
            key=lambda i: int(debug_sink["passes"][i].get("best_count", 0)),
        )
        debug_sink["deciding_pass_idx"] = best_idx
        debug_sink["hint"] = None
    return None


def _append_pass_diag(
    debug_sink: dict,
    *,
    dub: float,
    n_tried: int,
    n_passed: int,
    n_below_min: int,
    n_above_max: int,
    n_in_range: int,
    logl3_samples: list[float],
    n_votes: int,
    match_mult_mean: float,
    best_count: int,
    best_ra: float | None,
    best_dec: float | None,
    significance: float,
    votes_arr: np.ndarray | None,
    debug_truth_radec: tuple[float, float] | None,
    votes_near_truth_2deg: int = 0,
    votes_near_truth_5deg: int = 0,
    best_count_at_truth: int = 0,
) -> None:
    rec: dict[str, Any] = {
        "dub": float(dub),
        "n_tried": int(n_tried),
        "n_passed": int(n_passed),
        "n_below_min": int(n_below_min),
        "n_above_max": int(n_above_max),
        "n_in_range": int(n_in_range),
        "n_votes": int(n_votes),
        "match_mult_mean": float(match_mult_mean),
        "best_count": int(best_count),
        "best_ra": best_ra,
        "best_dec": best_dec,
        "significance": float(significance),
    }
    if logl3_samples:
        arr = np.asarray(logl3_samples)
        rec.update(
            {
                "log_L3_n": int(len(arr)),
                "log_L3_min": float(arr.min()),
                "log_L3_p10": float(np.percentile(arr, 10)),
                "log_L3_med": float(np.median(arr)),
                "log_L3_p90": float(np.percentile(arr, 90)),
                "log_L3_max": float(arr.max()),
            }
        )
    if debug_truth_radec is not None:
        rec["votes_near_truth_2deg"] = int(votes_near_truth_2deg)
        rec["votes_near_truth_5deg"] = int(votes_near_truth_5deg)
        rec["best_count_at_truth"] = int(best_count_at_truth)
    if votes_arr is not None and len(votes_arr):
        rec["votes"] = votes_arr.copy()
    debug_sink.setdefault("passes", []).append(rec)
