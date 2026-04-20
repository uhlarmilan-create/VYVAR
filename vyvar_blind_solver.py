"""VYVAR Blind Plate Solver — Triangle Hash Matching.

Nájde aproximatívne RA/Dec stredu snímky bez akéhokoľvek hintu
z FITS hlavičky. Používa predgenerovaný index trojuholníkov z Gaia DR3
(gaia_triangles.pkl, generovaný skriptom GAIA_DR3/build_gaia_blind_index.py).

3D hash (L1/L3, L2/L3, normalizovaný log10 L3 v ″) + hlasovanie podľa centroidu;
metadata môže obsahovať aj RA/Dec vrcholov (8 stĺpcov) pre rozšírenia / diagnostiku.

Výstup: (ra_deg, dec_deg) alebo None ak sa zhoda nenašla.
"""

from __future__ import annotations

import itertools
import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from infolog import log_event
from config import AppConfig

LOGGER = logging.getLogger(__name__)

_CACHED_INDEX: dict = {}  # module-level cache: path → {tree, metadata}

CLUSTER_RADIUS_DEG = 1.0  # bolo 2.0 — menší klaster pre presnejší výsledok


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
        log_event(f"WARNING: Blind index: načítanie zlyhalo: {exc}")
        return None


def find_blind_hint(
    dao_stars: pd.DataFrame,
    index_path: str | Path,
    *,
    n_top: int = 30,
    distance_upper_bound: float = 0.004,
    min_votes: int = 3,
    plate_scale_arcsec_per_px: float | None = None,
    fov_deg: float | None = None,
) -> tuple[float, float] | None:
    """Nájde (ra_deg, dec_deg) polohu poľa bez FITS hintu.

    Args:
        dao_stars: DataFrame so stĺpcami x, y, flux — zoradený flux desc.
        index_path: Cesta k gaia_triangles.pkl.
        n_top: Koľko najjasnejších hviezd použiť (default 30).
        distance_upper_bound: Horná hranica v 2D hash KDTree (legacy; index hash_dim=3 ju ignoruje).
        min_votes: Minimálny počet hlasov v najlepšom klastri (default 3).
        plate_scale_arcsec_per_px: Nutné pre hash_dim=3 — L3 z px do ″ a normalizácia log L3.
        fov_deg: Priemer poľa [deg]; odfiltruje L3 väčšie ako ~0.9×FOV (v arcsekundách).

    Returns:
        (ra_deg, dec_deg) alebo None. Centroid najhustejšieho klastra hlasov
        so štatistickým testom významnosti voči náhodnému pozadiu.
    """
    idx = _load_index(index_path)
    if idx is None:
        return None

    hash_dim = int(idx.get("hash_dim", 2))
    if hash_dim == 4:
        log_event(
            "WARNING: Blind solver: starý 4D index — vygeneruj znova (build_gaia_blind_index.py)."
        )
        return None
    if hash_dim == 2:
        log_event(
            "WARNING: Blind solver: starý 2D index bez mierky — vygeneruj znova (build_gaia_blind_index.py)."
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
    log_L3_range = max(log_L3_max - log_L3_min, 1e-6)
    _dbg = bool(getattr(AppConfig(), "debug_platesolver", False))
    if _dbg:
        log_event(f"DEBUG: Index log_L3 rozsah: min={log_L3_min:.3f} max={log_L3_max:.3f}")

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

    # 3D KDTree je extrémne hustý; na ostrých toleranciách je zhôd málo.
    # Preto robíme multi-pass: začneme prísne a uvoľníme len ak cluster zlyhá.
    try:
        _idx_tol = float(idx.get("tolerance", 0.02))
    except (TypeError, ValueError):
        _idx_tol = 0.02
    if not math.isfinite(_idx_tol) or _idx_tol <= 0:
        _idx_tol = 0.02

    dub_candidates = [
        min(_idx_tol, 0.002),
        min(_idx_tol, 0.003),
        min(_idx_tol, 0.005),
        min(_idx_tol, 0.01),
        _idx_tol,
    ]
    # unique + sorted + keep in (0, inf)
    dub_candidates = sorted({float(d) for d in dub_candidates if math.isfinite(d) and d > 0})
    if not dub_candidates:
        dub_candidates = [0.002]

    knn_k = 8  # počet najbližších kandidátov v KDTree pre stabilnejšie hlasovanie

    log_event(
        f"INFO: Blind index: {len(metadata)} trojuholníkov, 3D hash (normalizovaný log L3), "
        f"vertices={'áno' if has_vertices else 'nie (legacy)'}"
    )

    # Vyber len hviezdy v strede snímky — obmedzíš maximálnu veľkosť trojuholníkov.
    x_max = float(dao_stars["x"].max())
    y_max = float(dao_stars["y"].max())
    x_cen = x_max / 2.0
    y_cen = y_max / 2.0

    L3_max_arcsec = 10 ** float(log_L3_max)
    if _ps > 0:
        # Vyber hviezdy v okolí R = L3_max/2 od stredu
        R_px = (L3_max_arcsec / _ps) / 2.0
        dist_from_center = np.sqrt((dao_stars["x"] - x_cen) ** 2 + (dao_stars["y"] - y_cen) ** 2)
        central_stars = dao_stars[dist_from_center <= R_px]
        if len(central_stars) >= 6:
            stars = central_stars.head(n_top)[["x", "y"]].to_numpy(dtype=np.float64)
            log_event(f"INFO: Blind solver: {len(central_stars)} hviezd v R={R_px:.0f}px od stredu")
        else:
            stars = dao_stars.head(n_top)[["x", "y"]].to_numpy(dtype=np.float64)
    else:
        stars = dao_stars.head(n_top)[["x", "y"]].to_numpy(dtype=np.float64)

    if len(stars) < 3:
        log_event(f"WARNING: Blind solver: príliš málo hviezd ({len(stars)} < 3).")
        return None

    _first_dub = float(dub_candidates[0])
    for _dub_3d in dub_candidates:
        _is_first_pass = float(_dub_3d) == _first_dub
        votes: list[np.ndarray] = []
        n_tried = 0
        n_passed = 0

        for i0, i1, i2 in itertools.combinations(range(len(stars)), 3):
            n_tried += 1
            p0, p1, p2 = stars[i0], stars[i1], stars[i2]
            d01 = math.hypot(p0[0] - p1[0], p0[1] - p1[1])
            d12 = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            d02 = math.hypot(p0[0] - p2[0], p0[1] - p2[1])
            L1, L2, L3 = sorted([d01, d12, d02])

            if L3 < 2.0 or L1 / L3 < 0.15:
                continue

            r1, r2 = L1 / L3, L2 / L3

            L3_arcsec = L3 * _ps
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
            # Filter: trojuholník musí byť v rozsahu indexu (inak zhoda nie je možná).
            if log_L3_norm < 0.0 or log_L3_norm > 1.0:
                continue
            query_vec = [float(r1), float(r2), float(log_L3_norm)]
            # Debug: vypíš prvých 10 query_vec (iba prvý pass a len v debug mode)
            if _dbg and _is_first_pass and n_passed <= 10:
                log_event(
                    f"DEBUG: Blind query_vec[{n_passed}]: "
                    f"r1={r1:.3f} r2={r2:.3f} "
                    f"L3_arcsec={L3_arcsec:.1f} log_L3={log_L3:.3f} "
                    f"log_L3_norm={log_L3_norm:.3f}"
                )

            dists, match_idxs = hash_tree.query(
                query_vec,
                k=knn_k,
                distance_upper_bound=_dub_3d,
            )

            # SciPy: pri k>1 sú návraty polia; mimo radius býva dist=inf.
            dists_arr = np.atleast_1d(dists)
            idxs_arr = np.atleast_1d(match_idxs)
            for dist, match_idx in zip(dists_arr, idxs_arr, strict=False):
                if not np.isfinite(dist) or float(dist) >= _dub_3d:
                    continue
                mi = int(match_idx)
                if 0 <= mi < len(metadata):
                    votes.append(np.asarray(metadata[mi, :2], dtype=np.float64))

        log_event(
            f"INFO: Blind solver(pass dub={_dub_3d:.4g}): {n_passed}/{n_tried} trojuholníkov prešlo filter, {len(votes)} hlasov"
        )

        if len(votes) < 2:
            continue

        votes_arr = np.array(votes, dtype=np.float64)
        if _dbg:
            # Debug: koľko hlasov padlo blízko DY Peg? (diagnostika pri konkrétnych dátach)
            dy_peg_ra, dy_peg_dec = 339.30, 17.34
            dra_dp = (votes_arr[:, 0] - dy_peg_ra) * math.cos(math.radians(dy_peg_dec))
            ddec_dp = votes_arr[:, 1] - dy_peg_dec
            sep_dp = np.sqrt(dra_dp**2 + ddec_dp**2)
            near_dy_peg = int((sep_dp < 5.0).sum())
            log_event(
                f"DEBUG: Blind votes near DY Peg (5°): {near_dy_peg}/{len(votes_arr)}"
            )
        sky_area_deg2 = 41253.0
        cluster_area = math.pi * CLUSTER_RADIUS_DEG**2
        expected_random = max(len(votes) * cluster_area / sky_area_deg2, 1e-9)

        best_ra: float | None = None
        best_dec: float | None = None
        best_count = 0

        for i in range(len(votes_arr)):
            ra_i, dec_i = float(votes_arr[i, 0]), float(votes_arr[i, 1])
            dra = (votes_arr[:, 0] - ra_i) * math.cos(math.radians(dec_i))
            ddec = votes_arr[:, 1] - dec_i
            sep = np.sqrt(dra * dra + ddec * ddec)
            in_cluster = sep < CLUSTER_RADIUS_DEG
            count = int(np.count_nonzero(in_cluster))
            if count > best_count:
                best_count = count
                best_ra = float(np.median(votes_arr[in_cluster, 0]))
                best_dec = float(np.median(votes_arr[in_cluster, 1]))

        significance = best_count / max(expected_random, 1e-9)
        log_event(
            f"INFO: Blind solver(pass dub={_dub_3d:.4g}): klaster={best_count}, "
            f"signifikantnosť={significance:.0f}x, expected_random={expected_random:.3f}"
        )

        if best_count >= min_votes and significance >= 5.0:
            log_event(
                f"INFO: Blind solver hint: RA={best_ra:.4f} Dec={best_dec:.4f} "
                f"({best_count} hlasov, {significance:.0f}x nad náhodou)"
            )
            return best_ra, best_dec

        log_event(
            f"INFO: Blind solver: klaster zamietnutý (dub={_dub_3d:.4g}, count={best_count}, sig={significance:.1f}x)"
        )

    log_event("INFO: Blind solver: žiadny pass neprešiel prahmi (min_votes/significance).")
    return None
