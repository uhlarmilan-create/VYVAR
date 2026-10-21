"""VYVAR blind plate-solving index (PLATESOLVE-2026-10).

Astrometry.net-style multi-scale index of 4-star patterns built from the local Gaia DR3
DB (Lang et al. 2010):

* **Uniformisation**: the sky is cut into cells; the brightest ``STARS_PER_CELL`` stars
  of each cell are index stars (bright stars everywhere, deeper where the sky is empty).
* **Bands**: pattern size bands d in [D/2, D], D = ``BAND0_DEG`` * 2**k. Band k uses cells
  of size D/2, so a field of diagonal ~1.3-4 D holds enough index stars and patterns.
* **Patterns**: each index star with 3 of its ``NEIGHBOURS`` nearest index stars, size
  inside the band, at most ``QUADS_PER_STAR`` per star; code = 5 sorted edge ratios
  (``astrosolve.pattern_codes``), computed in the tangent plane of the pattern.
* **Storage** (one folder per band, numpy memmaps): ``stars.npy`` (ra, dec float64,
  mag float32), ``cells.npy`` (star ranges per cell for cone lookup), ``quads.npy``
  (4 star indices, canonical order), ``codes.npy`` (float16 x 5), ``keys.npy`` (uint64
  bucket key, sorted) - looked up with ``searchsorted`` without loading into RAM.

Build: ``python catalogs/build_astro_index.py [--db <vyvar_gaia_dr3.db>] [--out <dir>]`` (one pass over
the DB by declination strips, then per band). Blind solve: :func:`solve_blind`.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import astrosolve as A

BAND0_DEG = 0.15          # smallest pattern size band upper edge (fields from ~10')
N_BANDS = 9               # 0.15 .. 38 deg
STARS_PER_CELL = 8
NEIGHBOURS = 8
QUADS_PER_STAR = 4
KEY_STEP = 2 * A.CODE_TOL  # bucket size per code dimension
KEY_BASE = int(math.ceil(1.0 / KEY_STEP)) + 1
INDEX_VERSION = 1
#: placeholder key that matches no bucket (real keys are < KEY_BASE ** 5)
_NO_KEY = np.uint64(2 ** 64 - 1)
#: image stars per pattern-search pass (astrometry.net --depth): every pass adds fainter stars
#: and searches only the patterns that contain at least one of them (BLIND-SOLVE-2026-10:
#: with 30 stars drafts 528 / 531 / 532 had no pattern in common with the index, with 50 they had)
BLIND_DEPTHS = (30, 50, 70)
#: patterns looked up per vectorised chunk
_LOOKUP_CHUNK = 50_000


def band_size(k: int) -> float:
    return BAND0_DEG * 2.0 ** k


# ---------------------------------------------------------------------------
# sky cells (dec strips, RA cells ~ square)
# ---------------------------------------------------------------------------
@dataclass
class CellGrid:
    size: float

    @property
    def n_dec(self) -> int:
        return int(math.ceil(180.0 / self.size))

    def n_ra(self, i_dec: np.ndarray | int) -> np.ndarray:
        dec_c = -90.0 + (np.asarray(i_dec) + 0.5) * self.size
        return np.maximum(1, np.ceil(360.0 * np.cos(np.radians(dec_c)) / self.size)).astype(np.int64)

    def offsets(self) -> np.ndarray:
        n = self.n_ra(np.arange(self.n_dec))
        return np.concatenate([[0], np.cumsum(n)])

    def cell_of(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        i_dec = np.clip(((dec + 90.0) / self.size).astype(np.int64), 0, self.n_dec - 1)
        nra = self.n_ra(i_dec)
        i_ra = np.minimum((np.mod(ra, 360.0) / 360.0 * nra).astype(np.int64), nra - 1)
        return self.offsets()[i_dec] + i_ra


def brightest_per_cell(ra: np.ndarray, dec: np.ndarray, mag: np.ndarray, grid: CellGrid, n: int) -> np.ndarray:
    """Indices of the ``n`` brightest stars of every cell."""
    cell = grid.cell_of(ra, dec)
    o = np.lexsort((mag, cell))
    c = cell[o]
    start = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
    rank = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
    return o[rank < n]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def unit_vectors(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    r, d = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def chord(deg: float) -> float:
    return 2.0 * math.sin(math.radians(deg) / 2.0)


def code_keys(codes: np.ndarray) -> np.ndarray:
    q = np.clip((codes / KEY_STEP).astype(np.int64), 0, KEY_BASE - 1)
    key = np.zeros(len(codes), dtype=np.uint64)
    for d in range(codes.shape[1]):
        key = key * np.uint64(KEY_BASE) + q[:, d].astype(np.uint64)
    return key


def neighbour_keys(codes: np.ndarray, tols: np.ndarray) -> np.ndarray:
    """Bucket keys that may hold codes within ``tols`` of ``codes``: (n, 2**5) per pattern.
    A tolerance below ``KEY_STEP`` spans at most two buckets per dimension; a repeated key
    in a row (both bounds in one bucket) is replaced by ``_NO_KEY``."""
    codes = np.asarray(codes, dtype=np.float64)
    t = np.asarray(tols, dtype=np.float64)[:, None]
    lo = np.clip(((codes - t) / KEY_STEP).astype(np.int64), 0, KEY_BASE - 1)
    hi = np.clip(((codes + t) / KEY_STEP).astype(np.int64), 0, KEY_BASE - 1)
    n, d = codes.shape
    bits = (np.arange(2 ** d)[:, None] >> np.arange(d)[None, :]) & 1  # (32, d)
    pick = np.where(bits[None, :, :] == 1, hi[:, None, :], lo[:, None, :])  # (n, 32, d)
    keys = np.zeros((n, 2 ** d), dtype=np.uint64)
    for j in range(d):
        keys = keys * np.uint64(KEY_BASE) + pick[:, :, j].astype(np.uint64)
    keys.sort(axis=1)
    dup = np.zeros_like(keys, dtype=bool)
    dup[:, 1:] = keys[:, 1:] == keys[:, :-1]
    keys[dup] = _NO_KEY
    return keys


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def iter_gaia_strips(db_path: str, strip_deg: float, ra_range: tuple[float, float] | None = None,
                     dec_range: tuple[float, float] = (-90.0, 90.0)) -> Iterator[np.ndarray]:
    """(n, 3) arrays ra, dec, g of the DB by declination strips."""
    from catalog_db import connect_ro  # noqa: PLC0415

    con = connect_ro(db_path)
    try:
        d = dec_range[0]
        while d < dec_range[1]:
            d2 = min(dec_range[1], d + strip_deg)
            q = "SELECT ra, dec, g_mag FROM gaia_dr3 WHERE dec >= ? AND dec < ? AND g_mag IS NOT NULL"
            args: list[float] = [d, d2]
            if ra_range is not None:
                q += " AND ra BETWEEN ? AND ?"
                args += list(ra_range)
            rows = con.execute(q, args).fetchall()
            yield np.asarray(rows, dtype=np.float64).reshape(-1, 3)
            d = d2
    finally:
        con.close()


def select_base_stars(db_path: str, *, ra_range: tuple[float, float] | None = None,
                      dec_range: tuple[float, float] = (-90.0, 90.0), progress: Any = None) -> np.ndarray:
    """One pass over the DB: brightest STARS_PER_CELL per finest cell (superset of every band)."""
    grid = CellGrid(band_size(0) / 2.0)
    out: list[np.ndarray] = []
    strip = grid.size
    n_strips = int(math.ceil((dec_range[1] - dec_range[0]) / strip))
    t0 = time.time()
    for i, arr in enumerate(iter_gaia_strips(db_path, strip, ra_range, dec_range)):
        if len(arr):
            keep = brightest_per_cell(arr[:, 0], arr[:, 1], arr[:, 2], grid, STARS_PER_CELL)
            out.append(arr[keep])
        if progress and (i % 50 == 0 or i == n_strips - 1):
            progress(f"base stars: strip {i + 1}/{n_strips}, {sum(len(x) for x in out)} stars, {time.time() - t0:.0f} s")
    return np.vstack(out) if out else np.zeros((0, 3))


def build_band(base: np.ndarray, k: int, out_dir: Path, progress: Any = None) -> dict[str, Any]:
    from scipy.spatial import cKDTree  # noqa: PLC0415

    D = band_size(k)
    grid = CellGrid(D / 2.0)
    sel = brightest_per_cell(base[:, 0], base[:, 1], base[:, 2], grid, STARS_PER_CELL)
    st = base[sel]
    cell = grid.cell_of(st[:, 0], st[:, 1])
    o = np.argsort(cell, kind="stable")
    st, cell = st[o], cell[o]
    n = len(st)
    v = unit_vectors(st[:, 0], st[:, 1])
    tree = cKDTree(v)
    kq = min(NEIGHBOURS + 1, n)
    dmax, dmin = chord(D), chord(D / 2.0)
    combos = np.array(list(__import__("itertools").combinations(range(1, kq), 3)), dtype=np.int64)
    pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    quads_all: list[np.ndarray] = []
    chunk = 20_000
    for c0 in range(0, n, chunk):
        idx = np.arange(c0, min(n, c0 + chunk))
        _, nb = tree.query(v[idx], k=kq, distance_upper_bound=dmax)
        nb = np.where(nb >= n, -1, nb)
        q = np.concatenate([np.repeat(nb[:, :1], len(combos), axis=1)[..., None],
                            nb[:, combos]], axis=2).reshape(-1, 4)
        q = q[(q >= 0).all(axis=1)]
        if not len(q):
            continue
        p = v[q]
        dd = np.linalg.norm(p[:, pairs[:, 0]] - p[:, pairs[:, 1]], axis=2).max(axis=1)
        ok = (dd >= dmin) & (dd <= dmax)
        q, dd = q[ok], dd[ok]
        # per star: QUADS_PER_STAR patterns with size closest to 3/4 D (most robust)
        target = chord(0.75 * D)
        o2 = np.lexsort((np.abs(dd - target), q[:, 0]))
        q = q[o2]
        first = q[:, 0]
        start = np.r_[0, np.flatnonzero(np.diff(first)) + 1]
        rank = np.arange(len(q)) - np.repeat(start, np.diff(np.r_[start, len(q)]))
        quads_all.append(q[rank < QUADS_PER_STAR])
    quads = np.vstack(quads_all) if quads_all else np.zeros((0, 4), dtype=np.int64)
    quads = np.unique(np.sort(quads, axis=1), axis=0)
    # codes in the tangent plane at each pattern's centre (as the camera sees it)
    codes = np.zeros((len(quads), 5), dtype=np.float32)
    canon = np.zeros_like(quads)
    for c0 in range(0, len(quads), 100_000):
        qq = quads[c0: c0 + 100_000]
        cv = v[qq].mean(axis=1)
        cv /= np.linalg.norm(cv, axis=1, keepdims=True)
        ra0 = np.degrees(np.arctan2(cv[:, 1], cv[:, 0])) % 360.0
        dec0 = np.degrees(np.arcsin(np.clip(cv[:, 2], -1, 1)))
        xi, eta = _tan_project_many(st[qq, 0], st[qq, 1], ra0[:, None], dec0[:, None])
        pts = np.stack([xi, eta], axis=2)
        cc, cn = _codes_from_points(pts)
        codes[c0: c0 + len(qq)] = cc
        canon[c0: c0 + len(qq)] = np.take_along_axis(qq, cn, axis=1)
    keys = code_keys(codes)
    o = np.argsort(keys, kind="stable")
    bdir = out_dir / f"band_{k:02d}"
    bdir.mkdir(parents=True, exist_ok=True)
    np.save(bdir / "stars.npy", st[:, :2].astype(np.float64))
    np.save(bdir / "mags.npy", st[:, 2].astype(np.float32))
    offs = np.searchsorted(cell, np.arange(grid.offsets()[-1] + 1))
    np.save(bdir / "cells.npy", offs.astype(np.int64))
    np.save(bdir / "quads.npy", canon[o].astype(np.int32))
    np.save(bdir / "codes.npy", codes[o].astype(np.float16))
    np.save(bdir / "keys.npy", keys[o])
    meta = {"band": k, "size_deg": D, "cell_deg": grid.size, "n_stars": int(n), "n_quads": int(len(quads))}
    (bdir / "meta.json").write_text(json.dumps(meta))
    if progress:
        progress(f"band {k} (D={D:.3f} deg): {n} stars, {len(quads)} patterns")
    return meta


def _tan_project_many(ra: np.ndarray, dec: np.ndarray, ra0: np.ndarray, dec0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, d = np.radians(ra), np.radians(dec)
    r0, d0 = np.radians(ra0), np.radians(dec0)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(r - r0)
    xi = np.cos(d) * np.sin(r - r0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(r - r0)) / cosc
    return np.degrees(xi), np.degrees(eta)


def _codes_from_points(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``astrosolve.pattern_codes`` for (n, 4, 2) points -> codes, canonical order."""
    pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    d = np.linalg.norm(p[:, pairs[:, 0]] - p[:, pairs[:, 1]], axis=2)
    ds = np.sort(d, axis=1)
    codes = ds[:, :5] / ds[:, 5:6]
    cen = p.mean(axis=1, keepdims=True)
    order = np.argsort(np.linalg.norm(p - cen, axis=2), axis=1)
    return codes, order


def build_index(db_path: str, out_dir: str | Path, *, ra_range: tuple[float, float] | None = None,
                dec_range: tuple[float, float] = (-90.0, 90.0), progress: Any = print) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # resumable: base star selection and finished bands are kept on disk
    bpath = out / "base_stars.npy"
    if bpath.is_file():
        base = np.load(bpath)
    else:
        base = select_base_stars(db_path, ra_range=ra_range, dec_range=dec_range, progress=progress)
        np.save(bpath, base)
    metas = []
    for k in range(N_BANDS):
        mp = out / f"band_{k:02d}" / "meta.json"
        if mp.is_file() and (mp.parent / "keys.npy").is_file():
            metas.append(json.loads(mp.read_text()))
        else:
            metas.append(build_band(base, k, out, progress))
    info = {"version": INDEX_VERSION, "band0_deg": BAND0_DEG, "n_bands": N_BANDS, "stars_per_cell": STARS_PER_CELL,
            "quads_per_star": QUADS_PER_STAR, "neighbours": NEIGHBOURS, "key_step": KEY_STEP,
            "ra_range": ra_range, "dec_range": dec_range, "bands": metas}
    (out / "index.json").write_text(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------------------
# blind solving
# ---------------------------------------------------------------------------
class Band:
    def __init__(self, bdir: Path) -> None:
        self.meta = json.loads((bdir / "meta.json").read_text())
        self.stars = np.load(bdir / "stars.npy", "r")
        self.mags = np.load(bdir / "mags.npy", "r")
        self.cells = np.load(bdir / "cells.npy", "r")
        self.quads = np.load(bdir / "quads.npy", "r")
        self.codes = np.load(bdir / "codes.npy", "r")
        self.keys = np.load(bdir / "keys.npy", "r")
        self.grid = CellGrid(float(self.meta["cell_deg"]))

    def lookup(self, codes: np.ndarray, tols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Index patterns within ``tols`` of ``codes`` (n, 5), all at once: returns
        (row of ``codes``, index pattern) pairs."""
        rows: list[np.ndarray] = []
        hits: list[np.ndarray] = []
        for c0 in range(0, len(codes), _LOOKUP_CHUNK):
            cc = np.asarray(codes[c0: c0 + _LOOKUP_CHUNK], dtype=np.float32)
            tt = np.asarray(tols[c0: c0 + _LOOKUP_CHUNK], dtype=np.float32)
            keys = neighbour_keys(cc, tt)
            owner = np.repeat(np.arange(len(cc)), keys.shape[1])
            k = keys.ravel()
            a = np.searchsorted(self.keys, k, "left")
            b = np.searchsorted(self.keys, k, "right")
            n = b - a
            m = n > 0
            if not m.any():
                continue
            n, a, owner = n[m], a[m], owner[m]
            total = int(n.sum())
            start = np.repeat(a, n)
            offs = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
            qi = start + offs
            ow = np.repeat(owner, n)
            ok = np.max(np.abs(np.asarray(self.codes[qi], dtype=np.float32) - cc[ow]), axis=1) <= tt[ow]
            rows.append(c0 + ow[ok])
            hits.append(qi[ok])
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.concatenate(rows), np.concatenate(hits)

    def cone(self, ra0: float, dec0: float, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = self.grid
        dmin, dmax = max(-90.0, dec0 - radius), min(90.0, dec0 + radius)
        i0 = int(np.clip((dmin + 90.0) / g.size, 0, g.n_dec - 1))
        i1 = int(np.clip((dmax + 90.0) / g.size, 0, g.n_dec - 1))
        offs = g.offsets()
        idx: list[np.ndarray] = []
        for i in range(i0, i1 + 1):
            nra = int(g.n_ra(i))
            dec_c = -90.0 + (i + 0.5) * g.size
            cosd = max(math.cos(math.radians(min(89.9, abs(dec_c) + g.size))), 1e-3)
            dra = radius / cosd + 360.0 / nra
            if dra >= 180.0:
                ra_cells = range(nra)
            else:
                c0 = int(math.floor((ra0 - dra) % 360.0 / 360.0 * nra))
                n_c = int(math.ceil(2 * dra / 360.0 * nra)) + 1
                ra_cells = [(c0 + j) % nra for j in range(min(n_c, nra))]
            for j in ra_cells:
                c = offs[i] + j
                a, b = int(self.cells[c]), int(self.cells[c + 1])
                if b > a:
                    idx.append(np.arange(a, b))
        if not idx:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        ii = np.concatenate(idx)
        ra, dec = np.asarray(self.stars[ii, 0]), np.asarray(self.stars[ii, 1])
        xi, eta = A.tan_project(ra, dec, ra0, dec0)
        m = np.hypot(xi, eta) <= radius
        return ra[m], dec[m], np.asarray(self.mags[ii])[m]


def load_index(index_dir: str | Path) -> list[Band]:
    p = Path(index_dir)
    return [Band(b) for b in sorted(p.glob("band_*")) if (b / "keys.npy").is_file()]


def solve_blind(
    det_xy: np.ndarray,
    det_flux: np.ndarray,
    shape: tuple[int, int],
    *,
    index_dir: str | Path,
    gaia_db: str | None = None,
    fwhm_px: float | None = None,
    scale_range_arcsec: tuple[float, float] = (0.05, 300.0),
    max_hypotheses: int = 60_000,
) -> A.SolveResult:
    """Solve without any position (astrometry.net style, Lang et al. 2010).

    Patterns of the brightest image stars (uniformised), deepened pass by pass
    (``BLIND_DEPTHS``), are looked up in every index band; hypotheses are tried large
    image patterns first. A hypothesis whose log-odds against the band's index stars
    reach ``LOG_ODDS_TUNE_UP`` is refined against the full Gaia DB (astrometry.net
    "tune-up": a 4-star similarity is exact only near its pattern) and becomes the
    solution only when it passes ``credible_solution``. Without a Gaia DB the index stars
    refine it and ``LOG_ODDS_ACCEPT`` applies."""
    ny, nx = shape
    diag = math.hypot(nx, ny)
    sigma = max(1.0, 0.5 * (fwhm_px or 2.0))
    order = A.uniform_brightest(det_xy, -det_flux, BLIND_DEPTHS[-1], (0, nx, 0, ny))
    vxy = det_xy[np.argsort(-det_flux)[:150]]
    bands = load_index(index_dir)
    accept_at = A.LOG_ODDS_TUNE_UP if gaia_db else A.LOG_ODDS_ACCEPT
    tried = 0
    best = A.SolveResult(False, message="no blind hypothesis verified")
    prev = 0
    for depth in BLIND_DEPTHS:
        n_img = min(depth, len(order))
        if n_img <= prev or n_img < 4:
            break
        ixy = det_xy[order[:n_img]]
        iq = A.make_quads(ixy, min_size=A.MIN_PATTERN_FRACTION * diag, max_size=diag, max_quads=10_000_000)
        iq = iq[iq.max(axis=1) >= prev]  # only patterns with at least one star new in this pass
        prev = n_img
        if not len(iq):
            continue
        icode, icanon, isize = A.pattern_codes(ixy, iq)
        itol = A.code_tolerance(isize, fwhm_px)
        cand_ii: list[np.ndarray] = []
        cand_band: list[np.ndarray] = []
        cand_qi: list[np.ndarray] = []
        for bi, band in enumerate(bands):
            D = float(band.meta["size_deg"])
            s_lo = (D / 2.0) / isize * 3600.0
            s_hi = D / isize * 3600.0
            sel = np.where((s_hi >= scale_range_arcsec[0]) & (s_lo <= scale_range_arcsec[1]))[0]
            if not len(sel):
                continue
            rows, qis = band.lookup(icode[sel], itol[sel])
            cand_ii.append(sel[rows])
            cand_band.append(np.full(len(rows), bi, dtype=np.int64))
            cand_qi.append(qis)
        if not cand_ii:
            continue
        all_ii, all_band, all_qi = np.concatenate(cand_ii), np.concatenate(cand_band), np.concatenate(cand_qi)
        # index stars of every candidate pattern in one read per band (memmap access per
        # pattern dominated the run time)
        all_ra = np.empty((len(all_qi), 4))
        all_dec = np.empty((len(all_qi), 4))
        for bi in np.unique(all_band):
            m = np.where(all_band == bi)[0]
            o = np.argsort(all_qi[m])
            sid = np.asarray(bands[int(bi)].quads[all_qi[m][o]], dtype=np.int64)
            st = np.asarray(bands[int(bi)].stars[np.sort(np.unique(sid))])
            pos = np.searchsorted(np.sort(np.unique(sid)), sid)
            all_ra[m[o]] = st[pos, 0]
            all_dec[m[o]] = st[pos, 1]
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
        for k in np.argsort(-isize[all_ii], kind="stable"):  # large image patterns first
            ii, band = int(all_ii[k]), bands[int(all_band[k])]
            D = float(band.meta["size_deg"])
            src = ixy[icanon[ii]]
            sra, sdec = all_ra[k], all_dec[k]
            cv = unit_vectors(sra, sdec).mean(axis=0)
            ra0 = math.degrees(math.atan2(cv[1], cv[0])) % 360.0
            dec0 = math.degrees(math.asin(cv[2] / np.linalg.norm(cv)))
            xi, eta = A.tan_project(sra, sdec, ra0, dec0)
            dst = np.column_stack([xi, eta])
            for mirror in (False, True):
                a, b, rms = A.fit_similarity(src, dst, mirror)
                s_as = abs(a) * 3600.0
                if not math.isfinite(rms) or rms > 0.02 * D or not (scale_range_arcsec[0] <= s_as <= scale_range_arcsec[1]):
                    continue
                w = A.wcs_from_similarity(a, b, mirror, ra0, dec0, shape)
                c_ra, c_dec = w.wcs_pix2world([[cx, cy]], 0)[0]
                rad = 0.55 * diag * s_as / 3600.0
                cra, cdec, cmag = band.cone(float(c_ra), float(c_dec), rad)
                lo, nm, w = A.verify_hypothesis(w, vxy, cra, cdec, cmag, shape, sigma_px=sigma)
                if not math.isfinite(lo):
                    continue
                tried += 1
                if lo > best.log_odds:
                    best = A.SolveResult(False, wcs=w, log_odds=lo, n_match=nm, mirrored=mirror,
                                         scale_arcsec=s_as, message="best rejected blind hypothesis")
                if lo >= accept_at:
                    if gaia_db:
                        res = A.refine_with_gaia(w, s_as, det_xy, shape, gaia_db=gaia_db, sigma=sigma)
                        if res.ok:
                            res = A._final_refine(res, det_xy, shape, gaia_db=gaia_db, sigma=sigma)
                    else:
                        res = A.refine_wcs(w, det_xy, cra, cdec, shape, sigma_px=sigma)
                    ok, why = A.credible_solution(res, len(det_xy), sigma)
                    if not ok:
                        best.message = f"verified blind hypothesis rejected: {why}"
                        continue
                    res.log_odds = max(res.log_odds, lo)
                    res.n_hypotheses = tried
                    res.extra["blind_band"] = int(band.meta["band"])
                    res.extra["blind_depth"] = int(n_img)
                    return res
                if tried >= max_hypotheses:
                    best.n_hypotheses = tried
                    return best
    best.n_hypotheses = tried
    return best


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the VYVAR blind plate-solving index from the local Gaia DR3 DB")
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ra", nargs=2, type=float, default=None, help="RA range (test builds)")
    ap.add_argument("--dec", nargs=2, type=float, default=(-90.0, 90.0), help="Dec range (test builds)")
    a = ap.parse_args()
    build_index(a.db, a.out, ra_range=tuple(a.ra) if a.ra else None, dec_range=tuple(a.dec))
