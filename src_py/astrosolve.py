"""VYVAR plate solver core (PLATESOLVE-2026-10, Milan 2026-10-09).

Astrometry.net-style solving (Lang et al. 2010, AJ 139, 1782) with ASTAP / Tetra3
style 4-star patterns:

* **Pattern code**: 4 stars -> the 6 pair distances sorted, divided by the largest ->
  5 ratios. Invariant to translation, rotation, scale and mirror. Stars of a pattern
  are put in a canonical order (distance from the pattern centroid) so that a code hit
  gives the 4 star correspondences.
* **Hypothesis**: from the 4 correspondences a similarity transform pixel -> tangent
  plane (scale, rotation, translation; both parities tried) -> TAN WCS.
* **Verification** (Lang et al. 2010, sect. 2.4): Bayesian log-odds of "the image is
  this sky" against "the detections are uniform noise"; foreground = mixture of a
  Gaussian around the nearest projected catalog star (sigma = positional error) and
  a distractor fraction; accept at log-odds >= ln(1e9).
* **Refinement**: iterative nearest matching + ``astropy.wcs.utils.fit_wcs_from_points``
  (TAN + SIP, standard convention); SIP order chosen by BIC with >= 5 matches per
  coefficient.

No plate scale is needed: the code is scale free; a scale hint only narrows the
catalog search radius. Brightest stars are used on both sides (image detections
and catalog stars uniformised over the field), as in astrometry.net / ASTAP.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOG_ODDS_ACCEPT = math.log(1e9)
#: astrometry.net --odds-to-tune-up: a blind hypothesis this good is refined against the full
#: catalog before the decision (credible_solution decides, not this threshold)
LOG_ODDS_TUNE_UP = math.log(1e6)
#: frame match rate (catalog matches / detections) below which OSC channel hand-over warns
MASTERSTAR_PLATESOLVE_MIN_MATCH_RATE = 0.60
DISTRACTOR_FRACTION = 0.25
CODE_TOL = 0.01
N_IMAGE_PATTERN_STARS = 30
N_CATALOG_PATTERN_STARS = 50
#: smallest pattern relative to the field diagonal (tiny patterns are dominated by centroid noise)
MIN_PATTERN_FRACTION = 0.10
# BLIND-SOLVE-2026-10 (Milan 2026-10-20). With few catalog stars in the field the foreground
# term of one match is huge (area / (K 2 pi sigma^2)), so a handful of chance matches passed
# ln(1e9). Measured on the true WCS of drafts 529 / 536: useful index bands put 20-300 stars in
# the field (band 3 of 529: 4 stars still gave log-odds 22.3 > 20.7).
# G03-02 (2026-10-29): the root cause (the 4 pattern stars vouching for their own hypothesis) is
# fixed - verify_hypothesis leaves them out. With them excluded, K = 5..19 catalog stars + 146
# random detections never reached ln(1e6) (3000 trials, max log-odds -17.8). The floor is kept
# for now (it also covers confirm_fits, which has no pattern); removing it is an open question.
#: a hypothesis is verified only against at least this many catalog stars inside the field
MIN_VERIFY_CATALOG_STARS = 20
#: the linear "tweak" re-fit needs this many pairs; the detections used in that fit are not
#: counted again in its log-odds (a fit on 4 pairs matches those 4 pairs by construction)
TWEAK_MIN_PAIRS = 10
#: a solution is accepted only after the TAN+SIP refinement against the full catalog with at
#: least this many 1:1 matches, this fraction of the detections, and an rms within this many
#: sigma (true solutions 529 / 536: 2467 / 2129 matches, 86 / 98 %, rms 0.15 / 0.18 sigma)
MIN_FINAL_MATCHES = 20
MIN_FINAL_MATCH_FRACTION = 0.10
MAX_FINAL_RMS_SIGMA = 1.0


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def tan_project(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic projection (degrees in, degrees out) about (ra0, dec0)."""
    r, d = np.radians(ra), np.radians(dec)
    r0, d0 = math.radians(ra0), math.radians(dec0)
    cosc = math.sin(d0) * np.sin(d) + math.cos(d0) * np.cos(d) * np.cos(r - r0)
    xi = np.cos(d) * np.sin(r - r0) / cosc
    eta = (math.cos(d0) * np.sin(d) - math.sin(d0) * np.cos(d) * np.cos(r - r0)) / cosc
    return np.degrees(xi), np.degrees(eta)


def tan_deproject(xi: np.ndarray, eta: np.ndarray, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.radians(xi), np.radians(eta)
    d0 = math.radians(dec0)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(np.cos(c) * math.sin(d0) + np.where(rho > 0, y * np.sin(c) * math.cos(d0) / rho, 0.0))
        ra = math.radians(ra0) + np.arctan2(x * np.sin(c), rho * math.cos(d0) * np.cos(c) - y * math.sin(d0) * np.sin(c))
    return np.degrees(ra) % 360.0, np.degrees(dec)


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------
def pattern_codes(xy: np.ndarray, quads: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Codes (n, 5), canonical star order (n, 4) and size (largest distance) of quads.

    ``quads`` (n, 4) are indices into ``xy`` (m, 2)."""
    p = xy[quads]  # (n, 4, 2)
    pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    d = np.linalg.norm(p[:, pairs[:, 0]] - p[:, pairs[:, 1]], axis=2)  # (n, 6)
    ds = np.sort(d, axis=1)
    size = ds[:, -1]
    with np.errstate(invalid="ignore", divide="ignore"):
        code = ds[:, :5] / size[:, None]
    cen = p.mean(axis=1, keepdims=True)
    order = np.argsort(np.linalg.norm(p - cen, axis=2), axis=1)
    canon = np.take_along_axis(quads, order, axis=1)
    return code, canon, size


def make_quads(xy: np.ndarray, *, min_size: float, max_size: float, max_quads: int = 400_000) -> np.ndarray:
    """All 4-star combinations of ``xy`` whose largest pair distance is in [min_size, max_size]."""
    n = len(xy)
    if n < 4:
        return np.zeros((0, 4), dtype=np.int64)
    combos = np.array(list(itertools.combinations(range(n), 4)), dtype=np.int64)
    p = xy[combos]
    pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    d = np.linalg.norm(p[:, pairs[:, 0]] - p[:, pairs[:, 1]], axis=2).max(axis=1)
    keep = (d >= min_size) & (d <= max_size)
    out = combos[keep]
    return out[:max_quads]


def uniform_brightest(xy: np.ndarray, mag_or_negflux: np.ndarray, n: int, extent: tuple[float, float, float, float]) -> np.ndarray:
    """Indices of up to ``n`` bright sources spread over the extent: brightest first per
    cell of a grid, cells visited round-robin (astrometry.net uniformisation)."""
    m = len(xy)
    if m <= n:
        return np.argsort(mag_or_negflux)
    x0, x1, y0, y1 = extent
    g = max(2, int(math.sqrt(n / 2)))
    cx = np.clip(((xy[:, 0] - x0) / max(x1 - x0, 1e-12) * g).astype(int), 0, g - 1)
    cy = np.clip(((xy[:, 1] - y0) / max(y1 - y0, 1e-12) * g).astype(int), 0, g - 1)
    cell = cx * g + cy
    order = np.argsort(mag_or_negflux)
    buckets: dict[int, list[int]] = {}
    for i in order:
        buckets.setdefault(int(cell[i]), []).append(int(i))
    out: list[int] = []
    rank = 0
    while len(out) < n:
        added = False
        for b in buckets.values():
            if rank < len(b):
                out.append(b[rank])
                added = True
                if len(out) >= n:
                    break
        if not added:
            break
        rank += 1
    return np.array(out, dtype=np.int64)


def code_tolerance(size_px: np.ndarray, fwhm_px: float | None) -> np.ndarray:
    """Code tolerance per image pattern: a centroid error of ~FWHM/6 px (>= 0.3 px) moves
    a distance ratio by ~4 * err / size; floor 0.002 (index-side projection / distortion),
    ceiling CODE_TOL. Large patterns get tight tolerances -> far fewer false hits."""
    err = max(0.3, (fwhm_px or 2.0) / 6.0)
    return np.clip(4.0 * err / np.maximum(size_px, 1.0), 0.002, CODE_TOL)


# ---------------------------------------------------------------------------
# transforms and WCS
# ---------------------------------------------------------------------------
def fit_similarity(src: np.ndarray, dst: np.ndarray, mirror: bool) -> tuple[complex, complex, float]:
    """dst = a * z + b with z = src (or conj(src) when mirror) as complex numbers.
    Returns (a, b, rms residual in dst units)."""
    z = src[:, 0] + 1j * src[:, 1]
    if mirror:
        z = np.conj(z)
    w = dst[:, 0] + 1j * dst[:, 1]
    zm, wm = z.mean(), w.mean()
    dz, dw = z - zm, w - wm
    den = np.sum(np.abs(dz) ** 2)
    if den <= 0:
        return 0j, 0j, float("inf")
    a = np.sum(dw * np.conj(dz)) / den
    b = wm - a * zm
    res = np.abs(a * z + b - w)
    return complex(a), complex(b), float(np.sqrt(np.mean(res ** 2)))


def wcs_from_similarity(a: complex, b: complex, mirror: bool, ra0: float, dec0: float, shape: tuple[int, int]) -> Any:
    """TAN WCS of the similarity pixel (0-based x, y) -> tangent plane (deg) about (ra0, dec0).

    G03-01: the similarity maps pixels into the plane tangent at (ra0, dec0), so that point is
    CRVAL and CRPIX is the pixel that maps onto it (z0 = -b / a; conjugated for the mirror
    parity); the CD matrix is the linear part. Moving CRVAL to the image centre while keeping the
    CD of the (ra0, dec0) plane dropped the meridian convergence and the plane curvature between
    the two tangent points (222 px at 9.8"/px, Dec 60, corner pattern). ``shape`` is unused: the
    refinement re-centres the WCS on the image."""
    from astropy.wcs import WCS  # noqa: PLC0415

    del shape
    if a == 0:
        raise ValueError("degenerate similarity (scale 0)")
    z0 = -b / a
    if mirror:
        z0 = z0.conjugate()
    # linear part: d(xi, eta)/d(x, y)
    if mirror:
        cd = np.array([[a.real, a.imag], [a.imag, -a.real]])
    else:
        cd = np.array([[a.real, -a.imag], [a.imag, a.real]])
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [z0.real + 1.0, z0.imag + 1.0]
    w.wcs.crval = [float(ra0) % 360.0, float(dec0)]
    w.wcs.cd = cd
    return w


def log_odds(
    det_xy: np.ndarray,
    cat_xy: np.ndarray,
    *,
    sigma_px: float,
    area_px: float,
    distractor: float = DISTRACTOR_FRACTION,
) -> tuple[float, int]:
    """Lang et al. 2010 verification: sum over detections (brightest first) of
    log[(1-d) * (1/K) * N(r; sigma) * A + d]. Each catalog star explains at most one
    detection. Returns (log-odds, number of matches within 3 sigma)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    k = len(cat_xy)
    if k == 0 or len(det_xy) == 0:
        return -math.inf, 0
    tree = cKDTree(cat_xy)
    used = np.zeros(k, dtype=bool)
    lo = 0.0
    n_match = 0
    norm = (1.0 - distractor) * area_px / (k * 2.0 * math.pi * sigma_px ** 2)
    dist, idx = tree.query(det_xy, k=min(3, k))
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    for dr, ir in zip(dist, idx):
        term = distractor
        for r, j in zip(dr, ir):
            if np.isfinite(r) and not used[j]:
                g = math.exp(-0.5 * (r / sigma_px) ** 2)
                if r <= 3.0 * sigma_px:
                    used[j] = True
                    n_match += 1
                term = norm * g + distractor
                break
        lo += math.log(term)
    return lo, n_match


# ---------------------------------------------------------------------------
# hypothesis verification (shared by hinted and blind solving)
# ---------------------------------------------------------------------------
def verify_hypothesis(
    w: Any,
    det_xy_bright: np.ndarray,
    cat_ra: np.ndarray,
    cat_dec: np.ndarray,
    cat_mag: np.ndarray,
    shape: tuple[int, int],
    *,
    sigma_px: float,
    exclude: np.ndarray | None = None,
) -> tuple[float, int, Any]:
    """Log-odds of a WCS hypothesis. Detections (brightest first) are balanced to the
    number of catalog stars in the field; fewer than ``MIN_VERIFY_CATALOG_STARS`` catalog
    stars in the field cannot verify anything. ``exclude`` (indices into ``det_xy_bright``)
    are the detections that generated the hypothesis - the 4 stars of the matched pattern
    fit it by construction and are left out of every log-odds and of the balance count
    (G03-02; Lang et al. 2010 sect. 2.4). A 4-star hypothesis is only exact near its
    pattern, so when it is promising (>= ``TWEAK_MIN_PAIRS`` pairs within a loose radius)
    the linear TAN is re-fitted once on those pairs ("tweak"); the detections used in that
    fit are left out of its log-odds, so the fit cannot vouch for itself.
    Returns (log-odds, matches, WCS used)."""
    from astropy.coordinates import SkyCoord  # noqa: PLC0415
    from astropy.wcs.utils import fit_wcs_from_points  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    if exclude is not None and len(exclude):
        keep = np.ones(len(det_xy_bright), dtype=bool)
        keep[np.asarray(exclude, dtype=np.int64)] = False
        det_xy_bright = det_xy_bright[keep]
    ny, nx = shape
    area = float(nx * ny)
    cpx, inside = _project_catalog(w, cat_ra, cat_dec, shape)
    ci = np.where(inside)[0]
    if len(ci) < MIN_VERIFY_CATALOG_STARS:
        return -math.inf, 0, w
    ci = ci[np.argsort(cat_mag[ci])][:300]
    n_det = min(len(det_xy_bright), max(20, len(ci)))
    dxy = det_xy_bright[:n_det]
    lo, nm = log_odds(dxy, cpx[ci], sigma_px=sigma_px, area_px=area)
    if lo >= LOG_ODDS_ACCEPT:
        return lo, nm, w
    loose = max(5.0 * sigma_px, 0.01 * math.hypot(nx, ny))
    dist, j = cKDTree(cpx[ci]).query(dxy, distance_upper_bound=loose)
    ok = np.isfinite(dist)
    if ok.sum() < TWEAK_MIN_PAIRS:
        return lo, nm, w
    try:
        sky = SkyCoord(cat_ra[ci[j[ok]]], cat_dec[ci[j[ok]]], unit="deg")
        w2 = fit_wcs_from_points((dxy[ok, 0], dxy[ok, 1]), sky, projection="TAN")
    except (ValueError, np.linalg.LinAlgError):
        return lo, nm, w
    cpx2, inside2 = _project_catalog(w2, cat_ra, cat_dec, shape)
    ci2 = np.where(inside2)[0]
    if len(ci2) < MIN_VERIFY_CATALOG_STARS:
        return lo, nm, w
    ci2 = ci2[np.argsort(cat_mag[ci2])][:300]
    unused = np.ones(len(det_xy_bright), dtype=bool)
    unused[np.where(ok)[0]] = False
    rest = det_xy_bright[unused]
    n_det2 = min(len(rest), max(20, len(ci2)))
    lo2, nm2 = log_odds(rest[:n_det2], cpx2[ci2], sigma_px=sigma_px, area_px=area)
    return (lo2, nm2, w2) if lo2 > lo else (lo, nm, w)


# ---------------------------------------------------------------------------
# solving against a catalog
# ---------------------------------------------------------------------------
@dataclass
class SolveResult:
    ok: bool
    wcs: Any = None
    log_odds: float = -math.inf
    n_match: int = 0
    rms_px: float | None = None
    scale_arcsec: float | None = None
    rotation_deg: float | None = None
    mirrored: bool | None = None
    sip_order: int | None = None
    n_hypotheses: int = 0
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _project_catalog(wcs: Any, ra: np.ndarray, dec: np.ndarray, shape: tuple[int, int], margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    x, y = wcs.all_world2pix(ra, dec, 0)
    ny, nx = shape
    inside = np.isfinite(x) & np.isfinite(y) & (x > -margin) & (x < nx - 1 + margin) & (y > -margin) & (y < ny - 1 + margin)
    return np.column_stack([x, y]), inside


def solve_against_catalog(
    det_xy: np.ndarray,
    det_flux: np.ndarray,
    shape: tuple[int, int],
    cat_ra: np.ndarray,
    cat_dec: np.ndarray,
    cat_mag: np.ndarray,
    *,
    ra0: float,
    dec0: float,
    sigma_px: float,
    scale_range_arcsec: tuple[float, float] | None = None,
    max_hypotheses: int = 20_000,
    gaia_db: str | None = None,
) -> SolveResult:
    """Match image patterns to catalog patterns (catalog = stars around ra0/dec0) and
    verify hypotheses. Every hypothesis above ``LOG_ODDS_ACCEPT`` goes through
    :func:`accept_hypothesis` (refinement + ``credible_solution``); the first one that passes
    is returned, a rejected one does not end the search (G03-04)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    ny, nx = shape
    diag = math.hypot(nx, ny)
    if len(det_xy) < 4 or len(cat_ra) < 4:
        return SolveResult(False, message="fewer than 4 stars")
    # image side: brightest detections, spread over the frame
    i_sel = uniform_brightest(det_xy, -det_flux, N_IMAGE_PATTERN_STARS, (0, nx, 0, ny))
    ixy = det_xy[i_sel]
    iq = make_quads(ixy, min_size=MIN_PATTERN_FRACTION * diag, max_size=diag)
    if len(iq) == 0:
        return SolveResult(False, message="no image patterns")
    icode, icanon, isize = pattern_codes(ixy, iq)
    # catalog side: tangent plane about ra0/dec0, brightest uniformised
    xi, eta = tan_project(cat_ra, cat_dec, ra0, dec0)
    cxy_all = np.column_stack([xi, eta])
    rmax = float(np.max(np.hypot(xi, eta)))
    c_sel = uniform_brightest(cxy_all, cat_mag, N_CATALOG_PATTERN_STARS, (-rmax, rmax, -rmax, rmax))
    cxy = cxy_all[c_sel]
    cmax = 2.0 * rmax
    cq = make_quads(cxy, min_size=0.0, max_size=cmax)
    if len(cq) == 0:
        return SolveResult(False, message="no catalog patterns")
    ccode, ccanon, csize = pattern_codes(cxy, cq)
    tree = cKDTree(ccode)
    hits = tree.query_ball_point(icode, r=code_tolerance(isize, 2.0 * sigma_px))
    # verification catalog: brightest stars, a few times the detection count
    n_ver_det = min(len(det_xy), 150)
    det_order = np.argsort(-det_flux)[:n_ver_det]
    vxy = det_xy[det_order]
    # position of every detection in the verification list (-1 = not in it)
    ver_pos = np.full(len(det_xy), -1, dtype=np.int64)
    ver_pos[det_order] = np.arange(len(det_order))
    tested = 0
    best = SolveResult(False, message="no hypothesis passed verification")
    order_hits = np.argsort(-isize)  # large image patterns first (better leverage)
    for ii in order_hits:
        for cj in hits[ii]:
            scale_deg_per_px = csize[cj] / isize[ii]
            s_as = scale_deg_per_px * 3600.0
            if scale_range_arcsec is not None and not (scale_range_arcsec[0] <= s_as <= scale_range_arcsec[1]):
                continue
            # the field must fit in the catalog cone
            if diag * scale_deg_per_px > 2.2 * rmax:
                continue
            src = ixy[icanon[ii]]
            dst = cxy[ccanon[cj]]
            for mirror in (False, True):
                a, b, rms = fit_similarity(src, dst, mirror)
                if not math.isfinite(rms) or rms > 0.02 * csize[cj]:
                    continue
                w = wcs_from_similarity(a, b, mirror, ra0, dec0, shape)
                quad = ver_pos[i_sel[icanon[ii]]]
                lo, nm, w = verify_hypothesis(w, vxy, cat_ra, cat_dec, cat_mag, shape, sigma_px=sigma_px,
                                              exclude=quad[quad >= 0])
                if not math.isfinite(lo):
                    continue
                tested += 1
                if lo > best.log_odds:
                    best = SolveResult(False, wcs=w, log_odds=lo, n_match=nm, mirrored=mirror,
                                       scale_arcsec=s_as, message="best rejected hypothesis")
                if lo >= LOG_ODDS_ACCEPT:
                    res, why = accept_hypothesis(w, det_xy, shape, sigma_px=sigma_px, gaia_db=gaia_db,
                                                 cat=(cat_ra, cat_dec))
                    if res is None:
                        best.message = f"verified hypothesis rejected: {why}"
                    else:
                        res.n_hypotheses = tested
                        res.log_odds = max(res.log_odds, lo)
                        return res
                if tested >= max_hypotheses:
                    best.n_hypotheses = tested
                    return best
    best.n_hypotheses = tested
    return best


def _bic_sip_fit(xy: np.ndarray, sky: Any, order: int | None, proj_point: Any) -> tuple[Any, float, int]:
    from astropy.wcs.utils import fit_wcs_from_points  # noqa: PLC0415

    w = fit_wcs_from_points((xy[:, 0], xy[:, 1]), sky, proj_point=proj_point, projection="TAN",
                            sip_degree=order)
    px, py = w.all_world2pix(sky.ra.deg, sky.dec.deg, 0)
    res = np.hypot(px - xy[:, 0], py - xy[:, 1])
    n = len(res)
    k = 6 + (0 if not order else 2 * sum(1 for i in range(order + 1) for j in range(order + 1 - i) if i + j >= 2))
    rss = float(np.sum(res ** 2))
    bic = n * math.log(max(rss / n, 1e-12)) + k * math.log(n)
    return w, bic, k


def refine_wcs(
    w0: Any,
    det_xy: np.ndarray,
    cat_ra: np.ndarray,
    cat_dec: np.ndarray,
    shape: tuple[int, int],
    *,
    sigma_px: float,
    n_iter: int = 4,
) -> SolveResult:
    """Iterative 1:1 matching (radius 3 sigma) and TAN+SIP fit; SIP order by BIC."""
    from astropy.coordinates import SkyCoord  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    w = w0
    order_used: int | None = None
    for _it in range(n_iter):
        cpx, inside = _project_catalog(w, cat_ra, cat_dec, shape, margin=0.0)
        ci = np.where(inside)[0]
        if len(ci) < 6:
            return SolveResult(False, wcs=w, message="too few catalog stars in field")
        tree = cKDTree(cpx[ci])
        dist, j = tree.query(det_xy, distance_upper_bound=3.0 * sigma_px)
        ok = np.isfinite(dist)
        # 1:1 - keep the closest detection per catalog star
        best: dict[int, tuple[float, int]] = {}
        for di in np.where(ok)[0]:
            cj = int(j[di])
            if cj not in best or dist[di] < best[cj][0]:
                best[cj] = (float(dist[di]), int(di))
        if len(best) < 6:
            return SolveResult(False, wcs=w, message="too few matches in refinement")
        cat_i = ci[np.array(list(best.keys()))]
        det_i = np.array([v[1] for v in best.values()])
        xy = det_xy[det_i]
        sky = SkyCoord(cat_ra[cat_i], cat_dec[cat_i], unit="deg")
        centre = w.pixel_to_world((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)
        cands = []
        for order in (None, 2, 3, 4, 5):
            k_need = 6 if order is None else 6 + 2 * sum(1 for i in range(order + 1) for jj in range(order + 1 - i) if i + jj >= 2)
            if len(xy) < 5 * k_need:
                break
            try:
                wf, bic, _k = _bic_sip_fit(xy, sky, order, centre)
            except (ValueError, np.linalg.LinAlgError):
                continue
            cands.append((bic, order, wf))
        if not cands:
            return SolveResult(False, wcs=w, message="WCS fit failed")
        cands.sort(key=lambda t: t[0])
        _, order_used, w = cands[0]
    px, py = w.all_world2pix(sky.ra.deg, sky.dec.deg, 0)
    rms = float(np.sqrt(np.mean((px - xy[:, 0]) ** 2 + (py - xy[:, 1]) ** 2)))
    cd = w.wcs.cd if w.wcs.has_cd() else w.pixel_scale_matrix
    det = float(np.linalg.det(cd))
    scale = math.sqrt(abs(det)) * 3600.0
    rot = math.degrees(math.atan2(cd[1, 0], cd[0, 0]))
    return SolveResult(True, wcs=w, n_match=len(xy), rms_px=rms, scale_arcsec=scale, rotation_deg=rot,
                       mirrored=det > 0, sip_order=order_used, message="solved")


def credible_solution(res: SolveResult, n_det: int, sigma_px: float) -> tuple[bool, str]:
    """Final acceptance of a refined solution (BLIND-SOLVE-2026-10): enough 1:1 matches,
    a minimum fraction of the detections and an rms inside the matching sigma. Every
    solver path applies it after the refinement; a hypothesis that fails is not a
    solution and the search goes on."""
    if not res.ok:
        return False, f"refinement failed: {res.message}"
    need = max(MIN_FINAL_MATCHES, int(math.ceil(MIN_FINAL_MATCH_FRACTION * n_det)))
    if res.n_match < need:
        return False, f"only {res.n_match} matches after refinement (need {need})"
    if res.rms_px is None or not res.rms_px <= MAX_FINAL_RMS_SIGMA * sigma_px:
        return False, f"refined rms {res.rms_px} px above {MAX_FINAL_RMS_SIGMA} sigma ({sigma_px:.2f} px)"
    return True, "solved"


def accept_hypothesis(
    w: Any,
    det_xy: np.ndarray,
    shape: tuple[int, int],
    *,
    sigma_px: float,
    gaia_db: str | None,
    cat: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[SolveResult | None, str]:
    """Decision on a verified hypothesis, shared by the hinted and the blind solver (G03-04).

    Refined against ``cat`` (ra, dec) when given (the hinted cone), else against the Gaia stars
    of its own footprint; with ``gaia_db`` then re-fitted against a Gaia catalogue cut to the
    field and deep enough for all detections (:func:`_final_refine`). The result is a solution
    only when it passes :func:`credible_solution`; otherwise ``(None, reason)`` and the caller
    keeps searching."""
    if cat is not None:
        res = refine_wcs(w, det_xy, cat[0], cat[1], shape, sigma_px=sigma_px)
    elif gaia_db:
        scale = float(np.sqrt(abs(np.linalg.det(w.pixel_scale_matrix)))) * 3600.0
        res = refine_with_gaia(w, scale, det_xy, shape, gaia_db=gaia_db, sigma=sigma_px)
    else:
        raise ValueError("accept_hypothesis needs a catalogue or a Gaia DB")
    if res.ok and gaia_db:
        res = _final_refine(res, det_xy, shape, gaia_db=gaia_db, sigma=sigma_px)
    ok, why = credible_solution(res, len(det_xy), sigma_px)
    return (res, why) if ok else (None, why)


# ---------------------------------------------------------------------------
# image detections
# ---------------------------------------------------------------------------
def detect_stars(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Detections for solving: (xy (n, 2) 0-based, flux, FWHM0). Same data-derived
    primitives as frame QC (frame_metrics): neighbour-difference noise, FWHM0 from
    half-maximum radii, DAO at 5 sigma."""
    from photutils.detection import DAOStarFinder  # noqa: PLC0415

    import frame_metrics as fm  # noqa: PLC0415

    a = np.asarray(img, dtype=np.float64)
    sigma = fm.pixel_noise_sigma(a)
    if sigma is None:
        return np.zeros((0, 2)), np.zeros(0), None
    sky = float(np.nanmedian(a))
    fwhm0 = fm.first_fwhm_guess(a, sky, sigma)
    if fwhm0 is None:
        return np.zeros((0, 2)), np.zeros(0), None
    from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER  # noqa: PLC0415

    tbl = DAOStarFinder(fwhm=float(fwhm0), threshold=fm.DETECT_NSIGMA * sigma,
                        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER)(np.nan_to_num(a - sky, nan=0.0))
    if tbl is None or len(tbl) == 0:
        return np.zeros((0, 2)), np.zeros(0), fwhm0
    cx = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
    cy = "y_centroid" if "y_centroid" in tbl.colnames else "ycentroid"
    xy = np.column_stack([np.asarray(tbl[cx], float), np.asarray(tbl[cy], float)])
    flux = np.asarray(tbl["flux"], float)
    ok = np.isfinite(xy).all(axis=1) & np.isfinite(flux) & (flux > 0)
    return xy[ok], flux[ok], fwhm0


# ---------------------------------------------------------------------------
# Gaia (local VYVAR DB)
# ---------------------------------------------------------------------------
def query_gaia_cone(db_path: str, ra0: float, dec0: float, radius: float, *, want: int = 3000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The brightest ``want`` Gaia stars (all when fewer) within the great-circle cone of
    ``radius`` deg about (ra0, dec0), brightest first.

    RA box from ``catalog_db.ra_boxes_for_cone`` (G03-05: exact cos dec at any Dec, RA wrap,
    poles). G03-45: no fixed G ladder (it capped the catalogue at G 17.5 and re-ran the SQL per
    step): one ``ORDER BY g_mag LIMIT n`` read per RA interval, n = ``want`` x box area / cone
    area, doubled until the cone holds ``want`` stars or the box is exhausted. The n brightest of
    the box contain every cone star brighter than the n-th, so the result is exactly the
    ``want`` brightest of the cone."""
    from catalog_db import connect_ro, ra_boxes_for_cone  # noqa: PLC0415

    want = max(1, int(want))
    ra_boxes, dmin, dmax = ra_boxes_for_cone(ra0, dec0, radius)
    r = math.radians(max(float(radius), 0.0))
    cone_sr = 2.0 * math.pi * (1.0 - math.cos(r))
    box_sr = sum(math.radians(b - a) for a, b in ra_boxes) * (math.sin(math.radians(dmax)) - math.sin(math.radians(dmin)))
    n = int(math.ceil(want * max(1.0, box_sr / cone_sr))) if cone_sr > 0 else want
    # unary + keeps SQLite on the (ra, dec) index (the g_mag index would scan the whole sky)
    sql = ("SELECT ra, dec, g_mag FROM gaia_dr3 WHERE ra BETWEEN ? AND ? AND dec BETWEEN ? AND ? "
           "AND g_mag IS NOT NULL ORDER BY +g_mag LIMIT ?")
    con = connect_ro(db_path)
    try:
        while True:
            parts = [np.asarray(con.execute(sql, (a, b, dmin, dmax, n)).fetchall(), dtype=np.float64).reshape(-1, 3)
                     for a, b in ra_boxes]
            exhausted = all(len(p) < n for p in parts)
            arr = np.vstack(parts)
            arr = arr[np.argsort(arr[:, 2], kind="stable")][:n]
            if len(arr):
                arr = arr[_angular_sep_deg(arr[:, 0], arr[:, 1], ra0, dec0) <= float(radius)]
            if len(arr) >= want or exhausted:
                break
            n *= 2
    finally:
        con.close()
    if not len(arr):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    arr = arr[:want]
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _angular_sep_deg(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> np.ndarray:
    r, d = np.radians(ra), np.radians(dec)
    r0, d0 = math.radians(ra0), math.radians(dec0)
    c = np.clip(math.sin(d0) * np.sin(d) + math.cos(d0) * np.cos(d) * np.cos(r - r0), -1.0, 1.0)
    return np.degrees(np.arccos(c))


# ---------------------------------------------------------------------------
# hinted solve (position known, scale optional)
# ---------------------------------------------------------------------------
def cone_ladder(diag_px: float, scale_hint_arcsec: float | None) -> list[float]:
    """Catalog cone radii (deg) to try. With a scale hint the matching radius comes
    first; then a x1.5 ladder from 4' to 15 deg."""
    ladder = [0.067 * 1.5 ** k for k in range(14)]  # 4' .. ~13 deg
    if scale_hint_arcsec and scale_hint_arcsec > 0:
        r0 = 0.75 * diag_px * scale_hint_arcsec / 3600.0
        ladder.sort(key=lambda r: abs(math.log(r / r0)))
    return ladder


def refine_with_gaia(w: Any, scale_arcsec: float, det_xy: np.ndarray, shape: tuple[int, int], *,
                     gaia_db: str, sigma: float) -> SolveResult:
    """Refine a WCS against the Gaia stars of its own footprint, deep enough for all
    detections."""
    ny, nx = shape
    c = w.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
    rad = 0.55 * math.hypot(nx, ny) * float(scale_arcsec) / 3600.0
    ra, dec, mag = query_gaia_cone(gaia_db, float(c.ra.deg), float(c.dec.deg), rad, want=max(3000, 3 * len(det_xy)))
    if len(ra) < 10:
        return SolveResult(False, wcs=w, message=f"only {len(ra)} Gaia stars in the footprint")
    return refine_wcs(w, det_xy, ra, dec, shape, sigma_px=sigma)


def _final_refine(res: SolveResult, det_xy: np.ndarray, shape: tuple[int, int], *,
                  gaia_db: str, sigma: float) -> SolveResult:
    """Re-fit with a catalog cut to this field and deep enough for all detections."""
    r2 = refine_with_gaia(res.wcs, float(res.scale_arcsec), det_xy, shape, gaia_db=gaia_db, sigma=sigma)
    if r2.ok and r2.n_match >= res.n_match:
        r2.log_odds, r2.n_hypotheses = res.log_odds, res.n_hypotheses
        return r2
    return res


def solve_hinted(
    det_xy: np.ndarray,
    det_flux: np.ndarray,
    shape: tuple[int, int],
    *,
    gaia_db: str,
    ra_hint: float,
    dec_hint: float,
    scale_hint_arcsec: float | None = None,
    fwhm_px: float | None = None,
    max_cones: int | None = None,
) -> SolveResult:
    """Solve with a position hint; the scale hint only orders the cone ladder."""
    ny, nx = shape
    diag = math.hypot(nx, ny)
    sigma = max(1.0, 0.5 * (fwhm_px or 2.0))
    tried: list[dict[str, Any]] = []
    for r in cone_ladder(diag, scale_hint_arcsec)[: max_cones or None]:
        ra, dec, mag = query_gaia_cone(gaia_db, ra_hint, dec_hint, r)
        if len(ra) < 10:
            tried.append({"radius_deg": r, "n_cat": int(len(ra)), "status": "few catalog stars"})
            continue
        res = solve_against_catalog(det_xy, det_flux, shape, ra, dec, mag, ra0=ra_hint, dec0=dec_hint,
                                    sigma_px=sigma, gaia_db=gaia_db)
        tried.append({"radius_deg": r, "n_cat": int(len(ra)), "status": res.message,
                      "log_odds": res.log_odds, "hyp": res.n_hypotheses})
        if res.ok:  # refined and credible (accept_hypothesis)
            res.extra["cones"] = tried
            res.extra["cone_radius_deg"] = r
            return res
    out = SolveResult(False, message="no cone gave a verified solution")
    out.extra["cones"] = tried
    return out


# ---------------------------------------------------------------------------
# FITS entry point (MASTERSTAR and any frame)
# ---------------------------------------------------------------------------
def header_scale_hint(hdr: Any) -> float | None:
    """arcsec/px from the header focal length and pixel (optics module, G03-19; only orders the
    cone ladder - a wrong value costs time, never the result)."""
    from optics import optics_hint  # noqa: PLC0415

    return optics_hint(hdr).arcsec_per_px


def default_index_dir(gaia_db: str | None, configured: str | None = None) -> str | None:
    from pathlib import Path  # noqa: PLC0415

    if configured:
        return configured
    if gaia_db:
        return str(Path(gaia_db).parent / "vyvar_astro_index")
    return None


def solve_fits(
    path: Any,
    *,
    gaia_db: str,
    index_dir: str | None = None,
    hint_ra_deg: float | None = None,
    hint_dec_deg: float | None = None,
    scale_hint_arcsec: float | None = None,
    write: bool = True,
    log: Any = None,
) -> dict[str, Any]:
    """Plate-solve a FITS image: hinted (position from the caller or header), then blind
    with the index. On success the WCS (TAN or TAN-SIP) replaces any WCS in the primary
    header together with VY_SOL* provenance keys. Returns a summary dict."""
    from pathlib import Path  # noqa: PLC0415

    from astropy.io import fits  # noqa: PLC0415

    _log = log or (lambda _m: None)
    p = Path(path)
    with fits.open(p, memmap=False) as hd:
        data = np.asarray(hd[0].data, dtype=np.float32)
        hdr = hd[0].header.copy()
    xy, flux, fwhm = detect_stars(data)
    out: dict[str, Any] = {"solved": False, "n_detections": int(len(xy)), "fwhm0_px": fwhm}
    if len(xy) < 8:
        out["message"] = f"only {len(xy)} detections"
        return out
    if scale_hint_arcsec is None:
        scale_hint_arcsec = header_scale_hint(hdr)
    if hint_ra_deg is None or hint_dec_deg is None:
        try:
            from fits_meta import pointing_hint_from_header  # noqa: PLC0415

            hra, hde, _src = pointing_hint_from_header(hdr)
            if hra is not None and hde is not None:
                hint_ra_deg, hint_dec_deg = float(hra), float(hde)
        except (ImportError, TypeError, ValueError):
            pass
    res: SolveResult | None = None
    method = ""
    if hint_ra_deg is not None and hint_dec_deg is not None:
        _log(f"PLATESOLVE: hinted at RA={hint_ra_deg:.4f} Dec={hint_dec_deg:.4f} "
             f"(scale hint {scale_hint_arcsec and round(scale_hint_arcsec, 4)} \"/px), {len(xy)} detections")
        res = solve_hinted(xy, flux, data.shape, gaia_db=gaia_db, ra_hint=hint_ra_deg, dec_hint=hint_dec_deg,
                           scale_hint_arcsec=scale_hint_arcsec, fwhm_px=fwhm)
        method = "hinted"
    if (res is None or not res.ok) and index_dir and Path(index_dir).is_dir():
        _log("PLATESOLVE: blind (index) - no usable position or hinted solve failed")
        from astro_index import solve_blind  # noqa: PLC0415

        res = solve_blind(xy, flux, data.shape, index_dir=index_dir, gaia_db=gaia_db, fwhm_px=fwhm)
        method = "blind"
    if res is None or not res.ok:
        out["message"] = (res.message if res is not None else "no position hint and no blind index")
        out["cones"] = (res.extra.get("cones") if res is not None else None)
        return out
    out.update({
        "solved": True, "method": method, "n_match": int(res.n_match), "rms_px": res.rms_px,
        "log_odds": float(res.log_odds), "sip_order": res.sip_order, "mirrored": res.mirrored,
        "plate_scale_arcsec_px": res.scale_arcsec, "rotation_deg": res.rotation_deg,
        "match_rate": float(res.n_match) / float(len(xy)), "n_hypotheses": res.n_hypotheses,
    })
    _log(f"PLATESOLVE: solved ({method}) scale={res.scale_arcsec:.4f}\"/px rot={res.rotation_deg:.2f} "
         f"mirrored={res.mirrored} matches={res.n_match} rms={res.rms_px:.3f} px SIP={res.sip_order} "
         f"log-odds={res.log_odds:.0f}")
    if write:
        _write_solution(p, res, method)
    return out


_SOLUTION_TIME_KEYS = ("MJDREF", "MJDREFI", "MJDREFF", "DATEREF", "DATE-OBS", "MJD-OBS", "DATE-AVG", "MJD-AVG")


def _write_solution(path: Any, res: SolveResult, method: str) -> None:
    """Replace the primary-header WCS with ``res.wcs`` plus the VY_SOL* provenance keys."""
    from astropy.io import fits  # noqa: PLC0415

    from utils import strip_celestial_wcs_keys  # noqa: PLC0415  (G03-21: the one WCS key list)

    with fits.open(path, mode="update") as hd:
        h = hd[0].header
        strip_celestial_wcs_keys(h)
        sol = res.wcs.to_header(relax=True)
        for k in _SOLUTION_TIME_KEYS:  # time keys of the frame stay (astropy writes MJDREF = 0.0)
            if k in h and k in sol:
                del sol[k]
        h.update(sol)
        h["VY_SOLVR"] = ("astrosolve", "VYVAR plate solver (PLATESOLVE-2026-10)")
        h["VY_SOLMD"] = (method, "hinted | blind | confirmed")
        h["VY_SOLLO"] = (round(float(res.log_odds), 2), "verification log-odds")
        h["VY_SOLNM"] = (int(res.n_match), "matched Gaia stars")
        h["VY_SOLRM"] = (round(float(res.rms_px or 0.0), 4), "fit rms [px]")
        h["VY_PLTS"] = (round(float(res.scale_arcsec), 6), "VYVAR plate scale arcsec/px")
        hd.flush()


def confirm_fits(path: Any, *, gaia_db: str, write: bool = True, log: Any = None) -> dict[str, Any]:
    """Verify and refine a WCS that is already in the header (WCS-CONFIRM-2026-10).

    A WCS handed over from another MASTERSTAR (sibling filter, OSC channel) is a hypothesis
    like any other: it is accepted only when the Bayesian log-odds against the Gaia stars of
    its own footprint reach ``LOG_ODDS_ACCEPT``, then refined (1:1 matching, TAN+SIP by BIC)
    exactly as a solved WCS. A rejected WCS is not written and the caller must stop - an
    unverified WCS would give wrong catalogue identities."""
    from pathlib import Path  # noqa: PLC0415

    from astropy.io import fits  # noqa: PLC0415
    from astropy.wcs import WCS  # noqa: PLC0415
    from astropy.wcs.utils import proj_plane_pixel_scales  # noqa: PLC0415

    _log = log or (lambda _m: None)
    p = Path(path)
    with fits.open(p, memmap=False) as hd:
        data = np.asarray(hd[0].data, dtype=np.float32)
        hdr = hd[0].header.copy()
    out: dict[str, Any] = {"solved": False, "method": "confirmed"}
    w = WCS(hdr)
    if not getattr(w, "has_celestial", False):
        out["message"] = "no WCS in header"
        return out
    w = w.celestial
    xy, flux, fwhm = detect_stars(data)
    out.update({"n_detections": int(len(xy)), "fwhm0_px": fwhm})
    if len(xy) < 8:
        out["message"] = f"only {len(xy)} detections"
        return out
    ny, nx = data.shape
    sigma = max(1.0, 0.5 * (fwhm or 2.0))
    scale = float(np.mean(proj_plane_pixel_scales(w))) * 3600.0
    c = w.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
    rad = 0.55 * math.hypot(nx, ny) * scale / 3600.0
    ra, dec, mag = query_gaia_cone(gaia_db, float(c.ra.deg), float(c.dec.deg), rad, want=max(3000, 3 * len(xy)))
    if len(ra) < 10:
        out["message"] = f"only {len(ra)} Gaia stars in the WCS footprint"
        return out
    vxy = xy[np.argsort(-flux)[:150]]
    lo, nm, w_v = verify_hypothesis(w, vxy, ra, dec, mag, data.shape, sigma_px=sigma)
    out.update({"log_odds": float(lo), "n_match": int(nm)})
    if not (math.isfinite(lo) and lo >= LOG_ODDS_ACCEPT):
        out["message"] = f"handed-over WCS rejected: log-odds {lo:.1f} < {LOG_ODDS_ACCEPT:.1f}"
        _log(f"PLATESOLVE confirm: {out['message']}")
        return out
    res = refine_wcs(w_v, xy, ra, dec, data.shape, sigma_px=sigma)
    ok, why = credible_solution(res, len(xy), sigma)
    if not ok:
        out["message"] = why
        _log(f"PLATESOLVE confirm: {out['message']}")
        return out
    res.log_odds = float(lo)
    out.update({
        "solved": True, "n_match": int(res.n_match), "rms_px": res.rms_px, "sip_order": res.sip_order,
        "plate_scale_arcsec_px": res.scale_arcsec, "rotation_deg": res.rotation_deg, "mirrored": res.mirrored,
        "match_rate": float(res.n_match) / float(len(xy)),
    })
    _log(f"PLATESOLVE: confirmed handed-over WCS scale={res.scale_arcsec:.4f}\"/px rot={res.rotation_deg:.2f} "
         f"matches={res.n_match} rms={res.rms_px:.3f} px SIP={res.sip_order} log-odds={lo:.0f}")
    if write:
        _write_solution(p, res, "confirmed")
    return out
