"""Frame QC per channel (QC-PER-GROUP-2026-10 + QC-METRICS-2026-10, Milan 2026-10-09).

One decision for the GUI and the night runner, computed separately for every
observation group (FILTER|EXPTIME|BINNING, one "channel"); channels are never pooled.
Per-frame metrics come from ``frame_metrics.measure_frame`` (calibration step).

* **Transparency** (relative zero point): every frame's star list is matched to the
  group's provisional reference (translation by pair voting, 180 deg rotation for a
  meridian flip), dZP = median(-2.5 log10(F_frame / F_ref)) over common stars,
  non-uniformity = max - min of the median dm in the 4 field quadrants, T = 10**(-0.4 (dZP -
  dZP_min)) relative to the clearest kept frame of the group (T = 1 there).
* **Reject only real failures** (one pass, no sigma clipping; limit = median +
  k * 1.4826 * MAD of the group, k = ``auto_fwhm_k_factor``, default 3):
  - no measurable FWHM (no usable stars),
  - elongation above the group limit (tracking / wind),
  - fewer than ``MIN_COMMON_STARS`` stars in common with the reference,
  - transparency loss: dZP above the group limit (one-sided),
  - non-uniform transparency above the group limit (patchy cloud).
  A wider FWHM alone is NOT a reason: the aperture scales with FWHM and the lower
  S/N is carried by the photometric errors.
  No limit for a group with fewer than ``MIN_FRAMES_FOR_LIMIT`` values or MAD = 0.
* **Best frame** (MASTERSTAR source) among the kept frames: max T / (FWHM * sigma_pix),
  the background-limited S/N of a faint point source (aperture ~ FWHM^2, measured
  pixel noise sigma_pix includes sky, read and dark noise). Provisional reference
  for dZP = max 1 / (FWHM * sigma_pix); the ranking does not depend on that choice.

Results: manifest (``inspection.is_rejected`` + ``qc_reason``) and ``frame_qc.json``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from infolog import log_event

MAD_TO_SIGMA = 1.4826
FRAME_QC_JSON = "frame_qc.json"
QC_STARS_JSON = Path("qc") / "qc_stars.json"
SPEC = "QC-PER-GROUP-2026-10 / QC-METRICS-2026-10"

#: Below this many measured frames the spread of a group cannot be estimated; no limit.
MIN_FRAMES_FOR_LIMIT = 5
#: Minimum common stars with the reference to measure dZP (median of >= 5 ratios).
MIN_COMMON_STARS = 5
N_MATCH_BRIGHT = 40

#: QC-FLOORS-2026-10 (Milan 2026-10-12): a frame is rejected only when its metric is above the
#: group limit AND above a physical floor - a tight group (good night) no longer rejects
#: usable frames (draft 526 S4 Cyg: elongation 1.205 > limit 1.198; Muniwin used those frames,
#: their scatter was 1.0-1.7x, errors 1.35x the rest). The weaker S/N is carried by the errors.
ELONG_FLOOR = 1.30
DZP_FLOOR_MAG = 1.0
PATCHY_FLOOR_MAG = 0.20


def _with_floor(limit: float | None, floor: float) -> float | None:
    return None if limit is None else max(float(limit), float(floor))


REASON_NO_FWHM = "no measurable FWHM"
REASON_ELONG = "elongation above group limit"
REASON_FEW_COMMON = "too few stars in common with the reference"
REASON_TRANSP = "transparency loss"
REASON_PATCHY = "non-uniform transparency"


@dataclass
class GroupQC:
    group_key: str
    setup: str
    n_frames: int
    fwhm_median: float | None
    fwhm_mad: float | None
    fwhm_limit: float | None  # no FWHM rejection any more; kept for display (None)
    elong_median: float | None
    elong_limit: float | None
    dzp_limit: float | None = None
    patchy_limit: float | None = None
    reference_path: str | None = None
    n_rejected: int = 0
    best_path: str | None = None
    best_fwhm: float | None = None
    best_sky: float | None = None
    best_noise: float | None = None
    best_transparency: float | None = None
    best_depth: float | None = None
    rejected: dict[str, str] = field(default_factory=dict)  # FILE_PATH -> reason
    frames: dict[str, dict[str, Any]] = field(default_factory=dict)  # FILE_PATH -> dzp, patchy, n_common

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def setup_name_for_group(group_key: str) -> str:
    """Folder name of a group under ``lights/`` (same rule as the importer)."""
    from fits_meta import observation_group_folder_name  # noqa: PLC0415

    return observation_group_folder_name(str(group_key))


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _robust_limit(vals: Sequence[float | None], k: float) -> tuple[float | None, float | None, float | None]:
    """(median, MAD, median + k*1.4826*MAD) - one pass, no clipping; limit None for
    fewer than ``MIN_FRAMES_FOR_LIMIT`` values or MAD = 0."""
    a = np.asarray([v for v in vals if v is not None and math.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return None, None, None
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if a.size < MIN_FRAMES_FOR_LIMIT or mad <= 0.0:
        return med, mad, None
    return med, mad, med + float(k) * MAD_TO_SIGMA * mad


def depth_score(fwhm: float | None, noise: float | None, transparency: float | None = 1.0) -> float | None:
    """T / (FWHM * sigma_pix); None when not defined."""
    if fwhm is None or noise is None or fwhm <= 0 or noise <= 0:
        return None
    t = 1.0 if transparency is None else float(transparency)
    return t / (fwhm * noise)


def group_key_of(row: dict[str, Any]) -> str:
    gk = str(row.get("OBSERVATION_GROUP_KEY") or "").strip()
    if gk:
        return gk
    flt = str(row.get("FILTER") or "NoFilter").strip() or "NoFilter"
    exp = _finite(row.get("EXPTIME"))
    return f"{flt}|{exp:g}|?" if exp is not None else f"{flt}|?|?"


# ---------------------------------------------------------------------------
# star lists (written at calibration time, read here)
# ---------------------------------------------------------------------------
def qc_stars_path(ap_root: Path | str) -> Path:
    """Draft-level star lists of the QC crops: ``<draft>/qc/qc_stars.json`` (OUTPUT-REVIEW-2026-10:
    one file keyed by FILE_PATH instead of one ``qc/stars/<frame>_<hash>.json`` per frame)."""
    return Path(ap_root) / QC_STARS_JSON


def save_qc_stars(ap_root: Path | str, lists: dict[str, tuple[Sequence[Sequence[float]], Any]]) -> None:
    """Merge ``{file_path: (stars (x, y, flux), shape)}`` into the draft's star-list file."""
    if not lists:
        return
    p = qc_stars_path(ap_root)
    data: dict[str, Any] = {}
    if p.is_file():
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            data = {}
    for fp, (stars, shape) in lists.items():
        data[str(fp)] = {"shape": list(shape) if shape is not None else None,
                         "stars": [[round(float(v), 3) for v in s] for s in stars]}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")


def load_qc_stars(ap_root: Path | str) -> dict[str, tuple[np.ndarray, tuple[int, int] | None]]:
    """``{file_path: (stars (n, 3), shape)}`` of the draft; empty when there is no file."""
    p = qc_stars_path(ap_root)
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, tuple[np.ndarray, tuple[int, int] | None]] = {}
    for fp, rec in (d.items() if isinstance(d, dict) else []):
        a = np.asarray(rec.get("stars") or [], dtype=np.float64).reshape(-1, 3)
        shp = rec.get("shape")
        out[str(fp)] = (a, tuple(int(x) for x in shp[:2]) if shp else None)
    return out


def match_to_reference(
    ref: np.ndarray,
    cur: np.ndarray,
    *,
    tol: float,
    shape: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Common stars of two star lists (x, y, flux). Translation by voting on pair
    differences of the brightest stars; tries a 180 deg rotation (meridian flip).
    Returns (flux_ref, flux_cur, xy_ref) of matched pairs, or None."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    if ref.shape[0] < MIN_COMMON_STARS or cur.shape[0] < MIN_COMMON_STARS:
        return None
    rb = ref[np.argsort(-ref[:, 2])][:N_MATCH_BRIGHT, :2]
    best: tuple[int, np.ndarray, bool] | None = None
    for rot in (False, True):
        if rot and shape is None:
            continue
        c = cur[:, :2].copy()
        if rot:
            c[:, 0] = (shape[1] - 1) - c[:, 0]
            c[:, 1] = (shape[0] - 1) - c[:, 1]
        cb = c[np.argsort(-cur[:, 2])][:N_MATCH_BRIGHT]
        d = (rb[:, None, :] - cb[None, :, :]).reshape(-1, 2)
        t = cKDTree(d)
        cnt = np.array([len(x) for x in t.query_ball_point(d, r=tol)])
        i = int(np.argmax(cnt))
        if best is None or cnt[i] > best[0]:
            near = t.query_ball_point(d[i], r=tol)
            best = (int(cnt[i]), np.median(d[near], axis=0), rot)
    if best is None or best[0] < MIN_COMMON_STARS:
        return None
    _, off, rot = best
    c = cur[:, :2].copy()
    if rot:
        c[:, 0] = (shape[1] - 1) - c[:, 0]
        c[:, 1] = (shape[0] - 1) - c[:, 1]
    c = c + off
    dist, idx = cKDTree(ref[:, :2]).query(c, distance_upper_bound=tol)
    ok = np.isfinite(dist)
    if ok.sum() < MIN_COMMON_STARS:
        return None
    # one-to-one: keep the closest current star per reference star
    pairs: dict[int, tuple[float, int]] = {}
    for j in np.where(ok)[0]:
        r = int(idx[j])
        if r not in pairs or dist[j] < pairs[r][0]:
            pairs[r] = (float(dist[j]), int(j))
    ri = np.array(list(pairs.keys()))
    ci = np.array([v[1] for v in pairs.values()])
    return ref[ri, 2], cur[ci, 2], ref[ri, :2]


def transparency_vs_reference(
    ref: np.ndarray, cur: np.ndarray, *, tol: float, shape: tuple[int, int] | None,
) -> tuple[float | None, float | None, int]:
    """(dZP mag, non-uniformity mag, n_common). dZP > 0 means the frame is fainter."""
    m = match_to_reference(ref, cur, tol=tol, shape=shape)
    if m is None:
        return None, None, 0
    fr, fc, xy = m
    good = (fr > 0) & (fc > 0)
    if good.sum() < MIN_COMMON_STARS:
        return None, None, int(good.sum())
    dm = -2.5 * np.log10(fc[good] / fr[good])
    xy = xy[good]
    dzp = float(np.median(dm))
    # Non-uniformity = spread of the field quadrants' median dm (max - min). Medians of
    # many stars average the photon noise of faint stars out; a patchy cloud makes the
    # quadrants differ. Quadrants split at the median star position.
    mx, my = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
    q_med = []
    for sx in (xy[:, 0] < mx, xy[:, 0] >= mx):
        for sy in (xy[:, 1] < my, xy[:, 1] >= my):
            sel = sx & sy
            if sel.sum() >= MIN_COMMON_STARS:
                q_med.append(float(np.median(dm[sel])))
    patchy = float(max(q_med) - min(q_med)) if len(q_med) >= 2 else None
    return dzp, patchy, int(good.sum())


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate_frame_qc(
    rows: Sequence[dict[str, Any]],
    *,
    k: float,
    reject: bool = True,
    stars: dict[str, tuple[np.ndarray, tuple[int, int] | None]] | None = None,
) -> dict[str, GroupQC]:
    """Per-group transparency, rejections and best frame. ``stars``: FILE_PATH ->
    (star list, crop shape); without it transparency is not evaluated."""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(group_key_of(r), []).append(r)
    out: dict[str, GroupQC] = {}
    for gk in sorted(by):
        grp = by[gk]
        paths = [str(r.get("FILE_PATH") or "") for r in grp]
        fw = {p: _finite(r.get("FWHM")) for p, r in zip(paths, grp)}
        el = {p: _finite(r.get("ELONGATION_MEAN")) for p, r in zip(paths, grp)}
        sky = {p: _finite(r.get("SKY_LEVEL")) for p, r in zip(paths, grp)}
        noise = {p: _finite(r.get("NOISE_SIGMA")) for p, r in zip(paths, grp)}
        # frames processed before QC-METRICS have no measured noise: fall back to sqrt(sky)
        use_noise = all(noise[p] is not None for p in paths if fw[p] is not None)
        sig = {p: (noise[p] if use_noise else (math.sqrt(sky[p]) if sky[p] and sky[p] > 0 else None)) for p in paths}

        f_med, f_mad, _ = _robust_limit([v for v in fw.values() if v is not None and v > 0], k)
        e_med, _e_mad, e_lim = _robust_limit(list(el.values()), k)
        e_lim = _with_floor(e_lim, ELONG_FLOOR)
        g = GroupQC(gk, setup_name_for_group(gk), len(grp), f_med, f_mad, None, e_med, e_lim if reject else None)

        # provisional reference: max 1/(FWHM*sigma) among frames with stars and normal elongation
        def _shape_ok(p: str) -> bool:
            return fw[p] is not None and fw[p] > 0 and (e_lim is None or el[p] is None or el[p] <= e_lim)

        ref_p = max((p for p in paths if _shape_ok(p) and depth_score(fw[p], sig[p]) is not None),
                    key=lambda p: (depth_score(fw[p], sig[p]), p), default=None)
        g.reference_path = ref_p
        have_stars = bool(stars) and ref_p is not None and ref_p in (stars or {})
        tol = max(1.5, float(f_med or 2.0))
        dzp: dict[str, float | None] = {}
        patchy: dict[str, float | None] = {}
        ncom: dict[str, int] = {}
        if have_stars:
            ref_list, ref_shape = stars[ref_p]  # type: ignore[index]
            for p in paths:
                s = (stars or {}).get(p)
                if s is None:
                    dzp[p], patchy[p], ncom[p] = None, None, 0
                    continue
                dzp[p], patchy[p], ncom[p] = transparency_vs_reference(
                    ref_list, s[0], tol=tol, shape=s[1] or ref_shape)
                g.frames[p] = {"dzp_mag": dzp[p], "patchy_mag": patchy[p], "n_common": ncom[p]}
            _, _, g.dzp_limit = _robust_limit([v for v in dzp.values() if v is not None], k)
            _, _, g.patchy_limit = _robust_limit([v for v in patchy.values() if v is not None], k)
            g.dzp_limit = _with_floor(g.dzp_limit, DZP_FLOOR_MAG)
            g.patchy_limit = _with_floor(g.patchy_limit, PATCHY_FLOOR_MAG)
            if not reject:
                g.dzp_limit = g.patchy_limit = None

        if reject:
            for p in paths:
                if fw[p] is None or fw[p] <= 0:
                    g.rejected[p] = REASON_NO_FWHM
                elif e_lim is not None and el[p] is not None and el[p] > e_lim:
                    g.rejected[p] = f"{REASON_ELONG} ({el[p]:.3f} > {e_lim:.3f})"
                elif have_stars and ncom.get(p, 0) < MIN_COMMON_STARS:
                    g.rejected[p] = f"{REASON_FEW_COMMON} ({ncom.get(p, 0)})"
                elif have_stars and g.dzp_limit is not None and (dzp.get(p) or 0.0) > g.dzp_limit:
                    g.rejected[p] = f"{REASON_TRANSP} (dZP {dzp[p]:.3f} > {g.dzp_limit:.3f} mag)"
                elif have_stars and g.patchy_limit is not None and (patchy.get(p) or 0.0) > g.patchy_limit:
                    g.rejected[p] = f"{REASON_PATCHY} ({patchy[p]:.3f} > {g.patchy_limit:.3f} mag)"
        g.n_rejected = len(g.rejected)

        # transparency relative to the clearest kept frame (T = 1 there, < 1 elsewhere)
        kept_dzp = [dzp[p] for p in paths if p not in g.rejected and dzp.get(p) is not None]
        dzp0 = min(kept_dzp) if kept_dzp else 0.0

        def _t(p: str) -> float | None:
            return 10 ** (-0.4 * (dzp[p] - dzp0)) if have_stars and dzp.get(p) is not None else None

        for p, fr in g.frames.items():
            fr["transparency"] = _t(p)
        best: tuple[float, str] | None = None
        for p in paths:
            if p in g.rejected:
                continue
            t = _t(p) or 1.0
            d = depth_score(fw[p], sig[p], t)
            key = (d if d is not None else -1.0, p)
            if best is None or key > best:
                best = key
        if best is not None:
            p = best[1]
            g.best_path, g.best_fwhm, g.best_sky, g.best_noise = p, fw[p], sky[p], sig[p]
            g.best_transparency = _t(p)
            g.best_depth = best[0] if best[0] >= 0 else None
        out[gk] = g
    return out


def write_frame_qc(ap_root: Path | str, groups: dict[str, GroupQC], *, k: float, reject: bool) -> Path:
    p = Path(ap_root) / FRAME_QC_JSON
    doc = {
        "spec": SPEC,
        "k": float(k),
        "reject_enabled": bool(reject),
        "rule": "per group, one pass: reject no FWHM | elongation, dZP, non-uniformity > max(med+k*1.4826*MAD, "
                f"floor {ELONG_FLOOR} / {DZP_FLOOR_MAG} mag / {PATCHY_FLOOR_MAG} mag) | "
                f"< {MIN_COMMON_STARS} common stars; best: max T/(FWHM*sigma_pix)",
        "groups": {gk: g.to_dict() for gk, g in groups.items()},
    }
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=True), encoding="utf-8")
    return p


def load_frame_qc(ap_root: Path | str) -> dict[str, Any]:
    p = Path(ap_root) / FRAME_QC_JSON
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def best_path_for_setup(ap_root: Path | str, setup_name: str | None) -> str | None:
    """Best frame of the group whose folder is ``setup_name`` (or of the only group)."""
    groups = (load_frame_qc(ap_root).get("groups") or {})
    if not groups:
        return None
    if setup_name:
        for g in groups.values():
            if str(g.get("setup") or "") == str(setup_name):
                return g.get("best_path") or None
        return None
    if len(groups) == 1:
        return next(iter(groups.values())).get("best_path") or None
    return None


def run_frame_qc(db: Any, draft_id: int, ap_root: Path | str, *, k: float, reject: bool = True) -> dict[str, GroupQC]:
    """Evaluate the draft, persist the decision (manifest + frame_qc.json), log it."""
    from draft_provenance import set_manifest_light_qc_decisions  # noqa: PLC0415

    rows = db.fetch_draft_light_rows_for_quality(int(draft_id)) or []
    _all = load_qc_stars(ap_root)
    stars = {str(r.get("FILE_PATH") or ""): _all[str(r.get("FILE_PATH") or "")]
             for r in rows if str(r.get("FILE_PATH") or "") in _all}
    groups = evaluate_frame_qc(rows, k=float(k), reject=reject, stars=stars or None)
    decisions: dict[str, str | None] = {}
    for r in rows:
        path = str(r.get("FILE_PATH") or "")
        g = groups.get(group_key_of(r))
        decisions[path] = g.rejected.get(path) if g is not None else None
    set_manifest_light_qc_decisions(db, int(draft_id), decisions)
    write_frame_qc(ap_root, groups, k=float(k), reject=reject)
    for g in groups.values():
        log_event(
            f"FRAME QC [{g.group_key}] n={g.n_frames} rejected={g.n_rejected} "
            f"FWHM med={None if g.fwhm_median is None else round(g.fwhm_median, 3)} "
            f"elong limit={None if g.elong_limit is None else round(g.elong_limit, 3)} "
            f"dZP limit={None if g.dzp_limit is None else round(g.dzp_limit, 3)} "
            f"patchy limit={None if g.patchy_limit is None else round(g.patchy_limit, 3)} "
            f"best={Path(str(g.best_path).replace(chr(92), '/')).name if g.best_path else '-'} "
            f"(FWHM={g.best_fwhm}, sigma={g.best_noise}, T={g.best_transparency})"
        )
    return groups
