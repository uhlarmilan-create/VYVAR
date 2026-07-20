#!/usr/bin/env python3
"""Diagnostic: blind solver log_L3 clip vs index range on draft_000380 (Chi_and_H).

Read-only w.r.t. draft artefacts; writes report + histogram PNGs under --out.
"""
from __future__ import annotations

import argparse
import itertools
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from vyvar_blind_solver import (  # noqa: E402
    CLUSTER_RADIUS_DEG,
    find_blind_candidates,
    find_blind_hint,
    iter_local_knn_triangle_indices,
    _index_k_neighbors,
)
from vyvar_platesolver import _verify_blind_candidates  # noqa: E402

DEFAULT_DRAFT = _ROOT / "Archive" / "Drafts" / "draft_000380"
DEFAULT_TRUTH_RA = 35.03
DEFAULT_TRUTH_DEC = 57.14
DEFAULT_PLATE_SCALE = 1.3
HIT_THRESHOLD_DEG = 2.0


def _angular_sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = (ra1 - ra2) * math.cos(math.radians((dec1 + dec2) / 2.0))
    ddec = dec1 - dec2
    return math.sqrt(dra * dra + ddec * ddec)


def _discover_masterstars(draft: Path) -> list[tuple[str, Path]]:
    hits: list[tuple[str, Path]] = []
    for fp in sorted(draft.glob("platesolve/*/MASTERSTAR.fits")):
        group = fp.parent.name
        hits.append((group, fp))
    return hits


def _load_index_meta(index_path: Path) -> dict[str, Any]:
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    log_L3_min = float(data["log_L3_min"])
    log_L3_max = float(data["log_L3_max"])
    meta = data.get("metadata")
    n_tri = int(len(meta)) if meta is not None else 0
    return {
        "mag_limit": data.get("mag_limit"),
        "tolerance": data.get("tolerance"),
        "hash_dim": data.get("hash_dim"),
        "n_triangles": n_tri,
        "log_L3_min": log_L3_min,
        "log_L3_max": log_L3_max,
        "log_L3_min_arcsec": 10 ** log_L3_min,
        "log_L3_max_arcsec": 10 ** log_L3_max,
        "k_neighbors": data.get("k_neighbors"),
        "cell_deg": data.get("cell_deg"),
        "stars_per_cell": data.get("stars_per_cell"),
    }


def _detect_dao_stars(fits_path: Path) -> tuple[np.ndarray, pd.DataFrame, int, int]:
    with fits.open(fits_path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
    ny, nx = data.shape
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=3.0, threshold=5.0 * std, min_separation=0)
    srcs = finder(data - med)
    n_dao = 0 if srcs is None else len(srcs)
    if srcs is None or n_dao < 3:
        return data, pd.DataFrame(columns=["x", "y", "flux"]), n_dao, nx
    df = srcs.to_pandas().rename(columns={"x_centroid": "x", "y_centroid": "y"})
    if "peak" in df.columns and "flux" not in df.columns:
        df["flux"] = df["peak"]
    elif "flux" not in df.columns:
        df["flux"] = 1.0
    df = df.sort_values("flux", ascending=False)
    return data, df, n_dao, nx


def _fov_deg_from_fits(nx: int, ny: int, plate_scale: float) -> float:
    return max(nx, ny) * plate_scale / 3600.0


def _analyze_image_triangles(
    dao_stars: pd.DataFrame,
    *,
    log_L3_min: float,
    log_L3_max: float,
    plate_scale: float,
    fov_deg: float,
    img_budget: int = 80,
    tri_k: int = 8,
) -> dict[str, Any]:
    """Replicate blind-solver star pick + triangle log_L3 loop (shape/FOV/range clip)."""
    log_L3_range = max(log_L3_max - log_L3_min, 1e-6)
    if dao_stars.empty or not {"x", "y"}.issubset(dao_stars.columns):
        return {
            "n_central": 0,
            "n_stars_used": 0,
            "n_tried": 0,
            "n_passed_shape_fov": 0,
            "n_below_min": 0,
            "n_above_max": 0,
            "n_in_range": 0,
            "pct_in_range": 0.0,
            "log_L3_samples": [],
        }

    x_max = float(dao_stars["x"].max())
    y_max = float(dao_stars["y"].max())
    x_cen = x_max / 2.0
    y_cen = y_max / 2.0
    L3_max_arcsec = 10 ** float(log_L3_max)
    R_px = (L3_max_arcsec / plate_scale) / 2.0
    dist_from_center = np.sqrt((dao_stars["x"] - x_cen) ** 2 + (dao_stars["y"] - y_cen) ** 2)
    central_stars = dao_stars[dist_from_center <= R_px]
    n_central = len(central_stars)
    budget = max(3, int(img_budget))
    if n_central >= 6:
        stars = central_stars.head(budget)[["x", "y"]].to_numpy(dtype=np.float64)
    else:
        stars = dao_stars.head(budget)[["x", "y"]].to_numpy(dtype=np.float64)

    n_tried = 0
    n_passed = 0
    n_below_min = 0
    n_above_max = 0
    logl3_samples: list[float] = []

    for i0, i1, i2 in iter_local_knn_triangle_indices(stars, k_neighbors=tri_k):
        n_tried += 1
        p0, p1, p2 = stars[i0], stars[i1], stars[i2]
        d01 = math.hypot(p0[0] - p1[0], p0[1] - p1[1])
        d12 = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        d02 = math.hypot(p0[0] - p2[0], p0[1] - p2[1])
        L1, L2, L3 = sorted([d01, d12, d02])
        if L3 < 2.0 or L1 / L3 < 0.15:
            continue
        L3_arcsec = L3 * plate_scale
        if L3_arcsec < 0.1:
            continue
        if fov_deg > 0 and L3_arcsec > fov_deg * 3600.0 * 0.9:
            continue
        n_passed += 1
        log_L3 = math.log10(L3_arcsec)
        logl3_samples.append(log_L3)
        log_L3_norm = (log_L3 - log_L3_min) / log_L3_range
        if log_L3_norm < 0.0:
            n_below_min += 1
            continue
        if log_L3_norm > 1.0:
            n_above_max += 1

    n_in_range = n_passed - n_below_min - n_above_max
    arr = np.asarray(logl3_samples) if logl3_samples else np.array([])
    stats: dict[str, Any] = {
        "n_central": n_central,
        "R_px": R_px,
        "n_stars_used": len(stars),
        "img_budget": budget,
        "tri_k": tri_k,
        "n_tried": n_tried,
        "n_passed_shape_fov": n_passed,
        "n_below_min": n_below_min,
        "n_above_max": n_above_max,
        "n_in_range": n_in_range,
        "pct_in_range": 100.0 * n_in_range / max(n_passed, 1),
        "log_L3_samples": logl3_samples,
    }
    if arr.size:
        stats.update(
            {
                "log_L3_min": float(arr.min()),
                "log_L3_p10": float(np.percentile(arr, 10)),
                "log_L3_med": float(np.median(arr)),
                "log_L3_p90": float(np.percentile(arr, 90)),
                "log_L3_max": float(arr.max()),
            }
        )
    return stats


def _plot_histogram(
    *,
    logl3_samples: list[float],
    log_L3_min: float,
    log_L3_max: float,
    group: str,
    out_png: Path,
) -> None:
    if not logl3_samples:
        return
    arcsec = [10 ** x for x in logl3_samples]
    idx_lo = 10 ** log_L3_min
    idx_hi = 10 ** log_L3_max
    med = float(np.median(arcsec))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(arcsec, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvspan(idx_lo, idx_hi, color="green", alpha=0.15, label=f"index [{idx_lo:.1f}\", {idx_hi:.1f}\"]")
    ax.axvline(med, color="crimson", linestyle="--", label=f"image median {med:.1f}\"")
    ax.set_xlabel("L3 side length (arcsec)")
    ax.set_ylabel("triangle count")
    ax.set_title(f"Blind solver image log_L3 - {group}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _plot_sky_scatter(
    *,
    votes: np.ndarray,
    truth_ra: float,
    truth_dec: float,
    best_ra: float | None,
    best_dec: float | None,
    group: str,
    out_png: Path,
) -> None:
    if votes is None or len(votes) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(votes[:, 0], votes[:, 1], s=8, alpha=0.35, c="steelblue", label=f"votes ({len(votes)})")
    ax.scatter([truth_ra], [truth_dec], marker="*", s=180, c="lime", edgecolors="black", label="truth", zorder=5)
    if best_ra is not None and best_dec is not None:
        ax.scatter([best_ra], [best_dec], marker="x", s=120, c="red", linewidths=2, label="winner cluster", zorder=5)
    circle = plt.Circle(
        (truth_ra, truth_dec),
        CLUSTER_RADIUS_DEG,
        fill=False,
        linestyle="--",
        color="lime",
        linewidth=1.5,
        label=f"truth +-{CLUSTER_RADIUS_DEG} deg",
    )
    ax.add_patch(circle)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"Blind solver vote sky-scatter - {group}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _deciding_pass(sink: dict[str, Any]) -> dict[str, Any] | None:
    passes = sink.get("passes") or []
    if not passes:
        return None
    idx = sink.get("deciding_pass_idx")
    if idx is not None and 0 <= int(idx) < len(passes):
        return passes[int(idx)]
    return max(passes, key=lambda p: int(p.get("best_count", 0)))


def _format_pass_lines(passes: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for i, p in enumerate(passes):
        lines.append(
            f"  pass[{i}] dub={p.get('dub')} "
            f"votes={p.get('n_votes')} match_mult_mean={p.get('match_mult_mean', 0):.2f} "
            f"near_truth<2 deg={p.get('votes_near_truth_2deg', 'n/a')} "
            f"<5 deg={p.get('votes_near_truth_5deg', 'n/a')} "
            f"best_at_truth={p.get('best_count_at_truth', 'n/a')}"
        )
        lines.append(
            f"         best_cluster count={p.get('best_count')} "
            f"RA={p.get('best_ra')} Dec={p.get('best_dec')} sig={p.get('significance', 0):.1f}x"
        )
        if p.get("log_L3_med") is not None:
            lines.append(
                f"         log_L3 med={p['log_L3_med']:.3f} "
                f"in_range={p.get('n_in_range')}/{p.get('n_passed')} "
                f"below_min={p.get('n_below_min')} above_max={p.get('n_above_max')}"
            )
    return lines


def _votes_branch_verdict(group_sinks: list[dict[str, Any]]) -> str:
    total_near_5 = 0
    total_near_2 = 0
    max_match_mult = 0.0
    any_truth_votes = False
    any_winner_elsewhere = False

    for sink in group_sinks:
        for p in sink.get("passes") or []:
            near_5 = int(p.get("votes_near_truth_5deg", 0))
            near_2 = int(p.get("votes_near_truth_2deg", 0))
            total_near_5 += near_5
            total_near_2 += near_2
            max_match_mult = max(max_match_mult, float(p.get("match_mult_mean", 0.0)))
            if near_5 > 0:
                any_truth_votes = True
            best_ra = p.get("best_ra")
            best_dec = p.get("best_dec")
            if near_5 > 0 and best_ra is not None and best_dec is not None:
                if int(p.get("best_count_at_truth", 0)) < int(p.get("best_count", 0)):
                    any_winner_elsewhere = True

    support = f"match_mult_mean max={max_match_mult:.2f} (vysoka ~ nediskriminuje)."
    if total_near_5 == 0:
        return (
            f"VETVA 1: prava konfiguracia chyba v indexe "
            f"(votes_near_truth<5 deg=0 napriec passmi; <2 deg={total_near_2}). "
            f"Dalsi krok = hlbsi/zhodny index alebo zhodna konstrukcia trojuholnikov. {support}"
        )
    if any_winner_elsewhere or total_near_5 > 0:
        return (
            f"VETVA 2: hlasy pri pravde existuju (<5 deg={total_near_5}, <2 deg={total_near_2}), "
            f"ale vitazny klaster je inde -> hash nediskriminuje / chyba geometricka verifikacia. "
            f"Dalsi krok = spatna projekcia + match-fraction (+ pripadne quady). {support}"
        )
    return f"NEJEDNOZNACNE: near_5={total_near_5}, near_2={total_near_2}. {support}"


def _hypothesis_verdict(groups: list[dict[str, Any]]) -> str:
    total_passed = sum(int(g.get("n_passed_shape_fov", 0)) for g in groups)
    total_below = sum(int(g.get("n_below_min", 0)) for g in groups)
    total_above = sum(int(g.get("n_above_max", 0)) for g in groups)
    total_in = sum(int(g.get("n_in_range", 0)) for g in groups)
    if total_passed == 0:
        return "INCONCLUSIVE: ziadne trojuholniky nepresli tvar/FOV filtrom."
    pct_below = 100.0 * total_below / total_passed
    pct_in = 100.0 * total_in / total_passed
    if pct_below >= 50.0:
        return (
            f"POTVRDENE: {pct_below:.0f}% obrazovych trojuholnikov pod log_L3_min "
            f"({total_below}/{total_passed}) -> dolny-koniec clip je hlavny killer."
        )
    if pct_in >= 30.0:
        return (
            f"VYVRATENE (dolny clip): {pct_in:.0f}% trojuholnikov v rozsahu indexu "
            f"({total_in}/{total_passed}); pricina inde (hlasovanie / hlbka indexu / falosne zhody)."
        )
    if total_above > total_below:
        return (
            f"Ciastocne: viac trojuholnikov nad log_L3_max ({total_above}) nez pod min ({total_below}); "
            "horny koniec clip."
        )
    return (
        f"NEJEDNOZNACNE: below_min={total_below}, above_max={total_above}, "
        f"in_range={total_in} z {total_passed} - pozri per-group histogramy."
    )


def _format_group_report(
    *,
    group: str,
    fits_path: Path,
    index_meta: dict[str, Any],
    n_dao: int,
    tri: dict[str, Any],
    hint: tuple[float, float] | None,
    sink: dict[str, Any],
    truth_ra: float,
    truth_dec: float,
    plate_scale: float,
    fov_deg: float,
) -> list[str]:
    lines = [
        f"=== Group {group} ===",
        f"FITS: {fits_path}",
        f"plate_scale={plate_scale:.3f} arcsec/px  fov_deg={fov_deg:.3f}",
        "",
        "Index:",
        f"  mag_limit={index_meta.get('mag_limit')} tolerance={index_meta.get('tolerance')} "
        f"hash_dim={index_meta.get('hash_dim')} n_triangles={index_meta.get('n_triangles')}",
        f"  log_L3 [{index_meta['log_L3_min']:.3f}, {index_meta['log_L3_max']:.3f}] dex",
        f"  arcsec [{index_meta['log_L3_min_arcsec']:.2f}, {index_meta['log_L3_max_arcsec']:.2f}]",
        "",
        "Image:",
        f"  DAO 5-sigma detections: {n_dao}",
        f"  central stars (R={tri.get('R_px', 0):.0f}px): {tri.get('n_central', 0)}",
        f"  stars used (budget={tri.get('img_budget', '?')}, kNN k={tri.get('tri_k', '?')}): "
        f"{tri.get('n_stars_used', 0)}",
        f"  n_tried={tri.get('n_tried', 0)}  passed shape+FOV={tri.get('n_passed_shape_fov', 0)}",
        f"  below_min={tri.get('n_below_min', 0)}  above_max={tri.get('n_above_max', 0)} "
        f"in_range={tri.get('n_in_range', 0)} ({tri.get('pct_in_range', 0):.1f}% in range)",
    ]
    if tri.get("log_L3_med") is not None:
        lines.append(
            f"  log_L3 image: min={tri['log_L3_min']:.3f} p10={tri['log_L3_p10']:.3f} "
            f"med={tri['log_L3_med']:.3f} p90={tri['log_L3_p90']:.3f} max={tri['log_L3_max']:.3f} dex"
        )
    lines.append("")
    lines.append("Blind result (verify + legacy fallback):")
    if hint is None:
        lines.append("  result: None (MISS)")
    else:
        sep = _angular_sep_deg(hint[0], hint[1], truth_ra, truth_dec)
        verdict = "HIT" if sep < HIT_THRESHOLD_DEG else "MISS"
        lines.append(
            f"  result: RA={hint[0]:.4f} Dec={hint[1]:.4f}  "
            f"sep_from_truth={sep:.2f} deg  {verdict} (threshold {HIT_THRESHOLD_DEG} deg)"
        )
        lines.append(f"  truth: RA={truth_ra} Dec={truth_dec}")
    passes = sink.get("passes") or []
    if passes:
        lines.append("")
        lines.append("Vote diagnostics (debug_sink passes):")
        lines.extend(_format_pass_lines(passes))
        deciding = _deciding_pass(sink)
        if deciding is not None:
            lines.append("")
            lines.append(
                f"  deciding pass: dub={deciding.get('dub')} "
                f"votes_near_truth <2 deg={deciding.get('votes_near_truth_2deg', 'n/a')} "
                f"<5 deg={deciding.get('votes_near_truth_5deg', 'n/a')} "
                f"best_at_truth={deciding.get('best_count_at_truth', 'n/a')}"
            )
    verified = sink.get("verified_candidates") or []
    if verified:
        lines.append("")
        lines.append("Geometric verification (top-N candidates):")
        near_truth: dict[str, Any] | None = None
        for row in verified[:20]:
            fc_ra = row.get("field_center_ra", row.get("center_ra"))
            fc_dec = row.get("field_center_dec", row.get("center_dec"))
            sep_t = None
            if fc_ra is not None and fc_dec is not None:
                sep_t = _angular_sep_deg(
                    float(fc_ra), float(fc_dec), truth_ra, truth_dec
                )
                if near_truth is None or (
                    sep_t is not None
                    and sep_t < float(near_truth.get("sep_from_truth_deg", 999))
                ):
                    near_truth = {**row, "sep_from_truth_deg": sep_t}
            sep_s = f" sep_truth={sep_t:.2f} deg" if sep_t is not None else ""
            lines.append(
                f"  cand[{row.get('idx')}] RA={row.get('center_ra'):.3f} Dec={row.get('center_dec'):.3f} "
                f"matched={row.get('n_matched')} n_cat={row.get('n_cat_in_frame', row.get('n_cat'))} "
                f"n_dao={row.get('n_dao', 'n/a')} fraction={row.get('fraction', 0):.3f} "
                f"votes={row.get('vote_count')} hash={row.get('hash_dist', 0):.4f}{sep_s} "
                f"{'WINNER' if row.get('accepted') else 'reject'}"
            )
        if near_truth:
            lines.append(
                f"  nearest-to-truth: matched={near_truth.get('n_matched')} "
                f"n_cat={near_truth.get('n_cat_in_frame', near_truth.get('n_cat'))} "
                f"fraction={near_truth.get('fraction', 0):.3f} "
                f"sep={near_truth.get('sep_from_truth_deg', 'n/a')} deg"
            )
        winner = sink.get("verify_winner")
        if winner:
            lines.append(
                f"  verify winner: RA={winner.get('ra'):.4f} Dec={winner.get('dec'):.4f} "
                f"matched={winner.get('n_matched')} n_cat={winner.get('n_cat_in_frame')} "
                f"n_dao={winner.get('n_dao')} fraction={winner.get('fraction', 0):.3f}"
            )
    return lines


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description="Diagnose blind solver log_L3 clip on draft_000380.")
    ap.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    ap.add_argument("--index", type=Path, default=None, help="fine PKL (default: config blind_index_fine_path)")
    ap.add_argument("--plate-scale", type=float, default=DEFAULT_PLATE_SCALE)
    ap.add_argument("--truth-ra", type=float, default=DEFAULT_TRUTH_RA)
    ap.add_argument("--truth-dec", type=float, default=DEFAULT_TRUTH_DEC)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = AppConfig()
    draft = args.draft.expanduser().resolve()
    index_path = Path(
        args.index or cfg.blind_index_fine_path or cfg.blind_index_path
    ).expanduser().resolve()
    out_dir = (args.out or (draft / "diag" / "blind_solver")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not draft.is_dir():
        print(f"ERROR: draft not found: {draft}")
        return 1
    if not index_path.is_file():
        print(f"ERROR: blind index not found: {index_path}")
        return 1

    masterstars = _discover_masterstars(draft)
    if not masterstars:
        print(f"ERROR: no MASTERSTAR.fits under {draft}/platesolve/*/")
        return 1

    index_meta = _load_index_meta(index_path)
    log_L3_min = float(index_meta["log_L3_min"])
    log_L3_max = float(index_meta["log_L3_max"])
    tri_k = _index_k_neighbors({"k_neighbors": index_meta.get("k_neighbors")})

    report_lines = [
        "Blind solver diagnostic - draft_000380 Chi_and_H",
        f"started_utc: {datetime.now(timezone.utc).isoformat()}",
        f"draft: {draft}",
        f"index: {index_path}",
        "",
        "=== Index summary ===",
        f"mag_limit={index_meta.get('mag_limit')} tolerance={index_meta.get('tolerance')} "
        f"hash_dim={index_meta.get('hash_dim')} n_triangles={index_meta.get('n_triangles')} "
        f"k_neighbors={index_meta.get('k_neighbors', 'legacy->8')}",
        f"log_L3 [{log_L3_min:.3f}, {log_L3_max:.3f}] dex  "
        f"arcsec [{index_meta['log_L3_min_arcsec']:.2f}, {index_meta['log_L3_max_arcsec']:.2f}]",
        "",
    ]

    group_stats: list[dict[str, Any]] = []
    group_sinks: list[dict[str, Any]] = []
    cfg_dbg = AppConfig()
    cfg_dbg.debug_platesolver = True

    for group, fits_path in masterstars:
        _, dao_df, n_dao, nx = _detect_dao_stars(fits_path)
        with fits.open(fits_path, memmap=True) as hdul:
            ny = int(hdul[0].data.shape[0])
        fov_deg = _fov_deg_from_fits(nx, ny, args.plate_scale)

        tri = _analyze_image_triangles(
            dao_df,
            log_L3_min=log_L3_min,
            log_L3_max=log_L3_max,
            plate_scale=args.plate_scale,
            fov_deg=fov_deg,
            img_budget=int(cfg_dbg.blind_img_star_budget),
            tri_k=tri_k,
        )
        group_stats.append(tri)

        hint = None
        sink: dict[str, Any] = {}
        if not dao_df.empty and len(dao_df) >= 3:
            cands = find_blind_candidates(
                dao_df,
                index_path,
                n_top=30,
                top_n=int(cfg_dbg.blind_verify_top_n),
                plate_scale_arcsec_per_px=args.plate_scale,
                fov_deg=fov_deg,
                app_config=cfg_dbg,
                debug_truth_radec=(args.truth_ra, args.truth_dec),
                debug_sink=sink,
            )
            gaia_db = Path(cfg_dbg.gaia_db_path).expanduser()
            if gaia_db.is_file() and cands:
                hint = _verify_blind_candidates(
                    cands,
                    dao_df=dao_df,
                    gaia_db_path=gaia_db,
                    fov_deg=fov_deg,
                    naxis1=nx,
                    naxis2=ny,
                    pixel_pitch_um=None,
                    focal_length_mm=None,
                    max_cat_mag=16.0,
                    app_config=cfg_dbg,
                    debug_sink=sink,
                )
            if hint is None:
                hint = find_blind_hint(
                    dao_df,
                    index_path,
                    n_top=30,
                    min_votes=3,
                    plate_scale_arcsec_per_px=args.plate_scale,
                    fov_deg=fov_deg,
                    app_config=cfg_dbg,
                    debug_truth_radec=(args.truth_ra, args.truth_dec),
                )
        group_sinks.append(sink)

        png_path = out_dir / f"log_l3_hist_{group}.png"
        _plot_histogram(
            logl3_samples=tri.get("log_L3_samples", []),
            log_L3_min=log_L3_min,
            log_L3_max=log_L3_max,
            group=group,
            out_png=png_path,
        )

        deciding = _deciding_pass(sink)
        winner = sink.get("verify_winner")
        scatter_votes = deciding.get("votes") if deciding is not None else None
        best_ra = winner.get("ra") if winner else (deciding.get("best_ra") if deciding else None)
        best_dec = winner.get("dec") if winner else (deciding.get("best_dec") if deciding else None)
        if scatter_votes is not None:
            votes_arr = np.asarray(scatter_votes)
            _plot_sky_scatter(
                votes=votes_arr,
                truth_ra=args.truth_ra,
                truth_dec=args.truth_dec,
                best_ra=best_ra,
                best_dec=best_dec,
                group=group,
                out_png=out_dir / f"vote_sky_scatter_{group}.png",
            )

        report_lines.extend(
            _format_group_report(
                group=group,
                fits_path=fits_path,
                index_meta=index_meta,
                n_dao=n_dao,
                tri=tri,
                hint=hint,
                sink=sink,
                truth_ra=args.truth_ra,
                truth_dec=args.truth_dec,
                plate_scale=args.plate_scale,
                fov_deg=fov_deg,
            )
        )
        report_lines.append("")

    clip_verdict = _hypothesis_verdict(group_stats)
    branch_verdict = _votes_branch_verdict(group_sinks)
    report_lines.extend(
        [
            "=== Hypothesis verdict (#1 log_L3 clip) ===",
            clip_verdict,
            "",
            "=== Hypothesis verdict (#2 votes near truth) ===",
            branch_verdict,
            "",
        ]
    )

    report_path = out_dir / "blind_solver_diagnostic_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nReport: {report_path}")
    print(f"Histograms: {out_dir / 'log_l3_hist_*.png'}")
    print(f"Sky scatter: {out_dir / 'vote_sky_scatter_*.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
