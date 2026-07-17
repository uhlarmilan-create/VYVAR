"""TODO-12g5 validation: local-bg SNR gate, stamp taper, hardened G2."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
ROOT = _bootstrap.REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from config import AppConfig
from hrd_colorfield import (
    background_neutrality_grid,
    build_star_exclusion_mask,
    color_field_stats,
    render_catalog_color_field,
    save_g2_heatmap_png,
)

OUT = ROOT / "tmp" / "todo12_hrd" / "run_0711_boost"
PRE = ROOT / "tmp" / "todo12_hrd" / "pre12g5"
G2_THRESHOLD = 0.03
CLUSTER_STD_TOL = 0.90

SETUPS = [
    (
        "draft425_V",
        ROOT / "Archive/Drafts/draft_000425/platesolve/V_20_2",
        {
            "cluster_core": (750, 940),
            "bright_center": (750, 940),
        },
    ),
    (
        "draft424",
        ROOT / "Archive/Drafts/draft_000424/platesolve/NoFilter_60_2",
        {
            "cluster_core": (1041, 698),
            "bright_center": (130, 1280),
        },
    ),
]

BOOSTS = (1.6, 2.2)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _cfg(boost: float) -> AppConfig:
    c = AppConfig()
    c.hrd_color_chroma_boost = float(boost)
    c.hrd_color_white_point = "field_median"
    return c


def _crop(img: np.ndarray, center: tuple[int, int], half: int) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = center
    return img[max(0, cy - half) : min(h, cy + half), max(0, cx - half) : min(w, cx + half)]


def _cluster_rb_std(img: np.ndarray, center: tuple[int, int], half: int) -> float:
    crop = _crop(img, center, half).astype(np.float64)
    gray = crop.mean(axis=2)
    thr = float(np.percentile(gray, 92))
    peaks = gray >= thr
    if peaks.sum() < 8:
        return 0.0
    rgb = crop[peaks] / 255.0
    rb = rgb[:, 0] / np.maximum(rgb[:, 2], 1e-6)
    return float(np.std(rb))


def _star_mask(ps: Path, h: int, w: int) -> np.ndarray:
    csv = ps / "masterstars_full_match.csv"
    if not csv.is_file():
        return np.ones((h, w), dtype=bool)
    df = pd.read_csv(csv, usecols=["x", "y"], nrows=50000)
    from hrd_analysis import field_annotation_pixel_scale

    sx, sy, ok = field_annotation_pixel_scale(ps, w, h, png_from_fits=True)
    if not ok:
        return np.ones((h, w), dtype=bool)
    xs = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=np.float64) * sx
    ys = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=np.float64) * sy
    return build_star_exclusion_mask(h, w, xs, ys, exclude_r=4.0)


def _ab_strip(before: Path, after: Path, out: Path) -> None:
    imgs = [Image.open(str(p)).convert("RGB") for p in (before, after)]
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    canvas = Image.new("RGB", (w * 2 + 4, h), (32, 32, 32))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * (w + 4), 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def _archive_prefix() -> None:
    PRE.mkdir(parents=True, exist_ok=True)
    if not OUT.is_dir():
        return
    for p in OUT.glob("*.png"):
        dest = PRE / p.name
        if not dest.is_file():
            shutil.copy2(p, dest)
    for sub in ("crops",):
        src_dir = OUT / sub
        if src_dir.is_dir():
            dst_dir = PRE / sub
            if not dst_dir.is_dir():
                shutil.copytree(src_dir, dst_dir)


def main() -> int:
    _archive_prefix()
    OUT.mkdir(parents=True, exist_ok=True)
    all_renders: list[dict] = []
    g2_failures: list[str] = []
    cluster_failures: list[str] = []
    strips: list[str] = []

    for label, ps, regions in SETUPS:
        phot = ps / "photometry" if (ps / "photometry").is_dir() else ps
        for boost in BOOSTS:
            outp = OUT / f"{label}_fm_boost{boost:.1f}_field_color.png"
            pre_path = PRE / outp.name
            t0 = time.perf_counter()
            path = render_catalog_color_field(ps, phot, _cfg(boost), outp)
            elapsed = time.perf_counter() - t0
            if path is None:
                print(f"FAIL render {label} boost={boost}")
                return 1

            img = np.asarray(Image.open(path).convert("RGB"))
            h, w = img.shape[:2]
            star_mask = _star_mask(ps, h, w)
            g2 = background_neutrality_grid(img, star_mask)
            heat_path = OUT / f"{label}_fm_boost{boost:.1f}_g2_heatmap.png"
            save_g2_heatmap_png(g2["metrics_grid"], heat_path)

            cluster_std = _cluster_rb_std(img, regions["cluster_core"], 70)
            pre_cluster_std = None
            if pre_path.is_file():
                pre_img = np.asarray(Image.open(pre_path).convert("RGB"))
                pre_cluster_std = _cluster_rb_std(pre_img, regions["cluster_core"], 70)

            entry = {
                "label": label,
                "boost": boost,
                "png": str(path),
                "render_seconds": round(elapsed, 3),
                "G2_worst_patch": g2["worst_patch_metric"],
                "G2_worst_xy": g2["worst_location_xy"],
                "G2_heatmap": str(heat_path),
                "G2_pass": g2["pass"],
                "cluster_rb_std": cluster_std,
                "pre_cluster_rb_std": pre_cluster_std,
                "pct_colored": color_field_stats(ps, phot, render_seconds=elapsed).get("pct_colored"),
            }
            all_renders.append(entry)

            if not entry["G2_pass"]:
                g2_failures.append(
                    f"{label} boost={boost} worst={g2['worst_patch_metric']:.4f} "
                    f"at {g2['worst_location_xy']}"
                )
            if label == "draft425_V" and boost == 2.2 and pre_cluster_std is not None and pre_cluster_std > 0:
                if cluster_std < CLUSTER_STD_TOL * pre_cluster_std:
                    cluster_failures.append(
                        f"{label} boost={boost} cluster_std={cluster_std:.3f} "
                        f"< {CLUSTER_STD_TOL}*{pre_cluster_std:.3f}"
                    )

            if pre_path.is_file():
                crop_name = "bright_center" if label == "draft424" else "cluster_core"
                center = regions[crop_name]
                half = 50
                pre_crop = OUT / "ab" / f"{label}_boost{boost:.1f}_{crop_name}_before.png"
                post_crop = OUT / "ab" / f"{label}_boost{boost:.1f}_{crop_name}_after.png"
                for src, dst in ((pre_path, pre_crop), (path, post_crop)):
                    im = np.asarray(Image.open(src).convert("RGB"))
                    c = _crop(im, center, half)
                    zoom = Image.fromarray(c).resize(
                        (c.shape[1] * 3, c.shape[0] * 3), Image.Resampling.NEAREST
                    )
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    zoom.save(dst)
                strip_out = OUT / f"{label}_ab_{crop_name}_boost{boost:.1f}.png"
                _ab_strip(pre_crop, post_crop, strip_out)
                strips.append(str(strip_out))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_head": _git_head(),
        "renders": all_renders,
        "strips": strips,
        "G2_threshold": G2_THRESHOLD,
        "G2_all_pass": len(g2_failures) == 0,
        "G2_failures": g2_failures,
        "cluster_std_tolerance": CLUSTER_STD_TOL,
        "cluster_all_pass": len(cluster_failures) == 0,
        "cluster_failures": cluster_failures,
        "pre_archive": str(PRE),
    }
    (OUT / "summary_12g5.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if g2_failures or cluster_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
