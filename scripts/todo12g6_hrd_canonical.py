"""TODO-12g6: canonical colorfield final renders + manifest (caption-stamped, default boost 2.2)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from config import AppConfig
from hrd_colorfield import (
    background_neutrality_grid,
    build_colorfield_caption,
    build_star_exclusion_mask,
    color_field_stats,
    hrd_color_bg_box_px_from_cfg,
    hrd_color_chroma_boost_from_cfg,
    hrd_color_chroma_snr_from_cfg,
    hrd_color_saturation_from_cfg,
    render_catalog_color_field,
    save_g2_heatmap_png,
    _prepare_color_stars,
)
from hrd_analysis import field_annotation_pixel_scale

ARCHIVE_SRC = ROOT / "tmp" / "todo12_hrd"
ARCHIVE_DST = ROOT / "tmp" / "todo12_hrd_archive_0711"
OUT = ROOT / "tmp" / "colorfield_final"

SETUPS = [
    (
        "d424",
        "NoFilter60_2",
        ROOT / "Archive/Drafts/draft_000424/platesolve/NoFilter_60_2",
        {
            "reddest": (1693, 1272),
            "densest": (1041, 698),
            "bright_bg": (130, 1280),
        },
    ),
    (
        "d425",
        "V20_2",
        ROOT / "Archive/Drafts/draft_000425/platesolve/V_20_2",
        {
            "reddest": (2727, 576),
            "densest": (1500, 1200),
            "cluster_core": (750, 940),
        },
    ),
]

WP_MODES = (
    ("fm", "field_median"),
    ("d65", "d65"),
)


def _git_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "nogit"
    except Exception:  # noqa: BLE001
        return "nogit"


def _archive_inventory() -> list[str]:
    if not ARCHIVE_SRC.is_dir():
        return []
    items: list[str] = []
    for p in sorted(ARCHIVE_SRC.rglob("*")):
        if p.is_file():
            items.append(str(p.relative_to(ARCHIVE_SRC)).replace("\\", "/"))
    return items


def _archive_old_runs() -> list[str]:
    if not ARCHIVE_SRC.is_dir():
        return []
    if ARCHIVE_DST.exists():
        raise SystemExit(f"archive destination already exists: {ARCHIVE_DST}")
    shutil.move(str(ARCHIVE_SRC), str(ARCHIVE_DST))
    return _list_archive_top(ARCHIVE_DST)


def _list_archive_top(base: Path) -> list[str]:
    if not base.is_dir():
        return []
    out: list[str] = []
    for p in sorted(base.iterdir()):
        out.append(p.name + ("/" if p.is_dir() else ""))
    return out


def _crop(img: np.ndarray, center: tuple[int, int], half: int) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = center
    return img[max(0, cy - half) : min(h, cy + half), max(0, cx - half) : min(w, cx + half)]


def _star_mask(ps: Path, h: int, w: int) -> np.ndarray:
    csv = ps / "masterstars_full_match.csv"
    if not csv.is_file():
        return np.ones((h, w), dtype=bool)
    df = pd.read_csv(csv, usecols=["x", "y"])
    sx, sy, ok = field_annotation_pixel_scale(ps, w, h, png_from_fits=True)
    if not ok:
        return np.ones((h, w), dtype=bool)
    xs = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=np.float64) * sx
    ys = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=np.float64) * sy
    return build_star_exclusion_mask(h, w, xs, ys, exclude_r=4.0)


def _save_crop(src: Path, center: tuple[int, int], name: str, crops_dir: Path) -> str:
    img = np.asarray(Image.open(src).convert("RGB"))
    crop = _crop(img, center, 40)
    zoom = Image.fromarray(crop).resize(
        (crop.shape[1] * 3, crop.shape[0] * 3), Image.Resampling.NEAREST
    )
    out = crops_dir / f"{src.stem}_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    zoom.save(out)
    return str(out.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    archived = _archive_old_runs() if ARCHIVE_SRC.is_dir() else _list_archive_top(ARCHIVE_DST)
    OUT.mkdir(parents=True, exist_ok=True)
    crops_dir = OUT / "crops"
    heat_dir = OUT / "g2_heatmaps"
    renders: list[dict] = []
    g2_failures: list[str] = []
    caption_failures: list[str] = []
    git_head = _git_short()

    for draft, setup, ps, regions in SETUPS:
        phot = ps / "photometry" if (ps / "photometry").is_dir() else ps
        crop_keys = ("reddest", "densest", "bright_bg") if draft == "d424" else (
            "reddest",
            "densest",
            "cluster_core",
        )

        for wp_tag, wp_mode in WP_MODES:
            cfg = AppConfig()
            if wp_mode != "field_median":
                cfg.hrd_color_white_point = wp_mode
            boost = hrd_color_chroma_boost_from_cfg(cfg)
            fname = f"{draft}_{setup}_{wp_tag}_b{boost:.1f}_color.png"
            out_png = OUT / fname
            stamp_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0)

            t0 = time.perf_counter()
            path = render_catalog_color_field(
                ps, phot, cfg, out_png, rendered_at_utc=stamp_utc
            )
            elapsed = time.perf_counter() - t0
            if path is None:
                print(f"FAIL render {fname}")
                return 1

            img = np.asarray(Image.open(path).convert("RGB"))
            h, w = img.shape[:2]
            sm = _star_mask(ps, h, w)
            g2 = background_neutrality_grid(img, sm)
            heat_name = fname.replace("_color.png", "_g2.png")
            heat_path = heat_dir / heat_name
            save_g2_heatmap_png(g2["metrics_grid"], heat_path)
            stats = color_field_stats(ps, phot, render_seconds=elapsed)

            crop_paths = []
            for key in crop_keys:
                crop_paths.append(_save_crop(path, regions[key], key, crops_dir))

            entry = {
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "draft": draft,
                "setup": setup,
                "white_point": wp_mode,
                "boost": boost,
                "saturation": hrd_color_saturation_from_cfg(cfg),
                "chroma_snr": hrd_color_chroma_snr_from_cfg(cfg),
                "bg_box_px": hrd_color_bg_box_px_from_cfg(cfg),
                "generated_at_utc": stamp_utc.replace(microsecond=0).isoformat(),
                "git_head": git_head,
                "g2_worst": g2["worst_patch_metric"],
                "g2_worst_xy": g2["worst_location_xy"],
                "pct_colored": stats.get("pct_colored"),
                "render_s": round(elapsed, 3),
                "crops": [p.replace("\\", "/") for p in crop_paths],
                "g2_heatmap": str(heat_path.relative_to(ROOT)).replace("\\", "/"),
            }
            renders.append(entry)

            if not g2.get("pass"):
                g2_failures.append(
                    f"{fname} worst={g2['worst_patch_metric']:.4f} at {g2['worst_location_xy']}"
                )

            cap_pat = re.compile(
                rf" rendered {re.escape(stamp_utc.strftime('%Y-%m-%d %H:%M'))} UTC @ {re.escape(git_head)}\."
            )
            _, colorable = _prepare_color_stars(
                ps / "masterstars_full_match.csv", ps / "_hrd_cache" / "hrd_enrich.json"
            )
            fm_teff = (
                float(np.median(colorable["teff_k"].to_numpy(dtype=np.float64)))
                if not colorable.empty
                else None
            )
            expected = build_colorfield_caption(
                white_point=wp_mode,  # type: ignore[arg-type]
                field_median_teff_k=fm_teff if wp_mode == "field_median" else None,
                chroma_boost=boost,
                rendered_at_utc=stamp_utc,
                git_short_hash=git_head,
            )
            if wp_mode == "field_median" and "white point = field median Teff" not in expected:
                caption_failures.append(f"{fname} missing fm WP line")
            if boost > 1.0 and f"chroma enhanced x{boost:.1f}" not in expected:
                caption_failures.append(f"{fname} missing boost suffix")
            if not cap_pat.search(expected):
                caption_failures.append(f"{fname} missing stamp")
            entry["caption"] = expected

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_head": git_head,
        "archive_moved_to": str(ARCHIVE_DST.relative_to(ROOT)).replace("\\", "/"),
        "archive_top_level": archived,
        "renders": renders,
        "G2_all_pass": len(g2_failures) == 0,
        "G2_failures": g2_failures,
        "caption_all_pass": len(caption_failures) == 0,
        "caption_failures": caption_failures,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if not g2_failures and not caption_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
