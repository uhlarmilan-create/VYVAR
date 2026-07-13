"""TODO-12g4 validation: chroma boost A/B renders + metric gates."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from config import AppConfig
from hrd_colorfield import color_field_stats, render_catalog_color_field

OUT = ROOT / "tmp" / "todo12_hrd" / "run_0711_boost"

SETUPS = [
    (
        "draft425_V",
        ROOT / "Archive/Drafts/draft_000425/platesolve/V_20_2",
        {
            "reddest": (2727, 576),
            "cluster_core": (750, 940),
            "mid_field": (1500, 1200),
            "background": (120, 120),
        },
    ),
    (
        "draft424",
        ROOT / "Archive/Drafts/draft_000424/platesolve/NoFilter_60_2",
        {
            "reddest": (1693, 1272),
            "cluster_core": (1041, 698),
            "mid_field": (1500, 800),
            "background": (120, 120),
        },
    ),
]

BOOSTS_FM = (1.0, 1.6, 2.2)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _cfg(boost: float, white_point: str = "field_median") -> AppConfig:
    c = AppConfig()
    c.hrd_color_chroma_boost = float(boost)
    c.hrd_color_white_point = white_point
    return c


def _crop(img: np.ndarray, center: tuple[int, int], half: int) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = center
    return img[max(0, cy - half) : min(h, cy + half), max(0, cx - half) : min(w, cx + half)]


def _mean_rb_core(img: np.ndarray, center: tuple[int, int], inner_r: int = 10) -> float:
    h, w = img.shape[:2]
    cx, cy = center
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    core = img[r <= inner_r].astype(np.float64) / 255.0
    if core.size == 0:
        return float("nan")
    return float(np.mean(core[:, 0] / np.maximum(core[:, 2], 1e-6)))


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


def _background_neutrality(img: np.ndarray, center: tuple[int, int], half: int) -> float:
    crop = _crop(img, center, half).astype(np.float64) / 255.0
    l = np.maximum(crop.mean(axis=2), 1e-3)
    return float(np.mean(np.abs(crop[:, :, 0] - crop[:, :, 2]) / l))


def _zoom_crop_file(src: Path, center: tuple[int, int], half: int, out: Path, zoom: int = 3) -> None:
    img = np.asarray(Image.open(src).convert("RGB"))
    crop = _crop(img, center, half)
    im = Image.fromarray(crop)
    im = im.resize((crop.shape[1] * zoom, crop.shape[0] * zoom), Image.Resampling.NEAREST)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


def _strip_three(before: Path, mid: Path, after: Path, out: Path) -> None:
    imgs = [Image.open(str(p)).convert("RGB") for p in (before, mid, after)]
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    canvas = Image.new("RGB", (w * 3 + 8, h), (32, 32, 32))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * (w + 4), 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_renders: list[dict] = []
    strips: list[str] = []
    g2_failures: list[str] = []

    for label, ps, regions in SETUPS:
        phot = ps / "photometry" if (ps / "photometry").is_dir() else ps
        boost_paths: dict[float, Path] = {}

        for boost in BOOSTS_FM:
            outp = OUT / f"{label}_fm_boost{boost:.1f}_field_color.png"
            t0 = time.perf_counter()
            path = render_catalog_color_field(ps, phot, _cfg(boost), outp)
            elapsed = time.perf_counter() - t0
            if path is None:
                print(f"FAIL render {label} boost={boost}")
                return 1
            boost_paths[boost] = path
            img = np.asarray(Image.open(path).convert("RGB"))
            stats = color_field_stats(ps, phot, render_seconds=elapsed)
            g2 = _background_neutrality(img, regions["background"], 50)
            entry = {
                "label": label,
                "white_point": "field_median",
                "boost": boost,
                "png": str(path),
                "pct_colored": stats.get("pct_colored"),
                "bp_rp_range": [stats.get("bp_rp_min"), stats.get("bp_rp_max")],
                "render_seconds": round(elapsed, 3),
                "mean_rb_reddest_core": _mean_rb_core(img, regions["reddest"]),
                "cluster_rb_std": _cluster_rb_std(img, regions["cluster_core"], 70),
                "G2_background_neutrality": g2,
                "G2_pass": g2 < 0.03,
            }
            all_renders.append(entry)
            if not entry["G2_pass"]:
                g2_failures.append(f"{label} boost={boost} G2={g2:.4f}")

        # d65 at boost 1.6
        d65_out = OUT / f"{label}_d65_boost1.6_field_color.png"
        t0 = time.perf_counter()
        render_catalog_color_field(ps, phot, _cfg(1.6, "d65"), d65_out)
        all_renders.append(
            {
                "label": label,
                "white_point": "d65",
                "boost": 1.6,
                "png": str(d65_out),
                "render_seconds": round(time.perf_counter() - t0, 3),
            }
        )

        # side-by-side strips (same crop, three boosts)
        for crop_name, center in (
            ("reddest", regions["reddest"]),
            ("cluster_core", regions["cluster_core"]),
            ("mid_field", regions["mid_field"]),
        ):
            crops_dir = OUT / "crops" / label
            paths = []
            for boost in BOOSTS_FM:
                cp = crops_dir / f"{crop_name}_boost{boost:.1f}.png"
                _zoom_crop_file(boost_paths[boost], center, 40, cp, zoom=3)
                paths.append(cp)
            strip_out = OUT / f"{label}_strip_{crop_name}.png"
            _strip_three(paths[0], paths[1], paths[2], strip_out)
            strips.append(str(strip_out))

    overall_g2 = len(g2_failures) == 0
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_head": _git_head(),
        "renders": all_renders,
        "strips": strips,
        "G2_all_pass": overall_g2,
        "G2_failures": g2_failures,
    }
    (OUT / "summary_12g4.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if overall_g2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
