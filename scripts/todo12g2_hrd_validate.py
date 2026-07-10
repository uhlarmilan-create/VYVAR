"""TODO-12g2 validation: A/B catalog-color field polish + hard visual gates G1-G4."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from config import AppConfig
from hrd_colorfield import color_field_stats, timed_render_catalog_color_field

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp" / "todo12_hrd"
PRE12G = OUT / "pre12g2"
RS_PER_CID = "458407464445792384"

SETUPS = [
    ("draft424", ROOT / "Archive/Drafts/draft_000424/platesolve/NoFilter_60_2"),
    ("draft425_B", ROOT / "Archive/Drafts/draft_000425/platesolve/B_20_2"),
]

CROP_SPECS = {
    "draft425_B": {
        "rs_per": {"center": (680, 985), "half": 40, "label": "RS Per"},
        "cluster_core": {"center": (720, 920), "half": 70, "label": "chi Per cluster core"},
        "background": {"center": (120, 120), "half": 50, "label": "background"},
    },
    "draft424": {
        "red_giant": {"center": (1155, 459), "half": 40, "label": "red giant"},
        "background": {"center": (100, 100), "half": 50, "label": "background"},
    },
}


def _git_head_short() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(ROOT),
                text=True,
            )
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _crop(img: np.ndarray, center: tuple[int, int], half: int) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = center
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    return img[y0:y1, x0:x1]


def _side_by_side(before: Path, after: Path, out: Path, zoom: int = 3) -> None:
    imgs = []
    for p in (before, after):
        if not p.is_file():
            return
        im = Image.open(str(p)).convert("RGB")
        if zoom > 1:
            im = im.resize((im.width * zoom, im.height * zoom), Image.Resampling.NEAREST)
        imgs.append(im)
    if len(imgs) != 2:
        return
    w = max(imgs[0].width, imgs[1].width)
    h = max(imgs[0].height, imgs[1].height)
    canvas = Image.new("RGB", (w * 2 + 4, h), (32, 32, 32))
    canvas.paste(imgs[0], (0, 0))
    canvas.paste(imgs[1], (w + 4, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out))


def _zoom_crop_file(src: Path, center: tuple[int, int], half: int, out: Path, zoom: int = 3) -> None:
    if not src.is_file():
        return
    img = np.asarray(Image.open(str(src)).convert("RGB"))
    crop = _crop(img, center, half)
    im = Image.fromarray(crop)
    if zoom > 1:
        im = im.resize((crop.shape[1] * zoom, crop.shape[0] * zoom), Image.Resampling.NEAREST)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(out))


def _mean_rb_core(img: np.ndarray, center: tuple[int, int], inner_r: int = 10) -> float:
    h, w = img.shape[:2]
    cx, cy = center
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    core = img[r <= inner_r].astype(np.float64) / 255.0
    if core.size == 0:
        return float("nan")
    rb = core[:, 0] / np.maximum(core[:, 2], 1e-6)
    return float(np.mean(rb))


def _background_neutrality(img: np.ndarray, center: tuple[int, int], half: int) -> float:
    crop = _crop(img, center, half).astype(np.float64) / 255.0
    l = np.mean(crop, axis=2)
    l = np.maximum(l, 1e-3)
    diff = np.abs(crop[:, :, 0] - crop[:, :, 2]) / l
    return float(np.mean(diff))


def _cluster_rb_std(img: np.ndarray, center: tuple[int, int], half: int) -> float:
    """Std of R/B at local brightness peaks in crop (star cores)."""
    crop = _crop(img, center, half).astype(np.float64)
    gray = crop.mean(axis=2)
    thr = float(np.percentile(gray, 92))
    peaks = gray >= thr
    if peaks.sum() < 8:
        return 0.0
    rgb = crop[peaks] / 255.0
    rb = rgb[:, 0] / np.maximum(rgb[:, 2], 1e-6)
    return float(np.std(rb))


def _spot_check_neutral(img: np.ndarray, center: tuple[int, int], half: int, n: int = 3) -> list[float]:
    """Sample low-but-nonzero background pixels; R/B should be ~1 when chroma is neutral."""
    crop = _crop(img, center, half)
    gray = crop.mean(axis=2)
    mask = (gray >= 12) & (gray <= np.percentile(gray, 35))
    ys, xs = np.where(mask)
    if ys.size < n:
        mask = gray >= 12
        ys, xs = np.where(mask)
    if ys.size == 0:
        return []
    step = max(1, ys.size // n)
    out: list[float] = []
    for i in range(min(n, ys.size)):
        idx = min(i * step, ys.size - 1)
        y, x = int(ys[idx]), int(xs[idx])
        r, _g, b = crop[y, x].astype(np.float64)
        out.append(float(r / max(b, 1e-6)))
    return out


def _archive_pre12g2() -> None:
    PRE12G.mkdir(parents=True, exist_ok=True)
    for name in (
        "draft424_field_color.png",
        "draft425_B_field_color.png",
        "draft424_color_crop_red_giant.png",
        "draft425_B_color_crop_red_giant.png",
        "draft425_B_color_crop_blue_ms.png",
    ):
        src = OUT / name
        if src.is_file():
            shutil.copy2(src, PRE12G / name)


def main() -> int:
    _archive_pre12g2()
    cfg = AppConfig()
    results: list[dict] = []
    gates: dict[str, dict] = {}

    for label, ps in SETUPS:
        phot_dir = ps / "photometry" if (ps / "photometry").is_dir() else ps
        out_png = OUT / f"{label}_field_color.png"
        path, elapsed = timed_render_catalog_color_field(ps, phot_dir, cfg, out_png)
        stats = color_field_stats(ps, phot_dir, render_seconds=elapsed)
        entry = {"label": label, "png": str(path) if path else None, "stats": stats}
        results.append(entry)

        if path is None:
            continue
        img = np.asarray(Image.open(str(path)).convert("RGB"))
        pre = PRE12G / f"{label}_field_color.png"
        specs = CROP_SPECS.get(label, {})
        ab_crops: dict[str, str] = {}

        for key, spec in specs.items():
            center = spec["center"]
            half = spec["half"]
            before_crop = PRE12G / f"{label}_ab_{key}_before.png"
            after_crop = OUT / f"{label}_ab_{key}_after.png"
            ab_out = OUT / f"{label}_ab_{key}.png"
            if pre.is_file():
                _zoom_crop_file(pre, center, half, before_crop, zoom=3)
            _zoom_crop_file(path, center, half, after_crop, zoom=3)
            if before_crop.is_file() and after_crop.is_file():
                _side_by_side(before_crop, after_crop, ab_out, zoom=1)
                ab_crops[key] = str(ab_out)
        entry["ab_crops"] = ab_crops

        if label == "draft425_B":
            g1 = _mean_rb_core(img, (680, 985), inner_r=10)
            g2 = _background_neutrality(img, (120, 120), 50)
            g3 = _cluster_rb_std(img, (720, 920), 70)
            g4 = _spot_check_neutral(img, (120, 120), 50, n=3)
            gates["draft425_B"] = {
                "G1_rs_per_mean_rb": {"value": g1, "pass": g1 > 1.3},
                "G2_background_neutrality": {"value": g2, "pass": g2 < 0.03},
                "G3_cluster_rb_std": {"value": g3, "pass": g3 > 0.05},
                "G4_unmatched_rb_spotcheck": {
                    "values": g4,
                    "pass": len(g4) >= 3 and all(0.85 <= v <= 1.15 for v in g4),
                },
            }
            entry["gates"] = gates["draft425_B"]

        if label == "draft424":
            g2 = _background_neutrality(img, (100, 100), 50)
            g4 = _spot_check_neutral(img, (100, 100), 50, n=3)
            gates["draft424"] = {
                "G2_background_neutrality": {"value": g2, "pass": g2 < 0.03},
                "G4_unmatched_rb_spotcheck": {
                    "values": g4,
                    "pass": len(g4) >= 3 and all(0.85 <= v <= 1.15 for v in g4),
                },
            }
            entry["gates"] = gates["draft424"]

    all_pass = bool(gates.get("draft425_B", {}).get("G1_rs_per_mean_rb", {}).get("pass")) and all(
        g.get("pass", False)
        for block in gates.values()
        for g in block.values()
        if isinstance(g, dict) and "pass" in g
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_head": _git_head_short(),
        "pre12g2_archive": str(PRE12G),
        "setups": results,
        "gates": gates,
        "overall_pass": all_pass,
    }
    out_json = OUT / "summary_12g2.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
