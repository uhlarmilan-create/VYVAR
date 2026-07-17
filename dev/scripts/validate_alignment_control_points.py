"""Compare astroalign transforms at max_control_points=200 (legacy effective) vs 80 (new default).

Milan: run on Chi/h draft 419/420 input frames BEFORE finalizing the perf change.

Example (adjust setup / draft paths to your Chi/h draft):

  python scripts/validate_alignment_control_points.py \\
    --draft-root Archive/Drafts/draft_000419 \\
    --setup B_60_2 \\
    --max-frames 5

Or explicit FITS:

  python scripts/validate_alignment_control_points.py \\
    --ref path/to/reference.fits \\
    --frames path/to/light1.fits path/to/light2.fits
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig
from utils import dao_detection_fwhm_pixels, seeded_numpy_default_rng, VYVAR_RANDOM_SEED
from vyvar_alignment_frame import _alignment_as_alignment_points, _alignment_detect_xy

OLD_MCP = 200
NEW_MCP = 80

PASS_TX_PX = 0.05
PASS_ROT_DEG = 0.01
PASS_SCALE = 1e-4
PASS_RMS_DELTA_PX = 0.05


def _load_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()
    return data, hdr


def _transform_metrics(t) -> dict[str, float]:
    rot_deg = float(np.degrees(t.rotation))
    tx = float(t.translation[0])
    ty = float(t.translation[1])
    scale = float(t.scale)
    return {
        "tx_px": tx,
        "ty_px": ty,
        "trans_norm_px": float(math.hypot(tx, ty)),
        "rotation_deg": rot_deg,
        "scale": scale,
    }


def _residual_rms_px(t, src: np.ndarray, dst: np.ndarray) -> float:
    mapped = np.asarray(t(src), dtype=np.float64)
    d = mapped - np.asarray(dst, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _run_find_transform(
    src: np.ndarray,
    dst: np.ndarray,
    mcp: int,
) -> tuple[object, float, float]:
    import astroalign

    mcp_eff = max(12, min(int(mcp), int(min(len(src), len(dst)))))
    t0 = time.perf_counter()
    with seeded_numpy_default_rng(VYVAR_RANDOM_SEED):
        t, (s_sub, d_sub) = astroalign.find_transform(
            source=np.asarray(src, dtype=np.float32),
            target=np.asarray(dst, dtype=np.float32),
            max_control_points=mcp_eff,
        )
    elapsed = time.perf_counter() - t0
    rms = _residual_rms_px(t, s_sub, d_sub)
    return t, elapsed, rms


def _compare_frame(
    frame_path: Path,
    ref_xy: np.ndarray,
    *,
    det_sigma: float,
    det_want: int,
    fwhm_px: float,
) -> dict:
    data, hdr = _load_image(frame_path)
    xy = _alignment_detect_xy(
        data,
        want_max=int(det_want),
        det_sigma=float(det_sigma),
        fwhm_px=float(fwhm_px),
        label=frame_path.name,
        log_sink=None,
    )
    n_det = int(len(xy))
    current_target = np.asarray(ref_xy, dtype=np.float32)
    n_fit = int(min(len(current_target), len(xy)))
    xy_fit = _alignment_as_alignment_points(xy[:n_fit], label="source", log_sink=None)
    ref_fit = _alignment_as_alignment_points(current_target[:n_fit], label="target", log_sink=None)
    n_fit = int(min(len(xy_fit), len(ref_fit)))
    if n_fit < 12:
        raise RuntimeError(f"{frame_path.name}: only {n_fit} fit points (need >= 12)")
    xy_fit = np.asarray(xy_fit[:n_fit], dtype=np.float32)
    ref_fit = np.asarray(ref_fit[:n_fit], dtype=np.float32)

    t_old, t_old_s, rms_old = _run_find_transform(xy_fit, ref_fit, OLD_MCP)
    t_new, t_new_s, rms_new = _run_find_transform(xy_fit, ref_fit, NEW_MCP)
    m_old = _transform_metrics(t_old)
    m_new = _transform_metrics(t_new)

    dtx = m_new["tx_px"] - m_old["tx_px"]
    dty = m_new["ty_px"] - m_old["ty_px"]
    dtrans = float(math.hypot(dtx, dty))
    drot = float(m_new["rotation_deg"] - m_old["rotation_deg"])
    dscale = float(m_new["scale"] - m_old["scale"])
    drms = float(rms_new - rms_old)

    fail = (
        dtrans >= PASS_TX_PX
        or abs(drot) >= PASS_ROT_DEG
        or abs(dscale) >= PASS_SCALE
        or abs(drms) >= PASS_RMS_DELTA_PX
    )

    return {
        "frame": frame_path.name,
        "detected": n_det,
        "fit_points": n_fit,
        "old_mcp": OLD_MCP,
        "new_mcp": NEW_MCP,
        "old": m_old,
        "new": m_new,
        "rms_old_px": rms_old,
        "rms_new_px": rms_new,
        "delta_trans_px": dtrans,
        "delta_rot_deg": drot,
        "delta_scale": dscale,
        "delta_rms_px": drms,
        "time_old_s": t_old_s,
        "time_new_s": t_new_s,
        "speedup": (t_old_s / t_new_s) if t_new_s > 0 else float("inf"),
        "fail": fail,
    }


def _draft_lights_dir(draft_root: Path, setup: str) -> Path:
    """Resolve input lights folder for a draft setup (VYVAR layout variants)."""
    setup = str(setup).strip()
    candidates = [
        draft_root / "processed" / "lights" / setup,
        draft_root / "non_calibrated" / "lights" / setup,
        draft_root / "calibrated" / "lights" / setup,
        draft_root / "detrended" / "lights" / setup,
        draft_root / "detrended_aligned" / "lights" / setup,
    ]
    for p in candidates:
        if p.is_dir():
            return p
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No lights dir for setup {setup!r}; tried: {tried}")


def _list_light_fits(lights_dir: Path) -> list[Path]:
    files = sorted(lights_dir.glob("proc_*.fits"))
    if not files:
        files = sorted(
            p for p in lights_dir.glob("*.fits")
            if p.name.upper() not in ("MASTERSTAR.FITS", "MASTERDARK.FITS", "MASTERFLAT.FITS")
        )
    return files


def _resolve_frames(args: argparse.Namespace) -> tuple[Path, list[Path]]:
    if args.ref and args.frames:
        ref = Path(args.ref).resolve()
        frames = [Path(p).resolve() for p in args.frames]
        return ref, frames

    if not args.draft_root or not str(args.setup).strip():
        raise SystemExit("Provide --ref + --frames, or --draft-root + --setup")

    draft_root = Path(args.draft_root).resolve()
    setup = str(args.setup).strip()
    lights_dir = _draft_lights_dir(draft_root, setup)
    all_files = _list_light_fits(lights_dir)
    if not all_files:
        raise FileNotFoundError(f"No light FITS in {lights_dir}")

    if args.ref:
        ref = Path(args.ref).resolve()
        frames = all_files
    else:
        ms_ref = draft_root / "platesolve" / setup / "MASTERSTAR.fits"
        ref = ms_ref.resolve() if ms_ref.is_file() else all_files[0]
        frames = [f for f in all_files if f.resolve() != ref.resolve()]
    if args.max_frames and args.max_frames > 0:
        frames = frames[: int(args.max_frames)]
    if not frames:
        raise FileNotFoundError("No light frames to compare (only reference found)")
    return ref, frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate astroalign control-point cap (200 vs 80).")
    parser.add_argument("--draft-root", type=Path, default=None, help="Draft dir (e.g. Archive/Drafts/draft_000419)")
    parser.add_argument("--setup", type=str, default="", help="Filter/exp/bin subdir under draft lights tree")
    parser.add_argument("--ref", type=Path, default=None, help="Reference FITS (default: first proc in setup)")
    parser.add_argument("--frames", nargs="*", default=None, help="Light FITS paths (with --ref)")
    parser.add_argument("--max-frames", type=int, default=5, help="Max light frames to test (draft mode)")
    parser.add_argument("--det-sigma", type=float, default=None, help="DAO sigma (default: cfg sips_dao_threshold_sigma)")
    parser.add_argument("--det-max", type=int, default=300, help="Detection want_max (ladder-style cap)")
    args = parser.parse_args()

    cfg = AppConfig()
    det_sigma = float(args.det_sigma if args.det_sigma is not None else cfg.sips_dao_threshold_sigma)
    ref_path, frame_paths = _resolve_frames(args)

    ref_data, ref_hdr = _load_image(ref_path)
    fb = float(cfg.sips_dao_fwhm_px)
    fwhm_px = float(dao_detection_fwhm_pixels(ref_hdr, configured_fallback=fb) or fb)
    ref_xy = _alignment_detect_xy(
        ref_data,
        want_max=int(args.det_max),
        det_sigma=det_sigma,
        fwhm_px=fwhm_px,
        label=ref_path.name,
        log_sink=None,
    )
    if len(ref_xy) < 12:
        print(f"ERROR: reference {ref_path.name} has only {len(ref_xy)} stars")
        return 2

    print("=== validate_alignment_control_points ===")
    print(f"Reference: {ref_path}")
    print(
        f"Ref stars detected: {len(ref_xy)}  DAO sigma={det_sigma:.2f}  "
        f"FWHM={fwhm_px:.2f}px  det_max={args.det_max}"
    )
    print(f"Compare astroalign max_control_points: OLD={OLD_MCP} vs NEW={NEW_MCP}")
    print(
        f"Pass criteria: |dtranslation| < {PASS_TX_PX} px, |drotation| < {PASS_ROT_DEG} deg, "
        f"|dscale| < {PASS_SCALE}, |dRMS| < {PASS_RMS_DELTA_PX} px; NEW materially faster"
    )
    print()

    results: list[dict] = []
    for fp in frame_paths:
        print(f"--- {fp.name} ---")
        row = _compare_frame(
            fp,
            ref_xy,
            det_sigma=det_sigma,
            det_want=int(args.det_max),
            fwhm_px=fwhm_px,
        )
        results.append(row)
        print(
            f"  detected={row['detected']} fit={row['fit_points']}  "
            f"dtrans={row['delta_trans_px']:.4f}px drot={row['delta_rot_deg']:.5f}deg "
            f"dscale={row['delta_scale']:.6f} drms={row['delta_rms_px']:.4f}px"
        )
        print(
            f"  time old={row['time_old_s']:.3f}s new={row['time_new_s']:.3f}s "
            f"speedup={row['speedup']:.2f}x  "
            f"rms old={row['rms_old_px']:.4f} new={row['rms_new_px']:.4f}  "
            f"{'FAIL' if row['fail'] else 'ok'}"
        )

    max_dtrans = max(r["delta_trans_px"] for r in results)
    min_speedup = min(r["speedup"] for r in results)
    mean_speedup = float(np.mean([r["speedup"] for r in results]))
    any_fail = any(r["fail"] for r in results)

    print()
    print("=== SUMMARY ===")
    print(f"Frames tested: {len(results)}")
    print(f"Max |dtranslation| across frames: {max_dtrans:.4f} px (threshold {PASS_TX_PX})")
    print(f"Speedup NEW vs OLD: min={min_speedup:.2f}x mean={mean_speedup:.2f}x")
    theoretical = (OLD_MCP * (OLD_MCP - 1) * (OLD_MCP - 2)) / (
        NEW_MCP * (NEW_MCP - 1) * (NEW_MCP - 2)
    )
    print(f"Theoretical triangle-count ratio C({OLD_MCP},3)/C({NEW_MCP},3) ~ {theoretical:.1f}x")

    if any_fail:
        print("RESULT: FAIL - at least one frame exceeds sub-pixel / RMS thresholds.")
        print("Consider bumping alignment_max_control_points (e.g. 100) or keeping ladder escalation.")
        return 1
    if min_speedup < 1.2:
        print("RESULT: WARN - transforms agree but speedup < 1.2x (field may not be dense enough to benefit).")
        return 0
    print("RESULT: PASS - transforms agree within thresholds; NEW is materially faster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
