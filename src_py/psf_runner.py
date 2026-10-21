"""Headless RUN ePSF on one draft / obs group (EPSF-SPATIAL-2026-10).

Thin CLI over ``epsf_stage.run_epsf_stage`` (INV-EPSF-STAGE-01): build-star selection on the
MASTERSTAR catalogue, per-frame ePSF + night spatial perturbation, PSF fit of the selected targets
(``--targets``, default config ``epsf_targets``) and their comparison stars, internal PSF light curves.

    python src_py/psf_runner.py --draft 535 --obs-group NoFilter_60_2 [--frames 10] [--skip-build] [--targets "EA, EW"]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from config import AppConfig
from database import VyvarDatabase


def _paths(archive_root: Path, draft_id: int, obs_group: str) -> dict[str, Path]:
    draft_dir = archive_root / "Drafts" / f"draft_{draft_id:06d}"
    ps_dir = draft_dir / "platesolve" / obs_group
    return {
        "ps_dir": ps_dir,
        "masterstar_fits": ps_dir / "MASTERSTAR.fits",
        "masterstars_csv": ps_dir / "masterstars_full_match.csv",
        "frames_dir": draft_dir / "detrended_aligned" / "lights" / obs_group,
    }


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = AppConfig()
    parser = argparse.ArgumentParser(description="VYVAR headless RUN ePSF")
    parser.add_argument("--draft", type=int, required=True, help="draft id")
    parser.add_argument("--obs-group", required=True, help="platesolve obs group, e.g. NoFilter_60_2")
    parser.add_argument("--archive", type=Path, default=Path(cfg.archive_root), help="archive root")
    parser.add_argument("--frames", type=int, default=None, help="only the first N frames (quick test)")
    parser.add_argument("--skip-build", action="store_true", help="reuse the selected build stars")
    parser.add_argument("--dry-run", action="store_true", help="print what would run")
    parser.add_argument("--targets", default=None, help="ALL or comma-separated VSX types / names / ids (default: config epsf_targets)")
    args = parser.parse_args(argv)

    from epsf_stage import EpsfStagePaths, run_epsf_stage

    p = _paths(args.archive, args.draft, args.obs_group)
    db_path = Path(cfg.database_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    db = VyvarDatabase(db_path)
    try:
        run_epsf_stage(
            params=None,
            paths=EpsfStagePaths(
                platesolve_dir=p["ps_dir"],
                frames_root=p["frames_dir"],
                masterstar_fits=p["masterstar_fits"],
                masterstars_csv=p["masterstars_csv"],
            ),
            cfg=cfg,
            progress_cb=print,
            db=db,
            draft_id=int(args.draft),
            do_build=not args.skip_build,
            max_frames=args.frames,
            dry_run=bool(args.dry_run),
            targets=args.targets,
        )
    finally:
        db.conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
