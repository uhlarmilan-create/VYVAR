"""E2E pipeline simulation for a post-observation session (e.g. D:\\BO_CVn).

Usage:
    python simulate_night_run.py
    python simulate_night_run.py --dry-run
    python simulate_night_run.py --source D:\\BO_CVn --eq 1 --tel 1

Simulates the VYVAR run equivalent to UI **Session Upload Automation -> RUN VYVAR**.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Defaults from vyvar.sqlite3 (QHY294MM / Carl-Zeiss 200mm)
_DEFAULT_EQUIPMENT_ID = 1
_DEFAULT_TELESCOPE_ID = 1


def _setup_logging(log_path: Path) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        # EXC-0468: ? -- intent unclear (sys.stdout.reconfigure(encoding='utf-8') / sys.stderr.reconfigure(encod... (EXCEPT-BULK 2026-07-08)
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="VYVAR night run simulation")
    parser.add_argument(
        "--source",
        default=r"D:\BO_CVn",
        help="Source directory with FITS files",
    )
    parser.add_argument(
        "--eq",
        type=int,
        default=_DEFAULT_EQUIPMENT_ID,
        help=f"Equipment DB ID (default {_DEFAULT_EQUIPMENT_ID}=QHY294MM Camera1)",
    )
    parser.add_argument(
        "--tel",
        type=int,
        default=_DEFAULT_TELESCOPE_ID,
        help=f"Telescope DB ID (default {_DEFAULT_TELESCOPE_ID}=Carl-Zeiss 200mm)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.json (default: project config.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan source only; no import or pipeline execution",
    )
    parser.add_argument(
        "--no-sysrem",
        action="store_true",
        help="Disable SysRem (default: enabled)",
    )
    parser.add_argument(
        "--sysrem-iter",
        type=int,
        default=3,
        help="SysRem iterations when enabled (default: 3)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("simulate_night_run.log"),
        help="Log file path (default: simulate_night_run.log in cwd)",
    )
    args = parser.parse_args()

    _setup_logging(args.log.resolve())

    from night_run import NightRunParams, run_night_pipeline

    sysrem_on = not args.no_sysrem
    logging.info("=" * 60)
    logging.info("VYVAR Night Run Simulation")
    logging.info("Source:    %s", args.source)
    logging.info("Equipment: ID=%d", args.eq)
    logging.info("Telescope: ID=%d", args.tel)
    logging.info("SysRem:    %s", "disabled via --no-sysrem" if args.no_sysrem else "from config.json")
    logging.info("Dry run:   %s", args.dry_run)
    logging.info("Log file:  %s", args.log.resolve())
    logging.info("=" * 60)

    params = NightRunParams(
        source_dir=Path(args.source),
        equipment_id=int(args.eq),
        telescope_id=int(args.tel),
        config_path=args.config,
        sysrem_enabled=False if args.no_sysrem else None,
        sysrem_n_iter=int(args.sysrem_iter) if args.no_sysrem is False else None,
        dry_run=bool(args.dry_run),
        progress_cb=lambda msg: logging.info("[Progress] %s", msg),
    )

    t_start = time.time()
    result = run_night_pipeline(params)
    elapsed = time.time() - t_start

    logging.info("=" * 60)
    if result.success:
        logging.info("SUCCESS in %.1fs", elapsed)
        logging.info("Draft:         %s", result.draft_id)
        logging.info("Draft dir:     %s", result.draft_dir)
        logging.info("Output dir:    %s", result.output_dir)
        logging.info("Light curves:  %d", result.n_lightcurves)
        logging.info("Frames:        %d", result.n_frames)
        if result.lc_rms_median == result.lc_rms_median:
            logging.info("LC RMS median: %.4f", result.lc_rms_median)
        if result.sysrem_improvement_pct == result.sysrem_improvement_pct:
            logging.info("SysRem improvement: %.1f%%", result.sysrem_improvement_pct)
    else:
        logging.error("FAILED in %.1fs", elapsed)
        for err in result.errors:
            logging.error("  ERROR: %s", err)

    for warn in result.warnings:
        logging.warning("  WARN: %s", warn)

    logging.info("--- Phase timings ---")
    for phase, sec in result.phase_timings.items():
        logging.info("  %-35s %.1fs", phase, sec)
    logging.info("=" * 60)

    logging.info("--- PERF verification (grep %s) ---", args.log.name)
    logging.info("Expected markers when full run completes:")
    logging.info("  [ProcFrameStore] Built:")
    logging.info("  [PERF-2] MASTERSTAR.fits loaded once")
    logging.info("  [PERF-3] Comp Gaia prefetch:")
    logging.info("  [PERF-4] comp_pool_rms:")
    logging.info("  [PERF-5] run_phase2a: using ProcFrameStore")
    logging.info("  [PERF-6] load_field_flux_matrix:")
    logging.info("  [SysRem]")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
