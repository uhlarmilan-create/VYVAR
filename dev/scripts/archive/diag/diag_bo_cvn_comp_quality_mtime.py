from __future__ import annotations

import datetime
import json
from pathlib import Path


def _print_safe(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "backslashreplace").decode("ascii"))


def main() -> None:
    phot_dir = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000278\platesolve\NoFilter_60_2\photometry")
    bo_cid = "1498613634033133184"

    data = json.loads((phot_dir / "lightcurves" / f"comp_quality_{bo_cid}.json").read_text(encoding="utf-8"))
    good = [k for k, v in data.items() if v == "good"]
    _print_safe(f"good comp v JSON: {len(good)}")
    statuses = {v for v in data.values() if isinstance(v, str)}
    _print_safe(f"vsetky stavy: {statuses}")

    files = {
        "comparison_stars_per_target.csv": phot_dir / "comparison_stars_per_target.csv",
        "comp_quality JSON": phot_dir / "lightcurves" / f"comp_quality_{bo_cid}.json",
        "photometry_summary.csv": phot_dir / "photometry_summary.csv",
    }
    for name, path in files.items():
        if path.exists():
            mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
            _print_safe(f"{name}: {mtime:%Y-%m-%d %H:%M:%S}")
        else:
            _print_safe(f"{name}: MISSING")


if __name__ == "__main__":
    main()

