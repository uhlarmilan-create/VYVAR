"""One git commit per module for 3b dead-function deletions."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
spans = json.loads((OUT / "dead_spans.json").read_text(encoding="utf-8"))
SKIP = {"psf_photometry", "psf_runner"}


def main() -> int:
    for m in spans["modules"]:
        mod = m["module"]
        if mod in SKIP:
            continue
        names = ", ".join(f["name"] for f in m["funcs"])
        path = f"src_py/{mod}.py"
        subprocess.check_call(["git", "add", "--", path], cwd=ROOT)
        subject = f"CONSOLIDATE-01A: drop dead functions in {mod}: {names}."
        if len(subject) > 180:
            msg = f"CONSOLIDATE-01A: drop dead functions in {mod}.\n\n{names}."
        else:
            msg = subject
        subprocess.check_call(["git", "commit", "-m", msg], cwd=ROOT)
        print("committed", mod)
    return 0


if __name__ == "__main__":
    sys.exit(main())
