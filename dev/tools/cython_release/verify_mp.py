# -*- coding: ascii -*-
"""Multiprocessing spawn check: compiled modules pickle by reference on Windows."""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

from module_list import REPO_ROOT, SRC_PY

LOG = REPO_ROOT / "tmp" / "cython_release" / "mp_spawn.log"


def _worker(out_q: mp.Queue) -> None:
    if str(SRC_PY) not in sys.path:
        sys.path.insert(0, str(SRC_PY))
    import comp_selection_per_target as cst
    import photometry_core as pc

    out_q.put(
        {
            "comp_file": str(cst.__file__),
            "pc_file": str(pc.__file__),
            "comp_compiled": str(cst.__file__).endswith((".pyd", ".so")),
            "pc_compiled": str(pc.__file__).endswith((".pyd", ".so")),
        }
    )


def verify() -> dict[str, object]:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q,))
    p.start()
    p.join(timeout=120)
    if p.exitcode != 0:
        raise RuntimeError(f"spawn worker exit {p.exitcode}")
    result = q.get(timeout=10)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(f"{k}={v}" for k, v in result.items()) + "\n", encoding="ascii")
    return result


def main() -> None:
    r = verify()
    print("MP spawn verify:", r)
    if not r.get("comp_compiled") or not r.get("pc_compiled"):
        raise SystemExit("MP verify FAIL: expected compiled .pyd in worker")


if __name__ == "__main__":
    main()
