#!/usr/bin/env python3
"""Regenerate a draft PDF with layout verify mode and report overflow violations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
ROOT = _bootstrap.REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify PDF layout overflow guard")
    ap.add_argument("--draft", type=int, required=True, help="Draft id, e.g. 362")
    ap.add_argument("--obs-group", default="NoFilter_60_2")
    ap.add_argument(
        "--draft-root",
        type=Path,
        default=ROOT / "Archive" / "Drafts",
    )
    ap.add_argument("--output", type=Path, default=None, help="Output PDF path")
    ap.add_argument(
        "--spot-pages",
        default="2,3,4",
        help="Comma-separated 1-based page indices to render as PNG",
    )
    args = ap.parse_args()

    draft_dir = args.draft_root / f"draft_{args.draft:06d}"
    if not draft_dir.is_dir():
        print(f"ERROR: draft dir not found: {draft_dir}", file=sys.stderr)
        return 1

    from photometry_report import generate_photometry_report

    out = args.output
    if out is None:
        out = (
            draft_dir
            / "platesolve"
            / args.obs_group
            / f"VYVAR_report_{args.obs_group}_overflow_verify.pdf"
        )

    pdf = generate_photometry_report(
        draft_dir,
        args.obs_group,
        Path(out),
        verify_overflow=True,
    )
    if pdf is None:
        print("ERROR: generate_photometry_report returned None", file=sys.stderr)
        return 1

    from pypdf import PdfReader

    reader = PdfReader(str(pdf))
    n_pages = len(reader.pages)
    violations = int(getattr(generate_photometry_report, "last_overflow_violations", 0))

    result = {
        "pdf": str(pdf),
        "pages": n_pages,
        "overflow_violations": violations,
    }
    print(json.dumps(result, indent=2))

    spot = [int(x.strip()) for x in str(args.spot_pages).split(",") if x.strip()]
    spot_dir = Path(pdf).parent / "_overflow_spot"
    spot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf))
        for pnum in spot:
            if 1 <= pnum <= doc.page_count:
                page = doc.load_page(pnum - 1)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                png = spot_dir / f"page_{pnum:03d}.png"
                pix.save(str(png))
                print(f"spot-render: {png}")
        doc.close()
    except ImportError:
        print("spot-render: skipped (PyMuPDF/fitz not installed)")

    return 0 if violations == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
