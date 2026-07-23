# Regenerates release/public_repo/INSTALL_VYVAR_EN.pdf and INSTALL_VYVAR_CZ.pdf.
# Run from repo root: python dev/tools/docs_pdf/build_install_vyvar_release.py
# -*- coding: ascii -*-
from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(os.getcwd())
OUT = ROOT / "release" / "public_repo"
S = getSampleStyleSheet()
B = S["Normal"]


def _md_to_pdf(md_path: Path, pdf_path: Path, title: str) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    story = [Paragraph(title, S["Title"]), Spacer(1, 8 * mm)]
    for raw in lines:
        text = raw.strip()
        if not text:
            story.append(Spacer(1, 3 * mm))
            continue
        if text.startswith("# "):
            story.append(Paragraph(text[2:], S["Heading1"]))
        elif text.startswith("## "):
            story.append(Paragraph(text[3:], S["Heading2"]))
        elif text.startswith("|"):
            story.append(Paragraph(text.replace("|", " "), B))
        else:
            story.append(Paragraph(text, B))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    doc.build(story)
    print("ok", pdf_path)


def main() -> None:
    _md_to_pdf(OUT / "INSTALL_VYVAR_EN.md", OUT / "INSTALL_VYVAR_EN.pdf", "VYVAR Install (EN)")
    _md_to_pdf(OUT / "INSTALL_VYVAR_CZ.md", OUT / "INSTALL_VYVAR_CZ.pdf", "VYVAR Instalace (CZ)")


if __name__ == "__main__":
    main()
