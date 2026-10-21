#!/usr/bin/env python3
"""Hand-repair non-dash U+FFFD / corrupted punctuation before ascii_migrate (BATCH-E STEP 6a)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "dev" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from ascii_migrate import _tracked_text_files, decode_text  # noqa: E402

PRIORITY = [
    "docs/VYVAR_MASTERSTAR_REFERENCE_ARCHITECTURE.md",
    "dev/results/CURSOR_RESULT_audit_t4.md",
    "dev/results/CURSOR_RESULT_dao_sigma_stability.md",
    "dev/results/CURSOR_RESULT_carry01_carry02.md",
    "dev/results/CURSOR_RESULT_audit_t1.md",
    "dev/results/CURSOR_RESULT_audit_t2.md",
    "dev/results/CURSOR_RESULT_audit_t3.md",
    "dev/results/CURSOR_RESULT_skysf_double.md",
    "dev/results/CURSOR_RESULT_audit_stage3_part0b.md",
    "dev/results/CURSOR_RESULT_sync_dev_to_github.md",
]

# (pattern, replacement) in application order
RULES: list[tuple[str, str]] = [
    (r"\?(\d+\.\d+) to \?(\d+\.\d+) deg", r"-\1 to -\2 deg"),
    (r"(\d\.\d+)\ufffd(\*\*)", r"\1x\2"),
    (r"\(~(\d+\.\d+)\ufffd\)", r"(~\1x)"),
    (r"3\.8\ufffd\?_pixel", "3.8*sigma_pixel"),
    (r"3\.8\ufffd\?_pp", "3.8*sigma_pp"),
    (r"3\.8\ufffd\?bkg2d", "3.8*sigma_bkg2d"),
    (r"3\.8\ufffd\?_", "3.8*sigma_"),
    (r"box_size \? 8\ufffdFWHM", "box_size ~ 8x FWHM"),
    (r"8\ufffdFWHM", "8x FWHM"),
    (r"2\ufffd amplif", "2x amplif"),
    (r"1\.25\ufffdlog", "1.25*log"),
    (r"([0-9a-f]{8})\ufffd(`)", r"\1...\2"),
    (r"([0-9a-f]{8})\ufffd n=", r"\1... n="),
    (r"([0-9a-f]{8})\ufffd;", r"\1...;"),
    (r"([0-9a-f]{8})\ufffd\)", r"\1...)"),
    (r"=06ed950\ufffd", "=06ed950..."),
    (r"rel_err\?1\.36", "rel_err~1.36"),
    (r"0\.39\?0\.45", "0.39-0.45"),
    (r"e of 3\.8\ufffd\?_pixel", "e of 3.8*sigma_pixel"),
    (r"ratio \?_pp", "ratio sigma_pp"),
    (r"max/min \? 1", "max/min ~ 1"),
    (r"cal \? pre", "cal -> pre"),
    (r"42\.5 \? 30\.6", "42.5 -> 30.6"),
    (r"\| \ufffd \|", "| - |"),
    (r"\?_pp ~45\ufffd52", "sigma_pp ~45-52"),
    (r"\?_pp ~45\?52", "sigma_pp ~45-52"),
    (r"\?_pp", "sigma_pp"),
    (r"\?bkg2d", "sigma_bkg2d"),
    (r"\?_pixel", "sigma_pixel"),
]


def repair_text(text: str, rel: str) -> tuple[str, list[str]]:
    log: list[str] = []
    out = text
    for pat, repl in RULES:
        new = re.sub(pat, repl, out)
        if new != out:
            for m in re.finditer(pat, out):
                line = out[: m.start()].count("\n") + 1
                before = out[max(0, m.start() - 12) : m.end() + 12].replace("\n", " ")
                after = re.sub(pat, repl, out[max(0, m.start() - 12) : m.end() + 12], count=1)
                log.append(f"{rel}:{line}: {before!r} -> {after!r}")
            out = new
    return out, log


def main() -> int:
    targets = {Path(p) for p in PRIORITY}
    all_logs: list[str] = []
    for rel in sorted(targets):
        path = ROOT / rel
        if not path.is_file():
            print(f"skip missing {rel}")
            continue
        text = decode_text(path.read_bytes())[0]
        new, log = repair_text(text, rel.as_posix())
        if new != text:
            path.write_text(new, encoding="utf-8", newline="\n")
            all_logs.extend(log)
    report = ROOT / "tmp" / "hand_repair_log.txt"
    report.write_text("\n".join(all_logs) + f"\n\ntotal repairs: {len(all_logs)}\n", encoding="ascii")
    print(f"hand-repaired {len(all_logs)} occurrences; log {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
