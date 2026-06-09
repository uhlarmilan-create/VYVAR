"""Validation matrix scoring -> validation_report.json + validation_report.md."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HONESTY_CAVEAT = (
    "Synthetic frames test what the generator injects (Moffat PSF + Poisson/read noise + "
    "gradient): a faithful but simplified sky. The harness validates LOGIC and recovery "
    "(does comp_qa flag the bad comp? does the solver recover WCS? does the trust gate gate?), "
    "not every real systematic (scintillation, true flat residuals, detector nonlinearity). "
    "Necessary, not sufficient -- real-data drafts remain the final word."
)

RNG_SEEDS = {
    "gen_frame": 42,
    "gen_series": 43,
    "gen_a9": 44,
    "v3d_fine": 367,
    "v3e_epsf": 370,
    "cr_pixels": 42,
    "series_variability": 43,
}


@dataclass
class ValidationItem:
    id: str
    description: str
    function_under_test: str
    citation: str
    expected: str
    recovered: str
    delta: str
    status: str  # PASS | FAIL | SKIP
    note: str = ""


@dataclass
class ValidationReport:
    items: list[ValidationItem] = field(default_factory=list)
    seeds: dict[str, int] = field(default_factory=lambda: dict(RNG_SEEDS))
    generated_at: str = ""
    honesty_caveat: str = HONESTY_CAVEAT

    def add(
        self,
        item_id: str,
        description: str,
        function_under_test: str,
        citation: str,
        *,
        expected: Any,
        recovered: Any,
        delta: Any = "",
        status: str,
        note: str = "",
    ) -> None:
        self.items.append(
            ValidationItem(
                id=item_id,
                description=description,
                function_under_test=function_under_test,
                citation=citation,
                expected=str(expected),
                recovered=str(recovered),
                delta=str(delta),
                status=status.upper(),
                note=note,
            )
        )

    def summary(self) -> tuple[int, int, int]:
        n_pass = sum(1 for i in self.items if i.status == "PASS")
        n_fail = sum(1 for i in self.items if i.status == "FAIL")
        n_skip = sum(1 for i in self.items if i.status == "SKIP")
        return n_pass, n_fail, n_skip

    def to_dict(self) -> dict[str, Any]:
        n_pass, n_fail, n_skip = self.summary()
        return {
            "generated_at": self.generated_at,
            "honesty_caveat": self.honesty_caveat,
            "seeds": self.seeds,
            "summary": {"pass": n_pass, "fail": n_fail, "skip": n_skip, "total": len(self.items)},
            "items": [asdict(i) for i in self.items],
        }

    def write_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="ascii") as f:
            json.dump(self.to_dict(), f, indent=2)

    def write_md(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n_pass, n_fail, n_skip = self.summary()
        lines = [
            "# VYVAR validation report",
            "",
            f"Generated: {self.generated_at}",
            "",
            "> " + self.honesty_caveat,
            "",
            f"**Summary:** {n_pass} pass / {n_fail} fail / {n_skip} skip "
            f"(total {len(self.items)})",
            "",
            "RNG seeds: " + ", ".join(f"{k}={v}" for k, v in sorted(self.seeds.items())),
            "",
            "| id | status | expected | recovered | delta | note |",
            "|----|--------|----------|-----------|-------|------|",
        ]
        for it in self.items:
            note = it.note.replace("|", "/").replace("\n", " ")
            if len(note) > 120:
                note = note[:117] + "..."
            lines.append(
                f"| {it.id} | {it.status} | {it.expected} | {it.recovered} | "
                f"{it.delta} | {note} |"
            )
        lines.extend(["", "## Fail diagnoses", ""])
        fails = [i for i in self.items if i.status == "FAIL"]
        if not fails:
            lines.append("(none)")
        else:
            for it in fails:
                lines.append(f"### {it.id} -- {it.description}")
                lines.append(f"- Function: `{it.function_under_test}` ({it.citation})")
                lines.append(f"- Expected: {it.expected}")
                lines.append(f"- Recovered: {it.recovered}")
                if it.note:
                    lines.append(f"- Diagnosis: {it.note}")
                lines.append("")
        with open(path, "w", encoding="ascii") as f:
            f.write("\n".join(lines) + "\n")

    def finalize(self, out_dir: Path) -> tuple[Path, Path]:
        self.generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out_dir = Path(out_dir)
        jp = out_dir / "validation_report.json"
        mp = out_dir / "validation_report.md"
        self.write_json(jp)
        self.write_md(mp)
        return jp, mp
