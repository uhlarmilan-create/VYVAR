"""Compare science outputs between two BO CVn drafts (UI vs headless equivalence).

Usage:
    python dev/scripts/draft_ui_equivalence_check.py draft_000452 draft_000453

Prints file counts, per-file SHA256 mismatches, C.4 acceptance metrics, infolog extracts.
Any divergence is reported; do not average over differences.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ANCHOR_ACTIVE = (
    REPO
    / "Archive/Drafts/draft_000435_snapshot_skysurface_20260716/platesolve/NoFilter_60_2/photometry/active_targets.csv"
)

SCIENCE_GLOBS = (
    "photometry/lightcurves/lightcurve_*.csv",
    "photometry/comp_quality_*.json",
    "photometry/comparison_stars_per_target.csv",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ps(draft: Path) -> Path:
    return draft / "platesolve/NoFilter_60_2"


def _compare_science(a: Path, b: Path) -> dict:
    out: dict = {"draft_a": a.name, "draft_b": b.name, "patterns": {}}
    for pat in SCIENCE_GLOBS:
        fa = sorted(_ps(a).glob(pat))
        fb = sorted(_ps(b).glob(pat))
        names_a = {p.name: p for p in fa}
        names_b = {p.name: p for p in fb}
        all_names = sorted(set(names_a) | set(names_b))
        mismatches = []
        for nm in all_names:
            pa, pb = names_a.get(nm), names_b.get(nm)
            if pa is None or pb is None:
                mismatches.append({"file": nm, "reason": "missing_in_one_draft"})
                continue
            sa, sb = _sha256(pa), _sha256(pb)
            if sa != sb:
                mismatches.append({"file": nm, "sha_a": sa[:16], "sha_b": sb[:16]})
        out["patterns"][pat] = {
            "count_a": len(fa),
            "count_b": len(fb),
            "mismatches": mismatches,
        }
    return out


def _acceptance(draft: Path) -> dict:
    ps = _ps(draft)
    log = sorted(draft.glob("infolog_*.txt"))
    log_text = log[-1].read_text(encoding="utf-8", errors="replace") if log else ""
    ms = pd.read_csv(ps / "masterstars_full_match.csv")
    dao_only = int((ms["source_type"] == "DAO_ONLY").sum())
    mf = ps / "MASTERSTAR.fits"
    bg_std = sigma_pp = dao_threshold = float("nan")
    if mf.is_file():
        with fits.open(mf) as hd:
            d = hd[0].data.astype("float32")
        _, _, std = sigma_clipped_stats(d - float(np.nanmedian(d)), sigma=3, maxiters=3)
        bg_std = float(std)
        dao_threshold = 2.1 * bg_std
        dx = d[:, 1:] - d[:, :-1]
        dy = d[1:, :] - d[:-1, :]
        mad = float(np.median(np.abs(np.r_[dx.ravel(), dy.ravel()])))
        sigma_pp = mad / 0.6745 / (2**0.5)
    act = pd.read_csv(ps / "photometry/active_targets.csv")
    vt = pd.read_csv(ps / "variable_targets.csv")
    exo_n = 0
    if "exo_host_obj_id" in vt.columns:
        exo_n = int(vt["exo_host_obj_id"].notna().sum())
    dao_pass1 = None
    m = re.search(r"\[DAO pass 1\]\s*(\d+)\s*detections", log_text)
    if m:
        dao_pass1 = int(m.group(1))
    def _grep(pat: str) -> str:
        m2 = re.search(pat, log_text, re.MULTILINE)
        return m2.group(0).strip() if m2 else ""
    return {
        "dao_pass1": dao_pass1,
        "masterstars_rows": len(ms),
        "dao_only_frac": dao_only / len(ms) if len(ms) else float("nan"),
        "bg_std": bg_std,
        "sigma_pp_unmasked_mad": sigma_pp,
        "dao_threshold": dao_threshold,
        "active_targets": len(act),
        "exo_promoted_vt": exo_n,
        "infolog": log[-1].name if log else None,
        "VSX-GAIA XM": _grep(r"VSX-GAIA XM:.*"),
        "FAZA 0 funnel": _grep(r"FAZA 0 funnel:.*"),
        "INV-PREP-01": _grep(r"INV-PREP-01 Preprocess gradient guard.*"),
        "INV-MS-01": _grep(r"INV-MS-01 MASTERSTAR purity guard.*"),
        "EXO TARGET": _grep(r"\[EXO TARGET\].*"),
    }


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: draft_ui_equivalence_check.py <draft_a> <draft_b>", file=sys.stderr)
        sys.exit(2)
    a = REPO / "Archive/Drafts" / sys.argv[1]
    b = REPO / "Archive/Drafts" / sys.argv[2]
    if not a.is_dir() or not b.is_dir():
        print("Both draft directories must exist under Archive/Drafts/", file=sys.stderr)
        sys.exit(1)
    report = {
        "science_compare": _compare_science(a, b),
        "acceptance_a": _acceptance(a),
        "acceptance_b": _acceptance(b),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
