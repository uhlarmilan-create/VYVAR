"""FRAME-QC-PARITY-01 phase 1: measure draft 516 vs 517 QC artifacts (read-only)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
DRAFTS = REPO / "Archive" / "Drafts"


def _read_qc(draft: str) -> pd.DataFrame:
    qc = pd.read_csv(DRAFTS / draft / "calibrated" / "lights" / "qc_metrics.csv")
    qc["frame"] = qc["src"].map(lambda s: Path(str(s)).name)
    return qc


def _headers(draft: str) -> pd.DataFrame:
    cal = DRAFTS / draft / "calibrated" / "lights" / "NoFilter_60_2"
    rows = []
    for fp in sorted(cal.glob("BO_CVn_Light_*.fits")):
        with fits.open(fp) as hd:
            h = hd[0].header
        rows.append(
            {
                "frame": fp.name,
                "VYQCPASS": h.get("VYQCPASS"),
                "VY_QCHFR": float(h.get("VY_QCHFR")) if h.get("VY_QCHFR") is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _infolog_hfr(draft: str) -> dict[str, float]:
    logs = sorted((DRAFTS / draft).glob("infolog_*.txt"))
    if not logs:
        return {}
    text = logs[-1].read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {}
    for m in re.finditer(
        r"Frame (BO_CVn_Light_\d+\.fits) REJECTED \(HFR: ([0-9.]+) > limit ([0-9.]+)\)",
        text,
    ):
        out[m.group(1)] = float(m.group(2))
    return out


def main() -> None:
    qc516 = _read_qc("draft_000516")
    qc517 = _read_qc("draft_000517")
    h516 = _headers("draft_000516")
    h517 = _headers("draft_000517")
    merged_qc = qc516.merge(qc517, on="frame", suffixes=("_516", "_517"))
    status_diff = merged_qc[merged_qc["status_516"] != merged_qc["status_517"]]
    merged_h = h516.merge(h517, on="frame", suffixes=("_516", "_517"))
    header_diff = merged_h[
        (merged_h["VYQCPASS_516"] != merged_h["VYQCPASS_517"])
        | (merged_h["VY_QCHFR_516"] != merged_h["VY_QCHFR_517"])
    ]

    rej517 = qc517[qc517["status"] != "ok"].copy()
    rej517 = rej517.merge(h517, on="frame")
    log_hfr = _infolog_hfr("draft_000517")
    rej517["infolog_hfr_px"] = rej517["frame"].map(log_hfr)

    proc516 = len(list((DRAFTS / "draft_000516" / "detrended_aligned" / "lights" / "NoFilter_60_2").glob("proc_*.csv")))
    proc517 = len(list((DRAFTS / "draft_000517" / "detrended_aligned" / "lights" / "NoFilter_60_2").glob("proc_*.csv")))

    out = {
        "qc_metrics_rows_516": int(len(qc516)),
        "qc_metrics_rows_517": int(len(qc517)),
        "status_ok_516": int((qc516["status"] == "ok").sum()),
        "status_ok_517": int((qc517["status"] == "ok").sum()),
        "prefilter_rejected_516": int((qc516["status"] != "ok").sum()),
        "prefilter_rejected_517": int((qc517["status"] != "ok").sum()),
        "status_diff_count": int(len(status_diff)),
        "header_diff_count": int(len(header_diff)),
        "vyqcpass_false_516": int((h516["VYQCPASS"] == False).sum()),  # noqa: E712
        "vyqcpass_false_517": int((h517["VYQCPASS"] == False).sum()),  # noqa: E712
        "infolog_hfr_rejections_517": len(log_hfr),
        "proc_csv_516": proc516,
        "proc_csv_517": proc517,
        "prefilter_rejected_frames": rej517[
            ["frame", "status", "fwhm_px", "VY_QCHFR", "VYQCPASS", "infolog_hfr_px"]
        ].to_dict(orient="records"),
    }
    out_path = REPO / "dev" / "results" / "context" / "session_20260821_frame_qc_parity" / "measurements.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "prefilter_rejected_frames"}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
