"""Every key the per-frame export worker reads from its state dict is written by export_per_frame_catalogs.

Guard for T10 G05-33: removing a worker-state key that is read only through ``st.get("key")`` fails
silently (the raw-frame peak loader returned None and the saturation columns were lost).
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src_py"
READ = re.compile(r"\bst(?:\.get\(|\[)\"([a-z_0-9]+)\"")
SET = re.compile(r"\bst\[\"([a-z_0-9]+)\"\]\s*=")


def test_worker_state_reads_are_written() -> None:
    readers = (SRC / "pipeline_catalog.py").read_text(encoding="utf-8")
    pre = (SRC / "pipeline_preprocess.py").read_text(encoding="utf-8")
    i = pre.index("def _load_raw_for_frame")
    readers += pre[i:pre.index("def load_qc_metrics_status_by_path")] if "def load_qc_metrics_status_by_path" in pre else pre[i:]
    reads = set(READ.findall(readers)) - set(SET.findall(readers))
    writer = (SRC / "frame_export.py").read_text(encoding="utf-8")
    written = set(re.findall(r"\"([a-z_0-9]+)\":", writer))
    assert "sat_diag_archive" in reads
    assert reads <= written, sorted(reads - written)
