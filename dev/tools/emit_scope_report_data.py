"""Emit rig-list markdown for CURSOR_RESULT_params_scope_audit.md."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "tools"))
import params_registry as pr
from classify_params_scope import EXPLICIT, _classify

reg = pr.load_registry()
print("count", len(reg))
print("overall", dict(Counter(e["scope"] for e in reg.values())))
for ph in pr.PHASES:
    c = Counter(e["scope"] for k, e in reg.items() if e["phase"] == ph)
    if c:
        print(f"{ph}: {dict(c)}")
print("---RIG---")
for k in sorted(k for k, e in reg.items() if e["scope"] == "rig"):
    base = {kk: vv for kk, vv in reg[k].items() if kk not in ("scope", "scope_confidence")}
    j = EXPLICIT[k][2] if k in EXPLICIT else _classify(k, base)[2]
    print(f"{k}\t{reg[k]['scope_confidence']}\t{j}")
