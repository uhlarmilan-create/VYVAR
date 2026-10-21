"""Emit scope remediation data for CURSOR_RESULT_params_scope_remediation.md."""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "tools"))
import params_registry as pr
from classify_params_scope import EXPLICIT

reg = pr.load_registry()
print("count", len(reg))
print("overall", dict(Counter(e["scope"] for e in reg.values())))
print("scope_key", dict(Counter(e["scope_key"] for e in reg.values())))
rg = Counter(e["scope_group"] for e in reg.values() if e["scope"] == "rig")
print("rig_groups", dict(rg))
lc = sorted(k for k, e in reg.items() if e["scope_confidence"] == "low")
print("low", len(lc))
for k in lc:
    print(f"  {k}")
print("---rig---")
for k in sorted(k for k, e in reg.items() if e["scope"] == "rig"):
    e = reg[k]
    note = EXPLICIT[k].note if k in EXPLICIT else "(mechanical)"
    print(f"{k}\t{e['scope_group']}\t{e['scope_key']}\t{e['scope_confidence']}\t{note}")
