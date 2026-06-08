"""One-off VYVAR root directory audit (read-only)."""
import os
from pathlib import Path
from datetime import datetime

root = Path(r"C:\ASTRO\python\VYVAR")

root_files = []
for f in sorted(root.iterdir()):
    if f.is_file():
        stat = f.stat()
        age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
        root_files.append(
            {
                "name": f.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "age_days": round(age_days, 1),
                "ext": f.suffix,
            }
        )

print("=== ROOT FILES ===")
for f in root_files:
    print(f"  {f['name']:<50} {f['size_kb']:>8} KB  {f['age_days']:>6.1f} days")

print("\n=== SUBDIRECTORIES ===")
for d in sorted(root.iterdir()):
    if d.is_dir() and not d.name.startswith("."):
        py_files = list(d.rglob("*.py"))
        all_files = [f for f in d.rglob("*") if f.is_file()]
        print(f"  {d.name:<30} {len(all_files):>5} files  ({len(py_files)} .py)")

print("\n=== CANDIDATE CLEANUP FILES (root *.py) ===")
suspicious_patterns = [
    "_fix_",
    "_test_",
    "_diag_",
    "_debug_",
    "_check_",
    "_audit_",
    "_rerun_",
    "_draft",
    "test_",
    "fix_",
    "temp_",
    "tmp_",
    "wip_",
]
for f in root_files:
    if f["ext"] == ".py":
        is_suspicious = any(p in f["name"].lower() for p in suspicious_patterns)
        marker = " <- cleanup candidate" if is_suspicious else ""
        print(f"  {f['name']:<50}{marker}")

scripts_dir = root / "scripts"
if scripts_dir.exists():
    print(f"\n=== scripts/ DIR ===")
    for f in sorted(scripts_dir.iterdir()):
        if f.is_file():
            stat = f.stat()
            age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
            print(
                f"  {f.name:<50} {round(stat.st_size/1024, 1):>8} KB  {round(age_days, 1):>6.1f} days"
            )
