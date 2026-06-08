"""Execute approved cleanup phases 1, 2, 4, 5, 6 (read-only report via prints)."""
from pathlib import Path

root = Path(r"C:\ASTRO\python\VYVAR")
scripts = root / "scripts"

# --- Phase 1 ---
deleted = []
patterns = [
    "*.log",
    "*_verify_console.txt",
    "*_console.txt",
    "*_verify_live.txt",
    "todo8_run2_only.txt",
    "_comp_refs_output.txt",
    "_comp_sel_body.txt",
    "_p2a_body_indented.txt",
    "_p2a_extract.txt",
    "_grep_photometry_core_calls.txt",
    "app_head_test.txt",
    "diff_cursor.txt",
    "hard_copy_phase2a_csv_only.*",
    "_draft343_run_config.json",
]

for pattern in patterns:
    for f in root.glob(pattern):
        if f.is_file():
            size = f.stat().st_size
            deleted.append((f.name, size))
            f.unlink()

total_mb = sum(s for _, s in deleted) / 1024 / 1024
print(f"Phase 1: deleted {len(deleted)} files, {total_mb:.1f} MB freed")
for name, size in sorted(deleted, key=lambda x: -x[1])[:10]:
    print(f"  {size // 1024:6d} KB  {name}")

# --- Phase 2 ---
tmp_files = list(root.glob("_tmp_*.py"))
for f in tmp_files:
    f.unlink()
print(f"Phase 2: deleted {len(tmp_files)} _tmp_*.py files")

# --- Phase 4 ---
archive = root / "scripts" / "archive"
subdirs = ["verify", "draft_runs", "fixes", "diag", "plots"]
for s in subdirs:
    (archive / s).mkdir(parents=True, exist_ok=True)

moves = {
    "verify": list(scripts.glob("_alg*_verify_*.py"))
    + list(scripts.glob("_todo*_verify_*.py"))
    + list(scripts.glob("_smoke_*_draft*.py"))
    + list(scripts.glob("_phase2a_only_*.py"))
    + list(scripts.glob("_reexport_*.py")),
    "draft_runs": list(scripts.glob("_complete_draft*.py"))
    + list(scripts.glob("_simulate_night_run_draft*.py"))
    + list(scripts.glob("_validate_draft*.py"))
    + list(scripts.glob("_rerun_phase01_*.py"))
    + list(scripts.glob("_gs11_*.py"))
    + list(scripts.glob("_draft3*.py")),
    "fixes": list(scripts.glob("fix_*.py"))
    + list(scripts.glob("patch_*.py"))
    + list(scripts.glob("gen_comp_*.py"))
    + list(scripts.glob("build_comp_*.py")),
    "diag": list(scripts.glob("diag_*.py"))
    + list(scripts.glob("_diag_*.py"))
    + list(scripts.glob("_dao_count_*.py"))
    + list(scripts.glob("_debug_*.py"))
    + list(scripts.glob("_comp_consistency_diag.py")),
    "plots": list(scripts.glob("plot_*.py")),
}

total_moved = 0
for subdir, files in moves.items():
    for f in files:
        dest = archive / subdir / f.name
        if not dest.exists():
            f.rename(dest)
            total_moved += 1

print(f"Phase 4: moved {total_moved} scripts to archive/")
for s in subdirs:
    count = len(list((archive / s).glob("*.py")))
    print(f"  archive/{s}/: {count} files")

# --- Phase 5 ---
diag_archive = archive / "diagnostics"
diag_archive.mkdir(exist_ok=True)

root_diags = [
    "diagnose_crowding_filters.py",
    "diagnose_ensemble.py",
    "diagnose_flux.py",
    "masterstar_wcs_dao_diagnostic.py",
    "generate_report_303.py",
    "plot_top_candidates_lightcurves.py",
    "debug_qc.py",
    "_audit_issues.py",
]

moved5 = []
for name in root_diags:
    src = root / name
    if src.exists():
        dst = diag_archive / name
        src.rename(dst)
        moved5.append(name)

print(f"Phase 5: moved {len(moved5)} root diagnostics to scripts/archive/diagnostics/")

# --- Phase 6 ---
src = root / "smoke_test_bprp_tier.py"
dst = root / "tests" / "test_bprp_tier_smoke.py"
if src.exists():
    src.rename(dst)
    print("Phase 6: moved smoke_test_bprp_tier.py → tests/test_bprp_tier_smoke.py")
else:
    print("Phase 6: smoke_test_bprp_tier.py not found (skipped)")

# --- Final count ---
root_files = [f for f in root.iterdir() if f.is_file()]
print(f"\nFinal root file count: {len(root_files)}")
