# -*- coding: ascii -*-
"""Delete named top-level functions from a Python module (AST line ranges)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def delete_funcs(path: Path, names: set[str]) -> list[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    ranges: list[tuple[int, int, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            start = int(node.lineno)
            end = int(node.end_lineno or node.lineno)
            ranges.append((start, end, node.name))
    if not ranges:
        return []
    lines = src.splitlines(keepends=True)
    # Drop preceding blank line when present.
    for i, (start, end, _n) in enumerate(ranges):
        idx = start - 1
        if idx > 0 and lines[idx - 1].strip() == "":
            ranges[i] = (start - 1, end, _n)
    keep = [True] * len(lines)
    removed = []
    for start, end, name in ranges:
        for i in range(start - 1, end):
            if 0 <= i < len(keep):
                keep[i] = False
        removed.append(f"{name}:{start}-{end}")
    new = "".join(ln for ln, k in zip(lines, keep, strict=True) if k)
    # Collapse 3+ blank lines.
    while "\n\n\n\n" in new:
        new = new.replace("\n\n\n\n", "\n\n\n")
    path.write_text(new, encoding="utf-8")
    return removed


def main() -> int:
    jobs = [
        (
            Path("src_py/psf_photometry.py"),
            {
                "_epsf_allowed_catalog_ids",
                "_epsf_positions_from_csvs",
                "_psf_fit_region_mask",
            },
        ),
        (
            Path("src_py/psf_runner.py"),
            {
                "step_1_build_epsf",
                "step_2_load_targets",
                "step_3_run_psf_on_frames",
                "step_4_build_summary",
                "step_5_calibrate_lightcurve",
            },
        ),
        (
            Path("src_py/photometry_core.py"),
            {
                "compute_snr_optimal_aperture_table",
                "_calibrate_snr_zero_point_for_draft",
                "_snr_table_radius_for_mag_bin",
                "_aperture_radius_from_snr_table",
                "_get_star_aperture_px",
                "resolve_draft_dir_for_snr_aperture_table",
                "load_snr_aperture_table_from_draft_dir",
                "_noise_floor_adu_from_image_array",
                "estimate_median_dao_fwhm_px_for_snr_table",
                "resolve_fwhm_px_for_snr_aperture_table",
                "estimate_median_sky_adu_per_px_for_snr_table",
                "estimate_star_free_median_sky_adu_per_px",
                "_load_star_xy_for_snr_ee",
                "_frame_data_for_snr_ee",
                "_measure_ee_curve_for_snr_table",
                "precompute_and_save_snr_aperture_table_for_draft",
                "_gaussian_ee_fraction",
                "_interp_ee_fraction",
                "_frame_dao_moment_fwhm_median_px",
                "_center_crop_with_offset",
                "_read_masterstar_fwhm_record_px",
            },
        ),
    ]
    for path, names in jobs:
        gone = delete_funcs(path, names)
        print(path, gone)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
