#!/usr/bin/env python3
"""CONSOLIDATE-01D: fill class/proposal/risk/d2_evidence. No config deletions."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE / "config_prerez_table.json"
MD = HERE / "d1_table.md"

SKIP = {"src_py/config.py"}

# Loader-migrated aliases that ARE config_runtime fields.
ALIAS = {
    "comp_iterative_clip_enabled": {
        "target": "comp_sparse_fallback_enabled",
        "loc": "config.py:995 loader 2073-2082",
        "d2": "After load, field is forced equal to comp_sparse_fallback_enabled. Default JSON False is overwritten. Flip of this key alone with sparse True is a no-op. Removable alias.",
    },
}

# Flip-only LEGACY switches (default is production; other value is superseded).
LEGACY_FLIP = {
    "global_comp_pool_enabled": {
        "superseded": "COMP-POOL-01 (default True = production global pool)",
        "legacy_value": "False",
        "loc": "photometry_core.py:16257",
        "d2": "Default True. False skips build_global_comp_pool and uses per-target masterstars. Flip-only; not data-conditioned. run_night_pipeline / Phase 0+1 takes True at default config.",
    },
    "export_err_mode": {
        "superseded": "calibrated (s, sigma_r); model = legacy quadrature",
        "legacy_value": "model",
        "loc": "photometry_core.py:10171",
        "d2": "Default calibrated. model skips ERR-CALIB apply. Flip-only. Production LC export uses calibrated at default.",
    },
    "err_background_mode": {
        "superseded": "F-BINGAIN-1 empirical empty-aperture; howell = full Howell variance",
        "legacy_value": "howell",
        "loc": "photometry_core.py:1299-1325 pipeline.py:10628",
        "d2": "Default empirical. Config value howell is flip-only (skips empirical term). Howell MATH remains reachable under empirical when sigma_bkg_ap is missing (howell_fallback) -- that is a data condition, not the key's howell value. Removing the howell KEY does not delete fallback math.",
    },
    "masterstar_accept_mode": {
        "superseded": "odds (Bayesian); fraction = legacy",
        "legacy_value": "fraction",
        "loc": "vyvar_platesolver.py:3711",
        "d2": "Default odds. fraction is flip-only in _masterstar_solve_acceptance. Production MASTERSTAR solve uses odds at default.",
    },
    "psf_ac_policy": {
        "superseded": "p4_none default; chi2_lt5_legacy is ePSF AC",
        "legacy_value": "chi2_lt5_legacy",
        "loc": "psf_photometry.py:410 pipeline.py:559",
        "d2": "Default p4_none. chi2_lt5_legacy is flip-only. On ePSF graph; phase 2 of this row needs --full-epsf.",
        "risk": "phase 2 of this row needs --full-epsf",
    },
}

# Documented as |dB-V| switch; production selection does not read it.
STALE_SWITCH = {
    "phase01_use_bprp_primary": {
        "class": "DIAG",
        "proposal": "needs-Milan",
        "risk": "Comment says False restores |dB-V| via comp_tier*_bv_limit (gone; those keys are _LEGACY_CONFIG_KEYS). Live readers are photometry_report.py:789 and ui_aperture_photometry.py:1703 (display). comp_selection_per_target does not read the flag.",
        "d2_evidence": "Default True. False does not restore |dB-V| selection. Flip only changes report/UI BV column hiding. Question: delete key as display-only, or restore a real |dB-V| branch?",
    },
}

# px / ratio pairs: default None on preferred unit => px path ALWAYS taken.
PX_PAIRS = {
    "qc_max_hfr": {
        "preferred": "qc_max_hfr_fwhm_ratio",
        "d2": "Default qc_max_hfr=5.0, ratio=None. unit_resolver.py:142-146 takes legacy px at default. Data/config condition (None) => MODE not removable-with-key.",
    },
    "qc_max_hfr_fwhm_ratio": {
        "fallback": "qc_max_hfr",
        "d2": "Default None. Setting it switches QC HFR to ratio x FWHM. Unused at default. MODE.",
    },
    "hrd_color_bg_box_px": {
        "preferred": "hrd_color_bg_box_arcsec",
        "d2": "Default px=96, arcsec=None. unit_resolver.hrd_color_bg_box_px takes px at default. MODE.",
    },
    "hrd_color_bg_box_arcsec": {
        "fallback": "hrd_color_bg_box_px",
        "d2": "Default None. MODE unused at default.",
    },
    "masterstar_centre_rms_max_px": {
        "preferred": "masterstar_centre_rms_max_arcsec",
        "d2": "Default px=1.20, arcsec=None. unit_resolver.masterstar_centre_rms_max_px takes px at default. MODE.",
    },
    "masterstar_centre_rms_max_arcsec": {
        "fallback": "masterstar_centre_rms_max_px",
        "d2": "Default None. MODE unused at default.",
    },
}

# Field name hidden behind a method; grep of the key under-counts.
WRAPPER_ACTIVE = {
    "phase01_tiers": {
        "class": "ACTIVE",
        "d2": "photometry_core.py:17083 calls cfg.phase01_tier_mags() which reads phase01_tiers (config.py:2641). Production Phase 0+1 mag-tier bounds. Naive key grep is config-only; NOT DEAD.",
    },
    "comp_color_tiers": {
        "class": "ACTIVE",
        "d2": "comp_selection_per_target.py:265 calls cfg.comp_tier_bprp_limits() from this field (config.py:2634). Production colour tiers. UI also edits the structured key.",
    },
    "comp_sparse_fallback_enabled": {
        "class": "MODE",
        "d2": "photometry_core.py:15073 via resolve_comp_sparse_fallback_enabled (config.py:244). Default True. Production sparse fallback. Key-grep misses pipeline because it uses the resolver.",
    },
    "comp_sparse_fallback_min": {
        "class": "ACTIVE",
        "d2": "photometry_core.py:15061 via resolve_comp_sparse_fallback_min. Production. Key-grep misses pipeline.",
    },
}

# Spec said alias of masterstar_dao_threshold_sigma; code loads both independently.
NOT_ALIAS = {
    "dao_detection_n_equiv": {
        "class": "ACTIVE",
        "proposal": "keep",
        "risk": "STOP vs spec: not a loader alias. Detection uses masterstar_dao_threshold_sigma (config.py:778). This field is zone-classifier T1 (pipeline.py:6226, 7307). Loaded independently (config.py:2142). Can diverge.",
        "d2_evidence": "Live on production MASTERSTAR zone path with default config. Not flip-only.",
    },
}

DEAD_CONFIRMED = {
    "qc_fwhm_limit": "config.py:736 persist only; night_run uses auto_fwhm / DB prefilter",
    "qc_elong_limit": "config.py:738 persist only; no QC reader",
    "psf_spatial_grid": "config.py:712 persist only; psf_photometry uses psf_spatial_order/enabled",
    "psf_spatial_min_stars_per_cell": "config.py:714 persist only",
    "gs11_comp_suspect_dilution": "config.py:1003 persist + clamp vs gs11_comp_max_dilution; production uses gs11_comp_max_dilution (photometry_core.py:15407)",
}

DIAG_FORCE = {
    "debug_platesolver": "platesolver verbose logs",
}


def prod_consumers(cons: list[str]) -> list[str]:
    out = []
    for c in cons:
        f = c.split(":")[0]
        if f in SKIP or "params_registry" in f:
            continue
        out.append(c)
    return out


def classify(row: dict) -> None:
    k = row["key"]
    td = row["type_default"]
    pc = prod_consumers(row["consumers"])

    if k in ALIAS:
        info = ALIAS[k]
        row["class"] = "ALIAS"
        row["proposal"] = "remove-alias"
        row["risk"] = f"loader-migrated alias of {info['target']} ({info['loc']})"
        row["d2_evidence"] = info["d2"]
        return

    if k in NOT_ALIAS:
        info = NOT_ALIAS[k]
        row.update({kk: info[kk] for kk in ("class", "proposal", "risk", "d2_evidence")})
        return

    if k in STALE_SWITCH:
        info = STALE_SWITCH[k]
        row.update({kk: info[kk] for kk in ("class", "proposal", "risk", "d2_evidence")})
        return

    if k in LEGACY_FLIP:
        info = LEGACY_FLIP[k]
        row["class"] = "LEGACY"
        row["proposal"] = "remove-key+dead-branch"
        row["risk"] = info.get("risk", f"legacy value {info['legacy_value']}; {info['superseded']}")
        row["d2_evidence"] = info["d2"]
        return

    if k in PX_PAIRS:
        info = PX_PAIRS[k]
        row["class"] = "MODE"
        row["proposal"] = "needs-Milan"
        if "preferred" in info:
            row["risk"] = (
                f"px path is the default production path; preferred {info['preferred']} is None. "
                "Is px still wanted as default, or should the preferred unit be set?"
            )
        else:
            row["risk"] = (
                f"preferred unit; unused at default (None). Fallback {info['fallback']} is live. "
                "Set this and drop px, or keep both?"
            )
        row["d2_evidence"] = info["d2"]
        return

    if k in WRAPPER_ACTIVE:
        info = WRAPPER_ACTIVE[k]
        row["class"] = info.get("class", "ACTIVE")
        row["proposal"] = "keep"
        row["risk"] = "live via method/resolver wrapper; do not treat as DEAD/DIAG"
        row["d2_evidence"] = info["d2"]
        return

    if k in DEAD_CONFIRMED:
        row["class"] = "DEAD"
        row["proposal"] = "remove-key+dead-branch"
        row["risk"] = DEAD_CONFIRMED[k]
        row["d2_evidence"] = "zero src_py consumers outside config.py load/save; getattr scan negative"
        return

    if k in DIAG_FORCE:
        row["class"] = "DIAG"
        row["proposal"] = "keep"
        row["risk"] = DIAG_FORCE[k]
        row["d2_evidence"] = "diagnostic switch"
        return

    if not pc:
        row["class"] = "DEAD"
        row["proposal"] = "remove-key+dead-branch"
        row["risk"] = "zero consumers outside config.py at tip (verify getattr)"
        row["d2_evidence"] = "key-grep empty outside config.py"
        return

    ui_only = all(
        ("/ui_" in c)
        or c.startswith("src_py/ui_")
        or "app.py" in c
        or "ui_help" in c
        or "ui_params" in c
        for c in pc
    )
    if ui_only:
        row["class"] = "DIAG"
        row["proposal"] = "move-to-diag"
        row["risk"] = "UI/app only at tip"
        row["d2_evidence"] = "no pipeline/night_run consumer in grep"
        return

    if td.startswith("bool"):
        row["class"] = "MODE"
        row["proposal"] = "keep"
        row["risk"] = "bool switch; both branches assumed wanted"
        row["d2_evidence"] = "not probed as LEGACY; default-path live"
        return

    row["class"] = "ACTIVE"
    row["proposal"] = "keep"
    row["risk"] = "live consumer outside config.py"
    row["d2_evidence"] = ""


def main() -> None:
    data = json.loads(TABLE.read_text(encoding="utf-8"))
    for row in data["rows"]:
        classify(row)
    counts: dict[str, int] = {}
    for row in data["rows"]:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    data["counts"] = counts
    data["needs_milan"] = [
        {"key": r["key"], "question": r["risk"]}
        for r in data["rows"]
        if r["proposal"] == "needs-Milan"
    ]
    data["n"] = len(data["rows"])
    data["note"] = (
        "CONSOLIDATE-01D D1+D2 2026-08-31. 279 config_runtime keys. "
        "No deletions. Extra AppConfig aliases outside this 279: "
        "blind_index_path (owner=internal), aavso_observer_code (owner=db_static)."
    )
    data["extra_aliases_not_in_279"] = [
        {
            "key": "blind_index_path",
            "owner": "internal",
            "class": "ALIAS",
            "proposal": "remove-alias",
            "target": "blind_index_fine_path",
            "loc": "config.py:564",
        },
        {
            "key": "aavso_observer_code",
            "owner": "db_static",
            "class": "ALIAS",
            "proposal": "remove-alias",
            "target": "observer_code",
            "loc": "config.py:641",
        },
    ]
    TABLE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    lines = [
        "| key | type/default | class | proposal | risk | d2_evidence | consumers |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in data["rows"]:
        cons = "; ".join(prod_consumers(r["consumers"])[:8])
        if len(prod_consumers(r["consumers"])) > 8:
            cons += "; ..."
        def esc(s: str) -> str:
            return (s or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| {key} | {td} | {cl} | {pr} | {rk} | {d2} | {co} |".format(
                key=r["key"],
                td=esc(r["type_default"]),
                cl=r["class"],
                pr=r["proposal"],
                rk=esc(r["risk"]),
                d2=esc(r["d2_evidence"]),
                co=esc(cons) if cons else "(config.py only)",
            )
        )
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("counts", json.dumps(counts, indent=2))
    print("needs-Milan", len(data["needs_milan"]))
    for item in data["needs_milan"]:
        print(" ", item["key"], ":", item["question"][:120])
    print("LEGACY", [r["key"] for r in data["rows"] if r["class"] == "LEGACY"])
    print("ALIAS", [r["key"] for r in data["rows"] if r["class"] == "ALIAS"])
    print("DEAD", [r["key"] for r in data["rows"] if r["class"] == "DEAD"])
    print("DIAG", [r["key"] for r in data["rows"] if r["class"] == "DIAG"])


if __name__ == "__main__":
    main()
