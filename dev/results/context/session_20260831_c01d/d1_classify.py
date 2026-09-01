#!/usr/bin/env python3
"""CONSOLIDATE-01D: fill class/proposal from consumers + known lists. No config deletions."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE / "config_prerez_table.json"

# AppConfig fields that are documented aliases of another live key.
ALIAS_KEYS = {
    "blind_index_path": ("blind_index_fine_path", "config.py:564"),
    "dao_detection_n_equiv": ("masterstar_dao_threshold_sigma", "config.py:779"),
    "comp_iterative_clip_enabled": ("comp_sparse_fallback_enabled", "config.py:995"),
    "aavso_observer_code": ("observer_code", "config.py:641"),
}

# Switch where default is the production branch; other value is superseded.
LEGACY_SWITCHES = {
    "global_comp_pool_enabled": {
        "superseded_by": "COMP-POOL-01 (default True is production)",
        "legacy_value": "False",
        "config_line": "config.py:1121",
    },
    "phase01_use_bprp_primary": {
        "superseded_by": "BP-RP colour (default True); False = |dB-V| tiers",
        "legacy_value": "False",
        "config_line": "config.py:910",
    },
    "export_err_mode": {
        "superseded_by": "calibrated (s, sigma_r); model = legacy quadrature",
        "legacy_value": "model",
        "config_line": "config.py:955",
    },
    "err_background_mode": {
        "superseded_by": "F-BINGAIN-1 empirical empty-aperture; howell = legacy",
        "legacy_value": "howell",
        "config_line": "config.py:1047",
    },
    "masterstar_accept_mode": {
        "superseded_by": "odds (Bayesian); fraction = legacy",
        "legacy_value": "fraction",
        "config_line": "config.py:831",
    },
    "psf_ac_policy": {
        "superseded_by": "p4_none default; chi2_lt5_legacy is ePSF graph",
        "legacy_value": "chi2_lt5_legacy",
        "config_line": "config.py:668",
        "risk": "phase 2 of this row needs --full-epsf",
    },
}

# px fallbacks: MODE if both branches wanted; else LEGACY.
PX_FALLBACKS = {
    "hrd_color_bg_box_px": "hrd_color_bg_box_arcsec",
    "qc_max_hfr": "qc_max_hfr_fwhm_ratio",
    "masterstar_centre_rms_max_px": "masterstar_centre_rms_max_arcsec",
}

SKIP_CONSUMER_FILES = {
    "src_py/config.py",
}


def prod_consumers(cons: list[str]) -> list[str]:
    out = []
    for c in cons:
        f = c.split(":")[0]
        if f in SKIP_CONSUMER_FILES:
            continue
        if "params_registry" in f:
            continue
        out.append(c)
    return out


def classify(row: dict) -> None:
    k = row["key"]
    td = row["type_default"]
    pc = prod_consumers(row["consumers"])
    row["d2_evidence"] = ""

    if k in ALIAS_KEYS:
        tgt, loc = ALIAS_KEYS[k]
        row["class"] = "ALIAS"
        row["proposal"] = "remove-alias"
        row["risk"] = f"loader-migrated alias of {tgt} ({loc})"
        row["d2_evidence"] = loc
        return

    if k in LEGACY_SWITCHES:
        info = LEGACY_SWITCHES[k]
        row["class"] = "LEGACY"
        row["proposal"] = "remove-key+dead-branch"
        row["risk"] = info.get("risk", f"legacy value {info['legacy_value']}; {info['superseded_by']}")
        row["d2_evidence"] = info["config_line"]
        return

    if k in PX_FALLBACKS:
        row["class"] = "LEGACY"
        row["proposal"] = "needs-Milan"
        row["risk"] = f"px fallback of {PX_FALLBACKS[k]} ({k} used when arcsec/ratio is None)"
        row["d2_evidence"] = "reachable when companion key is None (data/config condition -> MODE if wanted)"
        return

    if k in PX_FALLBACKS.values():
        row["class"] = "ACTIVE"
        row["proposal"] = "keep"
        row["risk"] = "preferred unit; px sibling is fallback"
        return

    if not pc:
        row["class"] = "DEAD"
        row["proposal"] = "remove-key+dead-branch"
        row["risk"] = "zero consumers outside config.py at tip (verify getattr)"
        return

    ui_only = all(
        ("/ui_" in c) or c.startswith("src_py/ui_") or "app.py" in c or "ui_help" in c
        for c in pc
    )
    if ui_only and "bool" not in td:
        row["class"] = "DIAG"
        row["proposal"] = "move-to-diag"
        row["risk"] = "UI/app only at tip"
        return

    if td.startswith("bool"):
        row["class"] = "MODE"
        row["proposal"] = "keep"
        row["risk"] = "bool switch; both branches assumed wanted until D2"
        return

    row["class"] = "ACTIVE"
    row["proposal"] = "keep"
    row["risk"] = "live consumer outside config.py"


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
    data["note"] = "D1 filled 2026-08-31; D2 evidence for LEGACY still to probe from production entry"
    TABLE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("counts", counts)
    print("needs-Milan", len(data["needs_milan"]))
    print("LEGACY", [r["key"] for r in data["rows"] if r["class"] == "LEGACY"])
    print("ALIAS", [r["key"] for r in data["rows"] if r["class"] == "ALIAS"])
    print("DEAD", [r["key"] for r in data["rows"] if r["class"] == "DEAD"])


if __name__ == "__main__":
    main()
