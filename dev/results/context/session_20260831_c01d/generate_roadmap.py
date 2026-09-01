#!/usr/bin/env python3
"""CONSOLIDATE-01D D3: rebuild docs/VYVAR_ROADMAP.md. No product-code change."""
from __future__ import annotations

import json
from pathlib import Path

from extract_roadmap_ids import HERE, REPO, ROADMAP, extract_ids

BEFORE = json.loads((HERE / "roadmap_ids_before.json").read_text(encoding="utf-8"))

# one-line state | owner | blocked-on
OPEN: dict[str, tuple[str, str, str]] = {
    "EXPORT-PARITY-01": (
        "Standing two-path export vs pipeline photometry defect (R5); PSF merge path fixed",
        "Cursor",
        "schedule",
    ),
    "EPSF-SHAPE-01": (
        "Root narrow ePSF core OPEN (FWHM 2.36 vs 3.30); routed to EPSF-CORE-01",
        "Milan+Cursor",
        "EPSF-CORE-01",
    ),
    "EPSF-XVAL-01": (
        "External ePSF gate: same ensemble/frames, independent PSF photometry reference; method unspecced",
        "Milan",
        "literature spec",
    ),
    "EPSF-ZP-OK-XRIG-01": (
        "Extend fit_ok_for_zp past wide 1:1; needs master dark+flat + CENSUS-01; Newton 518 pool 26 does not qualify",
        "Milan",
        "CalibrationLibrary + night with gated pool >=30",
    ),
    "MULTIFILTER-WCS-01": (
        "Sibling-seed VERIFIED WCS for z_90_4; catalog-recovery gate unrelaxed; 520 measurement 2.7%/0%",
        "Cursor",
        "Milan GO",
    ),
    "FRAME-QC-PARITY": (
        "Phase 2: Layer A log honesty + QC provenance stamp; draft 516 frame 29 n_stars 263 vs ~100",
        "Cursor",
        "not C8",
    ),
    "DEPTH-AUTH-01": (
        "Derive masterstar_gaia_census_target_depth_g from MASTERSTAR completeness vs Gaia; G=15.56 VSX absent",
        "Cursor",
        "not wired",
    ),
    "EPSF-CORE-01": (
        "Literature-parameter ePSF rebuild (multi-frame samples, osamp vs FWHM, smoothing)",
        "Milan+Cursor",
        "FUTURE",
    ),
    "EPSF-PERF-01": ("Forced linear refit path; deferred by Milan", "Milan", "FUTURE"),
    "INPUT-PATH-ARCH-01": (
        "Discussion: non-cal stays; raw-without-masters split",
        "Milan",
        "discussion",
    ),
    "SEL-GHOST-01": (
        "Not closed; origin/main stays 7c086e8 until Milan writes PUSH_AUTH",
        "Milan",
        "PUSH_AUTH",
    ),
    "PHASE0-BORDER-MARGIN-GEOMETRY": (
        "Phase 0 50 px margin is not EDGE r_out; not merged into EDGE-ANNULUS-01",
        "Cursor",
        "not EDGE",
    ),
    "TODO-MULTISET": (
        "Per-telescope-set config architecture (wide vs Newton)",
        "Milan+Cursor",
        "design",
    ),
    "TODO-GS8": (
        "Multi-night global matching + global ZP; descoped from HIGH; canonical unit is one night",
        "Milan",
        "FUTURE science case",
    ),
    "TODO-PSF-NEIGHBOR-SUB": (
        "Neighbour subtract + aperture residual; 2b deferred until blended fine-scale field",
        "Cursor",
        "blended fine-scale draft",
    ),
    "TODO-PSF-MULTIFRAME": ("Multi-frame ePSF stacking (isolation part done)", "Cursor", "FUTURE"),
    "TODO-PSF-ASYMMETRY": ("Tracking-smear diagnostics (BO CVn right-tail PSF)", "Cursor", "FUTURE"),
    "TODO-A": (
        "Median/sigma-clip MASTERSTAR stack of best N frames ranked by I_j; provenance; DAO recailbration",
        "Cursor",
        "audit Steps 1-6",
    ),
    "TODO-B": (
        "Zackay & Ofek proper coaddition; blocked on CR, uncorrelated inputs, per-frame PSF",
        "Cursor",
        "CR-REJECTION",
    ),
    "TODO-C": (
        "Admission gate vs detection threshold; CONTEXT-ONLY vs PHOTOMETRY-CANDIDATE flags",
        "Cursor",
        "audit Steps 7-8",
    ),
    "CR-REJECTION": ("Cosmic-ray rejection (L.A.Cosmic or equivalent); no CR step in src_py today", "Cursor", "TODO-A"),
    "CR-1": ("Same as CR-REJECTION (closure Step 9)", "Cursor", "TODO-A"),
    "A-1": ("Frame selection metric I_j for MASTERSTAR stack ranking", "Cursor", "TODO-A"),
    "A-2": ("Selection rule N_min=10 N_max=20 quality gate 0.5 x max(I_j)", "Cursor", "TODO-A"),
    "A-3": ("Median/sigma-clip stack replacing single-frame copy", "Cursor", "TODO-A"),
    "A-4": ("Stack provenance in header + pipeline_meta.json", "Cursor", "TODO-A"),
    "A-5": ("Recalibrate DAO threshold against stack noise/PSF", "Cursor", "TODO-A"),
    "C-1": ("Admission gate: predicted per-epoch SNR (g_lim + Labbe sigma_bkg_ap)", "Cursor", "TODO-C"),
    "C-2": ("CONTEXT-ONLY vs PHOTOMETRY-CANDIDATE catalogue flags", "Cursor", "TODO-C"),
    "T4-1": ("DECISION: detection noise on resampled frames (options A/B/C/D)", "Milan", "decision"),
    "A-1-OVERRIDE": (
        "Remove VY_FWHM_GAUSS as gaussian_fwhm_px_override; authorized in principle; own measured delta",
        "Cursor",
        "measured delta",
    ),
    "A-1-DECISION-4": ("Advanced r90 5.0-5.8 px target 5.31; not implemented", "Cursor", "schedule"),
    "TASK-A-REGRESSION": (
        "A2 CSV-write test never calls generate_masterstar_and_catalog",
        "Cursor",
        "test rewrite",
    ),
    "F-B01-F-B02": (
        "PASSTHROUGH runs may claim VYVAR calibration; PDF honesty",
        "Cursor",
        "calpath audit s14",
    ),
    "QHY294MM-RN-DOUBLE": (
        "DB RN 7.6 e- may be bin2 then scaled again to 15.2 e-",
        "Cursor",
        "low priority",
    ),
    "BPM-SIDECAR-PATH": (
        "No *_dark_bpm.json found; path dead/disabled/outside tree unresolved",
        "Cursor",
        "forensics",
    ),
    "D1-2-LINEARITY-RAMP": ("Exposure ramp at telescope; nothing else substitutes", "Milan", "telescope night"),
    "CORR-ERR-01": ("ZP common-mode vs diagonal budget; out of v1.0", "Milan", "research"),
    "WIDE-ERR-CROSSRIG": ("Per-rig when Newton/Boyden drafts exist", "Cursor", "other-rig drafts"),
    "C-EXPORT-GAP": ("Headless night_run omits AAVSO/VarAstro export", "Cursor", "schedule"),
    "INSTALL-GAIA-DEC-CUTOUT": (
        "INSTALL should lead with declination cutout decision for Gaia builder",
        "Cursor",
        "docs",
    ),
    "SYNTH-SKY-GENERATOR": ("WCS-true synthetic field generator for known-truth photometry", "Claude", "sub-pixel debug"),
    "V1-VALIDATION-PROTOCOL": (
        "Enrich validation packs: ePSF identity, DAOPHOT ref, PDF QA, E2E mini field",
        "Milan+Claude",
        "protocol",
    ),
    "DRAFT451-CAL-FRAME001": (
        "Draft 451 frame-001 calibrated product differs 659.6 ADU; root cause needs 451 cal logs",
        "Cursor",
        "logs",
    ),
    "SKY-SURFACE-BLAST-RADIUS": (
        "Drafts 438-451 inflated catalogues; confirm no AAVSO/VarAstro export from those drafts",
        "Milan",
        "export check",
    ),
    "GAIA-PM-COLUMNS": ("Gaia DB lacks pmra/pmdec; defer to DR4 ~Dec 2026", "Milan", "DR4"),
    "R-CVN-EMPTY-COMP": ("Empty-comp drop reports no_comps; confirm nothing further", "Cursor", "POST-453"),
    "INSTALL-MANUAL": ("New-user install manual + T460 installer including catalogs", "Milan", "TODO-LIB"),
    "TODO-9": ("Superseded/extended by INSTALL-MANUAL", "Milan", "INSTALL-MANUAL"),
    "TODO-LIB": ("Cython closed-source bundle; ties to installer", "Milan", "CYTHON-RELEASE follow-up"),
    "TODO-GEO": ("Backlog geography/site item", "Milan", "parked"),
    "TODO-SCENE-FORWARD-MODEL": (
        "Conditional on crowded-faint science; priority lowered after grouper-negative",
        "Milan",
        "FUTURE",
    ),
    "TODO-SEP-XVAL": ("SEP independent witness; aperture xval CLOSED", "Cursor", "parked"),
    "PUB-FIGS": ("Methods paper figures", "Milan+Claude", "PUBLICATION"),
    "PUB-JOSS-PREREQS": ("JOSS prerequisites", "Milan+Claude", "PUBLICATION"),
    "PUB-OUTLINE": ("Paper outline", "Milan+Claude", "PUBLICATION"),
    "PUB-POLICY": ("Publication policy", "Milan+Claude", "PUBLICATION"),
    "PUB-VALIDATION-SECTION": ("Paper validation section", "Milan+Claude", "PUBLICATION"),
    "PUB-VENUE": ("Venue choice", "Milan", "PUBLICATION"),
    "RELEASE-1": ("Release-1 checklist", "Milan", "v1.0"),
    "RELEASE-2": ("Release-2 checklist", "Milan", "v1.0"),
    "F-BINGAIN-1": ("Newton bin4 chi2 gate still open; do not flip ensemble to Broeg IVW until it passes", "Cursor", "Newton gate"),
    "SIGMA-BKG-VAR-01": ("Sigma background variance follow-up", "Cursor", "LOW"),
    "DB-RETIRE-01": ("Retire stale DB paths", "Cursor", "FUTURE"),
    "MS-POOL-POLICY-01": ("MASTERSTAR pool policy", "Cursor", "FUTURE"),
    "PRECAL-INPUT-CONTRACT-01": ("Pre-cal input contract", "Cursor", "MED"),
    "COMP-POOL-R": ("Comp pool R follow-up", "Cursor", "parked"),
    "K2-DATA-BLOCKER": ("K2 data blocker", "Milan", "data"),
    "K2-SLOPE-TRACE": ("K2 slope trace", "Cursor", "K2-DATA-BLOCKER"),
    "K2-SLOPE-UG": ("K2 slope UG", "Cursor", "K2-DATA-BLOCKER"),
    "MASTERSTAR-EPOCH": ("MASTERSTAR epoch / PM", "Cursor", "GAIA-PM-COLUMNS"),
    "NET-TEST-01": ("Network/test harness item still listed open", "Cursor", "LOW"),
    "RUN-WORKER-01": ("Run-worker follow-up", "Cursor", "LOW"),
    "SPARSE-TRUST": ("Sparse-field trust gate follow-up", "Cursor", "parked"),
    "STALE-LC-SWEEP": ("Stale LC sweep", "Cursor", "LOW"),
    "TIER1-OBSLOC-ZERO": ("Observer location zero hygiene", "Cursor", "LOW"),
    "TIER1-UI-DEBT": ("Tier-1 UI debt", "Cursor", "LOW"),
    "CAL-AGE-CLOCK": ("Calibration master age clock", "Cursor", "LOW"),
    "CAL-PASSTHRU-DEAD": ("Passthrough calibration honesty; related F-B01-F-B02", "Cursor", "F-B01-F-B02"),
    "EQUIP-BINNING-ASYM": ("Equipment binning asymmetry", "Cursor", "LOW"),
    "RN-HEADER-NONE": ("Read-noise has no FITS header source", "Cursor", "LOW"),
    "DB-DEFECT-DIAMETER": ("DB defect diameter", "Cursor", "LOW"),
    "GAIA-ID-FLOAT-GUARD": ("Gaia id float guard follow-up if any residual", "Cursor", "LOW"),
    "HRD-PLOT-TUPLE": ("HRD plot tuple hygiene", "Cursor", "LOW"),
    "PROC-MAG-NAMING": ("Proc mag naming", "Cursor", "LOW"),
    "PROD-SIGMA-FLOOR": ("Production sigma floor", "Cursor", "LOW"),
    "PROV-HEADLESS": ("Headless provenance", "Cursor", "LOW"),
    "PROVENANCE-GUARD": ("Provenance guard follow-up", "Cursor", "LOW"),
    "F-AIRMASS-CITE": ("Airmass citation hygiene", "Cursor", "LOW"),
    "F-BJD-1": ("BJD time-base follow-up", "Cursor", "LOW"),
    "F-EXCEPT-TIER1": ("Remaining tier-1 except hygiene", "Cursor", "LOW"),
    "F-HOWELL-3": ("Howell citation/path follow-up after F-BINGAIN-1", "Cursor", "F-BINGAIN-1"),
    "NOQA-TRUNCATED-EXCEPT-BULK": ("noqa truncated except bulk leftover", "Cursor", "LOW"),
    "TODO-BROAD-EXCEPT-HYGIENE": ("Broad-except tier-1 leftover (~25)", "Cursor", "LOW"),
    "WIDE-ERR-HONEYCUTT-PDF": ("Honeycutt SEM PDF honesty", "Cursor", "LOW"),
    "WIDE-ERR-POP-DELTA": ("Wide-err population delta", "Cursor", "LOW"),
    "WIDE-SLOPE-NOISE": ("Wide slope vs noise", "Cursor", "LOW"),
    "SIGMA-BUDGET-EMPIRICAL": ("Empirical sigma budget remaining Newton gate", "Cursor", "F-BINGAIN-1"),
    "SIGMA-PROV-FORENSIC": ("Sigma provenance forensic leftover", "Cursor", "LOW"),
    "SIGMA-SEM-CAUSE": ("SEM cause leftover", "Cursor", "LOW"),
    "BATCH-E-PARAMS-REGISTRY": ("Batch-E params registry leftover if any", "Cursor", "LOW"),
    "BIN-8-9-REGRESSION-01": ("Bin 8/9 regression leftover", "Cursor", "LOW"),
    "DAO-TOL-FLOOR-01": ("DAO tolerance floor leftover", "Cursor", "LOW"),
    "D10-1": ("D10-1 leftover from audit register", "Cursor", "LOW"),
    "ANCHOR-CHAIN-ACCEPT": ("Anchor chain accept leftover", "Cursor", "LOW"),
    "ANCHOR-CLEAN-BUILD": ("Anchor clean-build leftover", "Cursor", "LOW"),
    "ANCHOR-ERR-VERIFY": ("Anchor err verify leftover", "Cursor", "LOW"),
    "ANCHOR-GATE-SEED": ("Anchor gate seed leftover", "Cursor", "LOW"),
    "EPSF-PIN-CENSUS-01": ("ePSF pin census leftover / Newton 518", "Cursor", "EPSF-ZP-OK-XRIG-01"),
    "EPSF-NEWTON-518-01": ("Newton 518 ePSF STOP: gated pool 26 < 30", "Milan", "night with pool>=30"),
}

# False ID / dropped / superseded-not-this-arc
RETIRED: dict[str, str] = {
    "CLOSED-DECIDED": "Not a task id; status token from EDGE-ANNULUS-01 row.",
    "MASTERSTAR-EPSF-ALL": "Dropped 2026-06-02; plate scale is WCS-derived.",
    "TODO-RECUT-HARNESS-FIDELITY": "CLOSED superseded 2026-07-08; draft_387 zaloha gone.",
    "U-XVAL-COMP-RMS": "RETRACTED (audit register).",
    "TODO-COMP-P2P-RESIDUAL": "DONE already implemented; found stale 2026-07-19.",
    "TODO-DEV-PROCESS": "DONE 2026-07-08 as DEV-PROCESS-A + DEV-PROCESS-B.",
    "TODO-EPSF-1-FWHM-QC": "DONE 2026-06-08.",
    "TODO-FWHM-CONSISTENCY": "DONE 2026-06-09.",
    "APCORR-MIXEDFRAME": "DONE 2026-07-19 all-or-nothing COG per night.",
    "CYTHON-RELEASE": "DONE closed-source bundle preview 2026-07-23.",
    "CONFIG-MATERIALIZE-CHECK": "DONE 2026-07-24 BUNDLE-BOOTSTRAP-WIRING.",
    "CATALOG-PROVENANCE": "DONE 2026-07-29.",
    "A-6": "DONE 2026-08-07 DAO detection workstream closed.",
    "DAO-THRESHOLD-PARAMS": "CLOSED 2026-08-07; reopen only on two-rig empirical sweep.",
    "EXCEPT-BULK": "CLOSED 2026-07-08 silent broad-except census.",
    "F-428": "CLOSED 2026-07-15 draft_428 forensics.",
    "F-429": "CLOSED 2026-07-16 validate + regressions.",
    "F-431-HEADLESS-DIVERGENCE": "CLOSED 2026-07-16 / T3 (DECISIONS).",
    "INV-CAL-01": "CLOSED 2026-08-13 CAL-DIAG v2.",
    "INV-CAL-02": "DONE 2026-08-13 calibrated product stage integrity.",
    "CAL-DIAG": "CLOSED 2026-08-13; SUPERSEDED heading removed 2026-08-11 then implemented.",
    "SAT-DIAG": "DONE saturation and linearity limit gate.",
    "XVAL-AIJ-02": "DONE production 4-comp + two frame states.",
    "WIDE-ERR": "CLOSED WIDE-ERR-04 physical model g_pt + weighted SEM.",
    "FULL-ANCHOR-RECUT": "CLOSED 2026-08-27 ERA-04 lock.",
    "P1-RECUT": "CLOSED 2026-08-20 ERA-03 golden mini.",
    "A-1-435-RECUT": "CLOSED 2026-08-18; 435 retired by ROT policy; recut onto 516.",
    "VYVAR-INVARIANTS": "P1/P2 DONE 2026-07-19; remaining phases in git history.",
    "DEV-PROCESS-A": "DONE 2026-07-08 validation ledger.",
    "DEV-PROCESS-B": "DONE 2026-07-08 session_baseline_check.py --full.",
    "COMP-RMS-DEF-01": "Wired C3 2026-08-25 (k=5 LOO mag).",
    "COMP-RMS-DEF-01-B": "Wired C3 2026-08-25.",
    "ZONE-SAT-01": "Wired with COMP-RMS-DEF-01-B.",
    "EPSF-AC-01": "Closed in ePSF AC measurement arc 2026-08-24.",
    "EPSF-AC-02": "Closed/wired in ePSF AC arc; Newton ZP-OK still open as EPSF-ZP-OK-XRIG-01.",
    "EPSF-VALID-02": "CLOSED 2026-08-22 gated 67-star production ePSF on 516.",
    "EDGE-ANNULUS-01": "CLOSED-DECIDED Milan 2026-08-31: edge stars not used; full on-chip aperture+annulus.",
    "APERTURE-01": "Wired option i; later locked as APERTURE-01d.",
    "APERTURE-01b": "STOP 2026-08-26; no f* on accuracy grid.",
    "APERTURE-01c": "STOP 2026-08-26; AIJ PASS 2.7833 mmag; era04 not yet locked.",
    "APERTURE-01d": "LOCK 2026-08-27; annulus 2.7/5.2; AIJ 1.9503 mmag; era04 --full gate.",
    "ERA-03": "era03 freeze kept on disk; superseded as --full gate by era04.",
    "REG-520-01": "STOP 2026-08-24 measure; ghost/WCS notes carried in SEL-GHOST-01.",
    "DOCS-SYNC-517": "Superseded NEXT SESSION 2026-08-21.",
    "ARCHIVE-CLEANUP": "NEXT SESSION 2026-07-15; historical.",
    "FRAME-QC-PARITY-01": "Phase 1 heading superseded 2026-08-21; phase 2 remains FRAME-QC-PARITY.",
    "CLOSE-OUT": "Not a task id; heading token from stacked NEXT SESSION titles.",
    "SESSION-CLOSE": "Not a task id; heading token from stacked NEXT SESSION titles.",
}


KEEP_RETIRED = {
    "CLOSED-DECIDED",
    "CLOSE-OUT",
    "SESSION-CLOSE",
    "MASTERSTAR-EPSF-ALL",
    "TODO-RECUT-HARNESS-FIDELITY",
    "U-XVAL-COMP-RMS",
}


def main() -> None:
    before = set(BEFORE["ids"])
    extras_in_before = before
    retired_live = {k: v for k, v in RETIRED.items() if k in KEEP_RETIRED}
    closed: dict[str, str] = {
        k: v for k, v in RETIRED.items() if k not in KEEP_RETIRED and k in extras_in_before
    }
    unknown = []
    for i in sorted(extras_in_before):
        if i in OPEN:
            continue
        if i in retired_live:
            continue
        if i in closed:
            continue
        closed[i] = (
            "Closed, done, or superseded in stacked NEXT SESSION / DONE sections "
            "before CONSOLIDATE-01D; details in git history of this file."
        )
        unknown.append(i)
    missing_open = [i for i in OPEN if i not in extras_in_before]
    missing_ret = [i for i in retired_live if i not in extras_in_before]
    assigned = set(OPEN) | set(retired_live) | set(closed)
    dropped = sorted(extras_in_before - assigned)
    extra_assigned = sorted(assigned - extras_in_before)

    lines: list[str] = []
    a = lines.append
    a("# VYVAR - Roadmap (open work)")
    a("")
    a("Single source of truth for **open** tasks. Closed work lives in `VYVAR_JOURNAL.md`;")
    a("durable rationale in `VYVAR_DECISIONS.md`; current architecture in `VYVAR_STATE.md`.")
    a("")
    a("Rebuilt **2026-08-31 CONSOLIDATE-01D D3**: stacked dated NEXT SESSION sections")
    a("collapsed to one OPEN table, one CLOSED-this-arc list, and RETIRED lines.")
    a("No task id was dropped. Historical prose of the stacked sections is in git:")
    a("`git log -- docs/VYVAR_ROADMAP.md` (parent of the CONSOLIDATE-01D D3 commit).")
    a("")
    a("Cross-check: **EDGE-ANNULUS-01** is CLOSED-DECIDED in `VYVAR_DECISIONS.md`")
    a("(Milan 2026-08-31).")
    a("")
    a("---")
    a("")
    a("## OPEN")
    a("")
    a("| id | one-line state | owner | blocked-on |")
    a("|----|-----------------|-------|------------|")
    for i in sorted(OPEN):
        if i not in extras_in_before:
            continue
        st, owner, blocked = OPEN[i]
        a(f"| **{i}** | {st} | {owner} | {blocked} |")
    a("")
    a("Standing operator items without a hyphenated id (kept as prose, not an id row):")
    a("first AAVSO/VarAstro uploads BO -> FW (band CV) once a locked ledger exists;")
    a("`origin/main` stays `7c086e8` until Milan writes PUSH_AUTH.")
    a("")
    a("---")
    a("")
    a("## CLOSED this arc")
    a("")
    a("Closed, locked, or superseded during the 2026-06..2026-08 stacked-session era")
    a("(APERTURE / SEL-GHOST / ePSF / ERA-04 / audit closure). One line each.")
    a("")
    for i in sorted(closed):
        a(f"- **{i}** -- {closed[i]}")
    a("")
    a("---")
    a("")
    a("## RETIRED")
    a("")
    a("Dropped, retracted, status-token, or explicitly superseded-do-not-reopen.")
    a("")
    for i in sorted(retired_live):
        if i not in extras_in_before:
            continue
        a(f"- **{i}** -- {retired_live[i]}")
    a("")
    a("---")
    a("")
    a("## Parked (not blocking; ids already in OPEN if they had one)")
    a("")
    a("- CM-detrend differential (~10x lever; opt-in; needs transit injection-recovery).")
    a("- Newton-V colour-term (per-rig c1 from field BP-RP).")
    a("- Meridian-flip handling (Qatar-8 class).")
    a("- Pre-filled camera catalog for new-user onboarding (PARKED; design notes in git history).")
    a("- Magnitude-aware check-star threshold for the trust gate.")
    a("- Reserved check-star (hold-one-out; moves photometry anchor).")
    a("- AAVSO-standard output #4 G->B/V/Rc.")
    a("- Blind index 3rd rig tier (Noctutec 206/560) when a validated draft exists.")
    a("- Comet photometry mode (after variable-star pipeline; analysis only).")
    a("- TODO-GS7 paper draft (see PUBLICATION ids in OPEN).")
    a("")

    ROADMAP.write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = extract_ids(ROADMAP.read_text(encoding="utf-8"))
    after_ids = set(after)
    diff_drop = sorted(extras_in_before - after_ids)
    diff_add = sorted(after_ids - extras_in_before)
    payload = {
        "n_before": len(extras_in_before),
        "n_after": len(after_ids),
        "drop": diff_drop,
        "add": diff_add,
        "missing_open_not_in_before": missing_open,
        "missing_retired_not_in_before": missing_ret,
        "dropped_unassigned": dropped,
        "extra_assigned": extra_assigned,
        "n_open": len([i for i in OPEN if i in extras_in_before]),
        "n_closed": len(closed),
        "n_retired": len([i for i in retired_live if i in extras_in_before]),
        "closed_default_ids": unknown,
    }
    (HERE / "roadmap_id_diff.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("n_before", "n_after", "drop", "add", "n_open", "n_closed", "n_retired", "dropped_unassigned", "extra_assigned", "missing_open_not_in_before")}, indent=2))


if __name__ == "__main__":
    main()
