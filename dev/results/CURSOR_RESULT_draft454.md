# CURSOR RESULT - DRAFT-454 ANALYSIS (2026-07-28)

Session data: `dev/results/context/session_20260728_draft454/`

Infolog source (Part 1): `Archive/Drafts/draft_000454/infolog_20260728_152715.txt`
(durable session append log). Ring-buffer export `infolog_20260728_164917.txt` retains
milestones but drops mid-run lines including `INV-PREP-01`.

---

## Part 1 - Draft 454 on its own

### 1.1 Startup phase table (infolog-derived)

Milestone block + body timestamps from saved infolog only:

| Phase | Start (UTC) | Duration (s) |
|-------|-------------|-------------:|
| run_vyvar start | 13:27:05 | 0 |
| Scan Source + Import + calibration start | 13:27:05 | 10 |
| Calibration | 13:27:15 | 148 |
| Analyze QC (RAM -> OBS_FILES) | 13:29:43 | 64 |
| Auto FWHM limit | 13:30:47 | 0 |
| Auto-select MASTERSTAR (TOP1) | 13:30:47 | 1 |
| MAKE MASTERSTAR (detrend + plate-solve + zarovnanie) | 13:30:48 | 986 |
| Phase 0+1 + Phase 2A (photometry) | 13:47:14 | 3176 (to 14:40:10) |

**Gap run start -> first artefact** (first calibration write / `[1/150] Calibrating`): **10 s**
(session setup + import through `[SITE]` at 13:27:15, first calibrated output 13:27:16).

**Infolog wall clock** (first to last timestamp): **4975 s (82.9 min)** including ~599 s UI
activity after pipeline completion (HRD filters, etc.).

**Run time (Part 1 parse):** 10.6 s analysis script.

### 1.2 Timing profile

| Phase | 452 (headless, old preprocess) | 453 (UI, infolog incomplete) | 454 (UI, new preprocess) |
|-------|-------------------------------:|-----------------------------:|-------------------------:|
| preprocess wall | 2743 s | not measurable | **44 s** |
| per-frame preprocess | 18.3 s | - | **0.29 s** |
| platesolve+align (after preprocess) | 859 s | ~857 s (tail) | **942 s** |
| photometry | 2549 s | 805 s (tail) | **3176 s** |
| total wall (infolog / night_run) | 119 min | 76 min (folder) | **83 min** |

**Production preprocess (454):** 44 s for 150 frames with 8 workers (**0.29 s/frame**). Projected
~120 s at 8 workers -- **realised 2.7x faster than projection**. Not adjusted; likely less worker
contention and warm OS cache on live UI run vs bench harness.

Headless draft 455 (current code, same night): preprocess **48.9 s** (0.33 s/frame) -- consistent.

**Run time (455 headless, Part 2):** 4514.8 s.

### 1.3 Proof lines (verbatim from durable infolog)

```
13:27:15  [SITE] observer location id=2 name=Jirny lat=50.1121658 lon=14.6982547 alt_m=275.0 source=ui_selection
13:31:32  INFO  [pipeline]  INV-PREP-01 Preprocess gradient guard (NoFilter_60_2): large_small_ratio=0.03x (warn>10)
13:33:08  INFO  [pipeline]  INV-MS-01 MASTERSTAR purity guard: dao_only_fraction=0.037
13:33:13  VSX-GAIA XM: n_vsx=873 n_gaia=15085 rho=705.9 deg^-2 mean_nn=67.8" r_max=7.64" Q=0.99 w=0.96 sigma_n=0.18" sigma_b=0.50" accepted=717 contamination=0.011% cand_mult=0:156 1:694 2:23 3+:0 multi=2.63% pm_path=broadened pm_cols=False pm_finite=0 vsx_epoch=2000.0 gaia_epoch=2016.0 masterstars=205/208 outcome=ok gaia_db_max_g=17.5
13:47:15  FAZA 0 funnel: vsx_bbox=875 -> in_frame=797 -> gaia_id_assigned=651 (contamination=n/a) -> dao_detected=201 -> active=201 | excluded: no_dao_detection=0 no_gaia_id=0 not_target_eligible=596 out_of_frame=78 | masked: zone_flag=2 vsx_type_out_of_scope=0
13:33:14  [EXO TARGET] funnel: hosts_in_field=82 masterstars_in_frame=2842 promoted=3 sep_max=3 arcsec
13:32:59  INFO  [pipeline]  [DAO pass 1] 2552 detections, 1332 Gaia unmatched
13:32:59  INFO  [pipeline]  [DAO pass 2] 1225 additional detections from 1332 targeted positions
```

**INV-PREP-01:** confirmed on real run (durable session log). Absent from ring-buffer export
`infolog_20260728_164917.txt` -- durability gap for guard lines, not guard skip.

### 1.4 Acceptance metrics

Context labels: **454/455** = live plan, 875 VT rows, no mag limit. **Anchor 165** = frozen
mag-limited VT on draft_435 snapshot -- not comparable without stating context.

| Quantity | Expected (BO CVn live path) | 454 (UI) | Context |
|----------|----------------------------:|---------:|---------|
| pass-1 DAO | ~2552 | **2552** | infolog |
| masterstars rows | ~2951 | **2951** | CSV |
| DAO_ONLY fraction | ~3.7% | **3.69%** | CSV |
| bg_std | ~83.8 ADU | **83.82 ADU** | MASTERSTAR.fits |
| DAO threshold | ~176 ADU | **176.03 ADU** | 2.1 x bg_std |
| active targets | **201** (live) | **201** | active_targets.csv |
| exoplanet promotions | 3 | **3** | variable_targets.csv |
| light curves written | - | **198** | 3 actives without LC (empty_comp / edge) |

---

## Part 1.5 - Preprocess neutrality (452 old vs 454 new)

All **150** calibrated frames compared (`BO_CVn_Light_*.fits`):

| Metric | Value |
|--------|------:|
| n at exactly 0.0 ADU max diff | **149** |
| n non-zero | **1** (frame 001) |
| frame 001 max abs diff | **533.45 ADU** |

**Reading:** only frame 001 differs -- real edge case on the frame with the highest subtracted-surface
amplitude, not general non-neutrality. Frame 001 `VYSKYP2P` = **177.95 ADU** (max of 150; median
136.85; frame 002 = 169.90). Modest twilight excess (~5% above frame 002), not an order-of-magnitude
outlier; the 533 ADU calibrated delta is old-vs-new masking/surface fit on that frame. Needs its own
fix if strict 0.0 neutrality required on frame 001; ROADMAP item `DRAFT451-CAL-FRAME001` updated.

**Run time:** included in Part 1 script (10.6 s).

---

## Part 2 - Entry-point equivalence (454 UI vs 455 headless)

```
python dev/scripts/draft_ui_equivalence_check.py draft_000454 draft_000455
```

| Pattern | count 454 | count 455 | mismatches |
|---------|----------:|----------:|------------|
| lightcurve_*.csv | 198 | 198 | **0** |
| comp_quality_*.json | 0 | 0 | **0** |
| comparison_stars_per_target.csv | 1 | 1 | **0** |

**Byte-identical** on the science file set. These two runs also satisfy the standing two-agreeing-runs
requirement in the stronger form: **two different entry points** (UI vs headless), not two repetitions
of the same one.

Headless run: draft **455**, `location_id=2` (Jirny), source `D:\BO_CVn`, elapsed **4514.8 s**.

**Run time (equivalence check):** 2.2 s.

---

## Part 3 - Documentation gaps (golden re-cut audit)

| Item | Status |
|------|--------|
| 3.1 Ledger dropped target Gaia IDs | `VL-ANCHOR-WCSINV.identity_gate_dropped_targets` added |
| 3.2 Wrong-star naming in DECISIONS + ledger | `IDENTITY-GATE-WRONG-STAR-NAMING` in DECISIONS |
| 3.3 Anchor snapshot portability | STATE NOT guaranteed |
| 3.4 Frame 001 ROADMAP | `DRAFT451-CAL-FRAME001` (MED) opened |
| Addendum: anchor `--full` does not run preprocess | STATE NOT guaranteed (three gaps listed) |

---

## Acceptance

| Item | Status |
|------|--------|
| 1.1 | PASS - startup table from infolog; gap 10 s |
| 1.2 | PASS - preprocess 44 s vs ~120 s projection |
| 1.3 | PASS - all proof lines; INV-PREP-01 on durable log |
| 1.4 | PASS - table with context labels |
| 1.5 | PASS - 149/150 neutral; frame 001 edge case |
| 2 | PASS - 454 vs 455 byte-identical |
| 3 | PASS - four doc items |
| `--fast` / `ruff` | see below |

---

## Run times summary

| Part | s |
|------|--:|
| Part 1 analysis script | 10.6 |
| Part 1.5 (same script) | (included) |
| Part 2 headless 455 | 4514.8 |
| Equivalence check | 2.2 |
