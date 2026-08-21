CURSOR RESULT - 2026-08-20 (DRAFT-517-REVIEW)

What I did
Read-only review of draft 517 (first post-era UI run on BO CVn field) vs anchor
516 (9902d918). No code/config changes. Artifacts under
`dev/results/context/session_20260820_d517review/`.

## Part A - Provenance and infrastructure

| Item | draft 517 |
|------|-----------|
| Git tip | **8dea595** (era close; git_dirty=true scratch only, git_dirty_code=false) |
| Run mode | **UI** (`run_vyvar`, site `ui_selection`) |
| Entry point stamped | `run_phase2a` |
| PFS | **false** |
| err_background_mode | **empirical** |
| Era config | masterstar_dao 4.5 / dao_detection_n_equiv 4.5 / forced_photometry true |
| User-changed keys vs era defaults | **None observed** in stamped config_snapshot |

**Runtime (Rule 0.3)** - infolog UTC 2026-08-20:

| Phase | Window | ~Duration |
|-------|--------|-----------|
| Import + calibration | 16:25:20 - 16:27:58 | ~2.5 min |
| MASTERSTAR | 16:27:58 - 16:56:09 | ~28 min |
| Phase 0+1 + 2A | 16:56:09 - 17:28:52 | ~33 min |
| Postprocess / HRD / report | 17:28:52 - 17:37:43 | ~9 min |
| **Total** | 16:25:20 - 17:37:43 | **~74 min** |

**Certificate** (`dao_gaia_calibration.json`): **PASS**. Derived tolerances **2.5/2.5 px @ pass1_sigma 4.5 / pass2_sigma 4.0** - matches era anchor setup.

| Population | 517 p95 px (n) | 516 p95 px (n) | Drift mechanism |
|------------|----------------|----------------|-----------------|
| detection_identity | 1.420 (2829) | 1.411 (2870) | Fresh MASTERSTAR stack; -41 matches |
| seed_centroid | 2.420 (462) | 2.461 (589) | Fewer forced-seed paths on smaller census |
| empty_sky pass2 | 0.14% (3/2200) | 0.14% (3/2200) | Same |
| empty_sky seed | 0.27% (6/2200) | 0.18% (4/2200) | Within audit band; minor |

Validation gate: **PASS** (max_regression_pp 0.00063).

**Census** (gaia_source_state_census.csv) - side by side:

| source_state | 517 | 516 | delta |
|--------------|-----|-----|-------|
| DETECTED_P1 | 3082 | 3103 | -21 |
| DETECTED_P2 | 414 | 397 | +17 |
| FORCED_SEED | 39 | 93 | -54 |
| SEED_REJECTED | 450 | 1150 | -700 |
| BLENDED | 61 | 146 | -85 |
| EDGE | 85 | 101 | -16 |
| **Total rows** | **4131** | **4990** | -859 |

Accounting: **100%** on-chip both drafts (n_on_chip = n_total).

**INV gates during run:** No `InvariantViolation` lines in infolog. INV stamps in pipeline_meta all **ok=true** (WCS, DAG, CFG, SAT, CAL). Expand path logged (`[M1] catalog membership expand`) - normal, not a failure. **47** `[PIN]` log lines; pinned check **1497613731286514432** confirmed in infolog.

## Part B - Product vs anchor

**Product SHA: NOT identical.**

| | 517 | Anchor 516 |
|---|-----|------------|
| Core SHA | **342344d2** | **9902d918** |
| Core n | 111 | 121 |
| LC files | 55 | 60 |

**Input deltas FIRST (before interpreting product deltas):**

1. **Code path:** 517 on era tip **8dea595**; live 516 stamped **b8d5c74** (pre-era commit on prior run).
2. **Frame subset:** Phase 2A ran **134 frames** (16/150 lights QC-rejected for HFR>5); anchor 516 photometry used full accepted set.
3. **Fresh MASTERSTAR:** Smaller census table (4131 vs 4990); certificate populations shifted.
4. **Same field / same calibration library** - comparable science intent, not byte-replay of 516.

**Diff class (shared 55 LCs):** 34/55 show non-zero `mag_calib_final` delta vs 516; mix of metadata/ensemble path and frame-count effects. **BO_CVn delta 0.0 mmag; FW_CVn 3.1 mmag.** Among **45 shared pinned targets**, **19/45** at <=2 mmag max epoch delta.

**5 LCs only on 516:** 1485560025830226432, 1496037650087948160, 1496733984545821696, 1497169940906156032, 1497491273179203456.

**Pin path:** `pinned_ensembles_sha256=bb515414...` present; **48 targets**; **0 member drops** on disk vs pin file. Re-validation drops: **none filed**. Unpinned actives use default selection (250 phase0 targets; 55 LC products).

**LC census:** 55 LCs; lc_quality good=51 noisy=4 saturated=1. **CV CVn** skip on 517: **`zone_flag`** (saturated zone on active_targets) vs anchor **`per_frame_saturation`** - classification change worth noting for Milan; not a pipeline crash.

**Check meters** (check **1497613731286514432**, 134 epochs):

| Target | MAD mmag | Band |
|--------|----------|------|
| BO_CVn | **4.82** | 6.08-8.22 PASS |
| FW_CVn | **5.58** | 6.97-9.43 PASS |

Both tighter than era anchor meters (7.15 / 8.20) - consistent with fewer frames / refreshed ensemble, not a failure.

## Part C - Milan bright-stars observation (classify only)

UI layer: MASTERSTAR QA shows **Gaia markers** for all catalog rows; **green hollow DAO** only for DETECTED_P1/P2 when "Show detections (DAO)" enabled; **EDGE/BLENDED/SEED_REJECTED** correctly have **Gaia only**.

For **G<=13** without DETECTED_P1/P2: **22 census rows, all `EDGE`** (19 in masterstars_full_match). **Zero** BLENDED / SEED_REJECTED / FORCED_SEED at G<=13. All G<=13 DETECTED rows have **vy_dao_pass=1**.

| Verdict | Count | Example |
|---------|-------|---------|
| expected_named_category (EDGE) | 22 | 1500803208360486144 G=8.72 EDGE |
| display_gap | 0 | - |
| unclassified | 0 | - |

**Summary:** Milan's "bright Gaia, no DAO marker" sighting matches **EDGE** (and optionally saturated **DETECTED_P1** if DAO layer toggled off) - **era working as designed**, not a missing detection. Overlay truth: EDGE must **not** show green DAO; UI is consistent with census.

Full table: `dev/results/context/session_20260820_d517review/d517_review.json` part_c.

## Part D - Review verdict

**Is 517 a healthy era product?** **Yes, with named caveats.**

- Certificate PASS; census 100%; pins loaded; check meters in band; BO science-stable vs 516 (0 mmag).
- Expected first-run differences vs frozen anchor: **non-identical SHA**, fewer LCs (55 vs 60), frame QC subset, CV CVn skip reason path, tighter check MADs.

**Before new fields Milan should know:**

1. UI runs may QC-reject frames (HFR) - LC count can be < anchor without code bug.
2. Product SHA will differ from 9902d918 until same frame set + deterministic replay policy is defined (**MS-POOL-POLICY-01** scope).
3. MASTERSTAR QA: bright stars without green DAO at field edge are **EDGE** state - not missing DAO detections.
4. CV CVn: `zone_flag` on 517 vs `per_frame_saturation` on 516 - same target skipped, different gate label.

**ROADMAP candidates (named, no scope creep):**

| Pri | Item | Rationale |
|-----|------|-----------|
| HIGH | **MS-POOL-POLICY-01** | 517 SHA drift vs anchor from frame subset + fresh run; determinism policy needed |
| MED | **MS-QA-DISPLAY-01** | Optional UI cue for EDGE/BLENDED "Gaia-only by state" to reduce operator confusion |
| LOW | **CV-CVN-SKIP-CONSISTENCY** | Document zone_flag vs per_frame_saturation skip_reason taxonomy |

## Docs impact (DOCS-SYNC)

| File | Change |
|------|--------|
| `dev/results/CURSOR_RESULT_DRAFT_517_REVIEW.md` | First landing of DRAFT-517-REVIEW evidence (this file) |
| `docs/VYVAR_ROADMAP.md` | NEXT SESSION 2026-08-21 items from review verdict |
| `docs/VYVAR_DECISIONS.md` | 2026-08-20 product model + EMPTY-DAO-01 closure |

Frame QC mechanism detail superseded by `CURSOR_RESULT_FRAME_QC_PARITY_01.md`
(same session): live 516 vs 517 QC artifacts are identical; dual-layer QC vs
`--full` replay is the architectural split.

## Errors
None (review only).

## Files changed
None (review artifacts only).

Push not applicable.

Runtime (harness): 1.04 s. Review wall time ~74 min operator run documented above.
