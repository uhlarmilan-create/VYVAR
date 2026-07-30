CURSOR RESULT - PHASE0-IDENTITY-GATE - 2026-07-26

Tree: `cb78b25` (local main, +1 vs private tip `10608bb`). Science-path changes are
implemented but **not committed** pending Milan anchor STOP (STEP 5).

---

## What I did

Implemented Milan-approved design A-D across STEPs 0-5 (code + tests + partial docs).
STEP 0 anchor-gate extension is ready for its own commit; STEPs 1-5 science commits
**held** per task STOP.

---

## STEP 0 - Why anchor missed `a0e3431` + gate extension

### Root cause (CONFIRMED)

`dev/scripts/session_baseline_check.py` `--full` **does not regenerate** plan files.
It passes **frozen** inputs from `draft_000435/platesolve/NoFilter_60_2/`:

- `variable_targets.csv` (pre-built)
- `masterstars_full_match.csv`
- `MASTERSTAR.fits`

into `run_full_photometry_pipeline`, then compares photometry SHA to the snapshot.
Commit `a0e3431` changed **plan-time** `write_photometry_plan_files` export (873 vs 243
rows on identical frames in forensic), but the anchor continued using the **old frozen**
`variable_targets.csv` (245 rows on this machine). Photometry SHA stayed byte-identical;
the Phase 0 funnel change was invisible.

### Gate extension (implemented)

- New module: `src_py/phase0_funnel.py` (fingerprints + compare).
- `session_baseline_check.py`: after `--full` pipeline run, checks:
  - **Input** `variable_targets.csv` row count + `gaia_match_source` histogram.
  - **Output** `active_targets.csv` row count, `skip_photometry_true`, `zone_flag` histogram.
- Expected fingerprints for draft 435 in `EXPECTED_PHASE0_FUNNEL_BY_DRAFT`.
- Ledger `VL-ANCHOR-WCSINV` extended with `phase0_funnel_fingerprint` block.
- Tests: `dev/tests/test_session_baseline_check.py` (funnel compare + CSV read).

**Anchor draft_435 fingerprints (this machine):**

| Metric | Value |
|--------|-------|
| variable_targets_rows | 245 |
| gaia_match_source | masterstars=178, gaia_dr3_direct=64, no_match=1, masterstars_exo=2 |
| active_targets_rows (snapshot) | 169 |
| skip_photometry_true | 2 |
| zone_flag | linear=113, noisy1=10, noisy2=6, noisy3=38, saturated=2 |

---

## STEP 1 - Plan-time VSX -> Gaia (design A)

**Site:** `src_py/vsx_gaia_crossmatch.py` (new) + `pipeline.py:write_photometry_plan_files`.

- Removed fixed `max_sep = 10 arcsec` and 3"/7" quality bins.
- Density-aware match: fit `sigma` (Rayleigh) + local `rho` from frame; reliability
  R = L/(L+1) per Sutherland & Saunders 1992; accept set with expected contamination <= 1%.
- PM: propagate via `_apply_proper_motion` when `pmra`/`pmdec` present in masterstars/Gaia
  rows; else broaden sigma (`pm_absent_broadened_sigma` path).
- Fail loud: `VsxGaiaCrossmatchError` -> RuntimeError (no silent fixed-radius fallback).
- `gaia_match_quality` now reflects reliability tiers (`high`/`good`/`uncertain`/`poor`).
- `gaia_dr3_direct` / `no_match` rows remain in `variable_targets.csv` (comp veto preserved).
- Citations added to `CITATIONS.bib`: Marrese 2017/2019, Sutherland & Saunders 1992.

---

## STEP 2 - Phase 0 identity join (design B)

**Site:** `photometry_core.py:select_active_targets`.

- Deleted adaptive radius block (`phase01_match_radius_arcsec`, 5x plate scale, NN loop).
- Identity join: planner `catalog_id` (normalized) must exist in masterstars index (built from
  `name` column to avoid float corruption in `catalog_id`).
- VSX auto rows: only `gaia_match_source == masterstars` promotable.
- Exclusions: `no_dao_detection`, `no_gaia_id`, `not_target_eligible`.
- `zone_flag`: mask `saturated`, `catalog_only`, `neznama_zona`.
- Manual/exoplanet rows: identity join without VSX `gaia_match_source` gate.
- Removed `match_radius_arcsec` from `select_active_targets`, `run_phase0_and_phase1`,
  `run_full_photometry_pipeline` call chain.

**`phase01_match_radius_arcsec`:** removed from Phase 0 path only. Still in `config.py` /
registry (270 params unchanged this session) -- **pending removal** from registry/docs in
follow-up commit; no remaining science-path consumer in `src_py/`.

---

## STEP 3 - Depth from detection (design C)

No separate magnitude gate reintroduced. DAO membership in masterstars **is** the depth criterion.

**`MASTERSTAR_FAINTEST_MAG_FLOOR = 18.0`:** astrometry/MASTERSTAR detection floor only
(`dao_reconcile.py`, `pipeline.py` masterstar build). **Not read on the science photometry
path** after Phase 0 identity join (confirmed grep: no use in `photometry_core` Phase 0/2A
target promotion).

---

## STEP 4 - Observability

- FAZA 0 funnel INFO line at end of `select_active_targets` (counts + exclusion/mask buckets).
- WARN: empty `vsx_out_of_scope_types` with matching types but zero masks; low
  `gaia_id_assigned` fraction (<50% of in_frame).
- Invariants: `INV-CFG-01R` (WARN reverse), `INV-PHASE0-ID` (FAIL catalog_id identity).
- Docs: `docs/VYVAR_INVARIANTS.md` updated.

---

## STEP 5 - Dilution reporting + ANCHOR STOP

- `_attach_predicted_dilution_report()` adds `predicted_dilution_factor` to
  `active_targets.csv` (report-only; uses `dilution.compute_dilution_factor`; defaults unchanged).
- **`--full` anchor NOT run** (task: science path changed; expected FAIL on funnel + SHA).
  Milan must approve science-justified re-cut before `--full`, ledger restamp, or science commits.

### Expected anchor impact (from task + design)

| | draft_435 (old gate) | expected under new gate |
|--|----------------------|-------------------------|
| variable_targets.csv rows | 245 | unchanged at plan time* |
| active targets | 169 | lower (masterstars-only promotion) |
| planner/active catalog_id mismatches | possible (old NN) | **0** (INV-PHASE0-ID) |

\*Plan-time cross-match may also change `gaia_match_source` assignments when plans are regenerated;
frozen anchor VT file unchanged until re-cut.

### draft_450 replay

**BLOCKED on this machine** (no `draft_000450` tree under `Archive/`). Forensic from
`CURSOR_RESULT_phase0_target_gate_forensic.md` remains the reference: expect active drop
from 322 to <=283 (`masterstars` rows), 67 catalog_id disagreements -> 0.

---

## Validation

| Check | Result |
|-------|--------|
| New tests `test_phase0_identity_gate.py` | 8/8 PASS |
| Updated VSX / F428 / session_baseline tests | PASS |
| Full pytest (excl. slow golden) | 1162 passed, 3 fixed failures -> **re-run pending** |
| `--fast` session baseline | pytest reported FAIL in wrapper (1163 passed message; investigate wrapper exit code on next pass) |
| `ruff` | not run this session |
| `--full` | **NOT RUN** (STOP) |

---

## Files changed (uncommitted)

**STEP 0 (anchor gate):**
- `src_py/phase0_funnel.py` (new)
- `dev/scripts/session_baseline_check.py`
- `dev/validation/VYVAR_VALIDATION_LEDGER.json`
- `dev/tests/test_session_baseline_check.py`

**Science path (STEPS 1-5, held):**
- `src_py/vsx_gaia_crossmatch.py` (new)
- `src_py/pipeline.py`
- `src_py/photometry_core.py`
- `src_py/invariants_runtime.py`
- `dev/tests/test_phase0_identity_gate.py` (new)
- `dev/tests/test_select_active_targets_excludes_unmatched_vsx.py`
- `dev/tests/test_vsx_out_of_scope_types.py`
- `dev/tests/test_f428_fixbatch.py`
- `CITATIONS.bib`
- `docs/VYVAR_INVARIANTS.md`

**Docs still TODO for full task compliance:**
- `docs/VYVAR_DECISIONS.md` (supersede ~578-600)
- `docs/VYVAR_PARAMS.md` + config registry (`phase01_match_radius_arcsec` removal)
- `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`
- `dev/tools/docs_pdf/flow_doc_facts.py` + PDF rebuild

---

## Errors

None blocking implementation. draft_450 replay and `--full` anchor deferred by design STOP.

---

## Milan decision required (STEP 5 STOP)

1. Approve science-path commits (STEPS 1-5) and anchor re-cut?
2. Expected active-target drop on draft_435: confirm acceptable before `--full` + ledger restamp.
3. Push: **do not push** science commits until cleared.

---

## Files changed (commit hash)

None committed this session (STOP).

---

## FOLLOW-UP (2026-07-26)

Pre-STOP blockers B1-B5. Still **no commits, no push, no `--full`, no ledger restamp**.

### B1 - Green baseline (diagnosis)

**Previous `--fast` FAIL was not a wrapper bug.** Evidence:

| Run | Exit code | pytest subprocess | Detail |
|-----|-----------|-------------------|--------|
| Initial session | 1 | failed | 1162 passed; **3 test failures** (fixed in-session) |
| After fixes | 0 | passed | **1165 passed**, 26 skipped |
| After partial B5 config removal | 1 | failed | **1161 passed**; 4 registry-parity failures |

The session wrapper (`session_baseline_check.py`) forwards pytest's exit code unchanged
(`check_pytest` -> `subprocess.run(...); status = FAIL if rc != 0`). The "1163 passed"
message with FAIL was pytest exiting 1 while reporting many passes -- classic partial failure.

**3 failures fixed earlier (named):**

1. `test_select_active_targets_excludes_unmatched_vsx` - identity join requires
   `gaia_match_source=masterstars` on fixtures.
2. `test_vsx_out_of_scope_types` (3 tests) - same fixture update + removed
   `match_radius_arcsec` kwarg.
3. `test_f428_fixbatch::test_excluded_targets_sidecar_on_no_match` - exclusion reason
   renamed `no_dao_gaia_match` -> `no_dao_detection`.

**Current baseline (this follow-up):**

| Check | Result |
|-------|--------|
| `--fast` | **PASS** (exit 0; 1165 passed, 26 skipped) |
| `ruff check src_py dev/tests dev/scripts` | **PASS** (F821 included; 3 pre-existing noqa warnings in unrelated files) |
| Full `pytest dev/tests` | **1165 passed**, 26 skipped, 65 warnings |

---

### B2 - Anchor drop measurement (read-only join)

Sources (unchanged on disk):

- Active: `draft_000435_snapshot_skysurface_20260716/.../photometry/active_targets.csv` (169 rows)
- Planner VT: `draft_000435/platesolve/NoFilter_60_2/variable_targets.csv` (245 rows)

Join on `vsx_name`.

| Metric | Count |
|--------|------:|
| Active targets | 169 |
| Active with planner `gaia_match_source != "masterstars"` | **5** |
| Active with `catalog_id` != planner `catalog_id` (same vsx_name) | **4** |
| **Would drop under identity gate** | **5** |
| **New active count** | **164** |

The 4 catalog_id mismatches are a subset of the 5 non-`masterstars` actives (all
`gaia_dr3_direct`). The 5th is TOI-3919 (exo row; planner source empty/nan).

**Dropped targets table:**

| vsx_name | vsx_type | vsx_mag_max | planner source | planner sep (\") | active catalog_id | active mag |
|----------|----------|-------------|----------------|-----------------|-------------------|------------|
| TOI-3919 | | | (empty) | | 1497132660589966976 | 12.70 |
| Gaia DR3 1499883638682689920 | | 14.01 | gaia_dr3_direct | 0.125 | 1499883638682689408 | 9.40 |
| Gaia DR3 1500410236033012352 | | 14.41 | gaia_dr3_direct | 0.194 | 1500410613990135296 | 13.26 |
| Gaia DR3 1498513166158147968 | | 14.04 | gaia_dr3_direct | 1.416 | 1498513269237363456 | 11.74 |
| Gaia DR3 1499064433800590592 | | 13.67 | gaia_dr3_direct | 2.138 | 1499064399440851968 | 11.35 |

**Calibration vs chance-coincidence estimate:** Among planner VT rows, 67 have
`gaia_match_source != masterstars` (64 `gaia_dr3_direct` + 1 `no_match` + 2 `masterstars_exo`).
The task's ~4-5 spurious-promotion estimate at r=48.85" applies to **spatial NN promotion**
into actives, not to all non-masterstars VT rows. Measured **4** `gaia_dr3_direct` actives with
wrong-neighbor `catalog_id` aligns with the ~4-5 prediction. No discrepancy requiring model rework.

---

### B3 - Gate extension covers plan-time (option **a** implemented)

**Choice: (a)** Extend `--full` to regenerate plan files and compare fingerprint.

**Why not (b):** Plan regeneration is feasible on this machine (VSX + Gaia DBs present,
`write_photometry_plan_files` runs in ~4s). No architectural blocker.

**Implementation:** `dev/scripts/session_baseline_check.py` adds
`_check_plan_regen_fingerprint()` called at start of `run_full_baseline()` (before pipeline).
It writes to `{work_root}/plan_regen/variable_targets.csv` via `write_photometry_plan_files`
and compares row count + `gaia_match_source` histogram to `EXPECTED_PHASE0_FUNNEL_BY_DRAFT[435]`.

**Important:** With the current STEP 1 matcher on this field, regen **will FAIL** the new check
(VSX export skipped -- see B4). That is correct gate behaviour: a second `a0e3431`-class change
or a broken matcher cannot pass unnoticed. Frozen-input fingerprint alone remains necessary but
not sufficient; regen closes the plan-time gap.

**Not run:** `--full` per STOP.

---

### B4 - Real-data crossmatch diagnostic (anchor night)

Regenerated plan via `write_photometry_plan_files` on draft_435 inputs:

- **Result:** VSX export **skipped** (exception caught); only 3 `masterstars_exo` rows written.
- **Cause:** `VsxGaiaCrossmatchError: no reliability threshold achieves expected contamination <= 1.0%`

Direct diagnostic on anchor field (read-only; `tmp/b4_crossmatch_diagnostic.py`):

| Quantity | Value |
|----------|------:|
| VSX in-frame (bbox query) | 873 |
| Gaia masterstars in catalogue | 2842 |
| Field area | 21.37 deg^2 |
| rho | 133.0 deg^-2 |
| Fitted sigma (NN separations) | **101.27 arcsec** |
| PM path | `pm_absent_broadened_sigma` (no pmra/pmdec on masterstars CSV) |
| Reliability median | 0.00021 |
| Min achievable contamination | **none** (no threshold accepts any match) |

**Reading (not adjusted to match baseline):**

- Frozen anchor VT: **245 rows**, histogram `masterstars=178, gaia_dr3_direct=64, no_match=1,
  masterstars_exo=2` (old fixed-radius matcher at anchor cut).
- Current matcher on **873 in-frame VSX** (unbounded export, post-`a0e3431` geometry): sigma fit
  on all nearest-neighbour separations is dominated by chance pairs at large separation (~101").
  Reliability collapses; 1% contamination rule rejects the entire set.
- **Material difference:** yes -- regen cannot reproduce the recorded histogram. Per pre-registered
  rule, this is reported not patched. Milan must judge whether the matcher implementation matches
  the approved Marrese/Sutherland design intent on dense wide-field data before STOP clearance.

`pipeline.py` now stores `vsx_gaia_crossmatch` diagnostics in `variable_targets_diagnostics`
when VSX export succeeds.

---

### B5 - Parameter count and `phase01_match_radius_arcsec`

**Registered parameter count:** **271** (`docs/VYVAR_STATE.md:7` is correct).

Reconciliation:

- `dev/validation/params_registry.json`: **272** keys (includes entries not in AppConfig or legacy).
- AppConfig fields also in registry: **270**.
- STATE **271** = authoritative registered count used in docs/flow facts (272 registry keys
  minus one internal/non-public entry, per existing project convention).

Prior result file "270" was wrong.

**`phase01_match_radius_arcsec`:**

- **Science path:** no consumer after identity-gate refactor (verified grep + guard test).
- **Remaining references:** `dev/scripts/m67_continue368_gr_merge_sandbox.py` (dev sandbox only);
  stale docs/flow PDF text; registry + config.json persistence.
- **Action this follow-up:** field kept in `AppConfig` with DEPRECATED comment to preserve registry
  parity (partial removal caused 4 test failures). **Recommended next commit:** remove from
  registry, `config.json`, PARAMS guides, flow doc -> **270 registered params**.

---

### Blocker status

| Blocker | Status |
|---------|--------|
| B1 Green baseline | **CLOSED** |
| B2 Anchor drop measured | **CLOSED** (164 actives expected) |
| B3 Plan-time gate | **CLOSED** (option a implemented, not `--full`-verified) |
| B4 Real-data matcher diagnostic | **CLOSED** (reports failure on anchor field) |
| B5 Param count / phase01 | **CLOSED** (271 correct; phase01 removal deferred for registry sync) |

**New finding for Milan STOP:** STEP 1 matcher **fails loud** on anchor-night geometry (873 VSX).
Cannot approve anchor re-cut or science commits until matcher behaviour on this field is adjudicated.

---

## MATCHER-FIX (2026-07-26, B4 adjudication)

Milan adjudication: design A and <=1% rule unchanged. Two implementation defects fixed
(VSX->masterstars RHS; Q not fitted). **STOP still in force** -- no commits, push, `--full`,
or ledger restamp.

### What changed (F1-F5)

| Item | Implementation |
|------|----------------|
| F1 | `write_photometry_plan_files` queries **local Gaia DR3** over frame bbox via `query_gaia_for_frame_bbox` / `query_local_gaia` (same geometry as VSX bbox). Masterstars removed as match RHS. |
| F2 | Mixture fit: measured `rho`, fitted `sigma` + **Q**; Sutherland reliability tiers; <=1% gate on `mean(rho * pi * (sep/3600)^2)` over accepted set (pre-registered Poisson form from Milan B4 sanity table). |
| F3 | Degeneracy guard `sigma > 0.25/sqrt(rho)` [deg] -> `VsxGaiaCrossmatchDegenerateError` with explicit message (distinct from empty field / generic threshold failure). |
| F4 | INFO log: `VSX-GAIA XM: n_vsx=... n_gaia=... rho=... mean_nn=... sigma_fit=... Q_fit=... accepted=... contamination=... pm_path=... gaia_db_max_g=...` |
| F5 | `catalog_id` = accepted Gaia `source_id`; `gaia_match_source` **derived**: `masterstars` if ID in `masterstars_full_match.csv`, else `gaia_dr3_direct`, else `no_match`. All VSX rows kept in VT. |

**Files:** `src_py/vsx_gaia_crossmatch.py` (rewrite), `src_py/pipeline.py` (VSX block),
`dev/tests/test_vsx_gaia_crossmatch.py` (new), `docs/VYVAR_DECISIONS.md` (VSX-GAIA-MATCHER-TWO-STEP),
`docs/VYVAR_INVARIANTS.md` (INV-VSXGAIA-DEGEN plan-time note).

### Gaia DB depth (F1)

| Quantity | Value |
|----------|------:|
| `gaia_db_path` | `GAIA_DR3/vyvar_gaia_dr3.db` |
| `get_gaia_db_max_g_mag` | **17.5** |
| VSX in-frame with `mag_max > 17.5` | **141** |

The local DB is magnitude-capped at G=17.5. One hundred forty-one in-frame VSX rows have
`mag_max` above that limit; they are at elevated risk of `no_match` when no Gaia counterpart
exists in the capped DB. Reported explicitly, not silently worked around.

PM columns in the local Gaia DB are empty on this machine (`pm_finite=0`); fit uses
`pm_path=broadened` (+0.05 arcsec in quadrature).

### Anchor-night regen (`draft_000435` inputs -> `tmp/matcher_fix_regen_d435/`)

Well-conditioned fit (**not degenerate**):

| Fit diagnostic | Value |
|----------------|------:|
| VSX in-frame | 873 |
| Gaia in bbox | 15085 |
| Field area | 21.37 deg^2 |
| rho | 705.9 deg^-2 |
| mean_nn | 67.8" |
| sigma_fit | **0.18"** (astrometric scale, not ~100") |
| Q_fit | **0.99** |
| n_accepted (VSX with Gaia ID) | 574 |
| realized contamination | 0.002% |

**Histogram vs `draft_000450` baseline (873-row, no mag limit):**

| gaia_match_source | draft_450 baseline | this regen | delta |
|-------------------|--------------------:|-----------:|------:|
| (total rows) | 873 | **875** (+2 exo) | +2 |
| masterstars | 283 | **153** | **-130** |
| gaia_dr3_direct | 443 | **421** | -22 |
| no_match | 147 | **299** | **+152** |
| masterstars_exo | (not in baseline) | 2 | -- |

**Reading (observed, not tuned):**

- **Q ~ 0.99, sigma few arcsec:** observed -- fit is well conditioned on Gaia DR3 RHS (B4 defect 1 fixed).
- **no_match below 147:** **not observed** (+152 vs baseline). Primary driver likely the combination
  of G<=17.5 DB cap (141 faint VSX rows) plus reliability gate rejecting large-separation nearest
  neighbours for VSX without a deep-catalog counterpart.
- **masterstars near 283:** **not observed** (-130). Fewer accepted IDs land in the masterstars CSV
  than draft_450 recorded; worth checking whether draft_450 used a different Gaia DB depth,
  PM handling, or pre-fix matcher path. Not adjusted here.
- **gaia_dr3_direct near 443:** roughly observed (-22).

No fixed-radius fallback was used.

### Validation

| Check | Result |
|-------|--------|
| `dev/tests/test_vsx_gaia_crossmatch.py` | **6 passed** (degeneracy guard, sigma/Q recovery, end-to-end accept, contamination formula) |
| `dev/tests/test_phase0_identity_gate.py` | **8 passed** |
| Full pytest | **1171 passed**, 26 skipped (after INV registry parity fix) |
| `ruff check src_py dev/tests` | **clean** |

### Docs

- `docs/VYVAR_DECISIONS.md` -- VSX-GAIA-MATCHER-TWO-STEP (two-step separation, Q fitted, baseline withdrawal).
- `docs/VYVAR_INVARIANTS.md` -- INV-VSXGAIA-DEGEN (plan-time degeneracy; not end-of-run wired).

**Milan STOP:** Matcher now produces a well-conditioned fit on anchor night but **histogram
materially differs** from draft_450 baseline (especially `no_match` and `masterstars`). Science
commits remain blocked pending Milan review of these numbers and DB depth impact.

---

## MATCHER-FIX-2 (2026-07-26, sigma model)

Follow-up to MATCHER-FIX: root cause confirmed as **single-Rayleigh mis-specification**, not the
<=1% contamination rule (unchanged). STOP still in force.

### Adjudication accepted

- **G<=17.5 DB cap withdrawn** as `no_match` driver (128/141 faint rows were already `no_match`
  in draft_450 baseline).
- **sigma_fit=0.18"** was crushing the 1-2 arcsec tail; 130 lost `masterstars` rows are real DAO
  counterparts, not chance coincidences.
- **<=1% rule has headroom** to ~7.6 arcsec at anchor `rho`; rejections were from likelihood, not
  contamination budget.

### Implementation (G1-G3)

| Item | Change |
|------|--------|
| G1 | PM: propagate Gaia **2016.0 -> J2000.0** (`VSX_MATCH_EPOCH`) when `pmra`/`pmdec` finite; report `pm_path`, `pm_columns_present`, `n_pm_finite`, sep quantiles before/after. |
| G2 | True-match term = **two-Rayleigh mixture** (`Q`, `w`, `sigma_narrow`, `sigma_broad`); reliability uses full mixture `f(r)`; degeneracy on `sigma_broad` + astrometric-core tail checks. |
| G3 | WARN if `masterstars_accepted / masterstars_eligible < 80%` (eligible = NN Gaia ID in masterstars CSV within contamination radius). Logged in INFO line as `masterstars=N/N outcome=ok|warn_masterstars_low`. |

### PM path on this machine (G1)

| Field | Value |
|-------|------:|
| `pm_columns_present` | **false** (no `pmra`/`pmdec` in returned Gaia rows) |
| `n_pm_finite` | **0** |
| `pm_path` | **broadened** (+0.05 arcsec quadrature) |
| VSX epoch assumed | **J2000.0** (2000.0) |
| Gaia epoch | **2016.0** |
| Sep quantiles before PM (all NN) | p50=0.43" p90=60.6" p95=89.4" |
| Sep quantiles after PM | **identical** (no propagation applied) |

PM columns are optional in `database.query_local_gaia`; Milan's local DB on this machine does not
return usable PM for propagation. Epoch tail remains in the broad Rayleigh component.

### Fit diagnostics (anchor regen `tmp/matcher_fix2_regen_d435/`)

| Quantity | MATCHER-FIX | MATCHER-FIX-2 |
|----------|------------:|--------------:|
| Q_fit | 0.99 | **0.987** |
| w_fit | n/a | **0.964** |
| sigma_narrow | 0.18" | **0.18"** |
| sigma_broad | n/a | **0.50"** |
| rho | 706 deg^-2 | 706 deg^-2 |
| mean_nn | 67.8" | 67.8" |
| n_accepted | 574 | **683** |
| contamination | 0.002% | **0.005%** |
| Accepted sep p50/p90/p95 | n/a | **0.30" / 0.89" / 1.09"** |

### Histogram vs draft_450 baseline

| gaia_match_source | draft_450 | MATCHER-FIX | **MATCHER-FIX-2** | vs baseline |
|-------------------|----------:|------------:|------------------:|------------:|
| (VSX rows) | 873 | 873 | **873** (+2 exo) | +2 |
| masterstars | 283 | 153 | **191** | **-92** |
| gaia_dr3_direct | 443 | 421 | **492** | +49 |
| no_match | 147 | 299 | **190** | +43 |

**Reading (observed, not tuned):**

- **masterstars ~283:** not reached (191); large improvement over MATCHER-FIX (+38) but still
  short. G3 outcome **ok** (191/205 eligible = 93%). Residual gap likely PM-absent epoch tail plus
  683 vs ~726 total acceptances compared to baseline.
- **no_match ~128 genuinely uncatalogued:** not reached (190); improved from 299 but still above
  baseline 147.
- **Q >> 0.32, sigma astrometric not degenerate:** observed.
- **Contamination << 1%:** observed (0.005%).
- **<=1% rule unchanged:** confirmed throughout arc.

### Validation

| Check | Result |
|-------|--------|
| `dev/tests/test_vsx_gaia_crossmatch.py` | **8 passed** (two-population recovery, 1.5" regression, degeneracy) |
| Full pytest (spot) | **8/8** matcher tests green |
| `ruff check src_py/vsx_gaia_crossmatch.py` | **clean** |

### Docs

- `docs/VYVAR_DECISIONS.md` -- amended VSX-GAIA-MATCHER-TWO-STEP (mixture model, PM, 1% rule unchanged).
- `docs/VYVAR_INVARIANTS.md` -- INV-VSXGAIA-OUTCOME (G3 WARN).

**Milan STOP:** Material improvement over MATCHER-FIX; histogram still differs from draft_450 on
`masterstars` and `no_match`. PM propagation unavailable on this Gaia DB build may be a contributor.
No per-field constants or rule relaxation applied.

---

## MATCHER-FIX-3 (2026-07-26, acceptance rule)

Follow-up to MATCHER-FIX-2: remove the silent reliability threshold that capped accepted
separations at p95 ~ 1.09" while real matches extend to p95 ~ 1.85". The <=1% contamination
budget is now the **only** acceptance constant. STOP still in force.

### Adjudication accepted

- Reliability threshold was the de facto gate (0.005% contamination vs 1% budget = 200x margin).
- Broad Rayleigh + chance term are unidentifiable at 1-2"; pushing the fit further would still
  assign true-match tail to chance. Not a tuning problem.
- Gaia cross-match precedent: FoM **ranks** among positional candidates; no arbitrary FoM cutoff.

### Implementation (H1-H3)

| Item | Change |
|------|--------|
| H1 | `r_max = sqrt(0.01 / (pi * rho))` sets acceptance; 0 candidates -> `no_match`, 1 -> accept, >1 -> mixture reliability **ranking only**; reliability kept for `gaia_match_quality` tiers |
| H2 | Degeneracy guard **WARN** at plan-time (acceptance unaffected); candidate multiplicity logged; INFO line adds `r_max` + `cand_mult` |
| H3 | Two-step design, Phase 0 identity join, G3 check, existing tests unchanged (+ new 1.8" regression) |

### Acceptance diagnostics (anchor regen `tmp/matcher_fix3_regen_d435/`)

| Quantity | MATCHER-FIX-2 | **MATCHER-FIX-3** |
|----------|-------------:|------------------:|
| rho [deg^-2] | 706 | **706** |
| r_max | (implicit ~7.6") | **7.64"** |
| Q / w | 0.987 / 0.964 | **0.987 / 0.964** |
| sigma_n / sigma_b | 0.18" / 0.50" | **0.18" / 0.50"** |
| n_accepted | 683 | **717** |
| realized contamination | 0.005% | **0.011%** |
| Accepted sep p50/p90/p95 | 0.30 / 0.89 / 1.09 | **0.32 / 1.07 / 1.48** |
| fit_degenerate_warn | n/a | **false** |

**Candidate multiplicity (within r_max):**

| 0 | 1 | 2 | 3+ | multi-candidate fraction |
|--:|--:|--:|---:|-------------------------:|
| 156 | 694 | 23 | 0 | **2.63%** (expected ~1% at budget) |

PM unchanged on this machine: `pm_columns_present=false`, `n_pm_finite=0`, `pm_path=broadened`.

### Histogram vs draft_450 baseline

| gaia_match_source | draft_450 | MATCHER-FIX-2 | **MATCHER-FIX-3** | vs baseline |
|-------------------|----------:|--------------:|------------------:|------------:|
| (VSX rows) | 873 | 873 | **875** (+2 exo) | +2 |
| masterstars | 283 | 191 | **205** | **-78** |
| gaia_dr3_direct | 443 | 492 | **512** | +69 |
| no_match | 147 | 190 | **156** | +9 |

**Reading (observed, not tuned):**

- **masterstars ~283:** **not observed** (205); improved +14 over FIX-2 but still **-78** vs
  baseline. G3 outcome **ok** (205/208 eligible = 98.6%). Residual gap not compensated elsewhere
  per STOP instruction.
- **no_match ~128 genuinely uncatalogued:** not reached (156); improved from 190, now +9 vs
  baseline 147 (closer than FIX-2).
- **Accepted p95 extends to ~1.5":** **observed** (was 1.09"); matches the FIX-2 defect target.
- **Realized contamination << 1%:** **observed** (0.011%); acceptance is bounded by `r_max` at
  1% per source at the radius limit; mean over tight matches stays well below the budget.
- **<=1% rule unchanged:** confirmed; no second acceptance criterion added.

### Validation

| Check | Result |
|-------|--------|
| `dev/tests/test_vsx_gaia_crossmatch.py` | **10 passed** (+ `r_max` formula, ~1.8" regression) |
| `session_baseline_check.py --fast` | **OVERALL PASS** (1175 passed, 26 skipped) |
| `ruff check` (changed files) | **clean** |

### Docs

- `docs/VYVAR_DECISIONS.md` -- VSX-GAIA-MATCHER-TWO-STEP: acceptance via `r_max`, FoM ranking-only.
- `docs/VYVAR_ROADMAP.md` -- Gaia local DB lacks `pmra`/`pmdec` (rebuild item).

**Milan STOP:** Acceptance rule fix behaves as specified (p95 tail recovered, no reliability gate).
`masterstars` still **-78** vs draft_450 baseline (~283); report-only, no further compensation.
Science commits remain blocked pending Milan review.

---

## MATCHER-FIX-3-DIAG (2026-07-26, the missing 78)

Read-only diagnostic. No code changes. Scratch: `tmp/matcher_fix3_diag.py`, `tmp/matcher_fix3_diag_78.csv`,
`tmp/matcher_fix3_diag_results.json`.

**Inputs:** `draft_000450/.../variable_targets.csv` (baseline), `tmp/matcher_fix3_regen_d435/variable_targets.csv`
(FIX-3), `draft_000435/.../masterstars_full_match.csv` (FIX-3 regen DAO index), and for cross-check
`draft_000450/.../masterstars_full_match.csv`.

### Executive summary

**The name-column join-key hypothesis is not confirmed.** Current plan-time code builds `_ms_ids` from
the **`catalog_id` column only** (`pipeline.py` ~6539-6542), not from `name`. Phase 2A lookup uses
`masterstar_row_gaia_key()` (`gaia_catalog_id.py`: numeric `name` first, else `catalog_id`) -- an
inconsistency, but it does **not** explain this gap.

The dominant split is:

| Outcome | Count | Meaning |
|---------|------:|---------|
| Matcher agrees on `catalog_id`; relabel only | **77** | Same Gaia neighbour as baseline; FIX-3 marks `gaia_dr3_direct` because ID **not in draft_435 masterstars CSV** |
| Matcher chose different star / no match | **4** | Real matching disagreement (incl. float ID corruption in baseline VT) |

Net histogram gap: **81 lost** baseline `masterstars` labels, **3 gained** new ones -> **78 net**
(283 - 205).

**Critical confound:** all **283/283** baseline `masterstars` Gaia IDs appear in **draft_450**
masterstars CSV (6698 rows, 3993 unique IDs), but only **204/283** appear in **draft_435** masterstars
CSV (2951 rows, 2843 unique IDs). FIX-3 regen uses the **435** file; FIX-3 `masterstars=205` matches
that overlap (204 + 1). The "-78 vs draft_450" comparison mixes **different DAO catalog snapshots**,
not only a labelling bug.

### D1 -- name the 78 (81 lost, 3 gained)

**Summary split (81 rows baseline `masterstars` -> FIX-3 not `masterstars`):**

| Split | Count |
|-------|------:|
| `baseline_catalog_id == fix3_source_id` (matcher agrees) | **77** |
| IDs differ (matcher disagreement) | **4** |
| FIX-3 label `gaia_dr3_direct` | 80 |
| FIX-3 label `no_match` | 1 |

**77 equal-ID rows:** FIX-3 assigns the **same** Gaia `source_id` and separation (to machine precision)
as draft_450; only `gaia_match_source` changes because the ID is absent from `_ms_ids` built from
draft_435 `masterstars_full_match.csv`. All **77** IDs are in draft_450 masterstars `catalog_id`; **0/77**
are in draft_435 masterstars `catalog_id` (via `normalize_gaia_source_id`).

**4 differ-ID rows:**

| vsx_name | baseline_catalog_id | fix3_source_id | fix3 source | note |
|----------|--------------------:|---------------:|-------------|------|
| Gaia DR3 1486136822757775616 | 148613682275777**5744** | 148613682275777**5616** | gaia_dr3_direct | baseline ID last digits corrupted (float); FIX-3 picks name-consistent neighbour at 1.13" |
| Gaia DR3 1500387696044768384 | 150038769604476**8256** | 150038769604476**8384** | gaia_dr3_direct | same pattern; FIX-3 sep 0.53" vs baseline 4.07" |
| Gaia DR3 1498465333111277056 | 149846533**7402809472** | 149846533**3111277056** | gaia_dr3_direct | same pattern; FIX-3 tighter sep 0.013" |
| Gaia DR3 1502049466141453056 | 150204**9539149947136** | (none) | no_match | baseline sep 9.89"; ID mismatch; related ID in 435 CSV but outside `r_max` |

**Magnitude sanity (equal-ID subset vs draft_450 masterstars `mag`):** n=77, delta = mag(ms)-vsx_mag_max:
p50 **+0.02 mag**, p25 +0.01, p75 +0.02; |delta|>0.5: 3; |delta|>1.0: 2. Same as Milan's baseline check --
these are physically plausible matches, not random contamination.

**3 gained** in FIX-3 (not in baseline `masterstars`): `ASASSN-V J140153.60+422535.6`,
`Gaia DR3 1496820605445847296`, `Gaia DR3 1497436125799224960`.

**Full per-row listing (81 rows):** `tmp/matcher_fix3_diag_78.csv` (columns: vsx_name, vsx_type,
vsx_mag_max, baseline_catalog_id, baseline_gaia_match_arcsec, fix3_source_id, fix3_gaia_match_source,
fix3_gaia_match_arcsec, catalog_ids_equal).

### D2 -- join key lookup on draft_435 masterstars CSV

For each of the 81 FIX-3 `source_id` values (or baseline ID when FIX-3 empty), search
`masterstars_full_match.csv` **draft_435** with `normalize_gaia_source_id`:

| Bucket | FIX-3 source_id | Baseline catalog_id |
|--------|----------------:|--------------------:|
| In `name` (numeric Gaia id) | 0 | 3 |
| In `catalog_id` | 0 | 3 |
| In neither | **81** | **78** |

Zero FIX-3 IDs appear in either column of the 435 file. The 3 baseline IDs found in 435 are the
float-corrupted / mismatched cases above -- not the 77 equal-ID bulk.

**Conclusion:** D2 does **not** support "ID present in `catalog_id` but missing from `name` index."
The 77 lost labels are IDs **absent from the 435 DAO catalog entirely**, while present in draft_450's.

### D3 -- join key coverage (draft_435 masterstars, 2951 rows)

| Bucket | Count |
|--------|------:|
| Usable Gaia id in `name` only (`\d{12,22}`) | 0 |
| Usable Gaia id only in `catalog_id` | 38 |
| Usable id in both, identical | 2804 |
| Usable id in both, **different** | **0** |
| No usable id in either (`DET_*` names, empty ids) | 109 |

There is **no** "both but different" ambiguity on this file. The 38 `catalog_id`-only rows have
non-numeric `name` tokens (e.g. `DET_0002`); `_ms_ids` already indexes `catalog_id`, so switching
to `masterstar_row_gaia_key` would not recover any of the 81 lost rows on this snapshot.

### D4 -- G3 re-anchoring proposal (not implemented)

**Problem:** current G3 `masterstars_eligible` counts VSX rows whose **accepted matcher neighbour**
is in `_ms_ids` within `r_max` -- circular when the defect is "accepted ID correct but not in `_ms_ids`".

**Proposed external G3:**

```
eligible(vsx) = exists row ms in masterstars_full_match.csv
                with usable Gaia key K = masterstar_row_gaia_key(ms)
                and angular_sep(vsx, ms) <= r_max

masterstars_expected = |{ vsx : eligible(vsx) }|

WARN if masterstars_accepted / masterstars_expected < 0.80
```

**Why:** denominator comes from **DAO catalog geometry + r_max**, independent of matcher acceptance
or `_ms_ids` construction. Numerator remains `gaia_match_source=masterstars` after identity join.
On this diagnostic, applying draft_435 CSV + r_max=7.64" would yield a denominator closer to ~204
(the IDs actually present), exposing the 77 relabels as identity/index failures rather than reporting
205/208 = 98.6% "ok".

### Adjudication

| Hypothesis | Verdict |
|------------|---------|
| Name-vs-`catalog_id` join key defect | **Not confirmed** (0/81 recoverable; pipeline already uses `catalog_id`) |
| Matcher picking wrong stars for bulk of gap | **Rejected** (77/81 agree on ID) |
| Identity relabel because ID not in 435 DAO CSV | **Confirmed** (77/81) |
| draft_450 vs draft_435 masterstars catalog mismatch | **Confirmed confound** (283 vs 204 ID overlap) |
| G3 self-referential | **Confirmed** -- proposed fix in D4 |

**Expected small fix (report only, not implemented):** align `_ms_ids` with `masterstar_row_gaia_key`
for consistency with Phase 2A; re-anchor G3 as in D4. Neither explains the 77 without also using the
**correct masterstars CSV** for the baseline being compared (draft_450 has 1204 Gaia IDs not in 435).

**Milan STOP:** Diagnostic complete. No code changes. Next step is Milan's call: compare FIX-3 against
a regen that uses draft_450 masterstars inputs, or accept 205/204 as correct for the 435 snapshot.

---

## CLOSE (2026-07-26) -- STOP on active count

Milan cleared STEP 5 STOP. Commits landed; `--full` run **did not** reach re-cut / push (stop condition).

### Frozen-input claim (section 1 basis)

**Confirmed.** draft_435 anchor snapshot is **2026-07-16** (`10d610c0` provenance in `--full` log).
SKIPPROC / sky-surface regression (`013cb0c`, 2026-07-22) postdates the snapshot. `--full` consumes
frozen `MASTERSTAR.fits`, frozen `masterstars_full_match.csv`, frozen aligned lights, and frozen
input `variable_targets.csv` (245 rows). Detection/stack rebuild and plan-time matcher are outside
the byte-identity photometry contract.

**Ledger note:** `--full` exercises Phase 0 identity join on frozen plan files but **does not**
exercise the VSX->Gaia matcher (frozen `variable_targets.csv`). Matcher validation asset remains
the anchor-night regen (MATCHER-FIX-3).

### Commits (this close attempt)

| Commit | Summary |
|--------|---------|
| `266aff6` | STEP 0: `phase0_funnel.py`, `--full` funnel + plan-regen checks, ledger fingerprint block |
| `51a9b6e` | Science: identity join, `vsx_gaia_crossmatch.py`, docs, remove `phase01_match_radius_arcsec` (270 params) |
| `2a9cda0` | Regenerated `docs/VYVAR_FLOW_CZ.pdf` |

`--fast` PASS (1175 tests) after all commits.

### `--full` run 1 (`20260726T210249Z`, head `51a9b6e`, 2127s)

| Check | Result |
|-------|--------|
| full-plan-regen | **FAIL** (875 VT rows vs expected 245; new matcher histogram) |
| full-photometry-sha-core | **FAIL** (expected: science path changed; run `1c48d9fc...` n=325 vs snap `03d8fb64...` n=333) |
| full-photometry-sha-extended | **FAIL** (expected) |
| full-science-compare | **PASS** (n_lc=**162**, failures=0) |
| full-phase0-input-vt | **PASS** (frozen 245-row VT unchanged) |
| full-phase0-funnel | **FAIL** active **165** vs expected **169** |
| OVERALL | **FAIL** |

### Active count -- stop condition triggered

| Metric | B2 prediction | Observed |
|--------|---------------|----------|
| New active count | **164** | **165** |
| Dropped from 169 | 5 | **4** |

**Lost actives (4):** all `gaia_dr3_direct` mis-associations predicted in B2:

- Gaia DR3 1499883638682689920
- Gaia DR3 1500410236033012352
- Gaia DR3 1498513166158147968
- Gaia DR3 1499064433800590592

**Not dropped:** TOI-3919 (manual/exo row) -- remains active under identity gate (exo path bypasses
`gaia_match_source=masterstars` requirement). B2 counted it among the 5; implementation keeps it.

**Stop per task section 3:** active count is **not 164**; dropped set is **not** the predicted 5.
**No anchor re-cut, no second fresh run, no push.**

### Milan adjudication needed

1. Is **165** (drop 4 spurious `gaia_dr3_direct` only, keep TOI-3919) the intended outcome? If yes,
   update B2 expectation and re-cut fingerprint (`active_targets_rows: 165`).
2. Plan-regen check will continue to FAIL until regen histogram is re-baselined to MATCHER-FIX-3 output
   or regen is run against a regenerated VT policy Milan accepts.
3. Proceed with anchor re-cut only after (1) is explicit.

### Related read-only diagnostics (same session)

- `dev/results/CURSOR_RESULT_sigma_estimator_verify.md` -- sigma_pp unchanged (~46 ADU); bg_std estimator wrong on 450
- `dev/results/CURSOR_RESULT_skysurface_regression.md` -- SKIPPROC sky-surface blast radius + fix proposal

## CLOSE-2

Milan accepted **165 active** (169 - 4 spurious `gaia_dr3_direct` drops). Re-cut complete; two agreeing `--full` runs; pushed `535d863`.

### C1 -- TOI-3919 identity join (PASS, no stop)

Frozen `variable_targets.csv` row 243: `catalog=EXOPLANET`, `catalog_id=1497132660589966976`,
`gaia_match_source=masterstars_exo`. That `catalog_id` is present in `masterstars_full_match.csv`
(`source_type=GAIA_MATCHED`).

TOI-3919 reaches active **through** the identity join, not around it:

- `is_vsx_auto_selected_target()` returns **False** for `catalog=EXOPLANET`
  (`src_py/vsx_type_scope.py:65-80`) -- exo rows skip the VSX-only `gaia_match_source == masterstars` gate.
- VSX auto gate at `src_py/photometry_core.py:12752-12758` applies only when `is_vsx_auto` is True.
- Identity join at `src_py/photometry_core.py:12768-12772`: `ms_row = ms_by_cid.get(str(planner_cid))`;
  failure path `no_dao_detection` if absent.

### C2 -- 165 active vs n_lc=162

Three actives produced no light curve (`n_frames=0` in `photometry_summary.csv`, run `20260727T053550Z`):

| Target | zone_flag | skip_photometry | skip_reason | Notes |
|--------|-----------|-----------------|-------------|-------|
| CV CVn | saturated | True | zone_flag | Expected saturated-zone skip |
| Gaia DR3 1498278351706325248 | saturated | True | zone_flag | Expected saturated-zone skip |
| R CVn | linear | False | (empty) | Phase 2A empty-comp drop (`phase2a_empty_comp_drop=1`); active but no LC |

### Adjudication 165 vs 164

B2 over-counted by treating `masterstars_exo` as non-masterstars. True VSX-auto drop is **4**
(all predicted `gaia_dr3_direct` mis-associations):

- Gaia DR3 1499883638682689920
- Gaia DR3 1500410236033012352
- Gaia DR3 1498513166158147968
- Gaia DR3 1499064433800590592

TOI-3919 correctly retained (exo path exempt from VSX `gaia_match_source` gate only).

### Re-cut (VL-ANCHOR-WCSINV)

| Fingerprint | Value |
|-------------|-------|
| Frozen VT rows | 245 (`masterstars=178, gaia_dr3_direct=64, no_match=1, masterstars_exo=2`) |
| Plan-regen VT rows | **875** = **873 VSX + 2 EXOPLANET** |
| Plan-regen `gaia_match_source` | `gaia_dr3_direct=512, masterstars=205, masterstars_exo=2, no_match=156` |
| Active targets | **165** (`skip_photometry_true=2`; `skip_reason`: blank=163, zone_flag=2) |
| Zone histogram | linear=110, noisy1=10, noisy2=5, noisy3=38, saturated=2 |
| Core SHA | `1c48d9fc0056bc513c379fe1fb873e25215e5910f873e55e4a0b4fd8b5995d9f` n=**325** |
| Extended SHA | `744bce947ed47463227791bdac62dbec6885edfb1b637ae0a597f9d1973a71ba` n=**487** |
| Supersedes | core `03d8fb64...` / extended `bbfcc92e...` |

**Gate coverage (corrected):** plan-regen exercises the VSX->Gaia matcher; photometry SHA exercises
the identity join on frozen VT. Stronger than pre-B3 CLOSE wording.

**STATE.md NOT guaranteed:** anchor frozen inputs date 2026-07-16 and predate SKY-SURFACE regression
`013cb0c`; cross-ref `CURSOR_RESULT_skysurface_regression.md`.

Snapshot photometry mirrored from identity-gate run into
`draft_000435_snapshot_skysurface_20260716` before lock.

### Two fresh `--full` runs (byte-identical)

| Run | Timestamp (UTC) | Pipeline | Core/extended SHA | OVERALL |
|-----|-----------------|----------|-------------------|---------|
| 1 | `20260727T053550Z` | 2289s | `1c48d9fc...` n=325 / `744bce94...` n=487 | **PASS** |
| 2 | `20260727T061943Z` | 2299s | same | **PASS** |

Run1 vs run2: core SHA match **True**, extended SHA match **True**.

Also: `--fast` PASS (1175 tests), `ruff` clean, registered params **270** (STATE / PARAMS / registry).

### Commits pushed

- `0833c5c` -- anchor re-cut (session_baseline expectations, ledger fingerprints, STATE)
- `535d863` -- ledger verification stamp
- Prior identity-gate commits on same push: `266aff6`, `51a9b6e`, `2a9cda0`

Pushed to `origin/main` (`10608bb..535d863`).

