# CURSOR RESULT - SAT-RERANK-01 FOLLOW-UP

Date: 2026-08-17
Draft 515 photometry SHA (core): **8f107cf**
(`8f107cfa2f4af63335e51608f6df7658d03ba14fbcd82c12fc07c4dab06933b8`, n=193 files)
Full SHA (with comp_qa): `8dcfcfb77f6512c4535e90b4d981d1df3f9cc9393369ffbe6abb0054dba5c84d` (n=289)
Compared numbers from da9cce4: METER-DRAFT-DEP-01 (cross-SHA).
Push: NOT authorized.

Premise: this report stamps the post-DAG-trim, post-weight-rewrite draft 515
state and explains why Phase 2A emitted 96/97 light curves instead of 49.
It does **not** accept the 96-LC product. B4/B5 meters were not run.

## Part 1 - state stamp

### 1.1 Photometry SHA

Quantity: SHA256 of the photometry_sha core fileset (LC + Phase-2A
comparison pool), domain = draft_000515 on disk after the idempotent
weight rewrite of `comparison_stars_per_target.csv`.

| quantity | value |
|----------|-------|
| core SHA | 8f107cfa... (prefix 8f107cf) |
| core n files | 193 |
| full SHA | 8dcfcfb7... |
| full n files | 289 |
| prior SHA (da9cce4) | D515-ACCEPT / SAT-LIMIT product |

Git rebuild stamp on the harness log: `1a516c7` (Part A commits). The
photometry SHA above is the Archive product, not the git SHA.

### 1.2 DAG surgery audit (architect finding 3)

`stamp_pipeline_stage` is append-only. The rebuild died at
`photometry_core.py:12098` with INV-DAG-01: stage `phase2a` seq=6 goes
backwards (max stamped seq=7). Leftover stamps in `pipeline_meta.json`
from the da9cce4 night:

| name | seq | ts (UTC) | action |
|------|----:|----------|--------|
| masterstar | 3 | 2026-08-16T11:18:56 | kept |
| phase01 | 5 | 2026-08-16T14:56:57 | trimmed |
| phase2a | 6 | 2026-08-16T21:31:31 | trimmed |
| postprocess | 7 | 2026-08-16T21:31:31 | trimmed |

Re-stamped on 2026-08-17T10:07:21Z: phase01 seq=5, phase2a seq=6,
postprocess seq=7. masterstar ts unchanged.

Files rewritten after the crash (science LCs already on disk):

- `pipeline_meta.json` -- stages list only (trim + re-stamp)
- `comparison_stars_per_target.csv` -- columns `comp_weight` and
  `sigma_eff_mag` via `rewrite_comparison_stars_weights_csv`
  (`src_py/comp_weights.py`)

Idempotence proof (weight rewrite, cheap):

| step | sha256 of comparison_stars_per_target.csv |
|------|------------------------------------------|
| after first canonical rewrite | `ff2df81b6c6a4191df726fa4afe4c6462d0309673f0b945755ee309084c12550` |
| after second rewrite | identical (`ff2df81b...`) |
| after third rewrite | identical |

First rewrite after the DAG-completion pass was **not** byte-identical
to the DAG-completion write (CSV dialect / float roundtrip). After the
canonical rewrite, further rewrites are byte-identical. That is the
proof. Re-running `run_end_of_run_invariants(stamp_postprocess=True)`
would **append** another postprocess stamp; pipeline_meta cannot be
byte-identical under append-only DAG. Named spec defect, not a product
mutation of LCs.

Harness `dev/tools/draft_515_headless_phase012a.py` now trims leftover
phase01+ stamps before a rebuild so this crash does not recur.

### 1.3 `per_frame_saturation_enabled` registry

Pre-existing key. Not a silent 289th.

| quantity | value |
|----------|-------|
| registry n (`load_registry`) | **291** |
| AppConfig fields | **291** |
| config.json persisted keys | **256** (validate_config OK, 0 warnings) |
| key present | yes |
| scope / scope_key | universal / none |
| kind | static |
| config.json value | true (this rebuild) |
| AppConfig default | false |

Task cited 288 (ZONE-SIMPLIFY 2026-08-07). Live `docs/VYVAR_PARAMS.md`
already says Entries: 291. Flipping the flag to true did not add a key.
Spec defect: task 288 is stale, not a new-key defect.

## Part 2 - 96/97 gating (architect finding 1) -- HIGHEST PRIORITY

### 2.1 B6 table (this SHA 8f107cf)

Phase 1 photometry set = 97 targets with ensembles.
LC files written = 96. Silent drops = **0**.

Full per-target table: `dev/results/SAT_RERANK_01_gating_97.csv`
(catalog_id, name, zone_flag, skip_photometry, sat_clean_frac, n_comp,
lc_emitted, gate_reason_if_no).

The one gated-out target:

| catalog_id | name | zone | mag | gate |
|------------|------|------|----:|------|
| 1497037209236836736 | Gaia DR3 1497037209236836736 | linear | 15.1258 | skip_reason=`zone_flag` (PFS fallback; sat_clean_frac NaN). Mag > target_depth_g=15.0 -- this is one of the three previous `below_target_depth` rows. n_comp=8 (ensemble existed). |

RUN-HARDEN account of all 97: 96 emitted + 1 recorded gate. 0 silent.

Among 218 Phase 0 actives (not the 97): skip_reason histogram after PFS
rewrite is `no_comps` 118 + empty 96 + `zone_flag` 4 = 218.

### 2.2 Diff vs da9cce4 (97 -> 49)

Previous (da9cce4): 97 -> 49 LCs; 45 `zone_noise` + 3 `below_target_depth`;
0 silent. Zone on 97: linear 52 / noise 45 / saturated 0.

This SHA: 97 -> 96 LCs. Zone on 97: linear 51 / noise 45 / saturated 1.

Catalog zone counts (MASTERSTAR, n=3621 rows):

| zone | da9cce4 (SAT-LIMIT before) | this SHA |
|------|---------------------------:|---------:|
| linear | 2908 | 2884 |
| noise | 713 | 713 |
| saturated | 0 | 24 |

Noise catalog count **unchanged**. The 24 saturated rows came from linear
(2908-24=2884), SAT-LIMIT-01 reclassify, including C2. One of the previous
52 linear **targets** is now zone=saturated: CV CVn `1497007144465726080`
(G=5.94). It still has an LC (sat_clean_frac=1.0). Not a gated-out flip.

47 targets flipped gated-out -> emitted:

- 45 `zone_flag=noise` (all previous zone_noise set). All 45 now have LCs.
- 2 of 3 `below_target_depth` (linear, mag>15):
  `1497622939696499840` (15.131), `1498089270065726464` (15.248).
- 1 of 3 depth rows stayed gated (fallback), listed in 2.1.

49 + 47 = 96.

### 2.3 Root cause (file:line)

Not a change in zone-classifier inputs. Catalog noise stays 713.
`per_frame_saturation_enabled=true` did **not** replace stack
`peak_max_adu` for zone assignment. Proc-frame `sat_clean_frac` is 1.0
on every target with data (214/218); per_frame_sat_n_skipped=0.
The flag did not mask saturated epochs on this night. It rescued
whole-star skips.

Two coupled branches:

1. `apply_per_frame_saturation_to_active_targets`
   (`src_py/photometry_core.py:7324-7348`) blanks `skip_reason` then
   treats **any** `skip_photometry=True` as saturation-legacy
   (`decide_target_saturation_policy`, `:7170-7244`:
   `legacy = bool(legacy_skip) or zf == "saturated"`). Noise and
   below_target_depth skips are therefore "rescued" when clean_frac
   >= 0.5. pipeline_meta: `per_frame_sat_n_rescued=166` of 218 actives.

2. Phase 2A per-target loop (`photometry_core.py:10244-10249`):
   when PFS is ON, do not re-force skip from `zone_flag` in
   `{saturated, noise}`. That disables TARGET-DEPTH-02
   (`:14678-14680`, `:10247-10249` comment).

Phase 1 still flags noise (`skip_reason=zone_noise`) and depth
(`target_depth.json` still records `n_masked_below_target_depth=3`).
PFS then wipes those flags on `active_targets.csv` before photometry.

### 2.4 Verdict: **(b)** DEFECT

The 47 extra LCs are a silent weakening of TARGET-DEPTH-02 /
zone_noise, not a correct consequence of per-frame saturation stats.
Catalog noise count did not change. clean_frac is identically 1.0.
STOP. B4/B5 meters not run. EXPORT-HDR-01 not started.
No science-code patch in this follow-up (task: stop and report).

## Part 3 - BO ensemble (architect finding 2) -- read-only, not closed

Meters stopped under (b). Facts already on disk / in the rebuild log,
not a new photometry run.

### 3.1 True re-selection, leftover membership

Phase 1 log (`tmp/draft_515_sat_rerank_01.log`, UTF-16), target 15/97
BO CVn at 511.3 s:

```
[DEBUG BO CVn] Step A: global_comp_pool size = 1432
[DEBUG BO CVn] G: after RMS filter (+ MAD) -> 623
[DEBUG BO CVn] max_comp_rms=0.0800 n_comp_max=8
[DEBUG BO CVn] H: after n_comp_max truncation -> 4
[DEBUG BO CVn] Final: comp stars selected -> 4
```

This is a real `_select_comps_by_rms_then_color` pass, not a weight-only
rewrite of a frozen membership. Final IDs equal the old five minus C2
`1500748301498613248`. C2 is absent from every ensemble (B2 fire proof:
0 of 24 SAT-LIMIT IDs in `comparison_stars_per_target.csv`).

### 3.2 Fifth comp -- not closed

H=4 with n_comp_max=8 means the colour-ladder + isolation step between
G (623) and `head(n_comp_max)` returned 4, not a hard n_comp_max=4.
PowerShell wrapped `[COMP]` isolation/ladder lines, so the exact 5th
candidate vs thresholds is not in the log. Not reconstructed: verdict
(b) stop. "The pool had nothing better" is **not** claimed.

### 3.3 New BO ensemble (read of SHA 8f107cf CSV)

n_eff = (sum w)^2 / sum(w^2), w = `comp_weight` (pytics / sigma_eff).

| catalog_id | comp_weight |
|------------|------------:|
| 1497771992240531712 | 9241.188632 |
| 1499200223486564608 | 6486.886182 |
| 1497974027502858240 | 4701.614479 |
| 1497368849430107904 | 1886.405643 |

n_eff = **3.252** (dimensionless). References: old-with-C2 2.45;
leftover product-frame 1.20 (different weight definition). This n_eff
is not < 1.5. Same four IDs as the leftover set; weights are the
sigma_eff rewrite, not the 1.20 leftover weights.

### 3.4 FW CVn (read-only)

IDs identical to da9cce4 production FW ensemble (8 comps). n_eff = 6.313.
Check `1497368849430107904` is in the **BO** ensemble, not FW -- same
as 01B. Valid as an FW meter if/when B4 is authorized. Check
`1498020894186918144` is in the FW ensemble (01B used it only as BO's
meter). B4 not run.

## Part 4 - B4 + B5

Not run. Verdict (b). BIN-8-9-REGRESSION-01 stays OPEN (no new number).
01B 2x2 table not overwritten.

## Spec defects

1. Task registry count 288; live registry/AppConfig = 291. Flag is
   pre-existing. config.json persisted subset = 256.
2. `stamp_pipeline_stage` append-only: postprocess re-stamp cannot prove
   pipeline_meta byte-identity.
3. First weight rewrite after DAG completion was not byte-identical to
   the next rewrite; subsequent rewrites are.
4. Part 3.2 5th-candidate cut not measured (stop under b).
5. Parent rebuild log is PowerShell UTF-16; `[COMP]` lines wrapped as
   NativeCommandError.

Physics outranks the spec: TARGET-DEPTH-02 outranks turning on a
saturation-named flag.

## Docs impact

- this RESULT + `SAT_RERANK_01_summary.json` + `SAT_RERANK_01_gating_97.csv`
- STATE / JOURNAL / DECISIONS / ROADMAP: SAT-RERANK-01 blocked on (b)
- EXPORT-HDR-01 not started
- FLOW PDF: none (no intended behavior change shipped)

## Recurrence

Recurrence: n/a (first occurrence / not a bug-class test yet). The
coupling is a production defect; a test should assert PFS ON does not
clear `zone_noise` / `below_target_depth`. Not added here (stop).

## Files

- `dev/results/CURSOR_RESULT_SAT_RERANK_01.md`
- `dev/results/SAT_RERANK_01_summary.json`
- `dev/results/SAT_RERANK_01_gating_97.csv`
- `dev/results/SAT_RERANK_01_followup_raw.json`
- harness DAG trim: `dev/tools/draft_515_headless_phase012a.py`

## Errors

Rebuild process exit 1 = INV-DAG-01 leftover stamp (products complete).
Part 2 verdict (b) is the blocking error for acceptance meters.

`--fast`: OVERALL FAIL. pytest 1 failed, 1438 passed, 28 skipped.
Failed: `test_docs_sync_guard.py::test_flow_doc_config_facts` because
live `config.json` has `per_frame_saturation_enabled=true` while
`flow_doc_facts.py` documents the default false. Not patched: FLOW
facts must stay at the AppConfig default; turning facts to true would
lie. Git untracked RESULT artifacts WARN (expected). origin/main
differs (local ahead; push not authorized).
