CURSOR RESULT - ANCHOR-PAIR EVIDENCE (431/432) - 2026-07-16

**Headline:** `--finalize` did **NOT** cut a snapshot. Orchestrator stopped at SHA gate:
`STOP: run1 != run2 on core SHA - no anchor cut`. No `VL-ANCHOR-WCSINV` row; `session_baseline_check.py`
unchanged (`DRAFT_ID=424`, offline VL-ANCHOR-424 still suspended). No Archive snapshot dirs.

**Also critical (beyond SHA):** drafts 431/432 reproduce the **428-class inflated census**
(masterstars 6699 / matched 3993 / unmatched 2706), not the healthy 429 census (3054 / 2875 / 179).
Standing QA seed: identity p95~6.24 px (identical on both). Architect decision needed before any
re-cut.

Bundle: `tmp/anchor_evidence_431_432/`

---

## E1 - Core artifacts | **PASS** (copied)

| File | Bundle path |
|------|-------------|
| Orchestrator report | `anchor_pair_report.json` (from worktree `tmp/anchor_run_wt/tmp/anchor_pair_run/`) |
| pipeline_meta 431 | `meta_431/pipeline_meta.json` |
| pipeline_meta 432 | `meta_432/pipeline_meta.json` |
| Log tail (~150) | `anchor_pair_run_tail150.log` |
| Extras | `night_run_{1,2}.json`, `draft_manifest_{431,432}.json`, `e3_timestamps_431.json` |

---

## E2 - Independent gates

### E2.1 Provenance | **PASS**

**draft_431**
```json
{
  "git_hash": "715391b9d30a56ba2697d3b7b112826da34d7eba",
  "git_dirty": false,
  "entry_point": "run_phase2a",
  "stamped_at_utc": "2026-07-15T17:44:55.031326+00:00"
}
```

**draft_432**
```json
{
  "git_hash": "715391b9d30a56ba2697d3b7b112826da34d7eba",
  "git_dirty": false,
  "entry_point": "run_phase2a",
  "stamped_at_utc": "2026-07-15T19:16:12.282518+00:00"
}
```

Both: `git_dirty==false`, hash == `715391b...` (**PASS**).

### E2.2 Core + extended SHA (recomputed NOW) | **FAIL - MISMATCH**

| | draft_431 | draft_432 |
|--|-----------|-----------|
| core SHA | `f79c1eab272acced3120aa27113b04ad19b16d9b6dc0cd1ea0078f67f3e4ed6c` | `65921558ea4068c82cb3a7f2073bbc342c170d3266159635acb7363dc7a48c92` |
| core n | 365 | 365 |
| extended SHA | `b195b0516403004ff86ab084b4dd72852577bb09e4393807dce98c4128d64201` | `0284a92100c59bf16699bc3b36f6c4e4e0014b878bfecddefcf5cf128511033c` |
| extended n | 546 | 546 |

**MATCH core: NO. MATCH extended: NO.**

**`err` scope:** LC CSVs are hashed as whole files in `compute_photometry_sha` (core patterns include
`lightcurve_*.csv`). Column `err` is in `PHOTOMETRY_QC_COLS_LC` (not science set).
`compare_photometry_science_meaningful` reports `err_check.enabled=false`, so science compare can
be **benign** while core SHA fails.

**Diff diagnosis (181/365 core files differ; all are LCs):**
- Differing columns only: `err` (181 LCs; median-of-medians Delta~0.0085 mag; p95 maxabs~0.191) and
  `delta_mag_sysrem` (181 LCs; p95 maxabs~0.118).
- Science mag/flux/BJD columns: 0 failures in science-meaningful compare (`benign: true`, 182 LCs).

**Verdict:** Anchor SHA gate (incl. `err`) **FAILS** - orchestrator correctly refused finalize.

### E2.3 Standing QA identity series | **PASS (present; values alarming vs 429)**

Identical on both metas (seeds the series):

| key | 431 | 432 |
|-----|-----|-----|
| n | 3993 | 3993 |
| p50_px | 0.6148 | 0.6148 |
| p95_px | **6.2398** | **6.2398** |
| p99_px | **9.1983** | **9.1983** |
| max_px | 22.1248 | 22.1248 |

(`wcs_roundtrip_pass=true`, p99~1.2e-11 on both.)

---

## E3 - draft_431 FRESH vs REUSED | **VERDICT: FRESH**

| Evidence | Value |
|----------|-------|
| Kill PID 14716 | ~`2026-07-15T16:58:20Z` |
| Launch PID 7792 | ~`2026-07-15T16:59:23Z` |
| `draft_000431/` root ctime | `2026-07-15T16:59:26.686Z` (**3 s after launch**) |
| calibrated/ ctime | `16:59:33Z` |
| processed/ ctime | `17:01:38Z` |
| platesolve/ ctime | `17:01:51Z` |
| detrended_aligned earliest mtime | MASTERSTAR `17:04:23Z` |
| calibrated earliest mtime | Light_001 `16:59:35Z` (after launch) |
| `draft_manifest.json` | `updated_utc=2026-07-15T16:59:33+00:00`, `calibration_mode=vyvar_calibrated` |
| Frame counts 431 vs 432 | identical: Raw 150, calibrated 150, detrended FITS 140, proc CSV 139 |
| Import timing | run1 `smart_import_session`~6.66 s; run2~1.23 s - both `Step 2: Import session` then full `[1/150] Calibrating...` (no skip-existing for calibration) |

Pre-launch Archive already contained a stale `draft_000431` from the killed stream, but the fresh
importer **recreated** that directory after launch (Windows ctime reset; all stage trees and
calibrated/processed/platesolve timestamps post-kill/post-launch). Content is not the partial kill
leftover.

**Decision rule:** FRESH -> 431 *would* be eligible as snapshot source **if** E2 SHA passed.
E2 SHA **failed**, so no standee. Finalize did **not** cut from 431.

---

## E4 - Census + fixes sanity | **FAIL vs 429 expectations** (431==432)

| Metric | Expect (429) | 431 | 432 | Delta 431<->432 |
|--------|--------------|-----|-----|-----------|
| active targets | 167 | **184** | **184** | none |
| LC count (report / files) | 164 | 184 / 182 files | 184 / 182 | none |
| masterstars rows | 3054 | **6699** | **6699** | none |
| matched | ~2875 | **3993** | **3993** | none |
| unmatched | ~179 | **2706** | **2706** | none |
| `vsx_known_variable` true | ~190-210 id-join | **46** | **46** | none |
| excluded_targets | 78 | 59 (OOF46 + no_dao12 + no_id1) | same | none |
| identity QA p95_px | ~0 (429 coords) | **6.24** | **6.24** | none |

**Infolog absence (both drafts):** no `infolog_*.txt` under `Archive/Drafts/draft_000431|432`
(worktree cwd did not deposit draft-root logs). Therefore **cannot verify from disk**:
UTC header, single ePSF notice, REPAIR summary line, `[AC] run summary`, post_match /
optimizer identity gate lines, stamp join text, WCS PASS *count*. Proxies from meta only:
`wcs_roundtrip_pass=true`; identity QA block present; stamp count 46 implies **C1 id-join did
not achieve 190-210** on these runs.

**431 vs 432:** no delta on any enumerated census row above.

**Interpretation:** Both fresh same-commit runs reproduce **428-class pass-2 inflation**, not
429 health. This is independent of FRESH/REUSED and independent of SHA `err` nondeterminism.

---

## E5 - Export soft failures | **PASS (enumerated; soft; identical both runs)**

Same three targets, both runs (`method=aperture` export batch):

| catalog_id | Root cause | LC on disk |
|------------|------------|------------|
| `1497007144465726080` | `LC CSV missing` | absent |
| `1497121459315202560` | `no exportable LC points (flags/mag empty)` | present (~26 KB) |
| `1498278351706325248` | `LC CSV missing` | absent |

On **429:** no `export failure` batch lines. Those Gaia IDs appear only in COMP / selection noise,
not as export failures.

**Publication-facing?** Failures are in the aperture **export** path (`[EXPORT] ... method=aperture`).
They are **not** labeled AAVSO/VAR.ASTRO specifically in the log, but this is the same batch that
feeds publication exports - treat as **publication-adjacent soft fails** (3 targets lack exportable
LC), not silent UI noise. Flag for architect: not 'AAVSO file corrupt', but targets dropped from
export pack.

---

## E6 - Snapshot / finalize state | **NO CUT (correct given E2)**

| Item | State |
|------|-------|
| Orchestrator exit path | SHA stop: `byte_identical_core=false` -> return 2; `--finalize` body **not** reached |
| Snapshot path | **none** (`Archive/Drafts/*snapshot*` empty) |
| SHA registered | **none** |
| `session_baseline_check.py` | still `DRAFT_ID=424`, `SNAPSHOT_NAME=draft_000424_snapshot_sigma_floor_20260713`, core `bf3743a1...` |
| VL-ANCHOR ledger | only `VL-ANCHOR-DQ-430` (disqualified); **no** `VL-ANCHOR-WCSINV` |
| `--full` | still suspended via VL-ANCHOR-424 `status: suspended_offline` |

Because E3=FRESH, the 'if finalize cut from REUSED 431 -> STOP' branch does **not** apply.
Anchor cut is simply **blocked** by E2 SHA failure (+ E4 census alarm for any future re-cut).

---

## Files in bundle

```
tmp/anchor_evidence_431_432/
  CURSOR_RESULT_anchor_evidence.md   (this file)
  anchor_pair_report.json
  anchor_pair_run_tail150.log
  meta_431/pipeline_meta.json
  meta_432/pipeline_meta.json
  night_run_1.json
  night_run_2.json
  draft_manifest_431.json
  draft_manifest_432.json
  e3_timestamps_431.json
```

---

## Architect cheat sheet

1. **No snapshot exist** - nothing to revert.
2. **E3 FRESH** - kill leftover did not poison content; ID reuse alone is OK.
3. **E2 SHA FAIL** on `err` (+ SysRem companion) - nondeterministic across consecutive clean
   same-commit runs; science mag columns match.
4. **E4 census FAIL** - 431/432 ~ 428 contamination profile; not anchor-worthy even if SHA were
   patched to ignore `err`.
5. Next instruction (architect): diagnose why headless `715391b` regenerates 428-class census /
   p95~6.24 while Milan UI 429 was healthy; then decide err-nondeterminism policy before another pair.
