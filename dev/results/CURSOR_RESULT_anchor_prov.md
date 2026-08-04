CURSOR RESULT - 2026-08-04 16:45 UTC+2

ANCHOR-PROV: reconcile draft_435 anchor provenance (read-only).

Draft: draft_000435_snapshot_skysurface_20260716, setup NoFilter_60_2.

## STEP 1 -- Does the dirty set reconcile with f0b310e?

### pipeline_meta.json provenance (snapshot, mtime 2026-08-04 11:47:29 local)

| field | value |
|-------|-------|
| git_hash | 20dde2bcbacae25b14d532d3ef78524dd4e24d29 |
| git_dirty | true |
| git_dirty_code | true |
| git_dirty_code_files | src_py/comp_selection_per_target.py, src_py/config.py, src_py/pipeline.py |
| stamped_at_utc | 2026-08-04T09:25:48.110533+00:00 |
| entry_point | run_phase2a |

No porcelain block is stored in pipeline_meta.json. `_build_pipeline_provenance_block`
(`photometry_core.py:7047-7061`) captures porcelain at write time but does not persist
it in the JSON artifact.

### git_dirty_files (full list, 20 entries)

Each entry is `{path, content_sha256}`:

| path | content_sha256 |
|------|----------------|
| dev/results/CURSOR_RESULT_audit_stage3_part0b.md | 4fed1b2b96fb523ce0c9509df24608f1cdab7a6ee90eda0cf9d9b286c78250b0 |
| dev/results/CURSOR_RESULT_batch_E.md | de373abdaa29946361967572726ded1cfc8a40b316b47757676b91363651edbc |
| dev/scripts/audit_stage3_part0c_cohort_delta.py | bb40557ea2aa46ab3f8625163775ad43ebcfd4a8e1040ec87ebe7427a63561c6 |
| docs/VYVAR_AUDIT_CLOSURE_REGISTER.md | 78637ad744d9fef77c42f200d05f6a2b54961926bac2732e991fd4d6169234e3 |
| docs/VYVAR_AUDIT_FINAL.md | 0bdce08ea2c4bbff750194e3aff42e167843feee064c09c5937d6630efb2bb9e |
| docs/VYVAR_DECISIONS.md | 97cd90fa76480ee3b412f817cd2f23890c378985839702fea04f4704633d2b0c |
| docs/VYVAR_ROADMAP.md | 9f3029b056490f6c47baced4843429c4230a08c5625937ff1d40ca3e2de92bf2 |
| docs/VYVAR_STATE.md | 42543aabc63327575a3f948a42a9dc7aba37ad5b1bfb448933407aab113ee6d9 |
| docs/VYVAR_VALIDATION.md | 4e6f35e41985230979ca415bb09fd53096ea7c46ad591d4cd3cf3eff2f8a9853 |
| requirements.txt | f1430423babe58c79e44627e86aa0abef971f71e91eb009eeeaed4937018f1ff |
| src_py/comp_selection_per_target.py | 51bb6505f7249d3bbc5b394e3a4a3a8cb4a31b792ecb9f2f652b59a7d6b973cb |
| src_py/config.py | 5194bb953719d0caedb84839112df3a92cab137fd2bc18a6f314819ed2f68054 |
| src_py/pipeline.py | 483c8a7c7275c8bc5406cc30503ad34686e99088a2d2aa5e9f37f7fd1bf867db |
| dev/results/CURSOR_RESULT_wide_error_diag.md | 4b56250d44bc93bef3ce3a3ba9091312b1a296a9a2d4ba3c2ce1f0c5abc01711 |
| dev/tests/_tmp_batch_e_lc/ | DIR |
| dev/tests/test_batch_e_recut.py | 0bcd4e897b0207b7e65af76ed64fcd89ff50e4b8b85b9d4ac4e772f3bf372451 |
| dev/tools/batch_e_physical_recut.py | 973d41cf90391321c63e21c4ec33e67ec0641160afa1a7c132eeefa4e063082f |
| dev/tools/wide_error_budget_diag.py | d94c87c110de06f1a14248e3c5737b5811746c09e2f1533ddaa98be1555e07d6 |
| vyvar.sqlite3-shm | 38c5cb4c2df0461083e40514568b6e87e2faf8a3bfc11602f2d6ea11983b8d29 |
| vyvar.sqlite3-wal | bbf31aa2e279a1f097280e0b269208558f77112986d078c0299147b88aabb3a3 |

### git diff --name-only 20dde2b..ab0f669

All changed paths (25):

dev/results/CURSOR_RESULT_audit_stage3_part0b.md,
dev/results/CURSOR_RESULT_batch_E.md,
dev/results/CURSOR_RESULT_batch_E_physical_recut.md,
dev/results/CURSOR_RESULT_final_closure.md,
dev/results/CURSOR_RESULT_wide_error_diag.md,
dev/scripts/audit_stage3_part0c_cohort_delta.py,
dev/scripts/session_baseline_check.py,
dev/tests/test_batch_e_recut.py,
dev/tests/test_invariants_p1_seed.py,
dev/tools/batch_e_physical_recut.py,
dev/tools/wide_error_budget_diag.py,
dev/validation/VYVAR_VALIDATION_LEDGER.json,
docs/VYVAR_AUDIT_CLOSURE.md,
docs/VYVAR_AUDIT_CLOSURE_REGISTER.md,
docs/VYVAR_AUDIT_FINAL.md,
docs/VYVAR_DECISIONS.md,
docs/VYVAR_JOURNAL.md,
docs/VYVAR_LIMITATIONS.md,
docs/VYVAR_ROADMAP.md,
docs/VYVAR_STATE.md,
docs/VYVAR_VALIDATION.md,
requirements.txt,
src_py/comp_selection_per_target.py,
src_py/config.py,
src_py/pipeline.py

src_py/ only (3): comp_selection_per_target.py, config.py, pipeline.py

### Three-way set comparison

| set | paths |
|-----|-------|
| in dirty_files AND in 20dde2b..ab0f669 diff | 17 (all src_py + docs/results/scripts/tools listed above except _tmp and sqlite WAL/SHM) |
| in dirty_files but NOT in diff | dev/tests/_tmp_batch_e_lc/, vyvar.sqlite3-shm, vyvar.sqlite3-wal |
| in diff but NOT in dirty_files | dev/results/CURSOR_RESULT_batch_E_physical_recut.md, dev/results/CURSOR_RESULT_final_closure.md, dev/scripts/session_baseline_check.py, dev/tests/test_invariants_p1_seed.py, dev/validation/VYVAR_VALIDATION_LEDGER.json, docs/VYVAR_AUDIT_CLOSURE.md, docs/VYVAR_JOURNAL.md, docs/VYVAR_LIMITATIONS.md |

src_py: all three dirty src_py files are in the diff; no src_py in either asymmetric set.

### Content-level check (src_py dirty hashes)

Recorded `content_sha256` on the three src_py files matches the **current working tree**
but matches **neither** 20dde2b, f0b310e, nor ab0f669 for any of the three files.
Reconciliation is **name-level only**; the code at run time was a third uncommitted state.

### STEP 1 verdict

**RECONCILES** (name level) -- every src_py entry in git_dirty_files appears in the
20dde2b..ab0f669 diff. Unexplained non-src_py dirty entries at run time:
dev/tests/_tmp_batch_e_lc/, vyvar.sqlite3-shm, vyvar.sqlite3-wal. Content agreement with
f0b310e is **not** established (hashes differ from f0b310e for all three src_py files).

## STEP 2 -- What triggered the 10:20-11:47 run?

### Evidence found

| source | span (local) | entry / notes |
|--------|--------------|---------------|
| tmp/batch_e_physical_recut.log | 2026-08-04 10:14:43 - 11:47:29 | `_night_run_preprocess` -> `_night_run_platesolve` -> `run_full_photometry_pipeline`; target archive `Archive/Drafts/draft_000500` (headless harness `dev/tools/batch_e_physical_recut.py`) |
| snapshot pipeline_meta.json | stamped 2026-08-04 11:25:48 local (09:25:48 UTC); file mtime 11:47:29 | entry_point `run_phase2a`; git_hash 20dde2bc; POST batch E config |
| snapshot artifact mtimes | masterstars 10:20:09; comparison_stars 11:24:59; check_kmag 11:25:52-11:33:10 | under draft_000435_snapshot_skysurface_20260716 |
| tmp/wide_error_budget_diag.json | 2026-08-04 09:17:57 | read-only diagnostic (predates batch E chain) |

draft_000500 pipeline_meta.json: same git_hash, stamped_at_utc, and entry_point as
snapshot; mtime 11:47:29.

### Not found

- No files under `Archive/logs/` or repo `logs/` (`run_preflight_error_*` per
  `run_preflight_log.py`).
- No infolog under the snapshot draft tree.
- batch_e_physical_recut.log contains **no** path references to
  draft_000435_snapshot_skysurface_20260716.

### STEP 2 answer

Headless batch E physical re-cut on **draft_000500** is documented 10:14-11:47 local.
Concurrent snapshot writes (10:20-11:47) carry `run_phase2a` provenance on the snapshot
tree but **CANNOT DETERMINE** what invoked them or whether that invocation was UI or
headless -- no log covering the snapshot run exists in the searched paths.

## STEP 3 -- Is the snapshot still a snapshot?

### 3.1 Zip SHA256

| zip | exists | recomputed SHA256 | STATE record match |
|-----|--------|-------------------|-------------------|
| C:\ASTRO\backups\draft_000435_snapshot_skysurface_20260716.zip | yes | a35d22354666e359ce1bdd9a6eb207d5d768466a67fcdb77c22425eabb3f84a0 | **match** |
| C:\ASTRO\backups\draft_000435_anchor_live_20260716.zip | yes | a4bb42d255e542b4a516197d5efe1a6304602b331680ac554caf41a244070faf | **match** |

### 3.2 Zip vs live tree (platesolve/NoFilter_60_2/photometry/, no full extract)

| item | zip | live |
|------|-----|------|
| photometry/ entries (under prefix) | 1196 | (not counted separately) |
| check_kmag_*.csv count | 166 | 248 |
| check_kmag only in zip | 18 names | -- |
| check_kmag only in live | -- | 100 names |
| comparison_stars_per_target.csv size | 809531 | 1100385 |
| masterstars_full_match.csv size | 1071181 | 1443946 |
| pipeline_meta.json size | 19394 | 28408 |

### 3.3 pipeline_meta.json inside zip (extracted to tmp/anchor_prov_zip_pipeline_meta.json)

| field | zip | live snapshot |
|-------|-----|---------------|
| git_hash | 10d610c0e79ddbd67f91b6c01b1073ca2d3099dd | 20dde2bcbacae25b14d532d3ef78524dd4e24d29 |
| git_dirty | true | true |
| stamped_at_utc | 2026-07-16T13:35:35.955699+00:00 | 2026-08-04T09:25:48.110533+00:00 |
| admission_sat_peak_frac in config_snapshot | **absent** | **present (0.7)** |

Zip predates batch E re-cut (July 2026 provenance, no admission_sat_peak_frac).

Live snapshot photometry SHA (computed 2026-08-04, not re-run gate):
core `97fee58e67681e4a...` n=1121; extended `92403a445618f6a7...` n=1368.
draft_000500 on disk: core `5bccd85a94d95031...` n=497; extended `7fdcdca402ad47d0...` n=744.

WIDE-ERR 09:17 generation (162 successful fields) is **not** preserved in the live tree
(248 check_kmag sidecars, POST batch E meta) and is **not** in the zip (166 sidecars,
July 2026 meta).

## STEP 4 -- Which generation does the recorded anchor gate describe?

### Recorded fingerprints and timestamps

| location | core SHA (n) | extended SHA (n) | timestamp |
|----------|--------------|------------------|-----------|
| dev/scripts/session_baseline_check.py | 5bccd85a... (497) | 7fdcdca4... (744) | file in repo at ab0f669+ |
| dev/tests/test_invariants_p1_seed.py | same | same | file in repo at ab0f669+ |
| dev/validation/VYVAR_VALIDATION_LEDGER.json VL-ANCHOR-WCSINV | same | same | last_verified **2026-08-04**, commit 20dde2b |
| dev/results/CURSOR_RESULT_final_closure.md | same | same | dated **2026-08-04T11:30:00Z** |
| dev/results/CURSOR_RESULT_batch_E_physical_recut.md | computed on draft_000500 | same | **2026-08-04T09:50:00Z** (GATE 2 pending at write) |

Recorded values correspond to batch E physical re-cut fingerprints from **draft_000500**
(batch E physical recut result file, final_closure, ledger update on 2026-08-04).

### Relative to 10:20-11:47 snapshot writes

Ledger/fingerprint push timestamp (2026-08-04, final_closure 11:30 UTC) is **after** the
start of snapshot artifact writes (masterstars 10:20 local) and **after** batch E chain
end (11:47 local) if interpreted in local time for the latter.

Live snapshot SHA **does not match** the recorded 5bccd85a / 7fdcdca fingerprints.
Recorded gate numbers describe **draft_000500** batch E output, not the current snapshot
photometry tree. The 10:20-11:47 run rewrote snapshot files that the recorded gate was
later claimed to cover (final_closure "snapshot refreshed") but current on-disk snapshot
does not byte-match that generation.

## Verdict

**ANCHOR-COMPROMISED** -- recorded fingerprints do not match live snapshot photometry;
offline zip matches its recorded hash but predates batch E and differs from live (166 vs
248 check_kmag; July pipeline_meta); WIDE-ERR 09:17 generation is not recoverable from
zip or live tree; dirty set reconciles to f0b310e file names only (content hashes differ).

## ANCHOR-PROV-2 (2026-08-04)

### C1 -- Content reconciliation (EOL-safe)

`git hash-object` normalizes line endings; `content_sha256` in pipeline_meta is
sha256 of **raw working-tree bytes** (`photometry_core.py:6953-6957`, CRLF on Windows).
Those two measures are not directly comparable to each other or to git blob ids.

| file | git hash-object (worktree) | f0b310e | 20dde2b | ab0f669 | worktree matches |
|------|---------------------------|---------|---------|---------|-----------------|
| src_py/comp_selection_per_target.py | 4d3f875add2ee9f7f8db1298a836e6e3f76042d7 | 4d3f875a... | 48e4aba4... | 4d3f875a... | **f0b310e, ab0f669** |
| src_py/config.py | cac757ff4eba25694cb45731bab1cbea6596e22e | cac757ff... | ca54e64b... | cac757ff... | **f0b310e, ab0f669** |
| src_py/pipeline.py | e88a9bffdb56f5b287bfdb670312ed434d79c58a | e88a9bff... | 2ccfbd98... | e88a9bff... | **f0b310e, ab0f669** |

pipeline_meta `content_sha256` values (raw-byte sha256 at run time):

| file | content_sha256 in meta | matches current raw bytes |
|------|------------------------|---------------------------|
| comp_selection_per_target.py | 51bb6505... | yes |
| config.py | 5194bb95... | yes |
| pipeline.py | 483c8a7c... | yes |

LF-normalized sha256 of current raw bytes equals f0b310e blob content for all three
files. The prior "third uncommitted state" conclusion came from comparing meta
content_sha256 (raw CRLF bytes) to git commit blob hashes -- **NOT COMPARABLE**.

**Revised STEP 1 verdict: RECONCILES at content level** -- run-time code on the
11:25/11:47 run was the change set later committed as **f0b310e** (and ab0f669).
Name-level reconciliation from STEP 1 stands; content reconciliation now agrees.

### C2 -- WIDE-ERR generation inside the zip

From `C:\ASTRO\backups\draft_000435_snapshot_skysurface_20260716.zip`, 166
check_kmag_*.csv entries read in-zip (no full extract):

| metric | zip | tmp/wide_error_budget_diag.json |
|--------|-----|--------------------------------|
| n sidecar files / successful fields | 166 | 162 |
| distinct check_catalog_id | 2 | 2 |
| 1499906247391001088 | **164** | **160** |
| 1497528072458898432 | **2** | **2** |

Zip sidecars carry `kmag` only (no G column). G for 1499906247391001088 from zip
masterstars_full_match.csv: **8.743** (matches WIDE-ERR T1 G~8.74). Sidecar `kmag`
median for that id: ~8.47 (differential magnitude, not catalog G).

Reading applied: 164 + 2 in zip vs 160 + 2 in WIDE-ERR JSON -- same two-id
distribution; delta 4 files is consistent with 166 sidecars vs 162 production-LC
successes (sidecars present, LC rebuild failed or skipped on 4).

**Revised STEP 3 (partial):** the WIDE-ERR 09:17 generation **IS** the July zip
photometry generation for check-star identity and is **RECOVERABLE** from the zip.
It is **not** on the live snapshot tree (248 sidecars, POST batch E meta).

### C3 -- Which draft is the anchor gate bound to?

Verbatim bindings:

**dev/scripts/session_baseline_check.py:37-42**
```
DRAFT_ID = 435
SNAPSHOT_NAME = "draft_000435_snapshot_skysurface_20260716"
EXPECTED_PHOTOMETRY_SHA_CORE = "5bccd85a94d95031f80d372141ae0c61b0d8b0b2026c6bb15076d4e6a5e9b77e"
EXPECTED_PHOTOMETRY_SHA_EXTENDED = "7fdcdca402ad47d044ca7b34d1f1c0d09185d02016f94a1a3747cb0528862ea2"
```

**dev/tests/test_invariants_p1_seed.py:20-22**
```
SNAPSHOT = "draft_000435_snapshot_skysurface_20260716"
EXPECTED_CORE = "5bccd85a94d95031f80d372141ae0c61b0d8b0b2026c6bb15076d4e6a5e9b77e"
EXPECTED_EXT = "7fdcdca402ad47d044ca7b34d1f1c0d09185d02016f94a1a3747cb0528862ea2"
```

**dev/validation/VYVAR_VALIDATION_LEDGER.json VL-ANCHOR-WCSINV (lines 143-150)**
```
"description": "In-Archive BO CVn Anchor #3 sky-surface snapshot draft_000435_snapshot_skysurface_20260716"
"last_verified": "2026-08-04"
"commit": "20dde2b"
"notes": "... full production path from calibrated lights on scratch draft_000500; snapshot refreshed."
```

**dev/results/CURSOR_RESULT_batch_E_physical_recut.md:71-74**
```
| Tier | Physical (draft_000500) | Batch D ledger (GATE 1) | Delta |
| core | `5bccd85a94d95031f80d372141ae0c61b0d8b0b2026c6bb15076d4e6a5e9b77e` (n=497) | ...
| extended | `7fdcdca402ad47d044ca7b34d1f1c0d09185d02016f94a1a3747cb0528862ea2` (n=744) | ...
```

**dev/results/CURSOR_RESULT_final_closure.md:27-28** claims copy
"from `Archive/Drafts/draft_000500` to
`Archive/Drafts/draft_000435_snapshot_skysurface_20260716`" (calibrated,
detrended_aligned, platesolve).

Assessment:

- **(i) seeding error:** fingerprints computed on **draft_000500**; session baseline,
  invariants P1 seed, and ledger gate all name **draft_000435_snapshot** and expect
  those fingerprints on the snapshot tree. Live snapshot SHA does not match
  (core `97fee58e...` n=1121 vs expected `5bccd85a...` n=497). **Supported.**
- **(ii) deliberate rebaseline to draft_000500:** no file retargets DRAFT_ID or
  SNAPSHOT to 500; ledger still names 435 snapshot. **Not supported as deliberate
  rebaseline.**
- **(iii) legitimately comparable via derivation:** batch E input is
  `draft_000435/calibrated/lights` (`CURSOR_RESULT_batch_E_physical_recut.md:18`);
  that documents shared **input lights**, not photometry-byte identity between 500
  and snapshot. Post-recut SHA on 500 != live snapshot SHA. **Not supported for
  gate comparability.**

**C3 conclusion: (i)** -- fingerprint from draft_000500 seeded into gates bound to
draft_000435_snapshot; copy claimed in final_closure but not reflected in current
snapshot photometry SHA.

### Revised verdict

**Verdict changed** from ANCHOR-COMPROMISED to **ANCHOR-RECONSTRUCTIBLE**.

Why it changed:
- C1: run-time code **does** match f0b310e at content level (prior "third state" was
  an EOL-unsafe comparison artifact).
- C2: WIDE-ERR generation **is** recoverable from the July zip (164+2 id split).

Why not ANCHOR-INTACT:
- Recorded batch E fingerprints (`5bccd85a` / `7fdcdca4`) do **not** describe live
  snapshot photometry (still true).
- Live snapshot does **not** match the offline zip (248 vs 166 check_kmag; POST vs
  PRE batch E meta).

Original three pillars, restated:

| pillar | status |
|--------|--------|
| dirty set reconciles to f0b310e | **survives** (now content-level via git hash-object) |
| snapshot matches its zip | **failed** (live tree post batch E rewrite) |
| recorded gate describes current snapshot artifacts | **failed** (gate expects 500 fingerprints on 435 tree; live SHA mismatch) |

Recoverable generations: (a) WIDE-ERR / pre-batch-E check-star cohort from July zip;
(b) batch E fingerprints on draft_000500 on disk (core `5bccd85a...` n=497 verified
2026-08-04 read-only).

**Open item (unchanged, not resolved here):** no log accounts for writes into
draft_000435_snapshot_skysurface_20260716 between 10:20 and 11:47 local.

**ANCHOR-RECONSTRUCTIBLE**

