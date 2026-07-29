CURSOR RESULT - 2026-07-29 (session close 2026-07-26 .. 2026-07-28)

What I did
Closed the three open items from draft-454 analysis (infolog authority, INV-PREP-01 measurement
docs, CATALOG-PROVENANCE), produced disk cleanup plan + evidence extraction + manifest (no deletes),
synced STATE/DECISIONS/JOURNAL/ROADMAP, ran acceptance gates.

---

## Part 1 - Three fixes

### 1.1 Infolog authority (PASS)

**Premise:** Durable session log `infolog_20260728_152715.txt` on draft 454 contained `INV-PREP-01`;
ring-buffer export `infolog_20260728_164917.txt` did not.

**Fix:** `write_run_infolog()` finalizes the open durable session file (not a fresh ring-buffer dump).
Session file tagged `# authoritative: durable session log (complete operator record)`; ring-buffer-only
save tagged `# partial: ring-buffer tail only`. UI manual save and night_run auto-save both call
`write_run_infolog(draft_dir)` without `get_lines()`.

**Test:** `dev/tests/test_post453_infolog.py` -- guard lines + authoritative marker in session path;
partial label when no session.

### 1.2 INV-PREP-01 threshold anchored on measurement (PASS)

| Quantity | Value |
|----------|------:|
| Measured healthy (draft 454, BO CVn) | **0.03x** |
| SKIPPROC regression (draft 450 era) | **20-60x** |
| Threshold (constant) | **10x** |
| Margin at 10x | healthy **~330x below** warn; regression **2-6x above** |

Threshold **left at 10** -- separation enormous; original 1-5x estimate was ~2 orders high vs measured
healthy. Recorded in `docs/VYVAR_INVARIANTS.md`.

### 1.3 CATALOG-PROVENANCE (PASS)

**Stamped in run provenance** (`catalog_provenance.py` via `photometry_core._build_pipeline_provenance_block`):

| DB | path | size | rows | max_g_mag | fingerprint_sha256 (head/tail) |
|----|------|-----:|-----:|----------:|-------------------------------|
| Gaia DR3 | `GAIA_DR3/vyvar_gaia_dr3.db` | 53,137,264,640 B | 211,712,600 | 17.5 | `921ecb430eabd2f5d1c4815ea99bb08d2ee04734b8a45f66f60f0fe51126552d` |
| VSX local | `VSX/vyvar_vsx_local_v2.db` | 908,324,864 B | 7,827,904 | n/a | `13b4753f97c16a23f079026d9beab3eab0a1ebf3ea917f302e19e2f41b5086c5` |

Method: `sha256(size + first_1MiB + last_1MiB)` -- full SHA over 53 GB Gaia impractical.

**Compared in anchor gate:** `dev/scripts/session_baseline_check.py` adds `full-catalog-provenance`;
on mismatch reports `gaia_dr3: input catalogue changed` / `vsx_local: input catalogue changed` with
field deltas; photometry SHA core fails with *input catalogue changed* when catalog differs.

---

## Part 2 - Disk cleanup (evidence only; no deletes by agent)

### 2.1 Anchor needs (verified `session_baseline_check.py:475-482`)

Keep under `draft_000435`: `platesolve/<SETUP>/MASTERSTAR.fits`, `variable_targets.csv`,
`masterstars_full_match.csv`, `detrended_aligned/lights/<SETUP>/`. Plus snapshot
`draft_000435_snapshot_skysurface_20260716` and `draft_000435_p1mini`. `Raw/` and `calibrated/` not
required by `--full`.

### 2.2 Evidence extracted (PASS)

| File | SHA256 |
|------|--------|
| `dev/results/context/frame001_evidence/draft451_BO_CVn_Light_001.fits` | `15DED344DBC1CA4504FD419E38CC5A19DA16A6BBF001E5529B0005A0C2CFB041` |
| `dev/results/context/frame001_evidence/draft452_BO_CVn_Light_001.fits` | `E5B6B3E261842E833CE341ADAD669E5A686136734475D674AFA12387C5D3D774` |

533.45 ADU max diff on frame 001 (452 vs 454 calibrated compare).

### 2.3 Deletion plan for Milan (computed from on-disk sizes)

| Draft / path | Action | Size |
|--------------|--------|-----:|
| `draft_000435` | **keep**; delete `Raw/` + `calibrated/` only | trim **2.44 GB** (Raw 0.81 + cal 1.63) |
| `draft_000435_snapshot_skysurface_20260716` | **keep** entire | 5.95 GB |
| `draft_000435_p1mini` | **keep** entire | 0.69 GB |
| `draft_000438` | delete after evidence OK | 2.94 GB |
| `draft_000448` .. `draft_000455` (8 drafts) | delete after evidence OK | 39.64 GB |
| **Total potential freed** | | **~42.1 GB** |

Drafts 439-447, 440-446 not present on disk (already absent or never synced to this machine).

### 2.4 Manifest

`dev/results/context/deleted_drafts.md` -- one line per draft 438, 448-455 with date, field, entry,
masterstars rows, DAO_ONLY %, actives, note.

### 2.5 Backup flag for Milan

These three directories exist **only** under gitignored `Archive/` on one disk and **cannot be
regenerated** (detection behaviour changed since 2026-07-16; raw rebuild would not reproduce):

| Directory | Size |
|-----------|-----:|
| `Archive/Drafts/draft_000435` | 6.00 GB |
| `Archive/Drafts/draft_000435_snapshot_skysurface_20260716` | 5.95 GB |
| `Archive/Drafts/draft_000435_p1mini` | 0.69 GB |

**Operator action:** back up these three before any cleanup.

---

## Part 3 - Docs sync

Updated: `docs/VYVAR_STATE.md`, `VYVAR_DECISIONS.md` (INFOLOG-AUTHORITY, CATALOG-PROVENANCE),
`VYVAR_JOURNAL.md` (2026-07-29 session close entry), `VYVAR_ROADMAP.md` (CATALOG-PROVENANCE DONE,
DRAFT451 evidence SHAs).

---

## Gates

| Gate | Result | Run time |
|------|--------|----------|
| `--fast` | **PASS** (included in `--full` harness after ascii fix; 1206 pytest passed) | ~19 min pytest (within full run) |
| `ruff` | **PASS** | |
| anchor `--full` | **PASS** | pipeline **2689 s** (~44.8 min); total harness **3881 s** (~64.7 min) |

**Gate highlights:**
- `full-catalog-provenance` **PASS** (gaia rows=211712600 g<=17.5; vsx rows=7827904)
- `full-photometry-sha-core` **PASS** b7f980c09e238b85... n=325
- Infolog tests: 3 passed (`test_post453_infolog.py`)

**Dev note:** stale `infolog.cp312-win_amd64.pyd` shadowed source during first test attempt; renamed to
`.pyd.stale` for interpreted dev. Rebuild Cython release before compiled bundle use.

---

## Errors (if any)

None.

---

## Files changed

- `src_py/infolog.py`, `src_py/app.py`, `src_py/night_run.py`
- `src_py/catalog_provenance.py` (new)
- `src_py/photometry_core.py`
- `dev/scripts/session_baseline_check.py`
- `dev/tests/test_post453_infolog.py`
- `docs/VYVAR_INVARIANTS.md`, `VYVAR_STATE.md`, `VYVAR_DECISIONS.md`, `VYVAR_JOURNAL.md`, `VYVAR_ROADMAP.md`
- `dev/results/context/deleted_drafts.md`, `dev/results/context/frame001_evidence/*.fits`
- `dev/results/CURSOR_RESULT_session_close.md`
