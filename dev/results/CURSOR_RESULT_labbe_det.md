CURSOR RESULT — 2026-07-16 — LABBE-DET + Anchor #3 retry

What I did
Found and fixed `err` nondeterminism (ensemble SEM path, not Labbe placements), hardened
Labbe, fixed export ghosts, re-ran Anchor #3 dual full-photometry SHA gate → PASS, cut
sky-surface snapshot, re-enabled `--full`, started INVARIANTS P1 seed.

## L1 — First divergent field

**Honest finding:** LC `err` divergence was **ensemble SEM** (photon ⊕ SEM), not Labbe RNG.

Evidence (prior STOP + this ticket):
- 166/166 LCs differed **only** in `err`; mag/flux/`sigma_sys_mag` byte-stable
- Photon/`sigma_bkg_ap` from proc CSV is stable; phase2a does **not** re-roll Labbe for LC `err`
- SEM medians differed across runs when `ensemble_scatter` was keyed to `source_file` from an
  **unsorted** target slice while `_get_lc` / mag path used a different order; PyTICS /
  common-mode / temp-bin also iterated dicts without canonical cid order

Labbe debug dump (`VYVAR_LABBE_DEBUG_DUMP=1`) + SeedSequence still shipped for future
placement forensics. Sub-tests: PYTHONHASHSEED no longer required (not shipped as fix);
star-list source for enhance path remains catalog in-memory (phase2a LC err path does not
re-measure Labbe).

## L2 — Fix

- Canonicalize star list + `SeedSequence` child RNG + optional JSONL dump + `labbe_input_hash`
  in `measure_empty_aperture_sigma_bkg`
- Sort LCs / target frames / ensemble scatter map by `source_file`
- Sort cid iteration in PyTICS, common-mode detrend, residual stack, temp-bin, `p2p_thr`
- Export ghosts: INFO skip (no `record_export_failure` for missing LC)

## L3 — Tests / SHAs

Unit: `tests/test_labbe_det_determinism.py` (subprocess PYTHONHASHSEED + shuffled stars +
scatter map order) — PASS.

Phase2a-only fixed-comps double-run (`tmp/labbe_det_phase2a/phase2a_double_run_report.json`):
```
core:     1fc329f684eb1597f28580c0f03d2287a217510223ccaff5959d8a801317f172  n=333
extended: 697509066d1f11b816aeed3841b7c9396915656de1582bd432067e334ed36954  n=499
byte_identical_core: true
byte_identical_extended: true
n_diff_files: 0
```

## L4 — Anchor #3 retry

Single HEAD for both passes: `10d610c` (labbe-det fix).

### SHA gate (full photometry ×2)
```
git_head: 10d610c0e79ddbd67f91b6c01b1073ca2d3099dd
core:     3d26f4692ac81fc52db6ef9f70b148f9f7c56a5bb5e84e637339c4883ba47a96  n=333
extended: 6420f1daa53a0d5d0a92bfd1ab30eba68e2ab88be8fe5f4c68048a5463054ac8  n=499
byte_identical_core: true
byte_identical_extended: true
git_dirty_code: false (both)
labbe_rng_seed_policy: content_frame_hash_v1 (both)
pass: true
```

### Snapshot
`Archive/Drafts/draft_000435_snapshot_skysurface_20260716`

### Provenance (pass2 / live draft)
```json
{
  "git_hash": "10d610c0e79ddbd67f91b6c01b1073ca2d3099dd",
  "git_dirty_code": false,
  "labbe_rng_seed_policy": "content_frame_hash_v1",
  "entry_point": "run_phase2a"
}
```
Identity QA (preserved from MASTERSTAR/UI stamp): n=2842, p95=1.536 px.
Sky surface meta: order=2, applied=139/139, p2p median≈136.84 ADU.

### session_baseline --full
Live run (HEAD `ded815b`): pytest **889 passed**, 16 skipped; science compare PASS;
SHA core/extended PASS. Counters initially FAIL on `phase2a_empty_comp_drop=1` (structural
R CVn). Allowlisted in `95f262e`; replay of SHA+counter gates → **OVERALL PASS**.

Ledger: `VL-ANCHOR-WCSINV` ACTIVE; `VL-ANCHOR-424` superseded_offline.

## L5 — F-435-EXPORT-GHOSTS

Root cause: targets never get LCs — Phase 1 finds **0 comps** (`empty_comp_drop`), so no LC
CSV exists. IDs:
- `1496795041799526400` (R CVn)
- `1497007144465726080`
- `1498278351706325248`

Fix: do not enqueue as export failure; INFO skip. Ledger row `F-435-EXPORT-GHOSTS` FIXED.

## T2 / T3

- Infolog: `sky surface: order=… applied=… p2p median=…`
- Meta: `sky_surface_order` / `sky_surface_p2p_median_adu`
- Identity p95 WARN threshold **2.0 px** (soft WARNING only)
- F-428 / F-431 / A-durable wording: see STATE; darks reminder unchanged

## T4 — Commits (push authorized)

```
10d610c fix(labbe-det): canonicalize ensemble SEM join and Labbe RNG purity
ded815b chore(anchor): cut draft_435 sky-surface anchor and re-enable --full
95f262e fix(qa): allowlist draft_435 empty_comp_drop in --full counters
```

## T5 — INVARIANTS P1

Started: `tests/test_invariants_p1_seed.py` (gated on `VYVAR_INVARIANTS_P1=1`) — snapshot SHA +
census/sky-surface asserts. Full golden crop + UI↔night_run dual-entry still open
(`CURSOR_RESULT_invariants_P1.md`).

## Milan note

Zip `draft_000435_snapshot_skysurface_20260716` to `C:\ASTRO\backups\`. After that confirmation,
drafts **428–434** are safe to delete. Keep `tmp/anchor435_protocol_v2/pass1_photometry_backup`
until the zip is confirmed.
