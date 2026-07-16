CURSOR RESULT — 2026-07-16 — ANCHOR #3 STOP (SHA gate failed)

What I did
T1 protocol v2 on draft_435 through SHA gate. **STOP — no snapshot, no --full re-enable,
no VL-ANCHOR cut, no T4 push for anchor, T5 P1 not started.**

## T1 — gate chain

### Working tree / HEAD
```
HEAD: 3db08794e3bb446966e382daed4803bf8da220af
lineage: 3db0879 (docs) → 89842ff (T3 sky-surface) ✓
porcelain: untracked scratch only (.worktrees/, CURSOR_RESULT_*.md, scripts/* night_run helpers)
no modified tracked *.py at start
```

### Pass-1 (Milan UI photometry, preserved)
```
core_sha:     7156fecd71592a247649df07b2c89ae904d85c1ca12e6d3851952c90d458dbb8  n=333
extended_sha: e0686de60146bfa6516a11030d99c6ee8fc164d0c9d6b5bbadd80832ce34a135  n=499
```

**Pass-1 provenance (verbatim excerpt):**
```json
{
  "git_hash": "89842ff69b690a516863e504f4ebd70a850ba82c",
  "git_dirty": true,
  "git_dirty_code": false,
  "entry_point": "run_phase2a",
  "labbe_rng_seed_policy": "content_frame_hash_v1"
}
```
(identity p95 from meta: `matched_world2pix_identity_p95_px` = 1.536099044764055)

### Pass-2 (full `run_full_photometry_pipeline` on same draft after LC wipe)
```
core_sha:     3d26f4692ac81fc52db6ef9f70b148f9f7c56a5bb5e84e637339c4883ba47a96  n=333
extended_sha: 6420f1daa53a0d5d0a92bfd1ab30eba68e2ab88be8fe5f4c68048a5463054ac8  n=499
```

**Pass-2 provenance (verbatim excerpt):**
```json
{
  "git_hash": "3db08794e3bb446966e382daed4803bf8da220af",
  "git_dirty": true,
  "git_dirty_code": false,
  "entry_point": "run_phase2a",
  "labbe_rng_seed_policy": "content_frame_hash_v1",
  "git_dirty_code_files": [],
  "git_dirty_scratch_files": [".worktrees/", "CURSOR_RESULT_anchor_evidence.md", "...", "scripts/anchor435_protocol_v2.py", "..."]
}
```

### SHA gate
```json
{ "byte_identical_core": false, "byte_identical_extended": false, "pass": false }
```
Pass-1 photometry **restored** to `Archive/Drafts/draft_000435` from
`tmp/anchor435_protocol_v2/pass1_photometry_backup`.

---

## Per-column diff census (isolation re-run)

Re-ran **phase2a-only** with **fixed** `active_targets.csv` + `comparison_stars_per_target.csv`
from pass-1 (isolates Labbe / err from Phase 0+1 reselection).

```json
{
  "mode": "phase2a_only_fixed_comps",
  "byte_identical_core": false,
  "byte_identical_extended": false,
  "n_diff_files": 166,
  "column_diff_counts": { "err": 166 },
  "n_lc_with_err_diff": 166,
  "core_sha_pass1": "7156fecd71592a247649df07b2c89ae904d85c1ca12e6d3851952c90d458dbb8",
  "core_sha_pass2": "17cdc64c7eb959053c8da015663c200d7c9f33606f41c2065537f4bc402c07ba"
}
```

**Finding:** every lightcurve differs **only** in column `err` (mag/flux/`sigma_sys_mag` byte-stable).
Example `lightcurve_1485540612577549568.csv`: 139/139 rows differ; maxabs(err)=0.029053;
medianabs≈0.0095.

**Labbe seed probe:** `FILENAME` header is **None** on processed frames; seed still
deterministic from DATE-OBS/FRAME/NAXIS/r_ap. Same seed + same star list → identical
`sigma_bkg_ap`. Same seed + **perturbed star exclusion list** → different sigma.
→ Content seed is live, but empty-aperture placements still depend on the free-pixel mask
(star x/y set). Something in that path is not identical across the two photometry runs.

Artifacts:
- `tmp/anchor435_protocol_v2/protocol_v2_report.json`
- `tmp/anchor435_protocol_v2/sha_diff_census_phase2a.json`
- `tmp/anchor435_protocol_v2/pass1_photometry_backup/`

**No snapshot cut. No VL-ANCHOR row. No session_baseline --full update.**

---

## T2 (read-only while T1 ran; not applied — STOP)

Sky-surface **is** persisted:
- FITS: `VYSKYORD=2`, `VYSKYP2P` on all 139 `proc_*.fits` (median p2p ≈ 136.84 ADU)
- QC rows: `processed/lights/qc_metrics.csv` has `sky_surface_*` columns; applied=139/139
- **Not** in `pipeline_meta` / infolog summary (architect observation confirmed)
- Deferred: infolog line + meta keys (would be post-snapshot metadata commit — N/A until SHA passes)

Identity p95 WARN threshold 2.0 px: **not applied** (STOP).

## T3 (partial observations; ledger not closed for new anchor)

- Export soft-fails **still present** on phase2a re-run:
  ```
  [EXPORT] batch finished with 3 export failure(s)
  failed target ids: 1496795041799526400, 1497007144465726080, 1498278351706325248
  reason: LC CSV missing (method=aperture)
  ```
  → fresh ticket still warranted when anchor resumes (E5 not gone).
- F-428 / F-431 closeout wording / VL-ANCHOR ACTIVE: **deferred** until SHA gate green.
- A-durable / darks reminder: unchanged.

## T4 / T5

- **No commit/push** for anchor (gate failed). Diagnostic scripts under `tmp/` +
  `scripts/anchor435_protocol_v2.py` untracked.
- **INVARIANTS P1 not started** (blocked on Anchor #3).

## Errors

**SHA gate FAIL — Labbe `err` nondeterminism under protocol v2 (first live proof failed).**
Science columns otherwise match when comps are fixed. Next architect decision: harden Labbe
exclusion-mask / seed inputs so `err` is byte-stable, then re-run T1 from SHA gate.

## Milan note (when gate eventually passes)

Zip snapshot to `C:\ASTRO\backups\`; then drafts 428–434 safe to delete. **Not yet.**
