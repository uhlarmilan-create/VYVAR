CURSOR RESULT - 2026-07-30 AUDIT STAGE 3 PART 0c

What I did
Stratified the Part 0b photometry delta tail and restricted the comparison to the anchor
snapshot cohort (`draft_000435_snapshot_skysurface_20260716` vs scratch `draft_000499`).

## Provenance

| Field | Value |
|-------|-------|
| `git_hash` | `75e08cc07c91882402dd16aa105348258eaf67e1` |
| `git_dirty` | `true` (uncommitted audit scripts + JSON) |
| Harness | `dev/scripts/audit_stage3_part0c_cohort_delta.py` |
| Raw JSON | `tmp/audit_stage3_part0c_results.json` |

---

## 0c.2 — Cohort mismatch (like-for-like)

| Cohort | Light curves |
|--------|-------------:|
| Rebuild (`draft_000499`) | 230 |
| Anchor snapshot | 162 |
| **Common** (compared) | **156** |
| Rebuild-only | 74 |
| Anchor-only (missing in rebuild) | 6 |

`variable_targets.csv` rows: rebuild **875**, anchor snapshot **245**.

**Why the plan expanded:** Phase 0 on the rebuild regenerated the full VSX-in-field planner
table (875 rows, none flagged `skip_photometry`). The frozen anchor snapshot retained the
older 245-row plan; photometry there wrote 162 LCs. The 74 rebuild-only targets are all present
in the rebuild planner (median Gaia G ~14.5 mag; none skipped). They are additional VSX
candidates the regen picked up, not a photometry bug.

**Like-for-like delta** (156 common targets only — the only meaningful comparison):

| Metric | ? `mag_calib_final` | ? `err` |
|--------|--------------------:|--------:|
| median | **?0.0078 mag** | +0.00043 mag |
| p95 | **0.430 mag** | 0.039 mag |
| max | **2.560 mag** | 0.948 mag |
| n epoch pairs | 20?192 | 20?192 |

Part 0b's delta table already paired common LCs only; restricting to the anchor target set
does not shrink the tail — the tail is **in** the 156 shared targets.

The 6 anchor-only targets are absent from the rebuild LC set (planner/regen did not select
them for photometry in this run).

---

## 0c.1 — Delta tail stratification

### Overall tail (|?mag| ? p95 ? 0.66 mag; 1?010 epoch pairs)

| Stratifier | Finding |
|------------|---------|
| Target magnitude | Tail splits evenly: **515** epoch pairs with target G < 14, **495** with G ? 14 (median target G in tail **13.71**). Not exclusively faint-end. |
| `n_good_comp` / ensemble | **`ensemble_changed` = 100%** of tail epochs — comparison ensemble size/membership differed between runs for every tail point. |
| Trust flag | Empty on all tail rows in both runs (no trust stratification signal). |
| Bright targets (G < 14) | **133** common bright targets; per-target \|?mag\| p95 max **1.91 mag** (e.g. `1498135552633294976` max **3.36 mag**). **Bright, well-measured targets moved materially** — not a faint-limit artefact alone. |

**Plain verdict:** The median shift (?0.008 mag) is negligible. The p95/max tail is **partly**
explained by ensemble/plan differences (100% ensemble change in the tail), but **bright
targets also show large per-target deltas** — that is the more serious finding: it is not
only faint-end noise.

Worst per-target |?mag| max (common cohort): up to **3.36 mag** (`1498135552633294976`).

---

## Contradictions with Part 0b

| Part 0b statement | Part 0c correction |
|-------------------|-------------------|
| "156 common LCs" | Confirmed; plus **6 anchor-only** absent from rebuild. |
| Delta on mixed cohort | Same numbers on anchor-restricted cohort (comparison was already common-only). |
| "230 vs 162" unexplained | **74 rebuild-only** targets documented; planner 875 vs 245 rows. |

---

## Files changed

- `dev/scripts/audit_stage3_part0c_cohort_delta.py`
- `dev/results/CURSOR_RESULT_audit_stage3_part0c.md`

**STOP GATE 0c** — awaiting Milan review.
