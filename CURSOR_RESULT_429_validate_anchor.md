CURSOR RESULT — 2026-07-16 (UTC+2)

What I did
Validated draft_429 against F-428-WCS-INV checklist (Part A), proved pass-2 contamination on
428 vs 429 (Part B), shipped regression fixes C1–C3 (`fc177be`, pushed), started headless anchor
pair 430/431 from `D:\BO_CVn` (Part D **IN PROGRESS**). Docs updated (Part E).

---

## Part A — draft_429 validation

| # | Item | Result | Evidence |
|---|------|--------|----------|
| A1 | `pipeline_meta.json` provenance | **FAILED-AS-WRITTEN** → superseded by anchor pair | `git_dirty=true` on 429 (`695348b`); **provenance gap** — dirty file list not persisted in meta (Milan dev tree state unknown). **Hard gate:** 430/431 must have `git_dirty=false` or no snapshot cut. |
| A2 | Config snapshot diff 428 vs 429 | **PASS** (no user drift) | 5 keys differ — all map to **code-default / auto density overrides**, not `config.json` edits: `annulus_inner_fwhm` 5.75→4.75, `comp_max_delta_bprp` 0.64→0.79, `phase01_comparison_max_comp_rms` 0.08→0.1, `phase01_comparison_min_dist_arcsec` 90→60, `hrd_enrich_tap_timeout_s` null→20.0 (new persisted field). 428 values match `DENSITY_OVERRIDES` deltas in `config.py`; 429 matches current defaults at `695348b`. |
| A3 | `variability_candidates.csv` | **PASS** | 2 rows; 0 known VSX in candidates; both `vsx_known_variable=False`. `ms_vsx_true=87` (positional stamp bug — fixed in `fc177be`). **+3 CSV arithmetic:** `Gaia≤10″=242 → CSV=245` because 3 VSX rows without resolvable Gaia ID are appended to the export (245 total = 242 Gaia-matched + 3 no-ID appendix; 1 additional row has empty `catalog_id` in VT). |
| A4 | `excluded_targets.csv` | **PASS** | 78 rows: `out_of_frame=47`, `no_dao_gaia_match=30`, `no_catalog_id=1` — reconciles Phase-0 line `select_active_targets: … out_of_frame=47 excluded_no_dao_match=30` + 1 no-ID (`infolog_20260715_181704.txt` 15:24:47). |
| A5 | PDF overflow + HRD note | **PASS** | PDF: `VYVAR_report_NoFilter_60_2_20260715.pdf` emitted 16:17:04; Milan run overflow **0**. HRD page: `HRD enrichment skipped: Gaia TAP timed out after 20.0s` (`infolog` 16:09:32); `_hrd_cache/summary.json` records skip reason. |
| A6 | masterstars coords + v5 forensics | **PASS** | `coord_source`: gaia_catalog=2875, final_wcs=179. Matched sep vs Gaia DB: median 0.0″, p95 ~8e-11″, 0 rows >2″. v5 on 429: **`MISASSIGNED_164 n=0`** (`tmp/f429_coord_forensics_v5.txt`) — zero CATALOG_PROJECTION_OFF in the 164-cohort; priority T2 offsets ~1–7″ sep_wcs_gaia (residual peak-test on 6 stars, not census misassignment). |
| A7 | `[BORDER] Glob found 0 aligned` | **PASS (benign)** | Fires pre-alignment in 428 and 429 (`infolog_429` 15:10:28, 15:10:31) — stage-order artifact before detrended frames exist. **LOW ledger row:** BORDER-PREALIGN (silence or reorder). |
| A8 | BO CVn `lc_median_mag` 9.639→9.756 | **Quantified** | Same 4 comp IDs both drafts; **n_good_comp 4→3** via **ensemble sigma-clip** (not the relaxed `max_comp_rms` gate 0.08→0.10). Comp `1499053747922698240` (G≈10.79, faintest) clipped from active ensemble. **+0.118 mag** median ZP shift bounded by faintest-comp loss. |

### Infolog evidence (429, `infolog_20260715_181704.txt`)

| Check | Line (approx) | Value |
|-------|---------------|-------|
| UTC header | L2 | `# timestamps: UTC` |
| ePSF once | 15:11:18 | single skip notice |
| REPAIR summary | 15:10:26 | `kept_placeholder=179` |
| post_match identity | 15:10:16 | `ok=2805 warn=25 fail=0` |
| optimizer identity | 15:10:22 | `warn=4 fail=0` |
| WCS invertibility PASS | 15:10:19–25 | all Grip SIP refits |
| finalize coords | 15:10:26 | `gaia_catalog=2875 final_wcs=179` |
| VSX stamp (pre-fix) | 15:10:16 | `join=0 positional_fallback=87` |
| DAO census | 15:10:11 / 15:10:26 | pass1 raw=2816; pass2 +1180; JSON matched=2875 |

Script: `scripts/validate_429_wcsinv.py` → `tmp/validate_429_wcsinv_out.json` (run locally).

---

## Part B — Pass-2 contamination (428 vs 429)

| Metric | draft_428 | draft_429 |
|--------|-----------|-----------|
| `n_raw_dao` (pass-1) | **8927** | **2816** |
| DAO pass-2 additions | (not logged; implied ~6100+) | **1180** / 1285 targeted |
| masterstars rows | 6699 | 3054 |
| matched | 3974 | 2875 |
| unmatched | **2724** | **179** |
| matched only in 428 | — | **1172** catalog_ids |

**1172 disappeared rows:** median mag **15.14**; sep(ms sky → Gaia) median **17.0″**, p95 **80.5″**, **1142/1172** >2″ — faint/noise end, loose astrometry consistent with SIP-displaced pass-2 boxes on 428.

**Verdict: CONFIRMED.** 428's inflated census (8927 raw, 2724 unmatched, +1172 spurious matched) is dominated by pass-2 DAO on Gaia targets searched through corrupt forward SIP (~12″ bookkeeping offset per v5); 429 pass-2 (+1180) on healthy WCS yields 179 unmatched. Per-target science for well-detected stars unaffected (v5). **429 is the first healthy post-cleanup draft.**

DECISIONS row added: `F-428-PASS2-CONTAMINATION`.

Script: `scripts/pass2_contamination_428_429.py`.

---

## Part C — Regression fixes (`fc177be`, pushed)

| ID | Fix | Acceptance |
|----|-----|------------|
| **F-429-STAMP-WIRE** | Single VSX stamp **after** `finalize_masterstar_sky_coords` in `pipeline.py`; removed early call | Unit tests `test_stamp_post_finalize_after_optimizer_assigns_ids`, `test_stamp_before_optimizer_has_zero_id_join`. Live: expect `catalog_id join≈190–210` on 430. |
| **F-429-AC-SUMMARY** | `log_event` for `[AC] run summary:` in `photometry_core.py` | Infolog will show line after Phase 2A on next run. |
| **F-429-TAP-RETRY** | **Verified** on 429: `_hrd_cache/summary.json` `enrich_attempts=3`. Added INFO `log_event` per retry in `hrd_enrich.py` for future visibility. | |

**pytest:** 873 passed, 16 skipped (full suite on `fc177be`).

---

## Part D — Anchor pair (**RESTART** 2026-07-16)

**Disqualified attempt:** in-flight run on `draft_000430` killed. Mixed commits (`fc177be` during
run start, `a3536a0` landed mid-run); tree dirty during run 1. **Non-anchor** — ledger
`VL-ANCHOR-DQ-430`; draft retained on disk for reference only.

**Fresh pair:** `scripts/anchor_pair_run.py` from clean git worktree (`launch_anchor_pair_clean.py`).
Dynamic draft IDs + snapshot name `draft_{run1}_snapshot_wcsinv_{date}`. Gates: `git_dirty=false`,
`git_hash` match, `matched_world2pix_identity_*` in both metas, core SHA run1==run2 (incl. `err`).
`--finalize` cuts snapshot, adds `VL-ANCHOR-WCSINV`, re-enables `--full`.

Monitor: `tmp/anchor_pair_run.log`, `tmp/anchor_pair_run/anchor_pair_report.json`.

**Standing QA series (first two entries):** reported in `identity_qa_series` when pair completes.

---

## Part D — Anchor pair (original attempt — superseded)

**Rationale:** 429↔428 byte-identity void — WCS-INV legitimately changed detection census. RUNBOOK
two-fresh-runs rule → same-commit headless pair.

| Step | Status |
|------|--------|
| Post-C commit | **DONE** `fc177be` |
| **git_dirty hard gate** | **430 AND 431 must be clean** — else STOP, no snapshot |
| Headless run 430 | **RUNNING** |
| Headless run 431 | Pending |
| 430==431 core SHA gate (incl. `err`) | Pending |
| **matched_world2pix_identity p95/p99** | New standing QA — `pipeline_meta.json` + anchor report |
| Snapshot cut `draft_000430_snapshot_wcsinv_20260716` | Pending (requires clean provenance + SHA gate) |
| VL-ANCHOR ledger + `--full` re-enable | Pending |
| `session_baseline_check` SHA update | Pending |

**Allowed diffs vs 429 (sanity):** `vsx_known_variable` stamped count (~190–210 id join vs 87 positional); `pipeline_meta.provenance.git_hash`; HRD TAP cache contents; report PDF timestamp. **Not allowed:** matched census order-of-1000 shift, target count ≠167, science-column drift on shared targets.

Monitor: `tmp/anchor_pair_430_431_run.log`, `tmp/anchor_pair_430_431/anchor_pair_report.json`.

---

## Part E — Docs

Updated: `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_DECISIONS.md`, `CHANGELOG.md`, `docs/VYVAR_JOURNAL.md`.

**Open (non-blocking):** A-durable UI live test (save watched `.py` during alignment) — not confirmed exercised on 429; tag next routine UI run.

---

## Errors

None blocking. `git_dirty=true` on 429 provenance noted. Anchor pair completion pending long run.

---

## Files changed

- `pipeline.py`, `photometry_core.py`, `hrd_enrich.py`, `tests/test_f428_msstamp_coord.py`
- `scripts/validate_429_wcsinv.py`, `scripts/pass2_contamination_428_429.py`, `scripts/anchor_pair_430_431.py`
- Docs (STATE, ROADMAP, DECISIONS, JOURNAL, CHANGELOG)
- **Commit:** `fc177be` (pushed `origin/main`)
