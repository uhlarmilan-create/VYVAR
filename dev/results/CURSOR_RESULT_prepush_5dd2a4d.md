CURSOR RESULT - 2026-08-12

Pre-push verification for commit `5dd2a4d`. Do not push. Milan authorizes.

---

## A. `--fast` A/B - Prediction 1 FAILED

Working docs stashed; clean `682f40c` vs restored `5dd2a4d` (main). Uncommitted literature docs were stashed and not part of either run.

### A.1 Clean `682f40c` - `session_baseline_check.py --fast` (raw)

```
SESSION BASELINE CHECK (fast)
------------------------------------------------------------------------
Check                        Status Detail
------------------------------------------------------------------------
git-branch                   PASS   HEAD
git-head                     PASS   682f40c
git-staged                   PASS   none
git-untracked-known          WARN   14 known untracked
git-untracked                WARN   CURSOR_TASK.md; dev/results/MEMO_ensemble_zp_clip_literature.md; dev/tests/_tmp_
config-paths                 PASS   all present
pytest                       FAIL   1290 passed, 27 skipped
manifest-db-parity           PASS   draft_id=435
ledger                       PASS   v1 15 items
ledger-todo                  WARN   VL-ANCHOR-424, VL-ANCHOR-DQ-430
deps-outdated                WARN   numpy 2.4.4->2.5.2 (+94 other) - gated upgrade, see docs/DEPS_POLICY.md
------------------------------------------------------------------------
OVERALL: FAIL
```

Pytest short summary (same tree; baseline truncates failure names):

```
FAILED dev/tests/test_robust_frame_fwhm.py::test_robust_fwhm_on_frame62_not_cr_scale
FAILED dev/tests/test_robust_frame_fwhm.py::test_qc_fwhm_elongation_not_segmentation_cr
2 failed, 1290 passed, 27 skipped, 49 warnings in 396.70s
```

### A.2 With change `5dd2a4d` - `--fast` (raw)

```
SESSION BASELINE CHECK (fast)
------------------------------------------------------------------------
Check                        Status Detail
------------------------------------------------------------------------
git-branch                   PASS   main
git-head                     PASS   5dd2a4d
git-staged                   PASS   none
git-untracked-known          WARN   14 known untracked
git-untracked                WARN   CURSOR_TASK.md; dev/results/MEMO_ensemble_zp_clip_literature.md; dev/tests/_tmp_
git-origin-main              WARN   differs from origin/main (682f40c); consider git pull
config-paths                 PASS   all present
pytest                       FAIL   1290 passed, 27 skipped
manifest-db-parity           PASS   draft_id=435
ledger                       PASS   v1 15 items
ledger-todo                  WARN   VL-ANCHOR-424, VL-ANCHOR-DQ-430
deps-outdated                WARN   numpy 2.4.4->2.5.2 (+94 other) - gated upgrade, see docs/DEPS_POLICY.md
------------------------------------------------------------------------
OVERALL: FAIL
```

Pytest short summary:

```
FAILED dev/tests/test_ascii_policy.py::test_tracked_text_files_are_ascii
FAILED dev/tests/test_field_map_no_catalog_only.py::test_target_field_map_png_no_catalog_only_legend
FAILED dev/tests/test_robust_frame_fwhm.py::test_robust_fwhm_on_frame62_not_cr_scale
FAILED dev/tests/test_robust_frame_fwhm.py::test_qc_fwhm_elongation_not_segmentation_cr
4 failed, 1290 passed, 27 skipped, 49 warnings in 352.82s
```

### A.3 Comparison

| test | fails on clean HEAD | fails with change |
|---|---|---|
| `test_robust_fwhm_on_frame62_not_cr_scale` | yes | yes |
| `test_qc_fwhm_elongation_not_segmentation_cr` | yes | yes |
| `test_tracked_text_files_are_ascii` | no | yes |
| `test_target_field_map_png_no_catalog_only_legend` | no | yes |

**Prediction 1 = FAILED.** Two tests pass on clean `682f40c` and fail on `5dd2a4d`. Per task: stop; do not diagnose; A.4 not run.

---

## B. Draft 435 archive integrity

**Reference:** `C:\ASTRO\backups\draft_000435_anchor_live_20260716.zip` (whole tree as of 2026-07-16; 1932 members). Sibling: `Archive/Drafts/draft_000435_snapshot_skysurface_20260716`. Immediate pre-rerun backup covers **only** BO CVn LC: `tmp/_435_lc_before_zpclip_rm.csv`. Manifest has no per-file checksums.

| Class vs July live zip | Count |
|---|---|
| Hash match | 1381 |
| Hash changed | 551 (549 with mtime 2026-08-12) |
| Current-only (no zip ref) | 846 |

BO CVn LC restored and hash-identical to backup + July zip. Calibrated/FITS in zip set: 0 hash diffs. **Not restored:** ~549 photometry outputs (39 other LCs, trusts, comps, AAVSO, summaries, `pipeline_meta`, BO CVn sidecars).

**Verdict: partially restored.** Unknown: 846 post-July-only files; no minutes-before full-tree checksum.

Evidence: `tmp/_audit_435_integrity.json`, `dev/results/CURSOR_RESULT_zp_clip_removal.md`.

---

## C. INV-DAG-01

### C.1 Mechanism

`stamp_pipeline_stage` fails when new stage `STAGE_ORDER` index is less than max already stamped (`idx < max_seq`).

- Order: `invariants_runtime.py:22-32`
- Check: `invariants_runtime.py:494-507`
- Fail site after LC write: `photometry_core.py:10648` (`phase2a` stamp with `postprocess` already at seq=7)

### C.2 Normal E2E vs this sequence

**Only re-entry** on a draft that already has later stages stamped. Cold/forward E2E stamps in order. Run evidence: first 509 photometry re-run against completed meta (`CURSOR_RESULT_zp_clip_removal.md`; terminal `301354.txt`).

### C.3 Real requirement vs mis-specified

Forward order is intentional (registry + `test_invariants_p2.py`). The failure here was **bookkeeping against historical `postprocess`**, not science running out of order. Stamp happens **after** work. Re-run aperture photometry on a completed draft is a supported path that this frontier check blocks. **Verdict:** real forward intent; **mis-specified for photometry re-stamp** on completed drafts.

### C.4 LC numbers vs trim

Reported 509 metrics (check_scatter 0.00863, n=134, etc.) measured from LCs written **before** stage trim. Trim only edits `pipeline_meta.json` `stages` - does not change LC CSVs. A later successful re-run can rewrite LCs; the quoted prediction numbers are from the first write.

---

## D. Saturation on this field

**Method (stated before result):** On each LC-matched raw FITS, WCS?xy, apply per-frame offset from BO CVn local peak, then mag-guided local peak (half?22) to avoid brighter neighbors. Compare to live sat limit and 85% linearity/admission fraction. Artifact: `tmp/_prepush_sat_d_v3.json`.

### D.2 Thresholds

| Threshold | Value | Source |
|---|---|---|
| Saturation ceiling | **65535** ADU | FITS BITPIX=16 + BZERO=32768 via `_effective_saturation_limit` ? `"bitpix"` (`pipeline.py:5245-5264`, `5285-5324`). No `SATURATE`/`MAXLIN` in raw header. |
| Proc-recorded limit (509) | **65138.47** | `proc_*.csv` `saturate_limit_adu` |
| Proc-recorded limit (435) | **65535** | same column |
| 85% guard | **55704.75** (=0.85-65535) | used for `likely_saturated` / COG / admission (`pipeline.py:5597`, `photometry_core.py:12035-12039`, `comp_selection_per_target.py:798+`) |
| Equipment DB | C3-26000 `SATURATE_ADU=65535` exists in `EQUIPMENTS`; draft?equipment link not in this DB | `database.py` / EQUIPMENTS |

LC-frame flag uses `peak > sat_limit` (`photometry_core.py:2417-2460`). `likely_saturated` uses `peak >= sat_frac * limit` (default 0.85).

### D.1 / D.3 / D.4 - raw peak ADU (min / median / max); frames over threshold

**Draft 509** (134 LC frames; 5 comps + check `1497313255374892800`):

| star | role | min | median | max | >65535 | ?65535 | >85% |
|---|---|---:|---:|---:|---:|---:|---:|
| 1498613634033133184 | target | 12352 | 17492 | 24168 | 0 | 0 | 0 |
| 1497771992240531712 | comp | 13572 | 18544 | 24124 | 0 | 0 | 0 |
| 1499200223486564608 | comp | 6396 | 15406 | 21008 | 0 | 0 | 0 |
| 1499053747922698240 | comp | 1740 | 6746 | 7916 | 0 | 0 | 0 |
| 1497974027502858240 | comp | 1628 | 5584 | 7724 | 0 | 0 | 0 |
| 1497368849430107904 | comp | 1584 | 1944 | 6892 | 0 | 0 | 0 |
| 1497313255374892800 | check | 4760 | 10618 | 13800 | 0 | 0 | 0 |

**Draft 435** (139 LC frames in current archive; 4 comps + check `1497442379271632384`):

| star | role | min | median | max | >65535 | ?65535 | >85% |
|---|---|---:|---:|---:|---:|---:|---:|
| 1498613634033133184 | target | 12352 | 17532 | 24168 | 0 | 0 | 0 |
| 1497771992240531712 | comp | 13572 | 18600 | 24124 | 0 | 0 | 0 |
| 1499200223486564608 | comp | 6396 | 15400 | 21008 | 0 | 0 | 0 |
| 1499053747922698240 | comp | 1740 | 6748 | 8620 | 0 | 0 | 0 |
| 1497974027502858240 | comp | 1720 | 5624 | 8204 | 0 | 0 | 0 |
| 1497442379271632384 | check | 10392 | 28300 | 41576 | 0 | 0 | 0 |

**Definite answer:** target, selected comps, and check stars are **not** saturated on raw (all peaks ? 85% of 65535). Field-wide bright cores near 65535/68567 exist but are **other** stars, not this photometry set.

### D.5 What the pipeline does if a star crosses (code destinations)

| Destination | Behavior | file:line |
|---|---|---|
| Per-frame LC `flag` | `saturated` if `peak_max_adu > sat_limit` | `photometry_core.py:2417-2460` |
| LC export column | `flag` preserved through postprocess / outlier path | `photometry_core.py:4558-4559`, `9761-9800`, CSV write `5072`/`5105` |
| Summary | `n_saturated` count | `photometry_core.py:10173`, `10209` |
| Comp pool / weighting | exclude `is_saturated` / `likely_saturated` / zone saturated | `photometry_core.py:13671-13680`; `comp_selection_per_target.py:383-386`, `798-822` (admission vs 85% col) |
| Aperture / COG reference set | unsaturated required (`peak > sat_frac * limit` rejected) | `photometry_core.py:12035-12039` |
| AAVSO/VarAstro export | drops rows with `flag` in `{saturated, no_data, ...}` | `export_reports.py:885-890` |

On these drafts: BO CVn LC `flag` is `normal` for all frames; comps marked `is_saturated=False` / `zone=linear` in selection CSVs. Flags did not fire for this set.

### D.6 Measurable effect

**Design (stated first):** If any of these stars exceeded 85%/full limit, compare check-star scatter and target LC std with vs without that star in the ensemble; also count saturated LC flags. If none exceed, effect is null by construction.

**Result:** No star in the set crosses either threshold on any frame ? **no measurable saturation-driven effect** on 509 or 435 BO CVn light curves. (Leave-one-out of the brightest selected comp changes check std by ?0.3 mmag, unrelated to saturation flags.)

---

## PUSH READINESS

- A: **NO** - Prediction 1 FAILED (`ascii_policy`, `field_map` pass clean, fail on `5dd2a4d`).
- B: **NO** - draft 435 only partially restored; not a verified full-tree anchor.
- C: **YES for commit content** - INV-DAG-01 is re-entry stamp friction; reported 509 LC numbers predate trim and are not invalidated by trim alone.
- D: **YES** - target/comps/check are not saturated on raw; white-cores -may be- is answered no for this set.

**Does not clear push:** A (new `--fast` failures on the change); B (435 archive integrity).
