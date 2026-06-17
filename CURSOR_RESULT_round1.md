CURSOR RESULT — Round 1: four known fixes + pre-flip demo — 2026-06-17

## What I did

Implemented and verified the four Round-1 fixes (A-durable, B-cap, completeness gate, log
flood) audit-first on `draft_000413` (Boyden V454 CrA, g_60_2), reusing the existing aligned
frames (no re-alignment). Ran a no-code pre-flip demo. Epoch-quality and aperture-skirt remain
out of scope (Round 2).

**Headline science finding (needs to stay on the record): Fix 2 is NOT byte-identical on the
original targets.** Expanding the in-frame VSX target list (26 ? ~100) also expands the *global
comparison-pool veto* (the same `variable_targets` list drives `build_global_comp_pool`), so 6
of 19 original targets shift by ?0.12 mag. This is a deterministic comp-purity *improvement*
(newly-recognised variables are correctly removed from the comparison ensemble), but it changes
existing magnitudes — Milan reviewed and **accepted the coupling** (keep B as implemented, report
as a comp-purity improvement).

---

## Fix 1 — A-durable (MP reload robustness; no numeric effect)

**Audit (file:line):**
- Module-level binding source: `import vyvar_alignment_frame` (`pipeline.py:31`); `pickle`
  (`pipeline.py:8`); `ProcessPoolExecutor` (`pipeline.py:17`).
- Pool dispatch: `pipeline.py:12925-12931`. Single-process loop: `pipeline.py:12914-12917`,
  `12948-12951`.

**Fix:**
- Resolve the MP funcs by **fresh module attribute at call time**
  (`vyvar_alignment_frame._astrometry_align_mp_init` / `._astrometry_align_mp_task`,
  `pipeline.py:12923-12924`) so the object handed to the spawn pool is exactly what
  `sys.modules` resolves — even if Streamlit's watcher reloaded `vyvar_alignment_frame` after
  `pipeline.py` was imported.
- Wrap `pool.map(...)` to catch `pickle.PicklingError` and fall back to the existing
  single-process loop with a UI log line (`pipeline.py:12937-12947`). No partial per-frame state
  exists at submission time (PicklingError is raised before any flush), so the fallback is clean.

**Acceptance — PASS** (`tmp/fix1_fallback_sim.py`):
```
[A]  fresh module-attribute lookup current after reload: True (stale from-import binding: False)
[A2] pickling stale from-import binding raises PicklingError: True
     "Can't pickle <function _astrometry_align_mp_task ...>: it's not the same object as
      vyvar_alignment_frame._astrometry_align_mp_task"   (? the exact production root cause)
[A2] pickling fresh module-attribute succeeds: True
[B]  fallback reached on PicklingError: True; fallback result correct: True
RESULT: reload-robustness=True  authentic-root-cause=True  fallback=True => PASS
```
No new config param.

---

## Fix 2 — B-cap (spatial-first VSX target selection; science-changing)

**Audit (file:line):**
- Cap source: `query_local_vsx` caps at `catalog_query_max_rows=15000` with **no `ORDER BY`**
  (`database.py`, `config.py`).
- Cone query: `_query_vsx_local` (`pipeline.py:~4442`). variable_targets export builder:
  `write_photometry_plan_files` (`pipeline.py:5531`), VSX export query at `pipeline.py:5765-5791`.
- **Consumers of the cone/VSX result enumerated before changing:**
  1. variable_targets export (the one fixed) — `write_photometry_plan_files`.
  2. comparison-pool **veto** — `_vsx_for_veto` from the *unconditional* cone query
     (`pipeline.py:5620`) ? `select_comparison_stars_spatial_grid(...)` (`pipeline.py:5713-5721`)
     — left unchanged at the plan level.
  3. phase-1 comp exclusion — `variable_targets.csv` ? `_vt_cid_exclude` ?
     `build_global_comp_pool(..., variable_target_catalog_ids=...)` (`photometry_core.py:12415-12444`).
  4. stub auto-restore — `ensure_full_variable_targets_if_presel_stub` ?
     `write_photometry_plan_files` (`photometry_core.py:3304-3345`).

**Fix:** new `_query_vsx_local_frame_bbox` (`pipeline.py:4512`) queries local VSX within the frame
bbox (+50 px margin, matching the in-frame pixel filter) with `max_rows=None`. The bbox is
sub-degree ? tiny SQL result ? the global cap never truncates ? spatial-first completeness
independent of row order. Wired at `pipeline.py:5784-5791`. RA wrap handled via centre-relative
offsets. No new param.

**Impact / acceptance:**
- In-frame VSX **26 ? 100** export rows (92 active after in-frame/chip filter ? 67 photometry
  summary). Band fills.
- **V0454 CrA (m9.9), KQ CrA (m12.2), KM CrA (m12.2), KT CrA (m11.8) now appear** in
  `variable_targets.csv` — all at Dec ? ?39.5 (the northern slice the cap dropped); all 4 were
  **absent** in the capped 26-row set.
- Comparison-star plan output (`write_photometry_plan_files`) is **byte-identical** old-vs-new
  (`comparison_stars.csv` SHA `89b642ea1766…`, 143 rows), because the plan-level comp veto uses
  the unconditional cone query — `tmp/fix2_isolated_bcap.py`.
- **BUT originals are NOT byte-identical end-to-end.** Clean OLD-vs-NEW control
  (`tmp/fix2_e2e_oldnew.py`, both full re-runs, identical except the target-list width):
  OLD 19 summary, NEW 67 summary (+48 added); **6 of 19 common targets shift** (max |?mag|=0.122,
  |?rms|=0.253, |?rms_ooe|=0.464). Mechanism (`tmp/fix2_mechanism2/3.py`): `variable_targets`
  drives the **global** comp veto; the now-complete list purges newly-recognised variables from
  the ensemble ? (a) some targets lose a variable-as-comp directly, and (b) surviving comps'
  field-wide RMS drops (variables had inflated ensemble scatter) ? comp weights re-rank ? magnitude
  shifts. Deterministic; scientifically correct. **Milan-accepted.**

*Note:* an earlier "old-code control" was **invalid** — `ensure_full_..._stub` writes the restored
list back to `vt_path.parent`, which clobbered the 26-row backup to 100 rows, so the toggle never
fired. The clean control gives each run its own dir.

field_map (uncapped g, band filled): `tmp/round1_field_map_g_uncapped.png`.

---

## Fix 3 — Completeness gate (verdict logic; truncation guard preserved)

**Audit (file:line):** `audit_photometry_completeness` (`night_run.py:385`); verdict consumed at
`night_run.py:1058-1079`, `1103-1107`.

**Fix:** the verdict is taken against **measurable** targets. A missing target (active, no summary
row) counts as *unmeasurable* (honest, must NOT fail) when it is fainter than the achieved depth
(faintest measured target's catalog mag). Missing-but-detectable (? depth) counts as a *measurable
miss* and still fails (silent-truncation / false-success guard intact). Depth is derived from the
data ? **no new param**. Conservative fallbacks (no mag / nothing measured) treat misses as
measurable so truncation can never masquerade as honest.

**Acceptance — PASS:**
- Unit tests `tests/test_completeness_gate_measurable.py`: 4/4 (honest-unmeasurable passes;
  truncation fails; nothing-measured fails; bright-missing fails).
- Real data: the exact task case — g **19/22 = 86.4%** (would FAIL old 90% gate) — now **PASS**
  (`measurable_ratio=1.000`; 3 missing are below depth 13.82). Uncapped run 67/71 also PASS
  (`measurable_ratio=1.000`; 4 below depth 14.97).

---

## Fix 4 — NoDetections log flood (hygiene; no numeric effect)

**Audit (file:line):** `_dao_targeted_pass2_unmatched_gaia` per-cutout `DAOStarFinder`
(`pipeline.py:6438-6450`) — fires one photutils `NoDetectionsWarning` per empty cutout
(thousands per defocused/sparse frame); used by both per-frame and master DAO passes.

**Fix:** suppress `NoDetectionsWarning` inside the targeted-cutout loop, count misses
(`n_empty_cutouts`), and emit a single summary line per stage
(`pipeline.py:6420-6480`): `"[DAO pass 2] N/M targeted cutouts had no detection (NoDetectionsWarning
suppressed)"`. No change to detection behaviour. No new param.

---

## Verify (g-only, reused draft_413 aligned frames)

| Fix | Acceptance | Result |
|-----|-----------|--------|
| A | photometry byte-identical; fallback reached | sim PASS; A touches no numbers |
| B | 26?~100 in-frame; named variables appear; originals byte-identical | 26?100 ?, V0454/KQ/KM/KT ?; originals **shift 6/19 ?0.12 mag** (comp-veto coupling, Milan-accepted) |
| 3 | honest run passes; truncated still fails | 86.4% case now PASS; truncation FAILS (tests) |
| 4 | log readable; warning summarized | summary line implemented |

## Demo (no code) — pre-flip-only LCs + trust on the uncapped g set

`tmp/demo_preflip.py` (+ `tmp/demo_preflip_stats.py`) re-ran photometry on the 59 pre-flip frames
(`proc_V454CrA_*`; post-flip are `proc_V454CrAR_*`) and computed trust with existing functions.

Named bright variables (pre-flip only):
```
name        trust  check_mmag  lc_rms_mmag  n_pts  n_clean  driver
V0454 CrA   RED       47.4       122.7       59     2       noisy_moon LC + sparse comps (122 mmag ~ real variability, m9.9)
KQ CrA      RED        n/a        41.5       59     5       no check-star verification
KM CrA      RED       96.1        40.1       59     7       check-star scatter high
KT CrA      RED        n/a        53.2       59     3       no check-star verification
```
Field-wide pre-flip (68 targets): **best lc_rms 13.6 mmag, p25 40.5, median 69**; check-scatter
n=40 median 84, p25 80, **min 28 mmag**. RED drivers: check_scatter_high=38, no_check_star=28,
noisy_moon=3, no_clean_comps=1. **All 68 RED.**

**Reading:** the *photometric floor* is tens of mmag pre-flip (best 14-40 mmag; V0454 check
47 mmag), confirming "defocus helps bright" and that the all-RED night verdict is dominated by the
post-flip collapse — but bright targets do **not** reach GREEN pre-flip either. Pre-flip RED is
driven by check-star verification gaps for bright stars, thin/sparse comp ensembles for the
brightest, and lunar background (73% moon, 21° sep). These are distinct from the post-flip collapse
and sharpen the Round-2 scope (epoch-quality + bright-star check-star/comp handling).

## Output / findings (key paths)

- `pipeline.py` (Fix 1 dispatch+fallback; Fix 2 `_query_vsx_local_frame_bbox`+wiring; Fix 4 suppression)
- `night_run.py` (Fix 3 gate)
- `tests/test_completeness_gate_measurable.py` (Fix 3 unit tests, 4/4)
- `tmp/round1_field_map_g_uncapped.png` (band-filled g field map)
- throwaway: `tmp/fix1_fallback_sim.py`, `tmp/fix2_isolated_bcap.py`, `tmp/fix2_e2e_oldnew.py`,
  `tmp/fix2_mechanism2.py`, `tmp/fix2_mechanism3.py`, `tmp/demo_preflip.py`, `tmp/demo_preflip_stats.py`

## Errors

None outstanding. (One invalid control — clobbered 26-row backup — diagnosed and replaced with a
clean per-dir control.)

## Files changed

- `pipeline.py` — Fix 1 (A-durable), Fix 2 (B-cap), Fix 4 (log flood)
- `night_run.py` — Fix 3 (completeness gate)
- `tests/test_completeness_gate_measurable.py` — Fix 3 unit tests (new)
- docs: `VYVAR_JOURNAL.md`, `VYVAR_DECISIONS.md`, `VYVAR_STATE.md`, `VYVAR_ROADMAP.md`

Commits (separate, attributable): A infra / B targets / gate / log — created after this writeup.
Push pending (Milan approved B + gate science).
