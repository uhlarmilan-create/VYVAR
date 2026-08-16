CURSOR RESULT - 2026-08-16 D515-ACCEPT-01

What I did
Full-field acceptance measurement of draft 515 headless RUN (launch SHA
da9cce4). Parts A-F are measurement only; Part G adds PUSH-STAMP-01.
No science-code changes. Push not authorized.

## Output / findings

### Part A - Completion and integrity

A1. RUN completed. Exit status 0. Ended after Phase 2A (Comp QA + Trust +
`Faza 2A hotovo`). ELAPSED_S 4244.0. Log:
`tmp/draft_515_headless_phase012a.log` (UTF-16 LE from PowerShell Tee-Object).
No crash; Parts B-F executed.

A2. Draft-level counts (quantity labels):
- frames ingested (aligned FITS under setup): 135
- frames photometered (proc_*.csv): 134
- Phase 1 photometry targets completed / total: 97 / 97
- LC files written: 49
- qa_degraded targets: 0 (none)

97 -> 49 gap: fully accounted; silent_missing_n = 0 (no INV-NO-SILENT).
Missing-LC reason histogram:
- zone_noise: 45
- below_target_depth: 3

A3. `session_baseline_check.py --fast` on tip dea6a54 (pre-accept commits):
OVERALL PASS; pytest 1423 passed, 27 skipped; git-head dea6a54.
(Closes the unrecorded-PASS gap after 4fe84b4 through tools/docs tips.)

### Part B - Runtime decomposition (measurement only)

B1. Wall times from log timestamps (run SHA da9cce4):
- Phase 0 wall: 0.3 s
- Phase 1 wall: 3905.9 s
- Phase 2A wall: 331.9 s (includes Comp QA)
- Comp QA alone (Comp QA -> Trust): 211.1 s
- Total ELAPSED_S: 4244.0 s

B2. Per-target / checkpoint walls (log is sparse; see caveats):
Phase 1 (n_intervals=14 sparse checkpoints, every ~8 of 97):
min 33.9 / median 273.4 / p90 428.4 / max 462.7 s.
Five slowest checkpoints: ASASSN-V J140843.29+402701.8 (16), RX CVn (24),
FY CVn (8), R CVn (1), ZTF J140300.57+413339.3 (32).

Phase 2A ciel lines only (n_intervals=14 sparse of 218):
min 0.0 / median 4.8 / p90 23.4 / max 25.2 s.
Five slowest: R CVn, FW CVn, CSS_J135929.8+421520, Gaia 1497070744341492864,
Gaia 1500705484968035968.

Caveat: Phase 1 and Phase 2A status lines are not every target; B2 numbers are
inter-checkpoint deltas, not true per-target walls. Instrumentation follow-up
if Milan needs true per-target timing.

B3. Dominance (one sentence each):
- Phase 0: brief VSX/active-target selection; not resolvable further in the log.
- Phase 1: dominated by per-target comparison-star selection /
  `_accumulate_per_frame_comp_metrics` (long gaps between status lines).
- Phase 2A: aperture photometry of ciel loop is short; Comp QA dominates late
  Phase 2A wall (~211 s).

No threshold, no verdict (pre-registered).

### Part C - EMPTY-DAO-01 rate

C1. Empty-DAO / forced-only frames: 0 / 134 = rate 0.0.
Setup: NoFilter_60_2 only (single rig).

C2. Pre-registered: rate = 0 -> EMPTY-DAO-01 stays OPEN (515 did not exercise
it). No code change.

### Part D - Fixed-meter acceptance (full field)

Science identity: `git diff --stat 4fe84b4..da9cce4 -- src_py/` is empty.
D interpretations are unconditional on science code.

D1. Check-LC MAD (1.4826*MAD*1000 mmag; not comp_rms, not LOO), run da9cce4:
- BO CVn check MAD: 6.71 mmag (subset IMPL-05 C: 8.59 mmag)
- FW CVn check MAD: 8.20 mmag (subset IMPL-05 C: 9.82 mmag)
Selected check catalog_id on 515 for both: 1497613731286514432
(named meter 1497145751650265600 is NOT the selected check).

D2. Ensemble membership:
- BO: n_comp=5; set identical to IMPL-05 C subset (set_equal_impl05c=true).
- FW: n_comp=8; set NOT identical to IMPL-05 C subset
  (set_equal_impl05c=false). Different draft/pool; reported here.

D3. Distribution over LC targets with check MAD (n=49 of 97; only targets
with written LCs have the meter):
min 6.25 / median 8.20 / p90 10.73 / max 11.83 mmag.
Targets >3x median: none.

D4. Pre-registered interpretation:
Neither BO nor FW degraded vs subset; both improved (~1.9 and ~1.6 mmag).
Degrade gate (>2 mmag worse) does not fire. Policy transfers; acceptance holds.
Spec wording "within ~1 mmag of subset" is approximate (deltas are
improvements, not matches).

Spec defect (named, not implemented wrong): task names check
1497145751650265600; production and IMPL-05 C used different check IDs.
D1 uses production `check_kmag_*.csv` MAD. D3 "all 97" cannot all have
check MAD without LCs; reported on 49 LCs.

### Part E - Aperture-table follow-ons

Estimator: PRE-IMPL Q2-style equal-weight peer mean of instrumental mags from
flux (-2.5 log10 flux); not flux-sum loo_diff_series. Sampled from
masterstars_full_match G bins (selected comps are bright-biased).

E1. BIN-8-9-REGRESSION-01:
- subset before r=9.5: 7.79 mmag (n=4)
- subset after per-mag: 12.35 mmag (n=4)
- full 515 median LOO G8-9: 11.99 mmag (n=15)
Pre-registered: persists at larger n -> item is REAL. Follow-up: re-examine
bright rows of aperture_scatter_table.json (not this task).

E2. FAINT-14-15-CONTAM:
- subset after: 172.78 mmag (n=8)
- full 515 median LOO G14-15: 40.98 mmag (n=11)
Ratio ~0.24 (< half). Pre-registered: large improvement changes status from
WATCH. Closing/reclassification is a follow-up decision; numbers recorded.

### Part F - COMP-RMS-DEF-01 dataset (extraction only)

F1-F2. Written: `dev/results/COMP_RMS_DEF_01_dataset.json` (698 rows).
Header block names columns and units. LOO is PRE-IMPL Q2-style from flux-derived
inst mag.

F3. Scatter summary (no interpretation beyond ordering note):
- correlation(comp_rms_mmag, loo_mmag): 0.919
- median ratio comp_rms/LOO overall: 1.002
- by G bin median ratio: 8-9: 0.79; 9-10: 1.04; 10-11: 1.02; 11-12: 1.16;
  12-13: 0.89; 13-14: 0.85; 14-15: n=0 among selected comps
Ratio roughly O(1) across bins -> RMS-first ORDERING likely preserved under a
monotone remapping; absolute thresholds still need a unified definition
(separate fix task).

### Part G - Process rule

Added PUSH-STAMP-01 to `docs/VYVAR_DECISIONS.md` (2026-08-16): content tip vs
origin SHA; ends stamp-on-stamp chains.

## Machine-readable numbers

`dev/results/D515_ACCEPT_01_numbers.json` (Parts A-F; run_sha da9cce4).
`dev/results/COMP_RMS_DEF_01_dataset.json` (Part F rows + summary).

## Spec defects / corrections proposed (not implemented)

1. Named check star 1497145751650265600 does not match production / IMPL-05 C
   meters. Correct step: accept production check_kmag MAD and report selected
   check id (done).
2. D3 "all 97 targets" check-MAD: only LC-bearing targets have the meter.
   Correct step: distribute over written LCs and state the n (done: n=49).
3. Part E/F must not use proc CSV column `mag` (catalog G, nearly constant).
   Correct step: instrumental mag from flux (done in measure tool).
4. Log UTF-16 from Tee-Object; parsers must decode UTF-16 when BOM present.
5. Sparse Phase 1/2A status lines cannot yield true per-target walls; B2 is
   checkpoint deltas. Instrumentation follow-up if needed.

## Errors (if any)

None blocking. --fast WARN: untracked accept artifacts (expected before commit);
origin/main differs (local ahead; push not authorized).

## Files changed

- docs/VYVAR_DECISIONS.md (PUSH-STAMP-01)
- dev/tools/d515_accept_01_measure.py
- dev/results/D515_ACCEPT_01_numbers.json
- dev/results/COMP_RMS_DEF_01_dataset.json
- dev/results/CURSOR_RESULT_D515_ACCEPT_01.md

Commit hashes: filled after local commits below.
