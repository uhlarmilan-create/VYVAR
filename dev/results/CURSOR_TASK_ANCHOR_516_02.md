# CURSOR TASK - ANCHOR-516-02 (architect copy)

Date issued: 2026-08-18
Status: **closed as measurement** (see CURSOR_RESULT_ANCHOR_516_02.md)
Push: only on Milan's authorization. No re-cut in this task.

# ANCHOR-516-02: Isolate the 515-vs-516 ERR drift; then decide anchor

Premise check: 516 (UI run, d5ef039, PFS OFF via config default) vs
515 (harness run, PFS ON override, product de6f7c8). MAG identical
48/48; ERR differs 46/48 up to 6.6 mmag; comp_qa sidecar count 96 vs
48; comparison_stars_per_target.csv differs; CV CVn skip_reason label
differs (zone_flag vs per_frame_saturation). Two variables are
confounded: PFS setting and git tip (515 Phase 2A meta says 6b23633;
516 ran d5ef039; code commit between them: 057ecdc PT pin). This task
separates them by measurement. No anchor re-cut until the drift has a
named mechanism.

## Part A - ERR drift decomposition (measurement, no code change)
1. Pick 3 LCs with max ERR delta + BO CVn. Decompose err_total into
   err_photon / sem_rel / sigma_sys_rel per epoch for 515 and 516.
   Name which term carries the 6.6 mmag. Report per-term max delta.
2. Diff comparison_stars_per_target.csv 515 vs 516: which columns
   differ (membership? weights? metadata/skip_reason only?). If
   weights differ while MAG is byte-identical, explain how from the
   code path (read, not infer).
3. Explain comp_qa 96 vs 48: what writes these sidecars and what
   keys the count (read the writer code).

## Part B - Controlled re-run (one variable)
4. Re-run 516 Phase 2A at d5ef039 with PFS ON via the same harness
   override used for 515. Everything else untouched.
5. Product SHA vs de6f7c8. Three outcomes:
   a) SHA == de6f7c8: identity proven; PFS was the whole story.
      Report why PFS moved ERR (mechanism, from code+data).
   b) MAG identical, ERR still differs: PFS is not the cause; the
      remaining variable is the tip/state. Bisect: is 057ecdc
      implicated? Read what actually differs in the err path.
   c) MAG differs: STOP, full report, architect review.
6. Runtime per part (Rule 0.3).

## Part C - Anchor decision input (no re-cut in this task)
7. One-page verdict: named mechanism for every observed delta
   (ERR, comp_qa count, skip_reason label). State whether UI-default
   and harness-override runs are expected to produce identical
   products on one tip, and if not, which run mode the anchor should
   canonically use. Architect + Milan decide the re-cut in
   ANCHOR-516-03.

## Constraints
- Do not delete anything. 515 remains product reference (de6f7c8)
  until this closes.
- Raw numbers committed under dev/results/context/session_20260818*/
  (Rule 0.2). Push only on Milan's authorization.

## Docs impact
None yet (measurement task). STATE/ROADMAP move in 516-03.
