# CURSOR RESULT - PARAMS-REGISTRY-CLOSEOUT

Date: 2026-07-17
Arc: UI block, wave 1 (parameters) - closeout
Baseline: origin/main `e782786`; local HEAD `4b6c012` (6 unpushed commits)

## What I did

Review verdict on PARAMS-REGISTRY-UI was PASS. Completed the three closeout
follow-ups (result commit, hidden-tier cleanup, min_comps provenance) with a
separate commit per step and full pytest green. Push is GATED - awaiting
Milan's explicit confirmation.

## STEP 1 - commit the result file

Commit `3d08213` `docs(result): PARAMS-REGISTRY-UI closeout`

```
 CURSOR_RESULT_params_registry_ui.md | 229 ++++++++++++++++++++++++++++++++++++
 1 file changed, 229 insertions(+)
```

## STEP 2 - registry cosmetic rule: hidden implies expert

Commit `4b6c012` `PARAMS-REGISTRY-CLOSEOUT STEP 2: hidden implies expert tier`

```
 docs/VYVAR_PARAMS.md            | 16 ++++++++--------
 tests/test_params_registry.py   | 11 +++++++++++
 validation/params_registry.json | 12 ++++++------
 3 files changed, 25 insertions(+), 14 deletions(-)
```

- 6 `widget=hidden` entries moved `tier: advanced -> expert` (all `phase=paths`):
  `blind_index_fine_path`, `blind_index_path`, `blind_index_wide_path`,
  `exoplanet_local_db_path`, `gaia_db_path`, `vsx_local_db_path`.
  (The other 8 hidden keys - `archive_root`, `blind_index_select_mode`,
  `calibration_library_root`, `database_path`, `project_root`, etc. - were
  already `expert`, so no change.)
- New guard test `test_hidden_widget_implies_expert_tier`: `widget == hidden`
  implies `tier == expert`; failure names offending keys.
- `docs/VYVAR_PARAMS.md` regenerated (summary now: basic 12, advanced 75,
  expert 217; kind/widget unchanged). Freshness test stays green.

No other tier/label changes. Milan's tiering review can adjust entries later
via a cheap JSON edit + `python tools/gen_params_md.py` regen.

## STEP 3 - comp_trust_min_comps=3 provenance (REPORT ONLY, nothing changed)

The draft_000435 anchor-run snapshot (git_hash 10d610c..., stamped
2026-07-16) recorded `comp_trust_min_comps=3` while the code default is 5.

1. **Live value + file path.** `comp_trust_min_comps: 3` in
   `C:\ASTRO\python\VYVAR\config.json` (line 72). This is the single
   config.json consumed by the UI/pipeline (only one exists in the repo).
   Code default is 5 (`config.py:1948`, `_i01("comp_trust_min_comps", 5, 3, 20)`).

2. **Git provenance (config.json IS git-tracked, not ignored).**
   Line history of the key:
   - `70c23d0` (Milan Uhlar, Sun Jun 14 2026, "solver: stale-hint Gaia cone
     recenter ... Brno production fix") first introduced the key as
     `comp_trust_min_comps: 5`.
   - `1c80219` (Milan Uhlar, Tue Jun 16 20:47:34 2026, "feat(comp): Phase-1
     graceful comp degradation (validated matrix 164157)") changed it
     `5 -> 3`. That commit body states the intent explicitly: "trust
     green_min=3; sigma scales with N". This is the current committed HEAD
     value (working tree matches HEAD, no uncommitted diff).
   - The Jun 16 change predates the 2026-07-16 anchor run, so the run honestly
     read 3 from config.json.

3. **Clamp interaction (`config.py:1949-1950`).** The clamp lowers
   `comp_trust_min_comps` to `phase01_comparison_n_comp_max` only when
   `min_comps > n_comp_max`. In config.json `n_comp_max=8` and
   `comp_trust_min_comps=3`, and `3 > 8` is False -> clamp does NOT fire.
   The 3 came DIRECTLY from the JSON value, not from the clamp interaction.
   (`n_comp_min=3` in JSON is a separate Phase-1 selection key and does not
   feed this clamp.)

4. **comp_iterative_clip_enabled=True (other science-relevant deviation).**
   Value source: `config.json:223 "comp_iterative_clip_enabled": true`
   (code default False, `config.py`). Git provenance: introduced as `true` in
   `70c23d0` (Sun Jun 14 2026, same Brno production-fix commit) and never
   changed since; committed HEAD value, no uncommitted diff. No judgment.

Note: no modification made to `config.json` or any default. The
keep-3-or-revert-to-5 decision is Milan's science call, to be taken in chat.

## pytest status

`python -m pytest -q` after STEP 2 commit: **903 passed, 19 skipped** (the new
hidden->expert guard test accounts for the +1 vs the prior 902). Green after
every closeout commit.

## Unpushed commits (STEP 4 push target)

6 commits on top of origin/main `e782786`, oldest first:
- `548e7ae` PARAMS-REGISTRY-UI STEP 1: machine-readable params registry + parity guard test
- `addb05a` PARAMS-REGISTRY-UI STEP 2: generated VYVAR_PARAMS.md + freshness test
- `547008a` PARAMS-REGISTRY-UI STEP 3: tiered Parameters dashboard generated from registry
- `e37826d` PARAMS-REGISTRY-UI STEP 4: PDF Configuration page in SUMMARY MEASURE REPORT
- `3d08213` docs(result): PARAMS-REGISTRY-UI closeout
- `4b6c012` PARAMS-REGISTRY-CLOSEOUT STEP 2: hidden implies expert tier

## STEP 4 - push

Superseded by PARAMS-REGISTRY-PUSH below (Milan authorized the push via that task).

---

# PUSH (PARAMS-REGISTRY-PUSH, 2026-07-17)

Milan authorized the push. Committed the two doc steps first, then pushed.

## Doc commits added before push

- `20cd3e1` `docs(result): PARAMS-REGISTRY-CLOSEOUT provenance + hidden-tier rule`
  (this result file).
- `8080cc3` `docs: stamp PARAMS-REGISTRY-UI arc (DECISIONS/JOURNAL/STATE)`:
  - DECISIONS: COMP-TRUST-MIN-COMPS entry (3 vs default 5 = INTENTIONAL).
  - JOURNAL: PARAMS-REGISTRY-UI arc entry (registry 304, guard tests, dashboard
    basic 12 / advanced 75 / expert 217, PDF Configuration page, min_comps closed).
  - STATE: UI block wave 1 (parameters) DONE; next data dashboards wave.

## Final pytest

`python -m pytest -q` -> **903 passed, 19 skipped, 31 warnings in 240.17s**. Green.

## Push

`git push origin HEAD:main` -> `e782786..8080cc3  HEAD -> main`.

Pushed HEAD: **`8080cc3`**.

`git log --oneline e782786..HEAD` (8 commits, newest first):

```
8080cc3 docs: stamp PARAMS-REGISTRY-UI arc (DECISIONS/JOURNAL/STATE)
20cd3e1 docs(result): PARAMS-REGISTRY-CLOSEOUT provenance + hidden-tier rule
4b6c012 PARAMS-REGISTRY-CLOSEOUT STEP 2: hidden implies expert tier
3d08213 docs(result): PARAMS-REGISTRY-UI closeout
e37826d PARAMS-REGISTRY-UI STEP 4: PDF Configuration page in SUMMARY MEASURE REPORT
547008a PARAMS-REGISTRY-UI STEP 3: tiered Parameters dashboard generated from registry
addb05a PARAMS-REGISTRY-UI STEP 2: generated VYVAR_PARAMS.md + freshness test
548e7ae PARAMS-REGISTRY-UI STEP 1: machine-readable params registry + parity guard test
```

## Working tree after push

Clean with respect to tracked files (no staged/unstaged modifications). Only
pre-existing untracked files remain, none related to this arc:
`docs/VYVAR_CODE_AUDIT.md`, `docs/round2_figs/v0454_lc_vyvar.png`,
`scripts/dy_peg_night_run_bvr.py`, `scripts/forensic_disc_ui_match2.py`,
`scripts/qatar8_night_run_v.py`.

## Note

This PUSH section was added as a 9th commit (`docs(result)`) immediately after
the 8-commit push and pushed on top; see that commit hash in the chat report.
