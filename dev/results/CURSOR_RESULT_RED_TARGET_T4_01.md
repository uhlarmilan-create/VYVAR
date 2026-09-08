CURSOR RESULT - 2026-09-08 RED-TARGET-T4-01

Date: 2026-09-08. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 1761d1d.
Class: WIRE. STOP on A3 / G2 drift. B+C landed. A reverted.

Standing authority: refute before executing.

## What I did

Wired (i-a) T4_FALLBACK (commit A `817f1f9`), ran G1 then G2
`--full` aperture-only. G2 FAILED. A3 is refuted. Reverted A
(`8af9a69`). Landed skip-manifest sidecar (commit B `bc945ce`)
and docs / this file (commit C). Pushed
`origin consolidate-01:consolidate-01` only.

## Refute -- A3 (governing)

A3: "None. M1: no unpinned 516 target has an empty cap (R CVn
is pinned). G2 --full must be BYTE-IDENTICAL."

False. M1 measured the RAW spatial pool vs cap 0.79 for R CVn
(0/3300). The production last rung is the same cap applied to
the post-MAD / post-RMS-ceiling / post-isolation set. Two
unpinned 516 targets already undershoot that quality-filtered
cap and take the legacy relax-to-full-set path (C-EXPORT-GAP
log: "color filter relaxed to full under-ceiling single-source
set", lim=0.640, n=95, admitted 8 by RMS):

- `1497007144465726080` CV CVn
- `1497683996951418880` HAT-188-0002048

A0 ranks that same quality set by |dBP-RP| instead of RMS.
That is a science change on live 516, not a no-op.

G2 on `817f1f9` (`tmp/session_baseline/20260908T153156Z`):

| gate | result |
|---|---|
| full-snapshot-sha-core-aperture | PASS era04_aperture `d55fcc9d` n=53 |
| full-photometry-sha-core-aperture | FAIL run `f6cb0416` n=53 vs snap `d55fcc9d` |
| full-photometry-sha-ext-aperture | FAIL run `8282f3ce` n=157 vs snap `cc8b532e` |
| full-science-compare | FAIL science_failures=1 tid `1497683996951418880` (HAT-188-0002048) |
| phase2a_empty_comp_drop | PASS allowlisted 3 (pins untouched) |
| OVERALL | FAIL |

`comparison_stars_per_target.csv` in that run: 16 rows with
`color_rms_t4_fallback` = 2 targets x 8 comps (CV CVn +
HAT-188). Pin-RMS trio still aborted
(`n_survivors=2 < n_min=3` on `1500467303261764096`).

Task: if G2 drifts, STOP and report -- needs an anchor plan
before landing. A reverted. Reland row: RED-TARGET-T4-RELAND.

## Architect error ledger

18. C-EXPORT-GAP review asserted a 1:1 link
    "3 export failures = phase2a_empty_comp_drop=3" from count
    coincidence without per-target evidence. M1 refuted it
    (the FAILURE IDs had comps; the drop trio is a pin-RMS
    abort). Same class as 9-17.

19. R-CVN-EMPTY-COMP-M1 task cited tier limits from code
    fallbacks (0.25/0.48/0.79) instead of the live config
    (`comp_color_tiers` 0.15/0.30/0.55/1.10). Same class.
    Ladder last rung remains cap 0.79; T4 1.10 is not on the
    ladder (`photometry_comp.py` ladder build).

## Part A -- T4_FALLBACK (reverted; not in HEAD)

Implemented on `817f1f9` then reverted `8af9a69`.

A0 (`photometry_comp.py` after the color-ladder loop): when
`math.isfinite(tb)` and `len(selected) < n_comp_min`, if the
quality set `out` >= n_min -> T4_FALLBACK (`selected=out`,
sort `_delta_bprp_abs` first); else empty +
`attrs["color_fallback"]=True`. NaN-BP-RP keeps the legacy
relax path (`:1328-1334` `_delta_bprp_abs=0.0` bypass
untouched). Pins: early-return
(`photometry_comp.py:2221-2256`) still first.

A1: `sel_note` `color_rms_t4_fallback`
(`comp_selection_per_target.py`). Conditional
`color_fallback` / `max_delta_bprp_used` on
`comp_quality_*.json` only when the rung fires
(`phase2a_target.py`) -- not a new always-on SHA column.
`comp_delta_bprp_map` already carries per-comp deltas.
No new `comparison_stars_per_target.csv` columns (those
would break G2 even on a no-op selection).

A2 synthetic tests a-e (plus last-rung=0.79 assertion):
6 passed on `817f1f9`. Removed with the revert.

Quality chain = SNR floor: RMS ceiling 0.080 mag ~ SNR 13.6
per epoch (recorded in D-RED-TARGET-T4-01).

## Part B -- skip manifest (landed)

Path: `photometry/lightcurves_skip_manifest.json`.
Writer: `src_py/skip_manifest.py`.
Hook: `photometry_phase2a.py` `_phase2a_finalize_exports`
(after PRE-IMPL-01 weight rewrite, before the final return).

Always written. `targets: []` when nothing is skipped.
Classes: `no_comps_stub` | `pin_rms_abort` | `export_empty` |
`other`. LC CSV bytes not touched.

SHA glob exclusion (`dev/tests/photometry_sha.py:100-105`):

- `**/photometry/**/lightcurve_*.csv`
- `**/photometry/**/comp_quality_*.json`
- `**/platesolve/**/comparison_stars_per_target.csv`
- `**/photometry/**/lightcurves/comp_qa_*.json` (ext)

Name `lightcurves_skip_manifest.json` matches none of these.
G2 not repeated after B (asserted in
`dev/tests/test_skip_manifest_01.py`).

Tests: 5 passed (glob exclusion, helper does not collect the
file, three classes on synthetic inputs, empty-list file,
`other`).

G2-run census (uncommitted hook already fired during the A
`--full`; sidecar is outside the SHA, so this did not cause
the drift): 203 listed targets; `export_empty` x3 (149884 /
149984 / 150041 class); `pin_rms_abort` x5 =
M1 trio `1497245497969274240` / `1498425548825498112` /
`1497227287309482624` plus `1497181966814590848` and
`1498064771572297856` (ZTF J141004.59+410003.7) -- pinned
targets with no LC in `active_df`; remainder `other`
(skip_photometry / never-exported). `no_comps_stub` = 0 on
this night because the counted empty-comp drops are pinned.

Deferred: `skip_reason` column in LC CSVs at the next
natural era re-cut (D-LC-SKIP-MANIFEST-01).

## Part C -- docs

- D-RED-TARGET-T4-01 (rule + A3 STOP / not in HEAD)
- D-LC-SKIP-MANIFEST-01 (sidecar + deferred CSV column)
- ROADMAP: R-CVN-EMPTY-COMP CLOSED (misnomer split);
  new OPEN PIN-RMS-ABORT-01 (Milan decision),
  LCRMS-NAN-01 (LOW), RED-TARGET-T4-RELAND (Milan GO)
- FAILURE wording for the lc_rms=nan class: unchanged
  (Milan (ii))
- JOURNAL 2026-09-08; STATE one-liner

## Gates

| gate | when | result |
|---|---|---|
| G1 `--fast --clean` | after A `817f1f9` | PASS 1630 passed, 34 skipped; clean-tree PASS |
| G2 `--full` aperture | after A | FAIL (A3 refute; hashes above) |
| G1 `--fast --clean` | after B `bc945ce` | PASS 1629 passed, 34 skipped; clean-tree PASS |
| G2 after B | not repeated | name outside SHA globs (asserted) |
| `--full-epsf` | not run | no ePSF-graph name touched |
| G4 live 516 | not rewritten | prefixes `bfa24039` / `13e77cf8` / `172f9540`; A reverted so selection path == `1761d1d` |

## Errors (if any)

G2 FAIL is the STOP, not an implementation crash. No
`--full-epsf`. Dirty-tree historical `dev/results/context/`
and `VYVAR_VALIDATION_LEDGER.json` not staged.

## Files changed

A (reverted, still in history as `817f1f9` / `8af9a69`):
`src_py/photometry_comp.py`,
`src_py/comp_selection_per_target.py`,
`src_py/phase2a_target.py`,
`dev/tests/test_red_target_t4_01.py`.

B `bc945ce`:
`src_py/skip_manifest.py`,
`src_py/photometry_phase2a.py`,
`dev/tests/test_skip_manifest_01.py`.

C:
`docs/VYVAR_DECISIONS.md`,
`docs/VYVAR_ROADMAP.md`,
`docs/VYVAR_JOURNAL.md`,
`docs/VYVAR_STATE.md`,
`dev/results/CURSOR_RESULT_RED_TARGET_T4_01.md`.

## STOP

T4_FALLBACK is the Milan (i-a) rule and is recorded as
D-RED-TARGET-T4-01. It is not in production HEAD. Reland
requires an explicit anchor plan (era recut or a scoped
G2 exception for the two unpinned quality-cap undershoots)
before `817f1f9` is re-applied. Skip-manifest and the
R-CVN-EMPTY-COMP row split are in.
