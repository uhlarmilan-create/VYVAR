# VYVAR -- Spec: short-baseline LC-quality classification (#3)

**Date:** 2026-06-10 (rev b, decisions folded in) -- **Author:** Claude (read-only spec).
**Status:** decisions resolved 2026-06-10 (see below); ready for Cursor pending Milan's "implement".
**ROADMAP item:** "`classify_lc_quality` min_frames for short-baseline sessions" (MEDIUM).
**Scope:** classification + trust routing + export eligibility for sessions with too few frames.
NOT a photometry-numeric change. ASCII-only.

---

## Problem

`classify_lc_quality` (`photometry_core.py:4772`) returns `no_data` when `n_frames < min_frames`
(`:4805`, default `min_frames = 20`). A clean ~12-frame session is therefore `no_data`, which has
**two** downstream consequences:

1. Trust gate: `no_data` is not in `_LC_QUALITY_OK = {good, noisy}` (`trust_flag_core.py:29`), so
   it becomes a **hard** warning (`:95`) -> **RED** (`trust_level`, `:116`).
2. Export: `no_data` is in the AAVSO export exclusion mask
   (`export_reports.py:619`, `bad = fl.isin(("no_data","saturated","edge_fail","nondetection"))`)
   -> the target is **dropped from the AAVSO export entirely**.

So a short-but-photometrically-clean session is both auto-RED and unsubmittable. The single
threshold conflates **data volume** (enough points for robust RMS/eta statistics) with a
**quality verdict** (the data we have is bad).

The call site does not override the defaults (`photometry_core.py:4964` passes no `min_frames` /
`min_normal_frac`), so `20` / `0.5` are effectively hardcoded -- they should be config-driven per
the config<->UI parity discipline.

### Use case driving the design: long-period variables (Mira)

Mira / LPV observation is a primary near-term use case. LPVs have periods of hundreds of days and
large amplitude; the science lives in the night-to-night trend, not the intra-night curve. The
standard AAVSO LPV contribution is one averaged magnitude per night from a few frames. A 5-frame
LPV night is therefore legitimate, clean science and **must remain submittable** -- it must not be
classified `no_data` (RED + excluded). This is the opposite regime from fast variables (eclipsing
binaries, RR Lyrae, delta Sct), where few frames genuinely means an unresolved curve.

## Resolved decisions (2026-06-10)

1. **`lc_quality_short_min_frames` default = 3** (config-driven). `[3, min_frames)` ->
   `short_baseline`; `< 3` -> `no_data` (below 3 no meaningful scatter exists). 3 keeps few-frame
   LPV nights submittable. Was proposed 8; lowered for the Mira use case.
2. **Terminal `short_baseline`** (confirmed). A short session gets no `noisy`/`good` sub-verdict;
   the RMS-noisy test is not run for it (unreliable below `min_frames`). `lc_rms` is still
   computed and reported.
3. **Exportable = YES**, YELLOW-tagged. Essential for LPV epoch submission to AAVSO.
4. **`short_baseline` routes YELLOW but is EXCLUDED from the `len(soft) >= 3 -> RED` escalation**
   (`trust_level`, `trust_flag_core.py:116`). It is a "review" signal, not a "shaky" one: a slow
   variable with few frames + thin comps + no check star should stay YELLOW (submittable), not
   RED. **Consequence: ROADMAP Finding E stays OPEN** -- `short_baseline` is not the genuine third
   soft source that should force RED. (This reverses the rev-a suggestion to close Finding E here.)
5. **Future (not now): vsx_type-aware frame thresholds.** The principled fix is to relax the frame
   requirement for LPV/Mira (known from the VSX cross-match `vsx_type`) and keep it strict for
   fast variables. Larger change; out of scope for this spec. Recorded as a follow-up.

## Design

Introduce a distinct terminal classification **`short_baseline`** for "too few frames but
otherwise usable," routed soft (YELLOW), kept exportable, and excluded from the hard soft-count
escalation.

### classify_lc_quality logic (revised order)

1. `saturated` (unchanged; zone wins first).
2. `nf < short_min_frames` (default 3) -> **`no_data`**.
3. `short_min_frames <= nf < min_frames` AND normal-fraction OK (`nf>0 and nn/nf >= min_normal_frac`)
   -> **`short_baseline`** (NEW terminal).
4. `nf < min_frames` reached here = normal-fraction failed in the short range -> **`no_data`**.
5. `nf >= min_frames` AND `nn/nf < min_normal_frac` -> **`no_data`** (existing case, unchanged).
6. noisy-from-zone / noisy-from-RMS / good (unchanged).

### Config keys (config<->UI parity)

Promote defaults to config; pass them at the call site (`photometry_core.py:4964`).

| Key | Default | Meaning |
|-----|---------|---------|
| `lc_quality_min_frames` | 20 | Frame floor for full good/noisy classification. |
| `lc_quality_short_min_frames` | 3 | Below this -> `no_data`; `[short, min)` -> `short_baseline`. Low default supports few-frame LPV/Mira nights (Decision 1). |
| `lc_quality_min_normal_frac` | 0.5 | Min unsaturated/normal fraction. |

Clamp `lc_quality_short_min_frames <= lc_quality_min_frames` (mirror the `n_comp_max >= n_comp_min`
clamp at `config.py:1317`). Register in `VYVAR_PARAMS.md`; expose under a "Data quality &
validation" Settings section (same section the ROADMAP earmarks for comp_qa/trust toggles).

### Trust routing (`trust_flag_core.py`)

- Add `_LC_QUALITY_SOFT = frozenset({"short_baseline"})`.
- In `classify_warnings` (`:94-96`): if `lq in _LC_QUALITY_SOFT` -> `soft.append("short baseline
  ({nf} frames, thin series)")`; the existing `elif` then handles genuine hard values.
- In `trust_level` (`:116`): the `len(soft) >= 3` escalation must **not** count `short_baseline`.
  Implementation: track soft sources excluded from escalation (e.g. count only "escalating" soft,
  or subtract the short-baseline note from the `>=3` test). `short_baseline` alone -> YELLOW;
  `short_baseline` + a genuine hard -> RED (hard always wins).
- Update the docstring gate semantics (`:8-11`) to document `short_baseline` as a non-escalating
  soft.

**Finding E:** remains OPEN (Decision 4). Update its ROADMAP/DECISIONS note: `short_baseline` is a
non-escalating soft, so the "third soft source -> RED" trigger has not arrived.

### Export eligibility (`export_reports.py:619`)

`short_baseline` is **left OUT** of the `bad` exclusion mask -> exported, carrying its YELLOW
trust note (`format_export_trust_note`, `trust_flag_core.py:339`).

## Parity surface (all enum sites)

| Site | Current | Change |
|------|---------|--------|
| `photometry_core.py:4772` | producer | add `short_baseline` branch + new params |
| `photometry_core.py:4964` | call site | pass the 3 config values |
| `trust_flag_core.py:29,95,116` | `_LC_QUALITY_OK`, hard routing, escalation | add `_LC_QUALITY_SOFT`; soft route; exclude from `>=3` |
| `ui_aperture_photometry.py:1554` | `_lc_qopts` filter list | add `short_baseline` |
| `ui_aperture_photometry.py:1555` | `_lc_qdefault` shown-by-default | add `short_baseline` (usable) |
| `ui_aperture_photometry.py:510` | colour map (`no_data` grey) | add amber colour (match YELLOW) |
| `export_reports.py:619` | exclusion mask | leave `short_baseline` OUT (exportable) |
| `photometry_report.py:510-511,680-694` | quality-summary counts | add a `short_baseline` row |
| `photometry_report.py:4048-4055` | per-star LC rendering branch | render as thin/caution |
| `method_lc_output.py:250` | appends `no_data` | **verify** same taxonomy; align if so |
| `config.py` | (none) | add 3 keys + clamp |
| `ui_settings.py` | (none) | expose under Data quality & validation |
| `docs/VYVAR_PARAMS.md` | registry | add 3 keys |

Note: `photometry_core.py:1301/1394/1600` also emit `"no_data"` but in a per-frame/zone taxonomy --
**verify** they are unrelated before assuming no change.

## Test data + byte-identity baseline (re-established)

**All drafts were deleted, including the byte-identity reference `draft_000366` (SHA `770966c3`).**
The baseline must be re-cut from a fresh run before this change can be regression-checked.

Plan (Cursor / Milan's machine; see **`VYVAR_CHIANDH_BASELINE_RUNBOOK.md`** for the full procedure;
uses `C:\ASTRO\python\VYVAR\Archive\Chi_and_H`, filters B/V/Ri):

1. **Re-cut baseline.** Fresh full Chi_and_H run **against the `zaloha` catalog** (the same DB the
   old reference used). Record the new numeric photometry SHA + comp_qa SHA as the new anchor.
   (Baseline is catalog-dependent: it will legitimately shift when the new DB / DR4 lands.)
2. **No-regression check.** Apply the `short_baseline` change; re-run full Chi_and_H. It is a
   many-frame field (~127), so every target has `nf >> 20` -> no `short_baseline` targets -> the
   numeric SHA must be **unchanged** vs step 1.
3. **Exercise the new path.** Truncated 5-frame and 12-frame subsets of Chi_and_H -> confirm
   `short_baseline` classification, YELLOW trust, and presence in the AAVSO export.
4. **Natural data going forward.** Real Mira nights will provide genuine short-session inputs.

## Tests

1. `classify_lc_quality` boundaries: `nf=2` -> `no_data`; `nf=3` -> `short_baseline`;
   `nf=min-1` -> `short_baseline`; `nf=min` -> `good`/`noisy`; short range with
   `nn/nf < min_normal_frac` -> `no_data`; saturated zone wins at any `nf`.
2. Trust routing: `short_baseline` alone -> YELLOW; `short_baseline` + thin-comp + check-soft
   (3 soft, one is short_baseline) -> **YELLOW** (NOT RED -- escalation excludes short_baseline);
   `short_baseline` + a genuine hard -> RED.
3. Export: a `short_baseline` target is present in the AAVSO export with `trust=YELLOW`.
4. Regression: new Chi_and_H baseline SHA stable across the change (step 2 above); truncated
   subset shows `no_data -> short_baseline` + YELLOW + exported.
5. config<->UI parity check for the 3 new keys.

## Definition of Done

All parity sites updated; 3 config keys added with clamp + registered in `VYVAR_PARAMS.md` +
exposed in Settings; `short_baseline` routes YELLOW, is non-escalating, and is exportable; Finding
E note updated (stays open); new Chi_and_H byte-identity baseline recorded and stable across the
change; tests 1-5 pass; 0 PDF overflow; file content ASCII-only.

## Follow-up (out of scope)

- **vsx_type-aware frame thresholds** (Decision 5): relax frame floors for LPV/Mira from VSX
  type; keep strict for fast variables. Record in ROADMAP when scoped.
