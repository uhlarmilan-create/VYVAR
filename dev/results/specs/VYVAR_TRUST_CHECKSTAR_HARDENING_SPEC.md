# VYVAR — Trust / Check-star Hardening Spec

Scope: close the residuals of audit Findings **A** and **B**, fold in **CS-1**, decide
**C**, and re-check **E**. All changes live in `trust_flag_core.py` (+ config wiring +
tests). Mission: "trust in the numbers" — a target must never be GREEN on the strength of
absent or too-thin verification.

Status of the parent findings (audit re-read, commit `39690b7`):
- **Finding A — main path already FIXED.** A target missing from the trust map maps to
  `_UNEVALUATED_TRUST = "RED"`. Only a residual remains (Section 1).
- **Finding B — main path already FIXED.** A missing check-star file -> soft
  "no check-star verification available" -> YELLOW. Only a residual remains (Section 2).

INVARIANTS (do not break):
- **Photometry byte-identity holds.** Trust is post-Phase-2A. Photometry SHA on draft_000387
  MUST stay `203254fd...` (core) / `95a5515a...` (full).
- **Trust distribution WILL change on 387** (intended). Re-baseline counts; do NOT require
  trust to stay equal.
- English/ASCII only.

## 1. Finding A residual — un-evaluated defaults to GREEN at trust_map build

**Fix:** `info.get("trust") or _UNEVALUATED_TRUST` (and parallel `trust_reason` guard).

## 2. Finding B residual + CS-1 — thin check-star -> spurious GREEN

**Fix:** `check_star_min_epochs` (default 5, clamp >= 3); `check_star_scatter` returns
`(scatter, n_check)`; `classify_warnings` emits `"insufficient check-star verification (n=...)"`
when `0 < n_check < min`.

## 3. Finding C — population std (ddof=0) at small N

**Decision:** apply `np.nanstd(km, ddof=1)` once min-epochs guard is enforced.

## 4. Finding E — len(soft) >= 3 reachability

Ensure clean `short_baseline` + thin comp + thin check -> **YELLOW**, not RED
(`short_baseline` excluded from escalation count).

## 5. Unit tests

`tests/test_trust_checkstar_hardening.py` per sections above; keep `tests/test_trust_flag.py` green.

## 6. draft_000387 validation

1. Photometry SHA unchanged.
2. Re-baseline trust counts: **1382 YELLOW / 106 RED** at `comp_trust_min_comps=5` (floor-5
   baseline on draft_387). Pre-floor-5 stored counts (1400/88) superseded.
3. Report targets with `n_check < min`.
4. No clean short_baseline -> RED regression.

## 7. Out of scope

CS-2/3/4, min_comp cross-field experiment.
