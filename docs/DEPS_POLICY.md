# VYVAR - Dependency upgrade policy

How VYVAR moves its scientific dependencies (numpy, astropy, photutils, and the
rest of `requirements.txt`) forward **without ever silently changing the
numbers**. The governing idea: every upgrade is either proven byte-identical
against the anchor, or it becomes a documented, deliberate re-anchor. No upgrade
is applied by accident, and no upgrade is feared - both outcomes produce
knowledge.

## Why we pin majors

`requirements.txt` holds compatible-range pins (e.g. `numpy>=2.4.4,<3`,
`astropy>=8.0,<9`, `photutils>=3.0,<4`). The upper bound holds the **major**, so
a fresh install today reproduces the majors the pipeline was validated on.
Patch/minor bumps inside a range are cheap; crossing a major is the gated ritual
below.

## The quarterly ritual (candidate cycle)

1. **Scout.** Read the upstream changelogs for the candidate versions. Map every
   behavior change onto VYVAR usage (grep the code). Distinguish: (a) changes
   that would *move our numbers* (default changes, algorithm refactors), (b)
   deprecations we merely need to migrate, (c) pure gains. Record it as a
   `dev/results/DEPS_SCOUT.md`-style artifact.
2. **Fresh environment.** Install the candidate versions in a clean venv.
3. **Full test suite.** `python -m pytest -q` must stay green.
4. **Anchor gate.** `python dev/scripts/session_baseline_check.py --full` -
   headless `run_full_photometry_pipeline` on the anchor dataset, compared
   byte-for-byte (SHA) against the recorded anchor (currently draft_435).
5. **Decide from the result:**
   - **Identical** ? free upgrade. Move the pin (if a range boundary changed),
     add a `docs/VYVAR_DECISIONS.md` entry, and record the validated version in
     the ledger. Done.
   - **Different** ? this is a *finding*, not a failure. Two legitimate paths:
     - **Adopt-and-re-anchor:** if the new behavior is correct/desired, confirm
       via overlay comparison, then re-anchor per the anchor-gate machinery and
       document the delta and its cause in DECISIONS + ledger.
     - **Hold and report:** if the delta is unexpected/unwanted, keep the pin,
       open a note (and report upstream if it looks like a regression).

No scenario is bad. Identical = a free upgrade with proof. Different = we learned
exactly how a dependency touches our science, on our data.

## In-range vs cross-major

- **In-range refresh** (patch/minor within the current pin, e.g. numpy
  2.4.3 ? 2.4.4): run pytest + `--full`. Expected byte-identical; if so, just
  record it - no pin move needed since the range already permits it.
- **Cross-major** (e.g. photutils 2.x ? 3.x, astropy 7.x ? 8.x): full scout +
  code migration (kwarg renames, default freezes) **before** the gate, then
  pytest + `--full`. Byte-identity is *not* guaranteed - performance refactors
  can legitimately reorder float ops - so budget for an adopt-and-re-anchor.

## The `--fast` nudge

`python dev/scripts/session_baseline_check.py --fast` surfaces an
**informational** `deps-outdated` line (`pip list --outdated`, filtered to the
pinned scientific packages). It is WARN/PASS/SKIP only - it **never** fails the
session and is offline-tolerant (SKIP if no index). It exists to keep the
candidate list visible, not to pressure an upgrade.

## Current watchlist (see `dev/results/DEPS_SCOUT.md` for detail)

- **photutils 3.1** (unreleased) - aperture photometry speedups + segmentation
  masking evaluation; not adopted until released and scouted.
- **numpy 2.5.x** - minor bump within `<3`; candidate for a future in-range
  cycle (pytest + `--full`).

CYCLE 2 (photutils 3.0 + astropy 8.0) executed 2026-07-20; see DECISIONS
`DEPS-CYCLE-2` and `dev/results/CURSOR_RESULT_deps_cycle2.md`.
