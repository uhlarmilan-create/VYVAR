# preview-VYVAR.0.9.0 (pre-release)

Commit: `226d269f8648419ee834dbf58b140599453f5f3a`

## Important - data processed with preview-20260723 should be reprocessed

That build was missing the order-2 sky-surface subtraction during preprocessing. The effect was an
underestimated background sigma, so the nominal 2.1 sigma detection threshold operated at about
1.56 sigma: roughly 3.5x more raw detections, of which about 40% had no catalogue counterpart
(against ~4% in a healthy run). Photometry of well-detected targets was not measurably affected,
but the target list and the comparison-star pool were drawn from an inflated catalogue.

## Changes since preview-20260723

- **Phase 0 target identity.** Targets were matched to catalogue entries by proximity within
  `max(5 x plate_scale, 10")` - 48.85" on a wide-field rig - and adopted the matched star's
  identifier. On a test field, 67 of 322 targets were measured on a different star than intended,
  47 of them more than 1 magnitude brighter. Target selection is now an identity join on the Gaia
  source id, with a single positional cross-match whose acceptance follows a 1% chance-coincidence
  budget derived from the measured local source density.
- **Exoplanet target promotion restored** (it had stopped producing rows silently).
- **Observer site resolution** unified across the UI and headless entry points, with no default
  fallback; a site that cannot be resolved now stops the run instead of substituting one. Affects
  BJD and exported site coordinates on headless runs.
- **Preprocessing is about 60x faster** (18.3 s to 0.29 s per frame), numerically neutral.
- **New runtime guards**: preprocess gradient (`INV-PREP-01`), masterstar catalogue purity
  (`INV-MS-01`), Phase 0 identity (`INV-PHASE0-ID`).
- **Catalogue provenance** (Gaia and VSX identity) recorded in run metadata and compared by the
  validation gate.
- **Operator log** is now a durable full-run record rather than a truncated tail.
- Light-curve column set no longer depends on which entry point started the run.

## Compiled build verification (2026-07-29)

| Gate | Result |
|------|--------|
| Modules compiled | 90 |
| Import smoke | 90/90 |
| `--fast` (compiled) | PASS (1198 passed, 30 skipped; 488 s) |
| P1 golden (compiled) | 7/7 (978 s) |
| Anchor `--full` (compiled) | PASS core `b7f980c0...` n=325; extended `2c43bbbf...` n=487; plan-regen 875; active 165; `full-catalog-provenance` PASS (2889 s pipeline; 3277 s harness) |

## Retiring preview-20260723

**Withdrawal reason:** missing order-2 sky-surface subtraction on the mono preprocess path inflated
pass-1 DAO detections and DAO_ONLY fraction (~40% vs ~4% healthy). Target lists and comp pools from
runs on that preview should be treated as suspect until reprocessed on this build or later.

**Recommendation:** keep git tag `preview-20260723` on the private repo (marks the commit that
carried the regression). Remove the GitHub pre-release and its assets when Milan is ready; the tag
remains provenance.

**Local copies of preview-20260723 artefacts (if needed for user support):**

- `tmp/cython_release/bundle/dist/VYVAR-preview-20260723-win64.zip` (~324 MB)
- `tmp/cython_release/bundle/dist/VYVAR-preview-20260723-linux-x64.tar.gz` (~489 MB)
- `tmp/cython_release/bundle/dist_win/VYVAR-preview-20260723-win64.zip`

## Bundle naming note

`bundle_name()` yields `VYVAR-preview-VYVAR.0.9.0-<platform>` (the word VYVAR appears twice).
Tag is `preview-VYVAR.0.9.0`; do not change without Milan.
