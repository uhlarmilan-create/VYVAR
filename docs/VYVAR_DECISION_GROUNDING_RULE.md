# VYVAR -- Decision-grounding rule

Date: 2026-06-15  
Status: **ADOPTED** (Milan)

---

## Rule

Any design fork brought to Milan must be grounded in:

1. Physics or mathematics,
2. Peer-reviewed literature, or
3. Documented field practice (e.g. AIJ, SPECULOOS, AAVSO guidance).

Bare engineering preference is **not** sufficient. No "recommended" label without a cited basis.

Grounding may supersede earlier recommendations. When code changes land, method citations belong
in `CITATIONS.bib` at call sites.

---

## Workflow discipline (this arc)

1. **Sandbox** -- prototype / measure under `tmp/phase*` (not committed).
2. **Measure** -- DoD metrics vs ground truth (AIJ / SIPS / constant calibrator).
3. **Milan review** -- PDF + numbers before production commit.
4. **Commit** -- source + tests + docs only; harnesses stay in `tmp/`.

See `docs/VYVAR_PROCESS.md` and `docs/VYVAR_JOURNAL.md` (2026-06-15/16 arc).
