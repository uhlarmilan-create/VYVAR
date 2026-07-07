CURSOR RESULT — 2026-07-07 (Fable audit follow-ups)

What I did
Stage A: F-BINGAIN-1 diagnostic (no code commit). Stages B+C: airmass attribution fix,
GAIA-ID guard live-tree check, docs addendum.

## Output / findings

### Stage A — F-BINGAIN-1 (diagnostic only; STOP for Milan)
Report: `tmp/phaseBinGain/bingain_diag.{md,json}`

**A0 inventory:** cache dominated by bin2+gain0 (8226 rows); QHY294MM eq1 DB gain/RN 3.17/7.6 e-.

**Resolver provenance (draft_424 bin2):** 12/12 frames -> `header_index_mapped` gain **3.17 e-/ADU**;
scaled-db (`exponent=2`) fraction **0%**. RN from db scaled to **15.2 e-** (bin x1).

**A1 photon transfer:** 8 consecutive BO CVn light pairs; g_eff (RN-corr) mean **~0.90** (field
lights; no local bin2 flats). **INCONCLUSIVE** for 3.17 vs 12.7 verdict.

**A2 ADU ceiling:** data_max ~68608 ADU on saturated stars (16-bit full-scale).

**Recommendation:** Do not change gain exponent yet; flat-sequence PT needed. Production wide-rig
path already uses 3.17 via index map, not db x4.

### Stage B — F-AIRMASS-CITE (committed)
- `pipeline.py:8887`, `:8985` -> Kasten & Young (1989)
- `kastenyoung1989` added to `CITATIONS.bib`; CORE-candidate note in `citations.py`

### Stage C — GAIA-ID-FLOAT-GUARD
Live-tree grep: no production `astype(float)` / `int(float)` on catalog_id/source_id paths.
`read_csv` hits carry `dtype={"catalog_id": str}`. **CLOSED (verified 2x).**
Note: `scripts/archive/diagnostics/_audit_issues.py` has legacy `int(float(...))` (archive only).

### Tests
`pytest tests/` — 535 passed.

## Errors (if any)
None.

## Files changed (Stage B+C commit)
- `pipeline.py`, `CITATIONS.bib`, `citations.py`
- `docs/VYVAR_MATH_PHYS_AUDIT.md`, `VYVAR_STATE.md`, `VYVAR_AUDIT_LEDGER.md`, `VYVAR_JOURNAL.md`
