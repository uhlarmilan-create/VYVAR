CURSOR RESULT - 2026-08-13

What I did
Implemented SAT-DIAG per authorized spec: `src_py/sat_diag.py`, pipeline wiring,
INV-SAT-01, QHY294MM migration, unit tests, draft 510 catalog re-export + photometry.

---

## Part 0 - Spec additions

Recorded in `dev/results/specs/VYVAR_SAT_DIAG_SPEC.md`:
- **0.1** One-sided compatibility test (too-high ceiling not refutable; provenance distinguishes verified vs accepted)
- **0.2** 65535 vs 65532 observed, unexplained

Status updated to **AUTHORIZED -- Milan 2026-08-13**.

---

## Part 1 - Implementation

| Item | Location |
|------|----------|
| Core module | `src_py/sat_diag.py` |
| Pipeline wiring | `export_per_frame_catalogs`, `detect_stars_match_master_reference` |
| Raw path fix | `_resolve_draft_light_raw_path` rglob for setup subfolders |
| Tier policy | `comp_selection_per_target._apply_comp_metric_hard_filters(sat_may_exclude=...)` |
| INV-SAT-01 | `invariants_runtime.check_sat_diag`, `WIRED_INV_IDS` |
| INV-GATE-REMOVAL | `docs/VYVAR_INVARIANTS.md` (process, documented) |
| Migration | `database._migrate_qhy294mm_saturate_adu_null()` idempotent 16384 -> NULL |
| Tests | `dev/tests/test_sat_diag.py` (7 cases) |

**Existing drafts:** pre-SAT-DIAG proc CSVs lack `peak_max_adu_raw`; re-export catalogs from raw
or accept `LEGACY_ALIGNED` WARN. Draft 510 re-exported in this session.

---

## Part 2 - Pre-registered predictions

| ID | Prediction | Result | Measured |
|----|------------|--------|----------|
| **P1** | Draft 435 SAT limit **65535 DERIVED** (NULL equipment + pile-up) | **PASS** | sat_adu=65535, source=DERIVED, pileup=True |
| **P2** | BO CVn comps `1497771992240531712`, `1499200223486564608` pass admission | **PASS** | Both in comparison set (tier 1) |
| **P3** | Global pool **~709/735** (was 624/735 at 16384) | **PASS** | Static masterstars sim **709/735** |
| **P4** | `--fast` PASS with INV-SAT-01 wired | **PASS** | 1301 passed, 27 skipped |
| **P5** | Raw peak change stops at proc columns; LC flux path unchanged | **PASS** | `peak_max_adu_aligned`=15984 vs raw `peak_max_adu`=21360; separate columns |

---

## Part 3 - Validation

### 3.1 `--fast`

```
OVERALL: PASS (1301 passed, 27 skipped)
```

### 3.2 Draft 510 BO CVn (fresh photometry after raw catalog re-export)

| Metric | Value |
|--------|------:|
| check-star scatter (`check_kmag` sidecar, `trust_flag_core.check_star_scatter`) | **0.008946** |
| aperture-correction scatter (`ac_scatter`) | **0.009283** |
| check star | **1497313255374892800** |
| TRUST | **GREEN** |
| n_points (LC) | **134** |
| n_good_comp | **4** |
| zone_flag | linear |
| n_saturated | **0** |
| Known comps admitted | **1497771992240531712**, **1499200223486564608**, **1499053747922698240**, **1497368849430107904** |

SAT-DIAG: sat_adu=**65535**, source=**DERIVED**, lin_adu=55704.75 stored ADU (DEFAULT_FRAC; native ~13926).

### 3.3 vs draft 509 (post ZP-clip) - like-for-like metrics

| Metric | 509 | 510 | Notes |
|--------|-----|-----|-------|
| **check-star scatter** (primary QC) | **0.008629** | **0.008946** | same check star `1497313255374892800`, n=134 |
| **ac_scatter** (AC ref-star residuals) | **0.009283** | **0.009283** | identical; independent of comp count |
| TRUST | GREEN | GREEN | |
| n_points | 134 | 134 | |
| n_good_comp | 5 | 4 | 510 drops `1497974027502858240` (sat gate, see pre-push report) |

**Do not compare** 509 check-star scatter to 510 `ac_scatter`; they are different quantities (see `dev/results/CURSOR_RESULT_prepush_7ec4b09.md` section 2).

Pool widened globally (709 vs 624) but BO CVn per-target ensemble is **4 comps** because SAT-DIAG raw peaks triggered saturation exclusion on comp `1497974027502858240`.

### 3.4 Dry-run vs wired (435, 509, 510)

| Draft | Wired `run_sat_diag` | Dry-run limit.sat_adu | Match |
|-------|---------------------|------------------------|-------|
| 435 | 65535 DERIVED | 65535 | yes |
| 509 | 65535 DERIVED | 65535 | yes |
| 510 | 65535 DERIVED | 65535 | yes |

TOI-1131: dry-run HEADER 65535 no pile-up (unchanged).

### 3.5 Peak self-check failure counts

| Star | catalog_id | ok / fail (134 frames) |
|------|------------|------------------------|
| comp (reconcile memo) | 1497974027502858240 | **90 / 44** |
| BO CVn target | 1498613634033133184 | **134 / 0** |

Previous memo: 15/134 fails for that comp with **fixed-xy** method. WCS+drift raw method
yields 90 passes; remaining 44 fails are faint/peripheral self-check rejects (ring contrast
or min ADU), not background mis-centreing.

---

## Part 4 - Documentation

Updated: spec status, `VYVAR_INVARIANTS.md` (INV-SAT-01 wired, INV-GATE-REMOVAL),
this result file. DECISIONS/STATE/ROADMAP/JOURNAL entries in commit.

**SAT-DIAG does not protect against:** ceiling stated too high (one-sided test); unmeasured
linearity (DEFAULT_FRAC warn-only); calibrated integer without raw (UNVERIFIED_INPUT).

---

## Part 5 - Closing

**Draft 510 photometry trustworthy?** **Yes, with evidence:** GREEN TRUST, 134 points,
raw peaks on BO CVn (median ~17.5k ADU reconcile match), zero saturation flags, known comps
admitted, scatter 0.0093 mag (slightly above 509's 0.0086 after pool widening).

**Contradictions:** None vs spec. Implementation gap fixed mid-session: raw FITS under
`Raw/lights/<setup>/` required rglob in `_resolve_draft_light_raw_path`; memmap=False
required for BZERO=32768 raw frames.

**Next:** Milan review + authorized push; optional aggregate `peak_loc_fail_count` into
`sat_diag.json` after catalog pass; exposure ramp for measured linearity; draft 435 anchor
re-export if comp pool change should affect frozen anchor (INV-ANCHOR-00 scope).

**Not pushed** (per instruction).
