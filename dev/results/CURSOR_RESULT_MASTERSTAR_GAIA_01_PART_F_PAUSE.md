# MASTERSTAR-GAIA-01 Part F — E3b PAUSE REPORT

Date: 2026-08-19  
Status: **PAUSE before Phase 1** (comp-pool survival 77.8% < 90% gate)

## E3b pass-2 survival (anchor 516 / NoFilter_60_2)

Reproduced pass-1 (307 detections) + pass-2 at **old** params (?=1.9, tol=5 px) on cone-on-chip Gaia (4131 rows). Pass-2 accept count matches production path: **3315** (baseline infolog 3314).

Applied **new** params (?=5.0, tol=2 px) at each Gaia seed that pass-2 accepted under old params:

| Metric | Value |
|--------|-------|
| Pass-2 detections (old params) | 3315 |
| Survive new pass-2 (DETECTED_P2) | **1796 (54.2%)** |
| ? FORCED_SEED (centroid?2, SNR?4) | 254 |
| ? SEED_REJECTED | 1265 |
| ? TOO_FAINT | 0 |

Raw: `dev/results/context/session_20260819_msgaia01/e3b_pass2_survival.csv`

## Comp pool survival (2356 members)

| Origin | Count |
|--------|-------|
| Pass-1 (always DETECTED_P1) | 250 |
| Pass-2 origin | 2106 |

| Final state (comp pool) | Count |
|-------------------------|-------|
| DETECTED_P1 | 250 |
| DETECTED_P2 | 1467 |
| FORCED_SEED | 116 |
| SEED_REJECTED | 523 |
| **Remain DETECTED (incl. FORCED_SEED)** | **1833 / 2356 = 77.8%** |
| **Strict DETECTED only (P1+P2, no seed)** | **1717 / 2356 = 72.9%** |

With `masterstar_forced_seed_comp_pool_enabled=False` (default), comp pool effectively sees **72.9%** strict detection survival.

**Gate:** 77.8% < **90%** ? **PAUSE** per amendment 1.

Raw: `dev/results/context/session_20260819_msgaia01/e3b_comp_survival.csv`, `e3b_survival_report.json`

## Architect decision needed

523 comp-pool members (22.2%) would lose pass-2 detection under tightened params; only 116 recover via FORCED_SEED admission. Options:

1. **Proceed anyway** — accept comp pool shrink; E4 MAG gate still binding; photometry may shift if comps drop.
2. **Enable FORCED_SEED comp-pool admission** — recovers +116 (still ~78%, below 90%).
3. **Intermediate pass-2 params** — re-sweep empty-sky vs survival tradeoff (e.g. ?=4, tol=2.5) with new empty-sky audit.
4. **Hybrid** — keep tightened empty-sky gate but relax pass-2 only for stars already in comp pool (membership-specific; needs explicit DECISIONS entry).

## Not executed (blocked by PAUSE)

- Clean rebuild Phase 0 ? 1 ? 2A
- E3a census / E4 MAG / E5 meters
- Anchor + P1 recut (516-04 procedure)
- Exports + SUBMIT-01
- Docs update (STATE/ROADMAP/DECISIONS/ledger)

## Runtime

E3b harness: **26 s** (`tmp/masterstar_gaia_01_e3b.py`)
