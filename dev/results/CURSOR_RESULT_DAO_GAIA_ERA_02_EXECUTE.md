CURSOR RESULT - 2026-08-19 (DAO-GAIA-ERA-02 execute)

What I did
Executed ERA-02 directive: P0 L4 harness fix + fire proof; full rebuild preserved
to `draft_000516_era_candidate`; pinned-comp Phase-2A; T-panel (T0-T3 + T1-abs).
Live `draft_000516` restored and SHA-verified 477dc8cf n=97. NO anchor recut,
NO exports, NO push.

## P0 — L4 era-native offset

- Fixed `_eval_l4()` in `tmp/dao_gaia_era_01_part_c_rebuild.py`: offset from
  **candidate LC** vs AIJ, not baseline `mag_calib_final`.
- Fire proof PASS: +227.6 mmag constant shift ? RMS **4.858 mmag** unchanged.

## Full rebuild + preservation

- Runtime ~70 min photometry on live draft (MS rebuild + Phase 0+1+2A).
- Copied to `Archive/Drafts/draft_000516_era_candidate`.
- Unpinned LCs backed up at `photometry/_lightcurves_unpinned/`.
- Live draft restored 477dc8cf after copy.

## Pinned-comp Phase-2A (decision split)

Baseline `comparison_stars_per_target.csv` frozen; candidate-era MS/flux/detection.

| Check star | Unpinned ? vs 477dc8cf | Pinned ? vs 477dc8cf |
|------------|------------------------|----------------------|
| **BO** | **+227.6 mmag** | **0.0 mmag** |
| **FW** | **+199.9 mmag** | **0.0 mmag** |

**Mechanism: selection-only CONFIRMED.** Per-target comp ensemble re-ranking drives
absolute ZP shift; flux extraction path is unchanged when comps are pinned.

Pinned Phase-2A runtime: **719 s**.

## T-panel (`dev/results/context/session_20260819_era02/era02_tpanel.json`)

### T0 Infrastructure — **PASS**

| Check | Result |
|-------|--------|
| Certificate | PASS 2.5/2.5, ? 4.5/4.0 |
| Census accounting | **100%** (4990/4990) |
| Empty-sky inv | PASS |

### T1 External quality (era-native XVAL + check MAD)

| Metric | BO | FW | GH |
|--------|----|----|-----|
| XVAL RMS (mmag) | **6.24** (matrix 4.86) | **2.64** (matrix 1.52) | n/a |
| Check MAD (mmag) | **5.15** (anchor band 6.08-8.22) | **8.75** (PASS) | **8.65** |

BO XVAL **6.24 mmag** with fixed offset (was false 228 mmag pre-P0). Slightly above
±1 mmag matrix band. BO MAD below 0.85 anchor band (tighter scatter, not looser).

### T1-abs Absolute ZP vs Gaia?Johnson V transform

| Star | Baseline \|?\| (mmag) | Candidate \|?\| (mmag) | Closer to catalog |
|------|---------------------|------------------------|-------------------|
| **BO** | 209 | **19** | **Candidate** (?190 mmag improvement) |
| **FW** | 380 | **180** | **Candidate** (?200 mmag improvement) |
| **GH** | **141** | 332 | **Baseline** (+192 mmag candidate regression) |

Candidate wins ZP accuracy on BO/FW check stars; GH shows **>50 mmag** regression
vs catalog transform (architect review item).

### T2 Shape gate (post-median-offset residual ? 10 mmag)

| Mode | Pass | Notes |
|------|------|-------|
| **Unpinned** | **18 / 48** | Large median offsets remain; epoch shape mostly ?10 mmag after offset |
| **Pinned** | **48 / 48** | Perfect shape match — confirms selection-only |

The 24/46 unpinned population from ERA-02-OPEN investigation: with shape gate
(not byte continuity), **18/48 pass unpinned**; **48/48 pass pinned**.

### T3 Pool / comp stability

| Metric | Value |
|--------|-------|
| Pool Jaccard | **0.704** (2356 base / 2240 cand) |
| Per-target Jaccard median (unpinned) | **0.333** |
| Per-target Jaccard (pinned) | **1.000** (by construction) |

## Verdict for Milan + architect

1. **Mechanism closed:** ensemble selection rebase, not flux measurement defect.
2. **Candidate product:** **DIFFERENT** ZP frame; **not worse** on BO/FW absolute
   catalog alignment; **mixed** on GH.
3. **L2 byte-continuity:** wrong gate for era migration (DECISIONS updated).
4. **ERA acceptance:** T0 PASS; T1 mixed (BO XVAL +1.4 mmag over matrix); T1-abs
   mixed (GH regression); T2 pinned 48/48; pinned experiment definitive.

## Artifacts

| Path | Content |
|------|---------|
| `Archive/Drafts/draft_000516_era_candidate/` | Preserved candidate tree |
| `.../photometry/_lightcurves_unpinned/` | Unpinned rebuild LCs |
| `.../photometry/lightcurves_pinned/` | Pinned-comp Phase-2A LCs |
| `dev/results/context/session_20260819_era02/era02_tpanel.json` | Full T-panel |

## Files changed

- `tmp/dao_gaia_era_01_part_c_rebuild.py` — L4 era-native offset (P0)
- `tmp/dao_gaia_era_02_execute.py` — ERA-02 orchestrator
- `tmp/dao_gaia_era_02_resume.py` — resume pinned + T-panel
- `docs/VYVAR_DECISIONS.md` — ERA-ACCEPT + L2 rescope + T1-abs

No `src_py` production edits. Live draft 477dc8cf verified.
