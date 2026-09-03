CURSOR RESULT - 2026-08-19 13:45 UTC+2 (iteration 4)

What I did
Built and ran `tmp/dao_gaia_stage_01_iter4.py`: pass2 born-owned
provenance (I6), FORCED_SEED layer (I7), full rescore (I8), final
overlay gallery (I9). Config: pass1=4.5 sky ? + pass2=4.0 + I6/I7.
No src_py/config edits.

## I6 - Pass2 provenance ownership

Architecture: pass2 acceptances are **born-owned** by seed Gaia ID;
no global re-match on pass2 peaks. Dedup only within pass1 (0.75 px
spatial) and same-owner pass1+pass2 duplicates. Sanity check flags
`AMBIGUOUS_OWNER` when centroid is >1 px closer to a different Gaia
star - reported, not reassigned.

| Check | iter3 (global rematch) | **iter4 (I6)** |
|-------|----------------------|----------------|
| CROWDED_MISS (MS) | 8 | **0** |
| G<=13 true_miss holes | 10 | **0** |
| AMBIGUOUS_OWNER flags (MS) | (n/a) | **8** |

All **10 iter3 census holes** (G<=13 TOO_FAINT from match collision)
now **DETECTED** - e.g. G=11.15, 11.79, 12.59 ... all pass2-owned.

`ambiguous_owner_flags.csv`: 8 pass2 peaks on MS (33 total across
4 frames) where centroid lies >1 px closer to a neighbour Gaia star
than to the seed; ownership kept on seed per spec. Typical crowded
field pairs (catalog_ids in file).

`holes_le13_final.csv`: **empty** (zero G<=13 holes).
`holes_le13_decompose_final.csv`: EDGE 22, other 12 (non-hole
buckets; no true_miss).

## I7 - FORCED_SEED final layer

After I6 combined detection, every on-chip Gaia G<=15 still without
owner ? forced acceptance (centroid ?2 px, SNR ?4, empty-sky audited).

| Frame | pass1 | pass2 | forced_seed |
|-------|------:|------:|------------:|
| MASTERSTAR | 2636 | 217 | **163** |
| Light_001 | 2478 | 354 | 191 |
| Light_076 | 2946 | 245 | 180 |
| Light_148 | 3016 | 230 | 159 |

Empty-sky audit (`forced_seed_empty_sky_audit.json`, n=2200):
**8 accepts = 0.36%** - passes G2 ?1% gate for the seed layer.

Overlay: **cyan filled circle** = forced_seed; **green hollow** =
pass1/pass2 detection (distinct from iter3).

G1-eye with FORCED_SEED in OK set:

| Band | without seed | **with seed** |
|------|-------------|---------------|
| G<=13 (MS) | 98.8% | **100.0%** |
| G<=14.5 (MS) | 93.1% | **95.8%** |

G4 census sums on all frames (`g4_ok=True`).

## I8 - Winner rescore (p1=4.5, p2=4.0, I6+I7)

`final_scores.csv` - all 4 frames:

| Frame | G1 strict ?13 | G1 strict ?14.5 | G1-eye ?13 | G1-eye ?14.5 | G1-eye+seed ?13 | G1-eye+seed ?14.5 | G2 | G3 |
|-------|--------------:|----------------:|-----------:|-------------:|----------------:|------------------:|---:|---:|
| MASTERSTAR | 98.8% | 92.5% | 98.8% | 93.1% | **100%** | **95.8%** | 0.09% | **1.43%** |
| Light_001 | 98.7% | 90.2% | 98.7% | 91.0% | 99.9% | 94.4% | - | 3.81% |
| Light_076 | 98.7% | 94.0% | 98.7% | 94.6% | **100%** | 97.2% | - | 3.50% |
| Light_148 | 98.7% | 94.4% | 98.7% | 94.8% | **100%** | 97.4% | - | 3.83% |

G2 computed on MASTERSTAR only (empty-sky set is MS-specific).

**Gate verdict (MASTERSTAR):**
- G1-eye+seed ?13: **PASS** (100%)
- G1-eye+seed ?14.5: **SHORT** (95.8% vs ~100% target)
- G2: **PASS** (0.09%)
- G3: **PASS** (1.43% ? ~1.5%)
- G4: **PASS**

State counts (MS): DETECTED 2689, FORCED_SEED 163, BLENDED 35,
EDGE 146, TOO_FAINT 3135.

## I9 - Final overlay gallery

Path: `dev/results/context/session_20260819_daostage01_iter4/overlays/win_p1_4.5_p2_4.0_i6_i7/`

Per frame (MASTERSTAR, Light_001/076/148):
- `overlay_full.png`
- `overlay_crop_{center,mid,corner}.png` (500x500)

Legend: green hollow = pass1/pass2 detection; **cyan filled** =
forced_seed; violet = blend; orange = saturated; gray dot = TOO_FAINT
G>14; **red X = Gaia G<=14 still TOO_FAINT**.

### Red X accounting (`red_x_remaining.csv`, n=41)

**Zero red X at G<=13** - le13 band clean.

All 41 red X are **G in (13, 14]**, state TOO_FAINT, individually
audited (pass2 + forced_seed at Gaia xy):

| Reject reason | n (MS) |
|---------------|-------:|
| snr_low (SNR < 4) | 12 |
| centroid_tol (pass2 peak >2 px from seed) | 2 |

Examples (MS):
- G=13.38: pass2 centroid_tol, seed SNR=20 but centroid_tol
- G=13.57: no pass2 peak, seed SNR=0.98
- G=13.89: no pass2 peak, seed SNR=-0.82 (sub-threshold flux)

These are **genuinely below detection** at current thresholds - not
collision artifacts. They explain the 13-14.5 G1-eye+seed gap
(~4.2% of eligible ? 14 stars on MS). Red X is **not extinct** at
the G<=14.5 band but each case has a named physical reason (SNR or
centroid gate).

## Verdict

I6 **confirmed**: born-owned pass2 eliminates CROWDED_MISS and
closes all 10 iter3 census holes; 8 AMBIGUOUS_OWNER flags logged.
I7 **confirmed** for le13 (100% with seeds) and empty-sky safety;
I8 **partial** - le14.5 still 95.8% on MS due to 14 SNR-limited
G=13-14 stars, not ownership bugs. G2/G3/G4 green on MS.
I9 gallery ready for Milan visual review.

**No production config accepted.** Remaining axis for le14.5: lower
seed SNR gate (G2 risk), wider pass2 centroid tol in crowded pairs,
or accept red-X stars as TOO_FAINT with Milan sign-off.

## Files

- `tmp/dao_gaia_stage_01_iter4.py`
- `dev/results/context/session_20260819_daostage01_iter4/`
  - `final_scores.csv`, `final_config.json`
  - `ambiguous_owner_flags.csv`, `red_x_remaining.csv`
  - `forced_seed_empty_sky_audit.json`
  - `overlays/win_p1_4.5_p2_4.0_i6_i7/`
- `dev/results/CURSOR_RESULT_DAO_GAIA_STAGE_01_ITER4.md`

Push not authorized. Wall **65.2 s**.
