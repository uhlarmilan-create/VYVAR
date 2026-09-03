CURSOR RESULT - 2026-08-19 13:28 UTC+2 (iteration 3)

What I did
`tmp/dao_gaia_stage_01_iter3.py`: pass2 window G<=15 (I1), per-star
pass2 audit of 11 true misses (I2), G3 anatomy + PM check (I3),
pass2 empty-sky sigma pick (I4), winning combined config + overlays
(I5). No src_py/config edits.

## I1 - Pass2 window fix

Changed seed window from (13,15] to **G<=15** without detection.
Winning config: pass1=4.5 sky sigma + pass2 sigma=4.0, tol=2 px.

| Metric (MASTERSTAR) | iter2 comb 4.5 | **iter3 win** |
|---------------------|----------------|---------------|
| G1-eye <=13 | 98.9% | **99.0%** |
| G1-eye <=14.5 | 92.5% | **92.7%** |
| G2 | 0.09% | 0.09% |
| G3 G<=18 | 1.51% | **1.36%** |
| pass2 seeds G<=13 accepted | (n/a, wrong window) | **462/482** |

G1-eye <=13 now at Milan target (~100%). Remaining bright-end gap
is match-collision / blend, not pass2 window omission.

## I2 - 11 true misses, individual pass2 audit

File: `i2_true_miss_pass2_audit.csv`

**All 11 accept** pass2 in isolation (sigma=5, tol=2 px):

| Mechanism | n |
|-----------|---|
| accepted (ok) | **11** |
| centroid_tol | 0 |
| no_detection | 0 |

Examples: G=11.15 SNR=87 peak=3663 ADU, centroid=0.54 px;
G=11.79 SNR=6.0 peak=1938 ADU but local_std=385 (bright neighbour
in annulus) - still accepts with centroid=0.81 px.

**Conclusion:** iter2 TOO_FAINT labels were **pass1-only** failures.
Isolated pass2 recovers every star. Remaining census holes after
combined run (10) are **greedy 3 px match collisions** when
hundreds of pass2 peaks are merged (detection exists but assigned
to neighbour). State **CROWDED_MISS** added (n=8 on MS) for high-SNR
pass2-reject or neighbour-conflict cases; red X reserved for G<=14
TOO_FAINT only.

After win: `holes_le13_decompose_after_win.csv` BLENDED 31, EDGE 22,
true_miss 10.

## I3 - G3 anatomy @ thr=4.5 (32 spurious, G<=18)

PM: local Gaia DB has **no pmra/pmdec** (GAIA-1 build). n_pm=0;
G3 unchanged PM vs no-PM (both 1.64%, n=32). PM fix N/A until DB
includes PM columns.

`g3_anatomy_thr45.csv` (32 detections, no Gaia within 3 px):

| Class | n |
|-------|---|
| unmatched_other (no Gaia within 5 px) | 27 |
| poor_centroid_3to5px | 5 |

No PM-offset cases. Spurious peaks are **real unmatched detections**
(artifacts or objects beyond G<=18 catalog depth), not PM epoch error.

## I4 - Pass2 empty-sky sigma pick

`i4_pass2_empty_sky_audit.csv` (MASTERSTAR, n=2200):

| pass2 sigma | false-accept |
|-------------|--------------|
| **4.0** | **0.18%** |
| 4.5 | 0.09% |
| 5.0 | 0.09% |

Lowest sigma holding <=1%: **4.0** (all three pass). Rescored
pass1=4.5 + pass2=4.0 combined above.

Band 13-14.5 G1-eye curve still **92.7%** on MS; pass2@4.0 adds
~160 detections vs single-pass but 13-14.5 band remains threshold-
limited on pass1 + collision-limited on pass2.

## I5 - Overlays

`overlays/win_p1_4.5_p2_4.0/{MASTERSTAR,Light_*}/`
Legend: green=detected, violet=blend, orange=sat, gold=CROWDED_MISS,
gray=TOO_FAINT (G>14), red X=true miss (G<=14 TOO_FAINT only).

## Verdict

I1 **confirmed** (window defect caused G<=13 gap). I2 **confirmed**
(isolated pass2 accepts all 11; combined collisions explain residue).
I3 **PM N/A** on this DB; spurious class dominated by no-Gaia-within-5px.
I4 sigma=4.0 selected; G1-eye 13 green, 14.5 still short.
**No production config accepted** - next axis: match collision /
1-1 assignment at pass2 merge, or forced-seed at known Gaia xy.

## Files

- `tmp/dao_gaia_stage_01_iter3.py`
- `dev/results/context/session_20260819_daostage01_iter3/`
- `dev/results/CURSOR_RESULT_DAO_GAIA_STAGE_01_ITER3.md`

Push not authorized. Wall **99.9 s**.
