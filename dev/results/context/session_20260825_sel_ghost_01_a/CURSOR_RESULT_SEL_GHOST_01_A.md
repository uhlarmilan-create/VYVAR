CURSOR RESULT - 2026-08-25 (SEL-GHOST-01 Part A, measure)

What I did
Association hunt on draft 520 `g_60_4` MASTERSTAR: why 8 selected
comps are DETECTED_P1 / GAIA_MATCHED with (x, y) tens of pixels from
world2pix(Gaia). Measurement only. No wiring, no photometry rerun, no
516 writes, live 520 read-only. Push: NO.

HEAD `b1f5b8cd3ab58e27b86000720b85dd09aaa7ea25` == `origin/main`.
Session: `dev/results/session_20260825_sel_ghost_01_a/` plus Rule 0.2
copy `dev/results/context/session_20260825_sel_ghost_01_a/`.

## Premise (Rule 0.1)

**What is compared:** the 8 V0612 Cam comps on live draft 520 `g_60_4`
(`catalog_id` listed in REG-520-01 M1) versus the Gaia DR3 sky position
of those same IDs, through the MASTERSTAR match/lock path that stamped
`source_state=DETECTED_P1`. REG-520-01 measured the 59 px residual and
stopped. This task asks whether H-MATCH-WIDEN (sky-match widen +
one-shot identity gate + lock-without-distance) is the association
mechanism.

**How they differ:** a correct Gaia lock would put MASTERSTAR (x, y)
within ~3 x FWHM (3.75 px, 2.10 arcsec at this plate scale) of
world2pix(Gaia). The 8 comps sit 19-151 px away (REG census). The
pipeline match-rate statistic counts any nonempty `catalog_id` on a
retained DAO row. Those two numbers are not the same quantity; 2.5
below reports both.

## Gates

| Gate | Result | Evidence |
|------|--------|----------|
| G1 HEAD == b1f5b8c | PASS | `git rev-parse HEAD` = `origin/main` = `b1f5b8cd3ab58e27b86000720b85dd09aaa7ea25` |
| G2 live 520 SHA before == after | N/A (2.4 skipped) | live `masterstars_full_match.csv` SHA256 `5ce9b07fe0490103b2e16f6fbe3b18ffc7cd987fbee8a334722cc2fd46c6a683` (read-only; unchanged after this measure) |
| G3 516 ePSF | PASS | `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/masterstar_epsf.fits` SHA256 `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` |
| G4 `--fast` | PASS | 1530 passed, 32 skipped, OVERALL PASS at HEAD `b1f5b8c`. git-untracked WARN is pre-existing session dirt + this measure's untracked result. db-quick-check WARN waived. |

## Code verification (H1-H4 at b1f5b8c)

| Claim | File:line | Verified? |
|-------|-----------|-----------|
| `match_sep_used = max(12.0, requested_arcsec)` | `pipeline.py:9231` | yes |
| Widen x1.5 cap 90" if match rate < 0.70 | `pipeline.py:9436-9453` | yes |
| Silent 16-iter widen toward 0.95, cap 96" | `pipeline.py:9454-9469` | yes; **no per-iteration log** |
| Tighten to 4.5" only if >= 92% of pairs survive | `pipeline.py:9471-9482` | yes |
| Greedy 1:1 | `pipeline.py:9322-9428` | yes; sort key is **separation**, not brightness (architect text overstated "bright first") |
| Identity gate `post_match_pixel_sep` fail_factor 3.0 | `wcs_invertibility.py:213-237`, `pipeline.py:9486-9521` | yes |
| `_fwhm_used = max(1.2, _base_fw / bfac)` | `pipeline.py:9084` | yes; this set 2.5 / 2 = **1.25 px** |
| Gate called once, then refine loop may rematch | `pipeline.py:9521`, `:9534+`, rematch `:9667/:9704/:9750` | code true; **this-set rematch did not run** (refine `continue` at `:9580-9588`) |
| Gate try/except "skipped" | `pipeline.py:9518-9519` | yes; **not taken** on this set |
| Lock from every existing `catalog_id`; born-owned skip `lock_tol_px` | `masterstar_gaia_accounting.py:1013-1024`, `:578-586` | yes |

Additional governing path (not in H2, required by the log):
`pipeline.py:5938-5940` `_vyvar_df_to_csv` rewrites `catalog_id` via
`gaia_catalog_id.catalog_id_series_for_masterstars_export:197-208`,
which copies a Gaia-looking `name` onto `catalog_id`. The identity
gate (`wcs_invertibility.py:337-344`) clears `catalog_id` / `catalog`
on fail and does **not** clear `name`. `_assign_catalog_at_threshold`
(`pipeline.py:9399-9400`) sets `name` to the Gaia ID when matched.

## 2.1 Log forensics (g_60_4)

Source: `Archive/Drafts/draft_000520/infolog_20260824_204055.txt`.
Extract: `session_20260825_sel_ghost_01_a/log_g60_4_masterstar.csv`.
Three MASTERSTAR blocks exist (g/i/r); tables below are **g_60_4 only**
(18:44, lines 1340-1444).

| Line | Text |
|------|------|
| 432 | `WCS Refined: Mean residual error = 1.44 pixels` (platesolve, not catalog-match refine) |
| 1340 | DAO 719 pts, binning DAO=2x |
| 1341 | SNR filter: **692/719** retained before match |
| 1342 | `Catalog match: zhoda 13% < 70 %, opakovanie s max separaciou 18.00 arcsec (pozadovane 12.00 arcsec)` |
| 1343 | `post_match_identity_gate: ok=52 warn=9 fail=286 (FWHM=1.25px)` (347 at gate; 52+9+286) |
| 1344 | `[DAO] Match rate 8.8% below 88%` (61/692 after strip) |
| 1345-1347 | `Catalog match: WCS refine zamietnuty (rms=269.38px > 10)` x3 (`continue`, **no** `_run_full_match_pass`) |
| 1348 | Gaia->DAO completeness **61/24571** |
| 1352 | `MASTERSTAR: VYVAR pary (33) zlucene do katalogu` |
| 1353 | MATCH STATS (raw): 692 stars, **61 matched**, unique catalog_id **62/24571** |
| 1359 | optimizer `matched_nonempty=347/692` (`id_col=catalog_id`, `sep_col=NONE`) |
| 1361 | optimizer initial jump NN sky <=180" |
| 1364 | Grip1 SIP `n_pairs=347` `rms_lin=81.0` `rms_sip=84.7` |
| 1375 | optimizer rematch totals **+0** |
| 1383 | wrote `masterstars_full_match.csv` **(347/692 catalog-matched)** |
| 1387-1411 | second optimizer pass: still 347, rematch +0 |
| 1415 | `matched_world2pix_identity_px: n=347 p95=146.799` |
| 1418 | catalog-derived membership **+14** -> n=706 |
| 1443 | MATCH STATS (optimized): 706 stars, **361 matched** (51.13%) |

`match_sep_arcsec_effective` / `_wcs_refine_iters`: stamped in-memory
at `pipeline.py:9834-9839`, **absent** from `photometry_plan.json`,
`dao_gaia_calibration.json`, and `pipeline_meta.json`. Last **logged**
`match_sep_used` = **18.00 arcsec**. Silent 0.95 loop unlogged.
Catalog-match refine iterations entered = 3 rejects, accepted = **0**.
No `post_match_identity_gate skipped:` line.

**P-A1 TRUE.** Widen 12->18" logged; catalog-match WCS refine lines
present (all rejected).

**P-A2 TRUE** for g_60_4: exactly one `post_match_identity_gate:` line
(1343), before refine (1345). H4 skipped path **FALSE**.

H2's *mechanism* (refine rematch restores ungated IDs) is **FALSIFIED
on this set**. Completeness after reject is still 61. The 347 IDs
reappear at optimizer input two seconds later, which is the name-export
path above, not `_run_full_match_pass` after refine.

## 2.2 Ghost provenance

Authoritative store: `platesolve/g_60_4/masterstars_full_match.csv`
(706 rows, 361 nonempty `catalog_id`). Raw `masterstars.csv` **absent**
(optimizer temp, deleted). Column `match_sep_arcsec`: **absent**.
`_fwhm_used` = 1.25 px (header `VY_FWHM=2.5`, log binning DAO=2x,
`:9084`). Plate scale from final WCS = 0.5618 "/px. Fail threshold
3 x FWHM = 3.75 px = 2.11 arcsec.

d_px below is `|world2pix(Gaia) - (x,y)|` with the **final MASTERSTAR
header WCS**. `census_d_px` is REG-520-01 M1b (census `x_gaia,y_gaia`).
CSV: `provenance_ghosts_g12.csv`.

### 8 ghosts (today's selected comps)

| catalog_id | G | x | y | vy_match_mode | vy_dao_pass | source_state | source_type | amb | d_px WCS | census_d_px | sky" |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1112112413285008896 | 16.19 | 1670.53 | 1710.65 | locked | 1 | DETECTED_P1 | GAIA_MATCHED | F | 38.4 | 35.5 | 21.6 |
| 1112115024625070720 | 15.23 | 2128.49 | 778.68 | locked | 1 | DETECTED_P1 | GAIA_MATCHED | F | 55.5 | 61.0 | 31.1 |
| 1111930718988511616 | 15.32 | 3500.51 | 776.52 | locked | 1 | DETECTED_P1 | GAIA_MATCHED | F | 125.6 | 117.9 | 70.7 |
| 1112119250872867200 | 16.50 | 3100.40 | 486.79 | locked | 1 | DETECTED_P1 | GAIA_MATCHED | F | 140.8 | 152.0 | 79.3 |
| 1112110042463052928 | 16.40 | 2700.45 | 1710.38 | locked | 1 | DETECTED_P1 | GAIA_MATCHED | F | 53.1 | 56.9 | 29.9 |
| 1111931371821079552 | 15.56 | 3530.46 | 474.49 | locked | 1 | DETECTED_P1 | GAIA_MATCHED | F | 26.7 | 18.3 | 15.0 |
| 1111737823417422464 | 16.86 | 1490.34 | 1704.43 | locked | 1 | DETECTED_P1 | GAIA_MATCHED | F | 102.1 | 97.4 | 57.4 |
| 1111922300852743808 | 14.48 | 2976.70 | 1872.64 | locked | 1 | DETECTED_P1 | GAIA_MATCHED | F | 38.6 | 40.8 | 21.6 |

All 8: `name` == `catalog_id` (Gaia digit string). None are
`leftover_promotion`, `forced_seed`, or `catalog_membership`.

**P-A3 TRUE.**

### 11 G<12 (census DETECTED, G<12)

| catalog_id | G | vy_match_mode | d_px WCS | census_d_px | WCS verdict | census verdict |
|---|---|---|---|---|---|---|
| 1112113680298377344 | 7.63 | locked | 5.99 | 0.27 | fail | ok |
| 1111931371823539456 | 9.23 | locked | 16.47 | 5.54 | fail | fail |
| 1112113066119992064 | 9.98 | locked | 6.46 | 0.66 | fail | ok |
| 1111920204908702336 | 10.07 | locked | 3.91 | 1.41 | fail | ok |
| 1112110695298081664 | 10.90 | locked | 8.02 | 2.34 | fail | warn |
| 1112130898824233216 | 11.04 | locked | 9.00 | 2.07 | fail | warn |
| 1112121862213003648 | 11.10 | locked | 9.64 | 1.86 | fail | ok |
| 1111749157833870208 | 11.23 | locked | 3.37 | 0.68 | warn | ok |
| 1111754659689117952 | 11.42 | locked | 7.30 | 1.74 | fail | ok |
| 1112121067641532160 | 11.61 | locked | 8.11 | 2.22 | fail | warn |
| 1111955148762490496 | 11.93 | locked | 13.76 | 1.47 | fail | ok |

Final-header world2pix is **not** the census geometry. INV-WCS-01 already
WARN'd p95=146.8 px on this file. Census is the REG-520-01 M1b residual.

## 2.3 Gate replay on the final table

`post_match_pixel_sep` fail_factor 3.0, `_fwhm_used=1.25`, final WCS,
cone RA/Dec. Matched rows with cone coords: 361.

| | ok | warn | fail | no cone coords |
|--|----|------|------|----------------|
| fieldwide (final WCS) | 0 | 8 | 353 | 0 |

Would lose `catalog_id` (fail): **353 / 361**. Ghosts: **8/8 fail**.
G<12: **10/11 fail** (1 warn).

Same gate using census `x_gaia,y_gaia` vs MS (x,y), 3.75 px:
fieldwide 66 ok / 9 warn / 247 fail / 39 unmatched-in-census.
Ghosts **8/8 fail**. G<12: **1/11 fail** (G=9.23 at 5.54 px), 3 warn, 7 ok.

**P-A5 FALSE** as written ("0 of the 11 G<12"). Final WCS strips 10/11
G<12. Census geometry strips 1/11. Neither is zero. Ghosts strip 8/8
on both geometries.

## 2.4 Sandbox replay

**Skipped.** Log exists and is sufficient for P-A1/P-A2. Exact
`match_sep_used` after the silent 0.95 loop is unstamped; that is a
log gap, not a missing log. A MASTERSTAR rerun is not required to
answer the registered predictions. Live SHA unchanged (G2 N/A).

## 2.5 Statistic under the gate

Retained DAO rows n = 692 (log 1341). Pipeline match rate on the
**final** table: 361/692 = **0.522** (optimized MATCH STATS uses 361/706
= 0.511 including +14 catalog-derived rows).

Honest rate = pairs with d_px <= 3 x FWHM:
- final WCS: (0+8)/692 = **0.012**
- census xy: (66+9)/692 = **0.108**

At the live gate (match-time WCS): ok+warn = 61/692 = **0.088**, which
is the 8.8% the 88% warning used. The 0.95 goal was chasing 347/692 =
0.50 *before* the gate, then the export restored that 0.50 into the
optimizer.

## 2.6 Literature

Marrese et al. 2019, A&A 621 A144 (Gaia DR2 precomputed cross-match;
the Gaia DR3 documentation Ch. 15, Marrese et al. 2022,
2022gdr3.reptE..15M, keeps the same algorithm) set the initial search
radius from combined positional uncertainty and epoch/PM,
`RI = H_gamma * PosErr_L,max + PM * dEpoch`, with `H_gamma = 5`
(containment ~1-5e-7), not from a detection match-rate target. A
maximum radius appears only as a local-density / compute cap. SCAMP /
SExtractor-style matching likewise scales the pairing radius to seeing
or positional error (Bertin). I found **no published precedent** for
iteratively widening a cross-match radius until 95% of image detections
receive a catalog ID. The architect's "no precedent" framing stands.

## Predictions

| ID | Verdict | Evidence |
|----|---------|----------|
| P-A1 | **TRUE** | log 1342 widen 18"; 1345-1347 refine rejected |
| P-A2 | **TRUE** | one gate line 1343, before refine; no `skipped` |
| P-A3 | **TRUE** | 8/8 `vy_match_mode=locked` |
| P-A4 | **TRUE** | 8/8 sky sep 15.0-79.3" is > 2.11" and <= 96"; 7/8 > logged 18" (silent 0.95 loop and/or WCS change) |
| P-A5 | **FALSE** | ghosts 8/8 stripped; G<12 not 0 (10/11 final WCS, 1/11 census) |

## Decision-rule outcome

First branch (P-A1 AND P-A2 AND P-A5 -> H-MATCH-WIDEN confirmed, fix =
gate after every match pass) **does not fire**: P-A5 is FALSE.

**H-MATCH-WIDEN as a single causal story is STOPPED.** Pieces:

- **H1 widening: confirmed.** Floor 12", logged 18", silent 0.95 loop
  can reach 96" with no log. Gate-time 347 pairs at FWHM=1.25 is the
  widened 1:1 leftover pairing. Greedy is closest-first, not bright-first.
- **H2 refine-rematch: falsified on this set.** Refine rms=269 px,
  rejected, no second `_run_full_match_pass`. The gate **did** strip
  286 IDs (fail=286). They came back because `name` kept the Gaia ID
  and CSV export copied `name` onto `catalog_id` before the optimizer.
  Optimizer identity (`astrometry_optimizer.py:475-480`) only gates
  **new** writes and uses header `VY_FWHM=2.5` (7.5 px), not `_fwhm_used`.
  Rematch totals +0; Grip held the restored 347.
- **H3 lock: confirmed.** 8/8 `locked`; born-owned path does not test
  `lock_tol_px`.
- **H4 skipped: falsified.** Gate ran.

Part B preview, what would change it:
(1) INV-MATCH-IDENTITY-01 after every match pass is **necessary and
not sufficient**. A gate that strips `catalog_id` but leaves `name`,
then export restores the ID, is not a gate. Also needed: clear `name`
on fail; stop `catalog_id_series_for_masterstars_export` from
rehydrating stripped IDs; apply identity to optimizer *existing* pairs
(or refuse 347-in after 61-out). (2) persist `gaia_dao_resid_px` still
useful. (3)-(4) selection residual still required: even a working
match-time gate does not protect comps if export/optimizer restore
ghosts. (5) 0.95 match-rate must not drive the radius: no literature
precedent; this set shows the widen feeding the 347 ghosts.

## Errors

None. 2.4 not run (log present). `match_sep_arcsec_effective` not on
disk; last logged value 18.00 arcsec.

## Files changed

- `dev/results/CURSOR_RESULT_SEL_GHOST_01_A.md` (this file)
- `docs/VYVAR_STATE.md` / `docs/VYVAR_JOURNAL.md` (one-liners)
- untracked: `dev/results/session_20260825_sel_ghost_01_a/` and
  `dev/results/context/session_20260825_sel_ghost_01_a/`
- no commit, no push

## `--fast`

`python dev/scripts/session_baseline_check.py --fast` after STATE/JOURNAL
one-liners: **OVERALL PASS** (1530 passed, 32 skipped) at HEAD `b1f5b8c`.
