# CURSOR RESULT - SAT-RERANK-01B

Date: 2026-08-17
Compared with: da9cce4 D515-ACCEPT / SAT-LIMIT product meters
(BO 7.0498 mmag / FW 10.6836 mmag, n=134, 01B formula) vs Part 3 product
SHA **36a53b0** (48 LCs). 8f107cf is quarantined; not cited for acceptance.
Push: NOT authorized.

Premise: SAT-RERANK-01 follow-up stopped at verdict (b). After
PFS-SEMANTICS-01, B3-B6 meters run on the 48-LC product. BIN-8-9 LOO is
the same estimator as D515-ACCEPT-01 part_e (proc-CSV equal-weight mean
peer), so it is comparable to the pre-registered gate; it is NOT a
re-photometry of frames.

JSON: `dev/results/SAT_RERANK_01B_meters.json`,
`dev/results/SAT_RERANK_01B_forced_meter.json`.

## 4.1 BO fifth comp (readable [COMP] file log)

UTF-8 log `tmp/draft_515_pfs_semantics_01.log` (PowerShell UTF-16 wrap
fixed by writing the file log from the harness, not `*>`).

Target BO CVn `1498613634033133184`. After RMS filter n_cand=623.

| cut | n in -> n out | best rejected | value vs threshold |
|-----|--------------:|---------------|--------------------|
| max_comp_rms ceiling | 623 -> 331 | 1498780652424659456 | rms=0.08045 vs 0.0800 mag |
| isolation >= 3.00 FWHM | 331 -> 258 | 1499202834827066624 | rms=0.006575, nn_fwhm=2.977 vs 3.00 FWHM |
| colour ladder step 1 | 258 -> 4 locked | 1500727513856914944 | rms=0.000383, d_bprp=0.782 vs lim 0.150 |

Colour ladder locked at delta_bprp<=0.150 because n=4 >= n_comp_min=3
(no widen). Clean pool after gates **n=4**. n_comp_max=8, admitted=4.

Best non-admitted by RMS: **colour cut** (not ceiling, not isolation, not
distance). Candidate 1500727513856914944 rms=0.000383 mag vs colour_lim
0.150 mag in BP-RP. "Nothing better in the pool" with numbers: clean pool
n=4, ceiling=0.0800 mag, isolation=3.00 FWHM, colour_lim=0.150.

## 4.2 Ensembles (SHA 36a53b0)

BO CVn `1498613634033133184` n_comp=4, n_eff=3.2519 (dimensionless,
Broeg-style 1/sum w_i^2 over sum w):

| catalog_id | comp_weight |
|------------|------------:|
| 1497771992240531712 | 9241.19 |
| 1499200223486564608 | 6486.89 |
| 1497974027502858240 | 4701.61 |
| 1497368849430107904 | 1886.41 |

FW CVn `1497343732462852864` n_comp=8, n_eff=6.3134:

| catalog_id | comp_weight |
|------------|------------:|
| 1497442379271632384 | 34481.93 |
| 1499906247391001088 | 38837.06 |
| 1497674651102612992 | 14252.84 |
| 1498020894186918144 | 16184.41 |
| 1498812233320666368 | 15302.07 |
| 1497370563121917952 | 14004.08 |
| 1497313255374892800 | 11751.72 |
| 1500486102335278592 | 11230.98 |

CHK_FW `1497368849430107904` sits in the BO ensemble, **not** in the FW
ensemble it meters. FW meter cell valid (not consumed). CHK_BO
`1498020894186918144` sits in FW, not in BO.

B2: 0 of 24 saturated IDs in any ensemble.

## 4.3 Fixed-meter check MAD (01B formula, 134 epochs)

Quantity: check_scatter_mad_mmag = 1.4826 * MAD(kmag) * 1000. Domain:
forced-check sidecar via `d515_accept_01b_same_meter.run_forced_check`.
SHA of archive photometry used as input: 36a53b0.

| meter | check id | MAD mmag | n | da9cce4 mmag | delta mmag |
|-------|----------|---------:|--:|-------------:|-----------:|
| BO | 1498020894186918144 | 8.5798 | 134 | 7.0498 | +1.530 |
| FW | 1497368849430107904 | 10.6836 | 134 | 10.6836 | +0.000 |

BO membership changed (C2 gone; 4 leftover comps). Supersede-with-pointer
to da9cce4 7.0498 mmag. Audit: `tmp/d515_01b_check_kmag_01B_BO.csv`.

FW ensemble IDs unchanged vs da9cce4. First FW pass used R CVn as carrier;
CHK_FW is not in FW or R CVn ensembles so the sidecar was missing (named
measurement defect, not a science number). Second pass prefers a co-target
whose ensemble already contains the check (BO CVn). MAD 10.6836156 mmag
matches da9cce4 10.6836 mmag at 134 epochs.

Production sidecars (515 check `1497613731286514432`, not the 01B meters):
BO 7.1506 mmag, FW 8.2010 mmag, n=134. Different check star; not comparable
to the 01B formula cells.

## 4.4 Full-field per-bin check MAD vs D515-ACCEPT-01 gate

Estimator: 1.4826*MAD of focus minus equal-weight mean peer inst mags
(PRE-IMPL Q2-style). Domain: proc CSVs (not rebuilt). SHA of this LOO
table: same numbers as D515-ACCEPT-01 part_e (da9cce4).

| bin mag | n | n_candidates_in_bin | median LOO mmag |
|---------|--:|--------------------:|----------------:|
| 8-9 | 15 | 27 | 11.988543441702657 |
| 9-10 | 15 | 65 | 13.687339415472659 |
| 10-11 | 15 | 103 | 16.450777056798568 |
| 11-12 | 15 | 246 | 20.70470833426409 |
| 12-13 | 13 | 506 | 33.009557272337105 |
| 13-14 | 12 | 865 | 38.89157651455142 |
| 14-15 | 11 | 1306 | 40.98479043713609 |

BIN-8-9 re-read: 11.988543441702657 mmag, n=15, gate 11.988543441702657
mmag n=15. **Byte-identical to D515-ACCEPT-01.** Verdict: **OPEN**.

SAT-LIMIT / PFS / ensemble re-rank cannot change this LOO until frames
are re-photometered. Bright-bin excess remains.

## Named defects

1. First FW 01B forced-check sidecar missing (inject path vs BO-ensemble
   check). Fixed by ensemble-aware carrier; do not invent a number for the
   failed pass.
2. BIN-8-9 unchanged because LOO is on proc CSVs. OPEN.
3. 01B tool previously stamped run_sha_of_archive_photometry as da9cce4
   even on a later product. JSON now records 36a53b0.

## Docs impact

- docs/VYVAR_ROADMAP.md -- SAT-RERANK-01 DONE with pointer; BIN-8-9 OPEN
- docs/VYVAR_STATE.md / JOURNAL.md -- meters + BIN-8-9 verdict
- FLOW: none

## Recurrence

Recurrence: existing d515_accept_01b_same_meter (carrier prefers ensemble
that already contains the check). n/a for BIN-8-9 (measurement, not a
bug-class fix).

## Files

- dev/tools/sat_rerank_01b_meters.py
- dev/tools/sat_rerank_01b_forced_meter.py
- dev/tools/d515_accept_01b_same_meter.py (carrier rule)
- src_py/comp_selection_per_target.py ([COMP] ladder/isolation lines)
- this file + JSONs
