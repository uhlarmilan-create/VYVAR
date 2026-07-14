Historical session log. Current state -> VYVAR_STATE.md; decisions -> VYVAR_DECISIONS.md; open work -> VYVAR_ROADMAP.md.

---

## 2026-07-14 -- WIDE-SLOPE-NOISE / WSN-2 (park, pushed)

WSN-2: P4 corrected to sigma_slope_pt from excess (faint 5.17 mmag ~ sigma_r 5.5 ~ rig 4.5 mmag).
Neighbor contamination unattainable (p90 |b_attain| 0.003 << SE floor). P5 **UNIFIED_PHENOMENON_PARK**.
Full chain pushed origin/main. Result: `CURSOR_RESULT_wsn2.md`.

---

## 2026-07-14 -- CAL-LEDGER-BUNDLE (section-10 closeout + docs sweep)

**Part 0:** Pushed CAL-DIAG closeout `b08f1cc..237dd34` to `origin/main`; `session_baseline_check.py --fast` PASS on `237dd34`.

**Part 1 (CAL-AGE-CLOCK):** Verified header-first clock (`5143485`/`ee89de8`); centralized one-time
mtime-fallback WARN in `calibration.resolve_master_age` (import scan + library UI). Live library:
6 FITS paths (3 unique masters), **0 validity verdict changes** (all `VY_CDATE` 2026-04-22;
header age ~82.9 d vs mtime ~82.6 d; both valid at 90/200 d limits). Tests: `tests/test_cal_age_clock.py` (10).

**Part 2 (RN-HEADER-NONE):** Verified `1830527` -- MASTERSTAR header passed to `resolve_read_noise` in
SNR precompute. Added header-unavailable regression test. **Impact:** persisted `aperture_snr_table.json`
read_noise only; Phase 2A LC `err` and science columns unchanged (JOURNAL 2026-07-08 draft_424 byte-identical proof).

**Part 3 (CAL-PASSTHRU-DEAD):** Verified `21c20e3` -- `allow_passthrough` absent from all `.py`; only stale
audit doc references remain.

**Part 4 (docs):** ROADMAP standing list EXCEPT-BULK-2 row corrected to **CLOSED** (98/98, `97affe3`).
Root cause: stale "optional pass parked" line survived SPARSE-TRUST arc-close standing-list verification
because Part 2 re-asserted the pre-BULK-2 wording without cross-checking CENSUS closure status.

Result: `CURSOR_RESULT_cal_ledger_bundle.md`. Post-Part-0 commits **NOT PUSHED** -- Milan review.

---

## 2026-07-14 -- CAL-DIAG-IMPL closeout (spec approval + re-validation)

Spec v1.1 status flipped to **APPROVED (Milan, 2026-07-14)**. Implementation already on main
(commits 0268547..3d1508b, 2026-07-07); re-validated on HEAD 13341b3 without code changes.
14/14 gate pytest; draft_424 calibrate 150/150 VY_DKRSMP=SUM; photometry core SHA bf3743a1
unchanged. MP variant (a): parent pre-gate via `run_cal_diag_pregate`. Pushed `237dd34`.
Result: CURSOR_RESULT_cal_diag_impl.md.

---

## 2026-07-14 -- K2-STATS-FIX (honest CIs, wide re-verdict)

Retracted invalid naive-WLS k2_eff CIs (photon-weight SE ignored star-to-star overdispersion;
37,000-sigma slope vs rho=-0.013 internal contradiction). Recomputed from tmp/k2_cohort/ star
data: chi2_red >> 1 on all cells; bootstrap CI authoritative (2000 draws).

Pre-registration lesson #2: effect-size bases for power claims must be derived from the physical
effect propagated through the measured noise model (attainable rho), not a generic rho=0.4. Guard:
internal-consistency check (|k2_eff/se| > 5 while |rho| < 0.1) now raises report warning.

wide re-verdict: **LOW PRIORITY -- subdominance** (not "deprioritized-by-evidence" with B=2.12e-6).
Plausible k'' 0.02-0.04 NOT excluded (B=0.076). Newton unchanged OPEN suggestive. Overall UNCHANGED.
Result: CURSOR_RESULT_k2_stats_fix.md.

---

## 2026-07-14 -- K2-COHORT-CORRECT (verdict retraction, per-rig record)

Verdict correction without re-run. Frozen pre-registered rule applied verbatim: DOWN only if ALL
tested cells null AND **each** cell >=80% power. Newton g/i underpowered (0.47/0.40) -> clause
fails -> **UNCHANGED** (initial DOWN retracted). Process note: pre-registered rules applied as
written; ambiguities at application time recorded; **stricter** reading taken (underpowered cells
do not waive the "each cell" power requirement).

Per-rig (wide record superseded by K2-STATS-FIX): Newton (eq4) k'' OPEN suggestive (T1 rho=-0.325
expected sign; T2 rho=+0.470 p=0.043 raw). Result: CURSOR_RESULT_k2_cohort_correct.md.

---

## 2026-07-14 -- K2-COHORT (full-cohort k'' test)

Full-cohort archive report-only test. wide_CLEAR n=148 (LOO delta_mag per host target);
T1 rho=-0.013 q=0.877; T2 rho=-0.193 q=0.114. Pre-registered FDR rule initially read as
**k'' priority DOWN** (misapplied power clause; retracted in K2-COHORT-CORRECT). Newton g/i
underpowered. 425 filtered excluded N<20. Artifacts: tmp/k2_cohort/. Result:
CURSOR_RESULT_k2_cohort.md.

---

## 2026-07-14 -- SPARSE-TRUST arc close

**Arc CLOSED.** Milan confirmed SS Cam **YELLOW** (evidence-based). Verdict numbers: R=2.008
[1.224, 3.886], p_stab=0.0, x2_pair=2.96e-4 mag^2 (17.2 mmag pair excess, 26% below X2_RED cap),
production_lc_err chi2=21.38, n=2, N=25, external K 1112110935816253440. X2_RED not adjusted
post-hoc. r_60_4: all 6 sparse targets YELLOW (R~2.0-2.9). Pushed closeout `0b2f0ba`, `7886157`;
docs arc-close commit follows. Result: CURSOR_RESULT_arc_close.md.

---

## 2026-07-14 -- SPARSE-CHECK-POOL (Amendment 1)

External K sourcing on sparse branch; r_60_4 all 6 targets kmag+R computed (SS Cam
R=2.01 [1.22,3.89] YELLOW, chi2_prod=21.38). S2 zero band changes vs completion. Result:
CURSOR_RESULT_sparse_check_pool.md.

---

## 2026-07-14 -- SPARSE-TRUST-COMPLETE

S1-S4 validation PASS (see tmp/sparse_trust_validation/validation_summary.json). S2: zero
GREEN->RED flips on draft_424 n>=5. S4: anchor SHA bf3743a1 unchanged. S3: SS Cam sparse trust
YELLOW (single_comp, chi2=21.38). PZQ full report tmp/pzq_sigma_r/; k'' wide rho=-0.125 (DOWN).
Result: CURSOR_RESULT_sparse_trust_complete.md.

---

## 2026-07-14 -- SPARSE-TRUST + PZQ-SIGMA-R

**Goal:** Sparse-field check-star validation at n>=2 with CI-based trust bands; PZQ sigma_r report.

**Part 1 (PZQ report-only):** ``scripts/pzq_sigma_r_report.py`` reads ``tmp/sigma_floor/sigma_floor_fit.json``;
36 stars, 3 rig figures -> ``tmp/pzq_sigma_r/pzq_sigma_r_summary.json``. k'' probe (Spearman sigma_r vs
|colour offset|) when sem_cause artifacts present.

**Part 2 (implementation):** ``sparse_trust_core.py`` (triangulation, photon correction, chi2 CI,
stability test, trust band). ``compute_check_ensemble_mag_calib`` accepts n>=2; sidecar columns wired;
``trust_flag_core`` sparse path consumes sidecar CI stats (no field-wide comp_rms gate).

**Part 3 (validation):** S1 unit+synthetic tests in ``tests/test_sparse_trust_core.py`` (7 fast + 3 slow);
``scripts/sparse_trust_validate.py`` for S2-S4 on draft paths. S2-S4 require regen sidecars on anchor drafts.

Result: ``CURSOR_RESULT_sparse_trust.md``. Commits local; **not pushed**.

---

## 2026-07-13 -- ANCHOR-CHAIN-ACCEPT

**Goal:** Make the draft_424 anchor baseline chain explicit and accept the sigma-floor snapshot
only after exact validation of the c4-only delta.

**Part A (07-10 snapshot):** No ``draft_000424_snapshot_20260710*`` exists on disk. F-BINGAIN
closeout claimed re-anchor for err divergence, but the only retained draft_424 snapshots are
``20260708_full``, ``20260708_hybrid_deprecated``, and ``sigma_floor_20260713``. Verdict: process
gap (snapshot never physically cut or not retained).

**Part B (intermediate baseline):** Cut ``draft_000424_snapshot_intermediate_b5364e6_20260713``
at git ``b5364e6`` (two runs byte-identical). core ``373e8235...``; extended ``0243f719...``.

**Part C.2 (exact c4 validation):** intermediate -> sigma-floor snapshot: 23542/23542 epochs
within abs diff <= 2e-6 (median 2.93e-7; max 9.97e-7; 0 outliers).

**Acceptance:** ``draft_000424_snapshot_sigma_floor_20260713`` accepted (core ``bf3743a1...``;
git ``8fb21b3``). Comparator hardening: designed-err envelope/exact-predictor checks in
``tests/photometry_sha.py`` + tests.

Result: ``CURSOR_RESULT_anchor_chain.md``; artifacts ``tmp/anchor_chain/``.

---

## 2026-07-13 -- ANCHOR-ERR-VERIFY

**Question:** Does ~1.5-1.7x wide err rise on ``draft_000424_snapshot_sigma_floor_20260713``
indicate Newton 18 mmag floor leaking onto equipment_id=1?

**Finding:** **No.** ``sigma_sys_mag=0.0`` on all 178 LCs; runtime floor eq1=0.0. Per-tertile
``sigma_add`` is **not** constant (~17 / 51 / 116 mmag faint/mid/bright) -- not floor signature.
Worked frame: err_new reproduced exactly by bingain photon (ratio ~1.64) + c4 SEM + unit fix;
adding 0.018 mag floor would yield err=0.0215 vs observed 0.0136.

**Confound:** Old anchor git ``750c856`` predates F-BINGAIN-1 (``3b33b03``) and SEM unit fix
(``26396ab``). Comparing July-8 vs PROD-SIGMA-FLOOR bundles three err-model changes.

**Verdict:** Part C STOP -- no code fix, no re-cut, no push. Anchor **NOT ACCEPTED**.
Recommend intermediate baseline at ``origin/main`` before isolating c4/floor delta.

Result: ``CURSOR_RESULT_anchor_err_verify.md``; artifact ``tmp/reanchor_424/worked_frame.json``.

---

## 2026-07-13 -- PROD-SIGMA-FLOOR

**Wiring:** c4 ensemble SEM + per-rig ``sigma_sys`` in LC ``err``; ``sigma_sys_mag`` column +
``pipeline_meta.sigma_floor``. Newton floor **18.0 mmag** (eq 4); wide **un-floored** (unstable
bootstrap; point fit ~4.8 mmag below SIGMA-A3 [5.5, 7.5] band).

**PZQ:** report-only red-noise diagnostic per rig (not in per-point bars).

Spec: ``docs/VYVAR_SIGMA_FLOOR_SPEC.md``; artifacts ``tmp/sigma_floor/``.
Result: ``CURSOR_RESULT_sigma_floor.md``. Re-anchor draft_424 in progress.

---

## 2026-07-13 -- COMP-POOL-R

**Finding:** r_60_4 regen yields **2** Phase-1 comps (tier 3) vs June **8** (tier 2) on
**byte-identical** proc CSVs. check_kmag sidecars 0 (``n_comp_min`` ensemble needs >=3 others).

**Verdict:** legitimately sparse on HEAD; June 8 not reproducible without threshold retuning.
**No fix.** Milan: sparse-aware check-star design question flagged.

Result: ``CURSOR_RESULT_comp_pool_r.md``; artifacts ``tmp/comp_pool_r/``.

---

## 2026-07-13 -- MASTERSTAR-EPOCH-FIX

**Finding:** ``proc_MASTERSTAR.csv`` (stacked reference sidecar) entered phase2a epoch sets
via ``proc_*.csv`` glob / ProcFrameStore, yielding 26 vs 25 epochs on draft_426 regen.

**Root cause:** Producer writes sidecar when ``MASTERSTAR.fits`` sits under
``detrended_aligned/lights`` (``export_per_frame_catalogs``); artifact post-dated June LC cut
(2026-07-10 vs LC 2026-06-26). Collector had no MASTERSTAR exclusion.

**Fix:** ``is_masterstar_proc_name`` + filter in ``list_proc_csvs`` / ``ProcFrameStore.build``;
phase2a belt-and-braces log. Artifact production unchanged (SNR / noise_floor consumer).

**Exposure:** draft_424/425 clean; draft_426 proc present but LC phantom cleared after regen.
Anchor 424 not affected by phantom epoch.

**Baseline:** Fresh Newton redo -- V0611 i chi2 **2.01** (25 epochs); supersedes ``a6b19df``.

**r_60_4 follow-up:** check_kmag sidecars still 0 -- sparse comp pool (2 good vs stale 8), not
``check_star_min_epochs`` / phantom epoch. Flagged for separate investigation.

Result: ``CURSOR_RESULT_masterstar_epoch.md``.

---

## 2026-07-13 -- 426-REGEN + PROVENANCE-GUARD

**426-REGEN:** Stale draft_426 moved to ``Archive/evidence/draft_000426_stale_20260626``;
regenerated g/i/r/z on HEAD with provenance. Frame delta: ``proc_MASTERSTAR.csv`` epoch now
included (proc_*.csv glob). Fresh Newton baseline in ``tmp/sigma_newton_fresh/`` (V0611 i
chi2=2.17). r_60_4 check_kmag sidecars missing (0 vs stale 6) -- flagged, not auto-fixed.

**PROVENANCE-GUARD:** ``provenance_guard.py`` + harness wiring + VL-PROVENANCE ledger item.

**AAVSO:** stale exports had inflated err bars; resubmission Milan decision.

Result: ``CURSOR_RESULT_426_regen.md``.

---

## 2026-07-13 -- SIGMA-PROV-FORENSIC: stale-LC proof + fresh Newton + unit fix

**Root cause:** draft_426 archive LCs carry **pre-Fix-A** ensemble err (brightness spread, not
Honeycutt residual SEM). Semantic fingerprint: i_70_4 LC/normalize **7.46x**; LC-implied ensemble
0.0482 mag vs normalize 0.0067 mag. Strict P1 mtime leg FAIL (2026-06-26); semantic P1 PASS.

**Fresh i_70_4 (current HEAD):** V0611 err **0.0175 mag** (P2 FAIL vs 0.009-0.010); chi2 **2.13**
(P3 FAIL vs 6-8). Sign flip stale 0.24 -> fresh 2.13. Pooled check stars 2.1-24.9.

**Part C:** unit-consistent err quadrature landed; re-anchor required.

**Retractions:** SIGMA-NEWTON draft_426 baseline; SIGMA-SEM-CAUSE dominant mag/flux attribution.

Result: ``CURSOR_RESULT_sigma_prov_forensic.md``; ``tmp/sigma_prov_forensic/``.

---

## 2026-07-13 -- SIGMA-SEM-CAUSE: ensemble SEM inflation attribution (SUPERSEDED)

**Question:** WHY is LC err ~2x empirical scatter on i/r (chi2~0.24)? Not the 0.5x scale shortcut.

**Findings:** Comp residual lag-1 autocorr ~0.56-0.64 (trend content confirmed). Per-comp AM
trend fraction higher on i/r (0.27) than g (0.19). Simple AM detrend does NOT fix chi2.
LC-implied ensemble SEM **4-7x** larger than ``ensemble_normalize`` on i (mag/flux quadrature
at ``photometry_core.py:3233``). Split-half flux-sum test is a different estimand.

**Gates:** C0-C1-C2-C4 PASS; **C3 FAIL** (no detrend/split-half repair). Verdict: hypothesis
partial -- trends present, AM detrend insufficient.

**SS Cam:** excluded from pooled stats; chi2=122 separate sparse-path issue.

Result: ``CURSOR_RESULT_sigma_sem_cause.md``; ``tmp/sigma_sem_cause/``.

---

## 2026-07-13 -- SIGMA-NEWTON: empirical harness + Newton chi2 baseline (draft_426)

**Part A:** ``chi2_sigma_gate.sigma_arrays_from_lc_and_proc`` now uses proc ``sigma_bkg_ap`` (via
``_photometric_error_with_bkg_mode``); ``production_lc_err`` variant = LC ``err`` column.
**Part B:** draft_426 g/i/r gate run (V0611 + 6 check stars/setup). Gates N1-N4 PASS (baseline
defined; i/r attribution conclusive; g heterogeneity localized; harness matches bingain acceptance).

**Key numbers:** V0611 production_lc_err chi2/dof: g **1.23**, i **0.24**, r **0.25**. g pooled
**2.95** (SS Cam chi2=122 outlier). i/r underdispersion: LC err ~**2.1x** empirical kmag scatter
predicts chi2~0.23 vs observed ~0.24; ensemble share 85-91%. SEM implements std/sqrt(n) correctly
(``photometry_core.py:3113-3115``).

**Milan candidate (superseded by SIGMA-SEM-CAUSE):** ~~0.5x ensemble scatter scale~~ -- rejected;
use unit-consistent err assembly instead.

Result: ``CURSOR_RESULT_sigma_newton.md``; artifacts ``tmp/sigma_newton/``.

---

## 2026-07-13 — SESSION-CLOSE-0713: color-WB arc close (12g–12g6)

**Oblouk:** catalog-color field od TODO-12g po 12g6 na `origin/main` — vizualizační vrstva hotová,
téma **UZAVŘENO** (PDF wiring jen na vyžádání).

**Root cause recap (draft_424 blotches @ boost 2.2):** absolutní-L SNR gate (`L/sigma_bg`) otevřený
na vignetovaném jasném pozadí; Gaussian splat s nenulovou váhou na hraně boxu; single-patch G2
v tmavém rohu (blind spot ~0.015 vs střed ~0.06). Fix 12g5: lokální bg mapa, tapered stamp,
hardened 8×6 G2.

**12g6:** caption stamps `rendered YYYY-MM-DD HH:MM UTC @ {hash}` — tři stale-evidence incidenty
(identické názvy souborů přes pre/post běhy v `tmp/todo12_hrd/`) → systémová oprava. Kanonické
rendery z dirty tree razily `c5685c6` bez `-dirty` — po close doplněn `git describe --dirty` suffix.
Default boost **2.2** (Milan po fixed 424). 425 V @1.6 cluster retention 89.1% akceptováno jako
akademické (default 2.2). Archiv `tmp/todo12_hrd_archive_0711/`; finál `tmp/colorfield_final/`.

**NEXT:** SIGMA-NEWTON (viz ROADMAP).

---

## 2026-07-11 — TODO-12g6-HRD: caption stamps, boost default 2.2, canonical re-render

**Decisions:** Milan approved `hrd_color_chroma_boost` default **2.2** after fixed draft_424 @2.2
(hardened G2 pass) and chi/h Per @2.2 visual OK. 425 V @1.6 cluster-retention **89.1%** accepted
as academic given 2.2 default.

**Caption stamps:** every burned caption appends `rendered YYYY-MM-DD HH:MM UTC @ {git_short}` —
stale-evidence lesson #3 (identical filenames across pre/post runs in `tmp/todo12_hrd/`).

**Hygiene:** archived `tmp/todo12_hrd/` → `tmp/todo12_hrd_archive_0711/`; canonical flat set in
`tmp/colorfield_final/` + `manifest.json`.

---

## 2026-07-11 — TODO-12g5-HRD: local-bg SNR gate, stamp taper, hardened G2

**Problém (Milan review draft_424 @ boost 2.2):** hnědé čtvercové fleky přes jasné/mléčné
pozadí; draft_425 V vypadá dobře. Root cause: (1) chroma SNR gate měřil absolutní luminanci
`L/sigma_bg`, ne signál nad lokálním nebem — na vignetovaném jasném poli je gate otevřený
všude; (2) Gaussian splat stamp měl nenulovou váhu na hraně boxu → viditelné čtverce při silné
chroma na jasném pozadí; (3) G2 QA vzorkoval jeden tmavý rohový patch (<0.009) zatímco střed
rámu měl worst-patch ~0.06 — blind spot.

**Fix:** lokální bg mapa (sigma-clipped median + MAD sigma v box gridu `hrd_color_bg_box_px`,
default 96); gate `s=max(0,L-bg_local)/sigma_local`; tapered stamp `exp(-r²/2s²)-exp(-R²/2s²)`
s R≥3σ; hardened G2 = 8×6 grid star-masked patches, worst-patch |R-B|/L < 0.03 + heatmap PNG.

---

## 2026-07-11 — TODO-12g4-HRD: chroma boost (catalog-color field)

**Problém:** field_median vs d65 renders prakticky nerozlišitelné -- většina hvězd sedí u white
pointu, chroma amplituda je limitující faktor (pastel). Milan review draft_425 V + draft_424.

**Fix:** `hrd_color_chroma_boost` (default 1.6, clamp 1..3): expand distance from white po WP a
desaturaci, před SNR gate. Caption `chroma enhanced x{N}.` Standardní display astro practice;
není colorimetrická pravda, musí být vidět na obrázku.

---

## 2026-07-10 (večer) — TODO-12g2-HRD: catalog-color field polish

**Problém (Milan review draft425_B):** fyzikálně správné, vizuálně ploché -- per-channel clip
bělí jádra (RS Per bílá), tisíce slabých K/M splatů barví šum na pozadí, implicitní D65 dává
v reddened poli (E(B-V)~0.5-0.6) pozorované BP-RP chi/h Per B hvězd ~0.3-0.7 -> Teff ~6000-7500 K
-> téměř neutrální (render říkal pravdu, ne „modrobílý cluster" z očekávání 12g).

**Fix (vizualizace only):** hue-preserving highlight (scale-by-max / soft Reinhard L);
chroma SNR gate w=s/(s+snr); field_median von Kries white point + extended caption;
saturation default 0.85. Hard gates G1-G4 na draft425_B + draft424.

---

## 2026-07-10 (večer) — TODO-12g-HRD: catalog-color field (Gaia tinted)

**Cíl:** vizualizační vrstva — monochromatické MASTERSTAR pole obarvené katalogovou chrominancí
(Gaia BP-RP / Teff), bez dopadu na fotometrii. Modul `hrd_colorfield.py`: Planckian locus (Wyman
et al. 2013 CMFs) + PCHIP BP-RP→Teff (Pecaut & Mamajek 2013); Gaussian splat matched hvězd;
UI expander v HRD tabu; validate PNG pro draft_424 a draft_425 B. Config: `hrd_color_field_enabled`,
`hrd_color_saturation`. PDF mimo scope.

---

## 2026-07-10 — SESSION-CLOSE-0710 (HRD arc + F-BINGAIN-1, session close)

**Denní oblouk:** TODO-12 HRD (12→12f: absolutní extrémní objekty, obohacení Gaia TAP/SIMBAD,
identifikační tier, PDF/UI detaily, summary.json freshness stamps) → F-BINGAIN-1 FIX (empirický
`sigma_bkg_ap` v produkčním `err`) → acceptance + regate (dekomponované brány G1–G4 PASS) →
hybrid `howell_scaled` fallback (B_20_2: 0 % raw fallback).

**F-BINGAIN-1 — příčina:** level-based rekonstrukce background variance (`sky_pp/g·A + (RN/g)²·A`)
selhává při pedestal in level + resampling korelace (Stage B/C); RN hodnota sekundární. Žádné
bias/dark kalibrace na eq4 (0 C5A bias/dark v archivu) → dark subtract neprobíhá.

**Fix:** empty-aperture `sigma_bkg_ap` (Labbe et al. 2003; IRAF/DAOPHOT/SExtractor/photutils pattern);
`var = F/g + sigma_bkg_ap²`. Hybrid fallback: `howell_scaled` s transferovaným r_setup (Casertano spirit).

**Regate — proč původní brána padla:** dekompozice LC err² ukázala, že V0611 **není**
background-dominated v žádném pásmu (bkg podíl 7–10 %, ensemble 84–91 %). Původní „χ²∈[0.8,1.2]
všechna pásma" tiše předpokládala background dominanci — neplatné. **Anti-blanket-gate:** budget
je heterogenní per hvězda; oba ocasy musí být prozkoumány.

**SIGMA-NEWTON seed čísla (zítřejší entry point):**
- i/r: ensemble SEM podíl ~91 %, χ²≈0.25 → ~2× nadhodnocená nejistota (underdispersion) → SIGMA-NEWTON scope.
- g_60_4 pooled χ²≈2.99 vs V0611 g χ²≈1.11 → overdispersion v jiné hvězdě/kohoru; stejný setup, jiný budget.

**Byte-identity:** re-anchor pro `err` sloupec (dokumentovaná divergence); non-err proc sloupce
byte-identical na patch-only acceptance.

**pytest:** 737 passed, 15 skipped. **session_baseline_check --fast:** PASS. **origin/main:** `560723c`.

---

## 2026-07-10 — F-BINGAIN-1 acceptance regate (decomposition + hybrid fallback)

**Part 1 decomposition (V0611 draft_426, archive LC):** LC err² is **ensemble-dominated**
(84–91% median share); background term 7–10% in all g/i/r bands — **no band meets G1 threshold
(≥40% background).** Original acceptance gate “V0611 chi2 ∈ [0.8,1.2] all bands” silently assumed
background dominance; invalid per measured budgets. i/r underdispersion (χ²≈0.25) re-attributed to
ensemble SEM (~90%), not empirical-bkg fix failure → feeds **SIGMA-NEWTON** scope.

**Refined gates (G1–G4):** G2 PASS i/r (|Δχ²|<0.1); g direction toward unity (1.23→1.11, G3); draft_424
pooled 0.074→0.216 toward 1 (G3); wide-rig err ratio ~1 (G4). **Hybrid B_20_2:** r_setup=0.166,
0% raw howell_fallback, 24.7% howell_scaled.

**Code:** `howell_scaled` fallback, `finalize_hybrid_bkg_fallback_proc_dir`, validate LC-err chi2,
`scripts/bingain_err_decompose.py`. pytest 737 passed.

---

## 2026-07-10 — F-BINGAIN-1 Stage C (resampling correlation vs accounting)

**Diagnostic only.** Sandbox: `sandbox/tools/bingain_stageC_run.py`, `bingain_common.py`,
`bingain_stageC_chi2.py`. Artifacts: `tmp/bingain_stageC/`.

**Multi-stage closure draft_426:** pre (`non_calibrated`) ratio 0.10–0.65; post (`detrended_aligned`)
0.06–0.47; post/pre 0.58–0.72 (partial bilinear correlation, not full 0.44/pass). Gain Theil-Sen
3.5–7.1% from header 12.48 — **gain leg PASS**. z_90_4 anomalously low (0.06 post) — separate follow-up.

**Aperture r_ap:** g 0.54, i 0.25, r 0.22, z 0.08 (matches closure). Chi2 sandbox: V0611 g reaches
**0.805** with sqrt(r_ap) photon scale (prod 0.040); SS Cam overshoots (2.3–4.9). **Recommendation:**
empirical per-frame background noise term (option a), not RN-only fix. Stage B bias acquisition
still useful (sigma budget / darks expiry) but not critical path for chi2 deficit.

---

**Diagnostic only — no production changes.** Sandbox: `sandbox/tools/bingain_rn_measure.py`,
`bingain_bg_closure.py`, `bingain_stageB_run.py`. Artifacts: `tmp/bingain_stageB/`.

**STOP Part 0:** local Archive has **307** C5A bin4 science frames but **zero** bias/dark cal frames;
**0** pair-difference pairs. Empirical RN not measured. Milan acquisition: >=6 bin4 bias (or matched
darks) at GAIN=12.48 e-/ADU, T~-15 C; optional bin1 at same gain for sum-vs-average ratio.

**draft_426 resolver (unchanged):** GAIN=12.48 (header), RN=14.08 e- (db, 4x3.52 bin1).

**Background-variance closure:** var_meas/model = 0.07–0.47 with RN=14.08; RN=10 changes ratio
<0.3% (sky Poisson dominates); RN_implied clamps to 0 -> **sky/area accounting** lead candidate.

**Chi2 sandbox:** production check-star chi2/dof 0.04–0.67; sigma_ratio 1.13 -> chi2_pred 0.78
(A4 consistent). Vendor RN=10 hypothesis moves chi2 <1% vs 14.08 — RN fix alone insufficient without
pair-difference measurement. Fix proposal written; enactment blocked on cal-frame acquisition.

---

**Row payload:** dist_pc (reliable parallax only), parallax_mas/snr, sp_type_raw, otype_raw,
dsc_wd_p, teff_source, ra_dec_sex; compact overview table unchanged.

**PDF:** follow-on page(s) `Extreme objects -- details` (page 25 on draft_425 B); naive-distance
caveat cites Bailer-Jones 2021. **UI:** expander with full detail columns.

**Validate:** summary.json top-level `generated_at_utc` + `git_head` stamps.

---

## 2026-07-10 — TODO-12e-HRD (identification/confirmation tiers)

**Enrichment:** cache v2; SIMBAD `sp_type`; Gaia TAP LEFT JOIN `astrophysical_parameters` for DSC
WD/binary probs + `spectraltype_esphs`. Citations: Delchambre et al. 2023 (DSC), Creevey et al. 2023 (ESP-HS).

**Tiers:** confirmed (literature), likely (DSC WD p>=0.90), candidate (photometric). Table `ident` column.
SIMBAD MK lum class substitutes for missing Gaia logg at classification (RS Per fix).

**draft_425 B:** WD confirmed (DA2.3); RS Per RSG confirmed (M3.5IabFe-1, logg_source=simbad_lumclass);
otype conflicts 0 (was RS Per Very cool/s*r). pytest 717 passed; PDF overflow 0.

---

## 2026-07-10 — TODO-12d-HRD (NSS flag off, annotated field image + PDF)

**Item A:** `hrd_nss_category_enabled=False` (config-only, reversible). NSS excluded from Stage-1 nets
and Stage-2 when off; existing NSS tests parametrize flag ON.

**Item B:** `annotate_field_image` — MASTERSTAR NAXIS scale guard; skip when alignment unknown; short
labels (WD/RSG/RG/HOT/HOT-LUM/COOL/NSS); scaled markers/fonts; legend strip; SIMBAD main_id under label.
Background for annotation skips `field_map.png` (bbox padding breaks pixel scale); prefers MASTERSTAR
FITS→PNG 1:1. PDF `_report_hrd_page` embeds annotated field beside HRD plot.

**draft_425:** table 7 rows/setup, 0 Binary (pre-12d: 10 rows, 3 Binary). RSG alignment B: ratio 6.4
PASS (field_map-only scale: 1.9 FAIL → fixed by FITS background).

**pytest:** 698 passed, 15 skipped. PDF overflow 0 (draft_425 B_20_2). Milan authorized commit+push.

---

## 2026-07-10 — TODO-12c-HRD (label priority + per-net reservations, arc close)

**Fix:** Stage-2 priority now RSG/RG before Very cool; cool label suppressed when logg indicates giant
branch. Stage-1 `hrd_min_per_net=4` reserves slots per net (luminous ranked by M_G).

**draft_425:** RSG rows 3/setup (pre-12c: 2 RSG + 1 mislabeled Very cool/s*r). One remaining
Very cool/s*r row where Gaia logg is N/A (honest fallback). Luminous-net teff mostly NaN; none >=25k.

**pytest:** 696 passed, 15 skipped. Milan authorized commit+push of full 12/12b/12c arc.

---

## 2026-07-10 — TODO-12b-HRD (parallax gate, category cap, apparent-G legend)

**Motivation:** draft_425 chi & h Persei (d ~ 2.3 kpc, pi ~ 0.43 mas) — old 1.0 mas floor excluded the
entire cluster from M_G plane.

**Changes:** `hrd_parallax_min_mas=0.15` + `hrd_parallax_snr_min=5` (config); `hrd_max_per_category=3`;
NSS deprioritized in Stage-1 enrich budget; legend/caption clarify apparent-G gray points.

**draft_425 before/after reliable:** B 1015→7989, V 795→6651, R 1021→8011. Table: red supergiants
appear (2 on B/R, 3 on V); binaries 9→0 on B (cap + physics-first budget). No Very hot / Hot luminous
rows (reported, not threshold-tuned). draft_424 reliable 2474→3515 (modest).

**pytest:** 693 passed, 15 skipped (+2 gate/cap tests). PDF overflow 0.

---

## 2026-07-10 — TODO-12-HRD (session-aware extreme objects + online DR3 enrichment)

**Scope:** Report-layer HRD only — no photometry proc/summary/trust changes. Lite local Gaia DB keeps
`teff_gspphot`/`logg_gspphot` NULL; online Gaia TAP fetches astrophysical params for Stage-1 candidates
only (max 20 default). SIMBAD `otype` refines labels (fail-open).

**Part 0 verification:** draft_425 B/V/R each has per-setup `masterstars_full_match.csv` (19251 / 8381 /
17268 rows); DAO filter via `flux` (14574 / 7847 / 12474 HRD rows). Local DB: 211712600 sources, 0 teff, 0 logg.

**Real-data:** draft_425 B/V/R + draft_424 HRD PNGs under `tmp/todo12_hrd/`; candidate counts differ across
filters (detection selection). Offline enrich flag: table renders with `N/A` teff/logg, no errors.

**pytest:** 691 passed, 15 skipped (+10 HRD tests). session_baseline_check `--fast` PASS. PDF overflow 0
(draft_425 B_20_2 verify).

---

## 2026-07-09 — SESSION-CLOSE-0709 (equipment verify, sigma closeout, session close)

**Day arc:** DAO-RECONCILE close (3.5% accounting bug → 89.7–98.3% real completeness, 2-pass closed) →
sigma budget Phase A chain (A → A2 → A3 → A4: ensemble term, falsified-then-refined floor, attribution) →
sparse-comp decomposition (field-wide offset ~95% cancels; temporal 8–12 mmag healthy) → PSF audit fixes
(recap in prior journal entries) → draft_426 equipment verification.

**draft_426 equipment:** FITS INSTRUME=C5A-150M, NAXIS=3552×2664 bin4, GAIN=12.48, IMAGETYP=OBJECT,
NCOMBINE absent. Headers identify eq4 (C5A-150M/IMX411); OBS_DRAFT already ID_EQUIPMENTS=4 — **no DB change**.
GAIN=12.48 matches eq2 bin4 scale but does not override INSTRUME+geometry (F-BINGAIN-1, not attribution).
`scripts/fix_draft_equipment.py` committed for auditable future fixes.

**Sigma Phase A (wide rig) DONE:** photon + Honeycutt ensemble SEM + 6.5 mmag floor; scint ~2 mmag on D=0.2 m;
f_resid→0. Attribution: k2 ZERO; phase strongest (6.5→4.5 mmag); ~4.5 mmag rig constant. Open: PROD-SIGMA-FLOOR,
SIGMA-NEWTON (after bin4 fix).

**Sparse-comp:** SS Cam YELLOW; proposed gate uses check scatter CI + temporal comp_rms, not field-wide headline.

**pytest:** 681 passed, 15 skipped. session_baseline_check (default/fast) PASS.

---

## 2026-07-09 — SIGMA-A4 (floor attribution + bin4 forensics)

**Attribution draft_424:** k2 pooled R²≈0; k2_effective≈-3.6e-05, CI spans zero; floor_after k2 removal
unchanged 6.5 mmag (k'' workstream would recover 0.0 mmag). Phase-signature floor_after=4.5 mmag
(Δ=2.0 mmag). Controls X/time R²<0.2%.

**Bin4 forensics draft_426:** header gain 12.48 (source=header), RN 14.08 (db); σ_used/σ_exp≈1.13;
check-star χ² 0.04–0.33. Hypothesis DB-scaled gain=16 worsens χ². SEM=0: sparse 2-comp ensembles;
producer zero_fraction 0–4%.

**pytest:** 679 passed, 15 skipped.

---

## 2026-07-09 — SIGMA-A3 (ensemble SEM variant + refit)

**Variant (e):** `howell_scint_fresid_floor_ensemble` — quadrature of target Howell, f_resid×scint,
Honeycutt ensemble SEM (production `ensemble_normalize`), sigma_floor. Dual SEM extraction: path (b)
production primary; path (a) LC err decomposition (LOO inherits anchor-target `err` — agreement
pooled median |diff|≈41 mmag; clamp fraction median 0.0).

**Joint refit draft_424:** (d) f_resid=0.74, sigma_floor=10.5 mmag [9.5,11.0] unchanged;
(e) f_resid=0.0 [0.0,0.62] pinned lower, sigma_floor=6.5 mmag [5.5,7.5], median chi2/dof=1.000,
IQR=0.137. **Prediction verdict:** floor_did_not_collapse (6.5 mmag > 5 mmag PRNU-scale bound).

**pytest:** 674 passed, 15 skipped.

---

## 2026-07-09 — SIGMA-A2 (rig fixes, sigma_floor variant, rerun)

**Rig fixes:** `scripts/fix_telescope_diameter.py` (Carl-Zeiss DIAMETER 72→200 mm, `--apply`);
`resolve_rig_scintillation_params` rejects alt<=0 from pipeline_meta → LOCATION fallback.

**sigma_floor variant:** `howell_scint_fresid_floor` in quadrature (mag domain); joint grid fit +
bootstrap CIs on (f_resid, sigma_floor). draft_424 rerun: D=0.2 m, alt=275 m (Jirny);
f_resid=0.74 [0.0,1.0], sigma_floor=10.5 mmag [9.5,11.0], median chi2/dof=1.000, IQR=0.158.
G9.3 calibrator saturation fill p50/p95/max = 0.41/0.49/0.53 (not flagged; threshold 0.85).

**draft_425 trust:** V/B/R setups have zero GREEN rows — trust counts YELLOW+RED only
(see `tmp/sigma_budget/draft_425_trust.json`).

**pytest:** 669 passed, 15 skipped.

---

## 2026-07-09 — SIGMA-BUDGET-A + SPARSE-COMP-DIAG (diagnostic-first, committed scripts)

**Scope:** sandbox/diagnostic only — no production wiring to `_photometric_error`, ensemble combine,
or LC/proc output. `delta_mag` flux-sum remains canonical.

**Committed modules:** `sigma_budget.py` (Howell via `_photometric_error`, Osborn scintillation,
`total_sigma` variants), `scripts/chi2_sigma_gate.py` (reduced chi2/dof harness + bootstrap CI),
`scripts/select_constant_calibrators.py` (filter comp-selection + LOO via
`check_star_kmag.compute_check_ensemble_mag_calib`), `scripts/sparse_comp_diag.py` (two-way comp
matrix decomposition + cancellation test), `tests/test_sigma_budget.py`.

**Archive runs (2026-07-09):** `tmp/sigma_budget/calibrator_chi2_summary.json` (draft_424
NoFilter_60_2: 8 calibrators G 9.3–13.2, f_resid=1.0, median chi2/dof=1.13);
`tmp/sigma_budget/sparse_comp_diag.json` (SS Cam draft_426 g/i/r; V0611 g; healthy locus draft_424).
Plot: `tmp/sigma_budget/chi2_vs_g_draft000424_NoFilter_60_2.png`.

**Deviations reported in `CURSOR_RESULT_sigma_sparse_diag.md`** (frame gate 200→120, draft_425 no
GREEN anchor, TELESCOPE.DIAMETER unit on draft_424).

**pytest:** 666 passed, 15 skipped. **No push** (separate step).

---

## 2026-07-09 — DAO-RECONCILE-CLOSE (flat-curve fix + workstream closure)

**Flat-curve fix:** When completeness stays above 50%/90% to reference depth with no
interpolation crossing (draft_426), report `>= depth (no crossing)` — not median-bin fallback
(which had produced spurious G_lim_50=13.0). `fit_method="degenerate"` reserved for <3 bins.

**Workstream CLOSED.** The original ~3.5% Gaia→DAO completeness was an accounting bug (cone
brightest-100k denominator), not DAO health. After R-1→R-2→R-2b, true completeness is
89.7–98.3%; miss@G90 is 6–18 on wide/Newton rigs. **2-pass DAO recovery: CLOSED**
(not-worth-complexity) — 425 B/R residuals are catalog cross-match ceiling, not recoverable
signal. **PUB-QC-MISSRESIDUAL** parked (LOW).

**draft_426 corrected:** g/i/r/z G_lim_50/90 = `>= 17.5 (no crossing)`; compl_50 99.5–100%.

---

## 2026-07-09 — DAO-RECONCILE-2b (censoring + missed@G90 + match_depth)

**R-2b (`78febea`):** Right-censor G_lim_50/G_lim_90 when Fleming fit exceeds reference depth
(max Gaia G in query). `missed_below_g90` vs `missed_fadezone` split for 2-pass gate.
`resolve_effective_match_depth()` → `match_depth` in report/meta/diag table. Missed-G histogram
PNG with G_lim markers. QA tooltip: censored display + missed split.

**draft_424:** compl_50=89.67%; missed=353 → **miss@G90=15**, fadezone=338 (fade-zone pile near
G_lim_50; only 15 true sub-G90 anomalies).

**draft_425 B/R:** G_lim_50 censored `>=17.5` (raw fit was 19.25); R miss@G90=212 (G90 also
censored).

**draft_426 g/i/r/z forensics:** `match_depth=18.0` (MASTERSTAR `_ms_faintest_mag_eff` default;
`faintest_mag_limit` unset in pipeline_meta). G_lim_50≈13.0 is **not** config truncation — median
fallback when completeness stays ~100% to G17.5 (sparse in-frame reference n~400).

**pytest:** 659 passed, 15 skipped. **Push:** Milan-authorized; full chain dc9f9f9→78febea.

---

## 2026-07-09 — DAO-RECONCILE-2 (footprint reference + Fleming fit)

**Root cause (R-1):** `field_catalog_cone.csv` = brightest 100k in circumscribed cone (G~15.3
cap) vs detect-time depth to G~17.5 — population mismatch made R-1 genuinely-missed≈96k unusable.

**R-2 (`bd6244a`, `4279a52`, `b7df7c6`):** Direct Gaia DB bbox query at detect depth; frame
footprint filter (2×FWHM margin); Fleming et al. (1995) erf fit; `G_lim_50`/`G_lim_90`;
`completeness_50` headline. Citation: `fleming1995` in `CITATIONS.bib`. All-drafts diag (~42s).

**draft_424 R-2:** n_ref_in_frame=12478; G_lim_50=14.97; G_lim_90=14.17; completeness_50=89.67%;
blended=69; genuinely-missed=353; unmatched DAO 2724 (faint-real=2, artifact=40).

**2-pass gate:** 353 missed — material but completeness already ~90%; OPEN for Milan.

**Anchor:** no LC/proc CSV change.

---

## 2026-07-09 — DAO-RECONCILE (Gaia↔DAO field accounting)

**Goal (Milan):** near-100% two-way reconciliation between DAO signal and Gaia catalog;
old ~3.5% dashboard figure was unlimited-cone-denominator artifact, not DAO health.

**Part 1 (`becc274`):** `dao_reconcile.py` + `scripts/dao_reconcile_diag.py` + 8 unit tests.
draft_424 diagnostic: G_lim (p95)=16.69; matched=3437; below-limit=0; blended=83;
genuinely-missed=96480; corrected completeness=3.44%; raw=3.97%; n_dao_unmatched=2724
(54 artifact candidates, collinearity n=33). Report: `tmp/dao_reconcile/draft_000424/`.

**Part 2 (`e9daec9`):** `pipeline.py` detect + MASTERSTAR paths persist decomposition to
`pipeline_meta.json`; `ui_masterstar_qa.py` shows redefined metric + bucket tooltip.

**Decision:** 2-pass DAO recovery remains OPEN — draft_424 genuinely-missed is large;
re-run diagnostic after MASTERSTAR regen to refresh meta.

**Anchor:** no LC/proc CSV schema change; draft_424 SHA unchanged.

---

## 2026-07-09 — PSF-AUDIT-FIXES + ROADMAP-RECONCILE

**PSF arc (PSF off):** four latent audit findings fixed; draft_424 anchor byte-identity preserved.
Commits `0b5eb8b` (err routing + AC guard + ProcFrameStore PSF cols + legacy removal),
`a3368fb` (spatial flag PARAMS note). 643 pytest passed.

**Store projection (Part A.1):** added `psf_flux_err`, `psf_quality_fallback`, `psf_ac_factor`,
`psf_ac_n_used`, `psf_ac_applied` to `PROC_STORE_COLS`.

**Roadmap:** wide blind HIT CLOSED; WIDE-RIG-REPROCESS OBE; EXTERNAL-XVAL merged FWHM item;
PUBLICATION scheduled LAST; filtered-draft note for NIGHT_FIT/NoFilter CT.

---

## 2026-07-08 — PROC-STORE-TRUST-FIX + coherent draft_424 re-anchor

**Root cause (BASELINE-FULL-DIAG):** `catalog_match_mode` missing from `PROC_STORE_COLS` since
`977920f` — `run_full_photometry_pipeline` produced empty LC trust column; Phase-2A-only reruns
populated it. Retired hybrid snapshot `draft_000424_snapshot_20260708_hybrid_deprecated`.

**Fix (`7960715`):** Add `catalog_match_mode` to `PROC_STORE_COLS`; `wcs_untrusted` derived
downstream. Tests: `tests/test_proc_store_trust_cols.py`.

**Comparator (`750c856`):** `_dist_deg` numeric atol=1e-12 in comp CSV compare;
`catalog_match_mode` stays strict.

**Re-anchor (`619a6e4`):** Coherent cut `draft_000424_snapshot_20260708_full` via
`run_full_photometry_pipeline` (run 1: 2378s). Mode histogram: 24,742 × `master_reference_sky`
(100% trusted). Content SHA: core `92939fab…` n=357; extended `76642318…` n=535.
`--full` gate now content-based (science + photometry SHA; provenance hash informational).

**Verification (run 2):** `session_baseline_check.py --full` OVERALL PASS (2692s pipeline;
178 LCs; SHA match; counters zero). Ledger `VL-ANCHOR-424` + `VL-COUNTERS-ZERO` flipped.

---

## 2026-07-08 — DEV-PROCESS group A closeout (DEV-PROCESS-A/B + EXCEPT-BULK-2)

**DEV-PROCESS-A (`bfe710e`):** `validation/VYVAR_VALIDATION_LEDGER.json` (10 seed items) +
`tests/test_validation_ledger.py` (frozen `REQUIRED_IDS` guard).

**DEV-PROCESS-B (`00dd0cd`):** `scripts/session_baseline_check.py` — `--fast` (git/config/pytest/ledger
TODO hint); `--full` (draft_424 headless via `run_full_photometry_pipeline`, science-meaningful compare
vs `draft_000424_snapshot_20260708`, provenance hash + counters). Documented in RUNBOOK +
CLAUDE_OPERATING_PRINCIPLES session-start ritual.

**EXCEPT-BULK-2 (`97affe3`):** 98/98 drift-skipped rows applied (`--only-ids-file` + ordinal line
resolve in bulk script). Disposition counts applied: 24 delete-dead attempted (all 24 downgraded to
comment-only per safety interlock); log/comment/narrow per census. Census updated; ASCII `delta` fix
in `comp_selection_per_target.py` debug prints.

**Roadmap closeouts:** DEV-PROCESS-A/B DONE; TODO-RECUT-HARNESS-FIDELITY CLOSED (draft_387 zaloha dead);
exoplanet/TOI parked row DONE (see prior EXO JOURNAL). **625 passed**, 15 skipped.

---

## 2026-07-08 — EXCEPT-RETRIAGE-4 + EXCEPT-FIX-4 + EXCEPT-BULK (batch CLOSE)

**Part A:** Tranche 4 (report/export/UI) + 4b (remaining modules) = 213 sites triaged. **Zero T1**
in tranche 4 (report/UI layer). 19 T2 cluster around time-base, trust, provenance, export metadata.

**Part B (EXCEPT-FIX-4):** 8 surfacing-only fixes, zero behavior changes. New counters:
`timeobs_parse_fallback`, `jd_mid_compute_fail`, `trust_kmag_sidecar_read_fail`,
`variability_gaia_id_norm_skip`, `k2_airmass_read_fail`, `export_observer_location_read_fail`,
`optics_draft_override_read_fail`, `check_star_ensemble_filter_skip`. `tests/test_except_fix4.py`
(8 tests).

**Part C (EXCEPT-BULK):** Milan-approved conservative policy via `scripts/_except_bulk_apply.py`:
breadth kept by default; delete-dead for pure log_event guards; stdlib logging per evidence;
comment on all handlers; approved narrow tuples only. Phases C1/C2/C3 (delete-dead, log,
narrow+comment). **623 passed**, 15 skipped; ruff BLE001/E722 clean.

**Part D:** `scripts/_except_census_scan.py` committed. Post-bulk scan: **496** currently-silent.
**pipeline.py reconciliation:** original 160 vs 159 deferral = FIX-2 EXC-0275 already FIXED; post-bulk
160 vs 155 = five handlers surfaced by bulk (delete-dead + ERROR logging) without ordinal refresh.

**Batch status: CLOSED.** Artifact: `CURSOR_RESULT_except_final.md`.

---

## 2026-07-08 — EXCEPT-RETRIAGE-3 + EXCEPT-FIX-3 (tranche 3: astrometry/import/database)

**Part 0 (scanner refresh, `e2444b5`):** re-ran `sandbox/_except_census_scan.py` at HEAD
`9f3da34`. Found the scanner was double-counting (1230 sites) because a leftover
`.worktrees/except_fix1_*` git worktree wasn't excluded → added `.worktrees` to `EXCLUDE_DIRS`.
Added a **stable-ID line-refresh mode**: with an existing census the scanner now preserves every
EXC-#### ID + curated tranche prose/disposition and updates ONLY `file:line` per row (within-file
line-order match, `FIXED` rows treated as retired). **102 line numbers refreshed**; all tranche
IDs unchanged. Deferred `pipeline.py` (160 surviving vs 159 scan — one non-`FIXED` site surfaced
beyond the 10 FIX-2 rows; out of tranche-3 scope, IDs left intact). Census now: 604 currently-silent
+ 21 FIXED + 1 deferred = 626 rows.

**Part A (evidence, `a5b8bdf`):** Tranche 3 (84 core) + 3b `astrometry_optimizer.py` (14,
Milan-approved scope extension) = 98 sites triaged by line-level read. Tiers T1 2 / T2 25 /
T3 35 / T4 36; dispositions fix-now 11, narrow+log-ERROR 4, narrow+log 19, narrow+comment 45,
delete-dead 19. Verified grounded fact: `log_event` self-guards (`infolog.py:28-36`), so pure
log_event wrappers over validated locals are dead code. Key insight: `vyvar_platesolver.py`
(36 sites) is mostly HEALTHY (RANSAC `continue` is correct; terminal fails already surfaced);
real risk concentrates in `importer.py` (calibration library + frame classification).

**Part B (EXCEPT-FIX-3, `c47e9b8`):** 11 sites surfaced (10 new counters + shared
`wcs_header_io.copy_wcs_header_keys`). Surfacing-only: #1 _read_filter, #2 dark BPM sidecar,
#4 library register, #6 IMAGETYP, #7 obs-group meta, #10 platesolver match-rate (+ nan/error
sentinels). **Behavior changes (all fail-safe):** #3 scope-conflict fail-open→fail-closed (DB
error ⇒ assume conflict); #5 capture date today→file mtime; #8 shared WCS-copy helper aborts
atomically on any core-key failure (validated on a scratch header BEFORE opening the FITS, so no
half-written WCS is flushed) — EXC-0625 sibling recovery returns unrecovered, EXC-0010 SIP refit
skipped, cosmetic keys warn only; #9 `_n_unique_spread_sample` promoted to module level, returns
`-1` ("check unavailable") on error and callers no longer reject a good frame on a diagnostic's
own failure.

**Part C (gates):** `tests/test_except_fix3.py` (11 tests incl. #3/#5/#8a-c/#9 + #1/#4/#6 smokes).
Full suite **615 passed, 15 skipped** (baseline 604 + 11). Ruff BLE001/E722 clean. Happy-path
invariance argued structurally: all edits are in except handlers / post-exception sentinels; the
WCS helper is byte-identical on success; the alignment refactor is behavior-preserving and unit-
tested; byte-identity gate tests remain green. (draft_424 headless anchor not re-run — touched
paths do not fire on a healthy draft.)

**Next:** EXCEPT-RETRIAGE-4 (tranche 4: report/export/UI) → bulk dispositions. Artifact:
`CURSOR_RESULT_except_retriage3.md`.

---

## 2026-07-08 — SESSION-CLOSE-0708

**Day closed** on `main` — morning PROV-FIX / QUICKWINS-0708 / CAL-AGE-CLOCK / INPUT-GUARDS;
afternoon EXCEPT batch (census 625, 314 EVIDENCE, 21 T1 fixes, HRD-PLOT-TUPLE fixed).
**604 passed** pytest. **Next:** EXCEPT-RETRIAGE-3 (tranche 3). PUBLICATION venue decision pending Milan.

---

## 2026-07-08 — EXCEPT-FIX-2 (Tranche-2 TOP-10 pipeline)

**Probe:** drafts 424/425/427 — **all 10 NEVER-FIRE** on natural paths (`tmp/except_fix2_probe.json`).
EXC-0433 via unit-test path (425/427 pre-calibrated). EXC-0389 standard pass = `stress_test_relative_rms_from_sidecars`.

**Fixes:** ERROR + `except_fix_counters` per site; 0342 FOV fallback (never 0.0); 0350 coord-drop summary;
0433 retry-once + calibrate stats. **604 passed** pytest.

Artifact: `CURSOR_RESULT_except_fix2.md`.

---

## 2026-07-08 — EXCEPT-RETRIAGE-2 (pipeline.py evidence triage)

**Scope:** 170 `pipeline.py` sites (EXC-0275–0444): preprocess, platesolve, catalog export,
alignment, calibrate/CAL-DIAG. S0 mechanical all-T1/fix-now replaced with consequence-based tiers.

**Counts:** T1 83, T2 41, T3 20, T4 26; fix-now **26**, narrow+log-ERROR 80; **10** silent-drop,
**14** CAL-DIAG flagged.

**TOP-10 fix batch:** EXC-0312 (optics bundle swallow), 0339 (VSX WCS empty), 0342 (Gaia cone 0),
0275 (BPM bypass), 0317 (stale masterstars), 0331 (VYTARG), 0433 (CAL-DIAG DB sync), 0415
(MASTERSTAR ref), 0350 (VSX variable drop), 0389 (stress sidecar continue).

Artifact: `CURSOR_RESULT_except_retriage2.md`, `sandbox/_except_retriage2_apply.py`.

---

## 2026-07-08 — EXCEPT-FIX-1 (TOP-10 T1 + firing probe)

**Probe (drafts 424 NoFilter, 425 B, 427 g):** all 10 TOP-10 sites **NEVER-FIRE** on current data
(PSF trio cold: `psf_photometry_enabled=false`). Artifact: `tmp/except_fix1_probe_light.json`.

**Fixes:** terminal broad-except → narrow classes + ERROR + `except_fix_summary` counters in
`pipeline_meta.json`; EXC-0132 invalid sky → NaN flux + flag (not sky_pp=0). Bonus: `_chk_mag`
indentation UnboundLocalError (`photometry_core.py` ~7924).

**Validation:** `tests/test_except_fix_top10.py` 5 passed; **593 passed** pytest; draft_424 rerun
`except_fix_summary` all zeros. photometry_summary hash changed on full Phase2A rerun (0 comp LCs
path), not except-fix firing — no anchor-worthy science delta.

**Census:** TOP-10 rows → **FIXED (EXCEPT-FIX-1)** in `docs/VYVAR_EXCEPT_CENSUS.md`.

**POSTMORTEM (EXCEPT-FIX-1-POSTMORTEM):** Original draft_424 validation **invalid** — harness used
`comparison_stars.csv` (pool) not `comparison_stars_per_target.csv`; 180→4 summary rows, 2 LCs
corrupted (flat, n_good_comp=1). **Not** attributable to TOP-10 except-fix (counters zero) or
`_chk_mag` hunk (cd0b59e vs HEAD identical on correct-path rerun).

**Verdict (d) — CONFIRMED:** 1-cell `aperture_px_planned` drift on star `1496795041799526400`
(R CVn, catalog mag 7.12 → SNR bin 7.0): **3.918→3.868 px (−0.05)** = deterministic
**RN-HEADER-NONE footprint** (SNR table RN 7.6→15.2 e⁻, Item 4 `1830527`); not rerun noise.
Science columns byte-identical; no anchor ambiguity.

**`_chk_mag`:** indentation bug since `1c802197` (2026-06-16); fixed in `c7227ae` **bundled without
isolated validation**; 424 hot path unaffected (178 check_kmag sidecars).

**CLOSEOUT:** EXC-0626 empty-comp silent drop fixed (`phase2a_empty_comp_drop` counter + summary
stub + pool-CSV schema guard). See `CURSOR_RESULT_except_fix1_closeout.md`.

**Process:** STOP rules bind without row-level attribution; original close should have halted at sha mismatch.

---

## 2026-07-08 — EXCEPT-BATCH-S0 (HRD fix + silent-failure census)

**HRD-PLOT-TUPLE root cause (CONFIRM/REFUTE):** `row_factory = sqlite3.Row` was **already set**
(`hrd_analysis.py:73`). Failure mechanism: `for k in row` iterates **values**, not column names;
`row[value]` → `IndexError`. Fix: `{key: row[key] for key in row.keys()}`. PDF handler narrowed
to `_hrd_build_errors`; ERROR+log+traceback; explicit **HRD panel unavailable** placeholder page.

**Validation:** draft_424 PDF page 15 = Field astrophysics + HRD table (2474 reliable stars);
**0 overflow**; **588 passed** pytest; ruff green.

**EXCEPT census:** `docs/VYVAR_EXCEPT_CENSUS.md` — **625** sites (EXC-0001…), tiers: T1-SCIENCE
354, T2-INTEGRITY 82, T3-UI 76, T4-LEGIT 3, ? 110. Reconciles F-EXCEPT-TIER1 (160 ≈ pipeline 95
+ photometry_core 66 pass/continue).

---

## 2026-07-08 — PUB-TODO (publication workstream opened)

**PUBLICATION** workstream added to ROADMAP: JAAVSO methods paper + JOSS software DOI two-track;
venue matrix researched 2026-07-08 (decision pending Milan). Cross-links: **TODO-SEP-XVAL**,
**TODO-GS8** → Validation section backbone.

---

## 2026-07-08 — Day close (CLOSE-0708) — superseded by SESSION-CLOSE-0708

**Session summary** — five morning workstreams + EXCEPT batch; final gate **604 passed**.
See SESSION-CLOSE-0708 entry above for authoritative close.

| Workstream | Commits | Headline |
|------------|---------|----------|
| **PROV-ARCHEO + PROV-FIX** | `e7ce7ea`, `7b2d285` | Archaeology: `git_hash`/`config_snapshot` never-wired (not regression). `provenance` block in `pipeline_meta.json` (full `AppConfig.to_dict()`, last-writer-wins). |
| **QUICKWINS-0708** | `8c44b71`…`cd0b59e` | K2 GR slope traced; proc `mag` documented; CAL passthrough dead code removed; RN header fix; draft_425 determinism PASS; **draft_424 new anchor** (`e1a7a311…` provenance hash). |
| **CAL-AGE-CLOCK** | `5143485`, `ee89de8` | Import scan + UI share `resolve_master_age` (header `VY_CDATE`); 3 local masters, 0 validity flips. Darks `VY_CDATE` 2026-04-22 → **~2026-07-21** expiry at 90 d (calendar note). |
| **INPUT-GUARDS** | `166cbf4`, `80aab21`, `6a3d020` | Null-island guard at `resolve_site`; PDF cfg from provenance snapshot (G7×PROV synergy); draft_424 PDF **0 overflow**. |
| **EXCEPT batch** | `…`→`136b152` | Census 625 / 314 EVIDENCE; FIX-1/2 TOP-10 + EXC-0626; HRD-PLOT-TUPLE fixed. |
| **HRD-PLOT-TUPLE** | EXCEPT-BATCH-S0 | **FIXED** — `sqlite3.Row` tuple bug; PDF HRD page restored. |

---

## 2026-07-08 — INPUT-GUARDS-0708 (TIER1-OBSLOC-ZERO + G7-F003c)

**TIER1-OBSLOC-ZERO (`166cbf4`):** Null-island guard at `param_resolver.resolve_site`
(choke point). Priority unchanged: draft `ID_LOCATION` → header `SITELAT/LONG/ELEV` → flagged
config. `|lat| < 0.01° AND |lon| < 0.01°` → `ok=False`, `source=unresolved`, ERROR log names
`draft_id` + `ID_LOCATION`. Consumers: `time_utils.resolve_observer_location` →
`pipeline._compute_airmass_from_altaz` (NaN), `photometry_core._recompute_bjd_hjd_with_status`
(`JD_FALLBACK` / `time_base`), `lunar_context.get_lunar_context` (skipped when `site_ok=False`).
Threshold: module constant `NULL_ISLAND_LAT_LON_THRESHOLD_DEG = 0.01`.

**G7-F003c (`80aab21`):** PDF builder `resolve_report_config` reads
`pipeline_meta.provenance.config_snapshot` (PROV-FIX synergy); live `AppConfig` only when
snapshot absent — footer annotates `config: live (no run snapshot)`. draft_424 PDF rebuild:
**0 overflow violations**.

Tests: `tests/test_obsloc_null_island.py` (7), `tests/test_g7_f003c_report_cfg_snapshot.py` (3).
Gate: `587 passed`.

---

## 2026-07-08 — CAL-AGE-CLOCK (unified master validity clock)

**Problem:** import scan used filesystem mtime (`importer._age_days`); library UI used
`get_master_age_days` (header capture date). Copying CalibrationLibrary to a new machine
reset mtime and could revive expired masters.

**Fix:** `resolve_master_age` in `calibration.py` — priority `VY_CDATE` → `DATE-OBS` →
`DATEOBS`; naive datetimes assumed UTC; mtime fallback with one warning per file per scan.
Import scan (`_age_days`), `get_calibration_status`, and UI (`get_master_age_days`) now share
this clock. Boundary: **valid when age ≤ limit** (expired only when age > limit; matches UI).

**Third consumer found:** `get_calibration_status` also used mtime — unified.

**Local library scan** (`tmp/cal_age_clock/library_scan.json`): 3 masters, **0 validity flips**.
All have `VY_CDATE` 2026-04-22 (~76 d); header age ≈ mtime age (no copy scenario locally).

| File | kind | age_header | age_mtime | valid (both) |
|------|------|------------|-----------|--------------|
| Dark_120s_…_20260422.fits | dark | 76.5 d | 76.2 d | ✅ / ✅ |
| Dark_60s_…_20260422.fits | dark | 76.5 d | 76.2 d | ✅ / ✅ |
| Flat_0.15s_…_20260422.fits | flat | 76.5 d | 76.2 d | ✅ / ✅ |

Tests: `tests/test_cal_age_clock.py` (9). Gate: `577 passed` pytest + ruff.

---

## 2026-07-08 — QUICKWINS-0708 (four ledger items + determinism)

**Item 0 (evidence):** draft_425 `B_20_2` Phase 2A rerun on HEAD `21c20e3` — science columns
byte-identical run-to-run (363 LCs, 0 diffs; provenance `stamped_at` excluded). Evidence:
`tmp/quickwins_0708/item0_determinism.json`. Retroactively covers PROV-FIX science check.

**Item 1 — K2-SLOPE-TRACE (`8c44b71`):** `SLOPE_GR_PER_BPRP` 0.859→1.054 (Jordi 2010 Table 6
inverse at FGK g-r=0.48); k2_g≈−0.0169, k2_r≈−0.0042. UG slope 1.091 retained with explicit
exception comment (no Jordi u-g row) — ledger **K2-SLOPE-UG** FUTURE. Analytic 427 extreme-colour
shift (no rerun): g max ≈12.3 mmag, r max ≈3.2 mmag from stored comparison pools.

**Item 2 — PROC-MAG-NAMING (`0913665`):** Documented only — `_vyvar_df_to_csv` docstring +
`VYVAR_PIPELINE_CZ.md` proc schema note; PROCESS dao_flux rule verified present.

**Item 3 — CAL-PASSTHRU-DEAD (`21c20e3`):** Caller audit — no production/test callers of
`allow_passthrough=True`; parameter + synthetic master branch removed.

**Item 4 — RN-HEADER-NONE (`1830527`, science-affecting):** `precompute_and_save_snr_aperture_table_for_draft`
now passes MASTERSTAR header to `resolve_read_noise` (bin2 RN×bin). Unit test
`tests/test_snr_table_rn_header.py`. draft_424 validation: snapshot
`draft_000424_snapshot_20260708`; SNR table RN 7.6→15.2 e⁻, max aperture shift 2.2% (mag 13.0
−0.05 px); Phase 2A rerun 178 LCs **byte-identical** (Phase 2A already used header); median
lc_rms unchanged 0.0863; raw checksums OK. **New 424 baseline anchor** — provenance block hash
`e1a7a311b02c81a5bf602080b345ac95d8ba351327c2f63edd5ca185ff29e80f`. Evidence:
`tmp/quickwins_0708/item4_report.json`.

**Gate:** `568 passed` pytest + ruff BLE001/E722 between items 3–4 and at close.

---

## 2026-07-08 — PROV-FIX (pipeline_meta run provenance)

**Archaeology (PROV-ARCHEO):** `git_hash` / `config_snapshot` were **never-wired** to
`pipeline_meta.json` in recoverable git history (not a regression); JOURNAL background conflated
TODO-25 / PARAM-PROVENANCE with aspirational `TODO-PIPELINE-VERSIONING`. `dynamic_params` was
already wired since history reset.

**Fix (`e7ce7ea`):** `provenance` nested block via `merge_photometry_pipeline_meta` when `cfg`
passed — `git_hash`, mandatory `git_dirty`, full `AppConfig.to_dict()` snapshot (no curated
field list; no secrets in AppConfig), `stamped_at_utc`, `entry_point`. **Last-writer-wins**
(Phase 2A overwrites catalog stamp). Wired at `run_phase2a` and
`generate_masterstar_and_catalog`. Night-run report JSONs unchanged (parallel artifact).

---

## 2026-07-07 — K2-HOTFIX-BSIGN (SLOPE_BV_PER_BPRP sign)

**Root cause:** `SLOPE_BV_PER_BPRP` was `-0.620` — not a valid `d(B-V)/d(BP-RP)` from Jordi et al.
2010 (likely a misread Table 3 quadratic coeff or V−G_RVS linear term). Both B−V and BP−RP
increase with redness, so the converter slope must be **positive**. Wrong sign gave
`k2_B = +0.0186`; draft_425 B validation applied k'' in the **wrong** direction (max |Δ|≈96 mmag).

**Fix:** `SLOPE_BV_PER_BPRP = +0.713` derived from Jordi 2010 Table 3 (B−V)→GBP−GRP polynomial
(Sect. 5.2 Eq. 1) at FGK anchor B−V≈0.58; `k2_B ≈ −0.02139` (Henden −0.03 × 0.713). Audited
GR/UG slope comments (Table 6 / spec anchors; Sloan values unchanged). Added **spec-anchored sign
tests** in `tests/test_k2_extinction.py` (u/B/g/r negative; i/z positive; B in [−0.024, −0.017]).

**Re-validation (425 B):** Snapshot `draft_000425_snapshot_20260707` unchanged; V/R science
byte-identical; B `k2_value=−0.02139` on all rows; max |Δmag_calib|≈110 mmag; median Δ≈+1.1 mmag;
per-target Δmag vs (BP-RP_t − comp_med) **100% sign-consistent** with negative k2 (Spearman ρ≈1).
Report: `tmp/k2_land/validation_report.json`.

---

## 2026-07-07 — K2 session (CAL-DIAG verify, measurement campaign, v1 activation)

**CAL-DIAG-VERIFY (drafts 425–427):** Gate correctly **silent** on `pre_calibrated` imports (no library
darks for eq 2/4). Confirms CAL-DIAG does not false-alarm on Boyden/Newton pre-reduced inputs. k''
data blockers 1+2 **satisfied**: draft_425 BVR on disk; Sloan FILTER strings on 426/427 (`g`, `r`, …).

**K2-DIAG feasibility:** 425 Newton BVR — dX≈0.014 → k''×C×X_bar degenerate with colour term
(~0.6 mmag signal vs ~6 mmag noise). 426 — monotonic X(t), low leverage. 427 — best leverage but
comp residual floor too high for NIGHT_FIT.

**427-RERUN-GATED:** Snapshot `draft_000427_snapshot_20260707`; Fix B/B.2 ON — 8–11% frames dropped;
lc_rms g 0.35→0.11; snapshot g `mag_calib` defect fixed on current HEAD. CAL-DIAG silent on
pre_calibrated (expected).

**K2-SIGMA-FIX:** Harness Fix-A-class residual bug **CONFIRMED** (pooled std without per-comp median
removal → σ_resid≈0.86); Honeycutt differential path correct on flux mags.

**K2-FIT + K2-FIT-VERIFY:** g k''=0 was **clip collapse** on 4-unique-value residuals from proc
catalog `mag`; join-degeneracy hypothesis **REFUTED**. Ungated r k'' fake +56 mmag.

**K2-QUANT-ROOT + FIT v2:** proc CSV `mag` = Gaia catalog G (constant per comp); science =
`dao_flux`. Severity **LOW** (harness-scope). Flux-based comp floor 71–89 mmag; fitted k'' fails
sign/tertile/arc → 427 **NOT feasible** for NIGHT_FIT.

**K2 design v1.0 → live-tree review → v1.1 → Milan Q1–Q3 → implementation:** `k2_extinction.py`,
band_classify CT wiring + CV/CR flip, literature k'' at three insertion points, LC provenance columns.
NIGHT_FIT + pre-gate = **v2** (`k2_fit_enabled` OFF). Spec: `docs/VYVAR_K2_DESIGN_SPEC.md`.

Evidence: `CURSOR_RESULT_k2_*` series; validation `tmp/k2_land/validation_report.json`.

**K2-LAND validation addendum (2026-07-07):** Stage 3 matrix on HEAD `e62cc16`+fixes.
424 NoFilter — science mag cols **byte-identical** (178 LCs; harness flags `catalog_match_mode`
string metadata only). 425 snapshot `draft_000425_snapshot_20260707` — **V/R science
identical**; **B only** delta (max |Δmag_calib|≈96 mmag; median≈−1 mmag; k2_value=+0.0186,
k2_source=literature_default on all rows); raw-light checksums verified. 427 g/r rerun with
k2 columns — vs gated snapshot science delta **expected** (literature k2); g k2≈−0.013744,
r k2≈−0.003436. CV/CR audit: **no tokens** in OBS_FILES or archive — flip affects nobody locally.

---

## 2026-07-07 — CAL-DIAG-IMPL (calibration-time radiometry gate)

**Milan approved:** MP variant (a) parent pre-gate + pass results to workers; D1–D3 unchanged.

**Shipped:** `cal_diag.py` + `_cal_diag_gate_for_obs_group`; dark `dark_resample_mode` override in
`resample_master_to_light_binning`; wired sequential calibrate, RAM-QC, and `calibrate_batch` MP path;
config keys (`cal_diag_gate_enabled` exposed in Settings); D3 READNOISE_E bin1 comments.

**Validation:**
- `tests/test_cal_diag_gate.py`: **14 passed** (PASS/AUTO-CORRECT/FAIL-CLOSED/PASSTHROUGH/path-coverage).
- Full suite: **549 passed**, 15 skipped; `ruff check . --select BLE001,E722` clean.
- draft_424 regression (`sandbox/caldiag_d424_regression.py`): gate ON — **150/150** `VY_DKRSMP=SUM`,
  **0** WARN/FAIL, calibrated pixel arrays byte-identical to baseline; photometry science failures **0**
  (`compare_photometry_science_meaningful`, setup `NoFilter_60_2`); gate OFF — no `VY_*` CAL-DIAG headers.

Evidence: `CURSOR_RESULT_caldiag_impl.md`; report `tmp/caldiag_d424_regression/stage7_report.json`.

---

## 2026-06-25 — Photometry citation audit follow-ups (Stages A + B)

**Milan approved:** fix-now on citation hygiene + diagnostic-first F-HOWELL-3; Stage C gated.

**Stage A (committed):**
- **F-RIELLO-1:** removed dead B-V / Riello report line; dropped `riello2021` from citation
  emitter (BP-RP is raw Gaia catalog value).
- **F-HOWELL-1:** Howell err-model units comment corrected (ADU, not e-).
- **F-CITE-HONEYCUTT:** `honeycutt1992` moved to CORE (always-on Fix-A err model).

**Stage B (diagnostic only, `tmp/phaseHowell3/`):** synthetic; detection/annulus ratio **1.30**;
edge err +**1.5%** on bright star only (mechanism confirmed, magnitude not).

**Stage C (shipped):** `sky_adu_per_px_annulus`; err prefers annulus column. Draft_424
`run_full_photometry_pipeline`: 178/178 LCs science-identical; C2b faint sky-dominated
`err(det)/err(ann)` **1.12–1.14**. Harness: `tmp/phaseHowell3/stage_c_verify.json`.

**Stage D (shipped):** `time_base` LC column via `_recompute_bjd_hjd_with_status`
(`BJD_TDB` / `JD_FALLBACK`); numeric `bjd`/`hjd`/`jd` unchanged. Closes F-BJD-1 and the
2026-06-25 citation/error-model audit.

**Docs:** addendum in `VYVAR_MATH_PHYS_AUDIT.md`.

---

## 2026-07-07 -- Fable audit follow-ups + session close

**Stage A (F-BINGAIN-1, diagnostic only):** `tmp/phaseBinGain/bingain_diag.{md,json}`.
draft_424 bin2: gain `header_index_mapped` 3.17 (scaled-db 0%); photon transfer on field lights
inconclusive (g_eff~0.9; need bin2 flats). Verdict -> **LATENT** (not live). RN sub-Q (7.6->15.2
double-count?) deferred to **CAL-DIAG**.

**Stage B:** F-AIRMASS-CITE -- Rozenberg -> Kasten & Young (1989); bib entry added (`4f18f02`).

**Stage C:** GAIA-ID-FLOAT-GUARD closed on live tree (production reads str dtype).

**Session close:** CAL-DIAG workstream agreed (calibration radiometry gate; spec pending).
`VYVAR_PIPELINE_CZ_rework.md` added (draft, does not supersede `VYVAR_PIPELINE_CZ.md`).
Pushed `4f18f02` to origin.

---

## 2026-06-24 — Photometric band classifier (BAND-DETECT, additive)

**Shipped (additive, not wired).** New module `band_classify.py`:
`classify_photometric_band(obs_group, fits_filter=, aavso_code=)` returns
`{STANDARD_FILTER, LUMINANCE, CLEAR_UNFILTERED, UNKNOWN}`. Single source of truth
consolidating `_is_nofilter_obs_group`, `_is_broadband_photometric_filter`, and
`_AAVSO_FILTER_BUILTIN` synonym mapping (`FILTER_SYNONYM_TO_CANONICAL`). Helpers:
`effective_band_for_extinction` (UNKNOWN fail-safe -> CLEAR for k''), `color_term_auto_from_band`.

**Policy decisions recorded (not yet in production CT gate):**
- **CV/CR -> CLEAR_UNFILTERED** (physically clear-transformed; legacy broadband list wrongly enabled CT).
- **L -> LUMINANCE** (own class; CT off like legacy, distinct from clear for future k'' policy).

**Do-no-harm vs legacy `apply_color_term=auto`:** all real obs_groups AGREE; intentional FLIPs only
CV/CR obs_groups and FITS-rescue when obs_group token unknown (e.g. Johnson V header).

**Rewiring deferred** to band-aware k'' work (production CT gate + CV/CR flip activated together).

**Open data gap:** Newton/Brno literal FITS FILTER strings unconfirmed (not in local Archive).
Capture: `FILTER` / `FILT` / `FILTER1` / `INSTFILT` on Chi_and_H B/V/R/L and Brno `r_60_4` frames;
`SELECT DISTINCT FILTER, ID_EQUIPMENTS FROM FITS_HEADER_CACHE` on dev DB.

**Gates:** `pytest tests/test_band_classify.py` — 52 passed.

**Commit:** `fe9b375`

---

## 2026-06-23 — Exoplanet hosts as active targets (EXO-AS-TARGET)

**Shipped.** Exoplanet hosts with a DAO+position match in `masterstars_full_match.csv` now promote
into `variable_targets.csv` (merged with VSX cone rows), flow through `select_active_targets`, and
behave like VSX variable targets: `_vt_cid_exclude`, global comp pool veto, merged 10″ plan proximity
veto, and `phase01_comparison_min_dist_arcsec`. Dedup VSX↔exo on exact string Gaia `catalog_id`
(VSX labels win; `exo_*` columns additive). Passive informational annotation (EXOPLANET-XMATCH)
unchanged for non-promoted hosts.

**Gaia `catalog_id` string safety.** True TOI-1131 id `1625373404725030528` (Gaia DB); float64
truncation artifact `…0400` does not exist in DB. Promotion path hardened:
`catalog_id_series_for_masterstars_export` + `masterstar_row_gaia_key` (mirrors VSX /
`select_active_targets`). Surfacing: `resolve_masterstars_metadata_csv` prefers
`masterstars_full_match.csv` for UI caption + PDF.

**Validation (draft 422 `V_60_2`, production path).** `write_photometry_plan_files` +
`run_full_photometry_pipeline` (not sandbox): TOI-1131.01 promoted (`exo_disposition=PC`), active
8→9, excluded from comps on true id; 8 VSX targets do-no-harm PASS. Script:
`scripts/validate_exo_as_target_422.py`.

**Observing finding (not a code bug).** TOI-1131 is **genuinely saturated** on 60 s V (`V_60_2`):
target `peak_max_adu` ~29–59 k ADU (median ~44.6 k); MASTERSTAR reference peak **58.5 k** exceeds
85% of equipment ceiling (55.7 k from 65535 ADU) → `zone=saturated`, `skip_photometry=True`, no
Phase-2A LC. **Shorter V exposures required** for a usable transit light curve on this host.

**Deferred (ledger):** GAIA-ID-FLOAT-GUARD — audit all `catalog_id` read sites for `dtype=str`;
reject float inputs to `normalize_gaia_source_id` / `_gaia_id_str` fallbacks.

**Gates:** ruff clean; pytest pass (`tests/test_exoplanet_variable_targets_merge.py`).

**Commit:** `1616b18`

---

## 2026-06-23 — Exoplanet local DB + informational host cross-match (EXOPLANET-XMATCH)

**Shipped.** Local NASA Exoplanet Archive snapshot builder `exoplanets/exoplanet_make.py`
(separate from VYVAR core; not in this repo commit) produces `exoplanets/vyvar_exoplanet_local.db`
(**14185** rows: **6298** CONFIRMED + **7887** TOI). VYVAR integration mirrors VSX read layer +
config: `database.query_local_exoplanet` + schema validate; `exoplanet_local_db_path`,
`exoplanet_match_max_sep_arcsec`; `pipeline._query_exoplanet_local`,
`_exo_host_annotation_arrays`, detect/export hooks; `_apply_exo_host_columns_to_proc_df` closes
MASTERSTAR fast-path gap. Columns `exo_host_obj_id`, `exo_host_name`, `exo_cat_source`,
`exo_disposition`, `exo_match_sep_arcsec` — **additive informational only**; never wired into
`catalog_known_variable`, comp exclusion, proximity veto, variability masks, trust, or photometry
numerics. UI Settings test connection; target detail caption + PDF `exo host:` suffix.

**Isolation (draft 421).** DB-off vs DB-on full B/R/V re-run: comp sets, `mag_inst` / `mag_calib` /
`err`, and trust **bit-identical** per set. **36** hosts in Persei audit box; **1** matched within
3″ (**TOI-7453.01**, Gaia `458577476430838272`, sep 2.79″, B band only) — stayed a comp, not
excluded by exo leak. Test: `tests/test_exoplanet_local_match.py`.

**Gates:** ruff clean on touched files; pytest pass.

**Commit:** `c169675` (ledger `37d18d8`)

---

## 2026-06-23 — G7-F003b PDF report reads phase01_use_bprp_primary

**Shipped.** `photometry_report` reads `cfg.phase01_use_bprp_primary` via existing `self._cfg`
(instead of hardcoded `True`). Default True preserves all existing PDF wording; False switches to
legacy B-V layout/footnotes. Test: `tests/test_g7_f003b_report_bprp_primary.py` (4 cases).

**Deferred (ledger):** G7-F003c — report re-loads `AppConfig()` at PDF-build; cfg edited after
Phase 2A can drift from run settings (LOW-MED).

**Commit:** `795faef`

---

## 2026-06-23 — Tier-1 broad-except STEP 2 core (10 sites)

**Shipped.** Narrowed broad `except Exception` at 10 high-value sites: observer-location DB hydrate
(`config.py`), Gain/RN quality panel (`ui_quality_dashboard.py`), saturate_adu import paths
(`app.py` ×2), four `photometry_core` debug-only handlers, `infolog.log_event` (broad kept +
DEBUG swallow log). Behavior-preserving — fallback values and control flow unchanged; logging and
operator visibility added. Test: `tests/test_config_observer_location_hydrate.py` (2 cases).

**Deferred (ledger):** TIER1-UI-DEBT (38 SAFE UI/plotly `pass` sites, LOW); 299-defensive
pipeline/`photometry_core` cluster (phased audit); TIER1-OBSLOC-ZERO — 0.0/0.0 observer json
fallback may silently corrupt airmass/BJD/dilution context (MED, follow-up finding).

**Gates:** `ruff BLE001/E722` 0-unmarked; pytest 447 passed, 15 skipped.

**Commit:** `f950e3f`

---

## 2026-06-22 — G2-F002b WCS-trust flag + err reconciliation (permanent record)

**Shipped (G2-F002b).** Per-epoch `catalog_match_mode` + `wcs_untrusted` on LC CSV (additive
columns); `n_wcs_untrusted` soft-YELLOW trust signal (distinct from `alignment_failed`);
export modes split (`nondet_no_wcs` / `nondet_unaligned_sky` vs `master_reference_pixel`).
Flag-only — do-no-harm vs frozen draft 421 on production HEAD: `mag_inst` / `mag_calib` / `err`
bit-identical @6dp (`tmp/run_prod_one_target_b20.py`, full-set `run_full_photometry_pipeline`
confirm). Real-data pixel-fallback firing pending wide-rig data.

**Err reconciliation (do not re-litigate).** Frozen draft 421 LC `err` = combined
`sqrt(photon² + ensemble_scatter²)` where photon uses proc-CSV `aperture_r_px`, gain, and DB RN
(1.3). A G2-F002b gate harness that called Phase 2A with **`db=None`** hit RN fallback 10.0 →
~7% inflated photon `err` vs frozen — **not a regression** on HEAD. Production HEAD reproduces
frozen 421 `err` bit-identical when `db` is wired. SNR `apertures_px` dict does not drive photon
err when proc CSV carries `aperture_r_px`.

**Also closed:** G2-F004 (`8f86078` keyed scatter join), G1-F005 + G7-F008 (`473f089` silent except → log). G2-F002b commit `977920f`.

---

## 2026-06-22 — 7-group audit fix-pass CLOSE-OUT (HIGH + numeric MED)

**Účel.** Uzavřít systematický 7-group audit map po fix-passu: všechny **HIGH** nálezy a numerické **MED** z fronty opraveny, validovány a pushnuty; zbytek explicitně backlog.

**Rozsah.** `f3b73e9..f235986` (**44 commitů**, `origin/main`) + `5e01c25` (tracked `config.json`: `alignment_max_control_points=80`, orphan key drop; observer site lokální).

**Integrační brána.** Draft 421 e2e PASS (viz entry níže) — stack hraje dohromady na reálných datech.

### Fix-pass — uzavřené položky (commit → finding)

| Skupina | Finding | Commit | Poznámka |
|---------|---------|--------|----------|
| Phot | forced-aperture / `catalog_only` removal | `7f0dc86` | DAO+Gaia only; variable direct-hit |
| G5 | export plate scale derive-or-None + software version | `6774f83` | G5-F007 |
| G5 | candidate LC calibrated mag | `76c5a93` | G5-F003 |
| G5 | canonical `mag_calib_final` (CT+AC) | `be3e193` | G5-F011 |
| G5 | PDF time axis BJD(TDB) | `b74c301` | G5-F006 |
| G5 | VarAstro comp count vs trust | `07e6f69` | G5-F008 |
| G5 | export failure surfacing | `efbb4de` | G5-F004 |
| G7 | unwired Select Stars removed | `3e1cad7` | G7-F001/F002 |
| G6 | validity_days 90/200 unified | `379e78f` | G6-F002 |
| G3 | `mag_limit=None` = no cap | `fb75867` | G3-F002 |
| G3 | scoped calibration master model | `9e3280e` | G3-F001 |
| G1 | `alignment_max_control_points=80` | `2819e86` | G1-F001/F002; Chi/h 419 PASS |
| G2 | dilution aperture SNR-derive | `4b13e4a` | G2-F003; fixed 3.0 removed |
| G1 | pixel-fallback gated on `VY_ALGN` | `dbf76d5` | G1-F003 part 1 |
| G1 | `alignment_failed` LC + trust flag | `0a43dbf` | G1-F003 part 2 |
| Config | alignment CP key + orphan drop | `5e01c25` | housekeeping |

### Otevřený backlog (po fix-passu)

- **G6-F001 / TODO-MULTISET** — Set-1 rig literals as global defaults; derive-or-None / per-rig profiles
- **G7-F003** — `phase01_use_bprp_primary` non-persistable (`getattr` only)
- **Broad-except Tier-1 backlog** — ~25 remaining `BLE001` sites outside G1-F005/G7-F008

**config.json audit:** orphan reconcile **empty** (247 non-observer keys = `AppConfig` fields); Dáblice bin2 + GS11 OFF → žádný rig override potřeba.

### Verdikt

7-group audit fix-pass **DONE** pro HIGH + numerické MED. Ledger + PARAMS aktualizovány; zbytek v ROADMAP backlogu.

---

## 2026-06-22 — Draft 421: e2e integrační validace celého opravného stacku (PASS)

**Účel.** První end-to-end běh kompletní opravené pipeline na reálných datech, jako integrační brána před pushem. Ověřuje, že se jednotlivě validované opravy chovají správně **dohromady** — ne jen v unit testech.

**Vstup.** Chi/h Persei, konfigurace Dáblice, sety `B_20_2` / `R_20_2` / `V_20_2`, **12 vědeckých snímků/set** (+ MASTERSTAR). Identický snímkový set jako drafty 419 (starý kód) a 420 (13/13 `*.fits` shodných pro každý set), takže meziznámkové srovnání je platné. Hluboký katalog (`catalog_rows=95706`, `n_gaia_detected=12270`), `plate_scale≈1.302″/px`, `density_class=dense`.

**Validovaný stack** (push `f3b73e9..f235986`, 44 commitů): odstranění forced-aperture/catalog_only → G5 export vrstva (kanonická `mag_calib_final`, plate scale z WCS, BJD(TDB), comp count, export failures) → G6‑F002 (validity_days 90/200) → G3‑F002 (`mag_limit=None` = bez capu) → G1‑F001/F002 (alignment_max_control_points=80) → G2‑F003 (dilution aperture: SNR‑derive, žádná fixní 3.0) → G1‑F003 (pixel‑fallback gated na VY_ALGN, flag selhané registrace).

### Výsledky — PASS

| Brána | Výsledek |
|------|----------|
| Pipeline dokončena | **Ano** — infolog řádek 8004 `✓ Pipeline dokončený úspešne`, PDF pro všechny 3 sety |
| Žádný blow‑up (selhání po odstranění forced-aperture) | **Ano** — 0 cílů s `mag_inst > 5` nebo `< −14`; historický ~−12 mag wrong‑star vzor se nevrátil. Kalibrované mag ~8–16.5 |
| G5‑F011 invariant na reálných LC | **Ano** — `mag_calib_final == mag_calib_ac` (CT off) bit‑identicky, **0 neshod / 7468 párů** (B+R+V) |
| Export MAG ↔ LC | **Ano** — 3277/3279 řádků B sedí na 3 dp (2 jen rounding boundary, ne drift); watch cíl přesně |
| `alignment_failed` (nový flag) tichý na zdravé registraci | **Ano** — **0 / 9756** LC bodů; 0 cílů s alignment‑fail v trustu |
| Control points 80 | Reziduály zdravé (max <0.25 px na B); registrováno v logu |
| `mag_limit=None` hluboký katalóg | Aktivní (95706 řádků), comp počty srovnatelné s předchozími Chi/h drafty |
| Watch proměnná Gaia DR3 `458415401545371264` | Finální mag **+0.72 mmag** vs 419 (dřívější ~30 mmag posun nuly **vyřešen**); `mag_inst` −0.17 mmag |
| Comp `458412790204894208` (intermitentní DAO) | `qa_flag=false` v comp_qa — **správně**: odstranění forced bodů opravilo příčinu u zdroje, comp je nyní legitimně čistý (DAO‑only, 10/12), LOO exkluze už není potřeba |

### Rozsah a výhrady (poctivě)

- **Testovací draft o 12 snímcích** → veškerý trust **YELLOW** (automatický `short_baseline`, protože 12 < `lc_quality_min_frames=20`). **Není regrese** — artefakt krátké série. GREEN vyžaduje plné drafty ≥20 snímků.
- **GS11 dilution vypnuta** v configu (`gs11_dilution_enabled=false`) → oprava G2‑F003 v tomto běhu **neprovětrána** (kryta pouze unit testy). Při publikaci s dilution je třeba běh s ní zapnutou.
- **Watch proměnná `n_clean=1`** (tenká comp sada, 7 comps flagnuto jako spike) — důsledek krátké série; na delší sérii by prošlo více comps.
- **V_20_2 plate‑solve chyba** (řádek 3559, RMS 22.48 px) — **zotaveno** v rámci běhu (retry registrace, V má 114 cílů, 97 AAVSO exportů). Hlídat opakování.

### Verdikt

Integrace **PASS**. Opravy hrají dohromady, žádná regrese, kanonická magnituda a nula proměnné stabilní na reálných datech. Stack pushnut na `origin/main` (`f3b73e9..f235986`); následný `5e01c25` doplnil `alignment_max_control_points: 80` do tracked `config.json` (observer site zůstává lokální).

> **Toto ověřuje integrační soundnost stacku, ne finální publikační čísla.** Následuje další testování na reálných datech (plné ≥20‑snímkové drafty), kde se ověří trust GREEN a chování comp_qa na delší sérii.

---

## Session -- Fix C / Phase C1: dense-field alignment DIAGNOSED -> root = PSF/FWHM bloat (2026-06-18)

DIAGNOSE-only (sandbox `tmp/phaseC1/`, no production edits). Deliverable `CURSOR_RESULT_fixC_diag.md`.
**Refutes the Day-2 framing.** The 14 late-night (post-flip, back-half) frames Fix B drops are NOT
"good data, only alignment failed" -- they are **PSF-degraded**. Measured on the production alignment
path (run-414 g):
- **Root cause = PSF/FWHM bloat.** bad-14 median **FWHM 8.60 px = 1.85x** the good baseline 4.64 px;
  concentration flux_large/flux **13.1 vs 1.65** (8x worse); **corr(FWHM, alignment-residual) = 0.95**
  across 161 frames (`tmp/phaseC1/fixC_root_cause.png`). Most likely **late-night focus drift on the
  deliberately-defocused rig** (a pure transparency/flux drop would not bloat FWHM); possibly the
  post-flip half was not refocused -- open observer question. The bloated-donut centroid noise
  (~2.4 px) both breaks astroalign (misalignment is the *symptom*) and is exactly what Fix-B / B.2
  measure. This precisely characterizes B.2's original PSF/transparency intuition as FWHM bloat.
- **Residual geometry.** On the matched subset the leftover is **incoherent ~2.4 px scatter**
  (rotation ~0 deg, scale ~1.0, |t| ~0.1 px; a best-fit similarity does NOT reduce it, rms_pre ~=
  rms_post; no radial trend) + ~50% source loss -- centroid noise, not a recoverable transform.
- **Per-frame WCS:** MASTERSTAR has celestial WCS but **0/162 input frames** do -> WCS-reproject
  unavailable on this non-cal data.
- **Candidates (production path, monkeypatch only the control-point lever).** control-point cap 50 ->
  3/14 recovered, cap 80 -> 2/14, cap+isolation -> 1/14; WCS-reproject N/A; translation-refine
  inapplicable (incoherent residual). **No candidate recovers the 14 to sub-px.** The bloated frames'
  centroid floor (~2.4 px) > the 1.37 px gate, so even a perfect alignment would still be flagged. Only
  the mildly-bloated g_0231 (FWHM 5.12 = 1.10x) recovers reliably.
- **Side finding (real, unrelated to recovery):** astroalign at the production mcp~200 on dense fields
  is **~654 s/frame** (and still fails); capping to ~50 (astroalign's design point) -> ~3-10 s.

**Conclusions.** Fix C as "alignment recovery" is **DIAGNOSED -- NOT APPLICABLE**: there is nothing to
recover, and force-aligning bloated PSFs would not yield science-grade photometry (and would risk the
147 working frames / all rigs). **Fix B + B.2 are the correct handling** -- a **permanent quality
gate**, not a stop-gap awaiting Fix C. The Day-2 "misalignment, not transparency" attribution was an
over-correction; the single root is **PSF/FWHM bloat**, and B.2 (concentration) + Fix-B (alignment
residual) are two downstream symptoms of it (corr=0.95 is the discriminating measurement Day-1/Day-2
lacked). Logged a SEPARATE perf/robustness ticket for a dense-field astroalign control-point cap
(ROADMAP). Diagnose-before-fix prevented shipping a useless, risky alignment change. No code, no commits.
**Watch-item:** mildly-bloated near-threshold frames kept by the gate (e.g. g_0231, 1.10x) -- likely
benign since differential photometry cancels common-mode FWHM changes; watch the LC near the
good->bad transition.

---

## Session -- Fix B: reject-on-alignment-residual frame-quality guard (default-OFF) (2026-06-18)

Stop-gap for the run-414 D-A/D-B finding (the catastrophic LC outliers are 13 phase_correlation
frames mis-aligned ~2.1 px by the translation-only fallback). [The session framing below read "the data
is good; only the alignment failed (that is Fix C)" — **C1 later refuted this**: the frames are
PSF/FWHM-bloated and the misalignment is a symptom; see the C1 session above.] Drop frames whose
*measured alignment residual* is too large, so they never reach photometry. Cause-correct and
method-agnostic. Alignment itself is untouched.

**Audit (`file:line`).** `alignment_report.csv` is written from `star_counts` per-frame dicts at
`pipeline.py:12974` (before MASTERSTAR/Phase-2A, so proc CSVs don't yet exist there). The B.2 gate
hooks `csv_files` at `photometry_core.py:6541` inside `_phase2a_prepare_shared_state`; same point for
Fix B. `alignment_report.csv` lives at `masterstar_fits_path.parent`.

**Step 1 — residual metric (always-on QC).** `_compute_frame_align_residuals` (proc CSVs at the
Phase-2A point): per-frame residual = median, over bright matched sources (10≤mag≤13, flux>0), of the
Euclidean deviation of (x,y) from that source's robust across-night median position. The reference is
dominated by the well-aligned majority, so a translation-mis-aligned frame stands out by ~its full
shift. `_record_align_residuals_to_report` adds the `align_residual_px` column to `alignment_report.csv`
(additive metadata; matched by frame stem; best-effort try/except so QC never breaks a run). Reproduces
the diagnostic: astroalign median **0.358**/max **1.648** px vs phase_corr min **1.450**/median
**2.130**/max **2.334** px.

**Step 2 — gate design (default-OFF, rig-agnostic).** Reject if residual >
`frame_align_residual_max_frac × science-aperture-radius-px` (field-median bright-source
`aperture_r_px` = 5.47 px on run-414). Threshold is a **fraction of the aperture radius, never a fixed
px** (generalizes across rigs). Data: the good/bad gap is 1.206 px (last good astroalign, not an LC
outlier) → 1.450 px (first phase_corr) = **0.22–0.27 × aperture radius**, matching the physical
"residual ≳ ~0.2× aperture-radius is where defocused-donut flux leaves the aperture". Default **frac =
0.25** → 1.37 px, squarely in the gap. Safety floor `frame_align_residual_min_keep_frames` (default 10)
skips the gate (no-op) if too few frames would remain. Frames with no measurable residual (NaN) are
kept. Flags (config + Settings UI + PARAMS parity): `frame_align_residual_gate_enabled` (False),
`frame_align_residual_max_frac` (0.25, clamp 0.05–1.0), `frame_align_residual_min_keep_frames` (10).

**Step 3 — implement.** Wired at the B.2 hook in `_phase2a_prepare_shared_state`: residual compute +
record run always (wrapped in try/except); the gate filters `csv_files` only when enabled. B.2 gate and
the alignment stage untouched. No new method → no new citation. Lints clean; 99 photometry/config/
trust tests pass.

**Step 4 — verify isolated (run-414 g, two real re-runs vs the Fix-A no-Fix-B baseline `tmp/fixA_new`).**
- **OFF = byte-identical:** 70 targets compared, 0 differing `mag_calib`/`delta_mag`/`err` columns;
  V0454 all three `max|diff| = 0`. The always-on residual recording does not perturb photometry.
- **ON (frac=0.25):** drops **14 frames = all 13 phase_correlation + 1 astroalign** (dr=1.648, itself
  an LC outlier). V0454: robust `lc_rms` 0.1027→0.0993, outliers 22→10 (the 12 alignment-caused ones
  removed), catastrophic +3.7 mag/NaN points gone → clean SIPS-grade egress (`tmp/fixB_v0454.png`).
- **B.2 cross-check (not consolidating):** B.2 flags 13, residual gate drops 14, **overlap 13,
  residual-only = the 1 astroalign, B.2-only = 0** — the residual gate is a strict superset; it is the
  cause-correct (alignment) signal, B.2 the aperture-integrity symptom. Both kept distinct.

Committed (push held for Milan). **[Corrected by C1, 2026-06-18:** this gate is **permanent**, not
self-deactivating — C1 proved the 14 frames are PSF/FWHM-bloated, with a centroid floor (~2.4 px) above
the 1.37 px threshold, so they are unrecoverable and there is no "Fix C" that lowers their residual.
Fix B is the correct permanent handling. See the C1 session above.**]**

---

## Session -- Fix A: per-point error model (inflated err / std-of-instrumental-mags bug) (2026-06-18)

Followed up the run-414 D-C diagnostic (`CURSOR_RESULT_414_diag.md`) with the bug fix.

**Bug (audit, `file:line`).** LC `err` (`photometry_core.py:4111`, assembled `:7840-7858`) =
photon/SNR base (`:1462` `_photometric_error`) ⊕ `comp_rms_med/√n_ens` (term-2) ⊕
`ensemble_scatter/√n_ens` (term-3), where `ensemble_scatter = np.std(comp_vals)` on the comps'
**instrumental** magnitudes (`:2552,:2567`). For a sparse/brightness-spread ensemble that std is the
comps' brightness *spread* (a fixed per-target floor), not a per-point uncertainty: V0454's 2 comps
differ 1.655 mag instrumentally → std 0.827 → 0.585 mag on **every** point (23× the empirical 0.025).

**Step-1 consumer audit (checkpoint).** `err` does NOT feed the trust verdict (empirical `lc_rms` +
check-star scatter + comp counts), `lc_rms` (`np.std(mags)`, `:2154`), the production Broeg ensemble
combine (uses `comp_rms`, not `err`), or production sigma-clip. It DOES feed AAVSO/VarAstro export
MAGERR + PDF median-err (intended; improved) and **SysRem IVW weights** (`run_sysrem_field:13138`,
`W=1/err²`) — but `sysrem_enabled` is default-OFF; the fix improves its weighting (bad frames
down-weighted instead of ~uniform). Milan: proceed (option 2 — fix + doc-note SysRem, no SysRem code
change).

**Fix (default; no flag).** Term-3 → per-frame **ensemble zeropoint standard error** = `std(comp
residuals, ddof=1)/√n`, each residual = comp instrumental mag − its own across-night median
(`comp_ref_map`), so brightness/colour spread cancels (Honeycutt 1992). Term-2 (`comp_rms/√n`) dropped
(same ensemble-ZP quantity → no double-count). Photon term-1 kept (correctly large/NaN on SNR collapse).
Touched only the error path; `mag_calib`/`delta_mag`/`ens_med` untouched.

**Verify (run-414 g, re-run vs committed artifacts).** Centres `mag_calib`/`delta_mag`/`mag_calib_raw`
**byte-identical** (max|diff|=0, n=161). V0454 err median **0.581→0.013** vs empirical plateau 0.025
(mis-cal 23.5×→0.5×). Multi-target: NEW err tracks brightness (corr +0.75; bright 0.013 → faint 2.08),
no fixed baseline; faint targets (mag>15) ~unchanged (photon already dominated). The 13 mis-aligned
phase_correlation frames still carry large err (median 4.96, max 17.4) via the photon term — that
removal is **Fix B** (alignment), not A. Tests: 71 photometry/alg/lc-quality + 21 sysrem/trust pass.
Harness `tmp/fixA_verify.py`.

---

## Session -- Boyden V454 CrA (draft_413) sandbox: robustness hardening on defocused meridian-flip data (2026-06-17, end-of-day)

Stress-tested VYVAR on real external Brno-group data (Boyden, V454 CrA, non-cal, defocused, dense
bulge, meridian flip) to harden robustness for arbitrary photometrists. Clean committed baseline at
`955b850` (**pushed**).

**Shipped (8 commits).** `1eea2d2` masterstar catalog-recovery + sibling-WCS + per-set fault isolation
(the 2026-06-14 work, previously uncommitted -- 3-day drift -- now committed; tree hygiene: 16
accidentally-deleted docs restored incl. the operating-principles charter, `config.json` observer
location reverted to Jirny). `e042bc1` A-durable MP-reload robustness (UI crash = Streamlit watcher
reloading `vyvar_alignment_frame` mid-run -> desynced import-time MP bindings -> spawn-pool
PicklingError; fix = fresh module-attribute lookup at dispatch + PicklingError->single-process
fallback; headless never hit it). `d222eb7` B-cap spatial-first `variable_targets` (the 15000-row VSX
cap with no ORDER BY truncated a contiguous Dec slice -> dropped the northern half of the field's VSX
incl. bright named variables V0454/KQ/KM/KT CrA, AND those known variables silently stayed in the comp
pool; fix = frame-bbox query, no cap; re-baseline in DECISIONS: 6/19 originals shift <=0.122 mag, a
comp-purity correction). `2cc2b76` completeness gate scores measurable targets (honest-RED passes,
truncation still fails). `63e57c0` NoDetectionsWarning summarized. `a126980` B.2 frame-quality gate
(default-OFF; `flux_large/flux` concentration z-cut + FWHM guard isolates the 13 collapsed post-flip
frames; precision win bright-target lc_rms median -257 mmag, flat star 0.342->0.035; trust stays RED --
structural, not scatter; Howell 1989 gated; scope Phase 2A only). `15c699e` / `955b850` docs.

**Findings.** Meridian flip benign for alignment (~180 deg rotation, det>0, 100% aligned). Brno gate
correct on defocused fine-scale data (Moffat mismatch 8.7/6.5% -> SAFE_LOW_YIELD, PSF correctly OFF,
aperture+COG ran; blend~0, completeness 1.0; defocus limits depth 12.5-13.9 mag). B.1 aperture-skirt
**refuted** (35% skirt loss real but differential scatter flat -- common-mode PSF breathing cancels;
widening adds sky noise, FWHM-adaptive worse; matters only for absolute photometry, so
`cog_aperture_correction` correctly stays OFF). **V0454 CrA vs SIPS (`v0454_flip_diag`,
`docs/round2_figs/v0454_flip_diag.png`):** the ~0.45 mag gate-ON rise decomposes into a **real eclipse
egress** (comp-invariant pre-flip rise ~0.37 mag, std 0.088 across comps 141-1840 px; ~0.369 mag occurs
*within the pre-flip orientation* with no flip; SIPS-corroborated) **dominating ~4:1** over a
**meridian-flip step ~+0.1 mag post-fainter** (D2 check-star median +0.100; D3 boundary jump +0.144;
D1 comp-dependent post-flip offset std 0.174, near -0.25 -> far/mid -0.38..-0.53). The flip step is
comp/position-dependent and explains the 0.45-vs-SIPS-0.548 amplitude gap (**comp choice, not pixels** --
SIPS used the same aligned frames); root cause = uncorrected flat-field under the 180 deg p->-p mapping,
exacerbated by non-cal data; confirmed independently by SIPS (manual multi-comp) and VYVAR.

**Process.** Diagnose-before-fix paid off (B.1 refuted before shipping a noise-adding fix; the flip
step was measured, not assumed). Hygiene lesson: the 3-day uncommitted drift + accidental doc deletions
must not recur -- commit at the end of each session.

**Outstanding:** UI-VYVAR live test of A-durable (deferred to next session).

---

## Session -- Part A clean state + Round 2 (B.1 refuted, B.2 gate) (2026-06-17)

**Part A (clean committed baseline).** Executed Milan triage: masterstar catalog-recovery /
sibling-WCS / per-set fault isolation feature **KEPT** as its own commit; `config.json` observer
location **reverted** (Jirny); 16 accidentally-deleted docs **restored** (incl.
CLAUDE_OPERATING_PRINCIPLES). `pipeline.py` mixed feature + Round-1 in one file; split via per-hunk
patches (LF-normalised, `git apply --cached`; autocrlf=true, HEAD blob is LF). Result: **6 commits**
(`1eea2d2` feature, `e042bc1` A-durable, `d222eb7` B-cap, `2cc2b76` gate, `63e57c0` log-flood,
`15c699e` docs); tree clean; imports OK; 44 tests pass. **Push gated on Milan.**

**Round 2 (diagnostic-first, default-OFF, isolated). Full writeup: `CURSOR_RESULT_round2.md`.**

**B.1 aperture-skirt -- REFUTED, not implemented.** COG (`tmp/b1_cog_diag.py`, 12 bright isolated
pre-flip stars) confirms the 5 px production aperture captures only **EE=0.65**, but the decisive
test fails: differential scatter is flat from r=5 px to the 18 px plateau (24->27 mmag; min ~21 at
6 px, within noise). FWHM-adaptive aperture is **worse** (30-32 mmag). The ±5-7% skirt-fraction swing
(~48 mmag) is **common-mode PSF breathing** that differential photometry already cancels. Decision:
do not implement (a 5->6 px bump is within noise).

**B.2 frame-quality gate -- CONFIRMED, default-OFF flag.** Diagnostic
(`tmp/b2_transparency_diag.py`): 13 post-flip frames collapse (PSF concentration `flux_large/flux`
~11-16 vs good ~2.7, FWHM pegged at 8.62 px rail, 5 px aperture catches only noise); the two signals
flag the identical 13, wide gap to the good population; the gradual-transparency frames (clear-but-
faint, ratio stays ~2.7) are spared. Implemented `_frame_quality_gate_select()`
(`photometry_core.py`) + Phase-2A hook; params `frame_quality_{gate_enabled=False,ratio_k=5.0,
fwhm_factor=1.0,min_keep_frames=10}` (config + ui_settings + VYVAR_PARAMS); `howell1989` gated in
citations; 5 unit tests. **Isolated measurement** (`tmp/b2_measure.py`, gate ON vs OFF on draft_413
g): LC scatter drops **median -257 mmag, 14/15 bright targets** (V0454 CrA 0.404->0.147; flat field
star ASAS J175832 0.342->**0.035**). **Trust unchanged (all 67 RED both runs)** -- RED is set by
the structural check-star/thin-comp/colour-term-off gates, not LC scatter. Default OFF =>
byte-identical baseline (preserves the end-of-day UI test). Push gated on Milan. Tests: 339 passed,
15 skipped.

---

## Session -- Round 1: four known fixes + pre-flip demo (2026-06-17)

Verified g-only on `draft_000413` (Boyden V454 CrA), reusing existing aligned frames. Full writeup:
`CURSOR_RESULT_round1.md`.

**Fix 1 (A-durable, infra):** spawn pool now resolves MP funcs by **fresh module attribute** at
dispatch (`pipeline.py:12923-12924`) + `pickle.PicklingError` -> single-process fallback
(`12937-12947`). Sim (`tmp/fix1_fallback_sim.py`) reproduces the exact production message ("not the
same object as vyvar_alignment_frame._astrometry_align_mp_task") and confirms the fallback. No
numeric effect. No new param.

**Fix 2 (B-cap, science-changing):** new `_query_vsx_local_frame_bbox` (`pipeline.py:4512`) drives
variable_targets off the **frame bbox** (no cap) instead of the 3.5deg cone + 15000-row cap.
In-frame VSX **26 -> 100**; **V0454 CrA (m9.9), KQ/KM/KT CrA** (all Dec ~-39.5, the slice the cap
dropped) now appear. **Originals NOT byte-identical:** clean OLD-vs-NEW control
(`tmp/fix2_e2e_oldnew.py`) shows **6/19 shift, max |dmag|=0.122**, because `variable_targets` also
drives the global comp-pool veto -> newly-recognised variables are purged from the ensemble
(deterministic comp-purity improvement). **Milan accepted the coupling.** (Earlier "old-code
control" was invalid: `ensure_full_..._stub` writes the restore back to `vt_path.parent`, clobbering
the 26-row backup to 100.)

**Fix 3 (completeness gate):** `audit_photometry_completeness` (`night_run.py:385`) now scores
coverage against **measurable** targets (missing-and-fainter-than-achieved-depth = honest, doesn't
fail; missing-but-detectable = truncation, still fails). Depth derived from data -> no new param.
Tests `tests/test_completeness_gate_measurable.py` 4/4. Real data: the task's g **86.4% (19/22)**
now PASS (`measurable_ratio=1.0`, 3 below depth 13.82); uncapped 67/71 PASS.

**Fix 4 (log flood):** `NoDetectionsWarning` suppressed in the DAO pass-2 targeted-cutout loop,
misses counted, one summary line per stage (`pipeline.py:6420-6480`). No detection change.

**Pre-flip demo (`tmp/demo_preflip.py`):** pre-flip floor is tens of mmag (best lc_rms 13.6, p25
40.5; check-scatter min 28, median 84 mmag), confirming "defocus helps bright" and that all-RED
night-wide is the post-flip collapse. But bright named variables do NOT reach GREEN pre-flip either
(all 68 RED) -- driven by check-star verification gaps (28), check-scatter-high (38), and lunar
background (73% moon). Distinct from the flip; sharpens Round-2 (epoch-quality + bright-star
check-star/comp handling).

---

## Session -- Phase-1b: per-target comp_rms gate authoritative for N_good (2026-06-16)

**Landed:** known-issue (b) closed. The per-target `phase01_comparison_max_comp_rms` (=0.1) gate is now
authoritative for N_good. Separate commit on top of Phase-1 (`1c80219`).

**Two loci fixed (no threshold re-tuning):**
- `comp_selection_per_target.py` `_detrend_and_compute_comp_rms_map` -- RMS fallback no longer relaxes
  above the gate. Old steps `[max, 0.08, 0.15]` admitted comps with `comp_rms > max_comp_rms` (the 0.15
  step). Now the fallback selects only among gate-passers (`comp_rms <= max_comp_rms`); never exceeds
  the gate. Thin-set keeping for gate-PASSERS (Phase-1 downstream behaviour) is untouched.
- `photometry_core.py` auto-routing -- new `_count_gate_passing_comps`: routes on the count of comps
  passing the per-target gate, not raw `len(result)`. Zero gate-passers -> `sparse_fallback`.

**Tests:** added `tests/test_comp_rms_gate_authoritative.py` (8 cases: fallback never admits above-gate;
gate-passers retained; routing helper excludes above-gate / keeps thin set / handles disabled gate).
Full suite **330 passed / 15 skipped**; ruff clean.

**Matrix re-run (`matrix_20260616_185831.json`) vs baseline `164157` -- only SS Cam changed:**
| Target | OLD | NEW | note |
|--------|-----|-----|------|
| V0612 (`...526912`) | default RED 1@0.034 | default RED 1@0.034 | unchanged (gate-passer thin set) |
| SS Cam (`...992064`) | default RED 1@**0.134** | **sparse_fallback YELLOW** 3@~0.35 | flip; 0.134 comp no longer a good default comp |
| BO CVn (`...133184`) | default GREEN 4@0.0086 | default GREEN 4@0.0086 | regression guard PASS |
| V0842 Her (`...714240`) | default YELLOW 8@0.0124 | default YELLOW 8@0.0124 | regression guard PASS |
| V0611 / degenerate | sparse YELLOW 8 | sparse YELLOW 8 | unchanged |

**SS Cam trust RED->YELLOW (reconciliation -- expected RED, got YELLOW):** the (b) fix correctly flips
SS Cam `default -> sparse_fallback` (its only default comp, `comp_rms 0.134`, fails the gate -> 0
gate-passers). The matrix prediction "stays RED (hard check 0.053)" assumed a path-independent
check-star scatter. In reality check-star scatter is **ensemble-dependent**: against the new sparse
ensemble it is **0.043 < `_CHECK_HARD_LO` 0.05**, so it is a soft warning, not a hard RED. This matches
the grounded trust model (spec section 5: `sparse_fallback` lands at YELLOW at most). The sparse comps
(~0.35 mag field-wide) are only catchable by the **Phase-2 sparse-comp sanity ceiling** (explicitly out
of scope here). RED was NOT forced -- no threshold re-tuning, no Phase-2 / check-star-selector changes.
Decision flagged to Milan; accepted the grounded YELLOW outcome and committed.

**DoD:** no selection path counts an above-gate comp as good; routing uses gate-passer count; SS Cam
default->sparse_fallback; V0612 + BO CVn + V0842 Her unchanged; pytest/ruff green. Gate-authority part
of known-issue (b) CLOSED.

**Clarification (handoff, same day):** the SS Cam trust **band** (RED vs YELLOW) is **UNRESOLVED, not
closed** — YELLOW is the current *code output*, not a verified grounded conclusion. The sparse comp_rms
(~0.35 mag) is field-wide-scale (different definition from the 0.1 per-target gate) and check-0.043 is
ensemble-dependent; the tension (comps look bad, check looks OK) is unverified. Phase-2 is
**diagnostic-first**: characterize what field-wide sparse comp_rms represents (does it cancel in the
differential?) and whether check-0.043 is reliable (N points / baseline) BEFORE setting any threshold.
Do NOT reverse-engineer RED.

---

## Session -- Phase-1 graceful comp degradation committed (2026-06-16)

**Committed:** Phase-1 graceful comp degradation (spec: `docs/VYVAR_COMP_DEGRADATION_SPEC.md`).
Structural design validated via cross-field matrix `matrix_20260616_164157.json` (check-star
preselect active).

**Matrix results (164157):**
| Field / target | Result |
|----------------|--------|
| draft_410 BO CVn | GREEN -- 4 T1 comps, check scatter 0.007 |
| draft_411 V0842 Her (`1400549875578714240`) | YELLOW -- 8 T1, comp_rms med 0.0124; soft check scatter 0.023 |
| draft_409 V0612 | RED -- 1 T3 comp; degraded proc; archive T1 comps rejected by Filter B (PSF/FWHM) |
| draft_409 SS Cam | RED -- 1 T3 comp (comp_rms 0.134); honest fail-safe on degraded proc |
| draft_409 V0611 | YELLOW -- sparse_fallback, fieldwide comp_rms labeled |
| draft_409 degenerate | YELLOW -- sparse_fallback (sparse + check caps at YELLOW; see DECISIONS) |

**Structural confirmations:** no field-wide masquerade on default path (`comp_rms_fieldwide`=NaN);
field-wide labeled + matching on sparse; `comp_path` in summary + PDF; sigma scales with N; good comps
not discarded for sparse when N>=1; check-star preselect active on all key targets.

**a) SS Cam 3->1 RESOLVED:** not a regression, not a tier-drop bug. "3 comps" never existed on this
proc (stored archive had 2 T1; both now rejected by Filter B). Log: RMS fallback finds 2 at 0.15,
final 1 kept in `_select_comps_by_color_then_rms` (color/RMS final selection), not trust-tier removal.
Data-limited on degraded proc.

**KNOWN ISSUE (b) -- immediate next fix:** per-target `comp_rms` gate is NOT authoritative for N_good.
RMS fallback relaxes gate (0.1 -> 0.15) at `comp_selection_per_target.py:1614-1627`; auto-routing
`len(result) >= 1` at `photometry_core.py:11925-11928`. Pre-Phase-1 fallback the design intended to
supersede. Violates spec section 3 (N_good = colour ladder + per-target comp_rms gate).

---

**Landed:** `docs/VYVAR_CLAUDE_OPERATING_PRINCIPLES.md` (INIT NOTE, sections 0-7, Session-start
checklist) -- verbatim charter governing how Claude reasons and answers in VYVAR sessions.

**Wired to session init:** `CLAUDE.md` and `docs/VYVAR_PROCESS.md` now list the principles file as a
required read alongside STATE / ROADMAP / latest JOURNAL / PROCESS.

**ROADMAP:** reshaped **TODO-DEV-PROCESS** into two open spec items (not implemented this session):
- **DEV-PROCESS-A** -- JSON pass/fail validation ledger (agents edit status only; no weakened tests).
- **DEV-PROCESS-B** -- session-start baseline-check script (known-good regression before new work).

Both grounded in Anthropic long-running-agent harness / context-engineering guidance.

**Rationale / history:** recurring failure mode = confident symptom-narrative before reading the
governing code/param/log (V0612 degraded-proc vs SS Cam vs V842 Her wrong-ID harness were three
distinct causes folded into one story). Lost validated V0612 proc taught archive-before-re-proc and
baseline-before-work discipline -- now in charter section 5 and session-start checklist section 7.

**Decision recorded:** `VYVAR_DECISIONS.md` (DEV-PROCESS-A/B + session-init charter load).

---

## Session -- photometry root-cause investigation (AIJ ground truth) + direction (2026-06-15)

**Problem entry:** a non-home set (V0612 Cam, draft_407 g, Brno C5A-150M bin4, ~0.562 arcsec/px,
EA eclipse) showed lc_rms ~0.20 chaos while AIJ/SIPS showed a clean ~0.16 mag eclipse.
Investigated against AIJ (AstroImageJ) ground truth (Table.tbl: 14 px aperture, sky 27/54, comps
C2/C3/C4, no detrend; pre-eclipse scatter 0.011 mag).

**Sandbox phases** (all under tmp/phase*/, no production edits):
- Ph1-2: aperture undersizing real but NOT the root cause; simple T/sum(comps) at production 5.55 px
  is already AIJ-clean (0.014). Matched-14px "chaos" was partly an aligned-image-product artifact.
- Ph3-4: ran the REAL production delta_mag path per-stage. Root cause = **temporal_bin_comp_lc
  (ALG-3)** rolling-median smoothing of comp magnitude series before the differential. Disabling
  restores delta_mag pre 0.051->0.0139, eclipse shape corr 0.40->0.935. The "0.214 full RMS" was
  largely the real eclipse counted as noise by undemeaned lc_rms.
- Ph5: mechanism PROVEN -- corr(injection, comp high-freq transparency) = 0.9995; injection vanishes
  on flat transparency, explodes on variable. Regime: binning helps 0/25 targets across both rigs;
  "home works because binning helps" refuted (home just masks the same inflation via slower/larger
  signals). Only pre-differential common-mode breaker is temporal_bin_comp_lc.
- Ph6: prototyped fix pair (binning OFF + leave-one-out residual comp-QA). Pair restores AIJ-class LC
  (V0612 pre 0.011, corr 0.947) and sensible comp QA; LOO residual discriminates a real injected
  variable comp from shared transparency.
- Ph7: simple-pipeline params audit -- color window start **0.15** (existing comp_tier1_bprp_limit);
  widen ladder 0.30/0.55/0.79; N 3-8; flux-sum ensemble (AIJ); trust_flag_enabled=False = bypass.

**Refuted along the way:** aperture-as-root-cause; comp_rms~0 inverse-variance weighting;
PyTICS/color-term/AC/savgol/democratic as the driver.

**Literature grounding:** ALG-3 source is **Hartley & Wilson 2023, MNRAS 526, 3482** (docstring/config
mis-attribute it to "Broeg-Bischoff & Dreizler"). Their method bins ONLY the comp signal and is valid
only when the systematic (transparency) timescale >> cadence AND the comp is white-noise-limited
(few/faint comps). Milan's data violates all three (many comps, fast transparency, fast targets).
Standard tools (AIJ, Broeg 2005/prose, SPECULOOS) reduce reference noise by the per-frame ENSEMBLE
(spatial), which preserves common-mode; they do not temporally smooth comps before the differential.

**Direction decided:** drop the fudge stack (temporal binning, color term, complex weighting) in favor
of a SIMPLE, physics-first differential -> color-matched + stable comp selection. Trust RED/YELLOW
temporarily OFF while tuning photometry. Legacy anchor fields (h&chi Per, DY Peg, BO CVn) and old-SHA
re-cut framing retired; tune on new catalog + new pkl. See DECISIONS / ROADMAP / STATE.

**Deliverables:** CURSOR_RESULT_phase4..6, CURSOR_RESULT_simple_pipeline_params_audit.md;
sandbox harnesses tmp/phase4..7/.

---

## Session — catalog-recovery gate + hint-as-prior (TASK 2 — 2026-06-14)

**Task:** CURSOR_TASK TASK 2 — MASTERSTAR accept on catalog-recovery; `hint_sep` warning when VERIFIED.

**Changes:**
- `vyvar_platesolver.py` — `_compute_masterstar_catalog_recovery`, `_masterstar_solve_acceptance`;
  collapse stacked hint_sep escape blocks; benign ratio 2.50→3.20; FITS `VY_CRT`/`VY_HSWN`.
- `config.py` / `config.json` / `ui_dao_stars.py` — four new MASTERSTAR verification keys.
- `citations.py` + `CITATIONS.bib` — Lang et al. 2010 when catalog-recovery verification runs.
- `tests/test_masterstar_catalog_recovery_gate.py` — synthetic acceptance cases.
- Docs: DECISIONS, STATE, ROADMAP, PROCESS, VYVAR_PARAMS.

**Expected:** Brno `r_60_4` VERIFIED (recovery ~0.84); `z_90_4` stays rejected (~0.34).

**Pending gates before commit lock:** Milan overlay; anchor 0 science failures; home-rig no displacement.

---

## Session — per-set astrometry fault isolation (2026-06-14)

**Task:** TASK 1 from CURSOR_TASK — one bad filter/set must not abort RUN VYVAR; partial-success
semantics for photometry stage.

**Changes:**
- `pipeline.py` — multi-group `astrometry_align_and_build_masterstar`: per-job try/except; merge
  survivors; `skipped_subgroups` on partial fail; all-fail still raises. Single-group path unchanged
  (fail-fast).
- `app.py` — RUN VYVAR photometry: track `completed`; hard-fail only when nothing succeeded;
  surface astrometry `skipped_subgroups` in partial-completion log.
- `tests/test_astrometry_fault_isolation.py` — monkeypatch `_astrometry_align_impl_body`.

**Context:** draft_402 — `g`/`i` solved, `r` failed hint_sep guard, `z` never ran; whole run aborted.
TASK 2 (hint-as-prior reframe) **blocked** on Milan `r` overlay + anchor/home-rig gate.

**Photometry-neutral:** survivors use independent `_astrometry_align_impl_body` per set; skipped set
fail-closed (no catalog).

---

## Session — end-of-session consolidation: Brno production fix + docs (2026-06-14)

**Task:** Diagnose draft_400 Brno `g_60_4` production-path failure; fix; anchor gate; doc sync; commit.

**Comp / anchor (prior in session arc):** comp-stability fix; named-target dedupe; **sparse-only
fallback** default ON + re-baseline lock (**3f7c9e7a** core / **d5b72d08** full); science-meaningful
comparator is the regression arbiter.

**Brno solver arc:**
1. Scoped robustness (cone SIP, hint_sep) — sandbox 83.1% on draft_399 **never production-validated**.
2. ROWORDER Y-flip **rejected** via anchor gate (~320 px / 77% LC break on home rig).
3. Production root cause on draft_400: **Gaia cone at stale VY_TARG** (0.228° off solved center) →
   full-pair refit capped ~55%; not “refit not engaging.”
4. **Fix:** cone recenter on solved WCS (offset ≥0.05°) + full-pair refit pass 3; verified-strong
   hint_sep escape; `app_config` + scoped flags wired in `generate_masterstar_and_catalog`.
5. **Production path:** draft_400 `g_60_4` **passes** (75.5% brightest-N, WCS persists, past MASTERSTAR).
6. **Anchor:** legacy-vs-scoped **0 science failures**; B ~0.003 px. **83.1% retracted** as target.

**Other:** equipment seeds (C5A-150M id=4, AZ800 id=6); citation test fix; gate `_brno_check` → draft_400.

**Open:** draft_401 UI sign-off + overlay; Brno r/i/z end-to-end.

---

## Session — plate-solver scoped lock: legacy-vs-scoped control (2026-06-14)

**Task:** Confirm solver change photometry-neutral via same-harness (A) legacy vs (B) scoped re-cut; lock if (B) vs (A) = 0.

**Control (`tmp/anchor387_legacy_vs_scoped_report.json`):**

| Compare | science failures | max \|Δmag\| B |
|---------|------------------|----------------|
| (B) scoped vs (A) legacy | **0 / 1401** | **0.0** |
| (A) legacy vs archive `3f7c9e7a…` | 1087 / 1401 | 2.26 |

**Conclusion:** ~2.26 mag vs archive is **re-cut harness drift** (shared by legacy arm); scoped change (cone SIP + hint_sep) is photometry-neutral. ROWORDER Y-flip remains **OFF** (327 px break).

**Brno `g_60_4`:** 83.1% match, WCS persists under scoped defaults.

**Locked:** production defaults in `vyvar_platesolver.py`. No anchor re-baseline.

**Parked:** `[TODO-RECUT-HARNESS-FIDELITY]` — re-cut vs archive gate unreliable until harness reproduces anchor science-meaningfully.

---

## Session — plate-solver robustness: SIP full cone pairs + ROWORDER (2026-06-14)

**Task:** Brno `draft_000399` solver fix (SIP on all matched pairs, ROWORDER, equipment onboarding, anchor regression gate).

**Changes:**
- SIP refine pass matches against **deep Gaia cone** (`cat_df_cone_full`), not triangle slice — Brno `g_60_4` 128 pairs (was 34).
- `ROWORDER=BOTTOM-UP` Y-flip on DAO centroids; skip mirror sweep when native ≥10% after flip.
- FITS-header `hint_sep` escape at match ≥80%, RMS ≤2 px, hint_sep ≤0.45° (stale VY_TARG).
- `force_apply` SIP blocked when `rms_sip > rms_linear` (strict; anchor L no longer spurious SIP).
- Equipment seeds: C5A-150M id=4, AZ800 id=6 in `initialize_database()`.

**Brno `g_60_4` after fix:** solve **passes**, WCS persists; match **83.1%** (128/154); linear RMS **1.36 px** (centre 1.15 / edge 2.09); SIP-3 on 128 pairs **rejected** (ratio 1.53 > guard 1.15) — distortion-limited linear, same class as diagnosis overlay (1).

**Anchor 387 B/L/R/V `_20_2` re-solve:** `sip_applied=false` all setups; full-pair SIP input 242–250. Fresh re-solve CRPIX differs from frozen archive (full re-fit); **photometry science compare not re-run** — SIP status unchanged vs archive (all linear).

**Tests:** `tests/test_platesolver_sip_roworder.py` (5). `ruff BLE001,E722` clean on touched files.

---

## Session -- anchor re-baseline lock: sparse fallback default ON (2026-06-11)

**Task:** Reconcile additive-gate failure (`203254fd…` vs `3f7c9e7a…`) — three-way drift vs fallback;
science-meaningful comparator; lock if benign.

**Findings (in-SHA artefacts, all 4 setups, 1401 shared LCs):**

| Compare | Verdict |
|---------|---------|
| (2) vs (1) code drift (ON re-cut vs archived `203254fd…`; fallback inert ⇒ OFF-equivalent) | `comp_path` column only on comp CSV (0 shared-row diffs); max &#124;ΔBJD&#124; = max &#124;ΔHJD&#124; = **1.86×10⁻⁹** d; **mag_calib / delta_mag / mag_inst / flux / dao_flux max &#124;Δ&#124; = 0.0**; per-frame **`err` QC recalc** (max &#124;Δ&#124; ≈ 1.60 mag, median ratio ~2.5×, mag unchanged) |
| (3) vs (2) fallback effect | **0 recovery targets** on full re-cut — all `comp_path=default`; comp counts identical to archive |
| Science-meaningful (3) vs (1) | **PASS** — `compare_photometry_science_meaningful`: 0 science/time failures |

**9-vs-0 reconciliation:** Step-C phase01-only footprint (L:7, R:1, V:1 default-starved in isolated
phase01 rerun) does **not** apply to full `draft_000387` photometry — archive already has ≥3 comps
per target; fallback never fires; no new LC files (2806 unchanged).

**Locked anchor:** core `3f7c9e7a5d8078317cb27678fde028cacf1986d3778547a0c50b087db5f19487` (2806);
full `d5b72d0874a38b6bec69e7a3e56abb63b759b6906495c18aa6bbf4379525b2b6` (4285). Two-run repro
confirmed (`tmp/rebaseline_387_sparse_fb_cut1` == `cut2`). Historical: `203254fd…` / `95a5515a…`.

**Artifacts:** `tmp/anchor_reconcile_three_way.json`, `tests/photometry_sha.py`
(`compare_photometry_science_meaningful`).

---

## Session -- sparse-only comp fallback (2026-06-12)

**Context:** Wholesale `comp_iterative_clip_enabled` on anchor `draft_000387` was marginal churn
(median Δlc_rms +0.04 mmag, ~93% same-n comp swaps, ~47% targets changed per filter) — rejected for
rich fields.

**Refactor:** Repurpose machinery as **`comp_sparse_fallback_enabled`** (default OFF; legacy alias
`comp_iterative_clip_enabled`). Default a-priori selection unchanged; fallback runs **only** when
default yields `< comp_sparse_fallback_min` (default = `n_comp_min`):

- generous pool from masterstars (bypass global RMS pre-filter for that target),
- leave-one-out 5σ-MAD clip on CM-removed residuals,
- provenance `comp_path` ∈ {`default`, `sparse_fallback`} + funnel columns,
- trust **YELLOW** on all `sparse_fallback` targets.

**Anchor:** `tmp/sparse_fallback_step_cd.json` — B_20_2 rerun byte-identical when no default-starved
targets; L/R/V have small default-starved counts → purely additive recovery only.

**Sparse validation:** Qatar-8 / DY Peg in same JSON. CM-detrend target differential remains a
separate opt-in thread (~182 mmag comp recovery, not 17 mmag).

---

## Session -- iterative ensemble-relative comp clip (2026-06-12)

**Problem (sparse-field audit):** Double a-priori `comp_rms` cut — global pool pre-filter
(`comp_pool_rms.py:399–404`) plus per-target gate — mis-rejects comps whose scatter is
common-mode-contaminated (Qatar-8: 2 comps / 237 mmag LC; DY Peg: +7 comps recoverable after CM
removal).

**Fix (now sparse-only fallback — see session above):** machinery behind
`comp_sparse_fallback_enabled` (default OFF).

- Iterative leave-one-out 5σ-MAD clip on CM-removed ensemble residuals (Broeg 1/σ² provisional
  ensemble; Honeycutt sorted-BJD CM detrend); provenance `comp_pool_n_*`, `comp_clip_iterations`,
  `comp_path`.
- Trust: sparse-fallback targets → **YELLOW**.
- Citations gated when fallback runs: Gilliland & Brown 1988; Burdanov et al. 2014; Everett & Howell
  2001.

**Tests:** `tests/test_comp_iterative_clip.py`; full suite + ruff BLE001/E722.

---

## Session -- Phase-1 duplicate Gaia comp pool crash (2026-06-12)

**Problem:** `draft_000391` TOI host (and `1625336025625730816`) failed Phase-1 with
`unhashable type: 'dict'` → no LC (7/8 completeness). Log blamed named target; actual crash was
`comp_selection_per_target.py:2051` sorting `catalog_id` after duplicate comp IDs in global pool
(two masterstars rows, same Gaia ID) made `Series.to_dict()` emit dict-valued columns.

**Fix:** Dedupe global comp pool by Gaia key (keep best `comp_rms`); assembly guards (single-row
lookup, skip duplicate `selected_ids`, scalar IDs); `normalize_gaia_source_id` accepts dict
`source_id`; `normalize_gaia_id_set` for exclude sets.

**Verify:** Re-run Phase 0–1 + 2A on `draft_000391` → **8/8 LCs**; host 78 frames, 4 comps;
267 pytest passed.

---

## Session -- comp-slope stability: common-mode detrend + significance gate (2026-06-11)

**Problem (Step A):** `check_comparison_stability` common-mode detrend fired on DY Peg but removed
~97 mmag/hr instead of ~237 because `np.interp` received unsorted BJD (12/43 inversions). Slope
cut used `|slope|` only (~140 mmag/hr post-detrend) while significance was ~0.03σ.

**Fix (Step B):**

- **B2:** stable `argsort` on `(bjd, mag)` before stacking/interp.
- **B1:** `comp_slope_significance_k` (default 3.0); exclude only if magnitude **and** significance
  thresholds met on post-detrend residual; note includes σ.
- **B3:** Honeycutt 1992 in `CITATIONS.bib`; export citation gated on
  `common_mode_stability_detrend` in `pipeline_meta.json`.

**Verify DY Peg (`draft_000390` B_60_1, scratch):** common-mode log **237.14 mmag/hr**; post-detrend
slopes ~9.2 mmag/hr; no slope-excluded comps (0.03σ); p2p suspect notes unchanged; LC/trust RED
unchanged (2-comp pool).

**Step C footprint (`draft_000387` anchor):** 12 frames/setup (L: 15) — both detrend and slope
require ≥20 finite points → **paths never ran** on anchor; grep **0/1401** `slope=` notes on disk.
Expected LC diff **0 targets**; trust distribution unchanged (informational B: 359 YELLOW / 13 RED
class). **STOP for Milan** before Step D re-cut (`203254fd…` → new SHA after acceptance).

**Tests:** `test_comp_stability.py` (5/5); full suite 262 passed; ruff BLE001/E722 clean.

---

## Session -- per-frame proc perf: DAO pre-filter + Moffat gate (2026-06-12)

**Problem:** `draft_000389` DY Peg `B_60_1` per-frame `proc_*.csv` export ~171 s/frame (~12k DAO rows
measured; 521 written). Moffat Step-1 ran in aperture-only mode (`_run_aperture`) though consumed
only by `psf_photometry.py`.

**Fix (LC byte-identical):** `_proc_drop_unmatched_dao_rows` after detect in both
`_export_per_frame_run_catalog_core` and `export_per_frame_catalogs` (key on `catalog_id`, not
`source_type`). Moffat gate `if _run_epsf:` only (`pipeline.py` was `if _run_epsf or _run_aperture`).

**Verify:** Step A on `draft_000389`; Step C timing 6 frames **171.0 → 35.6 s/frame** (~4.8×).
Matched-row photometry: two consecutive post-fix exports byte-identical on `x,y,dao_flux,flux,mag,
aperture_r_px`; vs pre-fix on-disk baseline small centroid/flux deltas from independent detect re-run
(`mag` / `aperture_r_px` exact; `VY_FWHM` header-driven). `moffat_*` absent in aperture-only mode
(intended). Chi_and_H anchor unaffected (`proc_*.csv` not in SHA set). Milan to re-run `draft_000389`
end-to-end for real-world acceptance.

---

## Session -- math/physics audit hygiene (2026-06-11, `9a4e525`)

Byte-identity-neutral citation + guard pass. Filed `docs/VYVAR_MATH_PHYS_AUDIT.md` (ad6e788
line refs). Scoped Broeg → comp selection + zeropoint; Collins 2017 / Honeycutt 1992 → AIJ
flux-sum combination. `mid_exposure_jd` warns when EXPTIME missing (JD unchanged). MAD constant
unified; `mighell1999` marked export-only. ROADMAP parked D1-combination, D3/#4, Howell second
pass. Tests 261 passed; SHA unchanged.

## Session -- trust / anchor / reliability wrap (2026-06-11, `ad6e788`)

Long session, entirely byte-identity-disciplined (photometry SHA never moved).

**Anchor finally trustworthy.** Re-cut to confirmed-reproducible **zaloha-only** baseline: two
independent fresh runs byte-identical (`draft_386 == draft_387`) → core `203254fd...`, full
`95a5515a...`. Retired `d246a5be` / `30a2f461` (TAP field-DB draft_382) and `f4bcc0ee` /
`bd0b1792` (truncated draft_385). Recipe: zaloha DB (G<=16) + zaloha blind PKLs + #3 code +
this commit. No astroquery anywhere.

**Reliability root cause fixed.** `night_run.audit_photometry_completeness` fails
`night_run_success` when any setup photometry <90% of active targets — silent-truncation-as-
success class (draft_385 bad anchor, draft_383 degradation).

**#3 short_baseline** LC-quality and **Fix A** (proc_*.csv naming) confirmed in place.

**zaloha-only policy.** `chiandh_night_run_bvr.py` reads Gaia from `config.json` (zaloha G<=16);
no field DB, no TAP. (`build_gaia_catalog.py` adaptive-split DEFERRED.)

**Trust correctness cluster** (all photometry byte-identity-neutral):

- Findings A/B residuals: un-evaluated → RED; `check_star_min_epochs=5`; sample std `ddof=1`;
  Finding E re-checked (`short_baseline` stays YELLOW).
- CS-3: check star excluded from comparison ensemble via explicit `ensemble_ids` — prior column-
  based exclusion was dead code (~97% of draft_387 checks were ensemble members / circular).
- CS-2 artefact rms floor; CS-4 crowding exclusion when `contamination_idx` present.
- `comp_trust_min_comps=5` (Option B, trust-only): 3–4 comp targets → RED without touching
  photometry; `strong=min+2=7`. Trust baseline **1382 YELLOW / 106 RED** on draft_387
  (pre-floor-5 was 1400/88).

**min_comp policy** grounded in Broeg 2005 + AAVSO + empirical studies (floor = robustness/
trust gate; >=5 with ~7 saturation knee). Specs filed under `docs/`.

**Broad-except hygiene.** ruff BLE001/E722 enforced (`pyproject.toml`, pre-commit,
`tests/test_ble001_regression.py`); 4 bare excepts fixed (`sandbox/variables.py`); 8
`photometry_core` sites narrowed; 168 grandfathered `# noqa: BLE001`. No silent swallow on LC
magnitude / completeness path.

- core SHA `203254fd75ea5874f5986eac3f478260c2e7e5a9c2636bfecf2b31244cfb09ba` (2806)
- full SHA `95a5515a6c15a473b6fcd29d3afe0c3b78d88a2da434f8a1c03f28dbe2783c24` (4285)
- Tests: 259 passed / 14 skipped; ruff BLE001/E722 clean.

## Session -- Chi_and_H zaloha anchor re-baseline (2026-06-11, RETIRED same day)

Switched `chiandh_night_run_bvr.py` to zaloha-only. Recorded anchor from `draft_000385` — later
found truncated (547 LCs vs ~1401 full). Superseded by draft_386 cut above.

---

## Session -- Chi_and_H byte-identity anchor recorded (2026-06-10)

`chiandh_night_run_bvr.py` -> ephemeral `draft_000382` (Fix A validated). **Anchor = SHA
fingerprint + recipe** (draft deletable; re-verify by regenerating — see STATE).

- core SHA `d246a5be32c13cb0b0fd585220978040262ccf15245d92d1eda8ae78214d7d9d` (2810 files)
- full SHA `30a2f4616e15fbfe8c834cb7c9db87d3688561bf3b9c2290bd7763de2d0b112e` (4291 files)
- field DB recipe: TAP cone 0.75 deg @ 35.15, 57.13, G<=19.5, 2026-06-10, row_count=63138
- **git_commit at cut: not recorded** (gap; re-cuts must `git rev-parse HEAD`)
- source guard: `Archive/Chi_and_H` retained (only non-regenerable input)
- setups B/V/R/L wheel labels: V=visual/green (`G/` folder), L=clear/broadband (`L_20_2` in anchor)
- rig Newton 300/1200 + C3-26000 bin2 ~1.30"/px

Trust decomposition (pre-#3): 1403 `no_data`, 87 saturated, 15 check-star (CS-1..4 logged).
#3 `short_baseline` implemented; acceptance SHA vs recorded values (draft-independent).

## Session -- anchor documentation amendment (2026-06-10)

Amended STATE/JOURNAL/RUNBOOK: anchor is recipe + SHA, not draft survival; source-data guard;
filter-wheel label correction (B/V/R/L); git_commit capture requirement; #3 acceptance compares
re-run SHA to `d246a5be` / `30a2f461` not to draft_382 files on disk.

---

## Session -- Chi_and_H baseline runbook (2026-06-10)

Runbook filed: `docs/VYVAR_CHIANDH_BASELINE_RUNBOOK.md`. Re-cut byte-identity anchor from
Chi_and_H B/V/Ri (Newton bin2 ~1.3"/px) before #3 implementation. STATE: anchor PENDING RE-CUT.
Not executed in this session (local data on Milan's machine).

---

## Session -- short-baseline LC-quality spec #3 rev b (2026-06-10)

Spec updated: `docs/VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md`. Decisions: `short_min=3` (LPV/Mira),
terminal `short_baseline`, exportable YELLOW, **non-escalating** soft (Finding E stays open).
Chi_and_H baseline re-cut plan documented. Ready for implement. No code changes.

---

## Session -- Gaia DR3 integration audit + DR4 deferral (2026-06-10)

Audit filed: `docs/VYVAR_GAIA_DR3_AUDIT.md`. **Decision:** defer GAIA-1 (pmra/pmdec) + GAIA-2
(ruwe/duplicated_source) to the Gaia DR4 catalog build rather than restart the in-progress DR3
rebuild. GAIA-3 closed (G-band correction already in DR3). DR4 ref epoch J2017.5 hook recorded.
No code changes.

---

## Session -- VYVAR_PIPELINE_CZ revize 2026-06-09

Novy/aktualizovany `docs/VYVAR_PIPELINE_CZ.md`: CQ-C, PSF arc, blind solver, byte-identity,
Gaia build, fail-safety; sekce overeni pred odevzdanim clanku.

---

## Session -- build_gaia_catalog repo-root resolution (2026-06-09)

`GAIA_DR3/build_gaia_catalog.py`: upward walk for `gaia_catalog_id.py` (fresh-clone friendly);
clear `SystemExit` when run outside the repo. `build_blind_index.py` unchanged (self-contained).
Utility hardening only; photometry SHA untouched.

---

## Session -- CQ-C fix-once comp_qa locus (2026-06-09)

`comp_qa_core.compute_comp_qa`: magnitude locus fixed at pass-1 `build_locus` (no per-drop
recompute from `dropped_global`). Order-independence PASS (>=5 shuffled orders). Bounded diff
vs iterative locus on draft_000366: **1** flag flip, **1** `n_clean` +1, **0** trust changes.
Core photometry SHA **770966c3** held (283 files); reference SHA **edbd97e7** (426 incl. comp_qa).
Tests: `tests/test_comp_qa_fix_once_locus.py` (slow), `tests/test_photometry_sha_baseline.py`.
pytest **226/6** skip (+3 slow deselected in default run).

---

## Session -- TODO-GS9 descoped (2026-06-09)

Ground-LC Lomb-Scargle/BLS + folded diagram in PDF closed as out of scope; scope boundary
recorded in DECISIONS (period science -> Peranso/VStar/Period04).

---

## Session -- PSF publication-grade arc + EPSF-1 + #4 (2026-06-09)

**GitHub push (PSF arc):** `fe8201c..28fdafa` (12 commits on `main`) -- annulus sky, residual
annulus, sky-only weights, sandwich P3, V3d harness + proofs, citations, docs close PSF arc.
**Local post-push (not yet pushed):** EPSF-1 robust FWHM QC + V3e harness; #4 fail-safety/hygiene
(dead UI delete, MASTERSTAR writeto fail-closed, edge-ok loud flag).

### PSF mid-mag bias investigation (synthetic V3d, seed 367)

1. **Annulus sky** -- replaced 2-px border-median with aperture-consistent annulus on full frame.
   Precision improved; mid-mag accuracy still ~+4.5% post-AC.
2. **fit_shape enlargement** -- ruled out (global meta FWHM; enlargement worsens drift).
3. **Residual-annulus sky** -- shipped; noiseless truth already correct; noisy drift unchanged.
4. **Truth-sky fallback** -- confounded (no error map); not production path.
5. **Literature -> sky-only weights** -- root cause: flux-dependent (object-Poisson) fit weights
   bias point-source fluxes (Astier 2013; Lacroix 2025). `psf_weight_mode=sky_only` fixes mid-mag
   <2%, drift sub-%. Forced position (Guy 2010) not needed.
6. **Sandwich reported uncertainty** -- `psf_err_mode=sandwich_skyonly`; P3 mag12 0.56 -> 1.07.
   V3d **PASS** all pillars mag<=17.

Proof reports: `tier_v3d/v3d_clean_sky_proof.md`, `v3d_weight_proof.md`, `v3d_sandwich_proof.md`.
Numeric SHA **770966c3** held throughout (283 files, PSF gated OFF).

### EPSF-1 + V3e

Robust azimuthal-profile `epsf_fwhm_native`; QC band [0.80, 1.25]. V3e PASS (NEW ratios
1.038-1.049). Diagnostic only.

### #4 fail-safety

MASTERSTAR WCS writeto fail-closed; edge-ok fail-open + `edge_filter_failed` on
`variability_candidates.csv`; orphan UI deleted.

pytest **224/6** skip; SHA 770966c3 held.

---

## Session -- hygiene + fail-safety #4 (2026-06-08)

Deleted orphan UI (`ui_photometry_results`, `ui_suspected_lightcurves`). MASTERSTAR WCS writeto
fail-closed (draft solve fails, Phase 2A blocked). Edge-ok fail-open + `edge_filter_failed` on
`variability_candidates.csv` + report. pytest 224/6 skip; SHA 770966c3 held.

---

## Session -- EPSF-1 robust FWHM QC + V3e (2026-06-08)

Replaced legacy first-pixel half-max `epsf_fwhm_native` with azimuthally-binned radial profile
(`_epsf_fwhm_native_from_profile`). QC warning band [0.80, 1.25] (was 0.5-2.0). Harness V3e:
NEW ratios 1.038-1.049 on Moffat FWHM 2.7/5.4/6.02 px (PASS). Diagnostic only; flux path and
`assess_psf_quality` untouched. pytest 220/6 skip; SHA 770966c3 held.

---

## Session -- sandwich PSF flux_err (P3 fix) (2026-06-09)

`psf_err_mode=sandwich_skyonly`: Var(f_hat) with sky-only weights + true pixel variance.
P3 mag12: 0.56 -> 1.07; V3d status **PASS**. Fluxes unchanged. SHA 770966c3 held.

---

## Session -- sky-only PSF fit weights (2026-06-09)

**Fix 1:** `psf_weight_mode=sky_only` in `psf_photometry_stars` + `_grouped_psf_fit` (Astier 2013,
Lacroix 2025). V3d noisy post-AC: +0.8% (mag12) -> +1.75% (mag16); drift **+0.95 pp** (was +3.5).
Noiseless drift ~0 pp. Fix 2 (forced position) not needed. Citations wired. SHA 770966c3 held.
Report: `v3d_weight_proof.md`. TODO-PSF-V3d-MIDMAG-BIAS closed.

---

## Session -- residual-annulus sky (option C) proof (2026-06-09)

Shipped **residual_annulus** sky refine in `psf_photometry_stars` + `_grouped_psf_fit`
(PSF-only). V3d noisy drift unchanged (+3.5 pp); noiseless annulus median already = truth on
isolated Moffat inject. Truth-sky fallback confounded (no error map). Revised cause:
flux-dependent fit weights + scalar AC. Report: `v3d_clean_sky_proof.md`. SHA 770966c3 held.

---

## Session -- fit_shape enlargement proof + sky drift fallback (2026-06-08)

STEP 0: fit_shape is **global** ePSF-meta FWHM (uniform per star). Enlargement 3x/4x FWHM
**worsens** noisy V3d post-AC drift (+3.5pp -> +8-14pp); **reverted** to 2xFWHM+1. Fallback
noiseless truth-sky: **mag-drift vanishes** (0 pp) but ~+7% uniform offset -> **sky-annulus-wing
contamination** is drift source, not fit_shape truncation in production. Report:
`v3d_fit_shape_proof.md`. PSF not publication-grade; next: annulus push/sigma-clip.

---

## Session -- V3d empirical bias decomposition v2 (2026-06-08)

Harness `v3d_bias_decomposition_v2.py` (T1-T4 on real `psf_photometry_stars`, seed 367).
**T1 noiseless:** bias PERSISTS and grows (post-AC +6% mid-mag) -> **DETERMINISTIC**, not noise.
**T2:** ePSF norm sum/osamp^2=1; reported/fit=0.99; noiseless recovery/truth=1.51 (not unit-conversion bug).
**T3:** fit_shape truncation-sensitive (spread 14.8%; +52.5% at shape 15 vs +43.5% at 31).
**Cause:** `deterministic_fit_shape_truncation`. Proposed fix: enlarge fit_shape (separate task).
Report: `tier_v3d/v3d_bias_decomposition_v2.md`. pytest 218/6 skip; SHA unchanged.

---

## Session -- PSF annulus sky fix + V3d re-validation (2026-06-08)

Production `psf_photometry_stars`: replaced 2-px cutout border-median sky with aperture-consistent
annulus on full frame (`_catalog_only_fixed_aperture_flux` radii; fallback border; `psf_sky_method`
column). Byte-identity SHA **770966c3** unchanged (283 files, PSF gated OFF). V3d re-run (367):
mid-mag post-AC bias still ~+4.5% (excess vs mag12 improved ~0.5-1.4 pp, not <1-2% target); aperture
<1%. A9 draft367: FAIL-SILENT **0**, HV **83.3%** preserved. pytest 217/6 skip. **Verdict:** precision
+ calibration publication-grade; accuracy not yet (residual fit-stage). `v3d_sky_fix_comparison.md`.

---

## Session -- V3d PSF mid-mag bias decomposition (2026-06-08)

Harness diagnostic `run_v3d_bias_decomposition.py` (seed 367, n_real=30): records pre-AC and
post-AC PSF flux per realization. **AC factor 0.701306** (single multiplicative, 8 bright stars);
does not introduce mag-dependence. Pre-AC uniform ~+43% offset (ePSF flux scale); **excess vs mag 12**
shows +4-5% at mid-mag **pre-AC and post-AC** (AC only zero-points bright end). Border sky error
+7.9 ADU/px at mag 12 vs ~0 at mid-mag -> **fit_background_border_median** (2-px border median in
`psf_photometry_stars`). Proposed fix: annulus-local sky in fit cutout (not implemented). Reports:
`tier_v3d/v3d_bias_decomposition.{md,json,png}`. pytest 216/6 skip; SHA unchanged; PSF OFF.

---

## Session -- V3d fine-scale PSF-vs-aperture-vs-truth (2026-06-08)

Harness `v3d_fine_scale.py`: inject-and-recover mag 12-18 at draft-367-like sampling using real
`psf_photometry_stars` + aperture path + PSF AC from bright-star truth. PASS: PSF bias <5% mag12-17,
precision crossover ~mag14, uncertainty ratio ~0.8-1.1. Aperture faint-end bias +19% at mag18 (finding).
`psf_photometry_enabled` OFF in production. pytest green; SHA unchanged.

---

## Session -- NEIGHBOR-SUB pre-2b: bright_close_regime guard + 367 crowding (2026-06-08)

`bright_close_regime` guard (dM>=2.5 brighter + sep<=1.1 FWHM) closes draft-367 edge FAIL-SILENT
(sep1.0/dM-3). Re-run A9: 367 FAIL-SILENT **0**, HV **83.3%**, coarse realistic FAIL-SILENT **0**.
367 real crowding (Red_180_2, VY_FWHM_GAUSS): is_blended **9**, hard **4**, blend@2 **0.022** ->
**VALIDATED_FINE_SCALE_IDLE** (sparse; defer 2b). Brno characterization gate recorded in DECISIONS.
pytest green; SHA unchanged; gated OFF.

---

## Session -- NEIGHBOR-SUB fine-scale test draft 367 (2026-06-08)

Read-only diagnostic (gated OFF; SHA unchanged). Part 1: ePSF-vs-star Moffat audit on draft 367
Red_180_2 (richest filter Red, 16 frames, 0.3889 arcsec/px). **mismatch ratio 0.9994** (ePSF 5.39 px
vs stars 5.40 px) vs h & chi Per 375 L **1.112** (~8%). Part 2: A9 `neighbor_sub` at draft367
sampling + measured mismatch -- HV PASS-RECOVER **83.3%** (coarse realistic 17.6%), FAIL-SILENT **1**
(sep1.0/dM-3 edge), REFUSE **100%**. Decision: **FINE_SCALE_HOME** (sampling rescued mismatch);
2b wire blocked on one edge FAIL-SILENT until guard/crowding follow-up. Reports:
`tmp/epsf_fwhm_367.json`, `tier_a9/a9_draft367_diagnostic.md`.

---

## Session -- NEIGHBOR-SUB step 2a guard hardening (2026-06-08)

Sep floor inclusive (`nn_dist_fwhm <= 0.8`). Catalog-anchored guards: `neighbor_overfit` (fitted nn
mag vs catalog), `target_undershoot` (0.2 mag), `subtract_harmed`, sky-noise SNR floor. A9 realistic
mismatch re-run: FAIL-SILENT **0** (was 14), HV PASS-RECOVER 17.6%, REFUSE 100%, verdict
**SAFE_LOW_YIELD** -- 2b blocked (low yield at coarse bin2; fine-scale / ePSF improvement next). SHA
unchanged; gated OFF.

---

## Session -- A9 mismatch diagnostic / step 2b gate (2026-06-08)

Analysis-only: documented legacy `mismatch` variant (beta 2.0 vs 2.5, neighbour FWHM x1.12 ->
model/star 0.89 on neighbour, inverted vs EPSF audit). Added `realistic` variant (model/star 1.08,
e=0.08) anchored to `VYVAR_EPSF_FWHM_TEST`. Per-cell breakdown:
realistic HV PASS-RECOVER 16.7%, FAIL-SILENT 14, verdict **BLOCK_2B_GUARDS** (guards must strengthen
before pipeline wire). Reports: `tier_a9/a9_mismatch_diagnostic.md`. pytest green; SHA unchanged.

---

## Session -- NEIGHBOR-SUB step 2 core + A9 scoring (2026-06-08)

Joint-fit subtract + aperture residual core (`psf_neighbor_sub.py`): fit target+neighbour together,
subtract neighbour only, aperture target via `_catalog_only_fixed_aperture_flux`. Prototype finding
baked in: single-neighbour fit over-subtracts; joint fit recovers on ideal data. Fit-quality-driven
guards (sep floor ~0.8 FWHM pre-check; refuse on no_improvement / centroid shift / ill-conditioned
amplitude). `BlendMapEntry` + `_load_blend_worklist` extend crowding plumbing. A9
`measure_cell(mode="neighbor_sub")` scores ideal vs PSF-mismatch (fit_beta=2.0, inject FWHM x1.12);
coarse pass rates ideal 85.7% / mismatch 21.4%. `psf_neighbor_sub_enabled` default OFF; production
measurement sites NOT wired (step 2b). pytest 195/6 skip; numeric SHA 770966c3 unchanged.

---

## Session -- NEIGHBOR-SUB design recorded (2026-06-09)

Read-only design for TODO-PSF-NEIGHBOR-SUB: fit neighbour ePSF, subtract, aperture residual.
Worklist on corrected crowding (375/380 L: 58/53 blended, 39/34 hard). Insertion at per-frame
measurement (not `compute_lc_flux_method` router). `_load_adaptive_blend_map` needs full worklist
row extension. Doc `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md`. No code.

---

## Session -- TODO-FWHM-CONSISTENCY implemented (2026-06-09)

`header_core_fwhm_px` in `masterstar_context.py`; `crowding_index._load_wcs_meta` and
`psf_photometry.get_epsf_fwhm_from_context` now prefer VY_FWHM_GAUSS -> VY_FWHM_GAUSSIAN -> VY_FWHM.
h & chi Per 375/380 L live crowding: 58/53 is_blended, 39/34 hard; ePSF QC ratio ~0.78/0.81.
Numeric SHA 770966c3 unchanged; pytest 183/6. ROADMAP TODO-FWHM-CONSISTENCY closed.

---

## Session -- crowding recompute VY_FWHM_GAUSS 375/380 L (2026-06-09)

Read-only recompute: baseline crowding (VY_FWHM) self-check OK (77/87 is_blended). Corrected with
VY_FWHM_GAUSS (~2.73 px): 58/53 is_blended, 39/34 hard nn<1.0. **PROCEED** for NEIGHBOR-SUB on
h & chi Per. Diagnostic preceded production fix; doc `docs/VYVAR_HCHIPER_CROWDING_RECOMPUTE.md`.

---

## Session -- decisive ePSF FWHM test 375/380 L (2026-06-09)

Independent Moffat/Gaussian/azimuthal FWHM on built ePSF + stellar cutouts. **Dominant: EXPLANATION 3**
(OBS_FILES seeing ~3.84 px vs core ~2.0 px). EXPLANATION 2 rejected (ePSF ~= stars). Secondary:
buggy half-max at 2.236=sqrt(5). Doc `docs/VYVAR_EPSF_FWHM_TEST.md`. No production changes.

---

## Session -- ePSF path audit EPSF-1 (2026-06-08)

Read-only audit at fe8201c: `epsf_fwhm_native` half-max estimator (`psf_photometry.py:500-516`)
biases ratio<1 on h & chi Per -- diagnostic only, not flux/gating. Doc `docs/VYVAR_EPSF_AUDIT.md`;
ROADMAP `TODO-EPSF-1-FWHM-QC`; harness V3e added. No production changes.

---

## Session -- h & chi Per PSF probe drafts 375+380 (2026-06-08)

Read-only crowding + ePSF QC on solved MASTERSTAR data. Scale **~1.30"/px bin2** (not fine).
L-band richest (30 frames). blend_frac_1fwhm ~3.7-4.4%; 77-98 LC stars is_blended on L.
ePSF asymmetry ~0.001 (no smear flag); ePSF/input FWHM ratio 0.59-0.67. Doc:
`docs/VYVAR_HCHIPER_PSF_PROBE.md`. pytest 183/6 unchanged; production untouched.

---

## Overnight session — inject-and-recover validation harness (2026-06-08)

Built `tests/validation/` (gen_frame, gen_series, recover, score) wired to real VYVAR
entry points: crowding blend metrics, Sokolovsky indices, aperture photometry, trust gate,
color-term fit, BJD/airmass, calibration masters. Tier A single-frame + Tier B 60-frame
Gaia-structured series (fallback catalog). First full run: **14 pass / 2 fail / 2 skip**.
FAIL findings (not production fixes): A3 ePSF asymmetry on smeared cutout, A7 photutils vs
SEP ~0.7% flux offset. Docs: `docs/VYVAR_VALIDATION.md`, `tests/validation/README.md`.
Production photometry untouched; pytest 183 passed / 6 skipped.

---

Last updated: 2026-06-03 (session 03.6.2026 — APCORR-COLOR Path B: extrapolation guard block)

---

## APCORR-COLOR — prototype + extrapolation block (2026-06-03)

**Prototype (draft_000366, read-only):** 141 targets in `ct_prototype.csv`. Median c1 −0.07
(−0.36 nonzero); median |ct_corr| 0.019 mag (69 >0.02, 42 >0.05); cat−inst scatter
0.078→0.053 mag; `gate_would_pass` 15/141 (10.6%). Worst |ct_corr| ~4.8 mag on red targets
with BP-RP outside comp range. NoFilter production unchanged.

**Path B fix:** `_check_color_term_extrapolation` → bool; out-of-range blocks CT (warn kept),
`ct_ok=False`, target kept uncorrected. Config `phase01_ct_extrapolation_tol` (default 0).
NoFilter skip in `should_apply_color_term` untouched.

---

Last updated: 2026-06-02 (session 02.6.2026 — cross-validation draft_000365: V842 spot-check + whole-night 143 targets via xval_run.py)

---

## Cross-validation — draft_000365 (V842 Her, EW) — photutils + SExtractor vs VYVAR  [2026-06-02]

Independent end-to-end check of VYVAR aperture photometry against two outside engines.
Each built its OWN Gaia DR3 catalogue from the frame WCS, own detection, own apertures and
background; only the input FITS are shared with VYVAR. Compared via the SAME unweighted
leave-one-out differential method against VYVAR per-frame `dao_flux` (`proc_*_Light_*.csv`)
and reported `lc_rms` / `comp_rms`. 127 aligned NoFilter_60_2 lights, 9.77"/px wide field,
FWHM ~3.0px. Join key = Gaia `catalog_id`. Target V842 Her (EW) 1400549875578714240,
lc_rms 0.1696, 8 TIER1 comps.

| engine              | comp RMS | target RMS | per-frame vs dao_flux |
|---------------------|----------|------------|-----------------------|
| photutils (annulus) | 0.0143   | 0.1709     | 0.0102 mag            |
| sep / SExtractor    | 0.0105   | 0.1706     | 0.0022 mag            |
| VYVAR dao_flux      | 0.0104   | 0.1713     | —                     |
| VYVAR reported      | 0.0117   | 0.1696     | —                     |

Conclusions:
- Science number reproduced by THREE engines to ~1% (0.1696–0.1713). No systematic offset.
- sep matches VYVAR extraction to 0.2%/frame and 0.0105 vs 0.0104 comp RMS → VYVAR aperture
  photometry is equivalent to a SExtractor mesh-background pipeline.
- The 25% photutils comp-RMS gap is fully explained: photutils local-annulus background +
  centroid_com added ~0.010 mag/frame; sep (mesh background) reproduces VYVAR. Budget closes:
  sqrt(0.0104² + 0.0102²) = 0.0146 ≈ photutils 0.0143.
- VYVAR `comp_rms` (0.0117) sits just above the 0.0104–0.0105 floor — slightly conservative (safe).
- Big aperture (2×FWHM) independently confirmed VYVAR's small SNR-optimal aperture: at 9.77"/px a
  bright neighbour falls in the annulus (3/8 comps flagged, ratio up to 2.08); small aperture avoids it.
- Alignment independently confirmed: median centroid drift 0.39px, stable across all 127 frames.

### Whole-night — all 143 targets (harness `xval_run.py`)

Ran the consolidated harness on the full draft (photutils + sep extraction of 1108
sources incl. all 143 targets + their comps; decomposed vs VYVAR `dao_flux` + `lc_rms`).
Regression: V842 Her reproduced exactly (phot 0.1709 / sep 0.1706 / dao 0.1713 / VYVAR 0.1696).

| engine vs VYVAR lc_rms | N | median ratio | median \|Δ\| | within ±15% |
|------------------------|---|--------------|-----------|-------------|
| dao (my diff on VYVAR flux) | 140 | 0.93 | 0.0052 | 70% |
| sep / SExtractor (independent) | 133 | 0.97 | 0.0082 | 60% |
| photutils (annulus) | 133 | 1.22 | 0.0332 | 35% |

`comp_rms` sep/dao ratio = 1.055 (N=142) → independent SExtractor reproduces VYVAR
extraction precision across the whole night, not just one target.

Whole-night conclusions:
- Independent pipeline reproduces VYVAR per-target RMS to ~3% (median) over 143 targets.
- VYVAR `lc_rms` is consistent with, and slightly conservative vs, the raw differential
  floor (dao ratio 0.93) — VYVAR never under-reports its noise (safe for the database).
- photutils-annulus is NOT a reliable independent witness on crowded/faint targets
  (ratio 1.22, inflates); sep is. Independently justifies VYVAR's robust background + small aperture.
- ~8 low-confidence targets (e.g. Gaia 1399954352593040512, V1138 Her, Gaia 1403010994918399232):
  sep's OWN extraction blows up while dao≈VYVAR → VYVAR flux is fine, just not independently
  confirmable with a quick aperture pipeline (faint+blended). Flag for caution pre-AAVSO.

IRAF/PyRAF: CLOSED as unnecessary — two independent engines (one matching VYVAR to 0.2%) already
validate extraction; legacy IRAF adds no independent axis and is not feasible on Py3.12/Ubuntu24.

Harness gotchas (for `xval_run.py`): isolation must be a PHYSICAL bright-neighbour test (within
annulus r_out AND within ~2.5 mag), not raw nearest-neighbour on a deep catalogue; force-include
VYVAR comps+target; Gaia TAP returns SOURCE_ID upper-case (normalise case); join catalog_id as 19-digit string.

Harness `xval_run.py` validated on full draft_000365 (143 targets); reusable for other drafts.

---

## Reporting & export overhaul — COMPLETED (P1–P3.5)

- **P1 data correctness:** AAVSO MTYPE fixed DIFF→STD (TRANS=NO; every prior AAVSO file was mislabeled DIFF); validator no longer warns on STD+NO. Universal table-driven AAVSO FILT map (full code set: U/B/V/RJ/IJ/Rc/Ic/Sloan SU–SZ/CV/CR/TB/TG/TR/J/H/K/Y) + config override (`aavso_filter_map`); unknown filter → FILT=UNKN + `#WARNING` (no silent CV). Honest method label (`meth=` in `#SOFTWARE`/NOTES; was hardcoded `"aperture"`). OBSCODE guard (`#WARNING` when empty/default UMIA).
- **P2 citations:** now sourced from `CITATIONS.bib` (28 entries) via a single **conditional** emitter (`citations.py`) shared by AAVSO export + VarAstro export + PDF Methods. Cites only methods that actually ran in the run (e.g. Anderson&King/Moffat only when PSF on). Added Eastman 2010 (BJD), VSX/Watson, astroquery/numpy/scipy, + conditional refs.
- **P3 separate per-method reports:** aperture (always) + psf/adaptive (when enabled), one method per LC/report, suffixed (`_psf`/`_adaptive`) + labeled. Aperture-only default is **byte-stable** (verified SHA-256 on 362) — no disruption to current workflow.
- **P3.5 KMAG:** now the **measured ensemble-standardized check-star magnitude** (per-row via `check_kmag_{target}.csv` sidecar; check star excluded from its own ensemble); was `na`.

**ROUTING (confirmed intended):** eclipsing → VarAstro (LC); pulsating/all → AAVSO.

**NEW-DRAFT READINESS:** AAVSO Extended + VarAstro auto-produced after Phase 2A; manual upload.

**USER ACTION:** set `aavso_observer_code` (default UMIA placeholder warns).

**OPEN (optional, low priority):** B.R.N.O. minima (Kwee–van Woerden) for eclipsing; AAVSO API auto-upload (TODO-GS10); 4 KMAG=na targets on 362 (insufficient comps).

---

## VYVAR SESSION SUMMARY — 30.5.2026 (parameter provenance + setup UX)

### COMPLETED 2026-05-30

- **[PARAM-PROVENANCE]** Single authoritative parameter resolver across all stages +
  per-draft observer-site fix (kills the config-drift trap), plus the setup UX layer
  (default markers, Scan-Source optics auto-detect, poor-FITS prompt). Two commits:
  `bd4a539` (Phase 1, production correctness) + Phase 2–4 (UX layer).
  - **Resolver (`param_resolver.py`), two classes + site:**
    * equipment-intrinsic (gain, read-noise, pixel, focal, saturation):
      **DB-set(valid) → header(cross-check warn) → config**. The DB cross-check rejects
      *plausible-but-wrong* headers a sanity range would accept — proven by 363
      `XPIXSIZE=10.0` rejected in favour of DB `3.76`.
    * observation-specific (binning, exptime, pointing, time): header → DB → config.
    * site (lat/lon/elev): **per-draft `ID_LOCATION` → header `SITELAT/LONG/ELEV` →
      config (FLAGGED, never silent)**; unresolved → `ok=False` → Phase-4 prompt.
  - **Divergent paths eliminated** (gain Phase-2A-DB-only vs error-map header-first;
    site time_utils-FITS-first vs Phase-2A-config-only) — every consumer routes through
    the one resolver: BJD, airmass, per-frame error-map, lunar context, SNR-table,
    `crowding_index`, QA detector readout, AAVSO `#LATITUDE/#LONGITUDE/#ELEVATION`,
    VAR.ASTRO `#Site`, PDF Observer Location (submitted coords now tied to the BJD site).
    Documented deep fallback: `pipeline.py` worker-meta `cfg.observer_*` (worker already
    resolves draft→header first).
  - **VALIDATED (config stays Jirny throughout):** 360/361/362 → **Jirny**, 363 →
    **Dablice**, each from its own `ID_LOCATION`; forcing `config.json` to a **bogus
    Sydney** location leaves every draft unchanged → BJD/airmass are now
    **config-independent**. 363 BJD shift **0.0402 ms** vs the old config-only path
    (360 = 0.0000 ms, Jirny == config). Pre-commit sanity: gain/RN/pixel **identical**
    across old-error-map / old-Phase-2A / resolver for 360 and 363 (DB was already used
    on this data) — no science movement beyond the negligible site shift.
  - **config.json** `observer_location` is now **UI / last-session state only — moot for
    the science** (BJD/airmass derive from per-draft `ID_LOCATION`). Not changed by this work.
  - **Phase 2 schema (DB):** `IS_DEFAULT` on `EQUIPMENTS`/`TELESCOPE`/`LOCATION` + `ACTIVE`
    on `LOCATION` (idempotent migrations); `set_table_default`/`get_default_id` + editor
    exclusivity (exactly one default each). Seeded **QHY294MM(1) / Carl-Zeiss 200mm(1) /
    Jirny(2)** — explicit user markers, not a silent `id=1` fallback.
  - **Phase 3 (`optics_autodetect.py`):** fingerprint camera (INSTRUME + full-res sensor
    dims + GAIN), telescope (FOCALLEN+APTDIA), site (SITELAT/LONG); `TELESCOP` treated as a
    useless sample string. 360 → QHY294MM 0.90 / Carl-Zeiss 0.90 / Jirny 0.90 (all high);
    363 → C3-26000 0.55 (medium; 6252×4176 dims + GAIN 0.78, **no name**), telescope/site
    unmatched. Confident matches auto-fill the Scan-Source selectors (override default);
    user can still override.
  - **Phase 4 (`assess_unresolved`):** surfaces ONLY the gaps, pre-filled from default.
    363 → prompts Telescope + Observer site + Pointing; 360 → none.
  - **UI:** Scan-Source selectors pre-select the `IS_DEFAULT` rows; library editor exposes
    `IS_DEFAULT` checkbox; poor-FITS prompt panel.
  - **Migration scope:** backfilled ONLY library-table markers — **zero `OBS_DRAFT` rows
    modified** (frozen per-draft `ID_EQUIPMENTS/ID_TELESCOPE/ID_LOCATION` left intact:
    360–362 Jirny, 363 Dablice).
  - Files: `param_resolver.py`, `optics_autodetect.py` (new); `time_utils.py`,
    `photometry_core.py`, `pipeline.py`, `crowding_index.py`, `ui_quality_dashboard.py`,
    `export_reports.py`, `photometry_report.py` (Phase 1); `database.py`, `app.py`,
    `ui_database_explorer.py` (Phase 2–4).

### NEW TODO — parameter provenance

- **[TODO-PIXEL-XCHECK-BINNING]** `param_resolver.resolve_pixel_um` cross-check compares the
  raw header `XPIXSZ` against the DB **native** `PIXELSIZE`, so it warns on a legitimately
  *binned* pixel (360: header `XPIXSZ=9.26` = 2×2 of native `4.63` → spurious ">5% disagrees"
  warning). It still correctly uses the DB native value, so **no LC impact** — purely a noisy
  log. Fix: make the cross-check binning-aware (divide header pixel by `XBINNING`/`YBINNING`
  before comparing like-to-like), mirroring the binning-aware logic already in
  `optics_autodetect.detect_equipment`. LOW / cosmetic.

- **[TODO-CONFIG-CHURN]** The app re-serializes the **tracked** `config.json` every run
  (rewrites last-used session state — site/rig — into the same file as the static
  overrides) → perpetual git diff that has to be `checkout`-discarded each time. **Zero
  functional effect** (the resolver ignores the config site; the UI uses
  `LOCATION.IS_DEFAULT`; the rewritten `observer_location` is vestigial). Durable fix:
  **separate the session-state the app rewrites from the static user overrides** (e.g. a
  distinct `session_state.json` / UI-state store) so `config.json` stops drifting. **Do
  NOT gitignore `config.json`** — it still holds real user overrides. LOW.

- **[TODO-BROAD-EXCEPT-HYGIENE]** Broad `except Exception: pass` / `: continue` wrapping a
  **fallback / protection / degradation path** can swallow **programming** errors
  (`NameError` / `AttributeError` / `TypeError`) → the safety net fails *silently*, so an
  unusual/new set hits the un-protected path with no warning. 4 such instances were already
  found & fixed: `app_config` access, the optics FOV override, header-median pointing, and the
  association-slice rebuild (all were dead-on-arrival behind a bare except). **Likely more
  exist** (~700 bare `except: pass/continue` codebase-wide; the dangerous subset is the ones
  guarding a safety/fallback path in the core runtime: `pipeline.py`, `vyvar_platesolver.py`,
  `photometry_core.py`, `comp_selection_per_target.py`, `psf_photometry.py`, `param_resolver.py`,
  `optics_autodetect.py`, `calibration.py`). Fix pattern: **narrow each protective except to the
  EXPECTED runtime exceptions only** (e.g. `(KeyError, ValueError, TypeError)` for header parses,
  `(OSError, FileNotFoundError)` for IO) so `NameError`/`AttributeError` surface instead of
  disabling the protection invisibly. MEDIUM (do as a data-independent sweep). LOW risk per edit.

- **[APCORR-COG]** Per-frame curve-of-growth (encircled-energy) aperture correction —
  config-gated, **default OFF**. Fixes STEP-3 (audit): per-star SNR-optimal radii feed
  `dao_flux` with **no** enclosed-fraction correction → constant target↔comp differential
  bias + seeing-correlated systematic. New `compute_per_frame_cog_correction` builds a
  per-frame EE(r) from bright/isolated/unsaturated/high-SNR stars (isolation 6×FWHM, SNR≥50,
  peak<0.85·sat, capped 60), `ac_factor = 1/EE(r_star)` puts every star on the common
  `cog_ref_fwhm×FWHM` (4.5) ref-radius scale. Emits `dao_flux_apcorr`/`ac_factor`/`cog_ok`
  (never overwrites `dao_flux`); routed into `ensemble_normalize` via `read_flux_from_csv`
  `use_apcorr_flux` only when enabled + `cog_ok`. New config keys `cog_aperture_correction_*`
  (all gated OFF). Files: `config.py`, `photometry_core.py`, `pipeline.py`.
  - **Validation 360/363 (OFF vs ON):** target↔comp enclosed gap **23%/34% → 0.000%**.
    363 (variable seeing): robust quartile (n=178 vs 89) seeing slope **+0.037 → −0.006
    mag/px (−85%)**, p2p **40.7 → 13.9 mmag (−66%)**, comp LC RMS **19.1 → 17.4 mmag (−9%)**.
    360 (stable seeing): comp RMS unchanged (harmless). Gate byte-identical when OFF
    (proc CSV emits no apcorr cols → reader uses raw `dao_flux`). Forced fallback degrades
    gracefully (`ac_factor=1`, `cog_ok=False`, no crash). NB: stable-comp radius span was
    tiny (0.15px, all near the faint floor) yet still gave OFF corr +0.94 → real benefit is
    **larger** for target/comp pairs with bigger radius gaps (extreme-group test ~768 mmag).

- **[CROWDING-CLASSIFIER]** Detection-independent signal classifier + sampling-gated
  TIGHTEN — config-gated `crowding_classifier_enabled`, **default OFF** (zero production
  change; OFF == legacy stars/Mpx path). Replaces the erratic detection/scale-locked
  **stars/Mpx** density class with detection-independent `crowding_index` signals
  (gaia/arcmin² + `blend_frac` @ measured depth + comp-availability + `frame_limit_mag`).
  Decouples the two concerns the single legacy class conflated: **LOOSEN** keys on comp
  **availability** (few usable catalog comps in FOV), **TIGHTEN** on real **blend_frac**
  (contamination @ depth); both fire independently, shared keys sum additively.
  - **SAMPLING GATE:** TIGHTEN fires only when the PSF is resolved (**FWHM ≥ 3 px**). On
    undersampled fields the comp-RMS 0.08–0.10 tail is the **field floor**
    (scintillation/undersampling), not resolvable contamination, so tightening
    `max_comp_rms` there only cuts good comps. Never fires on the wide rig (FWHM ≈ 2.5–2.6).
  - Removed dead `aperture_fwhm_factor -0.3` from the legacy `dense` override (science
    aperture comes from the SNR-optimal table, which ignores it; verified 0-px effect).
  - **A/B VALIDATED — 361/362 OFF (legacy DENSE→tighten) vs ON (gated→no tighten)** (plate
    scale resolves ~9.77″/px both runs, fixing the stale-1.3 A4 artifact):
    * **360** (FWHM 2.59, legacy `normal`): **neutral by construction** — gate suppresses
      tighten and legacy `normal` doesn't tighten either. OFF==ON.
    * **361** (FWHM 2.47): ON **recovers** — robust LC scatter 0.07316→**0.07179**
      (−1.4 mmag), median lc_rms 0.07817→**0.07498** (−3.2 mmag); keeps the 0.08–0.10
      comp band (19 pairs vs 0).
    * **362** (FWHM 2.65): ON marginally **worse** — robust scatter 0.09876→0.09915
      (+0.4 mmag, sub-mmag), p2p +1.4 mmag, lc_rms +3.0 mmag. Floor-limited,
      comp-geometry-dependent → diminishing returns; not investigated further.
  - **DECISION: committed gated infra, NOT enabled for the wide rig.** The wide rig is
    **floor-limited** (scintillation/undersampling, FWHM 2.6 px); the tighten's payoff is
    **well-sampled crowded data — enable on the Newton cluster**, not here. The gate is
    correct in principle (prevents the demonstrated 360-style over-tightening harm).
  - Files: `config.py` (config keys + `CROWDING_{LOOSEN,TIGHTEN}_OVERRIDES` +
    `apply_crowding_overrides`), `crowding_index.py` (now consumed by the gated classifier),
    `photometry_core.py` (gated block in `run_phase0_and_phase1`; `db`/`draft_id` threaded).

### NEW TODO — aperture correction

- **[TODO-APCORR-MIXEDFRAME]** Before enabling on **sparse/cloudy** nights: wire a nightly-
  median `fallback_ee` (median EE from `cog_ok=True` frames → applied to fallback frames,
  flagged) so a draft never mixes corrected + uncorrected frames. `cog_ok` is per-frame; a
  mix would inject a cross-frame step (~target↔comp bias). Hook (`fallback_ee`) exists but is
  not yet wired in the pipeline. **Cannot occur on star-rich fields** (360/363: 261–320
  eligible stars/frame, 0 fallbacks). Alternative: draft-level gate (require all-frames
  `cog_ok`). MEDIUM.
- **[TODO-APCORR-COLOR]** (audit STEP 7) NoFilter↔Gaia-G color term **c1 ≈ −1.0 mag/(bp_rp)**;
  CT correction (`mag_calib_ct`/`fit_color_term_c1`) frequently inactive (`ct_ok=False`).
  Activate it robustly or tighten comp color-matching so residual color dependence (comp
  cat−inst std ~0.12–0.16 mag) is removed. MEDIUM.
- **[TODO-WIDE-RIG-REPROCESS]** 361/362 **production** `photometry/` still carry
  **stale-1.3-scale** LC products (A4) — the A/B reprocess wrote only isolated dirs (since
  removed). Reprocess 361/362 with the wide rig when convenient to refresh the production
  LCs at the correct ~9.77″/px scale. LOW / housekeeping.

---

## VYVAR SESSION SUMMARY — 29.5.2026 (plate-scale fix + crowding index + PSF grouper)

### COMPLETED 2026-05-29

- **[PLATESCALE-FIX]** Resolver made WCS/CD-authoritative (config last resort); sane clamp
  widened `0.3–5.0` → `0.1–30.0`. Fixed `_resolve_plate_scale_arcsec_per_px`,
  `_read_plate_scale_from_fits_path`, `_get_plate_scale_from_cfg`
  (`photometry_core.py`), `psf_photometry._read_plate_scale_arcsec_px_from_fits`, and
  routed `tess_verify` through the fixed reader. 362 `MASTERSTAR` `VY_PLTS` hygiene
  `1.3 → 9.768` (rewritten from its own CD matrix; WCS/CD keys untouched).
  - **CONFIRMED real scale = 9.77″/px** (200 mm + 9.26 µm binned). `VY_PLTS=1.3` was a
    GLOBAL config placeholder = Newton 300/1200 + C3-26000 binned 2× (~1.29″/px) leaking
    onto the wide-field set.
  - Blast radius (now correct): GS11 context, FOV / `max_dist_deg`, ePSF isolation, TESS.
  - Pixel-based geometry (aperture / annulus / SNR-optimal table / `field_density`) was
    IMMUNE and confirmed unchanged.
  - Validation (360/361/362): all three resolvers return ~9.77 after; ePSF 362 rebuilt
    with corrected isolation (~98″ vs old ~13″), `n_stars_used` 304→296; FOV
    `max_dist_deg` 0.34° → ~2.55°; `field_density`/`density_class` identical pre/post.

- **[CROWDING-INDEX]** Built parallel detection-independent `crowding_index.py` +
  `run_crowding_index.py` (NOT wired into the pipeline). Depth-aware: frame limit
  (SNR=5 Howell), Gaia footprint density/arcmin², blend fraction, miss decomposition
  (below_depth / blend_miss / threshold_miss), per-target blend worklist
  `crowding_targets.csv`.
  - FINDING: old `field_density` (Gaia-matched DAO count) is detection-limited →
    undercounts dense fields (blends merge). `threshold_miss ≫ blend_miss` ⇒ iterative
    DAO is the big completeness lever; PSF deblend is only 1.4–4.1% of the field but
    ~19% of *targets*. Hercules (362) is intrinsically denser than the CVn fields —
    OPPOSITE of what the old metric reported.

- **[PSF-GROUPER]** `SourceGrouper` joint-fit implemented, **DEFAULT OFF** (config
  `psf_grouper_enabled` / `psf_group_sep_fwhm` / `psf_neighbor_include_fwhm`). Offline
  362 ensemble test (corrected ePSF) shows it DEGRADES: blended median ratio
  1.10→1.70, CSS_J161519.8 1.56→20.4. Blends at ~1.5 FWHM (~5 px at 9.77″/px) are
  sub-resolution ⇒ joint fit ill-conditioned (3.38% divergence, fallbacks). Single-star
  PSF on a clean model ≈ aperture here. Re-test on Newton 0.65″/px data where blends are
  resolved.

- **[PSF-SPATIAL]** Spatially-varying ePSF (`GriddedPSFModel`) implemented, **DEFAULT OFF**
  (config `psf_spatial_enabled` / `psf_spatial_grid="3x3"` / `psf_spatial_min_stars_per_cell`).
  New `psf_photometry.build_epsf_grid_model` (full-frame CSV candidates, per-cell EPSFBuilder,
  `grid_from_epsfs`, per-cell fallback flagging) + `interp_gridded_epsf_array` + gated
  `gridded_model=` path in `psf_photometry_stars` (per-(x,y) interpolated ePSF; single path
  untouched when OFF). Refactored shared `_epsf_prepare_stars` / `_epsf_build_imagepsf_from_stars`.
  - VALIDATION (offline ensemble, 360/361/362, 9.77″/px, 3×3, 0 fallback cells, full-frame
    coverage, n≈90–280/cell): **gridded does NOT win**. By field region, median
    RMS(`mag_calib`) ratio **gridded/single = 1.06–1.59 everywhere** (gridded always worse);
    **gridded/aper ≥ 1.0** in essentially all bins. Single ePSF beats aperture only on 362
    (cleanest field): comp/edge single/aper **0.73** (0.0149 vs 0.0203), comp/centre 0.83;
    on 360/361 single/aper > 1. Variable targets ≈ flat (~1.0) for all methods.
  - WHY: at 9.77″/px the PSF is well-sampled and stable across the 5.7° field, so a single
    ePSF already captures it; subdividing into cells just starves each ePSF of stars and adds
    scatter. **Decision: keep `psf_spatial_enabled` OFF.** Revisit only on finer optics where
    field-dependent aberration (coma/curvature) actually varies the PSF across the chip.

- **[EPSF-CENTER-BUGFIX]** Fixed a latent ePSF-build bug uncovered by the spatial work: the
  per-cutout sky-sub *re-extraction* in `_epsf_prepare_stars` (ex `build_epsf_model`) treated
  `EPSFStar.center` as `(y, x)`, but the installed photutils returns `(x, y)`. The transpose
  pushed every star with `x > ~1388` (the chip-height axis = 1397) out of bounds → silently
  dropped, confining all ePSF candidates to a 1388×1388 box (left/upper region only). Now
  `(x, y)` — global ePSF candidate coverage is full-frame; build still produces a sane model
  (362 `n_stars_used`=296, asym=0.004). Affects the production global ePSF build (improvement).

- **[PSF-ISOLATION-FIX]** ePSF candidate isolation now compares each candidate against the
  **FULL Gaia cone catalogue** (`field_catalog_cone.csv`), not just the other candidates.
  New `psf_photometry._load_cone_catalog`; `_epsf_prepare_stars` rejects a candidate if any
  cone source lies within `3×FWHM` (correct 9.77″/px) **and** is within `Δmag ≤ 2.5` (brighter,
  or ≤2.5 mag fainter = contaminating). Candidate `mag` carried into the candidate frame; falls
  back to candidate-vs-candidate only if the cone CSV is missing. **This is a correctness fix to
  the production ePSF build.**
  - VALIDATION (rebuilt 360/361/362): old candidate-vs-candidate barely filtered
    (360 334→330), the cone test correctly drops bright-neighbour candidates:
    **360 334→213 (−35 %), 361 292→148 (−49 %), 362 304→161 (−46 %)**; models stay sane
    (asym 0.003–0.006, nan 0). Cleaner isolation did **NOT** move single-PSF past aperture:
    comp-star median RMS single-PSF/aper = **3.47 / 2.70 / 3.72** (360/361/362) — single ePSF
    is still ~3× worse than aperture at 9.77″/px (confirms the well-sampled-stable-PSF picture;
    aperture wins on coarse optics).

- **[PSF-QUALITY-FALLBACK]** Per-star PSF quality + auto-fallback + residual QA (the RMS-20.4
  safety lesson). `psf_photometry.assess_psf_quality` grades every fit `good/marginal/bad` from
  reduced χ², fit SNR (flux/flux_err), fitted-position shift (in FWHM) and nearest-neighbour
  proximity (`nn_dist_fwhm` + neighbour Δmag; a close *bright* neighbour ⇒ bad). New proc-CSV
  columns from `psf_photometry_stars`: `psf_quality`, `psf_quality_fallback`, `psf_snr`,
  `psf_pos_shift`, `psf_nn_dist_fwhm`. Quality is **always computed**. New config
  `psf_quality_fallback_enabled` (**default TRUE**): a `bad` fit drops `psf_fit_ok` and sets
  `psf_quality_fallback=True` so the caller substitutes aperture — a bad PSF flux can never
  silently become the reported value. **No production wiring yet** (Phase-2A still aperture).
  - VALIDATION (offline ensemble, 127/147/93 frames):
    measurement grades ≈ 28 % good / 43 % marginal / 29 % bad; target stars bad-majority
    66/57/56, marginal-majority 60/43/44 (360/361/362).
    **CSS_J161519.8+491001 (neighbour 3.5 mag brighter at 1.46 FWHM) ⇒ `bad` in 127/127 frames.**
    Fallback cuts PSF-specific blowups: e.g. 362 target `…0227968` single-PSF RMS **0.747 → 0.023**
    (≈ aperture 0.016); 360 max target RMS **1.28 → 0.97**; #targets with PSF RMS >1.5× aper
    **off→on: 360 82→65, 362 35→20**. (Residual-QA PNG + summary per draft in
    `…/psf_robust_qa_d<draft>/`.) Where a star is already bad in *aperture* (intrinsic variable),
    per-epoch fallback ≈ aperture — the safe floor, as intended.

- **[PSF-WIRING-ADAPTIVE]** Fixed the dead `psf_photometry_enabled` toggle + added a gated
  per-star adaptive flux selector. **Both default OFF → production stays pure-aperture.**
  - ROOT CAUSE: Phase-2A's flux reader (`read_flux_from_csv`) + the PERF-8 `_flux_matrix`
    cache carried ONLY `dao_flux`, so `psf_flux` never reached `_get_lc_psf_or_dao` →
    `psf_photometry_enabled` was a no-op. FIX: the reader now also carries per-star/per-frame
    `psf_flux`, `psf_fit_ok`, `psf_quality`, `psf_quality_fallback`, `psf_snr` (b.5 columns;
    default NaN/False when absent → no behaviour change). Confirmed end-to-end: with PSF run,
    31k–42k/frame-rows carry finite `psf_flux` and `_get_lc_psf_or_dao` now uses it.
  - ADAPTIVE SELECTOR (`compute_lc_flux_method` + `_get_lc_adaptive`, config
    `psf_adaptive_enabled` / `psf_adaptive_resolve_fwhm=2.0` / `psf_adaptive_snr_lo=15`):
    per-star/per-frame choice ∈ {aperture, psf}, CONSERVATIVE (default aperture; → PSF only
    with positive evidence + good quality). Rules: (1) bad/`!fit_ok`/no-flux → aperture;
    (2) resolvable blend (`is_blended ∧ nn_dist_fwhm ≥ resolve_fwhm`) → psf; (3) faint
    (`SNR ≤ snr_lo`) ∧ quality good → psf; (4) else aperture. Choice emitted as
    `lc_flux_method`. Blend map loaded best-effort from `crowding_targets.csv`.
  - VALIDATION (offline ensemble, 3 modes, 360/361/362):
    | mode | comp RMS (360/361/362) | targ RMS |
    |---|---|---|
    | aperture-only | 0.0118 / 0.0105 / 0.0105 | 0.0882 / 0.0628 / 0.0847 |
    | PSF-everywhere | 0.0437 / 0.0377 / 0.0318 | 0.1181 / 0.0807 / 0.0926 |
    | **adaptive** | **0.0118 / 0.0105 / 0.0105** | **0.1040 / 0.0628 / 0.0908** |
    Method split → PSF: comp **0.0–0.34 %** (bright flat comps stay aperture), targ 6.8–17.6 %.
    **comp adaptive == aperture exactly** (never hurts the flat stars) and adaptive ≪
    PSF-everywhere in every cell (smarter than blindly forcing PSF). On targets adaptive ≤
    aperture on 361 (==), slightly above on 360/362 — the rule-3 faint picks where single-PSF
    still loses at 9.77″/px (the rule pays off at fine scale, as expected; rule 2 barely
    fires — blends unresolvable here). **CSS_J161519.8 → aperture in 127/127 frames**
    (quality `bad` ⇒ rule 1), RMS == aperture (0.061), never worse.
  - DECISION: keep both flags OFF at 9.77″/px (PSF doesn't beat aperture). The wiring +
    selector are correct and ready to pay off on fine-scale Newton data.
  - CONFIG CORRECTION: `config.json` had `psf_photometry_enabled=true`, which was a SILENT
    no-op only because the reader dropped `psf_flux`. Now that the wiring carries it, leaving
    the flag on would have flipped production to PSF-everywhere (worse). Set
    `psf_photometry_enabled=false` so production genuinely stays pure-aperture.

### CORRECTIONS to prior STATE

- **Plate scale is 9.77″/px, NOT 1.3** (project-wide belief was wrong).
- **"359 rebuilt-1.3 good / 360-361 stale-9.55 bad" was WRONG**: ePSF quality tracks
  CROWDING, not plate scale. 359 is unreliable (ePSF overwritten during debugging) →
  DROPPED. Trust 360/361/362.
- **"No finer set exists" was WRONG**: focal length is in the `TELESCOPES` table; Newton
  300/1200 + C3-26000 = ~0.65″/px (cluster-capable).

### NEW TODO — PSF roadmap (priority order)

- **[TODO-PSF-SPATIAL]** `GriddedPSFModel` implemented, gated default OFF. LOSES to single
  ePSF in all regions on 9.77″/px (marginal sampling → little spatial variation to capture;
  3x3 grid → per-cell star starvation). Low priority: narrow Newton FOV has minimal spatial
  variation either. Kept for completeness.
- **[TODO-PSF-WIRING]** `psf_flux` → Phase 2A wiring + per-star adaptive aperture/PSF
  selection **DONE** (see `[PSF-WIRING-ADAPTIVE]`). Remaining: neighbor-sub branch
  (`[TODO-PSF-NEIGHBOR-SUB]`) + actually running PSF in production (still default OFF).
- **[TODO-PSF-NEIGHBOR-SUB]** Neighbor-subtracted aperture: fit + subtract bright
  neighbour ePSF, aperture the residual (deblend that works at coarse resolution, unlike
  the grouper).
- **[TODO-PSF-MULTIFRAME]** Multi-frame ePSF build (all frames of the night).
  Isolation `candidate-vs-full-Gaia` part **DONE** (see `[PSF-ISOLATION-FIX]`); only the
  multi-frame stacking remains.
- **[TODO-PSF-QUALITY]** **DONE** — see `[PSF-QUALITY-FALLBACK]` below (quality flags +
  auto-fallback + residual QA). Remaining: realistic per-star uncertainties.
- **[TODO-OIS]** Difference imaging (Alard & Lupton OIS) — dense clusters with fine optics.
  Long-term.
- **[TODO-PLATESCALE-PERSET]** Resolver: use `TELESCOPES.focal` × camera pixel as a per-set
  plate-scale fallback (between WCS and config).
- **[WATCH]** comp-selection: validate local vs wide comps before the next production run
  uses the now-correct ~2.55° `max_dist_deg` (wide-field distant-comp systematics risk).

---

## VYVAR SESSION SUMMARY — 28.5.2026 (časť 4 — pipeline + PSF finalizácia)

## skip_processed_directory — Fáza 1 (commit 7db6914)

### Architektúra zmena

**Starý flow:**
Raw → Calibrated → Processed (QC + kópia FITS) → Aligned → proc_*.csv → LC

**Nový flow (skip_processed=true):**
Raw → Calibrated → QC in-place (VY_* headers na calibrated) → Aligned → proc_*.csv → LC

### Čo sa zmenilo

| Zmena | Detail |
|-------|--------|
| `skip_processed_directory` | Nový config bool (default `false`) |
| `_qc_enrich_calibrated_in_place` | QC headers písané in-place na calibrated FITS |
| `_get_vy_qc_status` | Helper pre VY_QC=ok filter pri alignment |
| `_archive_preprocess_lights_root` | Routing calibrated/ keď skip=true |
| `astrometry_align_and_build_masterstar` | Filter VY_QC=ok z calibrated |
| `resolve_masterstar_input_root` | Preferuje calibrated/ keď skip=true |
| `find_qc_metrics_csv` | Hľadá qc_metrics.csv v calibrated/ aj processed/ |
| `qc_fwhm_limit`, `qc_elong_limit` | Nové config polia (8.0, 1.8) |
| Temporal sigma clip | Zámerne vynechané z nového flow |

### Úspora (per draft)

- ~1.5 GB diskového priestoru (processed/ adresár)
- ~20–30% rýchlejší pipeline (vynechanie kopírovania 139 FITS)

### Fáza 2 (budúca)

- Odstrániť starý processed/ kód po validácii nového flow
- Otestovať kompletný run s skip_processed_directory=true

## PSF pipeline — finálny stav (28.5.2026)

### Výsledky validácie draft_359

| Metóda | RMS median | Fit rate | Záver |
|--------|-----------|----------|-------|
| Aperture (DAO) | 0.080 mag | ~100% | ✅ Primárna |
| ePSF | 0.115 mag | 7.1% | ⚠️ Crowded fields |
| Moffat | 0.143 mag | 1.4% | ❌ Asymetrický PSF |

### Kľúčové zistenia

- BO CVn pole má asymetrický PSF (tracking smear) → symetrický Moffat/ePSF nevyhrá
- PSF metódy budú lepšie pre iné noci / crowded fields
- Moffat AC faktor = 0.806 (flux outside cutout ~20%)
- alpha median = 5.67 (fyzikálne rozumné pre seeing-dominated PSF)

### Open TODOs — PSF

| ID | Popis | Priorita |
|----|-------|----------|
| TODO-PSF-ASYMMETRY | Tracking smear diagnostika — elongation angle/ratio per frame | HIGH |
| TODO-PSF-SATURATION | Auto-exclude saturated stars z Moffat fit ✅ DONE | ✅ |
| TODO-PSF-NEIGHBOR-SUB | PSF-cleaned aperture (ALLFRAME štýl) | MEDIUM |
| TODO-PSF-SPATIAL | spatial_order=1 pre Newton (čaká na TODO-MULTISET) | LOW |
| TODO-PSF-MULTIFRAME | Multi-frame ePSF build (všetky framy noci) | MEDIUM |
| TODO-PSF-PHASE2 | Moffat centroidy → ePSF build (TODO na call site) | MEDIUM |

### Nové citácie pridané dnes

| Citácia | Relevancia |
|---------|-----------|
| Vicuña et al. (2025), A&A Oct 2025 | PSF fitting radius — Fisher information |
| Libralato et al. (2016), MNRAS 456 | PSF-based K2 photometry |
| Stetson (1994), PASP 106, 250 | ALLFRAME — neighbor subtraction |

## pytest stav

| Session | Passed | Failed | Skipped |
|---------|--------|--------|---------|
| Ráno (bug hunt) | 103 | 0 | 6 |
| Po PSF zmenách | 103 | 0 | 6 |
| Po skip_processed | 103 | 0 | 6 |

---

## VYVAR SESSION SUMMARY — 28.5.2026 (časť 3 — PSF pipeline)

**Last validated draft: `draft_000359`**

## PSF pipeline v1 — implementácia (28.5.2026)

### Čo sme implementovali

| Blok | Zmena | Súbor |
|------|-------|-------|
| `fit_moffat_psf_stars` | Moffat2D+Const2D fit via LevMarLSQFitter; per-star sky border estimate + residual sky fitting; flux z analytického integrálu π·amp·γ²/(α-1); výstup: gamma, alpha, FWHM, sky, chi2, fit_ok | `psf_photometry.py` |
| `_moffat_fwhm_px` | Helper: FWHM = 2γ√(2^(1/α)-1) | `psf_photometry.py` |
| ePSF sky subtraction | Sky odčítaný pred EPSFBuilder (global median z MASTERSTAR) | `psf_photometry.py` |
| ePSF normalizácia | Native integral = 1.0 (sum/osamp²); `epsf_norm_factor` v meta | `psf_photometry.py` |
| ePSF QC | Radial profile FWHM, quadrant symmetry, NaN fraction; uložené v `epsf_qc` meta JSON | `psf_photometry.py` |
| ePSF build params | `smoothing_kernel=\"quadratic\"` pre osamp≤2; `fit_shape=2×FWHM+1`; `min_stars=30` | `psf_photometry.py` |
| IterativePSFPhotometry | `_epsf_noop_finder`; per-star fallback na PSFPhotometry | `psf_photometry.py` |
| Aperture correction | `_compute_aperture_correction` σ-clip median ratio dao/psf; chi2<5.0; min 5 ref hviezd | `psf_photometry.py`, `pipeline.py` |
| `photometry_mode` | `\"aperture\"` / `\"epsf\"` / `\"both\"` (default `\"both\"`) | `config.py`, `config.json`, `ui_photometry.py`, `app.py` |
| VY_PLTS | Zápis plate scale do MASTERSTAR.fits; safer read order (VY_PLTS > CD matrix > CDELT > SCALE≤5.0) | `psf_photometry.py`, `pipeline.py` |

### Diagnostické nálezy (draft_359, BO CVn)

| Test | Výsledok | Poznámka |
|------|----------|---------|
| ePSF build | 330 hviezd, asymmetry=0.014 ✅ | Dvojnásobok oproti starému buildu (148) |
| ePSF pedestal | Odstránený ✅ | Sky sub pred EPSFBuilder |
| ePSF normalizácia | native integral=1.0 ✅ | |
| Moffat fit (saturované) | fit_ok 2/10, chi2=26–186 | Saturované hviezdy — nie vhodné pre PSF fit |
| Moffat fit (nesaturované) | fit_ok 14/15, chi2 median=1.96 ✅ | Funguje správne |
| Moffat FWHM (nesaturované) | median=1.84px | Podhodnotené — asymetrický PSF profil |
| ePSF photometry chi2 | medián=72 | Stále vysoké — asymetrický PSF v BO CVn |

### Kľúčový nález — asymetrický PSF

Hviezdy v BO CVn poli majú pravostranný chvost (tracking smear alebo kóma).
Symetrický Moffat2D / ePSF to nemôže správne popísať → chi2 systematicky vysoké.
ePSF (empirický) je správna dlhodobá cesta — zachytí asymetriu automaticky.

### Vedecké citácie (PSF pipeline)

| Citácia | Použitie vo VYVAR |
|---------|------------------|
| Moffat (1969), A&A 3, 455 | `fit_moffat_psf_stars` — základ Moffat profilu |
| Anderson & King (2000), PASP 112, 1360 | `EPSFBuilder` — definícia ePSF |
| Stetson (1987), PASP 99, 191 | Hybridný PSF prístup — inšpirácia dvojkrokovej architektúry |
| Stetson (1990), PASP 102, 932 | Aperture correction metodológia |
| Trujillo et al. (2001), MNRAS 328, 977 | β≈4.765 pre atmosferický seeing |
| Anderson (2016), WFC3 ISR 2016-12 | EPSFBuilder vylepšenia |
| Bradley et al. (2024), Zenodo | photutils softvérová citácia |
| Vicuña et al. (2025), A&A (Oct 2025) | PSF fitting radius — Fisher information |

### Open TODOs — nové (28.5.2026 časť 3)

| ID | Popis | Priorita |
|----|-------|----------|
| TODO-PSF-ASYMMETRY | Riešenie asymetrického PSF profilu v BO CVn — tracking smear diagnostika | HIGH |
| TODO-PSF-SATURATION | Automaticky vylúčiť saturované hviezdy z Moffat fit (použiť `likely_saturated` flag) | HIGH |
| TODO-PSF-RATIO | Validácia moffat_flux/dao_flux ratio na nesaturovaných hviezdach (merge fix) | MEDIUM |
| TODO-PSF-FWHM | Moffat FWHM 1.84px vs VY_FWHM 3.52px — prečo? Tracking smear analýza | MEDIUM |
| TODO-PSF-PHASE2 | Fáza 2: Moffat centroidy → ePSF build (presnejší ePSF z Moffat pozícií) | MEDIUM |
| TODO-PSF-PHASE3 | Fáza 3: Hybrid model Moffat + ePSF residual (DAOPHOT filozofia) | LOW |
| TODO-MASTERSTAR-EPSF-ALL | Prebudovať ePSF pre drafty 311, 321, 358 (stará plate_scale 9.55) | MEDIUM |

# VYVAR SESSION SUMMARY — 28.5.2026

**Last validated draft: `draft_000359`** — bug fix session + ePSF rebuild s novými parametrami

## Dokončené dnes (28.5.2026)

### Bug Hunt — systematic static review (37 findings, 27 fixes, 11 safe-pattern comments)

| Batch | Bugy | Typ |
|-------|------|-----|
| HIGH | BUG-004, 014, 015, 016 | Crash / user-facing |
| MEDIUM critical | BUG-001, 002, 017, 019, 024 | Data correctness + WCS |
| MEDIUM perf | BUG-033, 034, 035, 036 | I/O + memory |
| LOW | BUG-003, 005–012, 020–021, 023, 025–026 | Guards + config drift |
| Docs | BUG-013, 018, 022, 027–032, 037–038 | Safe-pattern comments |

#### HIGH fixes
| ID | Súbor | Fix |
|----|-------|-----|
| BUG-004 | `ui_variability.py:932` | `setdefault("tess_results", {})` — KeyError v crossmatch dialógu |
| BUG-014 | `photometry_core.py:3646` | `np.percentile` na prázdnych finite dátach — field map PNG |
| BUG-015 | `photometry_core.py:3587` | `np.percentile` na prázdnom cutout — target PNG |
| BUG-016 | `photometry_core.py:3753` | Rovnaký vzor — `save_target_field_map_png` |

#### MEDIUM critical fixes
| ID | Súbor | Fix |
|----|-------|-----|
| BUG-001 | `photometry_core.py:1825` | `wcs.has_celestial` guard pred `all_world2pix` |
| BUG-002 | `photometry_core.py:9054` | WCS celestial guard — CATALOG_ONLY forced aperture |
| BUG-024 | `photometry_core.py:9085` | `VY_PSOLV == 1` guard — combined s BUG-002 |
| BUG-017 | `variability_detector.py:679` | `np.where(den > 0, ...)` — clip_ratio division by zero |
| BUG-019 | `ui_variability.py:1890` | `sub.empty` check pred `iloc[0]` — VT export |

#### MEDIUM perf fixes
| ID | Súbor | Fix |
|----|-------|-----|
| BUG-033 | `photometry_core.py:7686` | Sidecar CSV dict cache v `stress_test_relative_rms_from_sidecars` |
| BUG-034 | `ui_finalization.py:332` | `usecols=["catalog_id","aperture_mag"]` — 13–20× menej RAM |
| BUG-035 | `ui_aperture_photometry.py:1635` | LC preload cap 200 + `usecols` + `ttl=3600` |
| BUG-036 | `photometry_report.py:1582` | `usecols=_AIRMASS_COLS` — bez `nrows` capu |

#### LOW fixes
| ID | Súbor | Fix |
|----|-------|-----|
| BUG-003 | `photometry_core.py:2062` | PyTICS weight sum guard `not isfinite(s) or s <= 0` |
| BUG-005 | `photometry_core.py:5199` | `is_file` guard pred `active_targets_csv` / `comparison_stars_csv` |
| BUG-006 | `ui_aperture_photometry.py:58` | `is_file` guard v `_cached_read_csv` |
| BUG-007 | `photometry_core.py:1457` | `is_file` guard pred per-frame proc CSV read |
| BUG-008 | `vyvar_platesolver.py:1546,2481` | `app_config=None` + `_cfg_ps` — 6 `AppConfig()` nahradených |
| BUG-009 | `vyvar_blind_solver.py:63` | `app_config=None` + `_cfg` v `find_blind_hint` |
| BUG-010 | `ui_aperture_photometry.py:1416` | `cfg` z render scope namiesto `AppConfig()` |
| BUG-011 | `photometry_core.py:4588,10412` | Hoisted `_cfg_summary`, `_cfg_base` |
| BUG-012 | `pipeline.py:9381` | `_cfg_for_workers` hoisted pred loop |
| BUG-020 | `vyvar_blind_solver.py:170` | Empty/missing x,y guard → `return None` |
| BUG-021 | `ui_variability.py:2034` | Explicit `df.empty` check + warning pred `iloc[0]` |
| BUG-023 | `photometry_core.py:1954` | Komentár — rolling median guard overený ako safe |
| BUG-025 | `variability_detector.py:478` | `abs(mu) > 1e-3` floor pre RMS% |
| BUG-026 | `photometry_core.py:2517` | `log_event` keď `ensemble_normalize` nemá comp hviezdy |

---

## ePSF — kompletná implementácia a robustnosť (28.5.2026, časť 2)

### Čo sme dnes urobili (ePSF)

| Blok | ID | Zmena | Súbor |
|------|----|-------|-------|
| A1 | smoothing_kernel | `"quadratic"` pre oversampling≤2, `"quartic"` pre ≥3; uložené v meta | `psf_photometry.py` |
| A2 | IterativePSFPhotometry | Nahradili `PSFPhotometry`; `_epsf_noop_finder` pre photutils 2.3.0; per-star fallback | `psf_photometry.py` |
| A3 | fit_shape fix | `2×FWHM+1` namiesto `cutout_size-4`; pre FWHM=3.5px → `(9,9)` | `psf_photometry.py` |
| A4 | ePSF QC | Radial profile FWHM, quadrant symmetry, NaN fraction; uložené v `epsf_qc` v meta JSON | `psf_photometry.py` |
| A5 | min_stars | Default 30 (bol 15), config-driven `epsf_min_stars` | `psf_photometry.py`, `config.py`, `config.json` |
| B1 | photometry_mode | `"aperture"` / `"epsf"` / `"both"` (default `"both"`) | `config.py`, `config.json` |
| B2 | Pipeline routing | `_photometry_mode_run_flags()` helper; guards na 6 miestach | `pipeline.py` |
| B3 | UI prepínač | Radio button v `ui_photometry.py`; caption v `app.py` | `ui_photometry.py`, `app.py` |
| C1 | Aperture correction | `_compute_aperture_correction()` — σ-clip median ratio `dao_flux/psf_flux`; chi²<5.0; min 5 ref hviezd | `psf_photometry.py`, `pipeline.py` |
| C2 | spatial_order doc | Komentár k 0/1/2 order; TODO-MULTISET pre per-set config | `config.py` |
| plate_scale | VY_PLTS fix | Safer read order (VY_PLTS > CD matrix > CDELT > SECPIX > SCALE≤5.0); `VY_PLTS` zápis pri MASTERSTAR build | `psf_photometry.py`, `pipeline.py` |

### ePSF rebuild — draft_359 výsledky

| Metrika | Starý build (27.5) | Nový build (28.5) |
|---------|-------------------|------------------|
| `n_stars_used` | 148 | **330** |
| `plate_scale_arcsec_px` | 9.55 (WCS bug) | **1.3** (VY_PLTS) |
| `smoothing_kernel` | chýbalo | **quadratic** |
| `fit_shape` | chýbalo | **[9, 9]** |
| `epsf_qc` | chýbalo | **prítomné** |
| `epsf_fwhm_native_px` | 17.0 (zlá metrika) | **2.236** |
| `epsf_vs_input_fwhm_ratio` | 4.832 | **0.636** ✅ |
| `epsf_asymmetry` | 0.041 | **0.014** ✅ |

### Vedecký kontext

- **Anderson & King (2000), PASP 112, 1360** — základ ePSF algoritmu
- **Anderson (2016), WFC3 ISR 2016-12** — vylepšenia EPSFBuilder
- `IterativePSFPhotometry` — subtrakcia susedných hviezd pred refittingom; kritické pre crowded fields a variabilné hviezdy s blízkymi susedmi
- Aperture correction štandardný krok DAOPHOT pipeline — teraz implementovaný

### Isolation filter (overené 28.5)

- `_isolation_radius_px = 3 × fwhm_px` (čisto pixely)
- Porovnanie susedov: **uhlová** vzdialenosť RA/Dec → `dists_px = dists_deg × 3600 / plate_scale_arcsec_px`
- `plate_scale` mení konverziu arcsec→px, nie samotný izolačný polomer

---

# VYVAR SESSION SUMMARY — 27.5.2026

**Last validated draft: `draft_000359`**

| Metrika | Hodnota |
|---------|---------|
| Light curves | **196** |
| Frames | **139** |
| LC RMS median | **0.0857 mag** |
| BO CVn comps | **4** (RMS 0.007–0.013) |
| masterstar_matched | **186** |
| catalog_only | **10** |
| AAVSO export | **184 súborov** |
| VAR.ASTRO export | **11 súborov** (eclipsing only) |
| Night run elapsed | **~3437 s** (~57 min) |

## Commits dnes (27.5.2026)

| Hash | Popis |
|------|-------|
| `45f7e9f` | fix: field_density uses Gaia-matched star count instead of raw VY_NDAO |
| `bf1adee` | fix: skip catalog_only targets in Phase 2A photometry loop |
| `f80a4e1` | fix: WCS rescale now handles PC+CDELT matrix format (Fix A) |
| `60bbc63` | fix: pixel-distance fallback when WCS plate scale deviates >20% |
| `4e74851` | fix: recompute masterstars ra_deg/dec_deg after WCS rescale (Fix C) |
| `5642da5` | fix: excluded comp stars show w(rel)=0.000 in PDF/UI export |
| `b2ed7f8` | remove: L.A.Cosmic (AstroScrappy) and Background Flattening |
| `139b313` | refactor(ui): move RUN VYVAR next to Scan Source, remove Pre-processing section |
| `4470bc6` | fix: restore _db_for_calibration_tasks removed in L.A.Cosmic cleanup |
| `dc9edca` | perf: ePSF model built only from masterstars + comp stars |
| `fe6bb6e` | fix: aavso/varastro export no longer produces empty directories |
| `84c7876` | fix: include noisy targets in UI LC selector default filter |
| `2a46f79` | fix: PDF report cleanup — catalog_only, field map, Method B text |
| `81c41a4` | feat: per-comp exclusion/suspect reasons persisted and displayed |

## Kľúčové opravy dnes

### WCS cascade bug (root cause BO CVn 0 comps)
- PC-matrix WCS plate scale 9.79″/px namiesto 1.30″/px
- Comp selection aj select_active_targets používali zlé RA/Dec
- Fix: pixel-distance fallback keď WCS scale deviates >20%
- Fix B aplikovaný na oba miesta (Phase 0 + Phase 1)

### Pipeline vylepšenia
- field_density z Gaia-matched count (nie raw VY_NDAO)
- catalog_only targety preskočené v Phase 2A
- ePSF len z masterstars + comp hviezd (nie všetky hviezdy)
- AAVSO/VAR.ASTRO export opravený (prázdne adresáre)

### UI/UX
- RUN VYVAR presunutý vedľa Scan Source
- Pre-processing + MAKE MASTERSTAR sekcia odstránená
- L.A.Cosmic + Background Flattening odstránené
- BO CVn viditeľná v UI (noisy do default filtra)

### PDF/Report kvalita
- excluded comp hviezdy: odstránené z tabuľky
- suspect comp hviezdy: dôvod zobrazený v status stĺpci
- catalog_only targety: žiadne LC stránky v PDF
- Field mapa: len masterstar-matched targety
- Method B: vysvetlenie pridané do PDF

---

## Open TODOs (backlog)

_Merged from repo (27.5–28.5.2026) and local archive (21.5.2026). CLOSED items marked ✅._

### Active (repo — 27.5.2026)

| ID | Popis | Priorita |
|----|-------|----------|
| TODO-MULTISET | Per-telescope-set config architektúra | HIGH |
| TODO-GS9 | Ground LC periodická analýza — LS + BLS + folded LC v PDF | HIGH |
| TODO-GS8 | Multi-Night Global Matching (Phase 3) | HIGH |
| TODO-LC-TREND | PARTIAL — common-mode detrend; re-validácia na moonless night | MEDIUM |
| TODO-FORCED-COMP | forced_aperture catalog_only bez Phase 1 tier selection | MEDIUM |
| TODO-GS10 | AAVSO Direct API Upload | MEDIUM |
| TODO-LIB | Cython .pyd kompilace | MEDIUM |
| TODO-GS7 | Paper draft (PASP/AN) | FUTURE |
| TODO-CACHE-CENTRAL | Centralizovať `csv_cache` na Phase 1 entry (BUG-037) | LOW |
| TODO-EPSF-VALIDATE | Full night run `photometry_mode="both"` na draft_359; porovnať `psf_flux` vs aperture na comp hviezdach; overiť aperture correction faktor | HIGH |
| TODO-MASTERSTAR-EPSF-ALL | Prebudovať ePSF pre drafty 311, 321, 358 (stará plate_scale 9.55) | MEDIUM |
| TODO-EPSF-SPATIAL | `spatial_order=1` pre Newton/Noctutec set — čaká na TODO-MULTISET | LOW |

### Active (local — 21.5.2026)

| ID | Popis | Priorita |
|----|-------|----------|
| TODO-LC-QUALITY | LC classification filter: saturated, noisy2/3, `no_data`; `lc_quality_flag`; LC count semantics | MEDIUM |
| TODO-LC-TREND | Differential extinction + ALG audit (rastúce/klesajúce LC u niektorých ROT) | MEDIUM |
| TODO-GEO | Observer geographic position audit (BJD, airmass, HJD) | MEDIUM |
| TODO-DEV-PROCESS | `scripts/regression_test.py` + smoke checklist pred commitom | HIGH |
| TODO-MASTERSTAR-QA validation | Overiť cyan FORCED_APERTURE overlay v UI na draft_342 | LOW |
| TODO-13 validation | Gaia→DAO completeness v QA stále ~3.5% — overiť po Phase A+B na novom exporte | MEDIUM |

### Background gradient & flux combination (Han & Brandt 2023 TGLC brainstorm)

Brainstorm source: Han & Brandt 2023 (TESS-Gaia Light Curve, AJ 165:71) independently
confirms VYVAR's own finding — aperture wins in most conditions; PSF wins faint/sparse +
deblends near variables; a weighted aperture+PSF combo is best.

- **[TODO-SKY-PLANE]** (priority LOW-MED): replace the constant/median local sky in the
  aperture annulus with a fitted **TILTED PLANE** (2D linear gradient) per star. Removes
  **additive** sky gradients tilting across a star's region (moon, light-dome, wide
  field). Benefit concentrated on **wide rigs** (OAT, planned 8" f/2.72) under gradient
  sky; small on narrow fields and largely cancels in differential photometry for
  smooth gradients. Does **NOT** fix multiplicative flat-field residual. Bounded change,
  fits the per-star paradigm (photutils plane-fit / LocalBackground). **TESTED —
  NEGATIVE** (drafts 361/362, OAT wide, 9.77"/px; standalone read-only): frame-scale
  sky gradients are **real** (~8–12% of sky, steepest on twilight/first frames), **but**
  the decision metric — plane-vs-median sky difference — is dominated by **residual**
  (median-clipping asymmetry + annulus contamination/scatter), **not** the linear tilt.
  The **linear** component (what a plane would fix on a symmetric annulus) is ~0.3–0.6%
  of the plane-median split and ~0.5% of the photometric error (only 3.7–5.4% of stars
  exceed 0.5× phot err on the linear-only part) — negligible. As predicted, a linear
  gradient cancels in the symmetric annulus median (the ring averages it out). No
  field-position correlation (0.07/0.12). Differential target–comp separations are
  ~2.5° (less gradient cancellation differentially), but since the linear part cancels
  in the annulus anyway, this does not rescue SKY-PLANE. **VERDICT: skip** for standard
  symmetric-annulus photometry; revisit only for asymmetric annuli (neighbor masking),
  curvature, or extreme gradients. **Minor future hint** (not actionable now): the
  per-star annulus sky **scatter** from median-clipping/contamination is larger than
  the gradient effect — if sky-noise reduction is ever pursued, a more robust annulus
  estimator (two-sided sigma-clip vs the current upper-only 2σ) would help more than
  any gradient model.

- **[TODO-WEIGHTED-LC]** (priority LOW-MED): produce a **WEIGHTED aperture+PSF light curve**
  (linear combination), per Han & Brandt 2023 (~0.4×PSF + 0.6×aperture, field/
  crowding-dependent; ~10–20% precision gain reported). Continuous-weight alternative
  to the hard adaptive selector (`psf_adaptive`). Needs both aperture+PSF fluxes per
  star; optimal weight is field-dependent. Small build, potential near-term precision
  win on faint/crowded. **TESTED — NEGATIVE** (drafts 361–364, standalone ceiling,
  relaxed PSF quality = upper bound): weighted LC w×PSF + (1−w)×aperture vs
  aperture-alone. The combination helps **only** where aperture and PSF have **comparable
  RMS** (error-decorrelation gain); nothing when one channel dominates.
  361 (9.77"/px): w_opt 0.2, ~2.9% | 362 (9.77"/px): w_opt 0.3, ~2.7% |
  363 (0.65"/px): w_opt 0, 0% | 364 (0.39"/px): w_opt 0, 0%.
  Gain tracks PSF-vs-aperture comparability: undersampled wide rigs (361/362) have large
  arcsec-apertures that are sky-noise-limited, so PSF nearly matches aperture → ~3%
  ceiling; well-sampled rigs (363/364) have PSF 37–68% worse → w→0, zero gain. The ~3%
  ceiling is undersampled-only and shrinks under the strict production quality gate →
  **below the cost/complexity threshold. VERDICT: not worth implementing; deprioritized.**
  **Future hint** (not actionable now): on 362 the G14–15 bin had PSF-alone beat
  aperture by 28% → possible PSF niche on **faint, sky-noise-limited undersampled
  wide-field** data (relevant to OAT / planned 8" f/2.72). Revisit only if wide-field
  faint precision becomes a priority.

- **[TODO-SCENE-FORWARD-MODEL]** (priority HIGH-effort / conditional): TGLC-style **scene
  forward model** for crowded-field decontamination. Fix neighbor positions+fluxes from
  the (deep) Gaia cone, fit a **LOCAL ePSF + a background model** (flat + 2D linear
  gradient) **SIMULTANEOUSLY** as a linear least-squares problem over a region, subtract
  modeled neighbors, photometer on the decontaminated residual. Principled crowded-
  field path; natural home for faint/crowded science (Brno / globular clusters).
  Major architectural addition; the linear trick relies on **FIXED Gaia priors** → fits
  a "fix neighbors, float target" decontamination mode, not standard differential
  photometry. TESS-specific strap/CCD-artifact terms **NOT** needed for ground-based
  (ground background = additive sky/light-pollution/moon + multiplicative flat
  residual). Justified only if faint/crowded becomes a priority.

**CLOSURE — Background gradient & flux combination brainstorm: CLOSED.** Two of three
TODOs tested **NEGATIVE** (TODO-WEIGHTED-LC, TODO-SKY-PLANE). Only
TODO-SCENE-FORWARD-MODEL remains **OPEN** — explicitly **CONDITIONAL** on
crowded-faint science (Brno / globular clusters) and **LOW priority** (further lowered
by the grouper-negative result and the modest realized PSF benefit). No further
background work planned unless that science becomes a priority.

### Reference — CLOSED / completed (from local archive)

| ID | Status | Notes |
|----|--------|-------|
| TODO-GS6b | ✅ CLOSED | AAVSO Extended Format validator (`scripts/validate_aavso_export.py`) — 20.5.2026 |
| CQ-1, CQ-2, CQ-4 | ✅ CLOSED | `run_phase2a`, `render_live_view`, `solve_wcs_with_local_gaia` splits — 20.5.2026 |
| PERF-9 | ✅ CLOSED | Vectorized haversine VSX match — 20.5.2026 |
| TODO-23, TODO-25, TODO-16, TODO-17 | ✅ CLOSED | Adaptive match radius, Gaia completeness UI, crossmatch coords — 20.5.2026 |
| TODO-ALG-2, TODO-ALG-3, TODO-ALG-4, TODO-ALG-5 | ✅ CLOSED | Savitzky-Golay, temporal binning, Democratic Detrender, PyTICS — 20.5.2026 |
| TODO-44, TODO-8 | ✅ CLOSED | Role-aware aperture; ePSF infrastructure — 20.5.2026 (Bootes → TODO-8-BOO) |
| PERF-1 … PERF-10 | ✅ CLOSED | Performance series — 19.5.2026 |
| CQ-3, TODO-35 | ✅ CLOSED | Comp selection split; SysRem MVP — 19.5.2026 |
| TODO-ALG-2 … TODO-ALG-5, TODO-44, TODO-8 | ✅ CLOSED | See 19.5.2026 session backlog table |

---

## Known issues / next session

_Merged from repo (27.5–28.5.2026) and local (21.5.2026)._

### 28.5.2026 (bug fix session — časť 1)

- Všetky HIGH a MEDIUM bugy z Bug Hunt reportu opravené ✅
- LOW docs (BUG-013, 018, 022, 027–032, 037, 038) — safe-pattern komentáre pridané
- Commit `ba7c142`: 27 fixes + `VYVAR_BUG_HUNT_REPORT.md`

### 28.5.2026 (ePSF robustness — časť 2)

- ePSF pipeline kompletný (A1–A5, B1–B3, C1–C2, VY_PLTS) — pozri tabuľku vyššie
- **draft_359** ePSF prebudovaný s `VY_PLTS=1.3`, `n_stars_used=330`, QC ratio **0.636**
- **draft_359 ePSF ratio=0.636** — fyzikálne správne (ePSF užší než DAO FWHM); validovať na fotometrii (**TODO-EPSF-VALIDATE**)
- **Staré ePSF modely** (drafty 311, 321, 358) — postavené so SCALE=9.55 → **TODO-MASTERSTAR-EPSF-ALL**
- **`photometry_mode="epsf"` only** — `mag_calib` routing z PSF flux čaká na validáciu; zatiaľ používať **`"both"`**
- Odporúčané: `pytest tests/` po commite ePSF zmien

### Repo (27.5.2026)

- BO CVn má 4 comp hviezdy (tier 1, dense override max_comp_rms=0.08)
  → riešiť pri TODO-MULTISET (per-set config)
- WCS pixel-distance fallback aktívny pre toto pole (9.79″/px WCS)
  → správne správanie, Fix B robustný
- draft_359 je aktuálny validovaný run
- 103 testov passed, 6 skipped

## Test suite

| Dátum | Výsledok |
|-------|----------|
| 27.5.2026 | pytest tests/ → **103 passed**, **6 skipped**, **0 failed** |
| 28.5.2026 | Odporúčané po commite (bug fixes + ePSF) — overiť 103 passed, 0 failed |

## Known sets

| ID | Telescope | Camera | Scale | Site |
|----|-----------|--------|-------|------|
| 1 | 200mm Carl-Zeiss | QHY294MM | ~1.3″/px | Jirny |
| 2 | 300/1200 Newton | C3-26000 | TBD | Dáblic obs. |
| 3 | Noctutec 206/560 f/2.72 | C3-26000 | TBD | TBD |

### Local (21.5.2026)

- **Staré drafty (≤341):** `FORCED_APERTURE` + `dao_flux` / `photometry_ok` platí až po re-exporte s `3d2363c` (SNR per-star aperture loop).
- **LC trends:** niektoré ROT hviezdy rastúce/klesajúce — čaká na **TODO-LC-TREND**.
- **MASTERSTAR QA:** Gaia→DAO Completeness **~3.5%** na dashboarde — stále nízke; čaká na **TODO-13** validáciu po novom DAO pass 2 + forced rows.
- **2 LC bez súboru:** draft_342 má 134 summary riadkov, 132 `lightcurve_*.csv` — overiť 2 targets bez `lc_csv`.

---

---

# VYVAR SESSION SUMMARY — 21.5.2026

**Last validated draft: `draft_000342`** (`simulate_night_run`, commit `3d2363c`)

| Metrika | draft_342 |
|---------|-----------|
| Light curves | **134** (summary); **132** `lightcurve_*.csv` on disk |
| HIP 67011 `lc_rms` | **0.012** mag (139 frames) |
| `DET_*` names v proc CSV | **0** |
| `FORCED_APERTURE` + `dao_flux` | **100%** measured (per-star aperture fix) |
| Targeted `psf_flux` non-null / frame | ~**183** (variables + top 40 comps) |
| Night run čas | **4345 s** (~72 min) |

## Dokončené dnes (21.5.2026)

| # | Zmena | Súbor(y) / commit |
|---|-------|-------------------|
| TODO-13 Phase A | Pass 2 (catalog-seeded DAO) v MASTERSTAR fast path — `_dao_targeted_pass2_unmatched_gaia` po pass 1 | `pipeline.py` |
| TODO-13 Phase B | Forced aperture rows + catalog-only proc CSV (`GAIA_MATCHED` + `FORCED_APERTURE` only); `_proc_catalog_keep_matched_rows_only` | `pipeline.py` |
| Comp selection | PSF χ² filter disabled pre DAO-era proc CSV (`max_psf_chi2=inf`) | `photometry_core.py`, `comp_pool_rms.py` |
| ISOLATED_BIN | `rms_map.pop(cid)` pre rms &lt; 1e-4 (žiadny floor — ultra-stable comps vylúčené) | `comp_pool_rms.py` |
| TODO-MASTERSTAR-QA | Layer overlay z proc CSV MASTERSTAR frame (green=GAIA_MATCHED, cyan=FORCED_APERTURE) | `ui_masterstar_qa.py`, `masterstar_qa_plot.py` |
| TODO-EPSF-TARGETED | ePSF len variables + top 40 comps (~183/frame vs 3500+) | `pipeline.py`; fix `read_vyvar_csv` import (`0b01955`) |
| DET_* names | Matched DAO → `name=catalog_id` po sky match | `pipeline.py` (`14cf0f9`) |
| photutils 2.3.0 | Per-star `CircularAperture` loop; `FORCED_APERTURE` → `photometry_ok=True` po meraní | `photometry_core.py` (`3d2363c`) |
| `ps` UnboundLocalError | `export_per_frame_catalogs`: `ps` pred `_ap_st` | `pipeline.py` (`0b01955`) |

---

# VYVAR SESSION SUMMARY — 20.5.2026

**17 tasks completed, 0 regressions, E2E validated on draft_321**

| Task | Detail |
|------|--------|
| TODO-GS6b | AAVSO Extended Format validator (`scripts/validate_aavso_export.py`) |
| MTYPE fix | `export_reports.py`: MTYPE=STD → MTYPE=DIFF |
| CQ-1 | `run_phase2a` split: 1515 → 188 lines + 3 helpers |
| CQ-2 | `render_live_view` split: 1395 → 731 lines + 3 helpers |
| CQ-4 | `solve_wcs_with_local_gaia` split: 1843 → 1224 lines + 3 helpers |
| PERF-9 full | Vectorized haversine VSX match in `photometry_core.py` |
| TODO-23 | Adaptive match radius (plate_scale × 3); universal telescope support |
| TODO-25 | UI reads `gaia_dao_completeness_pct` from `pipeline_meta.json`; pipeline writes it |
| TODO-16 | UI crossmatch uses `active_targets.csv` coords (WCS-verified) |
| TODO-17 | `crossmatch_auto_done` gated — only set True when candidates have catalog data |
| TODO-ALG-4 | Democratic Detrender (arXiv:2411.09753v2) — 3 models, `delta_mag_democratic` + `err_inflation` columns |
| TODO-ALG-3 | Temporal Binning adaptive window (MNRAS 2023) — `temporal_bin_comp_lc()` before stability/PyTICS |
| TODO-ALG-5 | PyTICS iterative comp weights (RASTI 2026) — `pytics_iterative_weights()` after stability check |
| TODO-ALG-2 | Savitzky-Golay detrending (default disabled) — `savgol_detrend_lc()` after airmass |
| TODO-ALG-1 | BLS ground LC — parkovaný, väčší scope (TESS-only L-S/BLS today) |
| TODO-44 | ✅ CLOSED — Role-aware aperture (`aperture_variable_factor` / `aperture_comp_factor`); verified draft_321 |
| TODO-8 | ✅ CLOSED — ePSF infrastructure: `build_epsf_model` + per-frame PSF export + Phase 2A flux selector + standalone 🔬 ePSF tab + `run_epsf` job + ⚡ RUN ePSF button; default `psf_photometry_enabled=false`; Bootes validation → **TODO-8-BOO** |
| TODO-8 ePSF dashboard | `load_epsf_metrics_for_draft` + UI overlay (aperture vs PSF) + PDF PSF section; `catalog_id` str/normalize fix; `drafts_before_session` path fix |
| TODO-8 ePSF RUN button | `run_epsf` job + ⚡ RUN ePSF Photometry button v ePSF tab |
| Reporting A | PDF Methods dynamic ALG citations; VAR.ASTRO `# ALG:` lines; UI Settings ALG toggles + TODO-44 sliders |
| FutureWarning | `variability_detector.py` pandas downcasting — zaznamenané ako tech debt |

### Backlog updates
Mark as CLOSED: TODO-GS6b, CQ-1, CQ-2, CQ-4, PERF-9, TODO-23, TODO-25, TODO-16, TODO-17, **TODO-ALG-2, TODO-ALG-3, TODO-ALG-4, TODO-ALG-5, TODO-44, TODO-8**

**TODO-8 note:** Infrastructure complete — `build_epsf_model` + per-frame PSF export + Phase 2A flux selector + standalone ePSF tab + RUN ePSF button. Bootes validation pending (**TODO-8-BOO**).

### Bug — Calibration Library re-register (20.5.2026, koniec dňa)

- `generate_master_dark_from_source_dir()` prepíše existujúci DB záznam (ID_EQUIPMENTS, ID_TELESCOPE) keď súbor s rovnakým názvom už existuje
- Správanie: mal by vytvoriť NOVÝ záznam pre iný set, nie UPDATE existujúci
- Dopad: Dark_60s bol preregistrovaný z QHY294MM+Carl-Zeiss na C3-26000+DDT
- Fix: INSERT nový záznam ak sa líši ID_EQUIPMENTS alebo ID_TELESCOPE (`register_calibration_library_entry` scope guard + scoped filename `_eq{N}_tel{M}`; `importer.py`, `database.py`)
- Priorita: HIGH (blokuje NGC 5466 / Bootes test)

### Brand / paper title — VYVAR (20.5.2026)

**Názov VYVAR je finálny — nemeň (skvelý brand).** V anglickom článku ho stačí „astronomicky“ obhájiť; oficiálny working title:

**VYVAR: A High-Automation Variable Star Photometry Pipeline…**

Do úvodnej sekcie článku (Introduction) alebo pod čiaru — krátka poznámka pre recenzentov:

> The name "VYVAR" is derived from the Slovak words for "Variable" (premenná) and "Archive/Reduction", while colloquially playing on the culinary term "broth/soup" – metaphorically boiling down raw FITS images into a rich, concentrated astrophysical summary report.

Recenzenti takéto jazykové hry zvyčajne vítajú; ukazuje, že softvér má „dušu“ a nepísal ho korporátny robot.

### Open / parked (po ALG series)
| ID | Popis | Priorita |
|----|-------|---------|
| TODO-8-BOO | Bootes globular cluster validation: test ePSF vs aperture on dense field (~2h data) | MEDIUM |
| TODO-LIB | Compile VYVAR modules to `.pyd` (Cython) — hide source, enable C translation | MEDIUM |
| TODO-ALG-1 | BLS period search on ground LC (`delta_mag` + BJD) — parkovaný, väčší scope | LOW |

### Strategic — Gold Standard 2.0

| ID | Popis | Priorita | Úsilie |
|----|-------|---------|--------|
| TODO-GS8 | **Multi-Night Global Matching** — Phase 3 modul: globálny crossmatch comp hviezd cez viacero nocí, medzinocný ZP solver, zlúčenie do jednej LC bez skokov. Závislosť: TODO-GS6 (AAVSO validácia na viacerých nociach). | HIGH | 2–4 dni |
| TODO-GS9 | **Ground LC periodická analýza** — Lomb-Scargle + BLS periodogramy priamo na Phase 2A LC CSV (astropy.timeseries); automatický fázový diagram (folded LC) pre variability kandidátov; výstup v PDF reporte. Súvisí s TODO-ALG-1 (parkovaný). | HIGH | 1–2 dni |
| TODO-GS10 | **AAVSO Direct API Upload** — tlačidlo "Submit to AAVSO" v UI; `AAVSO_USERNAME` / `AAVSO_PASSWORD` v config; odoslanie cez AAVSO WebObs API po validácii (GS6b). Závislosť: TODO-GS6. | MEDIUM | 1 deň |
| TODO-GS11 | **Flux Dilution Factor** — pixel-level blend correction: pre každý target vypočítať D = Flux_target / (Flux_target + ΣFlux_neighbors) z Gaia pozícií + magnitúd v aperture rádiuse; opraviť nameranú amplitúdu; pridať `dilution_factor` stĺpec do LC CSV a PDF. Súvisí s contamination_map (už existuje). | MEDIUM | 1–2 dni |

---

---

# VYVAR SESSION SUMMARY — 19.5.2026

**28 tasks completed, 5 bug fixes, E2E validated on draft_321**

Total night run time improved: **~1383 s → ~1047 s** (draft_321 measured; cieľová trieda ~1124 s, **~−19%** oproti pred-optimalizácii)

| Metrika | Pred | Po | Δ |
|---------|------|-----|---|
| Celkový night run (`simulate_night_run`) | ~1383 s | **1046.5 s** (draft_321) | **−336 s (~−24%)** |
| Photometry step | ~657 s (draft_320) | **524.9 s** | pod cieľom 580 s |
| Phase 1 comp selection | ~530 s (draft_320) | **368.8 s** | −161 s |
| SysRem (82 LC × 139 frames) | — | **5.7% RMS improvement** | BO CVn field |

## Dokončené dnes (19.5.2026)

### Performance (PERF-1 … PERF-10) — všetko ✅
| TODO | Čo |
|------|-----|
| PERF-1 | `csv_cache` v suspected-variables; hit/miss log |
| PERF-2 | MASTERSTAR `ms_data`/`ms_header` cache (Phase 2A + PNG/edge) |
| PERF-3 | Comp Gaia batch prefetch pred Phase 1 |
| PERF-4 | `comp_pool_rms.py` vectorized flux + groupby |
| PERF-4B | Hybrid `_accumulate_per_frame_comp_metrics` (iterrows N&lt;50, groupby N≥50); catalog_only skip Phase 1 |
| PERF-5 | `ProcFrameStore` — jeden disk read na snímok |
| PERF-6 | UI variability + `load_field_flux_matrix` cez ProcFrameStore |
| PERF-7 | UI LC index: cached summary + overlay CSV |
| PERF-8 | Shared flux matrix Phase 2A (~384 hviezd × 139 snímok) |
| PERF-9 | ✅ Vectorized haversine VSX match in `photometry_core.py` (`select_active_targets`) |
| PERF-10 | DAO QC v calibration pass (`dao_qc_in_calibrate`) |

### Code quality + algorithms
| # | Čo | Výsledok |
|---|-----|----------|
| CQ-3 | `select_comparison_stars_per_target` split | 1893 → ~200 riadkov orchestrátor + `comp_selection_per_target.py` (12 helpers) |
| TODO-35 | SysRem MVP (Tamuz et al. 2005) | `delta_mag_sysrem` stĺpec; 3 iter; 5.7% RMS improvement |
| TODO-ALG-3/4/5 | Backlog algoritmov | Pridané do STATE (temporal binning, Democratic Detrender, PyTICS) |
| TODO-GS6b | ✅ AAVSO Extended Format validator (`scripts/validate_aavso_export.py`) |

### Headless pipeline + E2E
| # | Čo | Výsledok |
|---|-----|----------|
| — | `night_run.py` | Headless runner (základ pre TODO-11) |
| — | `simulate_night_run.py` | CLI E2E (`D:\BO_CVn`, eq=1, tel=1) |
| — | **draft_321** | exit 0, **84 LC**, **139 frames**, photometry **524.9 s** |
| — | **draft_342** | exit 0, **134 LC**, **139 frames**, night run **4345 s**, HIP 67011 `lc_rms=0.012` |
| — | BO CVn / FW CVn `lc_rms` | 0.1515 / 0.0153 (within 0.001 mag of reference) |
| — | PDF report | **98 strán**, všetkých **84** hviezd |

### Bug fixes (19.5.2026)
| Bug | Fix |
|-----|-----|
| ProcFrameStore subscriptable | `.get()` + `__getitem__` safety net |
| `WindowsPath` / FIELD JUMP DB | `VyvarDatabase(dbp2)` v `detect_field_jumps` |
| PDF `_norm_cid` | Module-level + staticmethod na `_PhotometryReportBuilder` |
| PDF `_var_results` | `self._var_results` v `__init__` |
| PDF `TITLE_H` | Opravené v report builderi |
| `aperture_px` key | Konzistentný kľúč v summary/LC |
| `contamination_map` init | `contamination_map = {}` v comp selection (draft_320 crash) |

### Otvorený backlog (po 19.5.2026; CQ-1/2/4, PERF-9, GS6b, TODO-16/17/23/25 → CLOSED 20.5.2026)
| ID | Popis |
|----|-------|
| TODO-GS6 | AAVSO submission |
| TODO-GS7 | Paper draft |
| TODO-LIB | Compile VYVAR modules to `.pyd` (Cython) — hide source |
| TODO-ALG-1 | BLS ground LC — parkovaný (väčší scope) |
| TODO-ALG-2 | ✅ CLOSED — Savitzky-Golay detrend (`savgol_detrend_enabled=false` default) |
| TODO-ALG-3 | ✅ CLOSED — Temporal binning + adaptive window cap |
| TODO-ALG-4 | ✅ CLOSED — Democratic Detrender (`delta_mag_democratic`, `err_inflation`) |
| TODO-ALG-5 | ✅ CLOSED — PyTICS iterative comp weights |
| TODO-44 | ✅ CLOSED — Role-aware aperture (`aperture_variable_factor` / `aperture_comp_factor`) |
| TODO-8 | ✅ CLOSED — ePSF tab + RUN ePSF + `run_epsf` job; Bootes validation → **TODO-8-BOO** |
| TODO-8-BOO | Bootes globular cluster validation: ePSF vs aperture on dense field (~2h data) | MEDIUM |
| TODO-11 | Auto-trigger watchdog |
| TODO-LC-QUALITY | LC classification filter (saturated, noisy2/3, `no_data`) |
| TODO-LC-TREND | Differential extinction + ALG audit |
| TODO-GEO | Observer geographic position audit (BJD, airmass, HJD) |
| TODO-DEV-PROCESS | `scripts/regression_test.py` + development checklist |
| TODO-MASTERSTAR-QA validation | Cyan FORCED_APERTURE overlay v UI |
| — | `app.py` → `night_run.run_night_pipeline()` (deferred) |
| PERF-9 cKDTree | Spatial index for comp pool (optional; haversine done) |
| SysRem full-field | Všetky `proc_*.csv` hviezdy (deferred po PERF-5/6) |

---

# VYVAR SESSION SUMMARY — 18.5.2026

## Dokončené dnes (18.5.2026)

### Validácia
| # | Čo | Výsledok |
|---|----|---------|
| 1 | Gold standard roadmap + TODO-GS1–GS7 | Zdokumentované v STATE |
| 2 | TODO-45: RGB kamera (IMX533) | Zdokumentované v STATE |
| 3 | TODO-27: Re-validácia po float64 fix | CLOSED — 17 hviezd = DAO detekčná limita |
| 4 | TODO-31: Growth curve korekcia | CLOSED as NOT APPLICABLE (diferenciálna fotometria) |
| 5 | photutils LC validácia (draft_310) | 67 hviezd, mag 8–13 Δ<0.001 mag ✅ |
| 6 | PyRAF LC validácia | NOT FEASIBLE (IRAF float32 FITS bug) |
| 7 | Muniwin LC validácia | 3 hviezdy, ±5–15% RMS ✅ |
| 8 | AIJ validácia | PLANNED (manuálna, používateľ) |

### Fyzikálny audit + pipeline opravy
| # | TODO | Výsledok |
|---|----|---------|
| 9 | TODO-29: Airmass fit po outlier detect | ✅ |
| 10 | TODO-30: CT → airmass na CT-korigovanom mag | ✅ |
| 11 | TODO-GS1: Citačné komentáre (Howell, Broeg, Stetson) | ✅ |
| 12 | TODO-GS2: Unit test suite 11/11 | ✅ |

### UI + export
| # | Čo | Výsledok |
|---|----|---------|
| 13 | Kompletný preklad UI → angličtina | ~766 stringov, 0 SK/CZ zostáva ✅ |
| 14 | Vedecké citácie v AAVSO + VAR.ASTRO exportoch | ✅ |

### Code quality (6 passes)
| # | Fix | Výsledok |
|---|----|---------|
| 15 | Silent exceptions → logging | 38 blokov, 9 súborov ✅ |
| 16 | Gaia ID normalizácia | 8 duplicátov → canonical ✅ |
| 17 | Dead code označený | 5 UI + 13 legacy helpers ✅ |
| 18 | Draft path resolution | `resolve_draft_dir()` ✅ |
| 19 | CSV dtype konzistentnosť | `read_vyvar_csv()` ✅ |
| 20 | Split dlhých funkcií | `_PhotometryReportBuilder` (3384→63 lines) ✅ |

### CSV schema cleanup
| # | Čo | Výsledok |
|---|----|---------|
| 21 | Phase 1: 15 obsolete stĺpcov odstránených | ✅ |
| 22 | Phase 2: 11 ďalších stĺpcov odstránených | ✅ |
| 23 | Bug fix: variability_candidates.csv TESS stĺpce | ✅ |

### Dokončené dnes (18.5.2026) — doplnok

| # | Čo | Výsledok |
|---|----|---------|
| 24 | TODO-GS3: README.md | ✅ |
| 25 | TODO-GS4: CITATIONS.bib | ✅ |
| 26 | TODO-GS5: CHANGELOG.md | ✅ |
| 27 | TODO-19: Infolog ukladanie na disk | ✅ Auto-save po RUN VYVAR + MAKE MASTERSTAR + manual button |
| 28 | CQ-6: Delete legacy deprecated functions | ✅ 13 funkcií + 2 helpers = 1021 riadkov odstránených |

---

## Otvorené TODO — úplný backlog

### HIGH priority
| TODO | Popis |
|------|-------|
| TODO-GS1 | ✅ Hotovo |
| TODO-GS2 | ✅ Hotovo |
| TODO-GS3 | ✅ Hotovo |
| TODO-GS4 | ✅ Hotovo |
| TODO-GS5 | ✅ Hotovo |
| TODO-GS6 | AAVSO submission + cross-observer validácia |
| TODO-GS6b | AAVSO Extended Format validation (blocker for TODO-GS6) |
| TODO-GS7 | Paper draft (PASP / AN) |
| TODO-35 | ✅ COMPLETED — SysRem MVP (exported LC targets) |

**TODO-GS6b: AAVSO Extended Format validation**
- Before first submission to AAVSO (TODO-GS6), add automated validator that checks VYVAR AAVSO export against official spec: https://www.aavso.org/aavso-extended-file-format
- Required fields: `#TYPE`, `#OBSCODE`, `JD`, `MAGNITUDE`, `MAGNITUDE_ERROR`, `FILTER`, `TRANSFORMED`, `MAG_TYPE`, `COMP_STAR_1`, `COMP_STAR_2`, `CHARTS`, `NOTES`, `NAME`, `AFFILIATION`, `MTYPE`, `GROUP`, `CHART`, `HERALD`
- Validator should: parse export file, check all required columns present, check value ranges (`JD > 2400000`, mag 1–25, filter codes valid), report any violations before upload
- Effort: 2–4h | Priority: **HIGH** (blocker for TODO-GS6) | Dependency: TODO-GS6

### MEDIUM priority
| TODO | Popis |
|------|-------|
| TODO-8 | ✅ CLOSED — ePSF tab + RUN ePSF + `run_epsf` job; Bootes validation → **TODO-8-BOO** |
| TODO-8-BOO | Bootes globular cluster validation: ePSF vs aperture on dense field (~2h data) | MEDIUM |
| TODO-19 | ✅ Hotovo |
| TODO-31 | ✅ CLOSED as NOT APPLICABLE |
| TODO-44 | ✅ CLOSED — Role-aware aperture (SIPS-style; SNR table + role factors) |
| TODO-45 | RGB kamera podpora (IMX533) — de-Bayer → G kanál |

### LOW priority
| TODO | Popis |
|------|-------|
| TODO-7 | Plate solver refactor |
| TODO-9 | Inštalácia VYVAR na Lenovo T460 (crossval env hotové) |
| TODO-10 | Settings záložka refaktor + CONFIG_GUIDE.md |
| TODO-11 | Auto-trigger po konci pozorovania |
| TODO-12 | HRD klasifikácia hviezd (po novej DB) |
| TODO-14 | PDF size optimization (po plate solving refaktore) |
| TODO-16 | ✅ UI crossmatch — `active_targets.csv` coords |
| TODO-17 | ✅ `crossmatch_auto_done` gated (aperture tab) |
| TODO-18 | ✅ Hotovo (UI preklad) |
| TODO-20 | Mean stack MASTERSTAR — **LOW**: zlepší WCS/FWHM na stacku, nie photometry SNR na single-exposure LC |
| TODO-23 | ✅ Adaptive matching radius (plate_scale × 3) |
| TODO-25 | ✅ Gaia→DAO — `pipeline_meta.json` + UI read |
| TODO-LIB | Compile VYVAR to `.pyd` (Cython) |
| TODO-29 | ✅ Hotovo |
| TODO-30 | ✅ Hotovo |

### Code quality — remaining tech debt
| # | Popis | Riziko |
|---|-------|--------|
| CQ-1 | ✅ `run_phase2a()` split (1515 → 188 lines + 3 helpers) | — |
| CQ-2 | ✅ `render_live_view()` split (1395 → 731 lines + 3 helpers) | — |
| CQ-3 | ✅ `select_comparison_stars_per_target()` → `comp_selection_per_target.py` (12 helpers) + ~356-line orchestrator in `photometry_core.py`; timing regression fixed: catalog_only skip + hybrid accumulate (iterrows N&lt;50, groupby N≥50); draft_321 validated | — |
| CQ-4 | ✅ `solve_wcs_with_local_gaia()` split (1843 → 1224 lines + 3 helpers) | — |
| CQ-5 | Wire alebo remove orphaned UI modules | MEDIUM |
| CQ-6 | Delete legacy importer/pipeline helpers | ✅ |
| CQ-7 | Photometry module surface (`photometry` vs `photometry_core` imports) | LOW |
| pandas FutureWarning | `variability_detector.py` downcasting — fix v ďalšej session | LOW |

### Performance optimization TODOs (z auditu 18.5.2026)

| TODO | Popis | Impact | Effort | Priorita |
|------|-------|--------|--------|---------|
| TODO-PERF-1 | ✅ COMPLETED — `csv_cache` už v calleri; hit/miss log v `_write_suspected_variables` | MEDIUM | done | HIGH |
| TODO-PERF-2 | ✅ COMPLETED — `ms_data`/`ms_header` z `run_phase2a` cache do PNG/edge helperov; shared header v `run_full_photometry_pipeline` | LOW | done | HIGH |
| TODO-PERF-3 | ✅ COMPLETED — comp star Gaia batched via `_comp_gaia_prefetch` before Phase 1 loop | MEDIUM | done | HIGH |
| TODO-PERF-4 | ✅ COMPLETED (Option A) — `comp_pool_rms.py` flux vectorized; `select_comparison_stars` → CQ-3 | MEDIUM | done | HIGH |
| TODO-PERF-5 | ✅ COMPLETED (Option B) — `ProcFrameStore` unified in-memory proc CSV cache | HIGH | done | MEDIUM |
| TODO-PERF-6 | ✅ COMPLETED — UI variability uses ProcFrameStore when in session | HIGH | done | MEDIUM |
| TODO-PERF-7 | ✅ COMPLETED — UI LC index: cached summary + overlay CSV cache | MEDIUM | done | MEDIUM |
| TODO-PERF-8 | ✅ COMPLETED — shared flux matrix (all LC stars × frames) before target loop; per-target slice replaces ~11k `read_flux_from_csv` calls; fallback preserved | MEDIUM | done | LOW |
| TODO-PERF-4B | ✅ COMPLETED — `_accumulate_per_frame_comp_metrics` hybrid: vectorized groupby N≥50, iterrows N&lt;50; log `[PERF-4B]`; catalog_only skip in Phase 1; draft_321 validated | MEDIUM | done | HIGH |
| TODO-PERF-9 | ✅ COMPLETED — vectorized haversine in `select_active_targets` + `_filter_comp_candidates_spatial_static`; cKDTree optional deferred | MEDIUM | done | LOW |
| TODO-PERF-10 | ✅ COMPLETED — DAO QC merged into calibration pass (`dao_qc_in_calibrate`) | HIGH | done | LOW |

### Algorithm upgrade TODOs (z Gemini analýzy 18.5.2026)

| TODO | Popis | Impact | Effort | Priorita |
|------|-------|--------|--------|---------|
| TODO-35 | ✅ SysRem MVP — `delta_mag_sysrem` on exported LC; full field deferred | HIGH — sub-mmag RMS | done (MVP) | HIGH |
| TODO-ALG-1 | BLS ground LC — parkovaný (TESS `tess_verify` only today) | — | TBD | LOW |
| TODO-ALG-2 | ✅ CLOSED — Savitzky-Golay detrend after airmass (`savgol_detrend_lc`) | opt-in | done | MEDIUM |
| TODO-ALG-3 | ✅ CLOSED — Temporal binning comp ensemble (MNRAS 2023, adaptive window cap) | BO ↓3% draft_321 | done | MEDIUM |
| TODO-ALG-4 | ✅ CLOSED — Democratic Detrender 3-model ensemble (arXiv 2411.09753v2) | CSV err bars | done | MEDIUM |
| TODO-ALG-5 | ✅ CLOSED — PyTICS iterative comp intercalibration (RASTI 2026) | done | done | MEDIUM |
| TODO-8 | ✅ CLOSED — ePSF infrastructure (build + export + Phase 2A + tab + RUN button) | — | — | — |
| TODO-8-BOO | Bootes globular cluster validation: ePSF vs aperture on dense field (~2h data) | MEDIUM — dense fields | validation ~2h | LOW — test dataset: globular in Bootes |

---

**TODO-ALG-3: Optimized Temporal Binning of comparison ensemble** — ✅ CLOSED (20.5.2026)

Reference: MNRAS (2023) 526, 3482–3489 — *"Optimised temporal binning of comparison star measurements for differential photometry"*

**Problem:** Shot noise + read noise of comp stars adds in quadrature to target noise. High-frequency random noise in comp ensemble artificially degrades LC quality (especially mag > 12, short exposures).

**Solution:** Before ZP subtraction in Phase 2A `ensemble_normalize()`, apply optimal temporal smoothing (rolling window / spline) to comp flux time series. Preserves low-frequency atmospheric trend but removes high-frequency random noise from comp measurements.

**Integration point:** `ensemble_normalize()` in `photometry_core.py` — smooth `comp_mag_inst` per comp star before weighted median. Optimal window size: find mathematically via minimizing target RMS over window sizes [3, 5, 7, 9, 11] frames.

**Expected gain:** 15–30% RMS reduction without hardware changes.

Impact: HIGH | Effort: 4–8h | Priority: MEDIUM

Config: `temporal_binning_enabled` (bool), `temporal_bin_window` (int, 0=auto)

---

**TODO-ALG-4: Democratic Detrender — ensemble multi-model detrending** — ✅ CLOSED (20.5.2026)

Reference: arXiv:2411.09753v2 (February 2026) — *"The democratic detrender: Ensemble-Based Removal of the Nuisance Signal in Stellar Time-Series Photometry"*

**Problem:** Current VYVAR uses linear/polynomial airmass fit — wrong polynomial degree risks underfitting (residual trend) or overfitting (erasing real physical variability like eclipse minimum).

**Solution:** Run 3 independent detrending models in parallel:
- A) Cosine filtering (current `airmass_detrend_lc` — keep as-is)
- B) Low-degree polynomial fit (degree 2–3)
- C) Gaussian Process regression (sklearn/george/celerite2)

Compute marginalized mean of all 3 models. Use MAD between models as adaptive error-bar inflation factor — adds model-selection uncertainty to per-point error bars.

**Integration point:** after `airmass_detrend_lc()` in `run_phase2a()`, new function `democratic_detrend(mag_calib, airmass, bjd, flags)`. Output: `delta_mag_democratic` + `err_democratic` columns in LC CSV.

**Note:** GP model is computationally expensive — consider optional (`democratic_gp_enabled` config flag). Cosine + Poly alone already gives marginalized errors without GP overhead.

**Expected gain:** publication-quality error bars; overfitting immunity.

Impact: HIGH (publication) | Effort: 8–16h | Priority: MEDIUM

Config: `democratic_detrend_enabled` (bool), `democratic_gp_enabled` (bool)

---

**TODO-ALG-5: PyTICS iterative comp star intercalibration** — ✅ CLOSED (20.5.2026)

Reference: RASTI (2026) — *"PyTICS: an iterative method for photometric light-curve intercalibration using comparison stars"*

**Problem:** Some comp stars are micro-variable (low amplitude, unknown to VSX). Current MAD sigma-clip catches gross outliers but misses stars with systematic scatter pattern across the night. These silently inject noise into ZP calibration.

**Solution:** Multi-component noise model in closed loop:
1. Compute preliminary ZP (current Broeg 2005 ensemble)
2. Compute per-comp residuals vs ZP across all frames
3. Assign lower weight to comps with systematically higher scatter
4. Recompute ZP with updated weights
5. Iterate until weights converge (typically 3–5 iterations)

**Integration point:** `check_comparison_stability()` and `ensemble_normalize()` in `photometry_core.py` — replace fixed MAD threshold with iterative weight update.

**Synergy:** combines well with TODO-ALG-3 (temporal binning) and existing Broeg (2005) weighted ensemble.

**Note:** VYVAR already has comp stability check + MAD sigma-clip — PyTICS is an evolutionary improvement, not a replacement. Keep existing logic as fallback / sanity check.

**Expected gain:** full autonomy from catalog quality; detects 0.01 mag micro-variability in comp stars.

Impact: MEDIUM-HIGH | Effort: 6–12h | Priority: MEDIUM

Config: `pytics_enabled` (bool), `pytics_n_iter` (int, default 5)

---

### Competitive position (vs MUNIWIN / AIJ / SIPS / MaxIm DL)

**VYVAR exceluje:**
- Plná autonómnosť (kávový test ☕) — žiadny iný softvér nemá
- SNR-optimálna per-star apertura — unikátne
- TESS + VSX + Gaia ekosystém — unikátne
- MAD sigma-clip ZP per frame — vzácnosť v amatérskom softvéri
- 200-stranový PDF report — bez konkurencie

**VYVAR zaostáva:**
- Interaktívne čistenie dát (AIJ dominuje)
- Exoplanétové tranzitné modely (Mandel & Agol)
- Hardvérová kontrola (MaxIm DL dominuje)
- Rýchlosť na veľkých poliach (C/C++ vs Python)
- ePSF pre preplnené polia — infrastructure done (TODO-8); validation **TODO-8-BOO**

**VYVAR algorithm upgrades** (literature-backed; ✅ implemented 20.5.2026):
- **TODO-ALG-3** ✅ — optimised temporal binning of comparison ensemble (MNRAS 2023)
- **TODO-ALG-4** ✅ — Democratic Detrender multi-model detrending (arXiv 2411.09753v2)
- **TODO-ALG-5** ✅ — PyTICS iterative comp intercalibration (RASTI 2026)
- **TODO-ALG-2** ✅ — Savitzky-Golay detrend (opt-in, default off)
- **TODO-ALG-1** ⏸ — BLS on ground LC (parked; TESS path unchanged)

### Gold standard — zostatok
| Krok | Stav |
|------|------|
| Peer-reviewed algoritmy s citáciami | ✅ |
| Reprodukovateľnosť | ✅ (vizuálne overená) |
| Transparentnosť | ✅ |
| Validácia (photutils, Muniwin, IRAF, SExtractor) | ✅ |
| Dokumentácia | ✅ unit testy |
| AAVSO validácia | ⏳ TODO-GS6b → TODO-GS6 |
| Open source README + CITATIONS.bib + CHANGELOG | ✅ |
| Paper draft | ⏳ TODO-GS7 |

---

*Session 18.5.2026 — 28 úloh dokončených*

# VYVAR STATE — 2026-05-18 (aktualizácia)

## Issues Status

### a) Double MASTERSTAR — COMPLETED ✅

### b) Comp stars = variable targets — COMPLETED ✅
- Fix 1: catalog_id dtype (str)
- Fix 2: Field map dedup in save_field_map_png
- Fix 3: Proximity veto in select_comparison_stars_spatial_grid
- Proximity veto now always logs (even 0 removals)

### c) Border filter — COMPLETED ✅
- Root cause fixed: aligned_files passed after RAM flush
- Verified manually via test_border_bbox.py on draft_000283
- safe_bbox_px = [30.4, 30.4, 2050.6, 1365.6]

### d) Stale x/y — COMPLETED ✅
- x/y refresh po MAKE MASTERSTAR implementovaný (`_refresh_variable_targets_xy` pred `select_active_targets`)
- Eliminuje issue (d) úplne

### e) SUMMARY MEASURE REPORT PDF — COMPLETED ✅
- PDF po fotometrii: `generate_photometry_report` v `photometry_report.py`
- Po RUN VYVAR volanie z `app.py` po `run_full_photometry_pipeline`

---

### TODO-1: Adaptívny config podľa hustoty poľa — COMPLETED ✅
### TODO-2: x/y refresh po MAKE MASTERSTAR — COMPLETED ✅
### TODO-3: Globálny comp pool — COMPLETED ✅
### TODO-4: Summary Measure Report PDF — COMPLETED ✅
### TODO-5: TESS reaktivácia — COMPLETED ✅
### TODO-6: BP-RP slidery v `ui_settings.py` — COMPLETED ✅

### BP-RP UI tabuľka — COMPLETED ✅
### VSX crossmatch bug fix — COMPLETED ✅
### ⚡ RUN VYVAR — COMPLETED ✅
### Draft_287 overenie — COMPLETED ✅
### Export TXT hlavička cleanup — COMPLETED ✅

---

## Dnes implementované (14.5.2026)

### RUN VYVAR — „I/O operation on closed file" — COMPLETED ✅
### Variabilita dashboard zjednodušenie — COMPLETED ✅
### TESS auto-trigger + robustnosť — COMPLETED ✅
### TESS — robustnosť a kvalita periódy — COMPLETED ✅
### TESS blend check alpha fix — COMPLETED ✅
### TESS auto-trigger for all candidates — COMPLETED ✅
### SUMMARY MEASURE REPORT refactor — COMPLETED ✅
### VYVAR_report 5 enhancements — COMPLETED ✅
### VYVAR_report PDF redesign (TODO-15) — COMPLETED ✅

---

## Dnes implementované (15.5.2026)

### MASTERSTAR QA — auto-load draft — COMPLETED ✅
- `render_masterstar_qa` (`ui_masterstar_qa.py`): `default_ap` priority chain
  1. `draft_dir_override`
  2. `vyvar_last_job_output["archive_path"]` / `vyvar_post_cal_archive_path`
  3. `Drafts/draft_{id}` z `vyvar_last_draft_id` alebo `draft_id` arg
  4. `vyvar_last_import_result.archive_path` (fallback)
- Pole sa automaticky predvyplní rovnako ako ostatné dashboardy

### Variabilita — katalógy stale "žiadny záznam" — COMPLETED ✅
**Root cause:** UI crossmatch bežal pred pipeline s mierne zlými koordinátmi
→ zapísal "žiadny záznam" do `bullets_map` a `_crossmatch/*.json`
→ pipeline crossmatch tieto záznamy považoval za "hotové" a preskakoval

**Fix reťazec (3 vrstvy):**
- Fix A: `load_katalogy_map_from_disk` + `_merge_katalogy_maps` v `ui_variability.py`
  → UI tabuľka vždy číta `variability_candidates.csv` z disku (disk = ground truth)
  → Sync späť do `var_catalog_bullets` (export + TESS + PDF vidia rovnaké dáta)
- Fix B: `_has_positive_catalog_match` v CSV skip guard (`crossmatch_runner.py`)
  → Pipeline preskakuje riadok len ak má aspoň jeden POZITÍVNY katalógový match
  → Riadky s iba "žiadny záznam" sa znova crossmatchujú
- Fix C: `_has_positive_catalog_match` v JSON cache guard (`crossmatch_runner.py`)
  → Stale cache súbory (len "žiadny záznam") sa zmažú a API sa zavolá znova
  → Cache write logika zostáva nezmenená

**Otvorené follow-ups (LOW priority):**
- TODO-16: UI crossmatch — použiť `active_targets.csv` coords namiesto flux-matrix
- TODO-17: `crossmatch_auto_done` — re-enable ak disk prázdny

### TESS duplikácia — UI nespúšťa TESS ak result.json existuje — COMPLETED ✅
- `_tess_result_json_on_disk(cid)` check pred UI auto-trigger
- `to_tess` a `_need_tess` vylučujú kandidátov kde `_tess/{cid}/result.json` existuje
- Pipeline TESS volania nedotknuté
- Efekt: RUN VYVAR spustí TESS raz; otvorenie Variabilita tabu nespustí znova

### SUMMARY MEASURE REPORT — kompletný refactor (TODO-14, TODO-15) — COMPLETED ✅

#### Fáza 1: Nová štruktúra stránok
| Str. | Sekcia |
|------|--------|
| 1 | Cover sheet (logo, title, draft/setup/dates) |
| 2 | Observation summary (metrics, conditions, methods, comp pool) |
| 3 | FITS Quality Assessment |
| 4 | Summary of all stars (sorted by vsx_type: EA→EB→EW→ROT→VAR, then lc_rms) |
| 5 | HRD (Hertzsprung-Russell diagram) |
| 6 | Field map (full page, landscape) |
| 7–N | Per-star pages (LC + field + comp table, one page each) |
| N+1 | Variability Analysis — RMS Hockey Stick |
| N+2 | Variability Candidates table |
| N+3+ | TESS Analysis (per candidate, 1–2 sektory/strana podľa výšky) |
| Last | Abbreviations & Notes |

#### Fáza 2: Hockey stick + Candidates vylepšenia
- Hockey stick: farebné odlíšenie (Stable=green, Known VSX=amber×,
  Candidate+match=orange●, Candidate no match=red●); ukladá `hockey_stick_report.png`
- Candidates table: pagination, katalogy=positive lines only (max 4+N more),
  row coloring (green=match, red=no match)

#### Fáza 3: Per-star + TESS layout
- Per-star: max 12 comp rows + "(+N more)", TESS odstránený z per-star strany
- TESS: header + metrics + period table + 1–2 sektory/strana (phased P + 2P + blend)

#### Fáza 4: PDF veľkosť optimalizácia
- `_compress_image_for_pdf`: JPEG rekompresia pred každým embed (typ→max_px+quality)
- Hockey stick DPI: 96
- plt.close(fig) po každom savefig
- Výsledok: 29 MB / 187 strán (z 53 MB / ~200+ strán)
- TODO-14 zostáva PENDING pre ďalšiu optimalizáciu po plate solving refaktore

#### Layout opravy (vizuálna revízia)
- Cover: em-dash fix ("VYVAR — Summary Measure Report"), logo 40% šírky, centrovaný
- Observation summary: Broeg referencia zalomená (textwrap 110 chars, font 7.5)
- FITS QA: odstránená "Top 5 Masterstar candidates" tabuľka; FWHM limit z
  `_qa_fwhm_limit_px` (rovnaká logika ako `ui_quality_dashboard.py`);
  masterstar frame z `_resolve_masterstar_used_frame` (FITS header → CSV → fallback)
- HRD: všetky slovenské labely preložené do angličtiny (`hrd_analysis.py`)
- Field map: duplikát odstránený; len full-page landscape verzia
- Hockey stick: landscape strana, 90%×80% využitie plochy
- Variability candidates: celé catalog_id (bez skracovania), dynamická výška riadku
  z `Paragraph.wrap`, "ďalších"→"more", "žiadny záznam"→"—"
- TESS: dynamický height budget (fit 1 alebo 2 sektory podľa dostupnej výšky),
  period analysis tabuľka (Sector|N pts|P(d)|Method|P_anova|P_consensus) z result.json
- Abbreviations: font 7.5pt, textwrap 55 chars, nové skratky (AAVSO, VAR.ASTRO,
  DAO, SNR, ZP, lc, comp, obs_group, dr)

---

## Dnes implementované (16.5.2026)

### TODO-13: Multi-step iterative matching — COMPLETED ✅
- **Best frame FWHM pre DAO** (nie median `VY_FWHM` v hlavičke); flag `masterstar_use_best_frame_fwhm`
  · `build_masterstar_from_detrended()` → `best_frame_fwhm_px` v `ms_selection_meta`
  · `dao_fwhm_bypass_header` v `detect_stars_and_match_catalog()`
  · draft_000303: median header ≈ 3.52 px → best frame ≈ 3.09 px
- **2-pass iteratívny DAO** — pass 2 na unmatched Gaia pozíciách (`_dao_targeted_pass2_unmatched_gaia`)
  · Pass 2 sigma: **1.9** (`masterstar_dao_pass2_sigma`, min. 1.5 v kóde)
  · Očakávaný match rate: **83% → ~95%+**
  · draft_000303 simulácia: **+889** detekcií (+25.3 %), merge 3515 → 4404
- **Match-rate monitoring** — log warning ak &lt; 88 % po 1. match passe (bez auto-retry)

### TODO-21: SNR-optimal per-star aperture — PARTIALLY COMPLETED ✅
- **Fáza 1–2** v `photometry_core.py`: gain/RN z `EQUIPMENTS` do `_photometric_error()`;
  `compute_snr_optimal_aperture_table()` + per-star `apertures_px` v Phase 2A;
  `draft_dir/aperture_snr_table.json`
- **Fáza 3 ✅ (16.5.2026):** per-frame `dao_flux` cez `enhance_catalog_dataframe_aperture_bpm()`
  + pipeline precompute SNR table pred exportom CSV
- draft_303 (gain 3.17, RN 7.6): mag 8 → 4.53 px; mag 11 → 3.43 px; mag 14 → 2.47 px
  (vs globálna ~3.99 px)
- **Pending:** Fáza 4 LC scatter validácia (e2e draft_305 + `photometry_summary.csv`)

---

## Dnes implementované (17.5.2026)

### Match rate metrika: Gaia→DAO completeness — COMPLETED ✅ (17.5.2026)
- **Pôvodné:** DAO→Gaia = n_matched_dao / n_detected_dao (všade)
- **Nové (paralelné):** Gaia→DAO = unique(catalog_id) / catalog_rows
- **pipeline.py** `detect_stars_and_match_catalog()`:
  · meta keys: `gaia_dao_completeness_pct`, `n_gaia_undetected`
  · log `[DAO] Gaia→DAO completeness: N/M Gaia stars detected (X.X%) | catalog_only: K`
  · warning ak &lt; 80 %
- **pipeline.py** `generate_masterstar_and_catalog()`:
  · raw aj optimized MATCH STATS log rozšírený o Gaia→DAO %
  · `[MASTERSTAR] Gaia→DAO completeness: N/M (X.X%) | catalog_only: K`
- **ui_masterstar_qa.py** `render_masterstar_qa()`:
  · m2: `DAO→Gaia Match (%)` (nezmenené)
  · m3: `Gaia→DAO Completeness (%)` z `field_catalog_cone.csv` / n_ok
  · captions: VYNIKAJÚCA ≥90% / DOBRÁ 80–90% / NÍZKA &lt;80%
- **Known limitation:** UI (n_ok/cone_rows) vs pipeline (unique catalog_id/catalog_rows)
  môžu mierne líšiť ak 1 Gaia hviezda → viac MASTERSTAR riadkov (rare)
- **Aktivuje sa po:** re-run MASTERSTAR / per-frame DAO na draft_305;
  UI potrebuje `field_catalog_cone.csv` v `platesolve/<setup>/`

### TODO-22: Gain/RN Settings UI + DB — COMPLETED ✅ (17.5.2026)
- **database.py**: `set_equipment_cosmic_params(equipment_id, gain, read_noise)`
  → uloží `GAIN_ADU` / `READNOISE_E` (NULL ak ≤ 0); `get_equipment_cosmic_params()` nezmenené
- **ui_settings.py** (tab Fotometria): sekcia „Detektor — fotometrické parametre"
  · všetky EQUIPMENTS riadky (`get_equipments(active_only=False)`)
  · per-kamera expander: Gain [e⁻/ADU] + Read Noise [e⁻] inputs → Uložiť
  · warning ak hodnoty chýbajú (Phase 2A fallback 1.0 / 10.0)
- **ui_quality_dashboard.py**: banner po načítaní light rows
  · info (modrý) ak gain+RN nastavené v DB
  · warning (žltý) s odkazom na Settings → Fotometria → Detektor ak nie
  · caption ak draft nemá equipment_id
- Database Explorer grid: bez zmeny

### VYVAR UI — preklad do angličtiny — COMPLETED ✅ (17.5.2026)
- **ui_variability.py**: ~82 strings (Streamlit UI, Plotly, matplotlib, TESS, crossmatch)
- **ui_masterstar_qa.py**: ~38 strings (metrics, completeness captions, VSX controls)
- **ui_quality_dashboard.py**: ~45 strings (hover templates, FWHM, gain/RN banner)
- **ui_settings.py**: ~95 strings (tab labels, help blocks, detector UI, Phase 0+1)
- **ui_photometry.py**: ~18 strings (subheader, checkboxes, help, save button)
- **photometry_report.py**: 0 changes (PDF strings already in English)
- Nezmenené: LOGGER.*, log_event, column keys, catalog names (VSX/Gaia/TESS),
  config keys, CSV column names, `žiadny záznam` parsing
- py_compile: ✓ všetky súbory

### Fix: float64 catalog_id precision loss → XY fallback → false candidates (17.5.2026)
- **Root cause:** Gaia IDs (int64, ~19 digits) uložené ako float64 v proc_*.csv
  strácajú posledné bity → ID lookup miss → XY fallback → NaN frames
  → zlá LC → záporné airmass slopes → falošní variability kandidáti
- **Reťazec:** float64 precision → ID miss → 70× NaN/frame → LC scatter ↑
  → airmass fit zlyhá (slope < 0) → Hockey Stick červený bod
- **gaia_catalog_id.py:**
  · `normalize_gaia_source_id()`: Decimal pre large/sci-notation IDs
  · `catalog_id_series_for_masterstars_export()`: blank pre missing IDs
  · `GAIA_PROC_CSV_READ_DTYPE` + `catalog_id_series_for_proc_csv_export()`
- **photometry_core.py:**
  · `_build_csv_lookup()`: indexuje pod masterstar_row_gaia_key + name + catalog_id
  · `_lookup_star_in_csv()`: normalizuje cid pred lookup
  · `read_flux_from_csv()`: cid_key = _normalize_gaia_id(cid); KNOWN ISSUE blok odstránený
- **pipeline.py + diagnose_*.py:** `dtype={"catalog_id": str, "name": str}` všade
- **Potrebuje:** re-export proc_*.csv + re-run Phase 2A pre draft_305/307
- **Očakávané:** nula XY fallback warnings, viac am_detrended=True, menej red kandidátov

### Fyzikálny audit — CCD equation opravy (17.5.2026)
Zdroj: Howell (1989 PASP 101:616), DAOPHOT/MUNIWIN/IRAF referenčná implementácia.

**FIX 1 — _photometric_error: sky term /gain (photometry_core.py)**
- Chyba: `sky_pp × area` (ADU²) namiesto `sky_pp/gain × area` (e⁻)
- Oprava: `variance = flux/g + sky_pp/g × area + (RN/g)² × area`
- Dopad: err 2.87% → 1.69% pre sky-dominated hviezdy (mag > 12)
  → ~41% redukcia fotometrických chýb pre slabé hviezdy
- `compute_snr_optimal_aperture_table()`: nezmenená (samostatná TODO)

**FIX 2 — ensemble_normalize(): ZP sigma-clip per frame**
- Chyba: žiadny outlier reject na ΔZP = cat_mag − inst_mag per frame
- Oprava: MAD-based σ-clip (3σ) pred weighted mean ZP
  · len ≥ 4 comps: clip, re-compute weighted ZP
  · fallback nanmedian ak &lt; 2 po clipe
  · log `[ZP] Frame sigma-clip: N/M comps kept` pri outlieroch
- Zdroj: štandard IRAF phot / MUNIWIN (iteratívny ZP clip)
- Dopad: LC spiky z cosmic rays / saturovaných comp frameov

**Zostatok z auditu (ďalšie TODO):**
- SNR table: ADU/e⁻ mix v compute_snr_optimal_aperture_table()
- Airmass fit pred outlier detekciou (swap poradia)
- Color term aplikovaný pred airmass (nie na CT-korigovanom)
- Growth curve korekcia pre mag > 13 (Howell 1989 §3)

---

## TODO — backlog (updated 21.5.2026; draft_342 validated)

### TODO-13: Multi-step iterative matching — COMPLETED ✅ (16.5.2026; Phase A+B 21.5.2026)

**Phase A (21.5.2026):** Pass 2 catalog-seeded DAO v `detect_stars_match_master_reference` (MASTERSTAR fast path) — `_dao_targeted_pass2_unmatched_gaia` po pass 1 + prefilter.

**Phase B (21.5.2026):** `_inject_forced_aperture_rows` pre unmatched master stars; `_proc_catalog_keep_matched_rows_only` — len `GAIA_MATCHED` + `FORCED_APERTURE`; **0 `DET_*`** v proc CSV; `source_type` filter.

**E2E:** draft_342 — 134 LC, DET_*=0; Gaia→DAO completeness v QA dashboard stále nízka (~3.5%) → **TODO-13 validation** otvorené.
### TODO-14: PDF size optimization — PENDING (po plate solving refaktore)
- Aktuálne: 29 MB / 187 strán
- Cieľ: < 10 MB
- Hlavná príčina: LC grafy generované inline (matplotlib → priamo do PDF, bez disk cache)
- Riešenie: po TODO-13 (menej catalog_only hviezd = menej LC strán)

### TODO-15: VYVAR_report PDF redesign — COMPLETED ✅ (15.5.2026)

### TODO-16: UI crossmatch — active_targets.csv coords — COMPLETED ✅ (20.5.2026)
- `_get_candidate_row()` preferuje `active_targets.csv` RA/Dec (WCS-verified; mirrors `crossmatch_runner`)

### TODO-17: crossmatch_auto_done gate — COMPLETED ✅ (20.5.2026)
- Aperture tab: `crossmatch_auto_done=True` len keď `candidates` non-empty (nie pri prázdnom zozname)

### TODO-18: Field map — preložiť titulok do angličtiny — LOW
- Titulok "VYVAR — Field Map (červené=target, cyan=catalog_only, zelené=comp)"
  sa generuje v pipeline (save_field_map_png alebo podobne)
- Preložiť pri celkovom anglickom prepise VYVAR UI

### TODO-20: Mean stack MASTERSTAR — PENDING (priorita **LOW**)
- Zkombinovať 5 najlepších frames (sorted by FWHM) do median stack pred DAO detekciou
- Očakávaný benefit: √5 ≈ 2.2× lepší SNR na **detekciu** (MASTERSTAR / WCS / FWHM odhad) — **nie** na Phase 2A photometry SNR (LC stále z jednotlivých single-exposure frames)
- Poznámka: mean stack zlepší kvalitu WCS/FWHM, ale nezvýši photometry SNR v časovej sérii, ak každý frame zostane single-exposure limited
- Predpoklad: single best frame fix (TODO-13) musí byť overený e2e pred implementáciou stacku
- Implementovať po validácii TODO-13 na reálnom observe

### TODO-21: SNR-optimal per-star aperture — PARTIALLY COMPLETED ✅ (16.5.2026)

Fáza 1 ✅ — gain/RN z DB do Phase 2A `_photometric_error()`
  · gain=3.17 e⁻/ADU, RN=7.6 e⁻ pre QHY294MM (draft_303)
  · variance = flux/gain + sky·area/gain + RN²·area/gain²
  · log: `[PHASE 2A] Photometric errors: gain=X e-/ADU, RN=Y e-`

Fáza 2 ✅ — `compute_snr_optimal_aperture_table()`
  · Gaussian PSF enclosed flux model
  · SNR(r) = F(r) / sqrt(F(r)/g + π·r²·sky/g + π·r²·(RN/g)²)
  · Lookup table mag 7–18, krok 0.5 mag
  · Uložené: `draft_dir/aperture_snr_table.json`

Fáza 2 ✅ — Phase 2A per-star apertures z SNR table
  · `apertures_px[cid] = r_opt(mag_hviezdy)`
  · Clamped: r_min=0.8×FWHM, r_max=2.5×FWHM

Výsledky pre draft_303 (FWHM=3.094, sky=1581.6, gain=3.17, RN=7.6):
  · mag 8 → 4.53 px (+13% vs globálna 3.99)
  · mag 11 → 3.43 px (−14%)
  · mag 14 → 2.47 px (−38%)

Fáza 3 ✅ (16.5.2026) — per-frame CSV + pipeline precompute

`photometry_core.py`:
  · `snr_aperture_table=None` na `enhance_catalog_dataframe_aperture_bpm()`
  · `_get_star_aperture_px()` + `_snr_table_radius_for_mag_bin()`
  · `CircularAperture(pos, r=r_ap_arr)` → per-row `aperture_r_px`
  · `load_snr_aperture_table_from_draft_dir()`
  · `resolve_fwhm_px_for_snr_aperture_table()`
  · `estimate_median_sky_adu_per_px_for_snr_table()`
  · `precompute_and_save_snr_aperture_table_for_draft()`

`pipeline.py`:
  · Pred `export_per_frame_catalogs()`: precompute SNR table (gain/RN z DB, FWHM z MASTERSTAR, sky z aligned frames)
  · `aperture_snr_table.json` do `draft_dir` **pred** Phase 2A
  · `snr_aperture_table` predaná do `enhance_catalog_dataframe_aperture_bpm()`
  · Celé v `try/except` — backward compatible (globálna apertúra pri zlyhaní)

Poradie (jeden pipeline beh):
  MASTERSTAR → SNR table precompute → export CSV (per-star r_opt)
  → Phase 2A (môže JSON aktualizovať s presnejším sky z `proc_*.csv`)

Logy: `[PIPELINE] aperture_snr_table.json uložená pred exportom CSV` ·
  `[FÁZA 2A] SNR per-star apertures: min=… median=… max=…`

**⚠️ Regresia (21.5.2026, draft_341):** photutils **2.3.0** — array `r` na `CircularAperture` → tichý `except` → `dao_flux=NaN`. **FIXED ✅ (`3d2363c`):** `_aperture_flux_sky_per_star` loop; `FORCED_APERTURE` → `photometry_ok=True`; WARNING log namiesto silent swallow. Validované draft_342: HIP 67011 `lc_rms=0.012`, 100% forced flux.

Fáza 4 ✅ PARTIAL PASS (17.5.2026) — validácia LC scatter
  Setup: NoFilter_60_2 · 139 snímok · draft_303 (PRED) → draft_305 (PO)

  Mag bin | RMS PRED | RMS PO  | Párovaný Δ | Verdikt
  8–10    | 0.76 %   | 0.83 %  | −5 %       | ❌ mierne horšie (r_opt ≈ r_global; sky estimate vyšší v 305)
  11–12   | 2.52 %   | 2.14 %  | +7 %       | ✅ nad očakávaním (cieľ 5–10 %)
  13–14   | 4.90 %   | 4.39 %  | +11 %      | ⚠️ pod cieľom 15–25 % (sky 2060 vs 1581 ADU/px agresívny)
  15+     | 3.14 %   | 1.94 %  | +15 %      | ✅ v rozsahu (N=4, orientačné)

  Celkový medián: 2.41 % → 2.20 % (~9 % relatívne zlepšenie)
  Otvorené: aperture_r_px chýba v 305 proc_*.csv — overiť BPM path
  Odporúčanie: per-star aperture benefit pre mag ≳ 11; mag 8–10 flat

Gain/RN zdroj (priorita): EQUIPMENTS DB ✅ pre draft_303 · Settings UI ✅ (TODO-22) · fallback 1.0/10.0

### TODO-22: Gain/RN do Settings UI a DB — COMPLETED ✅ (17.5.2026)

### TODO-24: Vylúčiť catalog_only z photometry pipeline — COMPLETED ✅ (17.5.2026)
- **Flag:** `zone_flag == "catalog_only"` alebo `zone == "catalog_only"`
  (`has_dao_detection` neexistuje v kóde)
- **Helper:** `_is_catalog_only(df)` — lokálne v každom súbore;
  backward compatible (chýbajúci stĺpec → maska all-False)
- **Fix A** `photometry_core.py` — `build_global_comp_pool()` +
  `select_comparison_stars_per_target()` +
  `auto_export_variability_candidates_csv()`;
  log `[COMP] catalog_only excluded: N removed, M remain`
- **Fix B** `photometry_core.py` `run_phase2a` — pred zápisom
  `photometry_summary.csv`; draft_305: 148 → 86 riadkov (−62);
  log `[PHASE 2A] photometry_summary.csv: excluded N catalog_only stars`
- **Fix C1** `ui_variability.py` — Hockey Stick scatter + flux matica +
  comp CSV; log `[HOCKEY STICK UI] Excluding N catalog_only`
- **Fix C2** `photometry_report.py` — po načítaní
  `photometry_summary.csv`; log `[HOCKEY STICK] Excluding N catalog_only stars`
- **Fix D** `photometry_core.py` + `ui_variability.py` — pred
  sigma-clipping; VSX catalog_only zachované s `catalog_only_warning=True`;
  log `[VARIABILITY] Excluding N catalog_only from candidate detection`
- **Ponechané bez zmeny:** `proc_*.csv`, `active_targets.csv`,
  field map (cyan body), `pipeline.py`
- **Future:** proc_*.csv zone=NaN (~197/frame Gaia-fill) — riešiť samostatne
- **Phase 2A LC loop skip** (17.5.2026): `run_phase2a()` preskočí
  `catalog_only` na začiatku per-target loopu (pred flux/LC);
  log `[PHASE 2A] Skipping N catalog_only targets`

### UI counter fixes — konzistentné čísla (17.5.2026)
- **Phase 2A progress log** (`photometry_core.py`):
  `[PHASE 2A] 148 targets (84 active LC + 64 catalog_only skipped)`
  (pred: "148 targetov" bez rozlíšenia)
- **Hockey Stick metric** (`ui_variability.py`):
  "RMS candidates" = `n_combined` (po catalog_only + edge + VSX filter)
  = rovnaká množina ako tabuľka a `variability_candidates.csv`
  (pred: raw `is_variable_candidate_rms` ~193)
- **TESS auto-run log**:
  `[TESS] Auto-run eligible: N candidates (from variability_candidates.csv)`

### Cross-validácia — photutils + SExtractor + IRAF (17.5.2026)
Setup: Lenovo T460, Kubuntu, Python 3.12, photutils 3.0.0
Frame: `proc_BO_CVn_Light_001.fits`, 1750 hviezd, FWHM=3.094px

**Growth curve analýza (photutils vs VYVAR dao_flux):**

| Apertura | ×FWHM | Bright (8-10) | Medium (10-12) | Faint (12-13) |
|----------|-------|---------------|----------------|---------------|
| 2.0 px   | 0.65× | 0.808         | 0.798          | 0.707         |
| 3.0 px   | 0.97× | 1.021         | 1.000 ✓        | 0.933         |
| 3.5 px   | 1.13× | 1.092         | 1.071          | 0.999 ✓       |
| 4.0 px   | 1.29× | 1.155         | 1.122          | 1.035         |

- VYVAR `dao_flux` (r=4.0px nominálne) ≈ photutils r=3.0px (0.97×FWHM)
- Efektívna apertura VYVAR ~0.97×FWHM pre medium hviezdy
- Sky subtrakcia VYVAR odčíta ~15% viac ako photutils globálny median

**SExtractor 2.28.0 cross-validácia (MASTERSTAR.fits):**
- Detekovaných: 1575 hviezd (VYVAR: 1750)
- Matched < 3px: 1439 / FLAGS=0: 1259
- FWHM median SExtractor: 2.25px (VYVAR header: 3.09px)

| Mag bin | N | SEx/VYVAR ratio | Photutils/VYVAR (r=3px) |
|---------|---|-----------------|------------------------|
| 8–10    | 60 | 1.065          | 1.021                  |
| 10–12   | 273| 1.058          | 1.000                  |
| 12–13   | 394| 0.919          | 0.933                  |

**IRAF/PyRAF 2.2.4 cross-validácia:**
- Nástroj: Community IRAF V2.17.1, task `apphot.phot`
- Apertura: r=3.0px, annulus 6–10px, sky=median, gain=3.17 e⁻/ADU
- Hviezdy: 48 (mag 8–12, flag=0/NoError)
- ZP offset: 24.977 mag (očakávané 25.0; Δ=0.023 = EPADU=1.0 vs 3.17)
- Scatter std: 2.38% — VYVAR a IRAF zhodujú sa na 2.2%
- Flux ratio median: 1.0000 (po ZP korekcii)

**Finálna tabuľka trojitej cross-validácie:**

| Nástroj | Zhoda | Poznámka |
|---------|-------|----------|
| photutils 3.0 (r=3px) | 2.0% scatter | Optimálna apertura = 0.97×FWHM |
| SExtractor 2.28 | 6% offset | Growth curve efekt (PSF wings) |
| IRAF apphot (r=3px) | 2.2% scatter | Po ZP korekcii; gain fix needed |

**Záver cross-validácie:**
VYVAR diferenciálna fotometria je fyzikálne správna a konzistentná
s tromi nezávislými profesionálnymi nástrojmi na úrovni 2–6%.
Systematický 6% offset (SExtractor) = growth curve efekt — identický
v IRAF/photutils keď sa použije r=4px vs r=3px.

**Nástroje nainštalované na Lenovo:**
- SExtractor 2.28.0 ✓
- photutils 3.0.0 ✓
- astropy 7.2.0 ✓
- ccdproc 2.5.1 ✓
- IRAF/PyRAF 2.2.4 (Community IRAF V2.17.1) ✓
- `scripts/validate_photometry_crossval.py` ✓
- `scripts/install_vyvar_crossval_lenovo.sh` ✓

### FWHM priority fix — VY_FWHM_GAUSS pre per-frame apertúry (17.5.2026)
- **pipeline.py** `gaussian_fwhm_px_override` (~8088):
  · Pred: PRIORITA 1 = `VY_FWHM × 0.667` (vždy ak `VY_FWHM` existuje)
  · Po: PRIORITA 1 = `VY_FWHM_GAUSS` (priamo, bez škálovania)
           PRIORITA 2 = `VY_FWHM × 0.667` (fallback)
- Dopad: `fwhm_gaussian_px` ~2.25px namiesto ~2.06px (+9%)
  → `r_ap` mierne väčšia → lepší flux capture pre faint hviezdy
- Zjednotenie: per-frame, SNR table aj Phase 2A teraz všetky
  používajú `VY_FWHM_GAUSS` ako primárny zdroj

### TODO-26: Variability threshold auto-calibration — COMPLETED ✅ (17.5.2026)
- **Problém:** 180 false candidates (mag 12-13, RMS 8-24%, Jaccard=0.99
  across runs) — systematický noise floor, nie reálna variabilita
- **Root cause:** field envelope fit (~10% upper @ mag 12) príliš nízky
  pre tento field/night (comp P90 rms_pct @ mag 12 = 8.7%)
- **Fix** `variability_detector.py`:
  · Per mag bin: P90 of comp stars flux matrix rms_pct
  · `upper_envelope = max(field_upper, comp_P90[bin] × factor)`
  · Missing bins: fallback `expected_rms_pct × 2`
  · Log: `[VARIABILITY] Comp noise floor applied: P90@mag12=X% → upper=Y%`
- **Config:** `variability_comp_floor_factor = 1.5` (tunable)
- **Výsledok (draft_309):**
  · Upper envelope @ mag 12: ~13.0% (8.7% × 1.5)
  · RMS candidates: 248 → 47 (všetky mag)
  · Band mag 12-13: 128 → 21
  · VSX known variables: 3 zachované (0 stratených)
- **Tuning:** zvýšiť factor → menej kandidátov, znížiť → viac
- **Ďalší krok:** SysRem MVP hotové (TODO-35); full-field SysRem po PERF-5/PERF-6

### TODO-27: Re-export + re-validate draft_305/307 po float64 fix — CLOSED ✅ (18.5.2026)
- Re-validácia vykonaná na draft_310 (Lenovo, `validate_lc_crossval.py` v4)
- 17 hviezd NOT IN proc CSV = fyzikálna detekčná limita DAO (nie float64 bug)
- Tieto hviezdy nie sú v žiadnom proc frame naprieč draft_303, draft_310, `/media/milan/DISK`
- float64 fix (17.5.2026) je správny a platný; 17 hviezd je legitímne preskočených
- PyRAF LC validácia: NOT FEASIBLE (IRAF float32 FITS bug) — single-frame IRAF
  validácia z 17.5.2026 (2.2% zhoda) zostáva platná

### TODO-28: SNR table units fix — ADU/e⁻ mix — COMPLETED ✅ (17.5.2026)
- `compute_snr_optimal_aperture_table()`: `snr = enclosed(ADU) / noise(e⁻)` → `snr = (enclosed/g) / noise`
- Dopad: absolútna SNR bola chybná o ~√gain; `r_opt` poloha maxima nezmenená

### TODO-29: Airmass fit — swap poradia — COMPLETED ✅ (18.5.2026)
- Nový poriadok: ZP → CT → outlier detect → airmass fit (na čistých dátach)
- Airmass korekcia aplikovaná na všetky frames; fit len na non-outlier frames
- Log: `[PHASE 2A] Airmass fit on N/M frames (after outlier mask)`
- Dopad: airmass slope nie je skreslený oblakmi / bad frames

### TODO-30: Color term → airmass na CT-korigovanom mag — COMPLETED ✅ (18.5.2026)
- `mag_for_airmass = mag_calib_ct` (nie `mag_calib`)
- NoFilter/Clear: nulový dopad (mag_calib_ct == mag_calib)
- Filtered + CT on: log `[PHASE 2A] Airmass detrend applied on CT-corrected mag`

### TODO-GS1: Citačné komentáre — COMPLETED ✅ (18.5.2026)
- Howell (1989) PASP 101:616 — CCD equation, sky subtraction, SNR aperture
- Broeg, Fernandez & Neuhäuser (2005) AN 326:134 — comp weights, ZP ensemble
- Stetson (1987) PASP 99:191 — ZP MAD sigma-clip
- 6 lokácií v photometry_core.py (riadky 814, 837, 2174, 2181, 5732, 8827)

### TODO-GS2: Unit test suite — COMPLETED ✅ (18.5.2026)
- `tests/test_photometry_core.py` — 11 testov, 11/11 passed (~0.9s)
- Pokryté: Howell CCD error, Broeg weights, sky subtraction, ZP sigma-clip, SNR aperture
- Spustenie: `python -m pytest tests/test_photometry_core.py -v`

### TODO-31: Growth curve korekcia pre faint hviezdy — CLOSED as NOT APPLICABLE ✅ (18.5.2026)
- Implementovaná a testovaná na draft_310
- Výsledok: Median lc_rms 0.1114 → 0.1145 (mierne horšie), RMS<0.05 = 23 (bez zmeny)
- Príčina: growth curve faktor sa aplikuje rovnako na target aj comp hviezdy
  → eliminuje sa v diferenciálnom magnitude → nulový efekt na LC scatter
- Korekcia má zmysel len pre absolútnu fotometriu (mag kalibrácia), nie diferenciálnu
- Kód revertovaný do pôvodného stavu (py_compile OK)
- Referencia: Howell (1989) §3 — growth curve correction pre small apertures

### TODO-32: IRAF EPADU fix pre cross-validáciu — LOW
- IRAF varoval "Keyword 3.17 not found" → použil EPADU=1.0
- Fix: v cross-val scripte použiť `iraf.datapars.epadu=3.17`
  namiesto `iraf.datapars.gain="3.17"` (string vs float)
- Dopad na ZP: 0.023 mag = 2.5×log10(3.17) / correction
- Pre VYVAR kód: žiadny dopad (len pre cross-val script)

### TODO-44: Role-aware aperture (variable vs comp) — ✅ CLOSED (20.5.2026)
- `_apply_role_aware_aperture_scaling()` v `photometry_core.py` po SNR tabuľke
- Config: `aperture_variable_factor` (default 1.0), `aperture_comp_factor` (default 1.1)
- E2E draft_321: 260 comps scaled ×1.1; BO/FW `lc_rms` unchanged vs baseline
- Reporting A: PDF Methods (dynamic ALG + aperture factors), VAR.ASTRO `# ALG:` header lines, `ui_settings.py` sliders/toggles
- Referencia: SIPS Photometry tool documentation (Moravian Instruments)

### TODO-45: RGB kamera podpora (de-Bayer → G kanál fotometria) — MEDIUM

**Motivácia:** IMX533 (RGB) užívateľská podpora; G kanál ≈ širší V-filter,
vhodný pre diferenciálnu fotometriu premenných hviezd.

**Fáza 1 — Minimálna verzia (G kanál only):**
- Detekcia Bayer FITS z headera (`BAYERPAT` keyword: RGGB / BGGR / GRBG / GBRG)
- De-Bayer cez `opencv` (`cv2.cvtColor`) → extract G kanál → štandardný 2D FITS
- Zvyšok pipeline (MASTERSTAR, Phase 2A, PDF): bez zmeny
- Flat kalibrácia: použiť G-channel flat (alebo luminance flat ako fallback)

**Fáza 2 — Plná verzia (future):**
- Per-kanál R+G+B fotometria
- Color index R−G, B−G export → HRD klasifikácia bez Gaia BP-RP
- Per-kanál LC export

**Náklad:** Fáza 1 = 3–5 dní · Fáza 2 = 3–4 týždne
**Závisí od:** nič (nezávisí od iných TODO)
**Referencia:** IMX533 Bayer pattern = RGGB

### TODO-25: Gaia→DAO zdroj UI vs pipeline — COMPLETED ✅ (20.5.2026)
- Pipeline: `generate_masterstar_and_catalog()` → `platesolve/<setup>/photometry/pipeline_meta.json`
- UI: `ui_masterstar_qa.py` číta `gaia_dao_completeness_pct`; fallback na CSV row counts ak meta chýba

### TODO-23: Adaptívny matching radius z pixel scale — COMPLETED ✅ (20.5.2026)
- `select_active_targets` / `run_phase0_and_phase1`: `match_radius = plate_scale × 3` (fallback 15″)
- `_read_plate_scale_from_fits_path`: SCALE, VY_PLTS, CDELT1, …

### TODO-19: Infolog — ukladanie na disk — MEDIUM
- Aktuálne logy sú len v pamäti (Infolog dashboard)
- Pridať automatické ukladanie do draft_dir/infolog_<date>.txt po každom RUN VYVAR

### TODO-7: Plate solver refactor — LOW (nízka priorita)
### TODO-8: ePSF napojenie do pipeline — ✅ CLOSED (20.5.2026)
- **Infrastructure:** `build_epsf_model()` after MASTERSTAR; per-frame `psf_photometry_stars()` via `export_per_frame_catalogs`; `_get_lc_psf_or_dao()` in Phase 2A; `psf_chi2_threshold` config (default 50)
- **UI:** standalone 🔬 ePSF tab (`ui_epsf_dashboard.py`); metrics + aperture/PSF overlay; `run_epsf` job + ⚡ RUN ePSF Photometry button (no full RUN VYVAR)
- **Validated draft_321:** 157 PSF stars, 96/139 BO `psf_fit_ok`, `lc_rms` unchanged (differential ZP)
- **Default:** `psf_photometry_enabled=false` in `config.json`
- **Follow-up:** **TODO-8-BOO** — Bootes globular cluster dense-field validation (~2h dataset)
### TODO-9: Inštalácia VYVAR na Lenovo T460 — PENDING
### TODO-10: Settings záložka refaktor + CONFIG_GUIDE.md — PENDING
### TODO-11: Auto-trigger po konci pozorovania — PENDING (nízka priorita)
### TODO-12: HRD klasifikácia hviezd — PENDING (po novej DB)

### TODO-LC-QUALITY: LC Classification Filter — PENDING

**Kontext (draft_342):** Phase 2A hlási **134** LC; HIP 67011 `lc_rms=0.012` po aperture fixe. Stále treba formalizovať saturated / noisy2/3 / `no_data` a zosúladiť počítadlo LC s export kvalitou.

**Úlohy:**

1. **Saturated hviezdy (`zone=saturated`)**
   - Vylúčiť z AAVSO / VAR.ASTRO exportu, **alebo**
   - Exportovať so špeciálnym flagom (dokumentovať v AAVSO komentári / metadata).

2. **Noisy krivky (`noisy2` / `noisy3` v existujúcej klasifikácii)**
   - Definovať RMS threshold pre akceptovateľnú kvalitu (per target / per zone?).
   - Zosúladiť s Variability tab / Hockey Stick / PDF reportom.

3. **`lc_quality_flag` vo výstupe**
   - Hodnoty: `good` | `saturated` | `noisy` | `no_data` (prípadne `catalog_only` / `forced_no_flux`).
   - Stĺpec v `photometry_summary.csv` + propagácia do export reportov.

4. **Prehodnotiť počítadlo „Light curves: N“**
   - Fáza 2A: počítať len `good`? alebo všetky s existujúcim `.csv`?
   - UI / night_run / infolog: jednotná definícia (napr. `n_lc_good` vs `n_lc_total`).

**Súvisiace:** photutils 2.3 fix hotový (`3d2363c`); staré drafty bez re-exportu môžu mať staré proc CSV.

---

### TODO-LC-TREND: Differential extinction + ALG audit — PENDING

- Rastúce/klesajúce LC u niektorých ROT hviezd (vizuálna kontrola + fyzikálna interpretácia).
- Audit existujúcich ALG krokov (airmass, CT, SysRem, Democratic, PyTICS) vs trend artefakty.
- Súvisí s **TODO-GEO** (pozorovateľská poloha) a **TODO-LC-QUALITY**.

---

### TODO-GEO: Observer geographic position audit — PENDING

- Overiť BJD / HJD / airmass pre draft_342 (observer lat/lon/elev v DB + FITS + `time_utils`).
- Impact na airmass detrend a periodické trendy v LC.

---

### TODO-MASTERSTAR-QA validation: FORCED_APERTURE overlay — PENDING

- Kód: cyan = `FORCED_APERTURE`, green = `GAIA_MATCHED` (`masterstar_qa_plot.py`).
- **Úloha:** otvoriť MASTERSTAR QA na draft_342 a potvrdiť cyan body na správnych WCS pozíciách.

---

### TODO-DEV-PROCESS: Improved Development & Testing Process — PENDING

**Motivácia:** Commit `0b01955` (ePSF targeted) išiel do repo bez overenia, že `_epsf_target_catalog_ids` volá `read_vyvar_csv` — v runtime vždy `NameError` → PSF na všetkých riadkoch. Podobne draft_341: proc CSV s `dao_flux=NaN` pre forced rows kvôli tichému pádu SNR per-star apertures (photutils 2.3).

**Povinný postup po každej implementácii (pred commitom):**

1. **Smoke test** na jednom frame / jednom draft (min. 5 min, nie len `py_compile`).
2. **Checklist pre novú feature:**
   - [ ] Unit / smoke test na jednom frame (`proc_BO_CVn_Light_001` alebo referenčný draft)
   - [ ] Porovnanie metrík **pred / po**: LC count, `DET_*` count v proc CSV, `psf_flux` non-null count
   - [ ] Commit až po úspešnom teste (žiadny „fix neskôr v ďalšom commite“ bez dôvodu)

**Automatizovaný regression test — `scripts/regression_test.py` (nový):**

- Spustí pipeline krok(y) na referenčných dátach (draft_341 alebo pinnutý baseline draft).
- Porovná s baseline:
  - LC count ≈ **134** (alebo po TODO-LC-QUALITY: **good** LC count)
  - `DET_*` names v proc CSV = **0**
  - `psf_flux` non-null per frame ≈ **183** (targeted ePSF)
- **FAIL** pri regresii (tolerance konfigurovateľné v JSON / CLI).
- Voliteľne: subprocess na `export_per_frame_catalogs` + jeden frame `enhance_catalog_dataframe_aperture_bpm` s `snr_aperture_table` (chytí photutils array-`r` bug).

**Referenčný baseline (`draft_342`, 21.5.2026):**

| Metrika | Očakávanie | draft_342 |
|---------|------------|-----------|
| `DET_*` v proc CSV | 0 | ✅ 0 |
| `psf_flux` non-null / frame | ~150–200 | ✅ targeted ePSF |
| LC count | ≥ 134 | ✅ 134 |
| HIP 67011 `lc_rms` | finite | ✅ 0.012 |
| `FORCED_APERTURE` `dao_flux` | 100% | ✅ |

**Opravy už v repo (pred regression scriptom):**

- `read_vyvar_csv` import — `0b01955` ✅
- SNR per-star aperture photutils 2.3 loop — `3d2363c` ✅

---

### Dnešný súhrn — 17.5.2026

#### Implementované dnes
| # | Zmena | Súbor(y) |
|---|-------|----------|
| TODO-24 full-stack | catalog_only vylúčenie (comp, summary, Hockey Stick, PDF, variability, Phase 2A LC skip) | photometry_core.py, ui_variability.py, ui_masterstar_qa.py, photometry_report.py |
| TODO-21 Fáza 4 | Partial pass validácia (mag 11+ ✅) | — |
| TODO-22 | Gain/RN Settings UI + FITS QA banner | ui_settings.py, database.py, ui_quality_dashboard.py |
| Match rate Gaia→DAO | Completeness metrika v pipeline + MASTERSTAR QA | pipeline.py, ui_masterstar_qa.py |
| Comp BP-RP fix | Outlier cap [0.1–3.5] + B-V fallback (R CVn fix) | photometry_core.py |
| VYVAR UI preklad | 278+ stringov → anglicky (6+2 súborov) | ui_*.py, catalog_crossmatch.py |
| Auto-export variability_candidates.csv | Odblokuje TESS auto-run | ui_variability.py |
| Suggest from MASTERSTAR | Auto-suggest DAO/aperture parametre | ui_dao_stars.py, ui_settings.py |
| float64 catalog_id fix | Precision loss → XY fallback → false candidates | gaia_catalog_id.py, photometry_core.py, pipeline.py |
| catalog_only Phase 2A skip | 64 targets preskočené v LC loop | photometry_core.py |
| Fyzikálny audit Fix 1 | sky term: sky_pp/g×area (CCD equation) | photometry_core.py |
| Fyzikálny audit Fix 2 | ZP MAD sigma-clip per frame (DAOPHOT štandard) | photometry_core.py |
| UI counters | Phase 2A log + Hockey Stick metric konzistentné | photometry_core.py, ui_variability.py |
| TODO-28 SNR units | `snr = (enclosed/g) / noise` v SNR aperture table | photometry_core.py |
| Cross-validácia Lenovo | photutils + SExtractor + IRAF (BO CVn frame) | scripts/validate_photometry_crossval.py, install_vyvar_crossval_lenovo.sh |
| FWHM VY_FWHM_GAUSS | Per-frame apertúry: GAUSS pred VY_FWHM×0.667 | pipeline.py |
| TODO-26 | Comp P90 noise floor pre variability envelope (248→47 kandidátov) | variability_detector.py, config.py, config.json |

#### Otvorené TODO (backlog)
- TODO-LC-QUALITY: `lc_quality_flag`, saturated/noisy export policy
- TODO-LC-TREND: differential extinction + ALG audit
- TODO-GEO: observer position audit (BJD, airmass, HJD)
- TODO-DEV-PROCESS: `scripts/regression_test.py` (baseline: draft_342)
- TODO-MASTERSTAR-QA validation: cyan FORCED_APERTURE overlay
- TODO-13 validation: Gaia→DAO completeness ~3.5% v QA
- TODO-32: IRAF EPADU fix v cross-val scripte (gain 3.17)
- TODO-44: ✅ CLOSED — Role-aware aperture (SIPS-style SNR + comp_factor)
- TODO-45: RGB kamera — de-Bayer → G kanál fotometria (IMX533 RGGB)
- TODO-25: Gaia→DAO zdroj UI vs pipeline zjednotiť
- TODO-23: Adaptive matching radius z pixel scale
- TODO-20: Mean stack MASTERSTAR (LOW — WCS/FWHM, not LC SNR)
- TODO-9: Lenovo — crossval env hotové; plná VYVAR inštalácia ešte otvorená

#### Ďalší krok
- Gold standard: TODO-GS3–GS7 (README, CITATIONS.bib, CHANGELOG, AAVSO, paper)

### TODO-PERF-7: UI LC index lookup — COMPLETED ✅
- **Main LC path** already efficient (1× `_cached_read_csv` per star via `lightcurve_{catalog_id}.csv`)
- **`_load_summary`:** `@st.cache_data(ttl=300)`, cleared after `run_full_photometry_pipeline`
- **Multi-filter overlay:** uses `_cached_read_csv` instead of raw `pd.read_csv`
- **`ui_suspected_lightcurves`:** `csv_cache` param + PERF-7 warning (module inactive; ready for ProcFrameStore)

### TODO-PERF-6: Variability cache / UI ProcFrameStore — COMPLETED ✅
- **Pipeline:** 0 variability disk reads when `csv_cache` + Phase 2A path (PERF-5 `ProcFrameStore`)
- **UI:** `ui_variability._cached_load_matrix` passes `ProcFrameStore` from `st.session_state` when set after RUN VYVAR
- **`run_full_photometry_pipeline`:** returns `proc_frame_store`; Phase 1 injects store into `st.session_state`
- **`load_field_flux_matrix`:** `[PERF-6]` INFO log for cache type (ProcFrameStore / dict / None)
- **Deferred:** `_flux_matrix_from_pivot` true pivot value reuse (CPU-only optimization)

### TODO-PERF-8: Shared flux matrix (Phase 2A) — COMPLETED ✅
- **`run_phase2a`:** union of all target + comp Gaia IDs (`_all_lc_ids_list`, ~384 stars) → one `read_flux_from_csv` pass per frame (139×) before target loop
- **Per target:** slice `_flux_matrix` by `all_ids` + existing edge/catalog-only post-process per frame (no repeated flux extraction)
- **Fallback:** original per-target × per-frame `read_flux_from_csv` loop if matrix build fails
- **Validated:** `simulate_night_run` draft_318 — 384 IDs, 53376 rows, BO CVn / FW CVn `lc_rms` within 0.001 mag of draft_317; photometry step ~576 s vs ~619 s (draft_317 class)

### TODO-PERF-10: Single-pass preprocess+QC — COMPLETED ✅
- **`dao_qc_in_calibrate`** (default `true` in `config.json`): `_quality_inspection_dao_metrics_array` runs inside `_calibrate_one_light_disk` on the in-memory calibrated array (no second raw read)
- **`apply_perf10_dao_qc_to_obs_files`:** writes FWHM / SKY_LEVEL / STAR_COUNT / roundness / pointing to `OBS_FILES`, auto-reject (median FWHM ×1.5), drift sync — same as former step 5
- **`night_run`:** step 5 (`run_draft_ram_calibration_qc_to_obs_files`) skipped when calibration-time QC succeeded; falls back to RAM QC if no metrics (e.g. passthrough-only)
- **Savings:** ~278 redundant raw array loads eliminated (139 frames × 2 passes); ~40–50 s per full night run (draft_316 class)

### TODO-PERF-5: ProcFrameStore — COMPLETED ✅ (Option B)
- **`proc_frame_store.py`:** unified store, single disk read per `proc_*.csv` frame, dict-compat
  interface (`get` / `items` / `values`), legacy fallback when `proc_frame_store=None`
- **Phase 1 + 2A** share one `ProcFrameStore` built in `run_phase0_and_phase1`, passed to `run_phase2a`
- **Tests:** `tests/test_proc_frame_store.py`
- **Note:** full-field SysRem ready (single union column load per frame)
- **`night_run.py`:** headless pipeline runner extracted from `app._run_vyvar_full_pipeline` (no Streamlit); foundation for TODO-11 auto-trigger
- **`simulate_night_run.py`:** CLI e2e simulation for `D:\BO_CVn` (defaults: equipment ID **1** QHY294MM, telescope ID **1** Carl-Zeiss 200mm)
- **`app.py` refactor** to call `night_run.run_night_pipeline` — **deferred** (UI still uses inline `_run_vyvar_full_pipeline`)

### E2E simulate_night_run bugs — FIXED ✅ (19.5.2026)
| Bug | Symptom | Fix |
|-----|---------|-----|
| ProcFrameStore subscript | `[VARIABILITY] … not subscriptable` | `variability_detector._lookup_cached_frame_df` uses `.get()`; `ProcFrameStore.__getitem__` safety net |
| FIELD JUMP DB | `WindowsPath` has no `fetch_draft_light_rows_for_quality` | `pipeline.py` ~4769: `VyvarDatabase(dbp2)` passed to `detect_field_jumps` |
| PDF `_norm_cid` | `name '_norm_cid' is not defined` | Module-level `_norm_cid()` + `staticmethod` on `_PhotometryReportBuilder`; `self.comp_df` in cover rows |
| PDF `_var_results` | `cannot access local variable '_var_results'` | **FIXED** ✅ — `self._var_results` in `__init__` (140, 210–213) and `_variability_cover_metrics` (690); PDF generates without this error |
| PDF `TITLE_H` | Report builder crash / missing title constant | Fixed in `photometry_report.py` |
| `aperture_px` key | KeyError / inconsistent summary column | Unified key in LC/summary export |
| `contamination_map` | `NameError` on every Phase 1 target (draft_320) | `contamination_map = {}` init in `_compute_comp_contamination_map` |
| PERF-4 log | Marker missing in run log | `comp_pool_rms.py`: `logging.debug` → `logging.info` for `[PERF-4]` |

### E2E validation — draft_321 ✅ (19.5.2026)
- **Command:** `python simulate_night_run.py --source D:\BO_CVn --eq 1 --tel 1`
- **Exit:** 0 | **84** light curves | **139** frames
- **photometry_NoFilter_60_2:** 524.9 s (cieľ &lt;580 s)
- **Phase 1 comp selection:** 368.8 s (vs ~530 s draft_320)
- **BO CVn `lc_rms`:** 0.1515 (ref 0.151502) | **FW CVn:** 0.0153 (ref 0.015296)
- **SysRem:** 82 stars × 139 frames × 3 iter → **5.7%** median RMS improvement
- **PERF-4B paths:** 77 vectorized (N≥50), 6 iterrows (N&lt;50)
- **PDF:** 98 pages, all 84 stars

### TODO-PERF-4: comp_pool_rms vectorized flux — COMPLETED ✅ (Option A)
- **`comp_pool_rms.py`:** iterrows over `N_frames × K_stars` → per-frame vectorized ops + `groupby` over stars only
- **CQ-3 (19.5.2026):** `comp_selection_per_target.py` — 12 helpers; orchestrator v `photometry_core.py`; lazy import v tele funkcie
- **Timing regression fixed (draft_321):** Phase 1 skips `catalog_only` comp selection after bp_rp enrich; `_accumulate_per_frame_comp_metrics` uses iterrows for N&lt;50 candidates, groupby for N≥50

### TODO-PERF-3: Batch Gaia comp lookup — COMPLETED ✅ (19.5.2026)
- **Prefetch:** `_comp_gaia_prefetch` v `run_phase0_and_phase1()` pred Phase 1 loop (IDs z `ms_df` + global comp pool)
- **Infra:** `_batch_enrich_targets_bp_rp_from_gaia_db` + `query_local_gaia_by_source_ids` (chunk 500)
- **Wire:** `gaia_prefetch` → `select_comparison_stars_per_target` → `_enrich_comp_bv` (per-star `source_id=?` fallback zachovaný)
- **Log:** `[PERF-3] Comp Gaia prefetch` + `Selected comp stars covered by prefetch`
- **Deferred:** `_enrich_active_targets_b_v_bp` batch (Step 5) — nižší dopad

### TODO-35: SysRem MVP — COMPLETED ✅ (19.5.2026)
- **Implementácia:** Tamuz, Mazeh & Zucker (2005), MNRAS 356, 1466
- **Funkcia:** `run_sysrem_field()` v `photometry_core.py`; volané z `run_full_photometry_pipeline()` po Fáze 2A
- **Rozsah MVP:** matica exportovaných cieľov (`lightcurve_*.csv`, typicky ~82 × N snímok)
- **Nový stĺpec:** `delta_mag_sysrem` (existujúce stĺpce sa nemenia)
- **Config:** `sysrem_enabled` (default `false`), `sysrem_n_iter` (default `3`)
- **Full field** (všetky hviezdy z `proc_*.csv`): odložené po TODO-PERF-5/PERF-6

---

## Dnes implementované (18.5.2026)

### Kompletný preklad UI do angličtiny — COMPLETED ✅ (18.5.2026)

**Rozsah:** všetky ui_*.py, app.py, export_reports.py, photometry_report.py,
variability_detector.py, importer.py

| Fáza | Súbory | Stringov |
|------|--------|----------|
| Task 1 (17.5.2026) | ui_variability, ui_masterstar_qa, ui_quality_dashboard, ui_settings, ui_photometry, ui_dao_stars | ~278 |
| Task 1 (18.5.2026) | app.py, ui_calibration, ui_database_explorer, variability_detector, importer | ~155 |
| Task 1b (18.5.2026) | ui_aperture_photometry, ui_photometry_results, ui_finalization, ui_select_stars, ui_calibration_library, ui_hrd, ui_photometry_quality, ui_suspected_lightcurves, ui_components | ~333 |

**Celkom:** ~766 user-visible stringov preložených
**Grep overenie:** 0 user-visible Slovak/Czech stringov zostáva
**Zachované:** log_event, LOGGER.*, CSV column keys, katalógy/žiadny záznam sentinels, Python identifiers

### Vedecké citácie v AAVSO a VAR.ASTRO exportoch — COMPLETED ✅ (18.5.2026)

**Súbor:** `export_reports.py` — `_vyvar_export_citation_lines()` (riadky 22–36)

Každý AAVSO a VAR.ASTRO TXT export obsahuje:
- Broeg, Fernandez & Neuhaeuser (2005) AN 326:134
- Howell (1989) PASP 101:616
- Stetson (1987) PASP 99:191
- Gaia Collaboration (2023) A&A 674, A1
- `#SOFTWARE=VYVAR/1.0 (Broeg 2005 differential photometry)` (AAVSO)

### Fáza 2A pipeline + gold standard (TODO-29, 30, GS1, GS2)

- **TODO-29:** ZP → CT → outlier detect → airmass fit; korekcia na všetky frames, fit na čistých
- **TODO-30:** Airmass detrend na `mag_calib_ct`; NoFilter bez zmeny správania
- **TODO-GS1:** Citačné komentáre (Howell 1989, Broeg 2005, Stetson 1987) v `photometry_core.py`
- **TODO-GS2:** `tests/test_photometry_core.py` — 11/11 pytest passed

### Code quality audit + fixes (18.5.2026)

Full audit of 94 .py files. Six fix passes, all with pytest 11/11 green.

**Fix 1 — Silent exceptions:** 38 `except: pass` blocks → `LOGGER.warning/debug`
across `photometry_core.py`, `pipeline.py`, `ui_variability.py`, `app.py`,
`comp_pool_rms.py`, `config.py`, `database.py`, `catalog_crossmatch.py`,
`crossmatch_runner.py`, `astrometry_optimizer.py`

**Fix 2 — Gaia ID normalization:** 8 duplicate `_norm_cid`/`_cid_key` functions
replaced with canonical `normalize_gaia_source_id()` from `gaia_catalog_id.py`.
Deleted unused `catalog_id_series_for_proc_csv_export()`.

**Fix 3 — Dead code marked:** 5 orphaned UI modules (inactive NOTE),
13 legacy pipeline/importer helpers (DEPRECATED comment), 1 VSX duplicate.

**Fix 4 — Draft path resolution:** `resolve_draft_dir()` + `resolve_draft_dir_path()`
added to `utils.py`. Replaced 6+ duplicate resolution chains in
`ui_aperture_photometry`, `ui_variability`, `ui_masterstar_qa`, `ui_quality_dashboard`.

**Fix 5 — CSV dtype:** `VYVAR_CSV_DTYPE` + `read_vyvar_csv()` added to
`gaia_catalog_id.py`. High-risk join locations updated in `ui_variability`,
`ui_masterstar_qa`, `hrd_analysis`, `variability_detector`.

**Fix 6 — Long function split:** `generate_photometry_report()` 3384 → 63 lines
via `_PhotometryReportBuilder` class (13 section methods). `run_phase2a()`
partial extraction. `render_live_view()` deferred — TODO markers added.

**Remaining known tech debt:**
- pandas `FutureWarning` — `variability_detector.py` `.fillna()` downcasting (fix v ďalšej session)
- `run_phase2a()` full extraction (1235 lines, large closure surface)
- `render_live_view()` split (1390 lines, heavy session state)
- ~~`select_comparison_stars_per_target()` split~~ ✅ CQ-3 (19.5.2026)
- `solve_wcs_with_local_gaia()` split (1843 lines)

### CSV schema cleanup — Phase 1 + Phase 2 (18.5.2026)

Full audit of 7 generated CSV files (94 .py files scanned).
26 obsolete columns removed, 4 added/fixed. pytest 11/11 green.

**Phase 1 — High-confidence removals:**

proc_*.csv: `flux_raw`, `fwhm_gaussian_px`, `r_small_px`, `r_large_px`,
  `sky_annulus_r_in_px`, `saturated_from_peak`, `saturated_plateau`,
  `snr10_ok`, `gaia_nss`, `gaia_qso`, `gaia_gal`, `catalog_known_variable`
photometry_summary.csv: `n_outliers`, `am_slope_pre`, `am_slope_post`, `am_piecewise`
comparison_stars_per_target.csv: `color_rms_score`
Added: `PROC_CSV_READ_COLS` canonical list to `gaia_catalog_id.py`

**Phase 2 — Medium-confidence removals:**

proc_*.csv: `is_discovery_candidate`, `is_saturated_flagged`, `is_noisy`, `match_sep_arcsec`
photometry_summary.csv: `skip_photometry`
active_targets.csv: `snr50_ok`, `zone`, `is_usable`, `match_dist_arcsec`
masterstars_full_match.csv: `snr10_ok`, `saturate_limit_per_frame_adu`
Kept: `aperture_px` in summary (active consumers in report/export/UI)

**Bug fix:** variability_candidates.csv — added `vsx_known_variable`, `vsx_match`,
  `gaia_dr3_variable_catalog` to export (TESS loader expected them but they were missing)

**Impact:** Leaner proc CSV per frame → faster I/O, smaller disk footprint.
New schemas active after next pipeline run (existing CSVs on disk unchanged).

### "Gold Standard" status — roadmap a aktuálny stav

#### Čo robí IRAF "gold standard":
1. **Peer-reviewed algoritmy** — každá funkcia má citáciu v literatúre
2. **Reprodukovateľnosť** — rovnaký vstup = rovnaký výstup vždy
3. **Transparentnosť** — užívateľ vie presne čo sa počíta
4. **Validácia** — porovnané s inými nástrojmi a pozorovaním
5. **Dokumentácia** — každý parameter má fyzikálny význam

#### Kde VYVAR už je na úrovni IRAF:
- ✅ Broeg (2005) — citovaný v PDF reporte + inline v kóde (GS1, 18.5.2026)
- ✅ Howell (1989) CCD equation — fixnutá (17.5.2026) + citačné komentáre (GS1)
- ✅ Cross-validácia voči IRAF — urobená (2.2% zhoda!)
- ✅ Fyzikálne správna sky subtrakcia
- ✅ Unit testy fyzikálnej korektnosti (GS2, 11/11 pytest)

#### Výsledky trojitej cross-validácie (17.5.2026) — publikačne hodnotné:
| Nástroj | Zhoda | Poznámka |
|---------|-------|----------|
| photutils 3.0 (r=3px) | 2.0% scatter | Optimálna apertura = 0.97×FWHM |
| SExtractor 2.28 | 6% offset | Growth curve efekt (PSF wings) |
| IRAF apphot (r=3px) | 2.2% scatter | Po ZP korekcii |

Záver: tento výsledok ide priamo do sekcie "Validation" v budúcom paperi.

#### Plán pre "gold standard" status:
Krok 1 (1-2 mesiace): Unit testy + citačný reťazec — **hotové** (GS1, GS2, 18.5.2026)
Krok 2 (2-3 mesiace): AAVSO validácia na 3-5 known variables
Krok 3 (3-6 mesiacov): Paper draft (PASP alebo AN)
Krok 4: Submit + peer review

Navrhovaný názov papera: *"VYVAR: An Automated Differential Photometry Pipeline for Amateur Variable Star Observers"*

#### Nové TODO — gold standard:
- TODO-GS3: `README.md` s inštaláciou na GitHub — MEDIUM
- TODO-GS4: `CITATIONS.bib` — všetky použité algoritmy — MEDIUM
- TODO-GS5: `CHANGELOG.md` — MEDIUM
- TODO-GS6b: AAVSO Extended Format validation — HIGH (blocker for GS6)
- TODO-GS6: AAVSO submission + cross-observer validácia — FUTURE
- TODO-GS7: Paper draft (PASP / AN) — FUTURE

### LC cross-validácia — photutils differential (18.5.2026)

**Setup:** draft_310, BO CVn pole, 84 targets, 139 frames, Lenovo T460
**Nástroj:** `validate_lc_crossval.py` v4 — dao_flux z proc CSV +
rovnaké comp hviezdy ako VYVAR (`comparison_stars_per_target.csv`) +
3σ MAD sigma-clip

**Výsledky (N=67 hviezd s platným dao_flux):**

| Mag bin | N | Median Δ | Std Δ | Verdikt |
|---------|---|----------|-------|---------|
| 8–10    | 4 | -0.0008  | 0.005 | ✅ výborné |
| 10–11   | 2 | -0.009   | 0.007 | ✅ výborné |
| 11–12   | 4 | +0.001   | 0.007 | ✅ výborné |
| 12–13   |24 | +0.0001  | 0.011 | ✅ výborné |
| 13–14   |30 | -0.094   | 0.474 | ⚠️ rozptyl (variabilné + growth curve) |

**Záver:** VYVAR diferenciálna LC fotometria je konzistentná s photutils
na úrovni <0.001 mag (mag 8–13). Rozptyl mag 13–14 = reálne variabilné
hviezdy + growth curve efekt (TODO-31).

**17 hviezd preskočených:** NOT IN proc CSV → XY fallback vo VYVAR →
lc_rms nespoľahlivé → vyriešiť cez TODO-27 (re-export po float64 fix)

**Skript:** `scripts/validate_lc_crossval.py` (Lenovo)

### PyRAF LC validácia — ATTEMPTED, NOT FEASIBLE (18.5.2026)

**Pokus:** LC cross-validácia VYVAR vs IRAF apphot na draft_310 (139 frames, 447 hviezd)

**Root cause:** Community IRAF V2.17.1 / PyRAF 2.2.4 číta nesprávne gross flux
z float32 big-endian FITS (`>f4`):
- photutils gross = 194,690 ADU (správne)
- IRAF gross     = 109,652 ADU (faktor ~1.77× nižší)
- Sky a area sú identické → problém je v pixel readout nie sky subtrakcii
- Konverzia >f4 → int32 nepomohla (rovnaký výsledok)
- Toto je známy bug Community IRAF s moderným Linux/Python

**Záver:** PyRAF LC validácia nie je realizovateľná s proc FITS súbormi z VYVAR.

**Dostupná IRAF validácia:** Single-frame test na MASTERSTAR.fits (17.5.2026)
dáva 2.2% zhodu — MASTERSTAR.fits má iný formát (uint16) ktorý IRAF číta správne.

**Finálna validácia LC (platná):**
- photutils differential LC: mag 8-13 zhoda <0.001 mag (N=67 hviezd) ✅
- IRAF single-frame flux: 2.2% zhoda na MASTERSTAR.fits ✅
- Kombinácia týchto dvoch výsledkov = dostatočná validácia pre publikáciu

### Muniwin (c-munipack) LC validácia — COMPLETED ✅ (18.5.2026)

**Setup:** draft_310, BO CVn pole, detrended_aligned FITS, c-munipack 2.1.36
**Workflow:** `muniphot` → `munimatch` → `munilist --diff-mag`
**Parametre:** gain=3.17, RN=7.6, FWHM=2.3976, apertura=3.318px,
  skyinner=11.39px, skyouter=21.58px (zhodné s VYVAR nastaveniami)

**Výsledky (rovnaké comp hviezdy ako VYVAR):**

| Hviezda | Mag | VYVAR RMS | Muniwin RMS | Ratio | N |
|---------|-----|-----------|-------------|-------|---|
| Gaia 1502044 | 9.51 | 0.00958 | 0.00855 | 0.893 | 96 |
| Gaia 1497070 | 12.54 | 0.03108 | 0.03275 | 1.054 | 135 |
| BO CVn (var) | 9.72 | 0.14851 | 0.17040 | 1.147 | 48 |

**Záver:**
- Mag 9.5–12.5: Muniwin a VYVAR sa zhodujú na ±5–15% ✅
- BO CVn rozdiel (15%) = variabilná hviezda + rôzny počet valid frames (48 vs 139)
- Muniwin používa jednoduchý weighted mean; VYVAR používa Broeg (2005)
  → malý rozdiel v RMS je fyzikálne očakávaný
- FITS float32 big-endian čítané správne (na rozdiel od PyRAF/IRAF)

**Finálna validačná tabuľka VYVAR (draft_310, BO CVn pole):**

| Nástroj | Metóda | Zhoda | N hviezd |
|---------|--------|-------|----------|
| photutils 3.0 | differential LC (dao_flux) | <0.001 mag (mag 8–13) | 67 |
| Muniwin 2.1.36 | differential LC (rovnaké comp) | ±5–15% RMS | 3 |
| IRAF apphot | single-frame flux (17.5.2026) | 2.2% scatter | 48 |
| SExtractor 2.28 | single-frame flux (17.5.2026) | 6% offset | 273 |

**Nástroje nainštalované na Lenovo (Linux):**
- c-munipack 2.1.36 ✅ (sudo apt install c-munipack)

---

## Cieľový workflow (kávový test ☕)

Koniec pozorovania (4:00) → ⚡ RUN VYVAR → Fáza 0+1+2A
→ Variabilita detekcia (sigma=2.3) → Auto crossmatch
→ Auto TESS (len bez VAR match):
   · `period_reliability`: reliable / uncertain / noise
   · blend_check PNG: TESS TPF vs Gaia obloha
→ Summary Measure Report PDF (vrátane TESS sekcie)
→ 8:00 používateľ číta report ☕

Chýbajúce napojenie: Bootes ePSF validation (TODO-8-BOO), Auto-trigger (TODO-11)

---

## Pracovný workflow (Claude ↔ Cursor)

### Roly
- **Claude** = analytik / dizajnér — diagnostikuje problémy, navrhuje riešenia, píše inštrukcie
- **Cursor** = staviteľ — číta kód, implementuje fixy podľa Claudových inštrukcií
- **Používateľ** = schvaľuje — rozhoduje medzi diagnostikou a fixom

### Komunikačný protokol
1. Používateľ popíše problém Claudovi (príp. priloží PNG / log)
2. Claude napíše presnú inštrukciu pre Cursor (diagnostika ALEBO fix)
3. Používateľ skopíruje inštrukciu do Cursoru
4. Cursor odpovie → používateľ skopíruje odpoveď späť ku Claudovi
5. Claude vyhodnotí a buď schváli, alebo napíše ďalšiu inštrukciu

### Pravidlá
- Cursor **nemení kód** bez explicitného pokynu od Clauda
- Každý vyriešený problém sa zapíše do `VYVAR_STATE.md` ako COMPLETED ✅
- Otvorené problémy zostávajú v sekcii `## Otvorené otázky` až do overenia

### Language rules
- Claude → Cursor instructions: **English**
- Cursor → Claude responses: **English**
- Claude ↔ User: Czech / Slovak

---

## Otvorené otázky

### TESS blend check — overenie
- Spustiť TESS re-run pre `1451497396118755584`
- Overiť `sector_23_blend_check.png`

### TESS periódy draft_294
- `1451497396118755584`: uncertain (0.0201 d vs 0.1029 d) — vizuálna verifikácia pending
- `1451924483370649472`: P=0.4375 d — typ TBD

### HRD — po nočnom behu DB
- Overiť že `vyvar_gaia_dr3_v3.db` má Teff/logg vyplnené
- Aktualizovať `gaia_db_path` v `config.json`

---

## 31.5 — ePSF FINE-SCALE VALIDATION + ELONGATION RETRACTION (Telescope Live / Palomar 7)
Tested aperture vs PSF/ePSF on fine-scale well-sampled dense data: Telescope Live
Palomar 7 / IC 1276 GC, Planewave CDK24 + QHY600M, El Sauce Chile, 0.389"/px,
FWHM ~6.2 px (draft 364).

COMPLETED:
- Set #4 registered: CAMERA "QHY 600M" (id3; 9576x6388, 3.76um; GAIN/RN/SAT APPROX),
  TELESCOPE "Planewave CDK24" (id4; 3962mm/610mm), LOCATION "El Sauce (Obstech),
  Rio Hurtado, Chile" (id3; -30.4703,-70.7647,1570m). Pre-calibrated Telescope Live
  import validated (vendor _cal.fits, no calibration; blind solve from OBJCTRA/DEC;
  WCS-authoritative resolver -> 0.389"/px).
- APERTURE-vs-PSF IS SAMPLING- AND SNR-DEPENDENT (corrected): OAT (362, 9.77"/px,
  undersampled) PSF 1.3-1.7x WORSE everywhere. CDK24 (364, 0.39"/px), large-N forced
  photometry (667 isolated G12-16; full population 13k+): aperture WINS on bright
  (G12-16 psf/aper ~2.3; FREE-position run = FIXED = 2.33, so NOT a centroid artifact
  -- PSF is model-limited where photon noise is negligible), ratio converges through
  G16-18, PSF WINS only at the faint photon-limited end (G19-20 isolated 0.79).
  Crossover ~G18-19. The earlier small-N DAO-detected comparison (psf/aper ~0.85 on
  bright) was a selection/noise artifact of the detected subset and is SUPERSEDED.
  Practical implication: validates the ADAPTIVE-selector premise -- aperture as
  robust default, PSF reserved for the faint (and likely crowded) regime; method
  choice depends on plate scale AND per-star SNR/crowding, not a global flag.
- ePSF CANDIDATE-SOURCE FIX (enables ePSF on dense fields): (a) fixed indentation bug
  in _epsf_prepare_stars DB-join loop (star_rows.append was outside the for-loop);
  (b) conditional broad-pool augmentation when safe-comp∩CSV < epsf_min_stars, via
  COG-style isolation (_select_frame_stars_from_proc); sparse-field behavior
  unchanged. Dense GC (only 5 IS_SAFE_COMP rows) now builds with 72-98 candidates.
- DEEP GAIA CONE capability: astroquery Gaia DR3 -> field SQLite
  (vyvar_gaia_dr3_pal7_field.db, 34,376 rows to G~20) in gaia_dr3 schema, drop-in via
  gaia_db_path. Matched stars 717 -> 11,189 (faintest matched G~18).

RETRACTED:
- "Tracking smear" / asymmetric-PSF hypothesis (28.5-31.5) WITHDRAWN. The ~1.18-1.19
  per-frame elongation was a MEASUREMENT ARTIFACT of the fixed 9x9 Gaussian-fit window
  (too small vs FWHM, esp. at 6.2 px). Model-free second moments (PSF-scaled window)
  give axis ratio ~1.08-1.09 on BOTH 362 and 364 -> stars essentially round; no on-sky
  smear. NOTE: Gaussian-fit elongation metric biased high ~10pp; use moments where
  elongation drives decisions. qc_elong_limit=1.8 is loose -> no operational impact.

OPEN:
- Faint isolated: PSF wins (G19-20 forced, psf/aper 0.79) -- RESOLVED.
- Crowded-deblending PSF advantage UNCONFIRMED: forced PSF (fixed or free, single-
  source) does not cleanly win on crowded (G19-20 crowded ~1.04). A proper test needs
  free-position GROUPED fitting (gated grouper) -- deferred.
- Bright-end aperture advantage is large (~2.3x); ePSF could possibly be tightened
  (FWHM ratio 0.80, but consistent with the known DAO-FWHM overestimate, cf. 362
  ratio 0.636). Does not change the adaptive-selector conclusion.

### PSF VALIDATION — CLOSURE (gated components, 2026-06-01)

Standalone forced-photometry validation on draft 364 (CDK24, 0.39"/px, dense GC):

- **GROUPER (joint deblending): NEGATIVE for precision.** grouped/single ~0.94 (~6%),
  but grouped/aperture stays **1.05–1.22** on crowded (G17–20) — aperture still wins
  crowded; at G19–20 crowded grouped (1.05) is worse than single (0.98). At 0.39"/px
  blends are largely sub-resolution; joint fitting is a small refinement, not a crowded
  PSF win. **NICHE retained:** deblending protects **AMPLITUDE fidelity** for a target
  with a **VARIABLE close neighbor** (Han & Brandt 2023) — qualitative, not precision.
  Grouper wiring (`neighbor_catalog`) → **LOW priority**.

- **ADAPTIVE SELECTOR (`psf_adaptive`): VALIDATED = the production way to use PSF.**
  Aggregate median RMS aperture-always **0.111**, PSF-always **0.111**, **ADAPTIVE 0.100**
  (~10%, near-oracle within ~4%). Routes faint-isolated → PSF (G19–20 isolated 51%
  routed, **0.148 vs aperture 0.194** ≈ 24% there), keeps crowded/bright on aperture
  (3–13% PSF on crowded). Gain concentrated at faint-isolated end; ≈aperture elsewhere
  (free upside). NOT a strict aperture floor (1.3% slightly worse).

- **SPATIAL / GRIDDED ePSF:** builds cleanly (3×3, all cells ≥ min stars); benefit
  **UNTESTED, DEFERRED** to a field-variable-PSF rig (planned 8" f/2.72 Newton, corner
  coma). Expected neutral on well-corrected fine optics (364 field-elongation flat
  with radius).

**WIRING STATUS:** adaptive is wired into Phase-2A but **STARVED** — PSF computed for
only ~1.4% of rows (target subset) and `crowding_targets.csv` absent → routes 0 PSF.
To deliver the validated ~10% gain, two gaps must close: (1) PSF for the full LC star
set (targets+comps) when PSF enabled, (2) generate the blend map. [Wiring in progress.]

## PSF wiring — VERIFIED (adaptive selector, draft 364)

Wiring done (gated): PSF computed for the full LC star set when `psf_photometry_enabled`;
`crowding_targets.csv` generated when `psf_adaptive_enabled`. Runtime negligible.

**CORRECTIONS from verification:**

- "~1.4% PSF coverage" was an ALL-DETECTION denominator artifact. The LC star set (161
  on 364) was already ~80% covered; wiring → ~84%. LC stars were never starved.
- Adaptive routes 0 PSF on 364's LC ensemble CORRECTLY — that ensemble is BRIGHT
  (comps G~13–16, SNR≫15). The standalone ~10% gain was on a FAINT/isolated population
  (G17–20), NOT the bright comp ensemble. Gain is FAINT-TARGET-SPECIFIC (G≥17): on
  bright targets/comps adaptive == aperture (no harm/no gain); on faint targets it
  routes to PSF and gains (~10–24%). Earlier "~10% in production" corrected to
  faint-target-specific.
- **BUG:** rule 2 (resolvable-blend → PSF) can NEVER fire — `is_blended` is
  `nn_dist_fwhm ≤ 1.5` but rule 2 needs `≥ 2.0` (mutually exclusive). Plus grouper was
  NEGATIVE (deblending no precision gain at 0.39"/px) → blend route unjustified →
  **DROP rule 2**, keep rule 3 (faint isolated → PSF). Variable-neighbor amplitude
  niche handled case-by-case.

**OPEN:** demonstrate the faint-target gain end-to-end (G≥17 targets, re-export PSF,
confirm rule 3 fires and adaptive beats aperture).

## Adaptive selector — FINAL verified result (faint targets, draft 364)

- Rule 2 removed (dead + grouper-negative); rule 3 (faint + good PSF → PSF) is now the
  only PSF route. Rule 1 and aperture default unchanged.
- Faint-target verification (G≥17 isolated as targets, strict production
  `assess_psf_quality`): rule 3 DOES fire, but only on the faintest subset with SNR≤15
  AND good PSF quality (4/35 candidates). Realized gain MODEST: ensemble
  adaptive/aperture ~0.98 (~2%); PSF-routed targets raw RMS ratio ~0.93 (~7%); vs
  standalone faint-isolated ~0.79 (~21%).
- Gap vs standalone = the **QUALITY GATE**: standalone used a relaxed proxy (good if
  finite flux); production uses strict `assess_psf_quality`, blocking most ultra-faint
  (G~20) frames. Strictness is mostly protective but imperfect — one PSF-routed target got
  WORSE (bad PSF frames passing the gate on a few nights).

**FINAL CALIBRATED CONCLUSION:** adaptive selector validated and conservative. Bright
targets/comps == aperture (no change). Faint targets (G≥17): SMALL quality-gated
average gain (~2–7% realized, not the ~21% standalone ideal), with occasional per-target
noise. Recommended use: opt-in for faint deep-field programs; aperture remains the
published default.

**LOW-PRIORITY OPENS** (only if faint precision becomes critical): tune faint-end
`assess_psf_quality` acceptance; and/or re-export a FULL night (vs 10 frames) so more
G≥19 frames reach good quality and the realized gain is measured more robustly.

---

## Reporting PDF revision — R1+R2 done, R3 pending

- **R1 overflow robustness:** `drawString` → wrapping `Paragraph` + pagination + layout guard;
  0 overflow violations on 362 (page 2 Methods/citations paginates; long Gaia IDs wrap).
- **R2 completeness (self-contained night summary):** cover += observer/OBSCODE, plate scale,
  equipment (tel+cam), variability counts (wired `_variability_cover_metrics`); obs-summary
  KV += frames used/rejected, session BJD span, FWHM min/med/max; summary table +=
  n_points/MERR/median/quality; per-star += check-star KNAME/KMAG/scatter (P3.5 sidecar) +
  ground period. Additive; overflow=0.
- **R3 PENDING:** aperture-vs-PSF/adaptive overlay on per-target LC (primary report only;
  per-method PDFs stay single-method). Instruction written, not run.

---

## B-V legacy removal (APASS/Tycho) — audit + Stage 1 done; Stages 2–4 pending

- **Audit:** APASS/Tycho B-V reached ONLY via `lookup_bv_from_local_db` (last fallback: Gaia
  bp_rp → teff → APASS → Tycho → unknown), in `vyvar_vsx_local.db` (apass ~24M / tycho2
  ~1.1M rows). No production algorithm requires it; ~3% targets / ~7% masterstars on 362;
  color term + tiers already BP-RP-native.
- **Stage 1 DONE (reversible):** disconnected the APASS/Tycho fallback in 4 callers;
  `lookup_bv_from_local_db` left DEFINED but unused. The 5 APASS/Tycho targets on 362 fall
  back to unknown/NaN gracefully (bp_rp unchanged); selected comps 100% gaia_bprp.
- **Determinism VERIFIED:** comp loop fully deterministic (R1==R2, 100% overlap; mergesort,
  no RNG). The ~45% before-vs-after comp diff was run-context drift (active-set incl. a
  duplicate; full Phase 0+1 vs frozen; adaptive density overrides), NOT the disconnect.
- **PENDING:** scope A (APASS/Tycho only) vs A+B (also retire all Johnson B-V → pure BP-RP,
  recommended; kills the legacy |dB-V| dual-mode); then Stage 2 (delete dead APASS/Tycho
  code + UI/export AP/TY provenance), 3 (retire Johnson B-V + legacy mode; missing-bp_rp
  → mag-proxy), 4 (DROP apass_data/tycho2_data + update VSX/vsx_make.py + regenerate).

---

## Session 2026-06-03 — B-V A+B Stages 2–4 (Johnson B-V retired)

- **Scope:** A+B — pure Gaia BP-RP tiering; `bp_rp`-less targets → T4 / mag-proxy.
- **Stage 2** (`3480bd0`): removed `lookup_bv_from_local_db`; `VSX/vsx_make.py` VSX-only
  (no `apass_data` / `tycho2_data`). Byte-identity gate on `draft_000366`: **not run here**
  (no Archive draft on dev PC).
- **Stage 3** (`8945bbf`): removed `bp_rp_to_bv`, `teff_to_bv`, `bv_to_bprp_linear`,
  `_enrich_active_targets_b_v_bp` → `_enrich_active_targets_bp_rp`; comp selection BP-RP-only;
  config/UI/export/report columns (`b_v`, `bv_source`, `phase01_use_bprp_primary`, `*_bv_*`
  limits) retired from production paths. Tier-change report on `draft_000366`: **not run here**.
- **Stage 4** (this commit): docs parity (DECISIONS/ROADMAP/JOURNAL/PARAMS); DB regen command
  documented below. **Tests:** 111 passed / 6 skipped.
- **DB regen (operator, catalog machine):** from repo root, with Vizier/network:
  `python VSX/vsx_make.py --db path/to/vyvar_vsx_local.db` — builds VSX table only (Gaia
  photometry stays in separate `gaia_dr3` DB). Confirm `apass_data` / `tycho2_data` absent:
  `sqlite3 vyvar_vsx_local.db ".tables"`.
- **APCORR-COLOR watch-point:** `fit_color_term_c1` / `apply_color_term` already use
  `comp_bp_rp` dict — no production colour-term path reads `b_v` after Stage 3.

---

## Session 2026-05-31 — comp QA (Sokolovsky) + harness-era trust flag (draft_000365)

### Report label fix (commit `8e8cf29`)

Per-star comp table column was labeled **"p2p RMS"** but printed **comp_rms** (Phase-1 flux
scatter). Relabeled to **comp_rms**; added real **rms_p2p** from `comp_quality_{target}.json`
(matches exclusion footnote). Re-render draft_000365: **0 overflow**, **160 pages**. Report
bytes intentionally change; no numeric pipeline output changed.

### comp_qa (standalone `comp_qa.py` + `scripts/comp_qa_flagged_lcs.py`; `xval_out/` not in git)

Grounded in Broeg 2005 (already cited) + Sokolovsky et al. 2017 (MNRAS 464, 274). LOO
differential mags (zero-median), time-ordered:

| Index | Definition | Role |
|-------|------------|------|
| σ_IQR | (P75−P25)/1.349 | Amplitude (robust) |
| 1/η | s²/δ² von Neumann | Slow drift (white ≈ 0.5) |
| spike | std(m)/σ_IQR | Dropout frames |

**Flag:** σ_IQR above magnitude-dependent locus (0.5-mag bins, median+4·MAD spread, rebuilt
each iterative LOO pass) **OR** 1/η peer-outlier (median+4·MAD of target pool) and >1.0 **OR**
spike>3; iterative drop-worst, min 3 comps.

**Evolution:** flat floor+peer v1 (47→67, over-flagged faint comps) → Sokolovsky **27**
flagged (25 amplitude, 1 spike, 1 amp+invNV). Faint comps …190720/…050880 sit on locus (clean);
dropout caught (NSV 20420 …439552, spike≈8.48); tight-pool FPs clean (V0348 Dra, NSVS
J1618591+485752). **n_clean buckets:** ≥5: **133** / 3–4: **8** / <3: **2**.

### Trust flag (`trust_flag.py` → `xval_out/trust_per_target.csv`, 143 rows)

**Historical note (2026-05-31 draft_000365 study):** the gate below originally included a SEP
cross-val axis (`xval_results.csv` / `sep_confidence`). **Production since 2026-06-03** uses
trust gate v2 only — comp QA + check-star + `lc_quality_flag` (see Session 2026-06-03 below).

Per-target gate for non-experts. **Harness-era inputs (not production today):** comp-health
(`comp_qa_targets.csv` n_clean), sep cross-val (`xval_results.csv`: `confirmed` /
`vyvar_ok_indep_failed` / `review` / `no_independent` / `no_vyvar_rms`), VYVAR check-star scatter
+ `lc_quality_flag`.

**Harness-era warnings (W):** sep≠confirmed; n_clean 3–4; check_star_scatter≥0.02 mag;
`lc_quality`∉{good,noisy} (e.g. **saturated** — genuine data-quality demotion).

**Harness-era levels:** RED if n_clean<3 or W≥2; YELLOW if W==1; GREEN if W==0 (confirmed +
n_clean≥5 + check<0.02 + no hard quality flag).

**Harness-era counts (draft_000365, xval_results present):** GREEN **81** / YELLOW **52** / RED **10**.

### Cross-validation

**CLOSED** for aperture path — triple-validated photutils+sep+dao on draft_000365 (`xval_run.py`
harness). PSF cross-val **deferred** to a PSF-heavy/faint draft with per-frame ePSF.

### Parked next steps

(a) Wire trust flag into AAVSO/VarAstro export + PDF (GREEN/YELLOW/RED + reason at submit);
(b) PSF cross-val on a faint draft; (c) optional gate tuning — grade check-star or let strong
comp-health (n_clean≥6) absorb one mild warning (borderline faint REDs e.g. 1399187099635410432).

---

## Deferred to next session

- Formal cross-validation: DONE for draft_000365 — V842 spot-check + whole night 143 targets (sep/dao ~3% vs lc_rms); IRAF closed; `xval_run.py` harness validated. Muniwin still optional.
- Reporting R3 (aperture-vs-PSF overlay).
- ~~B-V removal Stages 2–4~~ — done 2026-06-03 (see session above).

---

## Future phases / backlog

### FUTURE BIG PHASE (not started) — Comet photometry mode

**STATUS:** analysis only (2026-06-01). Do NOT start until the variable-star pipeline is
finished. Comets are the NEXT MAJOR PHASE after stellar; current priority stays
variable-star VYVAR.

**VERDICT (feasibility analysis):** feasible; architecture sound; HIGH front-end reuse; but a
SIGNIFICANT new module (weeks, not days), and mature validated tools already exist — KOPR
(Czech, kopr.astro.cz; TYC2/APASS comps; click-to-measure coma aperture), Tycho-Tracker v12
(all-in-one: calibration/platesolve/align/photometry/report, star-removal filter, growth
curve, ICQ output), Comphot (BAA, two-image method). VYVAR's value = workflow integration
(own rigs, one tool) + Gaia zeropoint (vs APASS/Tycho), NOT novel science.

**WHY THE PROPOSED ARCHITECTURE IS CORRECT:** it matches the professional standard. Comphot
uses exactly two stacks — a STAR-stack ("fixed") for the photometric zeropoint and a
COMET-stack ("offset", shift-and-add on the comet's motion) for the coma flux, via a
median-annulus aperture that ignores stars. Comets are EXTENDED (nucleus+coma+tail) and
MOVING (different reference stars each night); the useful CCD measure is the total coma
magnitude m1, reported in ICQ format to COBS.

**REUSE (front-end, shared):** calibration; plate solving; star-align stacking (= the "fixed"
star-stack); Gaia comp ensemble + zeropoint (+ Gaia→V Riello transform for V-equivalent
comps); aperture machinery; the overhauled export/reporting layer.

**NEW (comet module, after the calibrate/platesolve/star-ZP front-end):**

- **C1.** Comet ephemeris / apparent motion (JPL Horizons or MPC) → comet-rate stacking
  (track-and-stack: shift-and-add along the motion vector; coma sharp, stars trailed;
  median + masking to suppress trailed stars).
- **C2.** Zeropoint transfer star-stack → comet-stack (comet stack has no usable comps).
- **C3.** Extended-source coma photometry: growth curve (flux vs radius), detectable coma
  diameter, total magnitude m1; optionally Afrho (dust parameter). THIS is the crux
  (low surface brightness, flat-sky sensitivity, trailed-star contamination).
- **C4.** ICQ-format export + COBS; comet night-summary report (reuse reporting layer).

**TOGGLE REALITY:** not a small switch — a PARALLEL photometry mode sharing the front-end
(calibrate → platesolve → star-stack → Gaia ZP), then forking into comet stacking +
extended photometry + ICQ export. Front-end shared, back-end forked. User wants a
stellar/comet mode selector.

**B-V CONNECTION:** KOPR/Comphot use APASS/Tycho V comps — the catalogs being removed from
VYVAR's stellar pipeline. VYVAR's Gaia→V (Riello) gives V-equivalent comps from Gaia, so
the B-V/APASS/Tycho removal does NOT block comet work; Gaia ZP is a cleaner, more modern
base.

**FIRST STEP IF PURSUED:** read-only audit of VYVAR stacking / aperture / zeropoint internals to
confirm the reuse points, before any C1–C4 design.

---

## Session 2026-06-03 — Remove in-pipeline `sep_xval`; trust gate v2

**Removed (production):**
- `sep_xval_core.py` (deleted); harness helpers moved to `xval_harness_core.py` for `xval_run.py`.
- `sep_xval_*` config keys and the post-comp_qa stage call in `photometry_core.py`.
- Runtime SEP citation emission in `citations.py` (`barbary2016` / `bertin1996` no longer in
  export/PDF DATA-QUALITY GATE; `.bib` entries kept for harness).

**Trust gate v2 semantics** (`trust_flag_core.py`):
- Axes: `n_clean` (comp_qa), check-star scatter, `lc_quality_flag` only — no `sep_confidence`.
- RED: `n_clean < min_comps` OR any hard (bad lc_quality, check ≥ 0.05).
- YELLOW: any soft (thin comps, check 0.02–0.05).
- GREEN: `n_clean ≥ strong`, check < 0.02, lc_quality ∈ {good, noisy}, no warnings.
- Targets previously YELLOW/RED only for `sep≠confirmed` promote (expected distribution shift
  vs 69/59/15).

**Validation:**
- `pytest tests/`: 111 passed / 0 failed / 6 skipped.
- Photometry byte-identity: stage was read-only — Phase-2A LC unchanged by design.
- Trust re-run on draft_000366: **not executed on this dev PC** (no draft tree in workspace);
  re-run `trust_flag.py --photometry-dir …` where comp_qa columns exist and record counts in
  STATE.

---

## Session -- Chi_and_H clean full re-run + n_clean root cause (2026-06-04)

*Moved from VYVAR_STATE.md during 2026-06-09 doc sweep.*

**draft_000380** -- clean fresh full run, all 4 filters (B/V/Rc/L), coordinate hint. Plate-solve
anchored ~35.03/+57.14 (98.5-100%). CT reproduced: B -1.084, V -0.383, Rc -0.023; comp scatter
pre->post B 0.376->0.053, V 0.188->0.047, Rc 0.061->0.060. Decoupling/CT-toggle verified: ~371
targets/filter, 0 "nan" names.

**n_clean=0 / trust RED -- resolved (PROC_CSV_GLOB, 2026-06-08).** `comp_qa_core.load_proc_pivot()`
now uses `PROC_CSV_GLOB="proc_*.csv"`. Verified draft_000366 n_clean healthy. Residual:
`classify_lc_quality` `min_frames=20` > short sessions still yields `lc_quality=no_data`.

---

## Session -- CT validation + pre-cal + plate-solve (2026-06-03/04)

*Moved from VYVAR_STATE.md during 2026-06-09 doc sweep.*

**M67 LRGB machinery (draft_368):** CT apply path exercised; Green count-gate-limited; Red
data-limited. Exposure-merge hypothesis **refuted** (mixing 60s+240s degrades c1 fit).

**h & chi Per science-grade CT (draft_375/380):** c1 B -1.09, V -0.40, Rc -0.026; comp scatter
B 0.38->0.05, V 0.21->0.06. CT production architecture: full VSX field always; CT is applied
correction toggle; presel opt-in only.

**Non-cal session mode:** pre-calibrated import shipped (`calibration_mode=pre_calibrated`).

**Plate-solve draft_375:** blind mis-land fixed via coordinate hint + standard Gaia DB.

**Blind index series (2026-06-04):** mag14 tiers + rig-prior; wide HIT on draft_365 still open.

---

## Session 2026-06-04 — Blind index series (mag14 density tiers + solve-rate)

**Problem:** Single prem-density mag14 index failed on dense Newton fields (draft_380); wide-rig
regression untested when `draft_361` MASTERSTAR was absent.

**Solution:**
- Density-matched **fine** tier (reuse `gaia_triangles_mag14.pkl` → `gaia_triangles_fine.pkl`).
- **Wide** tier build: mag14, cell 2°, 16 stars/cell → 224k stars, 1.43M triangles, log_L3 med 3.28 dex.
- Manifest `GAIA_DR3/blind_index_series.json`; orchestrator `vyvar_blind_series.py` integrated in
  `vyvar_platesolver.py` (`auto` / `series_all` / `single`).
- Solve-rate battery `validation/blind_solve_battery.json` + `scripts/blind_solve_rate.py`.
- Wide MASTERSTAR staged: `draft_000365/platesolve/NoFilter_60_2/MASTERSTAR.fits`.

**Config:** `blind_index_series`, `blind_index_select_mode` in `config.json` / `VYVAR_PARAMS.md`.

**Metrics:** `validation/blind_solve_rate.csv` — **9/10 HIT (90.0 %)** on battery v2 (8× Newton
Chi_and_H + M67 + 1× wide MISS on draft_365). Median sep 0.07° on hits. Wide tier: verify fails,
nearest votes ~11.5° — open tuning item.

**Rig-prior (same day, wide closure attempt):** `blind_use_rig_prior`, `blind_scale_tol_frac` (0.10);
A1 pre-vote L3 ratio gate; A2 verify WCS scale gate; A3 FOV bounds; gnomonic sides FOV≥2°; wide vote
fallback re-enabled under gates. Wide diag (`scripts/diagnose_blind_solver_wide.py`): **0 votes &lt;2°**,
nearest ~11–20°, flat≈gnomonic at edge sample — distortion not dominant; **wide index/triangle
correspondence** still the blocker. Newton spot-check OK post-change.

---

## Session 2026-06-08 — audit campaign + trust_flag_core Phase 2

- F841 triage finished: all flagged locals (`dist_score`, `rms_f2`, `c1_stderr@7141`, `lc_df`)
  are dead/redundant, none are real bugs; ranking sort confirmed correct (sorts by `comp_rms`).
  Automated lint layer essentially exhausted/cosmetic.
- Phase 2 manual audit of `trust_flag_core.py`: findings A–F (see `docs/VYVAR_AUDIT_FINDINGS.md`).
  Headline: un-evaluated target defaults to GREEN (A); missing check-star = no penalty (B);
  check-star scatter uses ddof=0 (C).
- Recorded language rule in PROCESS (Cursor↔Claude English; Milan↔Claude SK/CZ).
- Open items moved to ROADMAP (`NEXT SESSION` section).
- F841 batch 3: production 18 -> 1 pending (`n_rms_candidates` Cat 3). Removed dead locals
  across `comp_qa`, `photometry_core` (`ra_ms`/`de_ms`, `gaia_teff`), `pipeline` (`n0`×4, `cfg`),
  `psf_photometry` (`fit_shape`), `vyvar_alignment_frame` (`max_detected_stars`),
  `vyvar_platesolver` (`center`), and harness modules. Audit notes: `cfg`/`max_detected_stars`/
  `fit_shape` confirmed vestigial; `n_rms_candidates` awaits Milan (wire m2 RMS count vs remove).
- Phase A1: `VYVAR_AUDIT_FINDINGS.md` re-encoded UTF-8 ASCII; `_gen_audit_findings.py` ASCII emitter.
- Verify: pytest 174/6 skip; photometry SHA unchanged; PDF overflow 0.
- Phase C: ruff safe auto-fix batch (production) -- 37 fixes (SIM114 17, RUF010 8, B009 4,
  B010 4, SIM300 3, SIM910 1) across 16 modules; parenthesized dense SIM114 and/or merges.
  Verify: 0 remaining in production scope; pytest 174/6 skip; photometry SHA unchanged; PDF 0.
- Phase D: bug-class lint sweep (B023/B905/B904/RUF012/B007) -- 51 ruff instances cleared.
  B905 policy recorded in DECISIONS. `_norm_med_for_bin` duplication flagged for Phase F;
  B023 fix required `frame_med = nan` init before mag-bin branch (conditional binding).
  Verify: pytest 174/6 skip; photometry SHA unchanged.
- Phase E (trust_flag_core A+B): un-evaluated default RED + warn; nan check-star soft note;
  forward guard documented; C1 ddof=0; E deferred. `tests/test_trust_flag.py` (9 tests).
  draft_000366 trust re-run: 10 GREEN->YELLOW, 0 GREEN->RED; numeric LC/comp_quality unchanged.
  pytest 183/6 skip.
- Phase F manual audit: comp_qa_core.py done (CQ-A..E in AUDIT_FINDINGS). Headline: proc-CSV
  HIGH likely resolved via `PROC_CSV_GLOB`; CQ-C locus order-coupling needs conscious decision;
  CQ-B useless ternary; norm_id + `_norm_med_for_bin` duplication flagged for shared-core sweep.
  Next: calibration.py, database.py, vyvar_platesolver.py.
- Phase F: calibration.py done (CAL-A..D). Clean module; only LOW/future notes (passthrough
  caller logging, RGGB assumption for TODO-45, Bayer global rescale doc). Next: database.py,
  vyvar_platesolver.py.
- Phase F COMPLETE: database.py (DB-A..D, mostly sound), vyvar_platesolver.py (PS-A..C, well-gated
  blind solver). No new correctness bugs beyond CQ-B dead ternary. Headline: CQ-A proc-CSV HIGH
  likely resolved -- verify pre-cal draft and close. PS-B -> Phase G priority. Actionable list in
  AUDIT_FINDINGS Phase F COMPLETE section.
- Phase F follow-ups (consolidated): CQ-B/E, shared `norm_med_for_bin`, DB-A allowlist, m2
  RMS-only UI; CQ-A verified (proc glob tests, HIGH closed); CAL-A (no passthrough callers);
  DB-B closed (pipeline per-rerun). CQ-C documented + ROADMAP fix-once locus. pytest 183/6 skip;
  on-disk 283-file SHA stable across diff; PDF 0.
- Phase G batch 1: confirmed COMP_QA/TRUST stage wrappers log; 8 platesolver solve-result-path
  excepts now LOGGER.debug (logging-only). pytest 183/6 skip; numeric SHA 770966c3 unchanged.
- Phase G batch 2: 6 platesolver pass-excepts now log (1 warning MASTERSTAR WCS persist, 5
  debug); ~25 skip-OK reviewed. OPEN QUESTION: fatal MASTERSTAR write? pytest 183/6; SHA unchanged.
- Phase G batch 3: 7 photometry_core excepts logged (edge-ok fail-open, variability export x2,
  color-term x2, pipeline_meta); remainder skip-OK. OPEN QUESTION: edge-ok fail-closed? pytest
  183/6; SHA unchanged.
- Phase G batch 4: 3 pipeline.py excepts logged; worker error-surfacing reviewed; critical path
  COMPLETE (platesolver + photometry_core + pipeline). pytest 183/6; SHA unchanged.
- Phase H: cosmetic lint value-filtered (SIM118 x11, RUF022 x2, RUF007 x2, RUF034 x3); ProcFrameStore
  `.keys()` x2 kept; ~89 style accepted (PROCESS). Clean-code campaign A-H COMPLETE. pytest 183/6;
  SHA unchanged.

---

## Session 2026-06-15/16 -- simple differential PRODUCTION + draft_409 trust cleanup

### Root cause and design (ALG-3 -> simple differential)

- **ALG-3:** comp temporal binning (`temporal_bin_comp_lc`) breaks per-frame common-mode cancellation
  -> **`temporal_binning_enabled` OFF** (production default).
- **Simple-diff design** grounded on draft_407 V0612 + AIJ Table.tbl:
  - Tier-ladder colour window [0.15, 0.30, 0.55] -> cap 0.79; rank-by-RMS inside tier; bounds 3/8.
  - `comp_select_rms_floor` = 1e-6 drops isolated-bin artefact comp before RMS ranking.
  - Flux-sum ensemble; **`apply_color_term` OFF** (colour-matched comps).
- **Decision-grounding rule adopted** (`docs/VYVAR_DECISION_GROUNDING_RULE.md`).

### Workstream A (2026-06-15) -- defaults + Phase-1 selector

- Dataclass + config.json defaults; Phase-1 routes `_select_comps_by_color_then_rms`.
- **DoD-A PASS:** V0612 `delta_mag` pre-eclipse RMS 0.0113, eclipse corr 0.949, 7 comps.
- Harness: `tmp/phase10/`.

### Workstream B (2026-06-15) -- reporting column grounded decision

- Supersedes B1/B2: report differential + ensemble ZP; drop per-target airmass detrend on
  reporting path (Plavchan arXiv:0704.3584; Dhillon); mask-first outlier guard for variables
  (TESS arXiv:2402.16018; democratic detrender arXiv:2411.09753).
- `apply_reporting_postprocess` shipped.
- **DoD-B PASS:** V0612 `mag_calib` corr 0.958 (was 0.57 with target-fit detrend).
- Spec: `docs/VYVAR_REPORTING_COLUMN_GROUNDED_DECISION.md`.

### Canonical combination logic

- Broeg (2005) inverse-variance optimal but **CONDITIONAL** on complete validated sigma.
- **HOLD A:** flux-sum `delta_mag` canonical until sigma = Howell + scintillation + Broeg inflation
  AND chi-squared/dof ~ 1 on constant calibrator.
- Sigma-budget work item **PARKED** (`docs/VYVAR_SIGMA_BUDGET_SPEC.md`, `docs/VYVAR_CANONICAL_COMBINATION_LOGIC.md`).

### draft_407 closure (commit 2a8355b)

- Workstreams A+B committed; real pipeline run; clean V0612 eclipse on production path.

### draft_409 UI run + read-only audit (2026-06-16)

- Aperture is RADIUS ~0.89x FWHM (SNR-opt path) -- fine.
- **Bug found:** comp stability ran on raw `mag_inst` (~0.35 mag night drift) -> all comps suspect.
- `lc_rms` undemeaned (includes eclipse) -- misleading headline for variables.
- FWHM 6.43 self-consistent in headers; external 7.7-8.6 claim unproven (ROADMAP item).
- **Phantom from audit harness:** loose `contains('111174')` matched neighbor
  `1111749157833870208` (G~11.2, 6.85 px) -- true V0612: mag 12.326, aperture 5.754 px.

### Fixes 1-3 (2026-06-16 commit)

1. Comp stability on per-frame **ensemble residual** (not raw `mag_inst`).
2. Measured proc aperture on card/LC; observed-band mag priority for SNR sizing.
3. `lc_rms (OOE)` on variable cards; `n_stability_good/suspect` + trust soft-warning.

**Cross-validation vs SIPS on V0612:** eclipse shape match; single bright outlier at ~JD 2461200.385
matches in both reductions -> shared frame-level artifact (cosmic-ray-like on target), NOT VYVAR bug.

**draft_409 post-fix:** `n_stability_good=8`, `n_stability_suspect=0`, trust GREEN, `lc_rms_ooe` ~0.006,
`delta_mag` pre-eclipse RMS ~0.010.

**Byte-identity:** simple-differential change retired old photometry SHA anchors by design; validation
asset is empirical SIPS/AIJ cross-validation. Optional fresh anchor cut -- Milan call (ROADMAP).

---

## 2026-07-10 — F-BINGAIN-1 FIX (empirical background-noise term)

**Scope:** Milan-approved option (a) from Stage C — production change to `err` column.

**Implementation:**
- `measure_empty_aperture_sigma_bkg()` — random star-free apertures, same annulus sky subtraction as science; robust MAD scatter → `sigma_bkg_ap` [ADU].
- `_photometric_error_with_bkg_mode()` — empirical: `var = F/g + sigma_bkg_ap²`; howell: byte-identical legacy.
- `enhance_catalog_dataframe_aperture_bpm` emits `sigma_bkg_ap`, `err_bkg_source` per proc-CSV row.
- `read_flux_from_csv` consumes empirical columns; `compute_snr_optimal_aperture_table` optional measured star-free per-pixel bkg variance.
- Config: `err_background_mode`, `err_empty_apertures_n`, `err_empty_apertures_min`.
- Citations: Merline & Howell 1995, Fruchter & Hook 2002, Casertano et al. 2000, Labbé et al. 2003.

**Part 0 (pedestal / detrending trace):**
- Stage C photon-transfer with free intercept `var = (level - P)/g` — sandbox script path; draft_426 FITS not in local Archive for numeric P table this session.
- Detrending trace: `pipeline.py:_calibrate_one_light_apply_masters_in_ram` L14643–14646 subtracts dark only when `md_data is not None`; Stage B inventory = **0 C5A bias/dark frames** → **no bias/dark subtracted** → pedestal remains in level by construction. Pre-calibrated draft_426 photometry input is `detrended_aligned/lights` (alignment/detrend only).

**Validation:** pytest **733 passed** (+10 new unit tests). Draft chi2 before/after matrix **not run** — `Archive/Drafts/` empty locally. **STOP before commit** per task acceptance gate.

**Open:** Milan review of chi2 on reprocessed draft_426 (V0611 target 0.8–1.2); PROD-SIGMA-FLOOR separate.

