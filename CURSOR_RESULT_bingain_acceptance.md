CURSOR RESULT ù 2026-07-10 (F-BINGAIN-1 acceptance run)

What I did
Diagnosed the prior empty-Archive finding (wrong path segment), hardened
`scripts/bingain_fix_validate.py` archive-root resolution, updated validation to use
production LC-`err` chi2 (sigma-budget path ignores empirical `sigma_bkg_ap`), ran
patch-only acceptance on drafts 424/425/426 (no DAO re-export), chi2 before/after
matrices, provenance stats, pedestal fits, pytest, and PDF overflow on draft_425 B.

**Verdict: FAIL ù do not commit/push.** Primary gate miss: V0611 chi2 on draft_426 i/r
remains ~0.25 (outside 0.8ù1.2). Secondary: draft_425 B_20_2 howell_fallback 24.7%
(>20% flag); draft_424 pooled chi2 shifted 0.074?0.216 (425 unchanged).

## Part 0 ù path diagnosis

The FIX session reported an empty `Archive/Drafts/` because diagnostics used
`C:\ASTRO\Archive\Drafts` (missing the `python\VYVAR` repo segment). Production
code always resolved the correct root via `AppConfig.archive_root` ?
`C:\ASTRO\python\VYVAR\Archive` from `config.json`. Eight draft folders are present
(including `draft_000424`, `draft_000425`, `draft_000426`). Fix applied:
`resolve_archive_root()` in `scripts/bingain_fix_validate.py` with explicit
`--archive-root` and hard exit when `Drafts/` is missing or contains no `draft_*`
folders.

## Output / findings

### Acceptance gates

| Gate | Result | Notes |
|------|--------|-------|
| draft_426 V0611 chi2 ? [0.8, 1.2] (LC err) | **FAIL** | g **PASS** (1.23?1.11); i/r **FAIL** (~0.25, unchanged) |
| Wide-rig 424/425 chi2 unchanged | **MIXED** | 425 identical; 424 pooled 0.074?0.216 |
| Wide-rig err ratio ~1 | **PASS** | read_flux median IQR ~1.0 on 424/425 |
| Non-err proc byte-identical | **PASS** | patch-only; g/424/425 verified; i/r/z stale-backup artifact |
| pytest green | **PASS** | 733 passed, 15 skipped |
| PDF overflow 0 (draft_425 B) | **PASS** | 0 violations, 390 pages |
| err_bkg fallback ?20% all setups | **FLAG** | B_20_2 24.7% howell_fallback |

### Chi2 matrix (production LC `err` vs check_kmag; primary gate metric)

**draft_426 ù V0611** (`1112127291051695744`)

| Setup | ?ù/dof before | ?ù/dof after | Gate |
|-------|---------------|--------------|------|
| g_60_4 | 1.233 | **1.110** | PASS |
| i_70_4 | 0.238 | 0.251 | FAIL (low) |
| r_60_4 | 0.253 | 0.256 | FAIL (low) |
| z_90_4 | ù | ù | no V0611 sidecar |

**draft_426 ù pooled check stars (n=6/setup)**

| Setup | pooled before | pooled after |
|-------|---------------|--------------|
| g_60_4 | 2.952 | 2.989 |
| i_70_4 | 0.584 | 0.598 |
| r_60_4 | 0.476 | 0.544 |

**draft_425 ù pooled (n=40/setup, wide rig)**

| Setup | pooled before | pooled after |
|-------|---------------|--------------|
| B_20_2 | 0.569 | 0.569 |
| V_20_2 | 5.521 | 5.521 |
| R_20_2 | 0.504 | 0.504 |

**draft_424 ù NoFilter_60_2 pooled (n=40)**

| before | after |
|--------|-------|
| 0.074 | 0.216 |

(Sigma-budget chi2 variant remains ~0.04 for V0611 g ù ignores empirical proc columns;
documented harness mismatch, not used for gate.)

### err_bkg_source provenance (% empirical / howell_fallback)

| Draft / setup | rows | empirical | fallback | flag |
|---------------|------|-----------|----------|------|
| 426 g_60_4 | 10840 | 100% | 0% | |
| 426 i_70_4 | 7589 | 100% | 0% | |
| 426 r_60_4 | 18264 | 100% | 0% | |
| 426 z_90_4 | 5240 | 100% | 0% | |
| 425 B_20_2 | 87284 | 75.3% | **24.7%** | **>20%** |
| 425 V_20_2 | 33807 | 100% | 0% | |
| 425 R_20_2 | 115551 | 92.0% | 8.0% | |
| 424 NoFilter_60_2 | 445658 | 100% | 0% | |

Mask/dilation: `err_empty_apertures_n=64` (clamp 16..256), `err_empty_apertures_min=16`;
exclusion radius `r_out + margin_px`; edge margin `r_out + r_ap + 1`.

### Wide-rig read_flux err ratio (empirical / howell, median + IQR)

| Setup | median | p25 | p75 |
|-------|--------|-----|-----|
| 424 NoFilter_60_2 | 1.011 | 0.995 | 1.017 |
| 425 B_20_2 | 1.000 | 0.9997 | 1.0001 |
| 425 V_20_2 | 1.002 | 1.0003 | 1.0058 |
| 425 R_20_2 | 0.999 | 0.999 | 1.000 |

LC-level err ratio (ensemble-dominated): median 1.0 on all wide-rig setups.

### Runtime overhead (patch-only path, seconds)

| Setup | patch | phase2a emp | phase2a howell |
|-------|-------|-------------|----------------|
| 426 g_60_4 | 120 | 13 | 12 |
| 426 i_70_4 | 135 | 13 | 14 |
| 426 r_60_4 | 134 | 12 | 12 |
| 426 z_90_4 | 147 | 4 | 4 |
| 425 B_20_2 | 130 | 503 | 502 |
| 425 V_20_2 | 96 | 117 | 117 |
| 425 R_20_2 | 159 | 491 | 494 |
| 424 NoFilter_60_2 | **688** | 346 | 420 |

Empty-aperture patch cost scales ~5 s/frame; phase2a unchanged vs baseline.

### Part 0 pedestal P fit (draft_426, gain=12.48 from FITS GAIN)

| Setup | P [ADU] | 95% CI lo | 95% CI hi | OFFSET/PEDESTAL header |
|-------|---------|-----------|-----------|------------------------|
| g_60_4 | 232 | ?1966 | 2431 | none |
| i_70_4 | 967 | 692 | 1242 | none |
| r_60_4 | 120 | ?81 | 320 | none |
| z_90_4 | 769 | 428 | 1110 | none |

Photon-transfer patch fit is noisy (wide CIs on g); no OFFSET/PEDESTAL FITS keys on
science frames ù pedestal remains in level per Stage B/C.

### Residual-budget analysis (FAIL drivers)

1. **V0611 i/r (?ù?0.25):** Pre-fix underdispersion; empirical bkg term barely moves LC
   `err` because ensemble/scint/floor dominate at small apertures (r_ap 0.25/0.22 px).
   Fix targets background-dominated g (r_ap 0.54); g moved 1.23?1.11 into gate.
2. **B_20_2 fallback 24.7%:** Crowded wide-rig B field ù ~25% of rows lack valid empty
   apertures; those rows retain Howell fallback (byte-identical legacy err).
3. **424 pooled ?ù shift:** Low absolute ?ù; photon-level err ratio still ~1; LC err
   ensemble-dominated but small photon perturbation visible in pooled check stars.

## Errors (if any)

- Phase2A tmp outputs: COMP_QA non-fatal (missing comparison_stars in tmp tree).
- draft_424 phase2a: 2 export failures (missing LC for 2 Gaia targets).

## Files changed (uncommitted ù FAIL stop)

- `scripts/bingain_fix_validate.py` ù archive root hard-fail; LC-err chi2 primary
- `scripts/bingain_acceptance_run.py` ù `--patch-only`, FITS gain, read_flux err ratio
- `scripts/bingain_patch_sigma_bkg.py` ù (from FIX session)
- `photometry_core.py`, `pipeline.py`, `config.py`, `proc_frame_store.py`, tests, docs
  ù (from FIX session, uncommitted)

Reports: `tmp/bingain_acceptance/run_{424,425,426}.json`,
`tmp/bingain_fix/validation_{424,425,426}.json`
