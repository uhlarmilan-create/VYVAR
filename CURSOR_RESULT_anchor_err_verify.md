CURSOR RESULT -- 2026-07-13 (ANCHOR-ERR-VERIFY)

What I did
Diagnosed the draft_424 re-anchor err rise (~1.5-1.7x) on snapshot
``draft_000424_snapshot_sigma_floor_20260713`` vs old anchor
``draft_000424_snapshot_20260708_full``. Checked floor leak, per-tertile
decomposition, n_comps distribution, and a fully worked (star, frame) err budget.
Verdict: **Newton floor leak NOT confirmed**; err shift explained by stacked err-model
changes (F-BINGAIN-1 photon term + SEM unit fix), not c4 alone and not sigma_sys on eq1.
**Part C STOP** -- no code fix, no re-cut, no push. Anchor remains **NOT ACCEPTED**.

## Diagnosis (Part A)

### A.1 sigma_sys_mag / pipeline floor

| Source | Distinct values | Expected (eq1 un-floored) |
|--------|-----------------|---------------------------|
| LC column ``sigma_sys_mag`` (178 LCs, 23542 epochs) | **0.0 only** | 0.0 |
| ``pipeline_meta.dynamic_params.sigma_floor`` | equipment_id=**1**, sigma_sys_mag=**0.0**, c4_correction=true | 0.0 |

**Floor leak via config keying: NOT CONFIRMED.**

Runtime lookup: ``resolve_sigma_sys_mag(1, cfg) -> 0.0``;
``resolve_sigma_sys_mag(4, cfg) -> 0.018``.

### A.2 Per-tertile err decomposition (mag_calib tertiles, pooled 178 LCs)

| Tertile | mag range | n epochs | err_old med | err_new med | ratio | sigma_add (mag) | sigma_add pt med |
|---------|-----------|----------|-------------|-------------|-------|-----------------|------------------|
| faint | 8.16 - 12.69 | 7848 | 0.015019 | 0.022629 | 1.507 | 0.016926 | 0.016330 |
| mid | 12.69 - 13.72 | 7847 | 0.037641 | 0.063497 | 1.687 | 0.051137 | 0.050435 |
| bright | 13.72 - 21.37 | 7847 | 0.085756 | 0.143924 | 1.678 | 0.115586 | 0.114177 |

Overall epoch ratio: median **1.622**, p25 **1.432**, p75 **1.838**; 3081 epochs in [1.45, 1.55].

**Floor-leak signature (constant sigma_add ~18 mmag across tertiles): REJECTED.**
sigma_add scales with base err (ensemble-dominated bright tertiles grow more in quadrature).

Faint tertile sigma_add ~17 mmag is coincidental with 18 mmag floor magnitude but is
**not** consistent with a constant additive floor (mid/bright sigma_add 51-116 mmag).

### A.3 n_comps distribution (draft_424 wide)

Per-frame (star 1499294845909337344, 139 epochs): **n=8** all frames.

Per-frame histogram across all 178 LCs (23542 epochs):
``{3: 2116, 4: 1226, 5: 552, 6: 798, 7: 4252, 8: 14598}``.

Per-target median n_comps histogram (178 LCs):
``{3: 16, 4: 9, 5: 4, 6: 7, 7: 32, 8: 110}``.

Photon term ratio (bingain/howell) for target 1499294845909337344 across epochs:
median **1.634**, p25 **1.475**, p75 **1.806**.

Per-target pool (``comp_quality_*.json`` tier counts, 178 targets):

| n_good (tier1+tier2) | targets |
|----------------------|---------|
| 8 | 127 |
| 7 | 2 |
| 6 | 2 |
| 5 | 5 |
| 4 | 7 |
| 3 | 7 |
| 2 | 4 |
| 1 | 5 |
| 0 | 19 |

Majority population n=8 -> c4 inflation ceiling on SEM only ~**8%** (not 50-70% on total err).

### Git provenance (confound)

| Commit | Description | Ancestor of old anchor ``750c856``? |
|--------|-------------|-------------------------------------|
| ``750c856`` | Old snapshot git (2026-07-08) | -- |
| ``3b33b03`` | F-BINGAIN-1 empirical bkg in err | **NO** |
| ``26396ab`` | SEM mag->rel unit fix at err combine | **NO** |
| ``8fb21b3`` | New snapshot git (PROD-SIGMA-FLOOR) | -- |

Old vs new anchor compares **three** err-model commits (bingain + unit fix + c4/floor wiring),
not c4/floor alone. Attributing the full 1.5-1.7x to "c4 on wide; designed" is incorrect.

## STOP evidence (Part C) -- worked frame

Star ``1499294845909337344``, frame ``proc_BO_CVn_Light_001.csv`` (faint check star).

| Quantity | Value |
|----------|-------|
| err_old (2026-07-08 anchor) | 0.008833 rel |
| err_new (sigma_floor snapshot) | 0.013634 rel |
| ratio | 1.544 |
| sigma_sys_mag LC column | 0.0 |
| floor at runtime (eq1) | 0.0 |
| floor if eq4 had leaked | 0.018 mag -> err would be **0.0215** (not observed) |
| n_comps (frame) | 8 |
| photon Howell-only (old model) | 0.008047 rel |
| photon bingain empirical (new) | 0.013183 rel |
| photon ratio bingain/howell | **1.638** |
| sigma_bkg_ap | 240.95 ADU |
| err_bkg_source | empirical |
| ensemble SEM (c4, mag) | 0.003776 |
| ensemble SEM (c4, rel) | 0.003478 |
| model: Howell + SEM-as-rel bug (old code) | 0.008889 ~ err_old |
| model: bingain + c4 + unit fix, floor=0 | **0.013634** = err_new |

Artifact: ``tmp/reanchor_424/worked_frame.json``.

**Mechanism:** F-BINGAIN-1 empirical background term raises photon err ~1.63x on this
frame; quadrature with correctly converted ensemble SEM yields observed err_new. No Newton
floor required.

## Root cause + fix (Part B)

**NOT TAKEN** (floor leak not confirmed). No code changes, no re-cut, snapshot not renamed.

## Comparator hardening (Part D)

**NOT TAKEN** (Part C STOP: no code modifications this task).

Note for follow-up: ``compare_photometry_science_meaningful`` excludes ``err`` from science
failures (QC column). Re-anchor gate passed science columns only; err divergence was not
bounded. Recommend bounded err envelope check in a future task when a valid re-anchor
baseline exists.

## Re-anchor SHAs

Contested snapshot (unchanged, **NOT ACCEPTED**):

- Path: ``Archive/Drafts/draft_000424_snapshot_sigma_floor_20260713``
- core SHA: ``bf3743a150d788283eab2ab51db7b31f59e6d1c481159208bbe3f573092ec975``
- extended SHA: ``dec5c637724e0ca536e97a01194ab8cc06df9471ce4813fcfd26024b9e880fd1``
- git: ``8fb21b3``

Old anchor (untouched): ``draft_000424_snapshot_20260708_full`` (git ``750c856``).

## Recommendation (Milan / Claude)

Before accepting any draft_424 re-anchor:

1. Cut an **intermediate baseline** at ``origin/main`` (``b5364e6``) + unit fix + bingain
   **without** c4/floor, then compare PROD-SIGMA-FLOOR delta only (expected err ratio
   envelope ~[0.96, 1.05] on wide).
2. Do **not** push or accept anchor on old-vs-new err ratio alone until baseline chain is
   explicit.

## Errors

None.

## Files changed

- ``CURSOR_RESULT_anchor_err_verify.md`` (this file)
- ``docs/VYVAR_SIGMA_FLOOR_SPEC.md`` (wide 4.8 vs 6.5 mmag paragraph)
- ``docs/VYVAR_STATE.md`` (ANCHOR-ERR-VERIFY verdict, anchor HOLD)
- ``docs/VYVAR_ROADMAP.md`` (ANCHOR-ERR-VERIFY entry)
- ``docs/VYVAR_JOURNAL.md`` (ANCHOR-ERR-VERIFY entry)
- ``tmp/reanchor_424/worked_frame.json`` (diagnostic artifact)

No production code changed. pytest / ruff: **not run** (no code touched).
