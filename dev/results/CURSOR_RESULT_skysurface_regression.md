> **PROVENANCE WARNING (added 2026-07-30).** Quantitative rows in this document for **draft_000449**, **draft_000450**, and other unavailable drafts from 2026-07-21 onward were produced after the in-place preprocess architecture landed (`013cb0c`, 2026-07-22) or in the same session window, and before the sky-surface idempotency guard (`84174ae`, 2026-07-30). During that window a repeated preprocess pass could subtract the sky surface twice, at a measured cost of order 500 ADU. Those drafts are no longer available, so their status is UNKNOWN, not clean. Rows sourced only from pre-013cb0c **draft_000435** family on disk remain anchor-validated separately. Treat post-era numbers as indicative, not validated.

# CURSOR RESULT - SKY-SURFACE-REGRESSION (2026-07-26)

Investigation + impact assessment. No code changes, no commits. Independent of PHASE0-IDENTITY-GATE
close arc.

Scratch: `tmp/dao_verify_bisect/`, prior results in `CURSOR_RESULT_masterstar_count_diag.md`,
`CURSOR_RESULT_dao_only_verify.md`, `CURSOR_RESULT_sigma_estimator_verify.md`.

---

## K1 - Exact behavioural delta

### Confirmed reading

**Yes:** order-2 sky-surface subtraction was a photometric step on the shared
`calibrated -> processed` path. SKIPPROC (`013cb0c`, 2026-07-22) removed the `processed/` copy
tree and replaced it with in-place QC. The sky-surface call lived inside the deleted
`_preprocess_calibrated_one` helper; it did **not** move to the new in-place path for mono frames.

**Not deliberate science removal.** `docs/VYVAR_DECISIONS.md` SKIPPROC-PERMANENT (2026-07-22) states
the goal was retiring `skip_processed_directory` and the copy tree as a "check artifact"; it does
**not** authorise dropping sky-surface subtract. T3-PREPROCESS-SKY-SURFACE (2026-07-16) explicitly
mandates order-2 surface subtract in shared preprocess for **both** UI and headless chains.

Quote from T3 decision:

> Per calibrated light frame, fit a 2D polynomial surface of order 2 to the background
> (source-masked + sigma-clipped fit), subtract it in the shared ``calibrated -> processed`` step
> used by BOTH chains

Quote from SKIPPROC-PERMANENT:

> ``skip_processed_directory`` removed; in-place QC + ``qc_metrics.csv`` allowlist is the only
> preprocess path. The ``processed/lights`` copy tree is retired (it existed only as a check
> artifact; allowlist supersedes).

Housekeeping language only; no mention of removing sky-surface from mono.

### Before `013cb0c`

| Item | Behaviour |
|------|-----------|
| Function | `_preprocess_calibrated_one` (deleted in `013cb0c`) |
| Sky step | `_fit_subtract_preprocess_sky_surface(data, order=preprocess_sky_surface_order)` when order > 0 |
| Output | `processed/lights/.../proc_*.fits` with `VYSKYORD` / `VYSKYP2P` headers |
| Mono | **Applied** (no `VY_CHANNEL` gate on the old copy path) |
| Config gate | `preprocess_sky_surface_order` (default **2**, 0 = off) |

### After `013cb0c` (current HEAD)

| Item | Behaviour |
|------|-----------|
| Function | `_qc_enrich_calibrated_in_place` (`pipeline.py` ~16894) |
| Sky step | Only when `is_channel and sky_order > 0 and not is_mosaic` |
| `is_channel` | `bool(hdr.get("VY_CHANNEL")) or apply_sky_surface` |
| Mono BO CVn | **`VY_CHANNEL` absent -> sky-surface NOT applied** |
| Output | In-place header QC on calibrated FITS; alignment reads calibrated or detrended_aligned directly |

`263c6e7` temporarily restored the old `proc_` copy + sky subtract in bisect replay; current HEAD
(`cb78b25`) matches skip-only again. OSC commits (`0f1c07f`, `224c442`) do **not** change mono
preprocess pixels (bisect confirmed).

---

## K2 - Blast radius

### Drafts on disk (signature: DAO_ONLY fraction)

| Draft | mtime era | CSV rows | DAO_ONLY | DAO_ONLY % | Affected? |
|-------|-----------|----------|----------|------------|-----------|
| draft_000435 | 2026-07-16 | 2951 | 109 | **3.7%** | **No** (predates 013cb0c) |
| draft_000435_snapshot_skysurface_20260716 | 2026-07-16 | 2951 | 109 | 3.7% | No |
| draft_000435_p1mini | 2026-07-16 | 2951 | 109 | 3.7% | No |
| draft_000438 | 2026-07-21 | 29210 | 20706 | 70.9% | Yes (post-dark regen) |
| draft_000439 | 2026-07-21 | 26338 | 17816 | 67.6% | Yes |
| draft_000441 | 2026-07-21 | 893 | 15 | 1.7% | Likely low (small field / depth) |
| draft_000444 | 2026-07-21 | 34485 | 25875 | 75.0% | Yes |
| draft_000448 | 2026-07-22 | 10943 | 5210 | 47.6% | Yes |
| draft_000449 | 2026-07-22 | 10943 | 5210 | 47.6% | Yes |
| draft_000450 | 2026-07-25 | 6698 | 2705 | **40.4%** | Yes |

Drafts without `masterstars_full_match.csv` on disk: draft_000440, draft_000442, draft_000443,
draft_000445 (not scored).

### Preview bundle `preview-20260723`

| Check | Result |
|-------|--------|
| Tag | `preview-20260723` @ `fe574c0` |
| `013cb0c` ancestor of tag? | **Yes** (`git merge-base --is-ancestor` exit 0) |
| Bundle includes SKIPPROC code? | **Yes** |

**The released preview build carries the regression.** It affects every user of that bundle on
every rig, not only Milan's local Windows runs.

### FI Boo field run (Linux, preview-20260723)

`draft_000001` is **not on this machine** (`Archive/Drafts/draft_000001` absent). JOURNAL records
FI Boo as draft_000001 from the Linux preview run.

**Milan needs to send:** `masterstars_full_match.csv` (or full `platesolve/.../masterstars_full_match.csv`
path) plus draft date/commit if known. Signature to check: `DAO_ONLY` row fraction >> 5%.

### Anchor snapshot draft_435

Built **2026-07-16** (`10d610c0` era), **before** `013cb0c` (2026-07-22). Frozen `MASTERSTAR.fits`,
frozen `masterstars_full_match.csv`, frozen aligned lights -- **unaffected by sky-surface regression.**
PHASE0-IDENTITY-GATE re-cut consumes frozen plan files; it does not rebuild detection.

---

## K3 - Science impact

### Detection catalogue vs target photometry

| Area | Impact | Evidence |
|------|--------|----------|
| MASTERSTAR / DAO catalogue | **High** | 40% DAO_ONLY on 450; ~2235 spurious at G<16 |
| Target aperture photometry | **Partially protected** | Per-target annulus sky subtraction removes local pedestal; smooth gradient over aperture scale (~10-30 px) is partially cancelled |
| Protection breakdown | **Incomplete at large scales** | Order-2 gradient spans hundreds of px; annulus sky at r~20 px does not remove field-wide tilt. Residual bias possible for targets near gradient extrema |
| Comparison-star pool | **Indirect** | Larger, noisier master catalogue may admit poor comps if selection keys off masterstar rows |
| Ensemble / trust flags | **Low direct** | Trust flags keyed on per-frame photometry QC, not DAO_ONLY fraction; unless comp stars picked from spurious detections |
| Wide vs Newton vs C9.25 | **Scales with field** | Wide fields have larger sky gradients -> larger p2p surface (T3 logged ~96 ADU on BO CVn); wide rig **more exposed** |
| AAVSO/VarAstro exports | **Unknown on disk** | No export manifest found in this pass. Drafts 438-450 with high DAO_ONLY % are **at risk** if science was exported from post-0722 masterstars without manual filtering |

**Honest classification:** primarily a **catalogue-hygiene / detection problem** for anchor-style
workflows; **potential photometric bias** at large spatial scales for wide-field targets if gradient
remains in detrended_aligned frames. Not automatically a "all LC wrong" event because per-frame
aperture sky subtract provides partial shielding.

---

## K4 - Process defect

### Why nothing caught it for four days

| Guard | Why it missed |
|-------|---------------|
| Unit tests | `test_preprocess_sky_surface.py` copy-mode tests **removed** in `013cb0c`; helper-only tests remain |
| `--fast` | No invariant on detrended-frame large-scale variance or DAO_ONLY fraction |
| Anchor `--full` | Uses **frozen** MASTERSTAR + CSV + aligned lights; never rebuilds preprocess or detection |
| `--full` on 2026-07-22 | Ran on draft_435 (pre-regression inputs) -> **PASS masked the regression** |
| Invariants | QC-01 gates alignment allowlist, not sky-surface or detection purity |

### Cheapest durable guard (proposal only)

**Preferred:** post-preprocess metric on one QC frame per obs_group:

```
large_small_ratio = var(gaussian_blur(frame, sigma=30)) / var(frame - gaussian_blur(frame, sigma=30))
```

Fail-closed WARN if ratio exceeds bound calibrated on draft_435 (anchor: ~1-5x on good frames;
regression shows 20-60x). **Measurable on output**, not a call-count check.

**Secondary:** masterstar export invariant:

```
DAO_ONLY_fraction <= 0.10   (WARN above; FAIL above 0.25 on anchor-like nights)
```

Justification: anchor night stable at 3.7%; 40% is a 10x anomaly impossible from depth alone
(DAO-ONLY VERIFY). Cheap to compute from CSV already written.

Weak guard (not preferred): assert `_fit_subtract_preprocess_sky_surface` called -- does not catch
wrong order, wrong mask, or silent skip.

---

## Fix proposal (described, not implemented)

### Order of work

1. **Restore sky-surface subtract** on the skip-only path for **mono and OSC** frames:
   - Call `_fit_subtract_preprocess_sky_surface` inside `_qc_enrich_calibrated_in_place` when
     `preprocess_sky_surface_order > 0`, **without** requiring `VY_CHANNEL` (restore T3 behaviour).
   - Stamp `VYSKYORD` / `VYSKYP2P` on calibrated FITS as before.
   - OSC: confirm Bayer mosaic path still correct (do not double-subtract channels).

2. **Sigma estimator review** (depends on SIGMA-ESTIMATOR-VERIFY S1):
   - If sky-surface restore alone returns `bg_std ~ 83` and pass-1 count ~ 2552, estimator may suffice.
   - If `bg_std` stays low while `sigma_pp` unchanged, **separate fix**: consider `mad_std` on
     median-subtracted data, annulus noise, or `sigma_pp`-style local differencing for DAO threshold.
   - **Do not fold estimator fix into sky-surface PR unless S1 post-restore still fails.**

3. **Guard (K4):** add `large_small_ratio` WARN to preprocess QC row + optional ledger stamp;
   add `DAO_ONLY_fraction` WARN to masterstar export diagnostics.

4. **Docs:** DECISIONS (SKY-SURFACE restore supersedes accidental SKIPPROC drop), INVARIANTS,
   STATE NOT-guaranteed entry, PARAMS (no change to `preprocess_sky_surface_order` default 2).

5. **Validation:**
   - Rebuild draft_450 inputs from identical calibrated lights -> expect DAO_ONLY **< 10%**,
     pass-1 raw DAO **~ 2550-2600** (not 8926), threshold ADU **~ 175** at unchanged sigma_pp.
   - Bisect replay: mono frame 050 mean abs diff vs 8815c45 proc output **< 1 ADU**.
   - `--fast` + new unit test: mono frame gets `VYSKYORD=2` after preprocess.

### Dependencies on SIGMA-ESTIMATOR-VERIFY

| Outcome | Action |
|---------|--------|
| S1: sigma_pp unchanged, bg_std low ( **confirmed** ) | Sky-surface restore is **necessary**; estimator fix may be **also necessary** |
| If post-restore bg_std ~ 83 | Sky-surface alone may suffice |
| If post-restore bg_std still ~ 62 | **Second defect** -- estimator fix required as separate item |

---

## Files changed

None (read-only).
