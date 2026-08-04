CURSOR RESULT - 2026-07-31 AUDIT STAGE 3 PART 0e

What I did
Per-frame identity forensics on the Part 0d delta tail. Tested neighbour contamination (E1),
traced aperture-position provenance (E2), compared WCS (E3), scoped position shifts (E4),
and assigned mechanism (E5). Read-only; no code changes.

## Provenance

| Field | Value |
|-------|-------|
| `git_hash` | `086d45a6fbac1c30a71765ef77cbd53e816f7e4b` |
| `git_dirty` | `true` (scratch harness under `tmp/` only) |
| Harness | `tmp/audit_stage3_part0e_cohort_shift.py` (read-only scan) |

---

# Summary

**Part 0d ensemble attribution is wrong for the focus target.** On frame 063,
`1498135552633294976` keeps the **same `catalog_id`** in both runs; there is **no brighter
catalogue neighbour** within 6 px of either aperture position. The flux ratio (3.41x ?
?1.36 mag) matches the calibrated delta; comp-catalog mean shift (?0.45 mag) does not.

The mechanism is **per-frame DAO detection centroid shift** on the same sky-matched star:
aperture centres differ by **3.48 px** (r ? 1.9 px), so the two runs sample barely
overlapping flux on the same PSF. **`catalog_match_mode = master_reference_sky`** in both runs.

**E1 neighbour test: negative.** Not M1 as 'wrong Gaia ID / brighter neighbour.'

**WCS differs slightly (~0.4 px on frame 063)** but explains only ~12% of the 3.5 px
position split; the rest is **DAO centroid offset from WCS** changing between runs.

**Verdict: M4** (refined) - same catalogue identity, **mis-centred aperture from unstable
DAO match centroid** upstream of photometry; optionally **M3** at sub-aperture WCS level for
the global ~0.4 px component.

---

# E1 - What is at the rebuild position?

Frames examined: **063** (worst), **085**, **014** for target `1498135552633294976`.

## Frame 063

| | Anchor proc | Rebuild proc |
|---|------------:|-------------:|
| `catalog_id` | 1498135552633294976 | 1498135552633294976 |
| x, y | 282.18, 618.15 | 281.98, 621.63 |
| `dao_flux` | 134.4 | 458.1 |
| `peak_max_adu` | 1666 | 1709 |
| `aperture_r_px` | 1.916 | 1.901 |
| `catalog_match_mode` | master_reference_sky | master_reference_sky |

**Sources within 6 px of anchor position (282.18, 618.15):** only `1498135552633294976` (both CSVs).

**Sources within 6 px of rebuild position (281.98, 621.63):** only `1498135552633294976` (both CSVs).

**Union within 6 px of either position:** **1 source** - the target itself. No second Gaia ID.
Nearest masterstar neighbour on reference grid: `1498136342907277184` at **~11 px**, G = 14.45
(only **0.24 mag** fainter than target G = 14.20) - **outside** the 6 px search disc and not
present in per-frame proc rows near either aperture.

**Brighter neighbour at rebuild position?** **No.**

## Frames 085 and 014

| Frame | Position shift (an?rb) | Flux ratio | Neighbours within 6 px |
|-------|----------------------:|-----------:|------------------------|
| 085 | **2.02 px** | 2.0x (?0.75 mag) | target only |
| 014 | **0.74 px** | 1.13x (?0.12 mag) | target only |

Large flux deltas track **large position shifts**, not catalogue identity swaps.

---

# E2 - Where does (x, y) come from?

**Path:** per-frame **DAO detection centroid**, then **sky match** to master catalogue - **not**
catalogue RA/Dec ? WCS ? pixel for the exported proc `x,y`.

| Step | Location | What happens |
|------|----------|--------------|
| 1 | `pipeline.py` ~7525-7534 | `DAOStarFinder` on aligned frame ? detection table |
| 2 | ~7637-7638 | `x`, `y` = DAO centroids (full-pixel grid) |
| 3 | ~7699-7734 | If WCS OK: `match_to_catalog_sky(master_coords)` with threshold `match_sep_arcsec` (default **8 arcsec**); match by **sky position**, identity = master `catalog_id` |
| 4 | ~7878-7879 | Output proc row: **`x`, `y` = DAO centroid**; `catalog_id` from matched master row |
| 5 | `photometry_core.py` ~2417-2420 | Phase 2A reads proc CSV **`x`, `y` unchanged** for aperture photometry |

**No centroid refinement or recentre in `photometry_core.py`** (confirmed grep). Aperture is
placed at whatever DAO centroid the pipeline locked to the catalogue row for that frame.

---

# E3 - Is the WCS the same?

Frame 063, target RA/Dec = 212.232608 deg, 41.382321 deg:

| | Anchor | Rebuild |
|---|--------|---------|
| FITS used | `proc_BO_CVn_Light_063.fits` | `BO_CVn_Light_063.fits` |
| CRVAL1, CRVAL2 | 209.608785, 41.185795 | 209.581902, 41.186708 |
| CRPIX1, CRPIX2 | 1009.296, 690.667 | 1016.688, 690.295 |
| PC matrix | ~?0.002714 px/? (both) | ~?0.002714 px/? |
| VY_ALGN | True | True |
| **WCS RA/Dec ? pixel** | **281.36, 620.43** | **281.19, 620.80** |
| **Proc x, y** | **282.18, 618.15** | **281.98, 621.63** |
| **Proc ? WCS offset** | **(+0.82, ?2.27) px** | **(+0.78, +0.83) px** |

| Offset type | ?x | ?y | |?| |
|-------------|---:|---:|--:|
| WCS projection alone (rebuild ? anchor) | ?0.16 | +0.37 | **0.41 px** |
| Proc aperture centres (rebuild ? anchor) | ?0.20 | **+3.48** | **3.48 px** |

**WCS shift is real but ~0.4 px - not ~3.5 px.** The dominant term is **change in DAO?WCS
offset** between runs (especially ?y ? 3.1 px). Same catalogue row, different detection
centroid locked by sky match.

---

# E4 - Scope

## Cohort (156 common targets, 139 common proc frames)

Per-target maximum proc `(x,y)` shift on shared `source_file` rows:

| Metric | Value |
|--------|------:|
| Median max shift | **0.55 px** |
| p95 max shift | **2.57 px** |
| Targets with **any** frame shift **> aperture_r** | **19 / 156** (12%) |
| Targets with max shift **> 2 px** | **18 / 156** |

## Five worst targets by valid (source_file) \|?mag\| p95 (Part 0d)

| target_cid | \|?\| p95 (valid) | max proc shift | frames shift > r_ap |
|------------|-----------------:|---------------:|--------------------:|
| 1498322916287022976 | 1.98 | 1.25 px | 0 / 137 |
| 1485540612577549568 | 1.85 | 0.58 px | 0 / 139 |
| **1498135552633294976** | **1.52** | **3.48 px** | **2 / 127** |
| 1498453414573142016 | 1.50 | 2.72 px | 3 / 133 |
| 1498341092588681856 | 1.41 | 1.56 px | 0 / 123 |

**Two mechanisms in the tail:**

1. **Ensemble-only** (148554..., 149832..., 149834...): sub-aperture position shifts (? 1.6 px),
   **zero** frames with shift > r_ap; large \|?mag\| from comp-set change (Part 0d valid pairing).

2. **DAO mis-centroid** (149813..., partly 149845...): shifts **> 2 px** on multiple frames,
   flux responds at fixed `catalog_id` - matches user's peak/flux/aperture argument.

Focus target is the **extreme DAO-centroid case**, not representative of all five.

---

# E5 - Verdict

| Code | Applicable? | Evidence |
|------|-------------|----------|
| **M1** neighbour identity | **No** | Same `catalog_id`; no second source within 6 px; no brighter neighbour at rebuild position |
| **M2** astrometric (WCS) | **Partial** | CRVAL/CRPIX differ; ~**0.4 px** WCS projection shift on frame 063 |
| **M3** both | **Partial** | WCS sub-aperture + large DAO centroid change combine on worst frames |
| **M4** something else | **Yes (primary)** | **Same-ID DAO centroid instability**: sky match assigns catalogue row, but exported `x,y` follow per-run DAO peak; when centroid jumps **> 2r_ap**, enclosed flux changes dramatically with similar peak |

**Recommended label:** **M4 - per-frame aperture placement error on correct catalogue identity**
(DAO centroid path in `detect_stars_match_master_reference`, not ensemble zero-point).

---

# Contradictions with Part 0d

| Part 0d claim | Part 0e finding |
|---------------|-----------------|
| 'Ensemble recomposition ?1.38 mag mean offset' as primary cause | **Wrong for focus target:** flux ratio explains offset; comp mean ? = ?0.45 mag |
| 'Differential pipeline amplifies ensemble swap' | **Not supported** - no such amplification in code; ensemble shifts zero point ~ comp ? |
| 3.36 mag max delta | Still **invalid positional pairing**; valid max **2.76 mag**, driven by mis-centroid + flux on frame 063 |
| Bright-target tail 'not faint-end artefact' | Partially true under valid pairing, but **split mechanism**: ensemble (most) vs DAO centroid (149813...) |

---

# Implications

1. **Do not attribute focus-target flux delta to ensemble change** - fix centroid/aperture
   placement upstream (DAO match on `master_reference_sky` path).
2. **19 targets** hit at least one frame with shift > aperture radius - bounded but non-zero
   identity/placement risk class.
3. Anchor re-cut blocked on: (a) Part 0c pairing fix, (b) DAO centroid stability or
   catalogue-position aperture option for photometry.

**STOP GATE 0e** - awaiting Milan review. No fixes applied.
