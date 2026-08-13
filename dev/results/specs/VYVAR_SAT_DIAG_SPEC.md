# VYVAR -- SAT-DIAG: Saturation and linearity limit gate (spec)

Status: **AUTHORIZED -- Milan 2026-08-13; implementation in progress.**
Date: 2026-08-13 (decisions section added 2026-08-13; Part 0 limits 2026-08-13).
Grounding: `dev/results/MEMO_saturation_limit_literature.md`,
`dev/results/CURSOR_RESULT_saturation_peak_reconcile.md`,
`dev/results/specs/VYVAR_CAL_DIAG_SPEC.md` (structure model).

**Governing principle (Milan, 2026-07-07, extended to saturation):** a user may
build masters and acquire lights in any binning and any readout mode; VYVAR must
verify from the data and adapt or continue, never assume a camera convention.
VYVAR must not need to know what the camera firmware does.

**Policy principle (architect, 2026-08-13):** the strength of the action must
follow the provenance of the number. A measured limit may exclude data. A derived
or defaulted limit may only warn. A user who fills in nothing gets warnings; a
user who measures their rig gets full protection. Nobody silently loses good
comparison stars to a number nobody verified.

---

## 1. Purpose

VYVAR today stores one equipment scalar (`EQUIPMENTS.SATURATE_ADU`), applies a
hardcoded 0.85 linearity proxy, measures star peaks on **aligned float frames**
(`pipeline.py:8050+`), and couples limit crossings directly to comp-pool
exclusion, PSF skip, and aperture-correction reference removal.

Measured on draft 510: raw ceiling **65535**; aligned max **~69000**; DB limit
**16384** (wrong units). Two BO CVn peak measurements on the same raw data
disagreed by factor **4.8** in the median because one used fixed master-grid
coordinates without WCS.

SAT-DIAG is a **camera-agnostic, data-verified** gate that:

1. Derives or resolves saturation and linearity limits in **image ADU** for the
   active `(equipment, readout mode, XBINNING, YBINNING)` configuration.
2. Measures per-star peaks on **raw** frames with a self-check gate.
3. Emits **flags** and **provenance**; downstream **policy** decides actions.

This gate must survive configuration-parameter reduction: it is enforced as an
**invariant** (`INV-SAT-01`), not as optional config toggles alone.

## 2. Scope / non-goals

**In scope:**

- Ceiling derivation from raw-frame histograms (Check A).
- Limit resolution order: header, then equipment table, then derived (Check B).
- Two levels: **saturation** and **linearity** (distinct).
- Raw peak measurement + self-check per star per frame.
- Provenance flag `sat_limit_source` and per-star flags carried forward.
- Separation of flag from consumer policy (documented defaults; individually
  overridable in a future policy layer).

**Out of scope (this spec):**

- Implementation code, UI design, or PARAMS registry entries (follow-on task).
- Replacing BPM / cosmic-ray masks.
- PSF model changes for saturated cores.
- Multi-night limit inheritance across drafts with different binning.
- Automatic exposure-ramp acquisition (telescope procedure; Milan only).
- CAL-DIAG reinstatement (separate decision; see section 13).

## 3. Architect decisions (2026-08-13 -- recommendations, not yet authorized)

Four open decisions from the draft spec are answered below. **Milan must
authorize before implementation.**

### Decision 1 -- Interim limit source and CONFLICT policy

Every stated ceiling must pass a **compatibility test** before use:

> A stated ceiling that lies **below** the maximum pixel value present in the
> raw frames is refuted by those frames. Pixels exist above it. It cannot be
> the ceiling.

Example: 16384 stated, 13024 pixels at 65535 measured.

**Resolution order:**

1. Header keyword (`SATURATE`, `MAXLIN`, `DATAMAX`, etc.) -- if present and compatible.
2. Equipment row for `(equipment, readmode, XBIN, YBIN)` -- if present and compatible.
3. Derived from raw frames (observed pile-up level).
4. Container ceiling from `BITPIX`/`BZERO` -- upper bound when no pile-up exists.

**CONFLICT policy: adapt and continue, loudly.** A refuted stated value is
replaced by the derived value; WARNING names both numbers; provenance =
`CONFLICT_DERIVED`. Follows CAL-DIAG philosophy (2026-07-07): verify from data,
adapt or continue, never assume convention, never continue silently.

**Fail closed only when:** nothing can be derived AND nothing is stated AND
`BITPIX` gives no bound (should not occur with real FITS).

**No-pile-up case is not an error.** No pile-up means container bound +
`DERIVED_NO_PILEUP` flag. **Must never** derive a ceiling from the brightest
unsaturated star in the field.

### Decision 2 -- Target structure

**Two levels** keyed by `(equipment, readmode, XBINNING, YBINNING)`. DB row is a
**hint, never an authority** without passing the compatibility test.

| Column | Meaning |
|--------|---------|
| `sat_adu` | Saturation ceiling, image ADU |
| `lin_adu` | Linearity level, image ADU |
| `lin_source` | `MEASURED` / `DERIVED` / `DEFAULT_FRAC` |
| `sat_source` | `MEASURED` / `HEADER` / `DERIVED` |
| `measured_utc` | Ramp measurement date; null if not measured |
| `tolerance_pct` | Departure-from-linearity tolerance used; null if not measured |

Missing row is normal: derive, warn, record `DERIVED`. **Migration must not
carry forward `EQUIPMENTS.SATURATE_ADU = 16384`.** Set to null; let SAT-DIAG
derive until a ramp measurement replaces it.

### Decision 3 -- Exposure ramp

**Proceed now with `DEFAULT_FRAC` and mandatory WARN.** Ramp when convenient.

- Default: `lin_adu = linearity_default_frac * sat_adu` (frac **0.85**).
- Provenance: `lin_source=DEFAULT_FRAC` -- never `MEASURED`.
- **Decision 4 forbids DEFAULT_FRAC from excluding anything.**

**Ramp procedure (when Milan runs it):** AAVSO method -- mean pixel value in a
central box vs exposure time on a uniformly illuminated field; fit line to
low-signal portion; linearity level = departure point beyond stated tolerance.
Store tolerance and date. **Per rig configuration** (each binning + readout
mode used).

### Decision 4 -- Consumer policies

**Three tiers keyed on provenance.** Every exclusion decided **once per draft**
(see `INV-COMP-MEMBERSHIP` -- no per-frame membership changes).

| Tier | Condition | May exclude? |
|------|-----------|--------------|
| **1 -- Hard saturation** | Pixel at container ceiling; `sat_source` in {MEASURED, HEADER, DERIVED, **CONFLICT_DERIVED**} | **Yes** -- comp pool, AC ref set, PSF ref set. Target: flag epoch, keep export row. |
| **2 -- Linearity, MEASURED** | `lin_source=MEASURED`; star crosses `lin_adu` | **Yes** -- comp pool, AC ref set. |
| **3 -- Linearity, DEFAULT_FRAC or DERIVED** | Unmeasured linearity knee | **Warn only** -- flag + trust panel; exclude nothing. |

Draft-level exclusion rule example: exclude a comp if it crosses the ceiling on
more than N% of frames (N stated in policy; not per-frame flip).

## 4. Design constraints (must survive)

The gate MUST remain correct when:

| Failure mode | Required behaviour |
|--------------|-------------------|
| `EQUIPMENTS.SATURATE_ADU` in wrong units | Compatibility test refutes; replace with DERIVED; `CONFLICT_DERIVED` |
| Equipment row absent / draft link missing | Derive, warn, record `DERIVED` -- normal, not failure |
| No `SATURATE` / `MAXLIN` in header | DERIVED from histogram, or BITPIX bound |
| `READMODE` uninformative; `GAIN=0` | Ignore; use binning keys + derivation |
| Same camera, different binning later | Lookup keyed by `(equipment, readmode, xbin, ybin)` |
| User fills equipment incorrectly | Compatibility test refutes; adapt loudly |
| No pile-up in field | Container bound only; never brightest-star ceiling |

## 4.1 Image ADU convention (single authority)

**Image ADU** everywhere in this spec means the per-pixel value used for
saturation histograms, limit comparison, and peak measurement on the **primary
HDU array** as loaded for science:

| FITS layout | Image ADU |
|-------------|-----------|
| `BITPIX=16`, `BZERO=32768`, `BSCALE=1` (unsigned 16-bit, QHY/C3 class) | Stored array value in **0..65535** (do **not** add `BZERO` again) |
| `BITPIX=16`, `BZERO=0`, `BSCALE=1` (native signed/unsigned 16-bit) | Stored array value |
| `BITPIX < 0` (float) | **Not valid input** for Check A or raw peak (see section 5.5) |

All consumers (Check A, compatibility test, peak self-check, equipment table
`sat_adu` / `lin_adu`) use **image ADU** in this sense. Converting to
"electrons" or pre-scaling by binning is out of scope unless a future equipment
column defines it explicitly.

### 4.2 QHY294MM quantisation note (draft 510, measured 2026-08-13)

On raw BO CVn frames (150 frames, 436.3 Mpx):

| Measurement | Result |
|-------------|--------|
| Pixels with `value mod 4 != 0` | **13 024** (**0.0030%** of total) |
| Unique off-grid values | **1** -- value **65535** only |
| Off-grid excluding 65535 | **0** / 436 270 076 (**0%**) |
| Near-ceiling values 65532..65496 | All **mod 4 = 0** (1--3 px each) |

**Reading:** Stored ADU below the digital clip lie on a **step-4 grid**, consistent
with 14-bit native samples left-shifted by 2 into a 16-bit container (2x2 binning).
**65535 is a clip**, not a native quantised level (`65535 mod 4 = 3`). Linearity
and ramp measurements should record whether they use **stored image ADU** or
**native ADU (= stored / 4)**; defaults in this spec remain in **stored image
ADU** unless a measured ramp row states otherwise.

**GAIN:** QHY headers record `GAIN=0.0`; equipment DB holds `GAIN_ADU=3.17` e-/ADU
(stored). SAT-DIAG does not resolve gain; noise-model work must not assume header
`GAIN` is usable.

### 4.3 C3-26000 control (TOI-1131.01.b, measured 2026-08-13)

78 frames, `BITPIX=16`, `BZERO=32768`, 2x2 binning, **calibrated** integer (Milan:
master dark + flat applied). Grid test on stored values:

| Measurement | Result |
|-------------|--------|
| Pixels with `value mod 4 != 0` | **381 830 229** (**75.0%**) |
| Pixels at 65535 | **51** total (26/78 frames touch max) |
| Off-grid excluding 65535 | **75.0%** (same -- grid does **not** hold) |

C3 data do **not** show the QHY step-4 grid; sky levels (~700 ADU) occupy all
residue classes. Quantisation structure is camera/path-specific, not universal.
SAT-DIAG on this set: **no pile-up** (below `N_pileup_min`); `DATAMAX=65535`
header wins; field essentially **does not saturate** (129 pixels >= 60000).

## 5. Check A -- ceiling derivation from raw frames

### 5.1 Inputs

- All **raw light** FITS in the draft obs_group (or a deterministic subsample if
  N > `sat_diag_max_frames`, default 30, evenly spaced in sorted filename order).
- Primary HDU data in **image ADU** (section 4.1).

### 5.2 Histogram property

Identify a **hard ceiling** value `V_ceiling` when:

1. Value `V_max` has pixel count `N(V_max) >= N_pileup_min`, AND
2. **Shoulder test:** let `V_lo` be the **highest occupied bin strictly below
   `V_max`**. Require `N(V_max) >= k * N(V_lo)` where `k >= pileup_ratio`
   (default **10**). **Special case:** if no bin is occupied at `V_max - 1`
   (common at digital clip -- e.g. QHY has **0** pixels at 65534), use `V_lo`
   as the nearest lower occupied bin. If **no** lower occupied bin exists but
   `V_max` equals the BITPIX container ceiling and `N(V_max) >= N_pileup_min`,
   treat as pile-up (digital clip at container). AND
3. `V_max` is within `BITPIX` range (16-bit unsigned: 65535).

Default thresholds (implementation constants, not user config):

| Constant | Default | Role |
|----------|---------|------|
| `N_pileup_min` | 100 | Minimum pixels at ceiling across sampled frames |
| `pileup_ratio` | 10 | Sharpness of pile-up vs shoulder |

Draft 510 reference: **13024** pixels at **65535**, **1** at 65532 -- `V_ceiling=65535`.

**65535 versus 65532 (observed, unexplained):** under a clean two-bit left-shift,
native maximum 16383 maps to **65532**, and pile-up would sit there. Measured
pile-up sits at **65535** with a single pixel at 65532. Something beyond the
shift maps the saturated value. This does not change behaviour (pixels clip at
65535 either way) but must not be written up as understood.

### 5.3 Saturation level

When pile-up detected:

`saturation_adu = V_ceiling` (image ADU).

Optional conservative margin (DERIVED mode only):

`saturation_adu_effective = saturation_adu * (1 - sat_margin_frac)` with default
`sat_margin_frac = 0.0` at derivation time; linearity margin applied separately
(section 7).

### 5.4 No pile-up case (nothing in field saturates)

When no value satisfies section 5.2 across the sample:

- `saturation_adu = BITPIX ceiling` (65535 for 16-bit unsigned) as **upper bound only**.
- `sat_limit_source` includes `DERIVED_NO_PILEUP`.
- Emit **WARN**: "No saturation pile-up detected; ceiling set to BITPIX maximum."
- **Do not** infer a lower limit from noise or background.
- Run continues; linearity level still resolved separately (section 7).

This case is valid and must not fabricate a knee. **Must not** set ceiling from
the brightest unsaturated star.

### 5.5 Input refusal (non-raw frames)

Check A and raw peak measurement **must not run** on:

| Input | Action |
|-------|--------|
| `BITPIX < 0` (float / calibrated science array) | **Refuse** -- no derived ceiling; `sat_limit_source=REFUSE_NON_RAW`; WARN |
| Master dark, flat, aligned, detrended products | **Refuse** (same) |
| Integer (`BITPIX > 0`) files that are not raw lights from the draft ingest path | **WARN `UNVERIFIED_INPUT`** -- derivation may be meaningless (see TOI control, section 14 validation notes) |

A wrong derived ceiling from flat-divided or float data is worse than no ceiling.

## 6. Check B -- limit resolution and precedence

Resolution order (section 3 Decision 1):

1. Header keywords via existing resolver (`param_resolver.py` saturation
   aliases: `SATURATE`, `MAXLIN`, `ESATUR`, `LINLIMIT`, `MAXADU`, `DATAMAX`) --
   if present **and compatible**.
2. Equipment DB row keyed by `(equipment_id, readmode, xbin, ybin)` -- hint only;
   if present **and compatible**.
3. Derived from Check A (pile-up level).
4. Container ceiling from `BITPIX`/`BZERO` when no pile-up (section 5.4).

### 6.1 Compatibility test (one-sided -- known limit)

For stated ceiling `S`: **compatible** iff `S >= max_pixel_value` across sampled
raw frames. Otherwise refuted -- do not use.

**Known limit (cannot be closed in software):** this test refutes a stated
ceiling that lies **below** the data (pixels exceed it). It **cannot** refute a
ceiling that is **too high**, because the data never reach it. Where pile-up
exists, Check A derivation covers the low-ceiling case. Where **no pile-up**
exists, SAT-DIAG accepts the stated value **untested** for the too-high direction.
TOI-1131 is exactly that state: 51 pixels at 65535, below the pile-up threshold,
so `DATAMAX=65535` won unchallenged. It happens to be correct there, but nothing
verified it.

This matters because the linearity level is a fraction of the ceiling. A ceiling
that is too high makes the linearity level too high, and non-linear stars pass.
PSFEx documents "too high" as the common failure mode in practice; the test as
built catches the rarer error (stated limit below data). **Only an exposure ramp
can close the too-high gap.**

Provenance must let a reader distinguish an **unverified accepted** stated value
(`HEADER`, `EQUIPMENT`, `DERIVED_NO_PILEUP`) from a **measured** derived ceiling
(`DERIVED`, `CONFLICT_DERIVED`). Do not record refuted values as plain `DERIVED`.

### 6.2 CONFLICT handling

Refuted stated value replaced by derived; provenance **`CONFLICT_DERIVED`** (never
plain `DERIVED` when a header or equipment value was refuted); ERROR infolog;
**continue run**. Fail closed only when nothing stated, nothing derived, no
BITPIX bound.

### 6.3 Rescaled or stacked-frame warning (`POSSIBLE_RESCALED_STACK`)

Following STDWeb's practice of comparing header saturation to the observed data
range ([arxiv:2411.16470](https://arxiv.org/abs/2411.16470)):

Emit **WARN `POSSIBLE_RESCALED_STACK`** when **all** of:

1. `bitpix_ceiling` is known;
2. `max_pixel` across the sample `< bitpix_ceiling * rescaled_max_frac` (default **0.85**);
3. A stated ceiling `S` (header or equipment, before refutation) exists with
   `S >= max_pixel` (passes compatibility) **and** `S < bitpix_ceiling * 0.95`;
4. Check A finds **no** pile-up shoulder at `bitpix_ceiling`.

**Action:** do **not** use `S` for science exclusion; fall through to DERIVED
(if pile-up exists elsewhere) or `DERIVED_NO_PILEUP` / BITPIX bound. Record
`POSSIBLE_RESCALED_STACK` in `sat_diag.json`.

**Catches:** master-flat division, re-scaling, or stacked products where the
header still advertises a low linearity limit but pixels no longer reach the
container. **Does not catch:** genuine low-well cameras that never hit the
container; fields with no bright stars (max naturally low) -- pair with pile-up
test and provenance.

## 7. Two levels -- saturation and linearity

| Level | Meaning | Source |
|-------|---------|--------|
| **Saturation** | Hard ceiling / A-D full scale | section 6 resolution |
| **Linearity** | Knee below linear response | Ramp (MEASURED) or DEFAULT_FRAC |

### 7.1 Equipment table schema

See section 3 Decision 2. Migration: null out `SATURATE_ADU=16384`; do not carry
forward.

### 7.2 Linearity from exposure ramp (measured)

When equipment table holds a ramp measurement:

- Fields: `lin_adu`, `measured_utc`, `tolerance_pct`, `lin_source=MEASURED`.
- Provenance: `linearity_source=MEASURED`.

Only Milan can populate this via telescope procedure (AAVSO ramp; section 3
Decision 3). SAT-DIAG **reads** it; it does not perform the ramp. Repeat per
binning + readout mode.

### 7.3 Linearity default (unmeasured)

When no ramp measurement exists:

- `lin_adu = linearity_default_frac * sat_adu` (default frac **0.85**).
- `lin_source=DEFAULT_FRAC`.
- **Mandatory WARN** in infolog and `sat_diag.json`: "Linearity level is a default
  fraction, not a measured knee."

An unmeasured default must never appear as `MEASURED` in provenance. **Tier 3
policy (section 9): warn only, exclude nothing.**

## 8. Peak measurement stage (raw only)

### 8.1 When

Once per raw light frame, **before calibration**, at the same cadence as DAO
detection inputs (per obs_group, per frame).

### 8.2 How (2026-08-13 anchored search)

For each star with known sky position and aligned-frame centroid `(x, y)`:

1. Map aligned pixel to raw pixel via shared sky position.
2. Per-frame drift from bright reference only (`mag_guided`, `half=22` on ref).
3. **WCS residual anchor:** expected raw = raw WCS(ra,dec) plus aligned DAO
   offset (aligned xy minus aligned WCS xy). Search within 12 px disk for a
   pixel passing self-check and brightness plausibility vs aligned peak.
4. Report `peak_max_adu_raw` = max in 7x7 box at verified pixel (`half=3`).

### 8.3 Verification gate (pass = right star)

All must pass for `peak_loc_ok=true` and `sat_peak_source=RAW_VERIFIED`:

| Test | Default threshold |
|------|-------------------|
| Self-check (8.3 legacy) | Local max, ring contrast >= 1.8, peak >= 4000 ADU |
| Anchor distance | Found pixel within **12 px** of aligned-mapped raw position |
| Brightness plausibility | `raw/aligned` in **[1/3, 3]** when aligned < 85% sat |

When aligned peak >= 85% sat (resampling can exceed raw ceiling), plausibility
ratio is skipped; anchor distance still required.

When verification fails: `peak_max_adu_raw` may be NaN; saturation uses
**aligned** peak (`sat_peak_source=ALIGNED_INTERIM`).

### 8.4 Peak source for saturation (provenance)

| `sat_peak_source` | Meaning |
|-------------------|---------|
| `RAW_VERIFIED` | Anchored raw search passed all gates |
| `ALIGNED_INTERIM` | Raw not verified; **aligned** peak drives saturation |
| `MIXED` | Draft uses both (aggregate in `sat_diag.json`) |

Draft `sat_diag.json` records `sat_peak_source`. Proc CSV column `sat_peak_source`
is per star per frame. FITS header `VY_SATPS` mirrors draft-level source.

**ALIGNED_INTERIM caveat:** Aligned peaks pass through resampling and can exceed
the raw container ceiling (draft 510 aligned ~69000 vs raw 65535). Interim mode
is **conservative on admission** (may admit comps raw would reject) and **cannot**
flag raw-only saturation. Document until raw search is fully trusted.

Columns: `peak_max_adu` (authoritative for gates), `peak_max_adu_raw`,
`peak_max_adu_aligned`, `likely_saturated_raw`, etc.

### 8.5 Existing drafts

Drafts calibrated before SAT-DIAG lack raw peak columns. On re-run:

- Re-measure raw peaks if raw files present.
- If raw archived but not re-processed: `sat_limit_source=LEGACY_ALIGNED` WARN;
  do not claim raw provenance.

## 9. Flags versus actions (Decision 4)

Crossing produces **flags**. **Actions** depend on tier and provenance (section 3
Decision 4).

| Flag | Condition |
|------|-----------|
| `likely_nonlinear` | `peak_max_adu_raw >= lin_adu` |
| `likely_saturated` | `peak_max_adu_raw >= sat_adu * 0.85` |
| `is_saturated` | `peak_max_adu_raw >= sat_adu` |

### 9.1 Three-tier consumer policy

| Tier | Source | May exclude pool/AC/PSF? |
|------|--------|--------------------------|
| **1** | Hard saturation; sat from MEASURED/HEADER/DERIVED/**CONFLICT_DERIVED** | **Yes** (draft-level rule) |
| **2** | `lin_source=MEASURED` | **Yes** pool + AC |
| **3** | `lin_source=DEFAULT_FRAC` or DERIVED **linearity** | **Warn only** |

**Tier 1 rationale:** A derived pile-up ceiling is a **physical measurement** of
the data (stars hitting the container), not an unverified scalar. When equipment
or header is refuted (`CONFLICT_DERIVED`), the adopted limit is still that
measurement -- exclusion is permitted. **Warn-only (Tier 3) applies to unmeasured
linearity knees only**, not to container saturation.

Target: always flag epoch; keep LC export row.

### 9.2 Draft-level membership (INV-COMP-MEMBERSHIP)

Exclusion decided **once per draft** (e.g. exceeds ceiling on >50% of frames).
Per-frame flip-flop forbidden.

### 9.3 Supersedes current behaviour

Current code excludes regardless of provenance. Tier 3 must not silently exclude.

## 10. Provenance

### 10.1 FITS headers

| Header | Values | Meaning |
|--------|--------|---------|
| `VY_SATSRC` | `HEADER` / `EQUIPMENT` / `DERIVED` / `DERIVED_NO_PILEUP` / `CONFLICT_DERIVED` / `LEGACY_ALIGNED` | Limit source |
| `VY_SATADU` | float | Saturation level used (image ADU) |
| `VY_LINADU` | float | Linearity level used |
| `VY_LINSRC` | `MEASURED` / `DEFAULT_FRAC` / `DERIVED` | Linearity provenance |
| `VY_SATBF` | int | Block factor (binning) key used |

Modelled on CAL-DIAG `VY_DKRSMP` / `VY_CDSKY` / `VY_CDSTAT`.

### 10.2 Sidecar

`archive/<draft>/sat_diag.json` -- per obs_group:

- derived ceiling, resolved limits, source, conflict details
- frame sample count, pile-up pixel count
- peak self-check failure summary

### 10.3 Proc CSV / pipeline_meta

- Per-frame: `saturate_limit_adu`, `linearity_limit_adu`, `sat_limit_source`,
  `peak_max_adu_raw`, `peak_loc_ok`, flags.
- Phase 2A merges additive `sat_diag` block into `pipeline_meta.json` (same
  pattern as CAL-DIAG `cal_diag` block).

## 11. Migration, anchors, and regression gates

### 11.1 Drafts 435, 509, 510

| Draft | Expected SAT-DIAG run |
|-------|----------------------|
| **435** | Raw present; limits DERIVED ~65535; BO CVn not saturated; comp pool wider than at 16384 |
| **509** | Same; photometry already run with wrong 16384 limit -- re-flag only on re-run |
| **510** | No photometry yet; SAT-DIAG should run before production |

Anchor `--full` does **not** cover raw peak measurement (INV-ANCHOR-00). A
**SAT-DIAG unit test suite** + draft-510 raw histogram fixture is the regression
gate for this class.

### 11.2 Invariants

**INV-SAT-01** (planned):

- Saturation limits in **image ADU** on **raw** peaks with self-check PASS.
- Aligned-frame peaks must not be sole saturation authority when raw exists.
- `sat_limit_source` recorded when gate runs.
- Tier 3 (`DEFAULT_FRAC`) must not trigger exclusion.

**INV-GATE-REMOVAL** (recommended process invariant -- section 13):

A verification gate may not be removed on byte-identity evidence alone.

### 11.3 Why this survives config reduction

CAL-DIAG was removed in `967f835` as five config parameters. SAT-DIAG is defined
as an **invariant-backed gate** with implementation constants, not as
`sat_diag_gate_enabled` alone. Removal requires an explicit DECISIONS entry and
invariant deprecation, not silent parameter deletion.

## 12. Config keys (future implementation)

Derivation thresholds and compatibility tests are **constants in code**, tested
not toggled. No `sat_diag_gate_enabled` switch -- gate is invariant-backed.

Draft-level exclusion threshold (e.g. 50% of frames) may become one visible policy
parameter after Milan authorizes Tier 1/2 defaults.

## 13. Separate finding -- CAL-DIAG removal and INV-GATE-REMOVAL

CAL-DIAG was removed deliberately in `967f835`. Acceptance criterion was **P1 core
SHA byte-identical**. A passing gate produces identical outputs, so byte-identity
**cannot detect removal of a passing gate** (same limitation as `INV-ANCHOR-00`).

Evidence: draft **435** carries `VY_DKRSMP=SUM`, `VY_CDSKY`, `cal_diag.json`;
drafts **509/510** carry none. Anchor was built with radiometry verification;
current drafts are not.

`INV-FLUX-01` verifies resample arithmetic, not radiometric convention (SUM vs
MEAN dark content). **Nothing today** verifies that a master dark resampled to
different binning does not over-subtract.

**Recommended process invariant (`INV-GATE-REMOVAL`):**

> A verification gate may not be removed or disabled on the evidence of
> byte-identity alone. Removing a gate requires either a demonstration that the
> condition it checked is now impossible, or an explicit recorded decision to
> accept the unverified condition, with the risk stated.

**Separate decision for Milan (not bundled with SAT-DIAG):** whether to reinstate
the CAL-DIAG dark-resample convention check. Open hole; must not stay unrecorded.

## 14. Validation (Definition of Done -- implementation task)

1. Unit tests: pile-up detection, no-pile-up case, conflict resolution, self-check
   rejects fixed-position background (BO CVn 3662 case).
2. draft-510 raw: `VY_SATSRC=DERIVED`, ceiling 65535, BO CVn 0/134 saturated.
3. Peak self-check failure count reported; no silent background peaks.
4. `pytest` green; invariant INV-SAT-01 wired.
5. **Controls:** QHY mod-4 grid (section 4.2); C3 TOI no-pile-up + header DATAMAX
   (section 4.3); float input refusal (section 5.5); `POSSIBLE_RESCALED_STACK`
   synthetic case.

---

*Discipline reminders: shared helper for all raw-light paths; byte-identity of
photometry when flags additive-only; English task text; JOURNAL entry at close.*
