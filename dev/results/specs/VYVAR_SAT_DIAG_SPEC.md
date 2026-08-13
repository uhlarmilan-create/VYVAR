# VYVAR -- SAT-DIAG: Saturation and linearity limit gate (spec)

Status: **DRAFT -- architect recommendations recorded 2026-08-13; awaiting Milan authorization.**
Date: 2026-08-13 (decisions section added 2026-08-13).
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
| **1 -- Hard saturation** | Pixel at container ceiling; `sat_source` in {MEASURED, HEADER, DERIVED} | **Yes** -- comp pool, AC ref set, PSF ref set. Target: flag epoch, keep export row. |
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

## 5. Check A -- ceiling derivation from raw frames

### 5.1 Inputs

- All **raw light** FITS in the draft obs_group (or a deterministic subsample if
  N > `sat_diag_max_frames`, default 30, evenly spaced in sorted filename order).
- Primary HDU data, linear ADU (respect `BZERO`/`BSCALE` if present).

### 5.2 Histogram property

Identify a **hard ceiling** value `V_ceiling` when:

1. Value `V_max` has pixel count `N(V_max) >= N_pileup_min`, AND
2. `N(V_max) >= k * N(V_max - 1)` where `k >= pileup_ratio` (default **10**), AND
3. `V_max` is within `BITPIX` range (16-bit unsigned: 65535).

Default thresholds (implementation constants, not user config):

| Constant | Default | Role |
|----------|---------|------|
| `N_pileup_min` | 100 | Minimum pixels at ceiling across sampled frames |
| `pileup_ratio` | 10 | Sharpness of pile-up vs shoulder |

Draft 510 reference: **13024** pixels at **65535**, **1** at 65532 -- `V_ceiling=65535`.

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

## 6. Check B -- limit resolution and precedence

Resolution order (section 3 Decision 1):

1. Header keywords via existing resolver (`param_resolver.py` saturation
   aliases: `SATURATE`, `MAXLIN`, `ESATUR`, `LINLIMIT`, `MAXADU`, `DATAMAX`) --
   if present **and compatible**.
2. Equipment DB row keyed by `(equipment_id, readmode, xbin, ybin)` -- hint only;
   if present **and compatible**.
3. Derived from Check A (pile-up level).
4. Container ceiling from `BITPIX`/`BZERO` when no pile-up (section 5.4).

### 6.1 Compatibility test

For stated ceiling `S`: **compatible** iff `S >= max_pixel_value` across sampled
raw frames. Otherwise refuted -- do not use.

### 6.2 CONFLICT handling

Refuted stated value replaced by derived; `CONFLICT_DERIVED`; ERROR infolog;
**continue run**. Fail closed only when nothing stated, nothing derived, no
BITPIX bound.

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

### 8.2 How

For each star with known `(ra, dec)` or master-grid position:

1. WCS to pixel on raw frame.
2. Per-frame drift correction from bright reference (same family as
   `CURSOR_RESULT_saturation_peak_reconcile.md` pre-push v3).
3. Mag-guided local peak in search window (`half=22`, tighter for faint stars).
4. Report `peak_max_adu_raw` = max in 7x7 box (`pipeline._box_peak_max_adu`,
   `half=3`).

### 8.3 Self-check gate (memo recommendation 6)

Per star per frame, before accepting `peak_max_adu_raw`:

| Test | Default threshold |
|------|-------------------|
| Local maximum | Centre equals 3x3 max |
| Ring contrast | `peak / median(ring 11-15 px) >= 1.8` |
| Minimum signal | `peak >= 4000` ADU |

Outcomes:

- **PASS** -- record peak and position.
- **FAIL** -- `peak_max_adu_raw = NaN`; `peak_loc_fail=true`; increment failure count.
  **Must not** return a background value silently.

Report aggregate: `peak_loc_fail_count` per star in `sat_diag.json` and
per-frame proc column `peak_loc_ok`.

### 8.4 Replaces aligned-frame peaks

Current behaviour (`pipeline.py:8050+`) computes `peak_max_adu` on the aligned
float array. SAT-DIAG supersedes this for saturation/linearity decisions:

- New canonical columns: `peak_max_adu_raw`, `likely_nonlinear_raw`,
  `likely_saturated_raw`.
- Aligned-frame peaks may remain for DAO diagnostics but **must not** drive
  saturation flags once SAT-DIAG is active.

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
| **1** | Hard saturation; sat from MEASURED/HEADER/DERIVED | **Yes** (draft-level rule) |
| **2** | `lin_source=MEASURED` | **Yes** pool + AC |
| **3** | `lin_source=DEFAULT_FRAC` or DERIVED linearity | **Warn only** |

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

---

*Discipline reminders: shared helper for all raw-light paths; byte-identity of
photometry when flags additive-only; English task text; JOURNAL entry at close.*
