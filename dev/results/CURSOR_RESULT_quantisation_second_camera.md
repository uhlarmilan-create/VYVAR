CURSOR RESULT - 2026-08-13

What I did
Part A: mod-4 quantisation test on QHY draft-510 raw. Part B: TOI-1131.01.b C3
inventory and SAT-DIAG dry-run. Part C: four spec fixes (+ quantisation/C3 notes).
Part D: commit + `--fast`.

---

## Part A - QHY mod-4 grid (draft 510 raw, 150 frames)

### Measurements (before interpretation)

| Metric | Value |
|--------|------:|
| Total pixels | 436 283 100 |
| `value mod 4 != 0` | **13 024** |
| Fraction off-grid | **0.00298%** |
| Unique off-grid values | **1** (65535 only) |
| Off-grid excluding 65535 | **0** / 436 270 076 (**0%**) |

Near-ceiling histogram (accumulated):

| ADU | Count | mod 4 |
|-----|------:|------:|
| 65535 | 13024 | 3 |
| 65532 | 1 | 0 |
| 65524 | 1 | 0 |
| 65520 | 1 | 0 |
| ... (65516..65496) | 1--3 each | 0 |

Background mode **5184** is divisible by 4.

### Interpretation

The step-4 grid **holds for every pixel except the digital clip at 65535**. The
near-ceiling shoulder values in the memo are on-grid singletons; the observation
was not an artefact of looking at the top ten only.

**Reading:** Consistent with 14-bit native samples left-shifted by 2 (2x2 binning)
into a 16-bit container -- native quantisation step **4 stored ADU**. **65535 is
A-D clip**, not a native quantised full-well level (`65535 mod 4 = 3`).

**Follow-ons (no changes made):**

- Linearity ramp rows should state stored vs native (=stored/4) ADU.
- `GAIN=0.0` in header is uninformative; DB `GAIN_ADU=3.17` is in stored ADU/e-.
- Quantisation step 4 is small vs typical sky noise at BO CVn (~2400 ADU); unlikely
  to dominate the noise model unless counting in native units.

---

## Part B - TOI-1131.01.b (C3-26000)

### B.1 Inventory

| Item | Value |
|------|-------|
| Path | `Archive/TOI-1131.01.b/` |
| Structure | **78** FITS files, flat (one file per subdirectory name) |
| Raw vs calibrated | **No separate raw tree**; Milan: calibrated with dark+flat |
| `BITPIX` | **16** (all 78) |
| `BZERO` / `BSCALE` | **32768** / **1** |
| `XBINNING` / `YBINNING` | **2** / **2** |
| `GAIN` (header) | **~3.12** e-/ADU (not 0) |
| `SATURATE` / `MAXLIN` | absent |
| `DATAMAX` | **65535** |
| Data type | `uint16`; max frame range **43429--65535** |

### B.2 / B.4 SAT-DIAG dry-run

| Item | Result |
|------|--------|
| Pile-up | **No** (51 px at 65535 total; n@max below `N_pileup_min`) |
| Derived ceiling | none |
| Header | **DATAMAX=65535** wins |
| Equipment (C3 id=2) | **65535** compatible |
| Provenance | **HEADER** |
| Field saturates? | **Essentially no** (129 px >= 60000 / 509 Mpx) |

This is the **no-pile-up branch on real data**.

### B.3 Refusal branch

Frames are **integer BITPIX=16**, not float. SAT-DIAG does **not** refuse (would
derive if pile-up existed; it does not). **Gap identified:** calibrated integer
without raw lights is not caught by float refusal (section 5.5 now warns
`UNVERIFIED_INPUT`). Flat-divided float remains the hard refusal case.

### B.5 Grid control (C3)

| Metric | Value |
|--------|------:|
| Off-grid (all pixels) | **75.0%** |
| At 65535 | 51 pixels |

**Grid of 4 does not hold on C3** -- contrasts with QHY. Low ADU values (~700)
occupy all residue classes.

### B.6 Equipment row

| Field | C3-26000 (id=2) |
|-------|-----------------|
| `SATURATE_ADU` | **65535** |
| Compatible? | **Yes** (26/78 frames reach 65535; refutation would not fire) |

---

## Part C - Spec fixes

Applied in `dev/results/specs/VYVAR_SAT_DIAG_SPEC.md`:

- **C.1** Shoulder rule: nearest lower occupied bin; digital-clip special case
- **C.2** Section 4.1 image ADU convention (single authority)
- **C.3** `CONFLICT_DERIVED` never plain `DERIVED` when refuted
- **C.4** Section 6.3 `POSSIBLE_RESCALED_STACK` (concrete thresholds)
- **C.5** Tier 1 includes `CONFLICT_DERIVED`; warn-only for linearity only
- **Part A/B data:** sections 4.2, 4.3, 5.5 input refusal

---

## Part D - Commit and DB reproducibility

### D.2 `SATURATE_ADU = NULL` reproduction

| Mechanism | Present? |
|-----------|----------|
| Journal entry | **Yes** (`docs/VYVAR_JOURNAL.md`) |
| SQL migration / seed script | **No** |
| `config.json` | **No** |
| Git-tracked DB | **No** (`vyvar.sqlite3` local) |

**Same class of problem as pre-checksum anchor:** another machine or DB rebuild
will restore `SATURATE_ADU=16384` unless Milan NULLs it again or a migration is
written. Recommend: one-line seed in equipment setup docs or SQL migration when
SAT-DIAG lands.

---

## Part E - Closing

| Question | Answer |
|----------|--------|
| Spec ready to implement? | **Yes**, with TOI `UNVERIFIED_INPUT` gap understood |
| Second camera showed | No pile-up; header DATAMAX; no QHY quantisation; minimal saturation |
| Next | Milan authorize SAT-DIAG; implement gate; draft 510 photometry on raw peaks |
