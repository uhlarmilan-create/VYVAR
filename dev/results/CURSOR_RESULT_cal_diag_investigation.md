CURSOR RESULT - 2026-08-13 CAL-DIAG investigation (establish, recover, propose)

What I did
Read-only investigation per CAL-DIAG task: traced live resample path, surveyed Archive drafts,
recovered deleted implementation from `967f835^`, reviewed INV-FLUX-01, researched external
pipelines, and drafted a rebuild proposal aligned with SAT-DIAG. **No code changes.**

---

## Part A - Can the condition still arise?

### A.1 When does a master dark or flat get resampled?

**Library convention.** CalibrationLibrary masters are treated as **native bin 1x1** regardless of
FITS header (`CALIBRATION_LIBRARY_NATIVE_BINNING = 1`, `calibration.py:23`). At calibrate time,
`get_processed_master()` (`calibration.py:505-610`) loads the master and calls
`resample_master_to_light_binning()` (`calibration.py:277-337`) when the effective light binning
exceeds master binning.

**Trigger conditions (file:line):**

| Condition | Location | Effect |
|-----------|----------|--------|
| `light_binning > master_binning` and integer ratio | `calibration.py:301-309` | `bf = lb // mb`; if `bf > 1`, block resample |
| `infer_spatial_block_factor(master_shape, light_shape)` returns `k ? 2` | `calibration.py:193-213`, `543-546` | Raises **effective** `eff_lb` above header `XBINNING` when shapes imply master is ~kx larger than light (header binning wrong or missing) |
| `lb < mb` or `lb % mb != 0` | `calibration.py:301-308` | `MasterResamplingError` - no resample, calibrate aborts for that master |
| Post-resample shape mismatch | `calibration.py:567-577` | Try `align_resampled_master_to_light_shape()` (kron expand); else `MasterResamplingError` |

**Primary calibrate path:** `_calibrate_one_light_apply_masters_in_ram()` (`pipeline.py:15778+`):

- Reads light `XBINNING`/`YBINNING` (`pipeline.py:15813-15815`).
- Assumes library native binning `_mb_lib = 1` by default (`pipeline.py:15799-15800`, `15822-15825`).
- Dark: `get_processed_master(..., kind="dark", master_binning=_mb_lib, dark_resample_mode="sum")` (`pipeline.py:15835-15843`); logs when `pm.resampled` (`pipeline.py:15844-15847`).
- Flat: same with `kind="flat"` (`pipeline.py:15878-15887`); flat always block-**mean** inside `resample_master_to_light_binning` (`calibration.py:335-336`).

**Secondary path:** `_dark_array_for_calibration()` (`pipeline.py:738-760`) - same
`get_processed_master(..., dark_resample_mode="sum")`.

**Resample arithmetic:**

- **Dark:** block **SUM** (default) or block **MEAN** if `dark_resample_mode="mean"` (`calibration.py:328-333`).
- **Flat:** block **mean** only (`calibration.py:335-336`).

**When resample does *not* happen:** `bf == 1` - light and master share effective binning and
compatible shapes (no spatial block factor).

---

### A.2 Has it happened in practice?

**Archive/Drafts survey** (all drafts with raw lights, 2026-08-13):

| Draft | Light binning | Light shape | Library master (from cal_diag / CalibrationLibrary) | Resample? | CAL-DIAG artifact | Calibrated `VY_DKRSMP` |
|-------|---------------|-------------|------------------------------------------------------|-----------|-------------------|------------------------|
| draft_000435 | 2x2 | 1397x2082 | `Dark_60s_..._Bin1_20260422.fits` (bin 1, 2795x4164) | **Yes** (`bf=2`) | `cal_diag.json` PASS/SUM | **SUM** (316 calibrated frames) |
| draft_000509 | 2x2 | 1397x2082 | Same library pattern (bin1 masters) | **Yes** (inferred: `_mb_lib=1`, `light_bx=2`) | None (gate removed before/during cal) | None |
| draft_000510 | 2x2 | 1397x2082 | Same | **Yes** (inferred) | None | None |

**Measurements:**

- CalibrationLibrary darks on disk: `XBINNING=1`, shape `(2795, 4164)` for both `Dark_60s_...` and `Dark_120s_...`.
- All three production drafts use **2x2 lights** on a **1x1 master** library ? **resample on every dark and flat application** in Milan's current workflow.
- `draft_000435/cal_diag.json`: `block_factor=2`, `convention=SUM`, `status=PASS`, `m_L=2496`, `m_S=98.1`, `sky_median=2399.5`.
- Header scan across Archive: **316** frames with `VY_DKRSMP=SUM`; **0** `MEAN_AUTOCORRECTED`; **0** `PASSTHROUGH`; **300** calibrated frames without CAL-DIAG headers (post-removal runs).
- Grep of `Archive/` for `MEAN_AUTOCORRECTED`, `CAL-DIAG ABORT`, `VY_CDSTAT=ABORT`: **no matches**.

**Interpretation:** Resampling is **routine** in production today, not a rare edge case. The
**convention-mismatch** condition CAL-DIAG Check A was designed to catch has **not** been observed
to fire (always PASS/SUM). Draft 424 regression (2026-07-14 impl memo): **150/150 SUM**, 0 WARN, 0 FAIL.

---

### A.3 `dark_resample_mode="sum"` - physics and correctness

**Hardcoded locations:** `pipeline.py:760`, `15842` (always `"sum"`). The resample function also
accepts `"mean"` (`calibration.py:283`, `328-333`) but nothing in the live calibrate path passes it.

**Physical meaning (charge vs sensitivity):**

| Kind | Operation | Physical basis |
|------|-----------|----------------|
| **Dark** | Block **SUM** | Dark current and bias are **additive** per physical pixel. Software 2x2 binning that **sums** charge in the light frame implies the superpixel's dark level is the **sum** of constituent pixel darks ? master must be block-summed when downsampling from bin1. |
| **Dark** | Block **MEAN** | Correct only if the **light** was binned by **averaging** driver output (each stored superpixel ? mean of native pixels). |
| **Flat** | Block **mean** | Flat is **multiplicative** (relative sensitivity). Block mean preserves the ratio of sensitivity within the superpixel; block sum would scale incorrectly by ~`bf^2`. |

**Why dark and flat differ:** Subtraction is linear and additive; division is relative. Summing dark
blocks conserves total offset charge; averaging flat blocks conserves fractional gain.

**Is hardcoded `"sum"` always correct?**

- **Yes** for the common case: CMOS/software binning that **sums** (QHY ASCOM pattern used here;
  draft 435/424 PASS/SUM confirms Milan's lights match SUM convention).
- **No** if the acquisition driver **averages** on binning - then SUM overshoots by ~`bf^2` and
  Check A's signature (`median(dark) > median(light)`) appears. That is exactly the condition CAL-DIAG
  targeted; today it would pass undetected because the gate is gone.

---

### A.4 What does `INV-FLUX-01` verify?

**Definition:** `invariants_runtime.py:122-165`, wired in `get_processed_master()` when
`kind=="dark"`, `bf>1`, `dark_resample_mode=="sum"` (`calibration.py:557-565`).

**Assertion:** After block-**sum** downscale, `sum(out) ? sum(src_trim)` within relative tolerance
`FLUX_SUM_REL_TOL` (default **1e-6**). Upscale mode checks uniform SUM-preserving upscale shapes.

**Would catch:**

- Broken reshape/trim arithmetic (lost rows/cols, wrong axes).
- Non-finite sums.
- Implementation bugs in block-sum aggregation.

**Would *not* catch:**

- Wrong convention (**SUM vs MEAN**) - INV-FLUX-01 runs only for `mode=="sum"`; it validates that
  SUM was executed correctly, not that SUM was the right choice.
- Wrong master pairing (exposure, temperature, gain) - geometry can match while radiometry is wrong.
- Post-subtraction implausible sky (negative beyond noise) - no check today.
- Flat convention errors - separate INV-FLUX-02 (median ? 1 after normalization).

**Distinction:** INV-FLUX-01 verifies **the arithmetic of an operation**; CAL-DIAG verified **that
the operation matched the light's binning physics** and that **subtraction yielded a plausible sky**.

---

### A.5 QHY mod-4 grid interaction

**Measurement (draft 510 raw, SAT-DIAG spec S4.2 / `CURSOR_RESULT_quantisation_second_camera.md`):**

- Below digital clip, stored ADU lie on a **step-4 grid** (14-bit native << 2 in 16-bit container).
- **65535** is clip (`65535 mod 4 = 3`); only **13?024** off-grid pixels, all at 65535.

**Interaction with dark subtraction and resampling:**

| Step | Mod-4 effect |
|------|----------------|
| Light ? dark, same grid, no resample | Differences preserve mod-4 structure (both operands on grid). |
| Dark block **SUM** resample | Sum of values each ? 0 (mod 4) is ? 0 (mod 4) ? grid preserved if all inputs on grid. |
| Dark block **MEAN** resample | Mean of mod-4 values **need not** be mod-4 (e.g. `{4,8,12,20}` ? mean 11) ? **breaks grid** on resampled dark. |
| Light ? MEAN-resampled dark | Subtrahend may leave mod-4 grid; difference statistics slightly non-quantised. |

**Practical impact for Milan's QHY data:** With hardcoded SUM and matched convention, resampled dark
stays on the mod-4 grid; subtraction preserves quantisation structure except at clipped pixels.
If a MEAN driver were auto-corrected in, mod-4 structure of the dark would break - a second-order
reason to prefer **detect-and-abort** over silent MEAN correction on quantised sensors.

---

## Part B - Recover what CAL-DIAG actually did

Source: `git show 967f835^:src_py/cal_diag.py` (486 lines, deleted 2026-08-11) and
`dev/results/specs/VYVAR_CAL_DIAG_SPEC.md` v1.1. Not from summaries alone.

### B.1 Checks performed

**Check A - pre-subtraction convention cross-check** (`cal_diag.py` ~lines 156-232, spec S5.4):

1. Load representative light; `m_L = median(light)`.
2. Resample dark with **SUM**; `m_S = median(dark)` after `_match_and_crop_pair`.
3. **PASS (SUM):** if `m_S <= m_L * (1 + rel_tol)` ? convention `SUM`.
4. **Else if `bf > 1`:** compute `m_M = m_S / bf^2`.
   - If `m_M <= m_L * (1 + rel_tol)` and autocorrect enabled ? **WARN**, convention
     `MEAN_AUTOCORRECTED`, reload dark with `dark_resample_mode="mean"`, re-verify.
   - Else ? **ABORT**.
5. **Else (`bf == 1`):** `m_S > m_L` ? **ABORT** (wrong master / hot dark / scaling - not binning).

Default **`rel_tol = 0.02`** (2%).

**Check B - post-subtraction sky-median sanity** (`cal_diag.py` ~lines 234-270, spec S5.5):

- After chosen convention: `s = median(light - dark)`, `sigma_r = 1.4826 * MAD(diff)`.
- **ABORT:** `s < -hard_sigma * sigma_r` (default **`hard_sigma = 5.0`**).
- **WARN (continue):** `-hard_sigma * sigma_r <= s < 0` (slightly negative sky).
- **WARN (continue):** `s > sat_warn_frac * saturation_adu` (default **`sat_warn_frac = 0.90`**),
  skipped if saturation unresolvable.

**PASSTHROUGH:** No dark path ? both checks skipped; convention `PASSTHROUGH` (spec S5.8).

**Cadence:** Once per key `(obs_group_key, dark_path, light_binning)` on **first light in sorted
order** (spec S5.2).

---

### B.2 Five config parameters

| Parameter | Default | Controlled |
|-----------|---------|------------|
| `cal_diag_gate_enabled` | `True` | Master on/off (byte-identical arrays when off - tested) |
| `cal_diag_autocorrect_enabled` | `True` | MEAN retry on Check A MEAN signature |
| `cal_diag_rel_tol` | `0.02` | Check A median tolerance (clamped 0-0.2) |
| `cal_diag_hard_sigma` | `5.0` | Check B hard-fail floor (clamped 3-10) |
| `cal_diag_sat_warn_frac` | `0.90` | Check B high-sky WARN (clamped 0.5-1.0) |

**Genuinely needed vs knobs:**

- **Core logic needs no user-facing knobs:** thresholds are stability margins, not science choices.
  SAT-DIAG precedent: fixed defaults in spec/invariant, not registry keys.
- **`gate_enabled`:** invited silent removal via config audit (happened structurally via `967f835`).
- **`autocorrect`:** policy choice, not physics - debatable under current iron rules (see B.4).
- **`rel_tol` / `hard_sigma` / `sat_warn_frac`:** tuning knobs that existed because they could;
  amateur tool can use conservative fixed constants.

All five were listed as reduction candidates in the parameter-budget audit that preceded removal.

---

### B.3 Did it ever fire?

| Evidence source | Non-PASS result |
|-----------------|-----------------|
| `Archive/Drafts/**/cal_diag.json` (3 files, all draft_435 variants) | **PASS / SUM only** |
| Calibrated FITS `VY_DKRSMP` (316 stamped frames) | **SUM only** |
| `VY_CDSTAT` | **PASS only** on stamped frames |
| `Archive/` grep `MEAN_AUTOCORRECTED`, `ABORT` | **None** |
| `test_cal_diag_gate.py` (deleted) | Synthetic AUTO-CORRECT, ABORT, WARN - **never production** |
| draft_424 regression (impl memo) | 150/150 SUM, 0 WARN, 0 FAIL |

**Conclusion:** In ~1 year of Milan's use, the gate **ran and passed** on real data with resample
(`bf=2`). **No production record** of `MEAN_AUTOCORRECTED`, `PASSTHROUGH` via CAL-DIAG, or ABORT.
Synthetic tests prove the failure modes were implemented; they never triggered on disk.

---

### B.4 Auto-correct behavior and Milan iron rules

**What it did:** On Check A MEAN signature (`m_S > m_L` but `m_S/bf^2 ? m_L`), re-ran dark resample
with block **MEAN**, set `convention=MEAN_AUTOCORRECTED`, `VY_CDSTAT=WARN`, logged **ERROR-level**
message, continued calibrate for that obs_group.

**Iron-rule tension (2026-08-13 context):**

- **2026-07-07 DECISIONS:** explicitly approved "**loud** auto-correction (SUM ? MEAN retry) or
  fail-closed abort."
- **2026-08-13 SAT-DIAG / INV-GATE-REMOVAL:** verification gates must not disappear on
  byte-identity; strength of action follows provenance; **detect ? silently fix**.
- Auto-correct is **not silent** (loud log + WARN header) but **does change radiometry without
  user confirmation** - qualitatively different from SAT-DIAG's "flag then policy decides."

**Assessment for today:** A rebuild should **default to fail-closed ABORT** on convention mismatch.
Optional MEAN path belongs only behind an **explicit user decision** (not a config registry key -
e.g. documented advanced override or future equipment-level provenance), not automatic correction.

---

### B.5 Original spec purpose (verbatim intent)

From `VYVAR_CAL_DIAG_SPEC.md` S1:

> VYVAR lets the user build masters at any binning and calibrate lights at any (higher or equal)
> binning. Today only **geometric** adaptation is guarded ... **No radiometric check exists** -
> nothing verifies that the resampled dark is physically consistent with the light before
> subtraction, and nothing verifies the subtracted sky is plausible. A camera/driver that AVERAGES
> on binning (instead of summing charge) would today produce silently negative calibrated skies and
> garbage photometry.
>
> CAL-DIAG verifies radiometry **from data, not camera conventions** - camera-agnostic, no per-camera
> hardcoding.

---

## Part C - How do others verify calibration?

### C.1 Radiometric verification vs performing subtraction

| Package | Performs dark/flat? | Verifies radiometric correctness of subtraction? |
|---------|-------------------|--------------------------------------------------|
| **ccdproc** | Yes (`subtract_dark`, `flat_correct`) | **No.** Requires **identical shapes** (`ccd.shape != master.shape` ? `ValueError`). Optional exposure scaling only. No post-subtraction sky test. ([ccdproc `subtract_dark`](https://ccdproc.readthedocs.io/en/stable/api/ccdproc.subtract_dark.html)) |
| **DRAGONS** | Yes (recipe primitives) | **No radiometric gate on science frames.** Calibrations validated via `caldb` association and human inspection; `cp_verify`-style metrics apply to **calibration product construction**, not per-science-frame dark convention. ([DRAGONS tutorials](https://dragons.readthedocs.io/)) |
| **BANZAI** | Yes (mandatory dark subtract) | **Partial on cal frames only:** compares new supers to previous supers (outlier rejection); science reduction assumes correct supers. Temperature scaling via `DRKTCOEF`. No science-frame sky-median gate. ([BANZAI docs](https://banzai.readthedocs.io/), [arXiv:1811.04163](https://doi.org/10.48550/arxiv.1811.04163)) |
| **LSST ISR / cp_verify** | Yes | **Verifies calibration products** (bias/dark/flat residuals vs DMTN-101 limits); certifies calibrations for date ranges. **Not** a per-observation SUM-vs-MEAN binning test on user-supplied mismatched masters. ([arXiv:2404.14516](https://arxiv.org/html/2404.14516), [DMTN-101](https://dmtn-101.lsst.io/DMTN-101.pdf), [lsst/cp_verify](https://github.com/lsst/cp_verify)) |
| **PhotometryPipeline** | **No** (expects pre-reduced inputs) | N/A - diagnostics are photometric zeropoint stability and control-star flatness, not ISR. ([PP diagnostics](https://photometrypipeline.readthedocs.io/en/latest/diagnostics.html)) |
| **STDWeb / STDPipe** | **No** (expects science-ready images) | Example preprocessing subtracts dark and divides by flat with **no verification**; optional masking uses `dark > median + 10*MAD(dark)` for hot pixels only. ([STDPipe preprocessing](https://stdpipe.readthedocs.io/en/latest/preprocessing.html)) |

**Plain finding:** No surveyed package implements CAL-DIAG **Check A** (median light vs median
resampled dark as a binning-convention falsification test) on science data.

---

### C.2 Masters at different binning from lights

| System | Handling |
|--------|----------|
| **ccdproc** | User must pre-match shapes; **no built-in resample** across binning. |
| **VYVAR** | Explicit RAM resample: dark SUM / flat MEAN; `CALIBRATION_LIBRARY_NATIVE_BINNING=1`. |
| **Professional observatories** | Masters typically built at **same binning as science**; binning mismatch treated as operator error, not auto-resolved. |
| **Amateur stacks (Siril, etc.)** | Generally assume matched resolution; user re-bin masters manually. |

**Verification when resampling exists:** External tools largely **assume discipline** (matched
binning). VYVAR's explicit cross-binning support is a **differentiating feature** - and leaves a
**VYVAR-specific verification gap** that professional pipelines avoid by convention.

---

### C.3 Common sanity checks elsewhere

| Check | Who does it |
|-------|-------------|
| Negative sky after dark subtract | **Not automated** in ccdproc/DRAGONS/BANZAI/STDWeb; visual inspection in amateur workflows. |
| Implausible flat normalization | LSST cp_verify (product level); VYVAR **INV-FLUX-02** (median ? 1). |
| Residual gradient / illumination | LSST beyond-ISR metrics; STDPipe chunk-wise photometry to catch flat errors; not a universal gate. |
| Master calibration outlier rejection | BANZAI super-cal comparison; LSST cp_verify. |
| Control-star flatness | PhotometryPipeline diagnostics (downstream, not calibration). |

These are **cheap and general**; CAL-DIAG Check B (sky median vs MAD floor) is in this class and
**not widely automated** for amateur cal pipelines.

---

### C.4 If nobody does what CAL-DIAG did

**Plain statement:** The **combination** of (1) automatic cross-binning master resample with (2)
data-derived SUM/MEAN convention test and (3) post-subtraction sky sanity **was unique to VYVAR**
among packages reviewed. Others either forbid shape mismatch or assume matched binning.

**Legitimate readings:**

- **Unnecessary for observatory pipelines** that never mix binnings - discipline by operations.
- **Necessary for VYVAR** - product promise ("masters at any binning") explicitly breaks that
  discipline; verification must live in the tool.

---

## Part D - Propose a shape

### D.1 Revert, rebuild, or drop

| Option | Verdict |
|--------|---------|
| **Blind revert** of `967f835` | **Not recommended.** Restores five config keys, auto-correct policy from 2026-07, and pregate wiring that failed the parameter-budget audit. Does not address INV-GATE-REMOVAL or SAT-DIAG lessons. |
| **Drop** | **Not recommended.** Resample is **routine** (all three drafts, every calibrate); INV-FLUX-01 does not cover convention or sky sanity; amateur/universal use case still allows MEAN-binning drivers. |
| **Rebuild** (recommended) | New **invariant-backed** gate, SAT-DIAG-shaped: derived from data, falsifiable compatibility test, provenance without config registry keys. |

---

### D.2 Rebuild sketch (SAT-DIAG pattern)

**Working name:** CAL-DIAG v2 / `INV-CAL-01` (exact ID for Milan).

**Structure (mirror SAT-DIAG spec):**

1. **Check A - binning convention compatibility (one-sided test, like SAT-DIAG ceiling test):**
   - Hypothesis: light was **SUM-binned** ? resampled dark (SUM) must satisfy
     `median(dark) <= median(light) * (1 + tol)`.
   - Refutation: `median(dark) > median(light)` at `bf > 1` with
     `median(dark)/bf^2 ? median(light)` ? **MEAN-driver signature** ? **ABORT** obs_group
     (no auto-MEAN in v2 default).
   - At `bf == 1`: same inequality; failure ? ABORT with "wrong master / hot dark / scaling".

2. **Check B - post-subtraction sky sanity:**
   - `median(light - dark) < -5 * MAD_sigma` ? **ABORT**.
   - Slightly negative or high-sky ? **WARN** in provenance only (continue).

3. **Provenance (mandatory, not optional config):**
   - FITS: `VY_DKRSMP` = `SUM` | `PASSTHROUGH` | `UNVERIFIED` (if gate skipped by bug - should not happen).
   - Draft: `cal_diag.json` or additive block in `pipeline_meta.json` (same as v1).
   - `VY_CDSTAT` = `PASS` | `WARN` | `ABORT`.
   - Drop `MEAN_AUTOCORRECTED` unless Milan explicitly re-authorizes auto-correction.

4. **Invariant registration:** `docs/VYVAR_INVARIANTS.md` - policy FAIL-CLOSED on ABORT;
   cross-reference spec in `dev/results/specs/VYVAR_CAL_DIAG_SPEC_v2.md` (update v1.1, do not resurrect v1 keys).

5. **Config keys: zero.** Fixed constants in spec (e.g. `rel_tol=0.02`, `hard_sigma=5.0`,
   `sat_warn_frac=0.90`) - same as SAT-DIAG fixed `sat_diag_max_frames`. Gate always on when dark
   applied; no `cal_diag_gate_enabled`.

6. **Cadence / wiring:** Once per `(obs_group, dark_path, light_binning)` pregate; all calibrate
   paths (sequential, RAM QC, MP parent pre-dispatch) - reuse v1.1 coverage table.

7. **Relationship to INV-FLUX-01:** Complementary. FLUX-01 guards resample arithmetic; CAL-DIAG
   guards convention choice and subtraction outcome.

**Justify any non-zero keys:** None proposed.

---

### D.3 What the proposal does *not* cover

(Mirror SAT-DIAG S2 out-of-scope list.)

- Flat-norm sanity at calibrate (v1 spec S2 deferred item).
- Master **pairing** correctness (exposure, temperature, gain match) beyond median sanity.
- Validity age-clock / library precedence (D1/D2 - unchanged).
- MEAN-binning auto-correction without explicit user authorization.
- RN exponent / param_resolver semantics (closed by DECISIONS; not re-litigated).
- Upsampling master to finer binning (`lb < mb`) - already hard-fail via `MasterResamplingError`.
- Camera-specific firmware tables or driver identification.

**One-sided test limit (explicit):** Passing Check A proves **SUM convention is consistent with
medians**; it does **not** prove absolute radiometric accuracy (bias level, amp glow, etc.).

---

### D.4 Priority if resample is routine but mismatch never fired

- **Urgency:** **Medium**, not emergency. Milan's QHY workflow resamples every run but has **never
  triggered** mismatch or sky ABORT in production.
- **Still warranted:** VYVAR targets users who build masters independently; binning convention is
  not observable from headers alone; the failure mode (negative sky, ruined photometry) is severe
  and cheap to detect.
- **Honest case:** A gate for a condition that has **not yet occurred locally** is **insurance for
  universality**, not a fix for a current Milan bug. Rank below active science blockers (white cores,
  exposure ramp) but **above nice-to-have** because removal was invisible to P1 SHA (`INV-GATE-REMOVAL`).

---

## Part E - Summary

| Part | Headline finding |
|------|------------------|
| **A** | Resample still happens whenever library bin1 masters meet bin2 lights (`bf=2` in all Archive drafts). Hardcoded dark SUM is correct for Milan's SUM driver, wrong for MEAN drivers. INV-FLUX-01 ? convention check. QHY mod-4 preserved under SUM resample; MEAN resample would break grid. |
| **B** | v1 gate: Check A (median convention + optional MEAN autocorrect), Check B (sky MAD floor). Five config keys. **Never fired non-PASS in production**; 316x SUM/PASS on disk. |
| **C** | No external package does VYVAR's cross-binning convention test; observatories assume matched binning. Sky-median sanity is rare as an automated gate. |
| **D** | **Rebuild** as invariant (`INV-CAL-01`), zero config keys, no default auto-MEAN; do not blind-revert. |

---

## DECISION REQUIRED

1. **Reinstate calibration radiometry gate?** (Yes / No / Defer)
2. **If yes: revert `967f835` verbatim or rebuild v2?** (Recommend: **rebuild v2**)
3. **Allow MEAN auto-correction on convention mismatch?** (Recommend: **No** - ABORT + loud message; optional future explicit override only)
4. **Register as invariant `INV-CAL-01` with zero config keys?** (Recommend: **Yes**)
5. **Fail-closed per obs_group on ABORT (skip calibrated output)?** (Recommend: **Yes**, same as v1.1)
6. **Priority relative to white cores / exposure ramp?** (Recommend: **after** those; **before** anchor-tool cosmetic work)

---

**Recommendation (one paragraph):**

Rebuild CAL-DIAG as **`INV-CAL-01`**, not a blind revert: keep the v1.1 physics (Check A median
convention test + Check B sky sanity) and provenance headers, drop the five config parameters and
default auto-MEAN correction, and register the gate in `VYVAR_INVARIANTS.md` so parameter-budget
audits cannot delete it silently again. Evidence supports reinstatement even though Milan's QHY data
have only ever **passed** SUM/`bf=2` - resampling is **routine** in every Archive draft, INV-FLUX-01
does not substitute, no external pipeline provides this check, and the product promise of arbitrary
master binning leaves amateurs exposed to a silent MEAN-driver failure mode. Priority is **medium**:
not blocking current BO CVn science, but cheap insurance aligned with SAT-DIAG's "verify from data,
never assume camera convention" principle and INV-GATE-REMOVAL.

---

## Files changed

None (investigation only).
