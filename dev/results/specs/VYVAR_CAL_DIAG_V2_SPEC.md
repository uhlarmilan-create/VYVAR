# VYVAR - CAL-DIAG v2 / INV-CAL-01: Derived calibration convention gate (spec)

Status: **IMPLEMENTED (2026-08-13, Milan authorized; not pushed).** Decision 2 amendment: split `INDETERMINATE` into `INDETERMINATE_NEGLIGIBLE` and `INDETERMINATE_UNMEASURED`.
Date: 2026-08-13.
Grounding: `dev/results/CURSOR_RESULT_dark_binning_physics.md`,
`dev/results/CURSOR_RESULT_cal_diag_investigation.md`,
`dev/results/specs/VYVAR_CAL_DIAG_SPEC.md` (v1.1, superseded by this document),
`dev/results/specs/VYVAR_SAT_DIAG_SPEC.md` (structure model).

**Governing principle (unchanged, Milan 2026-07-07):** a user may build masters at
any binning and calibrate lights at any (higher or equal) binning. VYVAR must
**derive and verify from data**, not assume camera firmware conventions.

**What changed from v1.1 (2026-08-13):**

- v1 **tested** SUM against a 2% median tolerance and optionally auto-corrected to MEAN.
- v2 **derives** the per-read pedestal from master-dark structure, **predicts** the
  convention-dependent sky separation `(bf-?1)-P`, and **resolves** SUM vs MEAN from
  light-dark median scaling - the same move SAT-DIAG made when it derived the ceiling
  instead of trusting `SATURATE_ADU`.
- **No silent auto-correction.** Mismatch ? ABORT with full numeric report.
- **Zero configuration keys.** Constants in this spec only.
- Registered as **`INV-CAL-01`** (not optional config).

**Physics anchor (measured, QHY294MM, draft 435/509/510):**

| Quantity | Value |
|----------|-------|
| Pedestal P (bin1 pixel, intercept) | **24.548 - 0.011 ADU** |
| Block factor bf (bin1 master ? bin2 light) | **2** |
| Predicted convention separation `(bf-?1)-P` | **73.6 - 0.03 ADU** |
| Observed `M(D_SUM) ? M(D_MEAN)` | **73.65 ADU** |
| Post-SUM-subtraction sky `M(L?D_SUM)` | **2399.53 ADU** |
| Simulated sky if MEAN dark used | **2471.5 ADU** (+72 ADU) |
| Archived cal vs synthetic SUM replication | **?0.0001 ADU** |

Current Milan workflow is **verified correct**; v2 would **pass silently** on his data.

---

## 1. Purpose

When a master dark is resampled to match a higher-binned light, VYVAR must apply the
**same charge-combining convention** the acquisition driver used (block **SUM** vs block
**MEAN**). `INV-FLUX-01` verifies that a chosen block SUM is arithmetically flux-conserving;
it does **not** verify that SUM was the right operator.

For **CMOS software binning** (digitize each native pixel, then combine in software), block
SUM of a bin1 master reproduces driver-side SUM binning. For **CCD on-chip binning** (one
ADC conversion per superpixel), block SUM of a bin1 master carries `(bf-?1)` extra
pedestals per superpixel - a **~3% sky error** at Milan's pedestal, invisible to
`INV-FLUX-01`.

CAL-DIAG v2 derives the pedestal **P** from master-dark data, predicts the SUM-MEAN
separation, resolves which convention the light median supports, and runs an independent
post-subtraction sky sanity check (Check B, retained from v1).

---

## 2. Scope / non-goals

### In scope

- Check **P** - pedestal derivation from available master dark(s).
- Check **C** - convention resolution (SUM vs MEAN resample) from derived pedestal + light/dark medians.
- Check **B** - post-subtraction sky sanity (negative sky, sky vs saturation).
- Provenance: `VY_DKRSMP`, `VY_DKRSMP_SRC`, `VY_CDSTAT`, `VY_CDSKY`, draft `cal_diag.json` block.
- Fail-closed per `obs_group` on hard failure (same isolation as v1.1 -5.6).
- Once per key `(obs_group_key, dark_path, light_binning)` on deterministic representative frame.

### Out of scope

- Flat-norm sanity at calibrate (v1 -2 deferred item).
- Master pairing correctness (exposure, temperature, gain match) beyond Check B symptoms.
- RN exponent / `param_resolver` semantics (closed in DECISIONS 2026-07-07).
- Upsampling master to finer binning (`bf < 1`) - already `MasterResamplingError`.
- CCD hardware-binning **correction** - gate detects inconsistency and ABORTs; user must supply matched binning or a bin2 master (no linear fix exists).
- Implementation code, UI design, PARAMS registry entries (follow-on task).
- Pedestal recording for noise model / SAT-DIAG (separate finding -11 - spec references only).

---

## 3. Physics model (normative)

Per bin1 master pixel at exposure time t (seconds):

```
v(t) = P + k-t + ?
```

| Symbol | Meaning |
|--------|---------|
| **P** | Pedestal per ADC read (stored image ADU, bin1 pixel) |
| **k** | Dark-current rate (ADU/s per bin1 pixel) |
| **?** | Noise / structure (averaged down in master stack) |

Resample to light binning with integer block factor **bf** (square binning: N = bf- pixels per superpixel):

| Operator | Superpixel dark (pedestal-dominated) |
|----------|--------------------------------------|
| **SUM** | `D_SUM ? bf--(P + k-t)` |
| **MEAN** | `D_MEAN ? P + k-t` |

**Convention separation (master-only, exact under uniform P,k across block):**

```
?_dark = M(D_SUM) ? M(D_MEAN) = (bf- ? 1)-(P + k-t)
```

For representative light L with the same crop geometry as calibration:

```
M(L ? D_SUM) ? M(L ? D_MEAN) = M(D_MEAN) ? M(D_SUM) = ??_dark
```

The post-subtraction sky difference is **independent of L** - convention discrimination
must use **light median scaling**, not post-sub sky alone.

**Light median scaling (CMOS digital binning):**

| Driver convention | Typical `M(L)` vs `M(D_MEAN)` |
|-------------------|-------------------------------|
| **SUM** (add 4 digitized pixels) | `M(L) / M(D_MEAN) ? bf- - (P+k-t+s) / (P+k-t)` |
| **MEAN** (average 4 pixels) | `M(L) / M(D_MEAN) ? (P+k-t+s) / (P+k-t)` |

When sky+star contribution to the median satisfies `s* = M(L) ? bf--M(D_MEAN)` (SUM case) or
`s* = M(L) ? M(D_MEAN)` (MEAN case), the discriminant reduces to comparing `M(L)/M(D_MEAN)`
against **`bf-`** (SUM) vs **`? 1`** (MEAN), with ambiguity when `(bf-?1)-P` is small.

**CCD on-chip binning:** true bin2 superpixel carries **one** pedestal; block SUM of bin1
master carries **bf-** pedestals. Linear SUM resample cannot represent this; Check C fails
with `CCD_LINEAR_INCONSISTENT` when SUM-subtracted sky fails Check B but MEAN-subtracted
sky passes (-7.4).

---

## 4. Checks

All medians use the same crop/match geometry as calibration (`_match_and_crop_pair`).

Constants (implementation - **not user config**):

| Constant | Default | Role |
|----------|---------|------|
| `CAL_PED_INTERCEPT_MIN_EXPTIMES` | 2 | Minimum distinct master exptimes for intercept pedestal |
| `CAL_PED_BOOTSTRAP_N` | 200 | Bootstrap draws for intercept ? |
| `CAL_PED_SUBSAMPLE_N` | 100000 | Pixels for exposure fit |
| `CAL_PED_CONSISTENCY_REL` | 0.05 | Relative tolerance: `?_dark` vs `(bf-?1)-P` |
| `CAL_CONV_SUM_SCALE` | 0.85 | SUM: require `M(L)/M(D_MEAN) ? bf--(1??)` |
| `CAL_CONV_MEAN_SCALE` | 1.15 | MEAN: require `M(L)/M(D_MEAN) ? ?` |
| `CAL_CONV_AMBIGUITY_BAND` | shared | Gray zone between scales ? INDETERMINATE |
| `CAL_SKY_HARD_SIGMA` | 5.0 | Check B hard-fail floor (MAD-based) |
| `CAL_SKY_SAT_WARN_FRAC` | 0.90 | Check B high-sky WARN vs saturation |
| `CAL_MAD_SIGMA` | 1.4826 | MAD ? ? scale |

Where `? = CAL_PED_CONSISTENCY_REL` (5% band on ratio tests unless noted).

### 4.1 Check P - pedestal derivation

**Inputs:** all scoped master darks for `(equipment, temperature-?T, gain, filter)` usable
for the obs_group - at minimum the resolved `dark_path` master; prefer every library dark
at the same temperature and gain with distinct `EXPTIME`.

**Method A - intercept (preferred, ?2 exptimes):**

1. For each master dark at exposure `t_i`, sample `CAL_PED_SUBSAMPLE_N` pixels (deterministic seed from dark path).
2. Per-pixel fit: `v = P + k-t` across available `t_i`.
3. **P** = intercept; **?_P** = bootstrap standard deviation (`CAL_PED_BOOTSTRAP_N` draws).
4. Record `k`, `?_k`.

**Method B - single-exposure fallback (exactly one master exptime):**

1. If dark-current slope is negligible vs P: **`P := M(dark_bin1)`**, `?_P := ?_pixel / ?N_stack` (conservative; stack has NCOMBINE frames).
2. **`k` unknown** - set `k := 0`, `k_status := UNKNOWN`.
3. Pedestal provenance: `SINGLE_MASTER_MEDIAN`.

**Negligible dark-current criterion (enables Method B validation):**

```
|k| - t_max < 0.05 - P
```

If two exptimes exist but `|M(dark,t2) ? M(dark,t1)| / |t2?t1| - t_max < 0.05-P`, treat as
pedestal-dominated (QHY294MM at ?10 -C: measured **k = 0.00107 ADU/s**, **0.06 ADU over 60 s**
vs **P ? 24.5 ADU** - qualifies).

**Method C - cannot derive:**

If no master dark, or master has non-finite median, Check P skipped (PASSTHROUGH path).

**Pedestal consistency (one-sided, when bf > 1):**

Measure `?_dark = M(D_SUM) ? M(D_MEAN)` from resampled masters at the light exposure time
(use `t_light` for k-t correction when k known).

**Pass:** `|?_dark ? (bf-?1)-(P + k-t_light)| ? CAL_PED_CONSISTENCY_REL - max(?_dark, 1 ADU)`.

**Fail:** `PEDESTAL_INCONSISTENT` - WARN, proceed with measured `?_dark` but do not trust
derived P for prediction; convention check uses measured separation only.

**What Check P falsifies:** internal inconsistency between intercept pedestal and observed
SUM-MEAN dark separation. **Cannot falsify:** whether P equals physical readout offset
(no bin2 reference dark required).

### 4.2 Check C - convention resolution

**Precondition:** `bf > 1` and dark applied. When `bf == 1`, skip Check C; convention `NONE`.

Compute from representative raw light L and bin1 master dark:

```
R = M(L) / M(D_MEAN)     (guard: M(D_MEAN) > 0)
Q = M(L) / M(D_SUM)      (guard: M(D_SUM) > 0)
?_dark = M(D_SUM) ? M(D_MEAN)
s_SUM = M(L ? D_SUM)
s_MEAN = M(L ? D_MEAN)   (counterfactual; not applied unless MEAN wins)
```

**Predictions (pedestal-dominated, k-t ? P):**

```
?_pred = (bf- ? 1) - P
s_SUM_expected = s_MEAN_expected ? ?_pred     (identities; s differs by ??_pred)
```

Uncertainties (reporting, not gating unless noted):

```
?_?_pred ? (bf- ? 1) - ?_P
?_s ? 1.4826 - MAD(L ? D_applied) / ?N_eff     (N_eff ? n_pixels; typically ? 1 ADU)
```

**Decision table:**

| Condition | Resolved convention | `VY_DKRSMP_SRC` |
|-----------|---------------------|-----------------|
| `R ? bf- - (1 ? ?)` AND `Q ? 1` | **SUM** | **DERIVED** |
| `R ? CAL_CONV_MEAN_SCALE` AND `Q < bf--(1??)` | **MEAN** | **DERIVED** |
| `R` in gray band AND `?_dark < CAL_RESOLV_LIMIT` (-4.2.1) | **INDETERMINATE** | **INDETERMINATE** |
| Gray band but `?_dark ? CAL_RESOLV_LIMIT` | **CONFLICT** | - (ABORT) |
| SUM derived but `s_SUM` fails Check B; MEAN counterfactual passes | **CCD_LINEAR_INCONSISTENT** | - (ABORT, -7.4) |

Apply dark using resolved convention. On **INDETERMINATE**, apply **SUM** (current pipeline
default), emit **WARN**, `VY_DKRSMP_SRC=INDETERMINATE` - do **not** auto-switch to MEAN.

**Physical bound (retained from v1, one-sided):**

Under matched convention, dark content ? light content:

```
M(D_applied) ? M(L) - (1 + ?)
```

Violation with `bf == 1` ? `WRONG_MASTER` (hot dark, pairing error) - ABORT.

#### 4.2.1 Resolvability limit (numeric)

Conventions become **indistinguishable** when the predicted separation is smaller than
combined pedestal uncertainty:

```
CAL_RESOLV_LIMIT = max( 3 - ?_P - (bf- ? 1),  1 ADU )
```

**QHY294MM reference (bf = 2, ?_P = 0.011 ADU):**

```
CAL_RESOLV_LIMIT = max(3 - 0.011 - 3, 1) = max(0.10, 1) = 1 ADU
```

Measured `?_dark = 73.65 ADU` ? **730- above** resolvability floor.

**Pedestal below which SUM and MEAN are indistinguishable (bf = 2):**

```
P_indist = CAL_RESOLV_LIMIT / (bf- ? 1) = 1 / 3 ? 0.33 ADU
```

Any camera with **P ? 1 ADU** and **bf = 2** is resolvable. At **bf = 2**, Milan's
**P = 24.5 ADU** exceeds the threshold by **~74-**.

**Frame-to-frame sky variation (~192 ADU std across 20 BO CVn frames) is not the test noise.**
Check C uses **one frame median**; the discriminant `R` is stable to ?1 ADU on a single
frame. Temporal sky variation affects Check B repeatability, not Check C on a representative frame.

**What Check C falsifies (two-sided where noted):**

| Claim | Falsified by |
|-------|--------------|
| "SUM resample matches this light" | `R < bf--(1??)` or `M(D_SUM) > M(L)` |
| "MEAN resample matches this light" | `R > CAL_CONV_MEAN_SCALE` when MEAN selected |
| "Pedestal model predicts observed ?_dark" | Check P consistency failure (WARN) |
| "Linear SUM resample is valid for this sensor class" | CCD pattern (-7.4) |

**What Check C cannot falsify:**

- Absolute radiometric accuracy (bias level, amp glow structure).
- Correct master **temperature / gain / age** pairing (Check B may catch symptoms).
- Non-linear spatial structure that scales neither as SUM nor MEAN.

### 4.3 Check B - post-subtraction sky sanity (unchanged merit)

After applying the **resolved** convention (not counterfactual):

```
s = M(L ? D_applied)
?_r = CAL_MAD_SIGMA - MAD(L ? D_applied)
```

| Condition | Status | Action |
|-----------|--------|--------|
| `s < ?CAL_SKY_HARD_SIGMA - ?_r` | **ABORT** | Fail-closed obs_group |
| `?CAL_SKY_HARD_SIGMA-?_r ? s < 0` | **WARN** | Continue |
| `s > CAL_SKY_SAT_WARN_FRAC - saturation_adu` | **WARN** | Continue (skip if saturation unresolvable) |
| else | **PASS** | Continue |

**One-sided:** detects implausibly **negative** sky beyond noise (hard fail) and
implausibly **high** sky (warn). **Cannot detect** sky that is wrong but positive and
within band (e.g. uniform +74 ADU offset would pass - Check C must catch convention errors).

---

## 5. Behaviour

### 5.1 No silent auto-correction

On **ABORT**, emit one ERROR per key:

```
INV-CAL-01 ABORT [obs_group]: convention=CONFLICT
  P=24.548-0.011 ADU  k=0.00107 ADU/s  bf=2
  ?_dark(meas)=73.65  ?_pred=73.64 ADU
  R=M(L)/M(D_MEAN)=101.6  (SUM expects ?bf--0.85=3.4)
  s_SUM=2399.53  s_MEAN(counterfactual)=2471.5
  Check B: s=... ?_r=...
  Action: use matched binning, acquire bin2 master, or verify driver binning mode
```

User decides. **No MEAN auto-switch.**

### 5.2 Provenance

| Keyword / field | Values | Meaning |
|-----------------|--------|---------|
| `VY_DKRSMP` | `SUM` \| `MEAN` \| `NONE` \| `PASSTHROUGH` | Operator applied |
| `VY_DKRSMP_SRC` | `DERIVED` \| `INDETERMINATE` \| `ASSUMED_SUM` \| `PASSTHROUGH` | How convention was established |
| `VY_CDSTAT` | `PASS` \| `WARN` \| `ABORT` | Gate outcome |
| `VY_CDSKY` | float | Post-subtraction sky median (applied convention) |
| `VY_CPED` | float | Derived P (bin1 ADU); absent if not derived |
| `VY_CDRES` | `bf`, `?_dark`, `R`, `s_SUM`, `s_MEAN` | Diagnostic JSON in `cal_diag.json` |

**DERIVED** = Check C resolved with confidence (SUM or MEAN).
**INDETERMINATE** = gray band; SUM applied with WARN.
**ASSUMED_SUM** = gate skipped (disabled bug path - must not occur when INV-CAL-01 wired).

Draft `cal_diag.json` additive block (same keys as v1; extended fields above).

### 5.3 Zero configuration keys

No `cal_diag_*` registry entries. Removal requires invariant deprecation per
`INV-GATE-REMOVAL`, not parameter-budget audit.

### 5.4 INV-CAL-01 registration (for `docs/VYVAR_INVARIANTS.md`)

| Field | Value |
|-------|-------|
| **ID** | `INV-CAL-01` |
| **Contract** | When master dark is resampled to light binning (`bf > 1`), calibration convention (block SUM vs MEAN) is **derived** from master-dark pedestal and light-dark median scaling; post-subtraction sky passes Check B. Provenance `VY_DKRSMP` + `VY_DKRSMP_SRC` recorded. Fail-closed per obs_group on ABORT. |
| **Enforced** | both (`cal_diag.py` + `invariants_runtime`) |
| **Policy** | FAIL (ABORT path) |
| **Evidence** | CAL-DIAG v2 spec 2026-08-13; supersedes v1 removed `967f835`; physics `CURSOR_RESULT_dark_binning_physics.md` |

---

## 6. Coverage and cadence

Same path coverage as v1.1 -5.1:

- `calibrate_lights_to_calibrated` sequential loop
- RAM QC calibration
- MP batch (parent pre-dispatch variant (a) preferred)

Once per `(obs_group_key, dark_path, light_binning)`; representative frame = first light
in sorted order within obs_group.

Shared dark cache keyed by convention (SUM vs MEAN) - no per-frame re-decision.

---

## 7. Case survival table

| Case | bf | Check P | Check C outcome | Check B | Result |
|------|-----|---------|-----------------|---------|--------|
| **CMOS SUM** (QHY294MM, verified) | >1 | P?24.5 from intercept or single-master | `R?102 ? 4-0.85` ? **SUM, DERIVED** | s?2399 PASS | **PASS** (Milan data) |
| **CMOS MEAN** driver | >1 | P derived | `R?1`, MEAN branch ? **MEAN, DERIVED** | s_MEAN PASS | **PASS** with MEAN dark |
| **CMOS MEAN light, SUM dark applied** (misconfig) | >1 | P derived | `R` high but production used SUM wrongly- if user forced SUM while MEAN lit: `s_SUM` low by ?_pred | **ABORT** Check B or CONFLICT on `R` | **ABORT** |
| **CCD on-chip bin** (one ADC/superpixel) | >1 | P derived; `?_dark` matches | SUM applied: **s_SUM fails** B; s_MEAN counterfactual **PASS** | split | **ABORT** `CCD_LINEAR_INCONSISTENT` - no linear fix |
| **Matched binning** (bin2 master, bin2 light) | 1 | P optional | **Skipped** (`VY_DKRSMP=NONE`) | s sanity only | Check B only |
| **Single master exptime** (common) | >1 | Method B: `P=M(dark_bin1)`; k=0 if pedestal-dominated | Uses measured `?_dark`; if `P>0.33 ADU` resolvable | B | **DERIVED** or **INDETERMINATE** if P tiny |
| **Single master, warm sensor** (k confounded) | >1 | k unknown; Check P consistency weakened | Wider gray band; may ? **INDETERMINATE** | B | **WARN**, SUM default |
| **No dark** | - | Skip | **PASSTHROUGH** | Skip | No cal_diag dark checks |
| **P below resolvability** (`P < 0.33 ADU` at bf=2) | >1 | P ? 0 | `?_dark < 1 ADU` ? **INDETERMINATE** | B | **WARN**, SUM default; honest "cannot distinguish" |
| **Wrong camera/gain/readmode master** | >1 | P wrong | `R` or Check P inconsistent | likely **ABORT** B | **ABORT** (wrong master) |
| **bf>1 but INDETERMINATE + wrong guess** | >1 | - | SUM assumed | subtle bias | **Risk bounded**: only when `P < 0.33 ADU`; at Milan P=24.5 not exposed |

**Defect criterion:** any case yielding **DERIVED** when truth is MEAN (or vice versa) without
INDETERMINATE/WARN. Review MEAN-row and CCD-row in implementation tests.

---

## 8. Failure behaviour (fail-closed)

On ABORT (Check B hard fail, CONFLICT, CCD_LINEAR_INCONSISTENT, WRONG_MASTER at bf=1):

- Skip calibration for **all frames** in obs_group.
- Delete partial outputs if any.
- Increment `cal_diag_aborted_groups`.
- Surviving obs_groups continue.

On WARN (INDETERMINATE, slight negative sky, high sky, PEDESTAL_INCONSISTENT): continue, stamp headers.

---

## 9. What this gate does **not** cover (explicit)

Mirror SAT-DIAG -2 one-sided limits:

- **Cannot refute** a convention choice when `(bf-?1)-P` is below resolvability (reports INDETERMINATE).
- **Cannot verify** absolute pedestal matches vendor OFFSET keyword (header is uninformative on QHY294MM).
- **Cannot fix** CCD hardware binning - only detects linear SUM/MEAN inconsistency.
- **Cannot verify** flat-field correctness (INV-FLUX-02 only).
- **Cannot verify** master temporal validity or temperature match.
- **Does not** require bin2 master dark on disk (derives from structure; direct bin2 comparison remains optional validation).

---

## 10. Plain language (Milan)

**What this catches**

- Calibrating 2-2 lights with a 1-1 master when the driver **averages** instead of **sums**
  - the failure mode CAL-DIAG v1 was built for, now with a **measured pedestal** instead of a 2% guess.
- **CCD-style** data where block SUM of a bin1 master is **linearly wrong** (~3% sky error at your pedestal).
- Obvious broken calibration: **negative sky**, wrong master, gross pairing errors (Check B).

**What it does not catch**

- Subtle errors below **~1 ADU** convention separation (pedestal indistinguishable - rare).
- Wrong master that still leaves a plausible positive sky.
- Flat errors, gain drift, RN model issues.

**What it would do on your data (435/509/510)**

- Derive **P ? 24.5 ADU**, predict **? ? 73.6 ADU**, measure **?_dark = 73.65 ADU** - consistent.
- See **R ? 102 ? 4** ? convention **SUM, DERIVED**.
- Check B: **s ? 2399 ADU** - PASS.
- **Silent pass.** No change to calibrated outputs. Stamps `VY_DKRSMP=SUM`, `VY_DKRSMP_SRC=DERIVED`.

**What changed vs v1**

- No auto-MEAN. No config keys. Pedestal from data. Would have caught a **73 ADU** convention error
  that v1's 2% median test could miss; your data do not have that error.

---

## 11. Separate findings (not INV-CAL-01 - record elsewhere)

Implementers must copy these to consumer docs when those tasks run.

### 11.1 Pedestal not in headers (finding 4.1)

QHY294MM: `OFFSET=0.0` in FITS while data carry **~24.5 ADU/bin1 pixel** pedestal.
**SAT-DIAG**, noise model, and any RN/sky algebra must **measure P from dark/bias data**,
not read `OFFSET`. Record in SAT-DIAG spec -4 (pedestal note) and `VYVAR_DECISIONS.md`.

### 11.2 Pedestal-dominated dark at ?10 -C (finding 4.2)

60 s and 120 s masters: **identical median 24.4706 ADU**; **k = 0.00107 ADU/s**.
Exposure-time matching of darks buys **?0.06 ADU over 60 s** - negligible vs sky.
Record in DECISIONS / calibration-library matching guidance; do not require multi-exptime
dark libraries at this temperature for QHY294MM.

---

## 12. Validation (Definition of Done - implementation task)

1. Unit tests: QHY SUM pass (draft 435 numbers), MEAN driver synthetic, CCD inconsistent synthetic,
   bf=1 Check-B-only, single-master fallback, P-indistinguishable ? INDETERMINATE, no-dark PASSTHROUGH.
2. Synthetic: `P=24.5`, `bf=2`, `?_dark=73.65`, `R=102` ? SUM DERIVED PASS.
3. Regression: draft 435 archived cal byte-identical when gate stamps headers only (or document additive headers).
4. `pytest` green; `INV-CAL-01` wired in `invariants_runtime.py` + registry row.
5. No new PARAMS registry keys (grep guard).

### 12.1 Pre-registered predictions (INV-CAL-01)

| ID | Predicate (normative) | Pass criterion |
|----|----------------------|----------------|
| **P1** | Draft 435: gate resolves SUM/DERIVED; pure `(L-D)/F` recalibration matches archive | 150/150 `np.array_equal`; max abs diff **0.0** |
| **P2** | Drafts 509/510: **same processing stage** as archive before pixel compare | Recalibrate pure `(L-D)/F`; if archive has `VY_SKYSF`, apply matching `VYSKYORD` sky surface via `apply_calibrated_stage_for_compare` **before** compare |
| **P3** | Pedestal/convention separation on draft 435 Light_001 | Gate `P`, `Delta_meas`, `Delta_pred`, `R` consistent at **5%** on `Delta`; SUM/DERIVED/PASS |

Harness: `dev/tools/inv_cal01_validate.py`. Stage helper: `cal_diag.apply_calibrated_stage_for_compare`.

**P2 choice (2026-08-13):** apply archived sky order to fresh pure cal. Rationale: archives are the
science products; stage is in `VY_SKYSF`/`VYSKYORD`.

**P3 note:** Spec anchor `P = 24.548 +/- 0.011` uses pixel subsample bootstrap (physics memo). Gate
Check P uses master-median intercept when k ~ 0 (`P ~ 24.471`, `sigma_p ~ 0` degenerate). Both
predict `Delta ~ 73.6 ADU`. Do not treat `+/- 0.011` as gate `sigma_p` without the pixel path.

### 12.2 Known hazard -- mutable `calibrated/` (not fixed by INV-CAL-01)

`calibrated/lights/` is a **two-stage mutable product**: **PURE** `(L-D)/F`, or **SKYSF_N** after
in-place preprocess (`VY_SKYSF`, `VYSKYORD`, `VYVARPR`). `VY_QCBG` / manifest `qc.background` can
describe pre-sky calibrate QC while pixels are sky-subtracted. Gates must check stage first
(`calibrated_stage_from_header`). Cost investigation time twice (P-10 sign error; P2 stage mismatch).
See roadmap **INV-CAL-02**.

---

## 13. Supersedes

- `dev/results/specs/VYVAR_CAL_DIAG_SPEC.md` v1.1 - retained as history; do not implement v1 auto-MEAN.
- CAL-DIAG removal `967f835` - reinstatement requires Milan authorization + this spec.

---

*Discipline: shared helper all calibrate paths; English spec; DECISIONS/JOURNAL/INVARIANTS
updates are a separate docs commit after Milan authorization.*
