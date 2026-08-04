# MASTERSTAR reference architecture

**Status:** NOT STARTED. Does not block the current anchor re-cut arc.

**Source:** Science Audit Tranches 3/4 + design session 2026-07-30.

**Related:** `dev/results/CURSOR_RESULT_audit_t3.md`, `dev/results/CURSOR_RESULT_audit_t4.md`,
`dev/results/CURSOR_RESULT_dao_sigma_stability.md`, `dev/results/CURSOR_RESULT_dao_only_verify.md`.

---

## Background

`build_masterstar_from_detrended` currently copies the single lowest-`VY_FWHM` frame
(`shutil.copy2`). Detection, astrometry and catalogue construction therefore rest on 1 of ~139
frames, and the DAO threshold inherits that frame's sky level, dither offset and seeing.

Measured consequences (draft_435, DAO-SIGMA-STABILITY): sigma spread **15.6%** across candidate
frames; pass-1 DAO count spread **~21%**.

**Literature:** deriving the master star list from a **combined** image is the standard
DAOPHOT/ALLFRAME architecture (Stetson 1994), used unchanged across the literature. Photometry
on individual frames with a stack-derived master list is exactly VYVAR's design - only the
"single frame" part is non-standard.

Current implementation: `src_py/pipeline.py` - `build_masterstar_from_detrended` ranks by
`VY_FWHM` and copies the best file.

---

## TODO-A - MASTERSTAR from a stack of the best N frames

**Priority: HIGH.** Cheap, literature-standard, removes the best-frame lottery.

### A1 - Frame selection metric

Rank frames by point-source information content:

```
I_j = F_j^2 / (sigma_j^2 * FWHM_j^2)
```

- `F_j` - transparency / photometric zero point
- `sigma_j` - background noise of the frame
- `FWHM_j` - seeing

Do **not** rank on FWHM alone. Measured on draft_435: FWHM-only selection systematically
prefers TWILIGHT frames, which are the noisiest of the night. Frames 001-005 (Sun altitude
-12.8 to -13.4 deg) have sigma_pp 45-52 ADU; frames 050+ (full dark) have 28-32 ADU. Sky
brightness is a larger lever on detection depth (~**1.7x**) than either resampling (~1.39x) or
dither (~1.15x).

### A2 - Selection rule

```
N_max = 20        (knee of the depth curve; see A3)
N_min = 10        (floor for robust median rejection)
quality gate: keep frames with I_j >= 0.5 * max(I_j)

if n_available < N_min: use all frames, and mark the reference SHALLOW in provenance
else:                   use up to N_max frames passing the quality gate,
                        never fewer than N_min
```

**Rationale for N_max = 20:** depth gain is 1.25*log??(N) mag.

| N | depth gain |
|---|------------|
| 5 | +0.87 mag |
| 10 | +1.25 mag |
| 20 | +1.63 mag |
| 30 | +1.85 mag |
| 139 | +2.68 mag |

Going 20 ? 139 buys 1.05 mag for 119 extra frames. Going 10 ? 20 buys 0.38 mag for 10. The
knee is 20-30.

**Rationale for N_min = 10:** median combination is the reason a stack rejects cosmic rays for
free. At N=5 a single hit contaminates 20% of the sample and sigma-clipping is ineffective.
VYVAR has **no** cosmic-ray rejection step (verified: `cosmic` occurrences in `src_py` are
gain/read-noise parameters only), so the stack **is** the CR defence.

**Secondary benefit of a cap:** PSF homogeneity. Not mixing 2? and 5? seeing gives a more
compact, better-defined stack PSF, which helps astrometry and ePSF neighbour subtraction. This
argument is independent of depth.

### A3 - Combination method

Median or sigma-clipped mean. Robust by design - this is why the classical pipelines use it.
Do not use a plain linear sum at this stage (see TODO-B for why the optimal linear method needs
CR rejection first).

### A4 - Provenance (mandatory)

- Write the exact list of frames used into the reference header and `pipeline_meta.json`.
- Deterministic tie-breaking, so two runs on identical inputs select the same N frames.
- Record `I_j` per candidate frame and the applied threshold.
- Without this the reference is not reproducible and we are back to the problem this task exists
  to solve.

### A5 - Recalibration required after A

`masterstar_dao_threshold_sigma` must be re-derived against the stack, not carried over. The
stack's noise, PSF and depth all differ from a single frame. Calibrate against a measured
false-positive rate, not against a target count.

### A6 - Consequence: `DAO_ONLY` fraction stops being a purity metric

Local Gaia DB cap is `g_mag = 17.5`. A 20-frame stack reaches ~16.6; a 139-frame stack reaches
~17.6. Detections beyond the catalogue cap become **legitimate** `DAO_ONLY` rows - real stars
we simply do not have catalogued.

Today `DAO_ONLY` means "probably spurious". After A it means a mixture. The metric must be split
by estimated magnitude relative to the 17.5 cap - the analysis already done once by hand in
`CURSOR_RESULT_dao_only_verify.md` (2235 rows below G=16 = spurious; 26 rows beyond 17.5 =
possibly real) becomes a permanent part of the metric rather than one-off forensics.

Update any guard that uses `DAO_ONLY_fraction` as a health signal.

---

## TODO-C - Separate the admission gate from the detection threshold

**Priority: HIGH.** Arguably more important than A, and independent of it.

One threshold currently answers two different questions:

| Question | Role |
|----------|------|
| **A:** Which stars exist in the field? | Catalogue, astrometry, crowding, blending |
| **B:** Which stars are worth photometering? | Target admission |

Because both are driven by the same DAO threshold, tuning it moves catalogue depth and
light-curve quality together. This is why the 2.1 ? 3.8 recalibration was so awkward.

### C1 - Why the ALLFRAME depth argument does NOT transfer to VYVAR

ALLFRAME can afford a very deep master list because it fits **all** frames **simultaneously**:
a star at S/N ~0.5 per frame reaches S/N ~6 in the joint solution over 139 frames. Depth is
recovered by combining epochs.

VYVAR produces **time series**. Combining epochs is precisely what must not be done - the
epochs **are** the signal. **The useful depth for time-series photometry is set by a single
frame, not by the stack.** This is a fundamental architectural difference, not a detail, and it
belongs in the methods paper as the justification for a separate admission gate.

### C2 - Proposed admission gate

Admit a target on **predicted per-epoch SNR**, not on detection in the reference.

All inputs already exist: `g_lim_50` / `g_lim_90` are computed, and the noise model (Labb-
empty-aperture `sigma_bkg_ap`) is implemented and audited. For a given magnitude, typical sky
and seeing of the night, compute expected SNR per epoch and threshold on that.

Physically grounded, measurable, and independent of how deep the reference goes.

### C3 - Expected effect (verifiable prediction)

The 82 spurious Group-B actives in draft_451 (G 14.6-15.3, median RMS 0.30 mag, 82% RED/noisy)
sit below the single-frame limit and would be rejected by C2 regardless of the DAO threshold.
Draft_451's own scatter table shows the knee where curves become scientifically useless at
G ~ 14-15.

### C4 - Keep the deep catalogue, flag its rows

A deep catalogue remains valuable even for stars never photometered:

- astrometry / WCS / SIP fitting (more stars = better solution)
- crowding and dilution (a faint neighbour inside the target aperture must be known even if never
  measured)
- ePSF build-star neighbour subtraction
- blending flags during comparison-star selection

So: keep deep rows, mark them explicitly as **CONTEXT-ONLY** vs **PHOTOMETRY-CANDIDATE**.

---

## TODO-B - Proper coaddition (Zackay & Ofek 2017, ApJ 836, 188)

**Priority: MEDIUM.** Optimal version of A. Multi-session project.

Do **not** start before A and the prerequisites below.

### B1 - Method (equations verified against the paper)

Coadd (Eq. 7):

```
R_hat = SUM_j [ (F_j / sigma_j^2) * conj(P_hat_j) * M_hat_j ]
        / sqrt( SUM_j [ (F_j^2 / sigma_j^2) * |P_hat_j|^2 ] )
```

PSF of the coadd (Eq. 10):

```
P_hat_R = sqrt( SUM_j (F_j^2/sigma_j^2) |P_hat_j|^2 ) / F_R
F_R     = sqrt( SUM_j F_j^2/sigma_j^2 )
```

where `M_j` is the **background-subtracted** frame j, `P_j` its PSF, `sigma_j^2` its noise
variance, `F_j` its transparency (photometric zero point).

### B2 - The payoff: sigma_R = 1 (Eq. 11)

The coadd's noise has standard deviation exactly 1 by construction. The DAO threshold becomes
literally N - no noise estimator at all. This dissolves the entire sigma_pp / Background2D /
twilight / dither problem in one step, and the paper confirms the detection statistic of Paper I
is reproduced by matched filtering R with its own PSF, so DAOStarFinder runs on R unchanged.

### B3 - Prerequisites VYVAR does not currently meet

| # | Requirement | Current state |
|---|-------------|---------------|
| 1 | Input noise must be **uncorrelated** | astroalign resampling correlates it. The paper's decorrelation property is conditional on uncorrelated inputs - it does **not** remove pre-existing correlation. Must coadd pre-alignment frames, or use a registration that preserves noise (Fourier phase-ramp shift is a candidate - **verify**, do not assume) |
| 2 | Artifact-free inputs | Method is linear, **not** robust. No median/sigma-clip. Cosmic rays and bad pixels must be removed first (van Dokkum 2001 / L.A.Cosmic). VYVAR has **no** CR rejection today |
| 3 | Per-frame PSF `P_j` | ePSF exists but is gated/optional. **Note:** ePSF is a **prerequisite** for coaddition, not a beneficiary of it |
| 4 | Per-frame background and variance | Partially available; must be local, not global |
| 5 | `F_j` from PSF photometry | Paper is explicit: aperture-based zero points make `F_j` seeing-dependent. VYVAR's primary path is aperture |
| 6 | Background-dominated noise limit | Does not hold for bright targets (BO CVn V~9.5). Acceptable because MASTERSTAR is detection-only, but must be stated as a scope limit in the paper |

### B4 - Honest expected gain

The paper reports a few percent to 25% improvement in survey speed over weighted coaddition
schemes (Annis et al. 2014; Jiang et al. 2014). That is the gain of B over A - modest.

The large gain is **A itself**: going from 1 frame to N frames. Do not justify B by the numbers
that belong to A.

### B5 - Implementation notes from the paper

- Compute `P_R` from Eq. 10; do **not** measure it from R. The authors tested both and the
  measured route is significantly worse. Store `P_R`.
- Proper coaddition finds more real sources **and** more false detections than weighted schemes
  (deeper). Threshold recalibration required again.
- Only FFT and simple operators; numerically stable, no division by small numbers, unlike
  deconvolution.

### B6 - Prerequisite task worth doing regardless

Cosmic-ray rejection is missing from VYVAR entirely. It is needed for B, it is good hygiene for
A, and it is a genuine gap today. Consider promoting it to its own task independent of this arc
(**CR-REJECTION** on the roadmap).

---

## Suggested order

1. **TODO-C** (admission gate) - independent, high value, unblocks threshold tuning
2. **CR-REJECTION** - standalone gap
3. **TODO-A** (stack reference) - standard, cheap, big depth win
4. **TODO-B** (proper coaddition) - optimal version, once 1-3 are in place

---

## Detection noise on resampled frames (Tranche 4 cross-link)

Stack-based reference (TODO-A) does **not** by itself fix correlated noise after astroalign
resampling on aligned detection frames. Options documented in Tranche 4 remain relevant until
detection operates on uncorrelated pixels or thresholds the convolved quantity directly
(`scale_threshold=False` + convolved RMS, or detect pre-align). See `CURSOR_RESULT_audit_t4.md`.

---

## Citations

| Key | Reference |
|-----|-----------|
| `stetson1987` | Stetson (1987) - DAOPHOT/FIND threshold convention |
| `stetson1994` | Stetson (1994) - ALLFRAME, master list from combined image |
| `zackay2017detection` | Zackay & Ofek (2017) ApJ 836, 187 - optimal coaddition for detection |
| `zackay2017proper` | Zackay & Ofek (2017) ApJ 836, 188 - proper coaddition |
| `vandokkum2001` | van Dokkum (2001) - L.A.Cosmic (if CR rejection lands) |
| `annis2014` | Annis et al. (2014) - weighted coaddition (comparison baseline) |
| `fruchter2002`, `casertano2000` | Correlated noise after resampling (Tranche 4) |
| `bertin1996` | SExtractor (cited elsewhere) |

All entries in `CITATIONS.bib`.
