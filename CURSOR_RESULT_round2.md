CURSOR RESULT — Round 2 (2026-06-17)

# Round 2 — diagnostic-first, default-OFF flags, isolated measurement (draft_413 g)

Both Round-2 items were diagnosed before any code. B.1 was **refuted by its own decisive test**
and is **not implemented**. B.2 was **confirmed** and ships behind a **default-OFF** flag
(`frame_quality_gate_enabled`) with config / Settings-UI / `VYVAR_PARAMS.md` parity and a gated
citation. **Science-code ? present for Milan's approval before push.** Part A (clean committed
baseline) was completed first (6 commits, tree clean, 44 tests).

Baseline = HEAD after Part A. All measurements reuse existing `draft_413` g artifacts
(`detrended_aligned/lights/g_60_2`, 161 frames: 59 pre-flip `V454CrA_*`, 102 post-flip `V454CrAR_*`).

---

## B.1 Aperture-skirt — DIAGNOSTIC REFUTES THE FIX (not implemented)

**Premise (Round 1):** the SNR-optimal science aperture (`aperture_r_px ? 5 px` for the median
comp; 7.27 px for V0454) is sized to the sharp core but under-captures the defocused donut skirt;
the out-of-aperture skirt fraction may vary frame-to-frame ? scatter.

**Diagnostic** (`tmp/b1_cog_diag.py`; 12 bright, isolated, unsaturated pre-flip g stars, 30 frames;
photutils curve-of-growth + differential scatter vs radius):

| radius | encircled-energy (COG) | differential scatter |
|-------:|-----------------------:|---------------------:|
| 5 px (production) | **0.65** | 24.1 mmag |
| 6 px | 0.76 | **21.5 mmag** (min) |
| 7.3 px | 0.85 | 26.9 mmag |
| 18 px (EE?0.99 plateau) | 0.99 | 26.7 mmag |
| 22 px | 1.00 | 26.4 mmag |

- COG confirms the premise: the 5 px aperture captures only **65%** of the encircled energy.
- **Decisive test fails:** differential frame-to-frame scatter is **flat** from r=5 px to the
  plateau (24 ? 27 mmag; the minimum, ~21 mmag, is at r?6 px and within noise). Widening to the
  skirt plateau does **not** reduce scatter — it slightly worsens it (added sky noise).
- A per-frame **FWHM-adaptive** aperture (the grounded "defocus-aware" alternative) is **worse**
  (30–32 mmag): scaling by a noisy per-frame FWHM injects scatter.
- **Why:** the skirt fraction swings ±5–7% frame-to-frame (nominal ~48 mmag), but that is
  **common-mode PSF breathing** which differential photometry already cancels. The ~24 mmag
  bright-star floor is not aperture-limited.

**Decision:** do **not** implement an aperture-skirt fix. A 5?6 px bump is marginal (~2–3 mmag,
within noise) and not worth a science change. Figure: `docs/round2_figs/b1_aperture_skirt.png`.

---

## B.2 Transparency / frame-quality gate — CONFIRMED, implemented behind a default-OFF flag

**Premise (Round 1):** post-flip dawn frames passed FWHM/elong QC but transparency collapsed,
dragging night-wide trust to RED and burying the GREEN-quality pre-flip half.

**Diagnostic** (`tmp/b2_transparency_diag.py`, `tmp/b_figs.py`; all 161 g frames, in time order):

- Two regimes (`docs/round2_figs/b2_transparency.png`):
  1. **Gradual transparency decline** through the night (large-aperture zeropoint `zpL` drifts
     24.1 ? 21.5 mag; flux falls *equally* in the small and large aperture, so PSF concentration
     `flux_large/flux ? 2.7` stays normal). These are **clear-but-faint** frames — must NOT reject.
  2. **Catastrophic collapse** — 13 post-flip frames with `flux_large/flux ? 11–16` (vs good ~2.7),
     FWHM pegged at the ~8.62 px measurement rail, and the 5 px science aperture catching only
     noise (`flux` collapses; `zp` 17.4 vs ~21 normal).
- Two independent signals **agree exactly**: `flux_large/flux ? 6` and `FWHM ? 8.5 px` each flag the
  **identical 13 collapsed frames**, with a wide gap to the good population (next ratio 5.0, next
  FWHM 8.16). All 13 are post-flip. The gradual-decline frames are correctly spared.

**Design (grounded):** a per-frame **PSF-concentration** statistic — the median of
`flux_large / flux` over bright, unsaturated sources — is a robust frame-quality discriminator: a
collapsed/blurred PSF pushes flux out of the SNR-optimal aperture so the large/small ratio spikes
(encircled-energy / curve-of-growth framework, **Howell 1989, PASP 101, 616**), while a
clear-but-faint frame keeps a normal ratio. The gate rejects frames whose ratio is a robust outlier
(`z = (ratio ? median)/(1.4826·MAD) > k`, primary) guarded by `FWHM ? factor·median-FWHM` so a
spurious ratio outlier on a sharp frame is spared. Targets the transparency collapse specifically,
not merely-faint frames.

**Default-OFF flag (config + UI + PARAMS + gated citation):**

| param | default | clamp | role |
|-------|---------|-------|------|
| `frame_quality_gate_enabled` | **False** | — | master toggle; OFF ? baseline byte-identical |
| `frame_quality_ratio_k` | 5.0 | 2.0–20.0 | robust z-cut on per-frame `flux_large/flux` (primary) |
| `frame_quality_fwhm_factor` | 1.0 | 0.8–3.0 | guard: reject only if FWHM ? factor·median (spares sharp) |
| `frame_quality_min_keep_frames` | 10 | 3–100000 | safety floor: skip gate if it would keep fewer |

- `config.py`: dataclass fields + `__post_init__` load/clamp + `to_json()`.
- `ui_settings.py`: toggle + 3 sliders under Photometry ? Data quality & validation (+ save block).
- `docs/VYVAR_PARAMS.md`: 4 rows (exposed = yes).
- `photometry_core.py`: `_frame_quality_gate_select()` + hook in `_phase2a_prepare_shared_state`
  (filters `csv_files` before the flux matrix ? `n_frames` / summary counts stay consistent).
  Default OFF returns the input unchanged (no reads, no logging) ? byte-identical.
- `citations.py` + `CITATIONS.bib`: `howell1989` gated to `frame_quality_gate_enabled` under the
  DATA-QUALITY GATE section.
- Scope: applied at Phase 2A (target LC + trust). Phase-0+1 comp selection is **not** gated yet
  (future extension). Verified: gate ON rejects exactly the 13 collapsed frames; OFF is a no-op.

**Isolated measurement (flag ON vs OFF, full Phase 0+1+2A re-run on draft_413 g; `tmp/b2_measure.py`):**

- Frames: OFF = all 161; ON = 148 (13 collapsed rejected). Targets: 67 in both (gate drops frames,
  not targets).
- **Light-curve scatter (`lc_rms`) drops sharply** — median **?257 mmag**, **14/15** bright targets
  improved:

| target | mag | lc_rms OFF | lc_rms ON | ? |
|--------|----:|-----------:|----------:|---:|
| V0454 CrA | 9.7 | 0.404 | 0.147 | ?257 mmag (residual = real variability) |
| KM CrA | 11.4 | 0.294 | 0.061 | ?232 mmag |
| ASAS J175832-3945.1 (flat field star) | 12.7 | 0.342 | **0.035** | ?307 mmag (?10×) |
| Gaia DR3 4035673… | 12.0 | 0.321 | 0.045 | ?276 mmag |
| ASASSN-V J175929 | 13.1 | 0.489 | 0.098 | ?391 mmag |

  Figure: `docs/round2_figs/b2_lc_beforeafter.png` — the collapsed-frame points are wild outliers
  (mag 8–12.6) that survive existing per-epoch sigma-clipping; the whole-frame gate removes them
  a priori.
- **Trust unchanged:** all 67 targets RED in BOTH runs (no transitions). RED here is set by the
  structural check-star / thin-comp / colour-term-off gates, **not** by LC scatter — consistent
  with the Round-1 pre-flip-demo finding. The gate is a large precision win but does not by itself
  flip trust GREEN; the thin-comp?RED chain still needs the comp-side work.

---

## Combined effect

B.1 is not implemented (decisive-negative), so the combined Round-2 effect = **B.2 alone**:
large LC-scatter reduction for bright targets, trust still gated RED by the comp/check-star chain.
No combined run was needed beyond the B.2 isolated measurement.

## Errors

None. Default path byte-identical (gate OFF), full suite green.

## Files changed (uncommitted; push gated on Milan)

- `config.py` — 4 frame-quality params (default OFF) + load/clamp + `to_json`.
- `photometry_core.py` — `_frame_quality_gate_select()` + Phase-2A hook.
- `ui_settings.py` — Settings UI toggle + 3 sliders + save.
- `citations.py`, `CITATIONS.bib` — `howell1989` gated to the gate flag.
- `docs/VYVAR_PARAMS.md` — 4 param rows.
- `tests/test_frame_quality_gate.py` — 5 unit tests (no-op OFF, rejects collapse, spares faint,
  spares sharp outlier, safety floor).
- `docs/round2_figs/*.png` — B.1 COG/scatter, B.2 transparency, B.2 before/after LC.
- Tests: 339 passed, 15 skipped (incl. 5 new gate tests).

## Verification

- COG/scatter: `tmp/b1_cog_diag.py`. Transparency: `tmp/b2_transparency_diag.py`, `tmp/b_figs.py`.
- Gate measurement: `tmp/b2_measure.py` (? `tmp/b2_off`, `tmp/b2_on`). LC figure: `tmp/b2_lc_fig.py`.
- Default-OFF byte-identical preserved (the end-of-day UI test runs the default path).
