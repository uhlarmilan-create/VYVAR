CURSOR RESULT - 2026-08-02 10:15 UTC+2

What I did
Closed four audit items by documentation only (batch A): A-1, A-9, D1-1, U-09; created
`docs/VYVAR_LIMITATIONS.md` for the methods-paper limitations section. No code or re-cut.

## A.1 - A-1 -> DOCUMENTED

**State:** A-1b **CONFIRMED**, **DOCUMENTED** (register item 16 / aperture closure section).

Text written (register):

> **A-1b CONFIRMED, DOCUMENTED.** The SNR aperture table assigns magnitude-dependent radii
> (`aperture_r_px`, clamp 1.916 px at the faint end) with **no curve-of-growth correction**, so
> the enclosed-flux fraction differs between target and comparison stars and does not fully cancel
> in the differential. Verdict robust (all 5x3 proxy cells > 10 mmag gate, Step 1d). Exact
> consolidated magnitude was not stabilised because the measurement apparatus, not the physics,
> blocked it across Steps 1e-1n; the physics expectation from
> `dev/tools/closure_a1_reference_fixture.py` is ~144 mmag for G 8-9 comparisons over the anchor
> r50 span. **Recommended fix:** enable `cog_aperture_correction_enabled` (option iii, Stetson
> 1990 growth-curve aperture correction), which normalises all stars to a common enclosed-flux
> scale and removes the mechanism directly. This is deferred to the numeric-change batch (D), not
> applied here.

Sub-findings recorded: S1 DEAD (`aperture_snr_sizing`); S3 role factors label-only; D5-1 Q1
per-frame FWHM No.

## A.2 - A-9 -> DOCUMENTED

**State:** register item 31 **DOCUMENTED**.

Text: FWHM estimators disagree (2.395 / 3.207 / 4.0-4.9 px); not blocking differential results;
required before absolute claims; use COG r50 as scale proxy.

## A.3 - D1-1 / CR-1 -> DOCUMENTED

**State:** CR-1 remains **QUEUED**; documented statement added to register and LIMITATIONS.

Text: no CR rejection in `src_py`; L.A.Cosmic or astroscrappy scheduled in batch E.

## A.4 - U-09 -> MEASURED + DOCUMENTED

**State:** register item 26 updated. BO CVn shutter-open verified; other rigs and QHY294
UNVERIFIED.

## A.5 - VYVAR_LIMITATIONS.md

**Created:** `docs/VYVAR_LIMITATIONS.md` -- one paragraph each for A-1, A-9, D1-1, D1-2, D5-2,
I-12, D11-1, D12-1, U-09, MASTERSTAR TODOs. D1-2 and D5-2 mechanism paragraphs reference batch B
(outcome B-open).

## Files changed

- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md`
- `docs/VYVAR_AUDIT_FINAL.md`
- `docs/VYVAR_PARAMS.md`
- `docs/VYVAR_LIMITATIONS.md` (new)
- `docs/VYVAR_STATE.md`
- `docs/VYVAR_ROADMAP.md`
- `dev/results/CURSOR_RESULT_batch_A.md` (this file)
