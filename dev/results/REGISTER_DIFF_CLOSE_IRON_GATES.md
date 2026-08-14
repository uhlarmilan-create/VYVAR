# Register diff for authorization -- CLOSE-IRON-GATES (2026-08-14)

Base: `4a3e855`. Proposed append to `docs/VYVAR_AUDIT_2026_REGISTER.md` (Wave 8).

| Register ID | wave | stage | class | severity | summary | disposition | status |
|-------------|------|-------|-------|----------|---------|-------------|--------|
| **ENC-STALE-01** | 8 | process | P | LOW | `--fast` PASS was carried across commit that introduced 7 non-ASCII docs; guard correctly failed. Fixed via `ascii_migrate.py`; PROCESS.md now requires commit SHA on gate results. | **FIXED** | CLOSED |
| **IRON-GATES-01** | 8 | invariants | P | HIGH | Iron rules (no-clip, no-CR, pixels, master combine, comp membership) were policy-only. Wired as static AST/grep gates in `dev/tools/iron_gates_scan.py` + `dev/tests/test_iron_gates.py`; rows in `VYVAR_INVARIANTS.md`; IDs in `WIRED_INV_IDS`. | **FIXED** | CLOSED (gates live; INV-PIXELS-01 review item open) |
| **SKY-CLIP-01** | 8 | photometry | P | HIGH | One-sided 2-sigma upper annulus sky clip in batch path; plain median in single-star path. Replaced with unified plain median (`_sky_pp_from_annulus_image`). Draft 510 FITS recomputation: median fractional flux change -0.058% (-0.027% target). | **FIXED** (code); **PENDING** (510 anchor re-cut) | PARTIAL |

## Milan decisions requested

| Item | question | alternatives |
|------|----------|--------------|
| INV-PIXELS-01 / nanmedian fill | `photometry_core.py:2678,12235,12470`, `psf_photometry.py:1993` replace non-finite pixels with frame `nanmedian` before photometry. Violates literal pixel-immutability? | (A) propagate NaN and flag measurement; (B) exclude star/frame; (C) keep fill (IRAF fills; photutils often errors on NaN). **No change in this task.** |
| Draft 510 anchor re-cut | Full photometry re-export + new checksum manifest after SKY-CLIP-01. | Authorize physical re-cut per Wave 7 procedure; manifest label `anchor_510_checksums_sky_median_20260814.json`. |

## Retractions

- **Q1-XVAL-MATCHED arm P1** (clip vs median offset): superseded by SKY-CLIP-01 fix; see `dev/results/CURSOR_RESULT_CLOSE_IRON_GATES.md`.
