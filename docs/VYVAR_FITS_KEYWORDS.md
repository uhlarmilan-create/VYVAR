# VYVAR FITS keywords - renamed to the FITS standard (FITS-KEYS-2026-10)

A FITS keyword has at most 8 characters (FITS standard 4.0, section 4.1.2.1). VYVAR wrote 14
keywords with longer names; astropy stored them as HIERARCH cards (an ESO convention, not part
of the standard) and warned on every frame. Milan 2026-10-26: rename, keep the standard.

No aliases: the old names are not read any more. Drafts processed before this change are
processed again.

| old name (HIERARCH) | new name | written by | meaning |
|---|---|---|---|
| VY_SKYPASS | VYSKYPAS | cal_stage | sky-surface subtraction passes on these pixels |
| VY_CALDATASUM | VYCALSUM | cal_stage | FITS DATASUM of the primary data array (INV-CAL-02) |
| VY_CALSTAGE | VYCALSTG | cal_stage | calibrated product stage (INV-CAL-02) |
| VY_CDSTAT | VYCDSTAT | cal_diag | CAL-DIAG gate outcome of the setup |
| VY_DKRSMP | VYDKRSMP | cal_diag | dark resample convention applied (SUM / MEAN / PASSTHROUGH) |
| VY_DKRSMP_SRC | VYDKRSRC | cal_diag | how the resample convention was established |
| VY_RDNOIS | VYRDNOIS | osc_extract | effective read noise after OSC extraction + binning [e-] |
| VY_CHANNEL | VYCHANNL | osc_extract | OSC extracted channel (OSC-01) |
| VY_BAYERMASK | VYBAYMSK | osc_extract | Bayer mask used for the extraction |
| VY_OSC_BIN | VYOSCBIN | osc_extract | binning after extraction |
| VY_OSCSRC | VYOSCSRC | osc_extract | source CFA mosaic file |
| VY_OSCWCS | VYOSCWCS | osc_align | WCS propagated from the oneRGGB MASTERSTAR |
| VY_OSCREG | VYOSCREG | osc_align | registration reused from oneRGGB |
| VY_FWHM_GAUSS | (removed) | MASTERSTAR | was a copy of VY_FWHM (FWHM-ONE-2026-10); readers use VY_FWHM |

Old names that were only read, never written by the current code, are no longer read either:
VY_FWHM_GAUSSIAN, VY_NSTARS, VY_NDETECT, VY_PLATESCALE, VY_REFFILE, VY_MSFRAME, VY_SRCFILE,
VY_SRCFRAME, VY_SRCPATH, VY_TARGRA, VY_TARGDEC (the written names are VY_FWHM, VY_NSTAR, VY_NDET,
VY_PLTS, VY_REF, VY_FRAME, VYTARGRA, VYTARGDE). The clean-up of the L.A.Cosmic keywords
VY_COSM / VY_COSMNPX (removed 2026-08-12) is gone.

Contract: dev/tests/test_fits_keywords.py (no VYVAR keyword longer than 8 characters, no header
write of a longer keyword, none of the old names in the code).
