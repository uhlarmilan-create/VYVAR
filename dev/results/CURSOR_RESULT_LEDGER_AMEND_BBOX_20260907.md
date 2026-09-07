CURSOR RESULT - 2026-09-07 LEDGER-AMEND-BBOX-20260907

What I did
Appended the safe-bbox VY_FWHM carve-out to D-FWHM-AUTH-01-CLOSE
(same entry, no new heading). Docs-only.

Cited locations verified, line numbers unchanged:
- src_py/pipeline_astrometry.py:1957-1959 still states Variant A2
  FWHM source stays MASTERSTAR VY_FWHM (call at :1965
  fwhm_from_header_vy_fwhm).
- session_20260831_c01b/REPORT.md A2 STOP: bbox r_out MS header
  27.01218 vs night-med 26.99701 (0.015 px bookkeeping).

No refutation.

--fast --clean: OVERALL PASS (1621 passed, 32 skipped; clean-tree PASS).
HEAD at gate start: 2e5ce31. Branch: consolidate-01.
No --full, no --full-epsf.

Files: docs/VYVAR_DECISIONS.md;
dev/results/CURSOR_RESULT_LEDGER_AMEND_BBOX_20260907.md
