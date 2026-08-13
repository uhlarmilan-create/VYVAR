CURSOR RESULT - 2026-08-13 overnight

What I did
Completed read-only audit Waves 1-5 of the full VYVAR workflow audit (supersedes TASK_FULL_AUDIT_2026-08-13). No code changes. Checkpointed register continuously. Ran numerical cross-validation (xval_run) on draft_510 BO CVn.

## Output / findings

### Deliverables written
| Wave | File |
|------|------|
| Register | `docs/VYVAR_AUDIT_2026_REGISTER.md` |
| 1 Map | `docs/VYVAR_AUDIT_2026_MAP.md` |
| 2 Gates | `docs/VYVAR_AUDIT_2026_GATES.md` |
| 3 Contracts | register entries + MAP S2 |
| 4 Science | `docs/VYVAR_AUDIT_2026_SCIENCE.md` |
| 5 Landscape | `docs/VYVAR_AUDIT_2026_LANDSCAPE.md` |

### Key metrics (draft_510)
- xval BO CVn target RMS: VYVAR 0.1454 vs photutils 0.1456 (PASS)
- Check-star scatter: 0.008629 GREEN
- comp_rms max 0.025 (well below 0.05/0.10 gates)
- CAL-DIAG PASS; pedestal_sigma_p=0.0 (OPEN)
- SAT-DIAG: 0 admission rejects; lin DEFAULT_FRAC

### New C-class findings
- C-ALIGN-01: alignment_detection_sigma unused
- C-TRUST-01: apply_color_term misread in trust path
- C-P2P-01: hardcoded 0.10 p2p ceiling
- C-EXPORT-GAP: headless path skips AAVSO export

### Carry-forward unchanged status
A-1 OPEN, WIDE-ERR OPEN, D1-2 DEFERRED, ZP-CLIP/SATURATE_ADU FIXED

## Errors (if any)
None. PowerShell required `;` not `&&` for xval first attempt.

## Files changed
- docs/VYVAR_AUDIT_2026_REGISTER.md (new)
- docs/VYVAR_AUDIT_2026_MAP.md (new)
- docs/VYVAR_AUDIT_2026_GATES.md (new)
- docs/VYVAR_AUDIT_2026_SCIENCE.md (new)
- docs/VYVAR_AUDIT_2026_LANDSCAPE.md (new)
- dev/results/CURSOR_RESULT_audit_2026_waves1_5.md (new)

**Stopped at Wave 6** per task - awaiting Milan review before deletions/code changes.
