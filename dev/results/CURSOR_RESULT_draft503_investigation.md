CURSOR RESULT - 2026-08-11

What I did
Investigated draft_503 regression evidence and astrometry warnings. Live re-run blocked:
draft_000503 not present under Archive/ on this machine.

## Output / findings

### Expected after FIX 1 (Milan to verify live)
- Infolog should show `[BORDER] Safe bbox (shrunk by r_out=...)` after alignment, not
  "Deferred: no aligned proc_*.fits ... border filter skipped"
- Edge stars excluded from comparison_stars.csv comp pool
- comp_rms back toward ~0.01 (not 0.26-0.54)
- TRUST flags should improve once garbage edge comps are gone

### draft_503 live re-run
- **BLOCKED**: `Archive/Drafts/draft_000503/` not found locally
- Milan must re-run BO CVn on draft_503 after pulling commits 30ad516 + zone fix

### Astrometry warnings (report only, no code change)

**1. Mirrored RA/Dec parity** (`astrometry_optimizer.py:837`)
- Log: "mirrored RA/Dec matches markedly better (possible WCS parity issue)"
- Emitted when mirrored coordinate match score beats direct match by a large margin
- Indicates possible WCS axis flip/parity mismatch between reference and solved WCS
- **Separate bug** if confirmed on draft_503; do not patch without Milan approval

**2. Post-match identity gate fail=42** (`astrometry_optimizer.py:1226`)
- Counts Gaia sources whose post-solve pixel position disagrees with expected after match
- fail=42 on a wide field (~9.55 arcsec/px) may reflect parity issue above or genuine
  astrometry offset; correlate with `[BORDER]` skip first (primary garbage-LC cause)

## Errors (if any)
None (investigation only).

## Files changed
- dev/results/CURSOR_RESULT_draft503_investigation.md (this file)
