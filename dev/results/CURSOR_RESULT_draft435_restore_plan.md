# Draft 435 restore plan (PROPOSAL ONLY - do not execute)

Generated: 2026-08-12T14:57:34

## Reference

- Whole-tree zip: `C:\ASTRO\backups\draft_000435_anchor_live_20260716.zip` (2026-07-16, 1932 members)
- Live tree: `C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000435` (2778 files)
- Snapshot sibling: `Archive/Drafts/draft_000435_snapshot_skysurface_20260716`
- Immediate LC backup: `tmp/_435_lc_before_zpclip_rm.csv` (BO CVn only)

## Summary counts

| bucket | count |
|---|---:|
| hash match (zip member, unchanged) | 1381 |
| hash changed vs zip | 551 |
| current-only (no zip member) | 846 |
| zip-only (missing from current) | 0 |
| changed with mtime 2026-08-12 | 549 |

## B.1 Current-only files (846) by group

- **report_cache_jpg**: 838
- **photometry_png**: 2
- **lightcurve_csv**: 2
- **lightcurve_sidecar_json**: 2
- **export_reports**: 1
- **platesolve_catalog**: 1

### B.1 group notes (what produced them)

- **report_cache_jpg**: `photometry/_report_cache/*.jpg` field preview JPGs for Streamlit/report. Post-July artifact; safe to delete or regenerate.
- **photometry_png**: See file list.
- **lightcurve_csv**: See file list.
- **lightcurve_sidecar_json**: See file list.
- **export_reports**: AAVSO/VarAstro text exports if present only in current tree.
- **platesolve_catalog**: See file list.

## B.2 Changed files (551) by group

- **lightcurve_sidecar_json**: 238
- **export_reports**: 184
- **lightcurve_csv**: 79
- **photometry_png**: 41
- **photometry_csv**: 6
- **draft_manifest**: 1
- **hrd_cache**: 1
- **pipeline_meta**: 1

### B.2 verdict per group

- **draft_manifest**: **Undecided**: Aug-11 rewrite; compare DB parity intent. Not from Aug-12 re-run.
- **hrd_cache**: **Keep current** if from July 17 intentional regen; else zip.
- **photometry_csv**: **Restore from zip** (active_targets, comparison_stars_per_target, summaries) - **critical for --full inputs**.
- **photometry_png**: **Restore from zip** (field maps, LC PNGs).
- **lightcurve_csv**: **Restore from zip** for all except BO CVn target LC already restored from `tmp/_435_lc_before_zpclip_rm.csv` (hash-match to zip). Aug-12 mtime = killed mistaken re-run.
- **lightcurve_sidecar_json**: **Restore from zip** (trust/comp_qa/comp_quality/check_kmag sidecars). Aug-12 re-run outputs.
- **export_reports**: **Restore from zip** (AAVSO/VarAstro batch from July anchor).
- **pipeline_meta**: **Restore from zip** unless Milan wants Aug-12 stage stamps; science SHA not contract for meta alone.

## B.3 Restore plan (actions - NOT EXECUTED)

### Action RESTORE-ZIP (549 Aug-12 photometry outputs + sidecars)

Restore from `draft_000435_anchor_live_20260716.zip` all members under:
- `platesolve/NoFilter_60_2/photometry/` except where KEEP below
- Includes: lightcurve CSV/JSON, trust, comp QA, AAVSO exports, summaries, `pipeline_meta.json`, field maps

**Reason:** Aug-12 mistaken photometry re-run (killed); only BO CVn LC manually restored.

### Action KEEP-CURRENT

- BO CVn `lightcurve_1498613634033133184.csv` (already matches zip + immediate backup)
- All calibrated/FITS/raw (already hash-match zip - no action)
- `draft_000435_snapshot_skysurface_20260716` tree (separate; do not touch via live restore)

### Action KEEP-CURRENT (post-July legitimate cache - 846 current-only)

- `photometry/pdf_embed/**` (337 files) - report embed cache; regenerate if missing
- `photometry/_report_cache/**` (~40 JPG) - UI preview cache
- Other current-only derived caches with Aug mtime: **keep or delete**, not zip-restore (no zip member)

### Action UNDECIDED

- `draft_manifest.json` (Aug-11 hash change; not Aug-12 re-run)
- `platesolve/NoFilter_60_2/_hrd_cache/summary.json` (Jul-17 mtime)

### Action DELETE (optional cleanup after Milan approval)

- None required for anchor integrity; optional prune of duplicate report caches if restored from zip overlaps.

## B.4 Baseline / anchor dependency on changed files

**Partial.** `--fast` manifest-db-parity uses **snapshot** `draft_000435_snapshot_skysurface_20260716`, not live photometry outputs - **551 changed LC/trust files do not affect --fast PASS.**

`--full` uses live draft **inputs**: MASTERSTAR, variable_targets, masterstars_full_match, detrended_aligned/lights. Check changed list for those paths.

Changed photometry **outputs** (LC, trust, AAVSO) are **not** `--full` gate inputs; they affect anchor **reference display** and manual validation only.

## B.5 Protection proposal (no implementation)

1. **Anchor checksum manifest** written at every verified-good state:
   - Path: `Archive/Drafts/draft_000435/anchor_checksums.json` (or sibling `dev/validation/anchor_435_checksums.json` if draft tree must stay pure outputs)
   - Content: `{version, git_head, created_utc, algorithm: sha256, files: {relative_path: {sha256, size, mtime_utc}}}`
   - Scope: entire draft tree OR minimum contract: calibrated+detrended+platesolve inputs+photometry outputs

2. **Write triggers:**
   - After `--full` PASS (session_baseline_check updates ledger)
   - After Milan-approved manual restore
   - Never auto-update on partial photometry re-run (require explicit stamp)

3. **Verification hook:**
   - `dev/tools/anchor_integrity_check.py --draft 435 --manifest anchor_checksums.json`
   - Optional WARN line in `--fast` when live draft diverges from manifest (distinct from snapshot gate)

4. **Pre-run guard:** before photometry on draft 435, UI/orchestrator confirms checksum snapshot or blocks with `ANCHOR_MUTABLE` warning.

## File lists

- Full changed list: `tmp/_audit_435_changed_list.txt` (551 entries in generator; file has 551)
- Current-only list: `tmp/_audit_435_only_current.txt` (846 entries)
- Machine inventory: `tmp/_audit_435_integrity.json`
