CURSOR RESULT - 2026-09-07 C-EXPORT-GAP-VERIFY-01

Date: 2026-09-07. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 59c69d2.
Class: MEASUREMENT ONLY. No production code change. No science change.
Live draft 516 was not written. Sandbox exports are EVIDENCE ONLY
(uploads remain HELD).

## Decision

R2. H1 fails (partial exports). C-EXPORT-GAP left OPEN. No ROADMAP
edit, no AUDIT-REGISTER edit, no DECISIONS entry, no commit, no push.

## Static trace (not refuted; line numbers corrected)

The call chain is real. File:line drift only:

- `run_night_pipeline` def is night_run.py:1295. Step 13 +
  `run_night_photometry` call is night_run.py:1841-1850
  (`existing_draft=False` on the full-night path).
- `run_night_photometry` def night_run.py:973. It also logs Step 13
  at :1037. Calls `run_full_photometry_pipeline` at :1116.
- `run_full_photometry_pipeline` photometry_core.py:718; calls
  `run_phase2a` at :897.
- `run_phase2a` photometry_phase2a.py:3868. Returns
  `_phase2a_finalize_exports` at :4078 (def :3459).
- Export loop: `export_all_method_lightcurve_reports` ~:3566/:3647;
  no-LC skip log :3612; summary
  `[EXPORT] lightcurves_reports: %d targets exported, %d skipped`
  :3679.
- `active_report_methods` report_methods.py:17-30 always includes
  `"aperture"`. No config key disables the loop wholesale.
- UI C3: night_run.py:6 docstring; wrapper `run_ui_night_photometry`
  :1278 defaults `existing_draft=True`.
- Prior 512 evidence: CURSOR_RESULT_DRAFT_512_EXTRACT.md:350-360
  (17 AAVSO + VarAstro companion set).

The chain itself is not wrong. Measurement proceeded.

## CLI vs G2 sandbox

`parse_night_run_cli` (night_run.py:223) has `--source`, `--camera`,
`--telescope`, `--site`, `--draft-dir`, `--config`, `--dry-run`.
`main()` always calls `run_night_pipeline` (full import/cal/align
then photometry). There is no photometry-only CLI flag.

The G2 sandbox copy has no importable raw FITS. Running the full CLI
against live Archive/Drafts/draft_000516 is forbidden.

Documented CLI (parsed; IDs resolved; `main()` / `run_night_pipeline`
NOT executed):

```
python src_py/night_run.py --source C:\ASTRO\python\VYVAR\tmp\session_20260907_cexportgap\sandbox --camera 1 --telescope 1 --site 2 --draft-dir C:\ASTRO\python\VYVAR\tmp\session_20260907_cexportgap\sandbox
```

Parsed ids from snapshot `draft_manifest.json` rig / CLI flags:
equipment_id=1, telescope_id=1, location_id=2 (wide-rig).

Measured entry (G2-equivalent photometry slice that Step 13 of
`run_night_pipeline` and C3 both call):

```
python -u tmp\c_export_gap_verify_01.py
```

which copies `draft_000516_snapshot_era04_20260826` via
`_copy_frozen_anchor_inputs` into
`tmp/session_20260907_cexportgap/sandbox` (never live Archive) and
calls `run_night_photometry(existing_draft=True,
draft_dir_override=sandbox, write_pdfs=False, epsf=False)` with G2
cfg (`k2_mode=literature`, `save_lightcurve_png=False`,
`per_frame_saturation_enabled=True`, `VYVAR_P1_FORCE=1`).

Elapsed: 1363.3 s. phot_errors=[]. Setup: NoFilter_60_2.

## Counts

| item | n |
|------|---|
| active_targets.csv | 253 |
| aperture LC CSV (`lightcurve_<cid>.csv`) | 53 |
| lightcurves_reports/aavso/*.txt | 50 |
| lightcurves_reports/varastro/* (all files) | 20 |
| varastro .txt | 10 |
| varastro _field.png | 10 |

G2 era04_aperture n=53 matches n_lc.

## Per-target skip accounting (253 active_targets)

Identity: 50 AAVSO written + 3 empty-point failures + 200 no-LC-CSV
skips = 253.

| class | n | governing branch | H1 class |
|-------|---|------------------|----------|
| `[EXPORT] skip ... no LC CSV` | 200 | photometry_phase2a.py:3612 | legitimate skip (H1) |
| AAVSO write ok | 50 | export_reports.py:1391-1393 | product present |
| `record_export_failure` empty points | 3 | export_reports.py:1000-1004; batch summary | NOT a no-LC skip; LC CSV exists |
| VarAstro write ok (eclipsing) | 10 of the 50 | export_reports.py:1409+ | product present |
| `[EXPORT] Skip varastro ... nie zakrytova` | 40 of the 50 | export_reports.py:1405-1407 `_is_eclipsing` | NOT in H1 legitimate-skip list |

The 3 LC-present AAVSO failures (flags/mag empty):

- 1498842882207281152
- 1499842372636900992
- 1500410236033012352

phase2a_empty_comp_drop counter was 3 (photometry drop); the three
export failures are the three LC files that did not yield exportable
points. 53 LC CSV = 50 AAVSO + 3 failed.

skip_with_lc (logged no-LC skip but LC present): 0.

## Export summary log line

```
[EXPORT] lightcurves_reports: 50 targets exported, 203 skipped (methods=aperture)
```

203 skipped = 200 no-LC-CSV + 3 empty-point failures.

`log_export_batch_summary`:

```
[EXPORT] batch finished with 3 export failure(s) across 3 target(s)
[EXPORT] failed target ids: 1498842882207281152,1499842372636900992,1500410236033012352
[EXPORT]   1498842882207281152 | method=aperture | no exportable LC points (flags/mag empty)
[EXPORT]   1499842372636900992 | method=aperture | no exportable LC points (flags/mag empty)
[EXPORT]   1500410236033012352 | method=aperture | no exportable LC points (flags/mag empty)
```

No `[EXPORT] init failed` / EXC-0176. No `time_base_refused`.
Loop ran (not wholesale omit). Partial products.

## Band letter (NoFilter group; report only, do not judge)

File: sandbox `lightcurves_reports/aavso/BO_CVn_20260423.txt`
(also copied to session sample_exports).

No `#BAND=` comment. Extended data row field 5 (0-based index 4):
**CV**. First data line:

`BO CVn,2461154.320827,9.470,0.008,CV,YES,STD,ENSEMBLE,...`

Expect CV per D10-1: observed CV. Not judged.

## G4 live 516

Before and after the sandbox run (read-only hashes):

| product | sha256 prefix | verdict |
|---------|---------------|---------|
| masterstars_full_match.csv | bfa24039778f437b... | PASS |
| MASTERSTAR.fits | 13e77cf8a1dcb4e7... | PASS |
| masterstar_epsf.fits | 172f95403beae36d... | PASS |

Expected prefixes csv `bfa24039` / fits `13e77cf8` / epsf `172f9540`.
Live draft 516 was not written.

## H1 / R1 / R2 (verbatim)

H1 required aavso/*.txt AND varastro/* for every target that has an
aperture LC CSV, minus logged `[EXPORT] skip ... no LC CSV`.

H1 fails:
1. 3/53 LC targets: AAVSO batch error (empty points).
2. 40/50 AAVSO-ok targets: VarAstro skipped as not eclipsing
   (`nie zakrytova`). That skip is not the H1-listed no-LC skip.
3. n_varastro txt = 10 vs n_lc = 53.

Partial = R2 even if mostly works.

R2 applied: capture governing branches (above), classify per-target
skip vs batch error, STOP. No production fix. ROADMAP C-EXPORT-GAP
left OPEN. Register left OPEN.

## Session evidence

`dev/results/context/session_20260907_cexportgap/`

- night_run_photometry.log
- inventory.json
- export_accounting.json
- g4_before.json / g4_after.json
- sample_exports/aavso_BO_CVn_20260423.txt
- sample_exports/varastro_BO_CVn_20260423.txt
- plus extra ASASSN samples

Full sandbox (too large to copy into git):
`tmp/session_20260907_cexportgap/sandbox/platesolve/NoFilter_60_2/photometry/lightcurves_reports/`

Harness (gitignored tmp): `tmp/c_export_gap_verify_01.py`

## Gates

`--fast --clean` OVERALL PASS (1621 passed, 32 skipped; clean-tree
PASS). HEAD 59c69d2. Branch consolidate-01. No `--full`, no
`--full-epsf`. Under R2: no commit, no push.
