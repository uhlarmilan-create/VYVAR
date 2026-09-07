CURSOR RESULT - 2026-09-07 SKY-SURFACE-BLAST-RADIUS-01

Date: 2026-09-07. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 03b0b86.
Class: FORENSIC READ-ONLY. No production code change. No draft
written, touched, or re-run. Nothing deleted.

## Architect error 15

FRAME-QC-PARITY-02C wiring-point suggestion cited stale line
ranges (app.py:817-853 / night_run.py:241-277) carried from the
phase-1 prestep report without re-verification on the current
tree; the suggestion would have missed the OSC path
(pipeline_calibrate.py:3235). Same class as errors 9-14: claim
from a prior artifact instead of the current governing code.
Caught by Cursor's refutation authority; the wire landed at the
true shared point (_qc_enrich_calibrated_in_place :3675).

## Refute check

Export location claim (C-EXPORT-GAP-VERIFY-01) holds on this tree:
`lightcurves_reports` is created at
photometry_phase2a.py:3572; AAVSO/VarAstro dirs are
photometry_report.py:851-852
(`<photometry>/lightcurves_reports/aavso` and `.../varastro`).
AAVSO data rows: STARID,DATE(BJD),...,FILTER
(export_reports.py:1366-1374). VarAstro target line
`# VAR Name:` :1484; data rows BJD :1537-1545.

Archive root is not hardcoded: read from config.json key
`archive_root` via comment-safe parse
(config.py:208 parse_config_text / :263 load_config_json;
config.json:774). Resolved:
`C:\ASTRO\python\VYVAR\Archive`. Standard draft path
`Archive/Drafts/draft_{id:06d}` (night_run.py:1025,
draft_provenance.py:682).

AppConfig was not constructed (it mkdir's archive_root,
config.py:2914). VyvarDatabase was not opened (it can write;
MASTER_SOURCES hygiene). SQLite used
`file:<path>?mode=ro` only.

## Step 1 - draft directories 438-451

Present Drafts on disk: 515-520 and 516 snapshots only.
Archive also has raw-session folders `2026-06-05_MZ_boyden` and
`2026-06-08_MZ_zdanice` (FITS only; not draft_NNNNNN).
No relocated `draft_000438`..`draft_000451` and no
`draft_manifest.json` with those ids under Archive.

| id | directory | relocated | DB date (read-only) | DB source |
|----|-----------|-----------|---------------------|-----------|
| 438 | ABSENT | none | 2026-07-21T15:40:59Z | OBS_QC_PROCESSING_RUN |
| 439 | ABSENT | none | 2026-07-21T16:36:31Z | OBS_QC_PROCESSING_RUN |
| 440 | ABSENT | none | (no leftover row) | -- |
| 441 | ABSENT | none | 2026-07-21T16:56:44Z | OBS_QC_PROCESSING_RUN |
| 442 | ABSENT | none | (no leftover row) | -- |
| 443 | ABSENT | none | (no leftover row) | -- |
| 444 | ABSENT | none | 2026-07-21T17:06:30Z | OBS_QC_PROCESSING_RUN |
| 445 | ABSENT | none | (no leftover row) | -- |
| 446 | ABSENT | none | (no leftover row) | -- |
| 447 | ABSENT | none | (no leftover row) | -- |
| 448 | ABSENT | none | 2026-07-21T18:16:55Z | MASTER_SOURCES |
| 449 | ABSENT | none | 2026-07-22T10:25:56Z | MASTER_SOURCES |
| 450 | ABSENT | none | 2026-07-24T20:42:09Z | MASTER_SOURCES |
| 451 | ABSENT | none | 2026-07-27T07:21:21Z | MASTER_SOURCES |

No OBS_DRAFT table (files-only). Leftover MASTER_SOURCES counts
(not a catalogue of exports): 438 n=8843; 439 n=8848; 441 n=878;
444 n=9164; 448 n=5733; 449 n=5733; 450 n=3993; 451 n=3993.

Corroboration (not the measurement): 
`dev/results/context/deleted_drafts.md` listed 438 and 448-451
for the 2026-07-28 operator cleanup. 439-447 were not in that
manifest; they are also absent on disk today.

## Step 2-3 - lightcurves_reports / fingerprint

No draft root exists, so no
`platesolve/<setup>/photometry/lightcurves_reports/` tree.
Fingerprint table is empty (header only):
`dev/results/context/session_20260907_skysurface_blast/fingerprint.csv`
and `fingerprint.md`.

## Step 4 - ad-hoc glob

Per-draft glob `*aavso*`, `*webobs*`, `*varastro*` (case-insensitive):
not applicable (no draft root). Extra Archive-wide filename scan
for those tokens: 0 hits (including current 515-520).

## Evidence

`dev/results/context/session_20260907_skysurface_blast/`
- scan_sky_surface_blast.py
- inventory.json
- fingerprint.csv
- fingerprint.md
- summary.json

No large product files. Live Archive not written.

## Gates

`--fast --clean` OVERALL PASS (1625 passed, 32 skipped;
clean-tree PASS) on dirty tree with this evidence uncommitted;
production modules unchanged. No `--full`. No draft run.

R-S1: NO export files exist anywhere in 438-451 -> local
generation never happened; external risk reduces to "did Milan
upload anything from these nights via any tool" - a pure
account-side question. Report states R-S1.

STOP. No ROADMAP/register closure in this task; the row closes only
after Milan confirms account-side (his one-line answer will be
recorded in a follow-up docs commit).
