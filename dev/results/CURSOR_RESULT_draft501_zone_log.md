CURSOR RESULT - 2026-08-05 (DRAFT-501 zone-column log inspection)

What I did
Read-only search for draft_501 run logs; grepped infolog for zone-annotation
markers; inspected pipeline_meta.json; compared masterstars vs pipeline_meta
mtimes. No pipeline runs.

## 1 -- Run log location

Searched locations:

| location | result |
|----------|--------|
| Archive/logs/ | directory does not exist |
| draft_000501/**/*.log | no *.log files |
| repo logs/ | directory empty / no runtime logs |
| Archive/Drafts/draft_000501/infolog_*.txt | FOUND |

Primary log (matches run window):

| field | value |
|-------|-------|
| path | Archive/Drafts/draft_000501/infolog_20260805_113441.txt |
| mtime (UTC) | 2026-08-05T09:49:38.0657943Z |
| size | 234251 bytes |
| draft header | Draft: C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000501 |
| session start | 09:34:39 UTC run_vyvar start |

pipeline_meta.json stamped_at_utc = 2026-08-05T09:49:12.536386+00:00
(infolog ends 09:49:38 -- same session).

No separate VYVAR runtime log in repo logs/ for this window. infolog is the
authoritative durable session record for draft_501.

## 2 -- Log marker search (infolog_20260805_113441.txt)

| # | marker | found | matched line |
|---|--------|-------|--------------|
| 1 | "MASTERSTAR coordinate finalization" | no | (exact string absent; this text only appears in pipeline.py:12454 SKIP path) |
| 2 | "_annotate_masterstars_flux_zones" or "annotate_masterstars" | no | (function name not logged) |
| 3 | "flux" AND "masterstars" together | yes | 09:40:49 DEBUG: per-frame debug_pixel_match (first frame): {"file": "TOI-1131.01.b_2025-04-22_23-05-09_V.fits", "use_fast": true, "master_cols": ["name", "ra_deg", "dec_deg", "mag", "b_v", "catalog", "catalog_id", "x", "y", "flux"], ...} |
| 4 | "noise_floor_adu" | no | closest: 09:38:30 MASTERSTAR levels: noise_floor(min)=33410.70, saturation_proxy(max)=98300.55 ; 09:38:42 DAO po SNR filtri ... noise_floor~33493.9 ADU |
| 5 | "saturate_limit_adu_85pct" | no | (string absent) |
| 6 | Traceback in 200 lines after "MASTERSTAR coordinate finalization" | no | no "Traceback (most recent call last)" anywhere in infolog |
| 7 | phase2a start marker | yes | 09:49:04 [RUN VYVAR (non-cal)] Faza 2A: aperture photometry + lightcurves... |

Additional critical line (post-finalize, pre-CSV-write path):

  09:38:52  MASTERSTAR source_type annotate failed: INV-MS-01: dao_only_fraction=0.417 (fail>0.25)

Context lines immediately preceding that failure:

  09:38:52  Astrometry optimizer: wrote ...\masterstars_full_match.csv (972/1668 catalog-matched).
  09:38:52  MASTERSTAR optimizer: forced final re-match pass completed.
  09:38:52  finalize_masterstar_sky_coords: gaia_catalog=972 final_wcs=696 (matched rows with DB hit=972/972)
  09:38:52  matched_world2pix_identity_px: n=972 p95=2.018 p99=2.983
  09:38:52  IDENTITY-QA WARN: p95=2.018 px > 2.0 px threshold
  09:38:52  masterstars bp_rp fallback: 0/5 doplnenych z Gaia DB
  09:38:52  MASTERSTAR source_type annotate failed: INV-MS-01: dao_only_fraction=0.417 (fail>0.25)

Coordinate finalization DID run (finalize_masterstar_sky_coords + identity QA
above); the exact "MASTERSTAR coordinate finalization" string is only emitted
on skip/failure per pipeline.py:12454.

Code-path inference (pipeline.py:12458-12493):
  - _annotate_masterstars_flux_zones is called unconditionally at :12459 (no log)
  - _vyvar_df_to_csv(df_final, csv_path) at :12491 is inside try block
  - INV-MS-01 InvariantViolation at :12483 aborts try before CSV write
  - outer except :12492-12493 logs "MASTERSTAR source_type annotate failed: ..."

On-disk CSV therefore retains the earlier astrometry-optimizer write (no zone
columns); annotated dataframe with zone/is_saturated/is_usable was not flushed.

## 3 -- pipeline_meta.json inspection

Path: Archive/Drafts/draft_000501/platesolve/V_60_2/photometry/pipeline_meta.json

| field | value |
|-------|-------|
| entry_point | run_phase2a |
| git_hash | 2c964cb660e8e0ecd3b9dfe29063e30fb1e2b54c |
| git_dirty | true |
| git_dirty_code | false |
| stamped_at_utc | 2026-08-05T09:49:12.536386+00:00 |

No top-level error or warning key. No masterstars_meta or det_meta block.
No noise_floor_adu or saturate_limit_adu at top level.

Masterstars-related top-level fields present:
  n_gaia_detected=972, catalog_rows=26504, gaia_dao_completeness_pct=98.17,
  gaia_dao_completeness_raw_pct=3.67, n_dao_unmatched=696

stages:
  masterstar  seq=3  ts=2026-08-05T09:39:00.287800+00:00  cold_start=true  gap=false
  phase01     seq=5  ts=2026-08-05T09:49:04.387783+00:00  cold_start=false gap=true
  phase2a     seq=6  ts=2026-08-05T09:49:26.520201+00:00  cold_start=false gap=false
  postprocess seq=7  ts=2026-08-05T09:49:26.527919+00:00  cold_start=false gap=false

invariants (masterstars/WCS relevant):
  INV-WCS-01  ok=false  policy=WARN  detail=matched_world2pix_identity_p95_px=2.018 (warn<2)
              ts=2026-08-05T09:38:52.638653+00:00
  INV-DAG-01  ok=true   detail=stamped masterstar seq=3 (cold_start)

INV-MS-01 NOT recorded in pipeline_meta invariants (failure was caught and
logged, run continued).

lc_quality_summary: no_data=22, good=0, total=22

provenance block (entire):

  "provenance": {
    "git_hash": "2c964cb660e8e0ecd3b9dfe29063e30fb1e2b54c",
    "git_dirty": true,
    "config_snapshot": { ... 271 config keys ... },
    "stamped_at_utc": "2026-08-05T09:49:12.536386+00:00",
    "entry_point": "run_phase2a",
    "labbe_rng_seed_policy": "content_frame_hash_v1",
    "catalog_databases": {
      "gaia_dr3": {
        "kind": "gaia_dr3_sqlite",
        "path": "C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\vyvar_gaia_dr3.db",
        "size_bytes": 53137264640,
        "mtime_utc": "2026-06-14T23:05:57.812991+00:00",
        "fingerprint_sha256": "921ecb430eabd2f5d1c4815ea99bb08d2ee04734b8a45f66f60f0fe51126552d",
        "fingerprint_method": "sha256(size + first_1MiB + last_1MiB)",
        "row_count": 211712600,
        "max_g_mag": 17.5
      },
      "vsx_local": {
        "kind": "vsx_local_sqlite",
        "path": "C:\\ASTRO\\python\\VYVAR\\VSX\\vyvar_vsx_local_v2.db",
        "size_bytes": 908324864,
        "mtime_utc": "2026-06-03T07:02:13.728443+00:00",
        "fingerprint_sha256": "13b4753f97c16a23f079026d9beab3eab0a1ebf3ea917f302e19e2f41b5086c5",
        "fingerprint_method": "sha256(size + first_1MiB + last_1MiB)",
        "row_count": 7827904
      }
    },
    "git_dirty_files": [
      {"path": "config.json", "content_sha256": "c93bfe7483c760815392e83444cd1a525e076a85e304d5dd522ca63f7858a892"},
      {"path": "dev/tools/wide_err_a2b.py", "content_sha256": "acf136e3fdf922a188583e0c9bb03d1ed1465e775db53bdf3ad6cba7ece06a04"},
      {"path": "dev/tools/wide_err_e1.py", "content_sha256": "e5fcbf90fcbf75cfc3ccdbea5a760be1080999220ebedd557271684ac862fbcd"},
      {"path": "dev/tools/wide_err_e2.py", "content_sha256": "f7469307723e35d29529ba70261f76322a1603d155907e7f142e3793e3187f93"},
      {"path": "dev/tools/wide_err_e3.py", "content_sha256": "81b52a18a1488ef26a63f76e4701abd7c69ad03d0a30543ad81e71b3bba077c2"},
      {"path": "dev/tools/wide_err_e4.py", "content_sha256": "93c7453048b7a0725f9e5dda1b132e8e8603d2b680f157b962e993ed577b5266"},
      {"path": "vyvar.sqlite3-shm", "content_sha256": "c5f95007f622ae66bbb0b2db66b6bec3cf2c1730f6d4e7ad050cabd2337bc05a"},
      {"path": "vyvar.sqlite3-wal", "content_sha256": "d4bcd11ee0b82d43ea68ce95011231dbe085f3a592c7fec7bbdd6dfd580fec1b"}
    ],
    "git_dirty_code": false,
    "git_dirty_code_files": [],
    "git_dirty_scratch_files": [
      "config.json",
      "dev/tools/wide_err_a2b.py",
      "dev/tools/wide_err_e1.py",
      "dev/tools/wide_err_e2.py",
      "dev/tools/wide_err_e3.py",
      "dev/tools/wide_err_e4.py",
      "vyvar.sqlite3-shm",
      "vyvar.sqlite3-wal"
    ]
  }

(config_snapshot elided -- 271 keys; full copy in pipeline_meta.json on disk)

## 4 -- Was Phase 1 skipped / masterstars reused?

| check | result |
|-------|--------|
| masterstars_full_match.csv mtime (UTC) | 2026-08-05T09:39:02.5617747Z |
| pipeline_meta.json mtime (UTC) | 2026-08-05T09:49:26.5279195Z |
| masterstars older than pipeline_meta | yes (by ~10 min 24 s) |
| masterstars_full_match.csv.bak or older copy | none found in draft_000501 tree |
| same-session masterstar build in infolog | yes (09:37:33-09:39:02 MAKE MASTERSTAR phase) |
| pipeline_meta stages.masterstar.cold_start | true (fresh masterstar in this run) |

Interpretation: masterstars IS older than pipeline_meta, but both belong to the
same continuous run (infolog 09:34:39-09:49:38). Masterstar phase completed
~09:39; phase01+phase2a ran ~09:46-09:49. This is NOT reuse from a prior
standalone Phase 1 on different code. Phase 2A entry_point stamped because the
operator session resumed at photometry; masterstar stage ran cold_start=true
within the same session.

NOT DRAFT501-PHASE1-SKIPPED.

## Files changed

None (read-only).

DRAFT501-LOG-EXCEPTION -- Phase 1 ran, _annotate_masterstars_flux_zones
                            raised an exception
