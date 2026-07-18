CURSOR RESULT ù 2026-07-18 CONFIG-HUMAN-EDIT

What I did
Made config.json editable in a plain text editor without the UI: parameters grouped into
sections by pipeline stage, with a comment for every group and every key, a file header
that explains the static (DB) / dynamic (FITS) / setting (config.json) model, a
comment-tolerant loader, an unknown-key typo safety net, and a standalone validator. Per-key
explanations now live in the registry (single source of truth), ported from the English
config guide. One commit per step, full pytest green after each, ASCII-only.

## Per-step commits + diff --stat

STEP 1 ù port guide explanations into registry help ù `1307b73`
```
 dev/tests/test_params_registry.py   |  24 ++
 dev/validation/params_registry.json | 538 ++++++++++++++++++------------------
 docs/VYVAR_PARAMS.md                |   2 +-
 3 files changed, 294 insertions(+), 270 deletions(-)
```
All 269 registry `help` fields replaced with the hand-authored per-key Explanation text
from VYVAR_CONFIG_GUIDE_EN.md (1:1 coverage; ASCII; none empty). New guard
`test_help_is_nonempty_ascii_and_not_placeholder`. VYVAR_PARAMS.md keeps no help column by
design (the guide remains the human-readable place; the generated table stays a compact
metadata index) ù only the volatile header line changed.

STEP 2 ù comment-tolerant loader + unknown-key warnings ù `742fee5`
```
 dev/tests/test_config_jsonc_loader.py |  96 ++++++++++++++++++++++++
 src_py/config.py                      | 135 +++++++++++++++++++++++++++++++++-
 2 files changed, 228 insertions(+), 3 deletions(-)
```
`strip_jsonc_comments` is a character state machine that removes `//` line comments only
OUTSIDE string literals (so `"http://x"` and `//` inside values survive); block comments
and trailing commas remain unsupported. `load_config_json` now parses JSONC, warns (never
raises) on malformed JSON pointing at the validator, and warns on unknown keys with the
closest registered field (difflib) while staying silent on migrated legacy aliases
(uppercase env-style + WAVE-B scalar tiers/aperture, via `_LEGACY_CONFIG_KEYS`).

STEP 3 ù canonical grouped + commented writer + live migration ù `86c6748`
```
 config.json                           | 935 +++++++++++++++++++++++++---------
 dev/tests/test_config_jsonc_writer.py |  99 ++++
 dev/tests/test_params_registry.py     |  16 +
 dev/validation/params_registry.json   |  17 +
 src_py/config.py                      | 136 ++++-
 src_py/params_registry.py             |  23 +-
 6 files changed, 989 insertions(+), 237 deletions(-)
```
`save_config_json` -> `render_config_jsonc`: file header + sections in pipeline order
(observer, calibration, qc, alignment, detection, photometry, comp_selection, trust,
extinction, reports, export, system, paths last), each opened by a group comment from the
registry `__meta__.phase_help` block; within a section keys are basic -> advanced -> expert
then alphabetical, each preceded by its one-line registry help. Non-registry keys fall into
a trailing "Other" section (never dropped). Registry gained a reserved `__meta__.phase_help`
(13 phases, ASCII); `load_registry` strips `__`-prefixed reserved keys so field parity
holds; new `load_phase_help` / `load_registry_meta`.

STEP 4 ù standalone validator ù `0b75a69`
```
 dev/scripts/validate_config.py    | 170 ++++++++++++++++++++++++++++++++++++++
 dev/tests/test_validate_config.py |  83 +++++++++++++++++++
 docs/VYVAR_CONFIG_GUIDE_CZ.md     |  23 ++++++
 docs/VYVAR_CONFIG_GUIDE_EN.md     |  22 +++++
 4 files changed, 298 insertions(+)
```
`dev/scripts/validate_config.py` reports (a) syntax errors with line/column, (b) unknown
keys with suggestions, (c) out-of-range values, (d) type mismatches (outermost-token aware;
Optional/None handled; ambiguous multi-type unions skipped); non-zero exit on any error.
"Editing without the UI" paragraph added to both guides (CZ pure ASCII).

STEP 5 ù docs & result ù (this commit)
PROCESS one-liner (config.json is generated-commented; registry help is the single source),
STATE + JOURNAL stamps, this result file.

## Value-equality assertion (live config.json migration)

Migration script re-rendered the live config.json in the new grouped/commented form and
asserted value-equality key-by-key before writing:
```
MIGRATION OK
keys: 249  (old==new: True)
old bytes: 9362  new bytes: 36129
```
`set(new) == set(old)` and `new[k] == old[k]` for every key. Only ordering and comments
changed; not one value moved. Post-migration `validate_config.py` on the live file:
```
Validating C:\ASTRO\python\VYVAR\config.json
  parsed OK: 249 keys

OK: no errors, 0 warning(s).
```

## Sample of the new config.json (header + first section)

```
// ===========================================================================
// VYVAR config.json -- pipeline settings, safe to edit in a text editor.
//
// This file holds ONLY user-tunable pipeline settings. Two other kinds of
// values live elsewhere and are NOT in this file:
//   - Static observatory facts (site coordinates, telescope, camera,
// catalogs) live in the DATABASE and are managed in the app (Settings ->
// Observatory).
//   - Dynamic per-run values (gain, read noise, frame size, plate scale,
// filter, exposure) are read from the FITS headers at run time and appear in
// the report's Resolved Facts section.
//
// Editing without the UI: '//' line comments are allowed (they are ignored on
// load). Trailing commas and block comments are NOT allowed. Unknown keys are
// ignored with a warning that suggests the closest real key. After editing,
// validate with:
//     python dev/scripts/validate_config.py
// Full explanations of every key: docs/VYVAR_CONFIG_GUIDE_EN.md (English) and
// docs/VYVAR_CONFIG_GUIDE_CZ.md (Czech).
//
// NOTE: saving from the UI regenerates this file, its grouping and its
// comments from the parameter registry -- any custom comments you add here
// are not preserved.
// ===========================================================================
{
  // === Observer & export identity ===
  // Who observed and from where: AAVSO identity and observing-site
  // coordinates. Observatory facts; the database LOCATION table is the source
  // of truth.
  // Your official AAVSO observer code (UMIA); stamped into every AAVSO
  // submission so the observation is credited to you.
  "aavso_observer_code": "UMIA",
  ...
}
```

## Note for direct-JSON dev scripts

A handful of untracked dev/scratch night-run helpers (e.g. dev/scripts/*_night_run_*.py)
read config.json with a bare `json.loads`. Those bypass the tolerant loader and would trip
on `//` comments; switch them to `config.parse_config_text(...)` if needed. All tracked
production/anchor paths go through `AppConfig` -> `load_config_json` and are unaffected.

## Gates

- Full pytest green after each step (final: 963 passed, 19 skipped).
- session_baseline_check.py --fast: PASS (963 passed, 19 skipped; only WARNs are the known
  untracked scripts / origin-main-ahead / ledger-todo).
- MANDATORY --full anchor gate: PASS -- byte-identical science, confirming the
  comment-tolerant loader on the startup path changed no pipeline output:
```
full-provenance              PASS   anchor git_hash=10d610c0e79d...
full-pipeline                PASS   2311s -> tmp\session_baseline\20260718T132649Z
full-science-compare         PASS   n_lc=166 failures=0
full-snapshot-sha-core       PASS   3d26f4692ac81fc5... n=333
full-photometry-sha-core     PASS   3d26f4692ac81fc5... n=333
full-photometry-sha-extended PASS   6420f1daa53a0d5d... n=499
full-counters-expected       PASS   allowlisted {"phase2a_empty_comp_drop": 1}
------------------------------------------------------------------------
OVERALL: PASS
```

## Push
GATED ù the whole stack (audit + wave B + config-human-edit) pushes together on Milan's
explicit word.
