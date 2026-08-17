# CURSOR RESULT - PUSH-AUTH 2026-08-17 (STOPPED, amendment retry)

Date: 2026-08-17
Compared with: Milan-authorized push of local tip f4f16d5 vs origin/main
a217e1d. Sync only. Amendment: do not push on the earlier PASS; retry
`--fast` after >= 30 min since the last 429; no exemption push.

**Push was NOT executed.**

## Amendment retry

Last 429 before cooldown: 2026-08-17T13:49:50Z
(`test_ui_material_icons.py::test_telescope_icon_is_not_used`).

Cooldown wait >= 30 min, then `--fast` started 2026-08-17T14:21:52Z
(~32 min later). P1 env unset.

| check | status | detail |
|-------|--------|--------|
| git-head | PASS | f4f16d5 |
| pytest | FAIL | 1441 passed, 28 skipped |
| OVERALL | FAIL | -- |

Solo confirmation after that FAIL:

```
python -m pytest dev/tests/test_ui_material_icons.py -q --tb=line
.F
FAILED test_telescope_icon_is_not_used
urllib.error.HTTPError: HTTP Error 429: Too Many Requests
1 failed, 1 passed in 3.72s
```

Sibling `test_repo_material_icon_literals_validate_against_pinned_streamlit`
PASSes (local streamlit validator). The `--fast` count vs the earlier
green run on this tip (1442 passed / 28 skipped) is a single missing
pass. pytest lastfailed live node under `dev/tests` is only
`test_telescope_icon_is_not_used`. No other new failure identified.
No exemption push.

## Inventory (unchanged)

12 unpushed commits, hashes not rewritten. Content tip **f4f16d5**.
origin/main **a217e1d**. `git pull --rebase` was already up to date.

```
f4f16d5 docs+results: PFS-SEMANTICS-01, SAT-RERANK-01B, EXPORT-HDR-01 close.
71925b7 tools: SAT-RERANK-01B meters and ensemble-aware 01B carrier.
28a5f49 EXPORT-HDR-01: NOTES n_comp, pytics weights, check sidecar, PFS matrix.
7431449 ledger: quarantine 8f107cf as VL-PFS-8F107CF.
b3be817 harness: 515 PFS per-run override, UTF-8 log, LC_TRIM; [COMP] ladder lines.
44ebabb INV-CFG-01: read vsx_out_of_scope_types from provenance.config_snapshot.
7544808 PFS-SEMANTICS-01: rescue on skip_reason; one INV-SAT-LIMIT peak-test.
1a516c7 results: XVAL-AIJ-01 epoch table and SAT-LIMIT-01 measurements.
a75544b docs: close XVAL-AIJ-01 independent-tool check and SAT-LIMIT-01.
a23e62c tools: SAT-LIMIT-01 catalog/knee measure and 515 reclassify harness.
708ba32 test: INV-SAT-LIMIT fire proofs for NaN clip and stack overshoot.
50cc9ac SAT-LIMIT-01: never silently admit a missing saturation clip.
```

## WIDE-ERR-04 leftovers (still uncommitted)

- `dev/results/CURSOR_RESULT_WIDE_ERR_04.md`
- `dev/results/WIDE_ERR_04_summary.json`

Scratch (not product): `dev/tests/_tmp_batch_e_lc/`, `src_py/tmp/xval_out/`,
`vyvar.sqlite3-shm`, `vyvar.sqlite3-wal`.

## PUSH-STAMP-01

Not stamped. Content tip remains **f4f16d5**.

## NET-TEST-01 (queued, not implemented)

ROADMAP next-session HIGH (3). Preferred: vendor pinned Streamlit 1.60.0
icon list. Alt: auto-skip with WARN on network errors. Fold-in:
`session_baseline_check.check_pytest` must print the failing node id.
No code change in this task.

## Docs impact

- `docs/VYVAR_ROADMAP.md` -- NET-TEST-01 queued (working tree, not in
  the f4f16d5 pack). CORR-ERR-01 row kept.
- this RESULT (uncommitted)

## Recurrence

Recurrence: queued as NET-TEST-01 (second time the `--fast` gate depended
on a live GitHub fetch this session).

## Amendment 2 -- NET-TEST-01 implemented (harness only)

Pulled forward. No science change.

Vendored `dev/tests/data/streamlit_material_icon_names.txt` from locally
installed streamlit **1.55.0** (`streamlit.material_icon_names.ALL_MATERIAL_ICONS`,
n=4188 names, one per line). Spec said 1.60.0; this machine's installed
package is 1.55.0, which is what the sibling literals test already used.
Physics outranks the spec: vendor the local package, record the version
in the file header. No urllib/urlopen remains under `dev/tests/`.

`test_vendored_material_icon_list_sanity`: n>=1000 and known names
search/settings/science/help present.
`test_telescope_icon_is_not_used` reads the vendored list (telescope
absent) and asserts no `:material/telescope:` in `src_py/`.

`check_pytest` now appends `fail=<node id>` from pytest FAILED/ERROR
summary lines.

One commit on top of f4f16d5 is the new PUSH-STAMP-01 content tip
(recorded after that commit). Then `--fast` / push / origin verify.

