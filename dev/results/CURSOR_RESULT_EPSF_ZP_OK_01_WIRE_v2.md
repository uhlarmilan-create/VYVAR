CURSOR RESULT - 2026-08-26 (EPSF-ZP-OK-01-WIRE v2)

What I did
Wired W1-W4 for the wide rig only (draft 516/517 pair equipment_id=1,
telescope_id=1 -> identity `1:1`). Production default
`psf_zp_membership=fit_ok_for_zp` on that allow-list. T1 writes
`output_directory` (tmp); live BO PSF LC is not rewritten by `--fast`.
W2 regenerated 60 internal PSF LCs on live 516. Hash guards held.
Pushed to `sel-ghost-01` by name. C6 not run.

HEAD after W4 `6924998`. Live ePSF SHA
`172f95403beae36d...` unchanged. `--fast --clean` OVERALL PASS twice
(1575 passed, 32 skipped). W3b live BO PSF SHA identical to W2.

## Premise (Rule 0.1)

**What is compared:** PIN-CENSUS-01 chi2<50 pin (BO 23/134, FW 0/134)
versus `fit_ok_for_zp` membership on stored columns (no refit),
wide-rig only.

**How they differ:** `psf_fit_ok` stays the strict recorded column.
ZP membership may also admit finite `psf_flux>0` and finite `psf_chi2`.
Unvalidated rigs stay strict and stamp EPSF-ZP-OK-XRIG-01.

## W1 membership

`src_py/psf_internal_lc.py`: `psf_fit_ok_for_zp = psf_fit_ok OR
(finite psf_flux AND psf_flux>0 AND finite psf_chi2)`. Config
`psf_zp_membership` in {fit_ok_strict, fit_ok_for_zp}, default
fit_ok_for_zp. Config `psf_zp_for_zp_validated_rigs` default `["1:1"]`.
No-manifest or unlisted rig -> fit_ok_strict + header line
`# psf_zp_membership: fit_ok_strict (psf_fit_ok_for_zp not validated
for this rig; see EPSF-ZP-OK-XRIG-01)`. Stamps
`psf_zp_membership_effective` and `psf_zp_membership_rig_validated`.
Scope: INV-PSF-LC-PIN-01 only.

Tests: chi2=80 finite flux -> strict drops, for_zp keeps; unvalidated
rig -> strict + INFO; validated -> for_zp + header. INV-PSF-SUBMIT-01
unchanged.

Dashboard `config_runtime` 279 -> 281.

## W2 live 516 regenerate

n_written=60. elapsed_s=17.8. Rig `1:1` validated. Effective
fit_ok_for_zp. Hash guard: 127 must-not-change files identical
(aperture LCs, AAVSO, VarAstro, ePSF). Positive control: one-byte
scratch mutation changed SHA. ePSF SHA
`172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20`.

| target | n_finite | demeaned RMS mmag | pred n / mmag | hit |
|--------|----------|-------------------|---------------|-----|
| BO CVn `1498613634033133184` | 134/134 | 8.495 | 134 / 8.5 | YES |
| FW CVn `1497343732462852864` | 134/134 | 5.218 | 134 / 5.2 | YES |

STOP gates (miss >1 epoch or 1 mmag): **PASS**. All-60 meters:
`dev/results/context/session_20260826_closeout/c4_w2_all60_meters.csv`.
Live BO PSF LC SHA after W2:
`77adbc039dd6f247832a77f9f170a1206ce4203ee1827466b5f2a94b25a19ed0`.

## W3 output_directory

`write_internal_psf_lightcurves` / CLI `--output-directory`. Default
remains the draft lightcurves dir. T1
`test_epsf_lc_log_01_draft516_bo_cvn` writes tmp_path and asserts live
BO PSF LC SHA unchanged. W3b: `--fast` twice after this STOP; live
BO PSF SHA must equal W2a product on both runs.

## W4 docs

DECISIONS: ZP-OK v2 wide-rig only. ROADMAP EPSF-ZP-OK-XRIG-01 (MED):
extension requires (1) master dark+flat in CalibrationLibrary for
that rig and (2) CENSUS-01 replay of pin-drop vs quality on that
night. Newton 518 gated pool 26 does not qualify. Closed: "ZP-OK v2
undecided" and "T1 rewrites live 516". INV-PSF-LC-PIN-01 membership
is config-selected and rig-scoped.

## Gates

| Gate | Status | Detail |
|------|--------|--------|
| `--fast --clean` (1) | FAIL flake | `test_database_sqlite_threading` during malformed-DB WARN; 1574 passed. Same test 13/13 in isolation. Not a C4 regression. |
| `--fast --clean` (2) | PASS | 1575 passed, 32 skipped at `6924998` |
| `--fast --clean` (3) | PASS | 1575 passed, 32 skipped at `6924998` |
| W3b live BO PSF SHA | PASS | `77adbc039dd6f247...` identical to W2a on all three runs |
| 516 aperture/AAVSO/VarAstro | PASS | W2 127-file hash guard, 0 changed |
| ePSF SHA | PASS | `172f95403beae36d...` |
| INV-PSF-SUBMIT-01 | PASS | unit tests |

HEAD `6924998`. Push: `git push origin HEAD:sel-ghost-01`.

## C6

Not run. Waits Milan GO in chat after this STOP plus C8.

## Errors

None on W1-W4 science path. C8-1 R1' table remains blocked (separate
STOP).

## Files changed

- `src_py/psf_internal_lc.py`, `src_py/config.py`
- `dev/validation/params_registry.json`, `docs/VYVAR_PARAMS.md`
- `dev/tests/test_psf_internal_lc.py`, `dev/tests/test_ui_params_dashboard.py`
- `docs/VYVAR_INVARIANTS.md`, `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`
- `docs/VYVAR_CONFIG_GUIDE_EN.md`, `docs/VYVAR_CONFIG_GUIDE_CZ.md`
- this STOP (locate table kept as appendix)
- `dev/results/context/session_20260826_closeout/c4_w2.json`

## Appendix - locate STOP (2026-08-25, not wiring)

STOP at locate. Did not wire W1-W3. Did not reconstruct the parked
v2 task from v1 or from memory. Push: NO. Live 516 untouched.
HEAD then `b1f5b8c`.

### Locate search (negative)

| Place | Result |
|-------|--------|
| `dev/tasks/` | empty (0 files) |
| `dev/results/CURSOR_TASK*ZP*` | none |
| `dev/results/context/session_20260824_*` | no ZP-OK task file |
| working tree name match `*ZP*OK*` / `*zp_ok*` | none (unrelated zp_clip results only) |
| `tmp/` | none |
| Desktop / Documents / Downloads | none |
| `~/.cursor` filename match | none |
| agent transcripts | **v2 task body never pasted as a user_query** |
| git `b1f5b8c` | confirmed absent (task said so) |

Pointers that named v2 but did not contain it:

- `docs/VYVAR_HANDOFF_2026-08-24.md` line 79
- `dev/results/CURSOR_RESULT_SESSION_CLOSE_20260824.md`
- ROADMAP: "EPSF-ZP-OK-01-WIRE stays parked" (no v2 file then)

v1 existed in chat 2026-08-24 18:03 (REJECTED for missing GO). v1 W3
was docs. The locate amendment's W3 was T1 `--fast` live-516 PSF LC
rewrite. Architect re-issued v2 2026-08-25; this STOP is that wire.
