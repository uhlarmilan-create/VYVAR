CURSOR RESULT - 2026-08-22T23:59:00Z (EPSF-VALID-02 S6 GO)

What I did
Executed S6 per FINAL task + three addenda: production edge-star build guard (Addendum 1),
S5 Part D retraction pointer (Addendum 2), N policy in DECISIONS + build meta (Addendum 3).
Swapped draft 516 to guarded gated 67-star ePSF; first gated build on 517 (66 stars); F6 PSF
merge on 516 science-light frames.

STOP-B signed; Milan authorized swap.

---

## Commits

| Commit | Description |
|--------|-------------|
| `f97615a` | Addendum 1: production ePSF edge-star build guard + tests |
| `8b98156` | S6 close: docs, INV-EPSF-BUILD-GUARD-01 wired, execution JSON |
| `777f10e` | ASCII fix in S5/S6 result docs |

HEAD: **`777f10e`**.

---

## Addendum 1 - edge-star build guard (production)

**Ported to** `src_py/psf_photometry.py` (`build_epsf_model` loop):

- Deterministic drop: edge-nearest star by `(dist_edge_px, catalog_id)` on non-finite ValueError.
- Logged in `masterstar_epsf_meta.json` -> `build_guard` (catalog_id, x, y, dist_edge_px, reason).
- Bounded: FAIL LOUDLY if next drop would exceed 10% of isolated gated pool.
- Meta also carries `n_policy` block (Addendum 3).

**Tests:** `dev/tests/test_epsf_build_guard.py` (3 passed).

---

## S6a - Draft 516 swap

| Item | Value |
|------|------:|
| Pre-gated backup | `masterstar_epsf_pre_gated_20260822.fits` (+ `_meta.json`) |
| Superseded n_stars_used | **1475** |
| New n_stars_used | **67** |
| Guard drops | **0** |
| Build path | Guarded production `build_epsf_model` (not sandbox artifact copy) |
| swap_date | 2026-08-22 |

**Meta excerpts** (`Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/masterstar_epsf_meta.json`):

```json
"build_guard": {"n_dropped": 0, "n_pool_baseline": 67, "dropped": []},
"n_policy": {"production_pool": "full Part C gated science-comp pool", "interim_top_n": "disabled"},
"swap_meta": {"superseded_model": "masterstar_epsf_pre_gated_20260822.fits", "superseded_n_stars_used": 1475}
```

Iteration curve: all-zero status-3 failures at n=67 (flat convergence).

---

## S6b - Draft 517 first production gated build

| Item | Value |
|------|------:|
| n_stars_used | **66** (67 science-scope, 1 lost to isolation) |
| Guard drops | **0** |
| Path | `Archive/Drafts/draft_000517/platesolve/NoFilter_60_2/masterstar_epsf.fits` |

---

## S6c - F6 PSF merge (516)

| Item | Value |
|------|------:|
| Frames total | **134** |
| Frames written (psf_flux populated) | **134/134** |
| INV-PSF-ADDITIVE-01 | **PASS** (dao_flux unchanged vs accept backup on 20/20 sample frames) |
| Aperture columns | Untouched (non-psf_* identity enforced at merge) |

Harness note: merge completed; post-run JSON capture hit `AttributeError` on invariants list shape
(fixed in `epsf_valid_02_s6_swap.py`); on-disk merge result verified independently.

Wall time: ~4.6 h (134 science-light frames, full science-set PSF photometry).

Artifacts: `dev/results/context/session_20260822_epsf_valid_02_s6/s6_execution.json`.

---

## Addendum 2 - S5 retraction

`dev/results/CURSOR_RESULT_EPSF_VALID_02_S5.md` Part D header marked:

> SUPERSEDED by `CURSOR_RESULT_EPSF_VALID_02_S5B.md` (metric defect: raw-flux offsets).

---

## Addendum 3 - N policy (docs + meta)

Recorded in:

- `docs/VYVAR_DECISIONS.md` (EPSF-VALID-02 S6 section)
- `masterstar_epsf_meta.json` -> `n_policy` on 516/517 builds
- `docs/VYVAR_INVARIANTS.md` (INV-EPSF-BUILD-GUARD-01)

Policy summary:

- Production: **full Part C gated science-comp pool**; INTERIM top-N=200 **disabled**.
- Certificates: scale-aligned per-star RMS < 3x median ERR budget; raw offsets = bookkeeping.

---

## S6d - Docs sync

| Doc | Update |
|-----|--------|
| `docs/VYVAR_STATE.md` | ePSF production-ready on 516/517; EPSF-VALID-02 CLOSED |
| `docs/VYVAR_DECISIONS.md` | Swap, INV-PSF-ADDITIVE-01, N policy, guard |
| `docs/VYVAR_ROADMAP.md` | EPSF-VALID-02 done; EXPORT-PARITY-01 HIGH stays |
| `docs/VYVAR_INVARIANTS.md` | INV-EPSF-BUILD-GUARD-01 |

---

## S6e - Gates

| Gate | Status | Detail |
|------|--------|--------|
| `--fast` | **OVERALL PASS** | HEAD `777f10e`; 1507 passed, 32 skipped (~581 s) |
| `--full` | **OVERALL PASS** | HEAD `777f10e`; core **9902d918** n=121, extended **472bc9e4** n=179 (~2541 s pipeline) |

---

## Gate status

| Check | Status |
|-------|--------|
| Production ePSF swap 516 | Done (on disk) |
| 517 first gated build | Done |
| PSF merge 134/134 | Done |
| Guard commit | `f97615a` |
| Milan push | Pending review |

---

## Files changed (this arc)

| File | Role |
|------|------|
| `src_py/psf_photometry.py` | Guard + meta blocks |
| `dev/tests/test_epsf_build_guard.py` | Guard tests |
| `dev/sandbox/epsf_valid_02_s6_swap.py` | S6 execution harness |
| `docs/VYVAR_*` | STATE, DECISIONS, ROADMAP, INVARIANTS |
| `dev/results/CURSOR_RESULT_EPSF_VALID_02_S5.md` | Part D superseded note |
| `dev/results/CURSOR_RESULT_EPSF_VALID_02_S6.md` | This deliverable |
| On-disk (not git): `Archive/Drafts/draft_000516/.../masterstar_epsf*` | Swapped model + backup |
| On-disk (not git): `Archive/Drafts/draft_000517/.../masterstar_epsf*` | First gated 517 model |
| On-disk (not git): 516 proc CSVs | PSF columns merged |

---

## Milan handoff

1. Review dashboard PSF curves by eye (516, post-gated model).
2. Push today's commit series when satisfied.
3. `--full` recut on frozen snapshot recommended before next anchor-sensitive work.
