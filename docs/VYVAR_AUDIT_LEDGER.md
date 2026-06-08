# VYVAR — Audit ledger

**Started:** 2026-06-08 · **Workflow:** tooling-first triage → targeted fixes → manual critical-path read

Stav modulov a disposícia nálezov. Obnoviteľný register — nie čierna diera.

**Stavy:** `todo` · `auditing` · `done` · `deferred`

| Modul | Stav | Nálezy (Fáza 0→1) | Disposícia |
|---|---|---|---|
| `photometry_report.py` | done | F821×27 → 0 (TYPE_CHECKING); F841×11 → 1; F811 cm/mm | fix batch 1 — kozmetika + dead locals |
| `photometry_core.py` | done | batch 2: removed redundant `c1_stderr@7141`; `lc_df` read-guard preserved | F841 `ra_ms`/`gaia_teff` deferred |
| `comp_selection_per_target.py` | done | removed `dist_score`, `rms_f2`; `g_teff` deferred | batch 2 cleanup |
| `comp_pool_rms.py` | done | F841 `avail_cols` | removed dead assignment |
| `comp_qa.py` | todo | F841 `lc_map` | triage — možná zabudnutá logika |
| `comp_qa_core.py` | todo | — | Fáza 2 manual read |
| `trust_flag_core.py` | auditing | Phase 2 findings A–F (see AUDIT_FINDINGS) | fixes → ROADMAP next session |
| `calibration.py` | todo | — | Fáza 2 manual read |
| `database.py` | todo | — | Fáza 2 manual read |
| `vyvar_platesolver.py` | todo | F841 `center` | Fáza 2 + F841 triage |
| `pipeline.py` | todo | F841×2 (`n0`, `cfg`) | deferred — obrie súbor, split track |
| `psf_photometry.py` | todo | F841 `fit_shape` | triage |
| `export_reports.py` | done | F401×2 | ruff --fix batch 1 |
| `config.py` | done | — | no F841 in scope |
| `ui_*` (aggregate) | deferred | F841×3 v UI | nízka priorita vs core |
| `tess_verify.py` | deferred | F841×2 | TESS side path |
| `xval_run.py` | deferred | F841×2 | offline harness |
| `psf_runner.py` | deferred | F841×1 | dev CLI |
| `orchestrator/` | done | F401/F541 | ruff --fix batch 1 |

## Dávka 1 (2026-06-08) — hotovo

| Check | Výsledok |
|---|---|
| F821 | **0** (bolo 27) |
| F811 | **0** (bolo 7) |
| F401/F541 | **0** (auto-fix + review) |
| F841 | **22** (bolo 44) |
| `pytest tests` | **174 passed, 6 skipped** |
| Byte-identita fotometrie | neoverená v tejto dávke (len refaktor mŕtvych premenných / importy) |

## Dávka 2 (2026-06-08) — session close

F841 batch 2 cleanup: removed `dist_score`, `rms_f2`, redundant `c1_stderr@7141`;
`lc_df@7786` read-guard preserved (`pd.read_csv` without binding); `g_teff`/`gaia_teff`
deferred (benign).

| Check | Výsledok |
|---|---|
| `pytest tests` | **174 passed, 6 skipped** |
| Photometry byte-identity (`draft_000366`, 284 artifacts) | **OK** — SHA-256 `ad12325d262e913dc57fa0e805e07c2115aec5005268c704177d7fb72856aa69` unchanged |
| PDF overflow (`draft_000366`) | **0** violations (160 pages) |
| commit | *(filled after push)* |

## Ďalšie kroky (Fáza 1 pokračovanie)

1. **F841 kritická cesta** — `comp_selection_per_target`, `photometry_core` (`c1_stderr`, `lc_df`), jeden fix = jeden test ak rizikové.
2. **`except Exception: pass`** — kritická cesta (683× broad except; 9× explicit pass) — po jednom, minimum log.
3. **Fáza 2 manual read** — poradie: `photometry_core` → `comp_selection_per_target` → `comp_qa_core` → `trust_flag_core` → `calibration` → `database` → `vyvar_platesolver`.
4. **Split track** — `pipeline.py` / `photometry_core.py` (samostatný spec, byte-identita).

## Nástroje

```bash
python tmp/_gen_audit_findings.py   # regeneruje docs/VYVAR_AUDIT_FINDINGS.md
python -m ruff check . --select F821,F811,F841 --statistics
python -m pytest tests -q
```
