# GAIA_DR3 — cleanup inventory (read-only scan)

**Dátum scanu:** 2026-06-08 · **Autor scanu:** Cursor (read-only)  
**Účel:** mapa `súbor → kto ho používa` pred upratovaním. **Žiadne súbory neboli mazané ani presúvané.**

Scan: 23 súborov v `GAIA_DR3/` (bez `__pycache__`), sha256 prvých 8 hex znakov, grep celej repo (`.py`, `.json`, `.md`, UI, …).

---

## Load-bearing cesty z konfigurácie

| Kľúč | `config.json` | `config.py` default | UI (Settings) |
|---|---|---|---|
| `gaia_db_path` | `GAIA_DR3\vyvar_gaia_dr3.db` | `""` (prázdny; JSON má prednosť) | `GAIA_DB_PATH` (`ui_settings.py`) |
| `blind_index_path` | `GAIA_DR3\gaia_triangles.pkl` | `gaia_triangles.pkl` | `BLIND_INDEX_PATH` (`ui_settings.py`) |
| `blind_index_series` | `GAIA_DR3\blind_index_series.json` | `blind_index_series.json` | *(nie je v UI; len config)* |
| `blind_index_select_mode` | `auto` | `auto` | — |

**Manifest `blind_index_series.json` → tier PKL:**

| Tier | Súbor | Použitie |
|---|---|---|
| `fine` | `gaia_triangles_fine.pkl` | Newton / fine plate scale (`vyvar_blind_series.py`, `blind_index_select_mode=auto`) |
| `wide` | `gaia_triangles_wide.pkl` | Wide rig (`plate_scale ≥ 5″/px`, draft_365, …) |

**Runtime orchestrátor:** `vyvar_blind_series.solve_blind_with_series` → `vyvar_platesolver` (blind fallback).

---

## Duplicitné skupiny (rovnaký sha256[:8])

| sha256[:8] | Súbory | Veľkosť každý |
|---|---|---|
| `35619b96` | `gaia_triangles.pkl`, `gaia_triangles_fine.pkl`, `gaia_triangles_mag14.pkl` | 1341.14 MB |

**Hypotéza potvrdená:** tri PKL sú **byte-identické** (1× dáta, 3× kópia na disku ≈ **4.0 GB navyše**).

---

## Hlavná tabuľka

| súbor | MB | mtime | sha256(8) | referenced_by (mimo seba) | trieda | navrhovaná akcia |
|---|---:|---|---|---|---|---|
| `vyvar_gaia_dr3.db` | 9599.75 | 2026-04-28 | `d7eb163c` | `config.json`, `config.py`, `ui_settings` (GAIA_DB_PATH), `blind_index_build.py`, `build_gaia_blind_index.py`, `build_blind_index_series.py`, `blind_density_runbook.py`, `vyvar_platesolver`/`database.py` (runtime), `tests/test_dilution.py`, pilot/chiandh skripty, … (19 ref) | **KEEP-LOADBEARING** | **keep** — produkčná Gaia SQLite |
| `blind_index_series.json` | 0.0 | 2026-06-04 | `4e3563bf` | `config.json`, `config.py`, `vyvar_blind_series.py`, `build_blind_index_series.py`, docs (8 ref) | **KEEP-LOADBEARING** | **keep** — manifest tierov |
| `gaia_triangles_fine.pkl` | 1341.14 | 2026-06-04 | `35619b96` | `blind_index_series.json` (tier `fine`), `build_blind_index_series.py`, docs (4 ref) | **KEEP-LOADBEARING** | **keep** — kanonický fine tier |
| `gaia_triangles_wide.pkl` | 673.78 | 2026-06-04 | `d179f7c4` | `blind_index_series.json` (tier `wide`), `diagnose_blind_solver_wide.py`, `diagnose_wide_true_triangle_shape.py`, `build_blind_index_series.py` (5 ref) | **KEEP-LOADBEARING** | **keep** — wide tier |
| `gaia_triangles.pkl` | 1341.14 | 2026-06-04 | `35619b96` | `config.json` (`blind_index_path`), `config.py`, `ui_settings` (BLIND_INDEX_PATH), `vyvar_blind_solver.py`, diagnózy, runbooky (17 ref) | **DUP** + **KEEP-LOADBEARING** | **consolidate** — po zmene `blind_index_path` → `gaia_triangles_fine.pkl` zmazať duplikát |
| `gaia_triangles_mag14.pkl` | 1341.14 | 2026-06-04 | `35619b96` | `build_blind_index_series.py` (zdroj pre fine copy), `blind_index_regression.py`, docs (4 ref) | **DUP** | **delete** po konsolidácii (ponechať `fine.pkl`) |
| `blind_index_build.py` | 0.01 | 2026-06-08 | `b4d68096` | `build_gaia_blind_index.py`, `gaia-dr3_index_solver.py`, `blind_density_runbook.py`, `build_blind_index_series.py` (import) | **KEEP-SCRIPT** | **keep** — kanonická implementácia index buildu |
| `blind_index_cells.py` | 0.0 | 2026-06-04 | `73fb798b` | `blind_index_build.py` (import), `tests/test_blind_knn_construction.py` | **KEEP-SCRIPT** | **keep** — knižnica per-cell cap |
| `build_gaia_blind_index.py` | 0.0 | 2026-06-04 | `9925e106` | `ui_settings.py`, `vyvar_blind_solver.py`, `config.py` docs, `solver_audit.txt` (7 ref) | **KEEP-SCRIPT** | **keep** — dokumentovaný entry-point (wrapper na `blind_index_build`) |
| `gaia_dr3_make_fast.py` | 0.02 | 2026-04-25 | `ded12837` | `scripts/pilot_palomar7_deep_gaia_ab.py` (import `init_db`, …) | **KEEP-SCRIPT** | **keep** — živý DB builder (field DB); výstup default `vyvar_gaia_dr3_v2.db` |
| `gaia_triangles_test.pkl` | 2960.36 | 2026-06-04 | `72df4330` | `scripts/blind_density_runbook.py` (`--test-index`), `gaia-dr3_index_solver.py` (dokumentácia) | **ORPHAN** (runtime) | **delete** alebo presun do `Archive/` — experimentálny index (~3 GB) |
| `gaia_triangles.pkl.bak_mag14_premdensity` | 4952.32 | 2026-04-20 | `e89fb93f` | *(0 ref mimo scan cache)* | **ORPHAN** | **delete** — záloha pred pre-density; regresia 10/10 hotová |
| `vyvar_gaia_dr3_v1.db` | 7797.38 | 2026-04-02 | `901fd567` | *(0 ref mimo scan cache)* | **ORPHAN** | **delete** po potvrdení, že `vyvar_gaia_dr3.db` je kompletná náhrada |
| `vyvar_gaia_dr3_v3.db` | 0.01 | 2026-05-18 | `3678fa65` | `gaia-dr3_make_v3.py`, `VYVAR_JOURNAL.md` (stub 8 KB) | **ORPHAN** | **delete** — nedokončený build (len schéma) |
| `vyvar_gaia_dr3_pal7_field.db` | 11.01 | 2026-06-03 | `24ed54b9` | `palomar7_*` skripty, `forced_photometry_pal7.py`, `pilot_palomar7_deep_gaia_ab.py`, … (13 ref) | **SCRATCH** | **keep** do ukončenia Pal7 pilotu, potom archivovať |
| `vyvar_gaia_dr3_chiandh_field.db` | 10.67 | 2026-06-03 | `5e8d69c6` | `chiandh_*` skripty (8 ref) | **SCRATCH** | **keep** do ukončenia χ And / h Per runu, potom archivovať |
| `vyvar_gaia_dr3_chiandh_field_build.json` | 0.0 | 2026-06-03 | `8d633379` | `chiandh_build_field_db.py`, seba | **SCRATCH** | keep s párovým `.db` |
| `vyvar_gaia_dr3_m67_field.db` | 1.77 | 2026-06-03 | `e8be89d4` | `m67_*` skripty (8 ref) | **SCRATCH** | **keep** do ukončenia M67 runu, potom archivovať |
| `vyvar_gaia_dr3_m67_field_build.json` | 0.0 | 2026-06-03 | `a48737b1` | `m67_field.db` (vnútri JSON) | **SCRATCH** | keep s párovým `.db` |
| `gaia-dr3_index_solver.py` | 0.0 | 2026-06-04 | `4f1067dc` | `build_gaia_blind_index.py` (historický odkaz), `solver_audit.txt` | **ORPHAN** (mŕtvy entry) | **delete** — duplicitný wrapper; nahradený `build_gaia_blind_index.py` / `blind_index_build.py` |
| `gaia-dr3_make.py` | 0.0 | 2026-04-02 | `919eb201` | `vyvar_gaia_dr3.db` (DB_NAME v skripte) | **ORPHAN** (mŕtvy) | **delete** alebo `Archive/` — HEALPix builder v1; produkcia už na `vyvar_gaia_dr3.db` |
| `gaia-dr3_make_v2.py` | 0.0 | 2026-04-24 | `ba53ed27` | *(0 ref)* | **ORPHAN** (mŕtvy) | **delete** — výstup `vyvar_gaia_dr3_turbo.db` (neexistuje na disku) |
| `gaia-dr3_make_v3.py` | 0.01 | 2026-05-16 | `70360d1c` | *(0 ref mimo seba)* | **ORPHAN** (mŕtvy) | **delete** — v3 stub; obsahuje hardcoded TAP credentials (**bezpečnostné riziko**) |

**Súhrn veľkosti:** ~**29.5 GB** v `GAIA_DR3/` (z toho ~**9.0 GB** potenciálne uvoľniteľné: v1.db + bak + test.pkl + 2× dup PKL).

---

## Maker skripty: živý / mŕtvy

| Skript | `py_compile` | Volaný z repo | Entry-point | Výstupná DB / PKL | MAG / DEC parametre |
|---|---|---|---|---|---|
| `blind_index_build.py` | OK | `build_gaia_blind_index.py`, `build_blind_index_series.py`, `blind_density_runbook.py` | `python blind_index_build.py --stars-per-cell N` | `--out` PKL (default `gaia_triangles.pkl`) | `--mag-limit` (default 16), `--cell-deg` 1.0, `--db` → `vyvar_gaia_dr3.db` |
| `blind_index_cells.py` | OK | import z `blind_index_build`, test | knižnica | — | per-cell brightest cap |
| `build_gaia_blind_index.py` | OK | dokumentácia, UI caption | `python build_gaia_blind_index.py` | `gaia_triangles.pkl` z `vyvar_gaia_dr3.db` | mag 16, cell 1°, SPC z env |
| `gaia-dr3_index_solver.py` | OK | len audit / historický odkaz | duplicitný wrapper | `gaia_triangles.pkl` | = `blind_index_build` defaults |
| `gaia-dr3_make.py` | OK | **mŕtvy** | HEALPix TAP | `vyvar_gaia_dr3.db` | MAG 16, DEC −20…90 |
| `gaia-dr3_make_v2.py` | OK | **mŕtvy** | Dec pásy turbo | `vyvar_gaia_dr3_turbo.db` | MAG 15.5, DEC −20…90, step 0.5° |
| `gaia-dr3_make_v3.py` | OK | **mŕtvy** (stub DB) | Dec pásy + HRD stĺpce | `vyvar_gaia_dr3_v3.db` | MAG 15.5, DEC −20…90, step 0.1° |
| `gaia_dr3_make_fast.py` | OK | **`pilot_palomar7_deep_gaia_ab.py`** (import) | strip TAP builder | `vyvar_gaia_dr3_v2.db` | MAG 16, DEC −20…90, RA/DEC step env |

**Živé build reťazce (2026-06):**

1. **Produkčná DB:** `vyvar_gaia_dr3.db` (pravdepodobne z `gaia_dr3_make_fast` alebo manuálne premenovaná v2).
2. **Blind index:** `scripts/build_blind_index_series.py` → `fine.pkl` + `wide.pkl` + manifest.
3. **Field DB (CT):** `*_build_field_db.py` skripty v `scripts/` + `gaia_dr3_make_fast` API.

**Odporúčaný jediný „script #1“ po upratovaní:** `blind_index_build.py` + `build_gaia_blind_index.py` (index), `gaia_dr3_make_fast.py` (DB).

---

## Overenie hypotéz (Claude)

| Hypotéza | Výsledok scanu |
|---|---|
| `mag14.pkl` == `fine.pkl` == `pkl` (sha256) | **POTVRDENÉ** (`35619b96`, 1341 MB × 3) |
| `gaia_triangles_test.pkl` ORPHAN | **Čiastočne** — len runbook/dokumentácia, nie runtime |
| `bak_mag14_premdensity` ORPHAN | **POTVRDENÉ** (0 ref) |
| `vyvar_gaia_dr3_v1.db` ORPHAN | **POTVRDENÉ** (0 ref) |
| `vyvar_gaia_dr3_v3.db` stub ORPHAN | **POTVRDENÉ** (8 KB, nedokončený) |
| `*_field.db` + `build.json` SCRATCH | **POTVRDENÉ** — aktívne v Pal7/M67/χAnd skriptoch |
| Z make skriptov živý jeden | **Čiastočne** — `gaia_dr3_make_fast.py` (DB) + `blind_index_build.py` (PKL); ostatné make v1/v2/v3 mŕtve |
| `gaia-dr3_index_solver.py` mŕtvy | **POTVRDENÉ** — nahradený wrappermi |
| `build_gaia_blind_index.py` = wrapper | **POTVRDENÉ** — stále dokumentovaný entry-point, volaný manuálne |

---

## Navrhované fázy upratovania (až po spoločnom OK)

1. **Konsolidácia PKL dup:** `blind_index_path` → `gaia_triangles_fine.pkl`; zmazať `gaia_triangles.pkl` + `gaia_triangles_mag14.pkl` (−2.7 GB).
2. **Veľké ORPHAN:** `bak_mag14_premdensity` (−4.9 GB), `gaia_triangles_test.pkl` (−3.0 GB), `vyvar_gaia_dr3_v1.db` (−7.8 GB).
3. **Mŕtve skripty:** `gaia-dr3_make.py`, `_v2.py`, `_v3.py`, `gaia-dr3_index_solver.py` → `Archive/` alebo delete.
4. **Field SCRATCH:** po uzavretí pilotov presun do `Archive/GaiaFieldDB/` (nie hneď).
5. **`vyvar_gaia_dr3_v3.db` stub:** delete (8 KB).

---

## Scan artefakty (tento beh)

Tento scan vytvoril pomocné súbory (nie súčasť produkcie):

- `GAIA_DR3/_hash_cache.json`
- `GAIA_DR3/_refs_cache.json`

Odporúčanie: zmazať po schválení inventára (alebo pridať do `.gitignore`).

---

*Žiadna akcia nebola vykonaná okrem zápisu tohto súboru.*
