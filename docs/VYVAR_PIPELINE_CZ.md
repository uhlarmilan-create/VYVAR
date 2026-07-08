# VYVAR -- Pipeline (cesky manual pro clanek)

**Verze:** 2026-06-09 (revize oproti PDF z 3. 6. 2026)

Jazyk: cestina, ASCII (bez diakritiky) -- v souladu s tvrdym pravidlem projektu.
Ciselne a parametricke udaje jsou ukotveny v realnem kodu (baseline commit `28fdafa` a zmeny
session 2026-06-09). Sekce [Overeni pred odevzdanim clanku](#overeni-pred-odevzdanim-clanku-doporuceni)
na konci shrnuje hodnoty, ktere doporucujeme jeste jednou potvrdit proti zivemu kodu / harness.

**Souvisejici anglicke dokumenty:** `VYVAR_STATE.md`, `VYVAR_DECISIONS.md`, `VYVAR_PROCESS.md`,
`VYVAR_VALIDATION.md`, `VYVAR_PARAMS.md`, `VYVAR_CALIBRATION.md`, `CITATIONS.bib`.

---

## Obsah

1. [Hlavicka / verze](#0-hlavicka--verze)
2. [comp_qa -- order-independent locus (CQ-C)](#1-comp_qa--magnitudovy-locus-je-nyni-order-independent-cq-c)
3. [PSF fotometrie (gated OFF)](#2-psf-fotometrie----publication-grade-na-syntetickych-fine-scale-datech-gated-off)
4. [Blind plate-solver](#3-blind-plate-solver----prestaveny-8-6-2026)
5. [Byte-identity](#4-byte-identity----princip-nezmenen-kotva-v-prechodu)
6. [Gaia katalog a build pro noveho uzivatele](#5-gaia-katalog----g--175-a-build-pro-noveho-uzivatele)
7. [Fail-safety / hygiena (#4)](#6-fail-safety--hygiena-4)
8. [Kalibrace magnitud -- datovy tok](#7-kalibrace-magnitud--datovy-tok)
9. [Overeni pred odevzdanim clanku](#overeni-pred-odevzdanim-clanku-doporuceni)

---

## 0) HLAVICKA / VERZE

- **Datum manualu:** 3. 6. 2026 -> **2026-06-09**.
- **pytest (default `-m "not slow"`):** 226 passed / 6 skipped pri pritomnem referencnim
  draftu `draft_000366`; bez draftu na disku 224 passed / 8 skipped (2x SHA-guard testy
  preskoci s `draft_000366 not present`). Tri pomale CQ-C testy jsou `@pytest.mark.slow`
  (deselect v default behu; spustit na vyzadani).

---

## 1) comp_qa -- magnitudovy locus je nyni ORDER-INDEPENDENT (CQ-C)

**Modul:** `comp_qa_core.py` (`build_locus`, `compute_comp_qa`).

### Mechanika locusu

- Body (instrumentalni magnituda, sigma_IQR) z leave-one-out analyzy komparacnich hvezd se
  binuji po **0.5 mag** (`_MAG_BIN = 0.5`).
- V kazdem binu: **median** jako lokalni ocekavani a robustni rozptyl **MAD * 1.4826**
  (s bezpecnym fallbackem, kdyz je MAD nulovy nebo bin maly).
- Hvezda se posuzuje proti locusu na sve magnitude -- slabe komparace se tedy neodmitaji jen
  proto, ze prirozene sumi.

### Zmena CQ-C (2026-06-09)

Locus je nyni **FIXNI**, pocitany **jednou** z celeho pass-1 poolu vsech (cil, komparace) bodu.

Drive se locus **prepoctival** uvnitr smycky vyrazovani z bodu filtrovanych pres narustajici
mnozinu `dropped_global` -- a protoze ta rostla v poradi zpracovani cilu, locus (a tedy i to,
ktere komparace se oznaci) zavisel na **poradi zpracovani** (a byl cirkularni: vyrazeni tvarovala
locus, ktery tvaroval vyrazeni). Iterativni prepocty byly odstraneny; `dropped_global` zustava
jen pro evidenci prezivajicich.

**Dusledek:** oznaceni komparaci je nyni nezavisle na poradi cilu -- vlastnost dulezita pro
reprodukovatelnost (a obhajitelnost v clanku).

**Overeno:** stejny vstup ve >= 5 ruznych zamichanych poradich cilu dava **bajtove identicky**
vystup comp_qa (legacy iterativni verze se napric poradimi lisila). Testy:
`tests/test_comp_qa_fix_once_locus.py` (fix-once), `tests/comp_qa_legacy_iterative.py`
(pre-CQ-C komparator).

**Dopad re-baseline** (referencni draft `draft_000366`): **1** zmena flagu, **1** cil se zmenou
`n_clean` (+1), **0** zmen trust. Maly, hranicni -> re-baseline prijat.

Viz `VYVAR_DECISIONS.md` (CQ-C), `VYVAR_PROCESS.md` (re-baseline tabulka).

---

## 2) PSF fotometrie -- publication-grade na syntetickych fine-scale datech (GATED OFF)

**Modul:** `psf_photometry.py`.

Tri rezimy, ktere delaji PSF na jemnem vzorkovani mereni-presnym a kalibrovanym:

### (a) Vahy fitu sky-only (`psf_weight_mode = sky_only`)

Vahy pixelu pri PSF fitu pouzivaji jen rozptyl oblohy:

    w_i^-1 = Var(sky) = sky/gain + (read_noise_e/gain)^2

Zdrojovy Poissonuv clen je z vah vypusten (reportovana `psf_flux_err` se nadale skaluje z plne
variance). Tim se odstrani textbookovy flux-dependent PSF-fit weighting bias (jasne/slabe hvezdy
by jinak mely PSF-modelove zavisly pomer toku). Grounded v Astier et al. 2013 (SNLS) a Lacroix,
Regnault et al. 2025 (ZTF SNe Ia scene modeling).

**Vysledek (V3d harness, fine-scale):** mid-mag bias +4.5% -> ~1%, drift vyrazne nizsi,
bez-sumovy pripad ~0; PSF presnejsi nez apertura zhruba od mag ~13.

### (b) Sendvicova variance (`psf_err_mode = sandwich_skyonly`)

Sky-only vahy jsou suboptimalni pro jasne hvezdy, takze chyba pocitana "optimalne" by podcenovala
skutecny rozptyl. Reportovana chyba je proto variance **skutecneho** vazeneho estimatoru:

    Var(f^) = [ sum w_i^2 * sigma_true,i^2 * P_i^2 ] / [ sum w_i * P_i^2 ]^2

kde `w_i = 1/sigma_sky^2` a `sigma_true^2 = sigma_sky^2 + f^ * P_i / gain`.

Meni jen reportovanou chybu, nikoli tok. **Vysledek:** P3 (reportovana/skutecna chyba) ~1 napric
mag 12-17 (drive ~0.56 na mag12 -> over-confident).

### (c) Odhad oblohy z rezidua (`psf_sky_method = residual_annulus`)

Obloha se bere z dat minus fitnute PSF modely -- robustni i pro hustsi pole.

### (d) EPSF-1 (diagnosticky)

Nativni FWHM ePSF z azimutalne binovaneho radialniho profilu (linearni interpolace v p=0.5),
nahrazuje drivejsi nerobustni odhad. Pouziva se jen v QC/varovani, nevstupuje do toku ani do
`assess_psf_quality`.

### GATING (klicove pro clanek)

PSF je **GATED OFF** v produkcnim LC. Na sirokem poli (Jirny, Carl-Zeiss, ~9.77"/px) vyhrava
apertura kvuli podvzorkovani. PSF se odemkne az pro jemne vzorkovani (Newton, Dablice,
~0.65"/px) **po pruchodu charakterizacni branou** na realnych hustych datech.

**DULEZITE:** vse vyse je validovano **POUZE na syntetickych fine-scale datech** (inject-and-recover,
V3d) -- realny dukaz na realnych hustych datech je teprve pred nami; charakterizacni brana je presne
proto, ze synteticka validace je nutna, nikoli postacujici.

Reproducibilni proof skripty (spustit z korene repa):

- `tests/validation/run_v3d_weight_proof.py` (sky-only vahy / mid-mag bias)
- `tests/validation/run_v3d_sandwich_proof.py` (P3 sendvicova chyba)
- `tests/validation/run_v3d_clean_sky_proof.py` (rezidualni obloha)

`.md`/`.json` reporty vzniknou v `tests/validation/data/tier_v3d/` az po spusteni daneho
skriptu (generovane vystupy nejsou v gitu).

---

## 3) Blind plate-solver -- prestaveny (8. 6. 2026)

Verze PDF z 3. 6. tuto prestavbu jeste nema. Aktualni stav (`vyvar_blind_solver.py` +
`GAIA_DR3/build_blind_index.py`):

- **Density-matched Gaia triangle index:** per-bunka cap na hustotu hvezd (`stars_per_cell = 95`
  pro fine tier pri vystavbe indexu), 8-NN trojuhelnikove hashe.
- **Image-side lokalni kNN** zrcadli per-star 8-NN indexu (`k` cteno z PKL jako jediny zdroj
  pravdy).
- **DBSCAN vote clustering** kandidatu (haversine metrika; `eps`, `min_samples`, `min_votes` jako
  prahy) misto holeho poctu hlasu.
- **Geometricke overeni** top-N kandidatu pres match-fraction (ne vote count) a cluster-level
  RANSAC overeni WCS.
- **Dvoutierovy index:** fine (dlouhe ohnisko, Newton) + wide (Carl-Zeiss ~9.77"/px), scale-aware
  orchestrator vybira tier podle meritka.
- **A/B:** `verify_mag_limit = 14` lepsi nez 16 (shodne 10/10 vyreseni, -28% runtime, lepsi
  odstup pravda/lez). Default v `config.py`: `verify_mag_limit = 14.0`.
- Index se stavi skriptem `GAIA_DR3/build_blind_index.py` (mag <= 14, tier `fine|wide|both`).

---

## 4) Byte-identity -- princip nezmenen, kotva v prechodu

- Princip stejny: fotometricke vystupy musi byt po zmenach kodu SHA-256 identicke s referencnim
  behem (read-only zmeny).
- **Jadrova sada** (`lightcurve_*.csv` + `comp_quality_*.json` +
  `comparison_stars_per_target.csv`, **283 souboru**) = **`770966c3...`** a **DRZI** i po CQ-C:
  comp_qa je read-only vrstva po Phase 2A, takze locus meni jen QA sidecary a reportovany
  `n_clean`, nikoli fotometrii.
- **Nova, sirsi referencni sada** (**426 souboru**) vc. `comp_qa_*.json` sidecaru =
  **`edbd97e7...`**, aby byl CQ-C nadale pokryty.
- Kotva `draft_000366` + stary katalog se prakticky **retiruje**: drafty byly smazany na dev
  stroji, katalog se prohlubuje na **G <= 17.5**; nova kotva se ustavi na cerstvem draftu
  postavenem na novem katalogu (default build skriptu je zatim G <= 16.5 -- viz
  `build_gaia_catalog.py --mag-limit`).

SHA helper (kotvy): `tests/photometry_sha.py`. Guard testy:
`tests/test_photometry_sha_baseline.py` (skip, pokud draft na disku chybi).

---

## 5) Gaia katalog -- G <= 17.5 a build pro noveho uzivatele

- Lokalni Gaia DR3 SQLite se stavi skriptem `GAIA_DR3/build_gaia_catalog.py` (full-sky, default
  G <= 16.5, cil prohloubeni **G <= 17.5**), resume-safe pres tabulku `strip_progress`,
  `source_id` jako INTEGER PRIMARY KEY.
- Trojuhelnikove indexy (PKL) se stavi `GAIA_DR3/build_blind_index.py` (mag <= 14, fine+wide).
- **Build pro noveho uzivatele:** skripty se pousti **z klonu repozitare** (sdili
  `gaia_catalog_id.py`, ktery resi presnost 19-cifernych `source_id` pres `Decimal` -- naivni
  `int(float)` by je poskodil). `--out` / `--fine-out` / `--wide-out` presmeruji velke soubory
  kamkoli. Skript hleda koren repa smerem nahoru a pri spusteni mimo klon konci jasnou hlaskou
  (ne `ModuleNotFoundError`). Viz `README.md` sekce "Building the Gaia catalog".

Priklad:

```bash
python GAIA_DR3/build_gaia_catalog.py --mag-limit 16.5 --out <path-to-db>
python GAIA_DR3/build_blind_index.py --db <path-to-db> --tier both
```

---

## 6) Fail-safety / hygiena (#4)

- **Zapis WCS MASTERSTARu je fail-closed:** pri chybe zapisu vyjimka -> `solved=False` ->
  stavajici guard blokuje Phase 2A pro dany draft (zadna ticha zastarala WCS dal).
  Modul: `vyvar_platesolver.py`.
- **Kontrola edge-ok je fail-open + hlasity flag:** pri selhani jsou vsechny hvezdy edge-ok, ale
  nastavi se `edge_filter_failed` / `edge_filter_note` pouze na `variability_candidates.csv`
  (mimo SHA sadu), `LOGGER.error` a status na titulce reportu. Modul: `photometry_core.py`.
- Odstraneny mrtve UI moduly (`ui_photometry_results`, `ui_suspected_lightcurves`; nahrazeny
  `render_aperture_photometry` / `render_variability_dashboard`).

---

## 7) Kalibrace magnitud -- datovy tok

### Proc CSV schema (`proc_*.csv`)

Sloupec **`mag`** = Gaia katalogove G (hodnota v dobe krizoveho parovani s katalogem; pro danou
hvezdu konstantni pres celou noc). **Neni** to instrumentalni magnituda snimku — pro vedeckou
fotometrii pouzivejte **`dao_flux`** (viz `docs/VYVAR_PROCESS.md`).

Kompletni linie sloupcu `mag_inst` -> `mag_calib` -> korekce (GS11, SG, CT, AC) ->
kanonicky **`mag_calib_final`**, tabulka spotrebitelu (export, PDF, `lc_rms`, variability),
a dokumentace sloupce `err` jsou v dedikovanem manualu:

**`docs/VYVAR_CALIBRATION.md`** (verze 2026-06-22, Path A commit `be3e193`).

Shrnuti: vsechny publikacni vystupy (AAVSO/VarAstro MAG, LC figury v PDF) ctou
`mag_calib_final`; scatter metriky (`lc_rms`, trust) zustavaji na `mag_calib` (CT/AC jsou
konstantni posuvy). Rozhodnuti: `VYVAR_DECISIONS.md` (Path A); audit G5-F011.

---

## OVERENI PRED ODEVZDANIM CLANKU (doporuceni)

Tyto konkretni hodnoty doporucujeme potvrdit proti zivemu kodu / vystupum harness tesne pred
odevzdanim (pochazi z behu session 2026-06-09; sandbox na `28fdafa`, harness zde znovu
nepoustel):

| Oblast | Hodnota k potvrzeni | Zdroj |
|--------|---------------------|-------|
| PSF mid-mag bias | +4.5% -> ~1% | `tests/validation/run_v3d_weight_proof.py` |
| PSF vs apertura | PSF presnejsi od ~mag 13 | `tests/validation/run_v3d_fine_scale.py` |
| P3 (sandwich err) | ~0.56 -> ~1 na mag12 | `tests/validation/run_v3d_sandwich_proof.py` |
| SEP cross-validace (offline harness, not pipeline) | ~0.2%/snimek | `xval_run.py` |
| Blind solver | 10/10 solve rate; `verify_mag_limit=14` | `scripts/blind_verify_mag_ab.py`, `scripts/blind_solve_rate.py` |
| DBSCAN prahy | `eps`, `min_samples`, `min_votes` | `vyvar_blind_solver.py`, `config.py` |
| Index hustota | `stars_per_cell=95` (fine build) | `GAIA_DR3/build_blind_index.py`, PKL meta |
| CQ-C re-baseline | 1 flag flip, 1x n_clean +1, 0 trust | `tests/test_comp_qa_fix_once_locus.py` |
| SHA kotvy | `770966c3` (283), `edbd97e7` (426) | `tests/photometry_sha.py`, `tests/test_photometry_sha_baseline.py` |

---

*Posledni revize: 2026-06-09. Commit reference: `a339b56` (origin/main).*
