# VYVAR -- Kalibrace magnitud -- datovy tok

**Verze:** 2026-06-22 (Path A: kanonicky sloupec `mag_calib_final`, commit `be3e193`)

Jazyk: cestina, ASCII (bez diakritiky) -- stejne jako `VYVAR_PIPELINE_CZ.md`.
Vsechny vzorce a cisla jsou ukotveny v aktualnim kodu (overeno po `be3e193`).

**Souvisejici:** `VYVAR_PIPELINE_CZ.md` (pipeline manual), `VYVAR_DECISIONS.md` (Path A + DAO+Gaia),
`VYVAR_FULL_AUDIT_LEDGER.md` (G5-F011), `CITATIONS.bib` (literatura pro ensemble, chyby, CT, AC).

---

## Obsah

1. [Linie sloupcu (diagram + tabulka)](#1-linie-sloupcu-diagram--tabulka)
2. [Korekce -- vzorec, vyznam, config gate, kod](#2-korekce--vzorec-vyznam-config-gate-kod)
3. [Tabulka spotrebitelu](#3-tabulka-spotrebitelu)
4. [Sloupec `err`](#4-sloupec-err)
5. [Krizove odkazy](#5-krizove-odkazy)

---

## 1) Linie sloupcu (diagram + tabulka)

Faze 2A per cil (`photometry_core.py`, smycka `_process_one_target_phase2a` ~7275+,
ulozeni `save_lightcurve_csv` ~3743+).

```
mag_inst
  |  read_flux_from_csv / apertura: flux -> mag, err (per snimka)
  v
mag_calib, delta_mag, ensemble_scatter
  |  ensemble_normalize()  [photometry_core.py:2254-2452]
  |  (+ volitelne GS11 dilution na mag_calib)
  v
[volitelne] mag_calib + dilution_delta_mag   [dilution.py:333-373]
  |
  +-- paralelne z mag_calib (pred reporting postprocess):
  |     mag_calib_ac = mag_calib + delta_m_corr   [photometry_core.py:7343-7349]
  |     mag_calib_ct = mag_calib + ct_correction  [apply_color_term, ~7394-7400]
  |
  v
apply_reporting_postprocess()
  |  mag_calib_raw = kopie mag_calib na vstupu  [photometry_core.py:3441]
  |  mag_calib (shipped) = mag_calib_raw kopie  [photometry_core.py:3461]
  |  outlier mask na mag_calib_ct             [photometry_core.py:3442-3459]
  v
[volitelne] Savitzky-Golay na mag_calib        [savgol_detrend_lc, ~7563-7576]
  |  po SG: mag_calib_ac = mag_calib + delta_m_corr (znovu)  [~7575-7576]
  v
[volitelne] democratic detrend -> delta_mag_democratic  [~7578-7590, CSV ~3863-3867]
  v
save_lightcurve_csv():
  mag_calib_final = mag_calib + CT + AC (scalar gates)  [compute_mag_calib_final ~3698-3735, ~3809-3854]
```

### Tabulka sloupcu v `lightcurve_*.csv`

| Sloupec | Vyznam | Kde se pocita (funkce, soubor:radka) |
|---------|--------|--------------------------------------|
| `mag_inst` | Instrumentalni mag z aperture flux | `_flux_to_mag` `photometry_core.py:621-625`; per-frame v `read_flux_from_csv` |
| `mag_calib` | Ensemble-kalibrovana diferencialni mag (ZP per snimka) | `ensemble_normalize` `photometry_core.py:2446-2452`; po GS11 `dilution.py:333-359`; po SG `savgol_detrend_lc` `photometry_core.py:3608+` volano ~7567 |
| `mag_calib_raw` | Audit snapshot `mag_calib` pred reporting postprocess | `apply_reporting_postprocess` `photometry_core.py:3441` |
| `mag_calib_ct` | `mag_calib` + color-term korekce (scalar `ct_correction`) | `apply_color_term` `photometry_core.py:2580-2620` volano ~7394; ulozeno `save_lightcurve_csv` ~3842 |
| `mag_calib_ac` | `mag_calib` (final po SG) + `delta_m_corr` kdyz `ac_ok` | Po SG ~7575-7576; CSV `save_lightcurve_csv` ~3800-3853 |
| **`mag_calib_final`** | **Kanonicka publikovana mag: final `mag_calib` + CT + AC** | `compute_mag_calib_final` `photometry_core.py:3698-3735`; CSV ~3809-3854 |
| `delta_mag` | `mag_inst(target) - mag_ensemble` (AIJ flux sum) | `ensemble_normalize` `photometry_core.py:2399` |
| `delta_mag_democratic` | Demokraticky detrendovana mag minus median (pokud ALG-4 ON) | `save_lightcurve_csv` ~3863-3867 |
| `ct_correction`, `ac_correction` | Scalar korekce per cil/noc (metadata) | CT: `apply_color_term` ~2612; AC: `compute_aperture_correction` ~2193 |
| `ac_scatter` | Median abs dev ?M ref stars (metadata, ne `err`) | `compute_aperture_correction` ~2194 |
| `err` | Photon/SNR + ensemble SEM | Viz sekce 4 |

**Poznamka:** `mag_calib_ct` v CSV je CT aplikovany na `mag_calib` **pred** Savitzky-Golay;
`mag_calib_final` aplikuje scalar `ct_correction` na **final** `mag_calib` (po SG), takze pri
`ct_ok` a zapnutem SG muze `mag_calib_ct` != `mag_calib + ct_correction`. Kanonicky export/figury
pouzivaji `mag_calib_final`.

---

## 2) Korekce -- vzorec, vyznam, config gate, kod

### 2.1 Ensemble zeropoint (`mag_calib`)

**Vzorec (per snimka):**

- `delta_mag = mag_inst(target) - ens_med`, kde `ens_med = -2.5 log10(sum 10^{-0.4 m_comp})`
  (`photometry_core.py:2376-2399`).
- `mag_calib = mag_inst(target) + weighted_median(cat_mag_j - mag_inst_j)` pres vybrane komparace
  (`photometry_core.py:2446-2452`).

**Vyznam:** diferencialni ensemble kalibrace k Gaia katalogu; airmass trend je odstranen
diferencialnim ensemble (ne per-target LSQ na reporting path).

**Literatura:** Broeg, Fernandez & Neuhùuser (2005) AN 326:134 (vaha 1/sigma^2 pri ZP);
Honeycutt (1992) PASP 104:435 (ensemble scatter pro `err`).

**Config:** `phase01_comparison_n_comp_min/max` (ensemble pocet kompar).

---

### 2.2 GS11 flux dilution (volitelne)

**Vzorec:** `mag_calib += dilution_delta_mag` kdyz `dilution_factor` v rozsahu
(`dilution.py:357-359`).

**Gate:** `gs11_dilution_enabled` (default **False**, `config.py:485`).

**Kod:** `apply_target_dilution_to_mag_calib` `dilution.py:333-373`, volano
`photometry_core.py:7329`.

---

### 2.3 Savitzky-Golay detrend (volitelne, ALG-2)

**Vyznam:** odstraneni pomaleho systematickùho trendu na `mag_calib` (airmass jiz v ensemble).

**Gate:** `savgol_detrend_enabled` (default **False**, `config.py:459`);
`savgol_window_frac` default 0.5 (`config.py:460`).

**Kod:** `savgol_detrend_lc` `photometry_core.py:3608+`, volano ~7567-7573.
Po SG se `mag_calib_ac` prepocita z noveho `mag_calib` (~7575-7576).

**Literatura:** Savitzky & Golay (1964).

---

### 2.4 Color term (CT) -> `mag_calib_ct` / scalar `ct_correction`

**Vzorec** (`apply_color_term`, `photometry_core.py:2590-2613`):

```
ct_correction = c1 * (target_bp_rp - bp_rp_comp_med)
mag_calib_ct  = mag_calib + ct_correction
```

**Gates (vrstvene):**

| Gate | Config / podminka | Default | Kod |
|------|-------------------|---------|-----|
| User toggle | `apply_color_term` | `"off"` (`config.py:447`) | `resolve_apply_color_term` `photometry_core.py:2866-2874` |
| Filter typ | NoFilter/Clear -> CT skip | auto: broadband only | `should_apply_color_term` `photometry_core.py:2696-2697` |
| Fit kvalita | `n_comp >= phase01_ct_min_comp`, `c1_stderr/c1 <= 0.5` | min_comp **7** | `should_apply_color_term` `photometry_core.py:2672-2673`, volano ~3050-3056 |
| Skupinovy fit | `state.group_color_term.apply_gate` | z fitu | `_compute_group_color_term_fit` ~3057-3064 |
| Extrapolace BP-RP | target BP-RP v rozsahu kompar | `phase01_ct_extrapolation_tol` | `_check_color_term_extrapolation` ~7387-7392 |
| Per-target `ct_ok` | finite target_bp_rp, c1!=0, finite bp_rp_comp_med | ù | `photometry_core.py:7401-7405` |

**Per-target/noc:** `ct_correction` je **scalar** (stejny pro vsechny snimky v LC).

---

### 2.5 Aperture correction Method B (AC) -> `mag_calib_ac` / `ac_correction`

**Vzorec** (`compute_aperture_correction`, `photometry_core.py:2075-2206`):

- Per ref comp: `?M = mag_large - mag_small` (large minus small aperture flux, ~2174-2178).
- `delta_m_corr = median(?M)` pres ref stars (~2193).
- `scatter_mag = median(|?M - delta_m_corr|)` (~2194).
- `mag_calib_ac = mag_calib + delta_m_corr` (~7346-7347, po SG ~7575-7576).

**Gates:**

| Gate | Config | Default | Kod |
|------|--------|---------|-----|
| Zapnuto | `aperture_correction_enabled` | **True** (`config.py:493`) | `photometry_core.py:7173-7180` |
| Min ref stars | `aperture_correction_min_ref_stars` | 3 | ~7178 |
| Contamination | `aperture_correction_max_contamination` | 0.15 | ~7179 |
| Scatter | `scatter_mag <= aperture_correction_max_scatter_mag` | 0.03 mag | ~2196-2197, config ~497 |

**Per-target/noc:** `delta_m_corr` je **scalar**. `ac_scatter` se uklada do CSV, **ne** do `err`.

**Poznam:** COG aperture correction (`cog_aperture_correction_enabled`, default False) je oddelena
cesta pri cteni fluxu ù neni soucast `mag_calib_ac` Method B.

---

### 2.6 Kanonicky soucet `mag_calib_final` (Path A)

**Vzorec** (`compute_mag_calib_final`, `photometry_core.py:3698-3735`):

```
mag_calib_final = mag_calib + (ct_correction if ct_ok else 0) + (delta_m_corr if ac_ok else 0)
```

Kdyz `ct_ok=False` a `ac_ok=True`: vysledek kopiuje `mag_calib_ac` (bit-identicke s AC-only
exportem pri CT-off config). Ulozeni: `save_lightcurve_csv` ~3809-3854.

**Dulezite:** CT a AC jsou **aditivni konstanty** na final `mag_calib` ù nemeni tvar variability
(nemenny scatter), nemenù `err` ani `lc_rms` (viz sekce 4).

**Rozhodnuti:** `VYVAR_DECISIONS.md` (Path A, 2026-06-22); audit G5-F011 v
`VYVAR_FULL_AUDIT_LEDGER.md`.

---

## 3) Tabulka spotrebitelu

| Spotrebitel | Sloupec | Soubor:radka |
|-------------|---------|--------------|
| AAVSO export MAG | `mag_calib_final` -> kopie do `mag_calib` pro body | `export_reports.py:645-686`, AAVSO loop ~904-911 |
| VarAstro body `mag_calib` | stejne (`_select_export_lc_rows`) | `export_reports.py:1054-1062` |
| VarAstro `delta_mag` | `delta_mag` (ensemble diferencial, **bez** CT/AC) | `export_reports.py:1057` |
| Main per-star PDF LC | `mag_calib_final` (fallback `mag_calib_ct`, `mag_calib`) | `_publication_lc_mag_column` `photometry_report.py:129-138`; plot ~1329-1348 |
| LC overlay PDF | stejne pres `_load_lc_xy_from_csv` | `photometry_report.py:1388`, overlay ~1443-1484 |
| Candidate PDF LC PNG | `mag_calib_final` | `_resolve_candidate_lc_mag_for_plot` `photometry_report.py:141-150` |
| `lc_rms`, `lc_rms_ooe` (summary) | **`mag_calib`** (ne final) | `photometry_core.py:7843-7844` |
| comp_qa locus os | `mag_calib` / `lc_median_mag` z summary | `comp_qa_core.py:24-25`, `comp_axis_mag` ~64-73 |
| Trust / export trust note | summary `lc_rms` + `n_clean` (comp_qa) | VarAstro header `export_reports.py:1000-1011`; `format_export_trust_note` ~848-852 |
| VarAstro header comp count | `n_good_comp` summary -> label **`n_ensemble_comp`** (stability good+suspect) | `export_reports.py:1000-1005`; **distinct** from trust `n_clean` (`comp_qa_core.py:465-470`) |
| Variability detection | instrumental `dao_flux` z `proc_*.csv` | `variability_detector.py:19`, `load_field_flux_matrix` default `flux_col="dao_flux"` ~277 |

**Publikacni vystupy** (AAVSO, VarAstro MAG, vsechny LC figury v PDF) nyni pouzivaji
**`mag_calib_final`**. Scatter/metriky kvality (`lc_rms`, trust) zustavaji na **`mag_calib`**
ù CT/AC jsou konstantni posuvy, scatter je invariantni.

**`n_ensemble_comp` vs `n_clean`:** summary `n_good_comp` (`photometry_core.py:7827-7829`) =
komparace stability `good` nebo `suspect` (ensemble pool). `n_clean` z comp_qa
(`comp_qa_core.py:465-470`) = komparace bez Sokolovsky flagu pro cil (trust). VarAstro header:
`n_ensemble_comp` (ne `n_good_comp`); trust pouziva `n_clean`.

**Legacy CSV** bez sloupce `mag_calib_final`: export fallback na AC precedence
(`mag_calib_ac` kdyz `ac_ok`) ù `export_reports.py:653-665`.

---

## 4) Sloupec `err`

### Photon / SNR base (per snimka)

**Vzorec** (`_photometric_error`, `photometry_core.py:628-649`):

```
variance = flux/g + sky/g*area + (read_noise/g)^2*area
err_photon = sqrt(variance) / flux
```

Literatura: Howell (1989) PASP 101:616, eq. 2.

Per-frame `err` v `target_frames` pochazi z aperture cteni (`read_flux_from_csv`).

### Ensemble SEM (term-3)

**Sestaveni** (`photometry_core.py:7502-7516`):

```
err = sqrt(err_photon^2 + ensemble_scatter^2)
```

`ensemble_scatter` = per-frame SEM zeropoint residual kompar
(`ensemble_normalize` `photometry_core.py:2389-2396`, Honeycutt 1992).

### Parovani s `mag_calib_final`

CT (`ct_correction`) a AC (`delta_m_corr`) jsou **konstantni posuvy** per cil/noc ?
**nemeni** `err`. Export paruje `mag_calib_final` (z `_select_export_lc_rows`) s puvodnim
`err` ù spravne pro konstantni kalibracni offset (viz G5-F002 RESOLVED non-issue).

`ac_scatter` je ulozen v CSV (`save_lightcurve_csv` ~3850), ale **neni** pridavano do `err`
(per-point inflace by misrepresentovala korelovanù systematik jako nahodny rozptyl).

---

## 5) Krizove odkazy

| Dokument | Obsah |
|----------|-------|
| `VYVAR_DECISIONS.md` | DAO+Gaia matched only (2026-06-22); Path A `mag_calib_final` |
| `VYVAR_FULL_AUDIT_LEDGER.md` | G5-F011 FIXED (`be3e193`); G5-F003 superseded |
| `VYVAR_PIPELINE_CZ.md` | Pipeline manual (CQ-C, PSF, blind solver) ù odkazuje sem |
| `VYVAR_CANONICAL_COMBINATION_LOGIC.md` | Ensemble flux-sum vs IVW (Broeg) |
| `CITATIONS.bib` | Broeg 2005, Howell 1989, Honeycutt 1992, Savitzky-Golay, democratic detrender |

**Commit implementace:** `be3e193` ù `fix(calib): canonical mag_calib_final (CT+AC) used by export and all figures (G5-F011)`.
