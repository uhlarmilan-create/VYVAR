# Pruvodce konfiguraci VYVAR (config.json) - CZ

_Sesterky dokument: `VYVAR_CONFIG_GUIDE_EN.md` (anglicky). Vychazi z registru
parametru (`dev/validation/params_registry.json`, 264 polozek) a z auditu
zdroju parametru (`dev/results/PARAM_SOURCE_AUDIT.md`), stav k 18. 7. 2026
(po redukci parametru WAVE-B). Pri zmene parametru aktualizujte tohoto
pruvodce spolu s registrem._

## Co je config.json?

`config.json` lezi v korenovem adresari VYVAR a je to hlavni soubor
nastaveni cele pipeline. Jsou v nem "knofliky", ktere ridi, JAK se vase
snimky zpracuji: jak se hledaji hvezdy, jak se vybiraji srovnavaci hvezdy,
jak se stavi svetelna krivka a co se objevi v reportech. Bezne ho editujete
pres stranku Nastaveni v aplikaci, ne rucne.

Ne kazda hodnota, kterou VYVAR pouziva, ale zije v tomto souboru. Parametry
jsou tri druhu a tento pruvodce kazdy z nich oznacuje:

- **Staticky (databaze)** - fakta o vasi observatori: stanoviste,
  dalekohled, kamera, katalogy. Ziji v databazovych tabulkach (LOCATION,
  TELESCOPE, EQUIPMENTS) a spravuji se v aplikaci, ktera je autoritativnim
  zdrojem. WAVE-B odstranila devet kopii techto fakt z config.json, takze
  se edituji jen na jednom miste: souradnice a nazev stanoviste
  (observer_lat / observer_lon / observer_alt_m / observer_location_name,
  hydratovane z vychoziho radku LOCATION, IS_DEFAULT), gain a
  read_noise detektoru (vyhodnocovane nejdrive z DB/FITS) a popisky meritka
  (plate_scale_arcsec_per_px, phase01_plate_scale_arcsec_per_px,
  export_arcsec_per_px, odvozene z WCS/optiky). CFG-REVIEW-2026-10 presunula
  do EQUIPMENTS dalsi dve vlastnosti kamery: ADC_BITS (drive
  gain_container_scale: nativni gain / 2^(16-ADC_BITS) je gain na
  16-bitovych snimcich, kdyz photon transfer neni k dispozici) a
  SIGMA_SYS_MAG (drive mapa sigma_sys_mag: chybove dno, kdyz noc nema dost
  kompu na jeho zmereni).
- **Dynamicky (FITS / za behu)** - hodnoty zmerene nebo spocitane pro kazdy
  beh: gain kamery, sumove cteni (read noise), rozmer snimku, uhlove
  meritko (plate scale), binning, filtr, expozice. VYVAR je cte z hlavicek
  FITS souboru nebo je odvodi (napr. z astrometrickeho reseni). Po WAVE-B
  uz nemaji zalohu v config.json.
- **Nastaveni (config.json)** - skutecne uzivatelske ladeni chovani
  pipeline. To je vetsina z 269 registrovanych parametru (config.json jich
  uklada 249; zbytek jsou fakta z databaze, hodnoty z FITS/za behu nebo
  interni zazemi).
  WAVE-B take zafixovala 20 internich parametru slepeho/astrometrickeho
  resice, ktere se v praxi nikdy neladily, a slucila 14 skalaru urovni a
  apertur do 3 strukturovanych klicu (comp_color_tiers, phase01_tiers,
  aperture_snr_sizing).
- **Interni** - technicke zazemi (cesty k souborum, hodnoty specificke pro
  pocitac). Nechte je byt, pokud presne nevite proc.

## Jak se hodnoty vyhodnocuji

Pro kazdy parametr plati jasna prednost. Obecne: vychozi hodnoty v kodu ->
config.json -> fakta z databaze -> hodnoty zmerene z FITS, pricemz u
dynamickych parametru vyhrava nejkonkretnejsi zdroj (gain zmereny v FITS
hlavicce prebiji zalohu v configu). Kazdy report obsahuje sekci
Configuration s UPLNYM snimkem konfigurace tak, jak ji beh skutecne pouzil,
vcetne vyhodnocenych dynamickych hodnot - vzdy tedy dohledate, jak vysledek
vznikl, i po letech.

Bezpecnost: config.json muze zapsat pouze vyslovna akce Ulozit v UI. Behy
pipeline ho nikdy nemeni.

## Editace bez UI

config.json muzete otevrit v libovolnem textovem editoru a menit hodnoty
rucne. Soubor se uklada rozdeleny do sekci (jedna na kazdou fazi pipeline) s
kratkym komentarem nad kazdou skupinou i kazdym klicem, takze vidite, co
ktere nastaveni dela, primo v editoru. Nekolik pravidel:

- Radkove komentare `//` jsou povolene a pri nacteni se ignoruji (blokove
  komentare `/* */` ani carky navic povolene NEJSOU - jinak jde o striktni
  JSON).
- Nezname klice se ignoruji s varovanim, ktere navrhne nejblizsi platny klic
  (pojistka proti preklepum); pipeline pro ne bezi s vychozimi hodnotami.
- Po editaci soubor zkontrolujte samostatnym validatorem:

    python dev/scripts/validate_config.py

  Nahlasi syntakticke chyby (s cisly radku), nezname klice (s navrhy),
  hodnoty mimo rozsah a nespravne typy a skonci nenulovym kodem, pokud je
  neco spatne.
- Ulozeni z UI soubor znovu vygeneruje (vcetne skupin a komentaru) z registru
  parametru, takze vlastni komentare pridane rucne se pri ulozeni z UI
  nezachovaji.

## Jak cist tabulky

Sloupec Typ: ctyri kategorie vyse. Sloupec Odkud se bere: skutecny zdroj
hodnoty (s poznamkou, kdyz ji muze prepsat FITS). Sloupec Kde se pouziva:
oblast kodu, ktera parametr cte, s jednim reprezentativnim odkazem do kodu.
Rozsah: tvrde limity vynucovane aplikaci, pokud jsou definovane.


## Pozorovatel a stanoviste

Kdo pozoroval a odkud. Jde o fakta observatore: identifikuji vas v exportech pro AAVSO a davaji souradnice stanoviste pro vypocet vzdusne hmoty (airmass) a casovych korekci. Spravuji se pres vyber lokality; zdrojem pravdy je databazova tabulka LOCATION.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `aavso_filter_map` | {} | Staticky (databaze) | config.json | sestaveni a validace konfigurace (`config.py:1196`) | Volitelna mapa vasich nazvu filtru na oficialni AAVSO kody pouzite v exportu (napr. 'NoFilter' -> 'CV'). |
| `observer_alt_m` | 275.0 | Staticky (databaze) | databaze (LOCATION) | hlavni UI aplikace (`app.py:2045`) | Nadmorska vyska stanoviste v metrech; soucast definice stanoviste pro vypocet airmass a casovych korekci. WAVE-B odstranila kopii v config.json - hydratuje se z DB podle observer_location_id. |
| `observer_code` | (prazdne) | Staticky (databaze) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:9690`) | Oficialni AAVSO kod pozorovatele tisteny v kazdem AAVSO exportu jako #OBSCODE (a v reportech). Varovani se emituje jen kdyz hodnota chybi/je prazdna. |
| `observer_lat` | 50.1121658 | Staticky (databaze) | databaze (LOCATION) | hlavni UI aplikace (`app.py:2043`) | Zemepisna sirka stanoviste ve stupnich. Vedecke behy berou stanoviste ze zaznamu LOCATION daneho draftu. WAVE-B odstranila kopii v config.json - hydratuje se z DB podle observer_location_id. |
| `observer_location_id` | - | Staticky (databaze) | databaze (LOCATION.IS_DEFAULT) | stranka importu / nocni runner | Neni v config.json: vychozi stanoviste je radek LOCATION s IS_DEFAULT (Database Explorer); beh muze zvolit jine stanoviste na strance importu nebo pres --location-id. |
| `observer_location_name` | (prazdne) | Staticky (databaze) | databaze (LOCATION) | hlavni UI aplikace (`app.py:2046`) | Citelny nazev vybraneho stanoviste (napr. Jirny, Dablice). WAVE-B odstranila kopii v config.json - hydratuje se z DB podle observer_location_id. |
| `observer_lon` | 14.6982547 | Staticky (databaze) | databaze (LOCATION) | hlavni UI aplikace (`app.py:2044`) | Zemepisna delka stanoviste ve stupnich; jak si vedecky beh stanoviste vyhodnocuje, viz observer_lat. WAVE-B odstranila kopii v config.json - hydratuje se z DB podle observer_location_id. |
| `observer_name` | Unknown Observer | Staticky (databaze) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:9691`) | Cele jmeno pozorovatele tistene v reportech. |

## Cesty k souborum a katalogum

Kde VYVAR na disku najde sva data: hvezdne katalogy (Gaia, VSX, exoplanety), indexy pro reseni souradnic, archiv a knihovnu kalibraci. Technicke zazemi specificke pro pocitac - jednou nastavit a zapomenout.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `archive_root` | (resolved at runtime) | Interni | config.json | hlavni UI aplikace (`app.py:76`) | Korenova slozka archivu pozorovani (surove i zpracovane drafty). Vyhodnocuje se pro dany pocitac pri startu. |
| `calibration_library_root` | (resolved at runtime) | Interni | config.json | hlavni UI aplikace (`app.py:233`) | Korenova slozka knihovny kalibraci (master darky/flaty trizene dle kamery, binningu a teploty). Vyhodnocuje se pro dany pocitac. |
| `database_path` | (resolved at runtime) | Interni | config.json | hlavni UI aplikace (`app.py:1918`) | Cesta k hlavni SQLite databazi VYVAR (drafty, vybaveni, lokality, vysledky). Vyhodnocuje se pro dany pocitac. |
| `gaia_db_path` | (prazdne) | Interni | config.json | vyber srovnavacich hvezd (`comp_selection_per_target.py:151`) | Cesta k lokalni celoblohove databazi Gaia DR3 (40M+ hvezd) pro identifikaci hvezd a vyber srovnavacich hvezd. |
| `objects_db_path` | catalogs/vyvar_objects.db | Interni | config.json | pristup ke katalogum (`catalog_db.py`) | Cesta k lokalnimu katalogu objektu: tabulka vsx_data (promenne hvezdy AAVSO VSX) a tabulka exoplanet_data (hostitele exoplanet z NASA Exoplanet Archive), sestavuje catalogs/objects_make.py. Slouzi k identifikaci znamych promennych a hostitelu exoplanet v poli. |
| `project_root` | VYVAR | Interni | pouze vychozi v kodu | hlavni UI aplikace (`app.py:2047`) | Koren instalace VYVAR; odvozuje se z umisteni kodu, nikdy needitovat. |

## Kalibrace

Odstraneni otisku kamery ze surovych snimku pomoci master darku a flatu, plus kontrolni brany, ktere odhali spatny nebo zastaraly master driv, nez potichu poskodi vedecky vysledek.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `calibration_master_ccd_temp_tolerance_c` | 0.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:743`) | Maximalni povoleny rozdil teploty CCD (deg C) mezi master darkem a svetelnymi snimky, ktere kalibruje. |
| `masterdark_validity_days` | 90 | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:1853`) | Kolik dni zustava master dark platny; starsi mastery vyvolaji vyzvu k nafoceni novych darku (aktualne termin ~21.7.). |
| `masterflat_validity_days` | 200 | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:1854`) | Kolik dni zustava master flat platny, nez si VYVAR rekne o novy. |

## Kontrola kvality snimku (QC)

Automaticke kontroly kazdeho snimku: ostrost hvezd (FWHM/HFR), protazeni, pozadi, minimalni pocty hvezd a volitelne brany, ktere spatne snimky vyradi z fotometrie. Patri sem i predzpracovani pozadi (sky-surface).

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `auto_fwhm_enabled` | True | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:408`) | Automaticky odvodi limit ostrosti (FWHM) pro kvalitu snimku ze statistiky seeingu dane noci misto pevneho cisla. |
| `auto_fwhm_k_factor` | 1.5 | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:418`) | Nasobitel medianu FWHM dane noci pri odvozovani automatickeho limitu kvality. |
| `preprocess_sky_surface_order` | 2; rozsah 0 .. 2 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1330`) | Rad polynomu modelu pozadi oblohy (sky-surface) odecitaneho pred fotometrii se zachovanim toku (2 = jemne odstraneni 2D gradientu; soucast anchoru draft_435). |
| `qc_after_calibrate_enabled` | True | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:14449`) | Spousti plne QC hned po kalibraci a uklada metriky kvality kazdeho snimku. |
| `detection_threshold_sigma` | 4.5; rozsah 2 .. 20 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py`) | Jediny detekcni prah hvezd, N x sigma oblohy, pro vsechny DAO detekce (QC, zarovnani, plate solve, MASTERSTAR, per-frame katalogy). Nahrazuje sips_dao_threshold_sigma, masterstar_dao_threshold_sigma, dao_detection_n_equiv, alignment_detection_sigma a qc_dao_detection_sigma (CFG-CLEAN 2026-10). |

## Zarovnani snimku

Sesazeni vsech snimku serie na spolecnou pixelovou mriz, aby kazda hvezda zustala po celou noc na stejnych souradnicich.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `alignment_max_control_points` | 80 | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:2163`) | Maximalni pocet kontrolnich bodu zarovnani na snimek; vice je robustnejsi, ale pomalejsi. |
| `alignment_max_stars` | 160; rozsah 10 .. 5000 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:122`) | Strop poctu hvezd uvazovanych parovacem pri zarovnani. |

## Detekce, reseni souradnic a masterstar

Hledani hvezd na snimcich, urceni oblohovych souradnic (plate solving vcetne slepeho reseni), stavba referencniho katalogu masterstar, parovani s katalogem Gaia a klasifikace hustoty pole a kandidatu promennosti.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `epsf_min_stars` | 30 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1335`) | Minimalni pocet vhodnych hvezd pro stavbu empirickeho PSF modelu (ePSF) pro PSF fotometrii. |
| `exoplanet_match_max_sep_arcsec` | 3.0; rozsah 0.5 .. 30 | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:5026`) | Maximalni uhlova vzdalenost (arcsec) pro ztotozneni detekovane hvezdy se znamym hostitelem exoplanety. |
| `frame_height_px` | 1397 | Interni | FITS hlavicka | fotometricke jadro (Phase 2A) (`photometry_core.py:14713`) | Vyska snimku v pixelech; za behu se meri z FITS NAXIS2. WAVE-B ji internalizovala (uz se neuklada do config.json). |
| `frame_width_px` | 2082 | Interni | FITS hlavicka | fotometricke jadro (Phase 2A) (`photometry_core.py:14712`) | Sirka snimku v pixelech; za behu se meri z FITS NAXIS1. WAVE-B ji internalizovala (uz se neuklada do config.json). |
| `masterstar_best_of_n` | 10; rozsah 1 .. 25 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1476`) | Kolik nejlepsich snimku se sklada/uvazuje pri stavbe reference masterstar. |
| `masterstar_dao_pass2_sigma` | 1.9 | Nastaveni (config.json) | pouze vychozi v kodu | kalibrace a zpracovani snimku (`pipeline.py:7411`) | Detekcni sigma druheho, hlubsiho pruchodu DAO na masterstar stacku. |
| `masterstar_use_best_frame_fwhm` | True | Nastaveni (config.json) | pouze vychozi v kodu | kalibrace a zpracovani snimku (`pipeline.py:11889`) | Pouziva FWHM nejlepsiho snimku pro detekcni jadra masterstar misto prumeru. |
| `phase01_chip_interior_margin_px` | 50 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14732`) | Okraj (px) od hrany cipu, ve kterem se hvezdy vylucuji z vyberu srovnavacich (okrajove efekty). |
| `phase01_plate_scale_arcsec_per_px` | 1.3; rozsah 0 .. 30 | Dynamicky (FITS / za behu) | vypocteno z WCS | fotometricke jadro (Phase 2A) (`photometry_core.py:10063`) | Meritko snimku pro Phase 1; za behu se vyhodnoti z WCS reseni. WAVE-B odstranila jeho zalohu v config.json - resolver je autoritativni. |
| `plate_scale_arcsec_per_px` | 1.3; rozsah 0.1 .. 30 | Dynamicky (FITS / za behu) | vypocteno z WCS | fotometricke jadro (Phase 2A) (`photometry_core.py:9934`) | Globalni meritko snimku (arcsec/px); za behu se vyhodnoti z WCS - cislo prevadejici pixely na uhly na obloze. WAVE-B odstranila jeho zalohu v config.json. |
| `plate_solve_fov_deg` | 1.0 | Dynamicky (FITS / za behu) | vypocteno (FITS + optika z DB) | hlavni UI aplikace (`app.py:2155`) | Odhad zorneho pole (stupne) predavany plate solveru; pocita se z rozmeru snimku a optiky. |
| `sips_dao_fwhm_px` | 2.5; rozsah 1 .. 8 | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:529`) | Predpokladane FWHM hvezd (px) pro detekcni preset DAO ve stylu SIPS. |
## Fotometrie

Mereni jasnosti hvezd: velikost apertury, oblozne mezikruzi (annulus), model chyb, aperturni korekce, volitelna PSF fotometrie a odecet sousedu a volitelne metody detrendovani. Srdce pipeline.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `annulus_inner_fwhm` | 2.7; rozsah 1 .. 10 | Nastaveni | config.json | PSF fotometrie (`psf_photometry.py:1940`) | Vnitrni polomer obloznich mezikruzi v nasobcich FWHM (APERTURE-01d; AIJ Sky_Inner 14 px na draftu 516). |
| `annulus_outer_fwhm` | 5.2; rozsah 1.5 .. 12 | Nastaveni | config.json | PSF fotometrie (`psf_photometry.py:1941`) | Vnejsi polomer obloznich mezikruzi v nasobcich FWHM (APERTURE-01d; AIJ Sky_Outer 27 px na draftu 516). |
| `aperture_fwhm_factor` | 1.35; rozsah 0.5 .. 6 | Nastaveni | config.json | per-frame katalogy (`photometry_shared.enhance_catalog_dataframe_aperture`) | Polomer apertury per-frame katalogu: r = f x QC FWHM snimku (APERTURE-01). Svetelne krivky pouzivaji k* na cil z pevne mrizky 1.0-2.0 x FWHM (APERTURE-K-2026-10). |
| `epsf_auto_run` | False | Nastaveni (config.json) | config.json | orchetrace night-run (`night_run.py`) | Kdyz true, RUN VYVAR / night_run automaticky spusti ePSF stupen (**beta**) po aperturni fotometrii. Vychozi VYPNUTO. Aperturni fotometrie je end-to-end validovana (AstroImageJ 4.86 mmag RMS, 134 epoch, BO CVn). ePSF je interni konzistentni s aperturou na referencni noci (era06); nezavisla externi a multi-rig validace jeste ceka. Tlacitka ePSF ve UI stupen vynuti. |
| `epsf_targets` | ALL | Nastaveni (config.json) | config.json | ePSF stupen (`epsf_stage.py`, `epsf_science_set.py`) | Ktere cile RUN ePSF meri: `ALL` = vsechny VSX cile s aperturni fotometrii (po `vsx_out_of_scope_types`; hostitele exoplanet a kandidati z hledani promennych nejsou katalogove promenne), nebo carkou oddelene VSX typy / jmena hvezd / catalog id se zastupnym `*` (`EA, EB, EW`, `E*`). Pouziva night run pri zapnutem `epsf_auto_run`; zalozka ePSF ho muze pro jeden beh prepsat (EPSF-TARGETS-2026-10). |
| `epsf_export_auto` | true | Nastaveni (config.json) | config.json | ePSF stupen (`epsf_stage.py`, `epsf_export.py`) | Soubory ePSF svetelnych krivek pro uzivatele (`photometry/lightcurves_reports/epsf/<hvezda>_<datum>_<setup>_epsf.txt`: bjd_tdb, delta_mag, err s popisnou hlavickou) se zapisi automaticky po dobehnuti RUN ePSF. false: zapise je tlacitko "Generate ePSF light-curve files" v zalozce ePSF. Neni to soubor pro AAVSO / VarAstro (EPSF-EXPORT-2026-10). |
| `err_empty_apertures_min` | 16 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1469`) | Minimalni pocet prazdnych apertur pro platny empiricky odhad pozadi. |
| `err_empty_apertures_n` | 64 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1462`) | Kolik prazdnych apertur se na snimek umisti pro empiricke mereni sumu pozadi. |
| `gain` | 1.0 | Dynamicky (FITS / za behu) | FITS hlavicka | kalibrace a zpracovani snimku (`pipeline.py:309`) | Gain detektoru (e-/ADU) prevadejici county na elektrony v modelu chyb; vyhodnocuje se z FITS hlavicky (s krizovou kontrolou proti DB). WAVE-B odstranila jeho zalohu v config.json - resolver je autoritativni. |
| `gs11_dilution_aperture_arcsec` | 0.0; rozsah 0 .. 120 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:165`) | Apertura (arcsec) pro vypocet dilution z katalogu; 0 ji odvodi z fotometricke apertury. |
| `gs11_dilution_enabled` | False | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:144`) | Zapina odhad dilution z katalogu (kolik svetla sousedu protece do apertur). |
| `gs11_dilution_mag_limit_delta` | 5.0; rozsah 0.5 .. 15 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:183`) | O kolik magnitud slabsi nez hvezda jeste dilution scitani pocita sousedy. |
| `gs11_target_min_dilution` | 0.5; rozsah 0.01 .. 1 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:6096`) | Minimalni prijatelna dilution cile, nez je cil oznacen jako silne blendovany. |
| `lc_outlier_enabled` | True | Nastaveni (config.json) | config.json + Settings UI | LC-OUTLIER-01 (`lc_outlier.py`) | Zapne izolovane spike + image evidence + frame_qc flagovani. Fotometrii nikdy nemaze. |
| `lc_outlier_n_sigma` | 5.0 | Nastaveni (config.json) | config.json + Settings UI | LC-OUTLIER-01 | Prah izolovaneho spike (|r| > n_sigma). Statisticka konvence. |
| `lc_outlier_adjacent_sigma` | 3.0 | Nastaveni (config.json) | config.json + Settings UI | LC-OUTLIER-01 | Sousedni same-sign residual chrani flare/zakryty. |
| `lc_outlier_frame_qc_n_sigma` | 5.0 | Nastaveni (config.json) | config.json + Settings UI | LC-OUTLIER-01 | Nocni MAD-sigma pro frame QC (FWHM/elong/sky). |
| `lc_outlier_evidence_n_sigma` | 5.0 | Nastaveni (config.json) | config.json + Settings UI | LC-OUTLIER-01 | Image-evidence MAD-sigma proti hvezdam podobneho fluxu ve snimku. |
| `lc_high_err_nsigma` | 5.0 | Nastaveni (config.json) | config.json + Settings UI | LC-FLAG-ERR-01 | Flag epochu kdyz `err_i > median(err) + n_sigma x 1.4826 x MAD(err)` na LC hvezdy. `flag_reason` jmenuje dominantni err komponentu. Statisticka konvence (vychozi 5). |
| `psf_chi2_locus_nsigma` | 5.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace | N-sigma reziduum od chi2(flux) lokusu pro psf_fit_ok. |
| `read_noise` | 10.0 | Dynamicky (FITS / za behu) | databaze (EQUIPMENTS) | kalibrace a zpracovani snimku (`pipeline.py:310`) | Sumove cteni detektoru (elektrony) v modelu chyb; vyhodnocuje se nejdrive z DB (vlastnost kamery), pak z FITS. WAVE-B odstranila jeho zalohu v config.json. |
| `save_lightcurve_png` | False | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:7386`) | Behem behu uklada i PNG nahledy svetelnych krivek jednotlivych cilu. |

## Vyber srovnavacich hvezd

Vyber souboru stalych hvezd, vuci nimz se cil meri (diferencialni fotometrie). Kriteria: shoda barvy (urovne dle Gaia BP-RP), rozdil jasnosti, vzdalenost, stabilita (RMS), izolovanost a pokryti snimky.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `comp_contamination_penalty_k` | 3.0; rozsah 0 .. 20 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2359`) | Sila penalizace vahy kontaminovanych srovnavacich hvezd pri vazeni souboru. |
| `comp_max_delta_bprp` | 0.79; rozsah 0 .. 5 | Nastaveni | config.json | vyber srovnavacich hvezd (`photometry_comp.py`) | Horni barevny limit zebricku srovnavacich hvezd (posledni krok po barevnych tridach 1-3). Sirsi jen pres T4 fallback, kdyz je v nem mene nez n_comp_min hvezd; zaroven hranice 4. radku comp_color_tiers. |
| `comp_max_slope_mmag_hr` | 5.0; rozsah 0 .. 500 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:116`) | Maximalni linearni trend (mmag/hod), ktery smi srovnavaci hvezda vykazovat, nez je vyrazena jako drifujici. |
| `comp_select_rms_floor` | 1e-06 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2211`) | Numericke dno RMS srovnavacich hvezd pri vazeni (brani numerickym explozim pri deleni). |
| `comp_slope_significance_k` | 3.0; rozsah 0 .. 10 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1092`) | Statisticka vyznamnost, od ktere se trend srovnavaci hvezdy pocita jako skutecny drift. |
| `comp_catalog_offset_max_mag` | 0.20; rozsah 0 .. 2 (0 = vypnuto) | Nastaveni (config.json) | config.json | Faze 2A vyber srovnavacich hvezd (`comp_smart.smart_select`) | Srovnavaci hvezda, jejiz rozdil katalog minus namereno se od ostatnich srovnavacich hvezd lisi o vic nez tuto hodnotu (mag), je vyrazena a nahrazena z rezervy (chybna katalogova hodnota, zamena hvezdy, nevideny soused). |
| `comp_color_tiers` | [{bprp: 0.15}, {bprp: 0.3}, {bprp: 0.55}, {bprp: 0.79}] | Nastaveni (config.json) | config.json | vyber srovnavacich hvezd (`photometry_comp.py`) | Barevne tridy \|BP-RP(srovnavaci) - BP-RP(cil)\| v poradi radku. Vyber vezme prvni tridu (radky 1-3, pak comp_max_delta_bprp), ve ktere je aspon n_comp_min hvezd, a uvnitr radi podle LOO RMS, barvy a vzdalenosti. Radek 4 je vse nad comp_max_delta_bprp (T4 fallback pro cervene cile); jeho 'bprp' kopiruje comp_max_delta_bprp. Vahy souboru jsou Broegovy vahy, ne vahy tridy. |
| `comp_pool_derived_admission` | True | Nastaveni (config.json) | config.json | globalni comp pool (`photometry_core.build_global_comp_pool`) | COMP-POOL-01: prijmani do fondu podle draftem odvozenych kriterii sumu/stability/diluce. |
| `comparison_stars_pool_n` | 0 | Nastaveni (config.json) | config.json | planovaci fond (`pipeline.select_comparison_stars_spatial_grid`) | Velikost planovaciho fondu; 0 = bez stropu (COMP-POOL-01). |
| `phase01_comparison_isolation_radius_px` | 25.0; rozsah 1 .. 200 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14762`) | Polomer izolace (px): srovnavaci hvezda v nem nesmi mit jasneho souseda. |
| `phase01_comparison_max_comp_rms` | 0.1; rozsah 0.01 .. 0.5 | Nastaveni | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14750`) | Absolutni horni mez LOO mag MAD srovnavaci hvezdy (INV-COMP-RMS-01). Brana je min(tato hodnota, k x foton). |
| `phase01_comparison_min_dist_arcsec` | 60.0; rozsah 0 .. 600 | Nastaveni | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14751`) | Minimalni vzdalenost (arcsec) mezi srovnavaci hvezdou a cilem proti vzajemne kontaminaci. |
| `phase01_comparison_min_frames_frac` | 0.2; rozsah 0.05 .. 0.95 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14752`) | Minimalni podil snimku, na kterych musi byt srovnavaci hvezda zmerena. |
| `phase01_comparison_n_comp_max` | 8 | Nastaveni (config.json) | config.json | kontrolni hvezda (`check_star_kmag.py:537`) | Maximalni velikost souboru; dle literatury se prinos scintilace nasycuje kolem 6-8 srovnavacich hvezd. |
| `snr_min` | 10.0; rozsah 3 .. 1000 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py`) | Limit slabych hvezd pro vsechny hvezdy: aperturni SNR pod touto hodnotou -> hvezda se nemeri; snimek, kde SNR cile klesne pod ni, se neexportuje. 10 = 0.11 mag (SNR-MIN-2026-10). |
| `phase01_comparison_n_comp_min` | 3 | Nastaveni | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14748`) | Minimalni pocet srovnavacich hvezd na cil (barevny zebricek se rozsiruje, dokud ho nedosahne). |
| `phase01_ct_extrapolation_tol` | 0.0 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:8927`) | Povolena extrapolace barevneho rozsahu vztahu barevneho clenu (0 = zadna extrapolace). |
| `phase01_ct_min_comp` | 7; rozsah 2 .. 30 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:249`) | Minimalni pocet srovnavacich hvezd pro fit vztahu barevneho clenu. |

## Duveryhodnost a kvalita vysledku

Jak VYVAR znamkuje vlastni vysledky: duvera GREEN/YELLOW/RED srovnavaciho souboru a kontrolni hvezdy, minimalni pocty epoch a snimku a prahy kvality svetelne krivky. Od 06/2026 je hodnota min-comps prahem pro GREEN, ne tvrdym minimem - mensi pocet srovnavacich hvezd elegantne degraduje na YELLOW s odpovidajicim zvetsenim chyb.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `check_select_rms_floor` | 0.0001; rozsah 0 .. 0.01 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2210`) | Dno RMS pri skorovani vyberu kontrolni hvezdy. |
| `check_star_min_epochs` | 5 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1100`) | Minimalni pocet epoch, ktere musi kontrolni hvezda pokryt, aby jeji verdikt kvality platil. |
| `comp_trust_min_comps` | 5 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1097`) | Prah GREEN duvery podle velikosti souboru (vychozi 5 dle literatury; produkce bezi na 3 se sigma skalovanym dle N - viz DECISIONS). Mene hvezd degraduje na YELLOW, ne RED. |
| `lc_quality_min_frames` | 20 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1094`) | Minimalni pocet snimku pro plnohodnotny verdikt kvality svetelne krivky. |
| `lc_quality_min_normal_frac` | 0.5; rozsah 0.1 .. 1 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1096`) | Minimalni podil normalnich (neoznacenych) bodu, ktery svetelna krivka potrebuje. |
| `lc_quality_short_min_frames` | 3 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1093`) | Minimalni pocet snimku pro kvalitni drahu kratkych serii (short-baseline). |
| `sparse_trust_T_green` | 1.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2207`) | Mez T-statistiky kontrolni hvezdy pro GREEN duveru na ridkych polich. |
| `sparse_trust_T_red` | 4.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2208`) | Mez T-statistiky, za kterou se duvera na ridkem poli meni na RED. |
| `sparse_trust_X2_RED` | 0.0004 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2209`) | Mez nadmerneho rozptylu, ktera meni duveru ridkeho pole na RED. |
| `trust_flag_enabled` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1183`) | Hlavni vypinac priznaku duvery po epochach zapisovaneho do vysledku a exportu. |

## Atmosfericka extinkce a barva

Korekce vlivu atmosfery: extinkce druheho radu (k2) zavisla na barve hvezdy a vzdusne hmote, a volitelne zachazeni s barevnym clenem. Dulezite hlavne pro nefiltovana nebo sirokopasmova pozorovani.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `apply_color_term` | auto | Nastaveni (config.json) | config.json | klasifikace fotometrickeho pasma (`band_classify.py:348`) | Zda se na magnitudy aplikuje transformace barevnym clenem ('off' ponechava instrumentalni system). |
| `k2_defaults_bprp` | {} | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1508`) | Literarni vychozi hodnoty k2 po pasmech vuci BP-RP, pouzite pri vypnutem nebo neuspesnem fitu. |
| `k2_mode` | literature | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1091`) | Zdroj k2: 'literature' (vychozi hodnoty) nebo nocni fit, je-li program fitovani zapnut. |

## Reporty a HR diagram

Co se objevi v souhrnnem PDF reportu: barevny HR diagram pole, online obohaceni zajimavych objektu (SIMBAD, Gaia) a jeho vizualni ladeni.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `hrd_dsc_confirm_prob` | 0.9; rozsah 0.5 .. 1 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:803`) | Prah pravdepodobnosti pro potvrzeni klasifikace kandidata na zaklade HRD. |
| `hrd_enrich_max_candidates` | 20; rozsah 1 .. 100 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:760`) | Strop poctu objektu odeslanych do online obohaceni na jeden report. |
| `hrd_enrich_tap_timeout_s` | 60.0; rozsah 5 .. 120 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:767`) | Casovy limit (s) TAP dotazu behem online obohaceni. |
| `hrd_max_per_category` | 3; rozsah 1 .. 20 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:788`) | Maximum zvyraznenych objektu na kategorii v legende HRD. |
| `hrd_min_per_net` | 4; rozsah 0 .. 20 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:795`) | Minimalni pocet objektu ponechanych na detekcni sit pri orezavani zvyrazneni HRD. |
| `hrd_nss_category_enabled` | False | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:799`) | Prida kategorii Gaia non-single-star do zvyrazneni HRD. |
| `hrd_online_enrich_enabled` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:753`) | Online obohaceni zajimavych objektu HRD ze sluzeb archivu Gaia. |
| `hrd_parallax_min_mas` | 0.15; rozsah 0 .. 10 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:774`) | Minimalni paralaxa (mas) pro zarazeni hvezdy do HRD v absolutnich magnitudach. |
| `hrd_parallax_snr_min` | 5.0; rozsah 1 .. 20 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:781`) | Minimalni SNR paralaxy pro duveryhodnou pozici v HRD. |
| `hrd_simbad_enrich_enabled` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:756`) | Dotazy do SIMBAD na typy objektu zvyraznenych v HRD. |

## Export

Vystupy pro okolni svet: odesilani do AAVSO/VarAstro a krizova analyza s TESS.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `tess_enabled` | False | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2100`) | Zapina blok krizove analyzy s TESS (porovnani vasi svetelne krivky s daty TESS). |

## System a vykon

Chovani na urovni stroje: pocet paralelnich pracovniku a rezerva RAM. Odvozuje se z vaseho hardwaru; prepisujte, jen kdyz musite.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `per_frame_mp_reserve_ram_gb` | 1.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1028`) | RAM (GB) drzena volna na pracovnika pri dimenzovani paralelniho zpracovani snimku. |
| `qc_preprocess_workers` | 1 | Interni | prostredi / stroj | kalibrace a zpracovani snimku (`pipeline.py:15384`) | Pocet paralelnich pracovniku predzpracovani; pocita se z CPU/RAM pri startu, lze prepsat promennou prostredi VYVAR_PARALLEL_WORKERS. |
