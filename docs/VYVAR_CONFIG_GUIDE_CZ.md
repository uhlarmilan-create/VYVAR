# Pruvodce konfiguraci VYVAR (config.json) - CZ

_Sesterky dokument: `VYVAR_CONFIG_GUIDE_EN.md` (anglicky). Vychazi z registru
parametru (`dev/validation/params_registry.json`, 269 polozek) a z auditu
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
  hydratovane z radku LOCATION vybraneho pres observer_location_id), gain a
  read_noise detektoru (vyhodnocovane nejdrive z DB/FITS) a popisky meritka
  (plate_scale_arcsec_per_px, phase01_plate_scale_arcsec_per_px,
  export_arcsec_per_px, odvozene z WCS/optiky).
- **Dynamicky (FITS / za behu)** - hodnoty zmerene nebo spocitane pro kazdy
  beh: gain kamery, sumove cteni (read noise), rozmer snimku, uhlove
  meritko (plate scale), binning, filtr, expozice. VYVAR je cte z hlavicek
  FITS souboru nebo je odvodi (napr. z astrometrickeho reseni). Po WAVE-B
  uz nemaji zalohu v config.json.
- **Nastaveni (config.json)** - skutecne uzivatelske ladeni chovani
  pipeline. To je vetsina z 269 registrovanych parametru (config.json jich
  uklada 249; zbytek jsou fakta z databaze, hodnoty z FITS/za behu nebo
  interni zazemi). Nektere jsou oznacene "auto-uprava za behu": nastavena
  hodnota je zaklad, ktery si pipeline muze prizpusobit poli (napr.
  povolovani/zprisnovani kriterii srovnavacich hvezd podle hustoty hvezd).
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
| `aavso_observer_code` | UMIA | Staticky (databaze) | config.json | sestaveni a validace konfigurace (`config.py:1193`) | Vas oficialni AAVSO kod pozorovatele (UMIA); vklada se do kazdeho AAVSO exportu, aby bylo pozorovani pripsano vam. |
| `observer_alt_m` | 275.0 | Staticky (databaze) | databaze (LOCATION) | hlavni UI aplikace (`app.py:2045`) | Nadmorska vyska stanoviste v metrech; soucast definice stanoviste pro vypocet airmass a casovych korekci. WAVE-B odstranila kopii v config.json - hydratuje se z DB podle observer_location_id. |
| `observer_code` | (prazdne) | Staticky (databaze) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:9690`) | Kratky identifikator pozorovatele tisteny v reportech a exportech (odlisny od AAVSO kodu). |
| `observer_lat` | 50.1121658 | Staticky (databaze) | databaze (LOCATION) | hlavni UI aplikace (`app.py:2043`) | Zemepisna sirka stanoviste ve stupnich. Vedecke behy berou stanoviste ze zaznamu LOCATION daneho draftu. WAVE-B odstranila kopii v config.json - hydratuje se z DB podle observer_location_id. |
| `observer_location_id` | 2 | Staticky (databaze) | config.json (autoritativni kopie v DB) | hlavni UI aplikace (`app.py:1965`) | Databazove ID aktualne vybraneho stanoviste (radek tabulky LOCATION). |
| `observer_location_name` | (prazdne) | Staticky (databaze) | databaze (LOCATION) | hlavni UI aplikace (`app.py:2046`) | Citelny nazev vybraneho stanoviste (napr. Jirny, Dablice). WAVE-B odstranila kopii v config.json - hydratuje se z DB podle observer_location_id. |
| `observer_lon` | 14.6982547 | Staticky (databaze) | databaze (LOCATION) | hlavni UI aplikace (`app.py:2044`) | Zemepisna delka stanoviste ve stupnich; jak si vedecky beh stanoviste vyhodnocuje, viz observer_lat. WAVE-B odstranila kopii v config.json - hydratuje se z DB podle observer_location_id. |
| `observer_name` | Unknown Observer | Staticky (databaze) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:9691`) | Cele jmeno pozorovatele tistene v reportech. |

## Cesty k souborum a katalogum

Kde VYVAR na disku najde sva data: hvezdne katalogy (Gaia, VSX, exoplanety), indexy pro reseni souradnic, archiv a knihovnu kalibraci. Technicke zazemi specificke pro pocitac - jednou nastavit a zapomenout.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `archive_root` | (resolved at runtime) | Interni | config.json | hlavni UI aplikace (`app.py:76`) | Korenova slozka archivu pozorovani (surove i zpracovane drafty). Vyhodnocuje se pro dany pocitac pri startu. |
| `blind_index_fine_path` | (prazdne) | Interni | config.json | UI Nastaveni (`ui_settings.py:1065`) | Cesta k jemnemu astrometrickemu indexu pro slepe reseni souradnic u uzkouhlych sestav. |
| `blind_index_path` | (prazdne) | Interni | pouze vychozi v kodu | UI Nastaveni (`ui_settings.py:1067`) | Starsi cesta k jedinemu indexu pro slepe reseni; nahrazena dvojici fine/wide s automatickym vyberem. |
| `blind_index_select_mode` | auto | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:868`) | Jak si slepy solver vybira index: 'auto' voli fine nebo wide podle zorneho pole sestavy; lze vynutit rucne. |
| `blind_index_wide_path` | (prazdne) | Interni | config.json | UI Nastaveni (`ui_settings.py:1066`) | Cesta k sirokouhlemu astrometrickemu indexu pro slepe reseni u sirokych sestav (napr. 200mm Zeiss). |
| `calibration_library_root` | (resolved at runtime) | Interni | config.json | hlavni UI aplikace (`app.py:233`) | Korenova slozka knihovny kalibraci (master darky/flaty trizene dle kamery, binningu a teploty). Vyhodnocuje se pro dany pocitac. |
| `database_path` | (resolved at runtime) | Interni | config.json | hlavni UI aplikace (`app.py:1918`) | Cesta k hlavni SQLite databazi VYVAR (drafty, vybaveni, lokality, vysledky). Vyhodnocuje se pro dany pocitac. |
| `exoplanet_local_db_path` | exoplanets/vyvar_exoplanet_local.db | Interni | config.json | kalibrace a zpracovani snimku (`pipeline.py:4986`) | Cesta k lokalni databazi NASA Exoplanet Archive pro krizove parovani detekovanych hvezd se znamymi hostiteli exoplanet. |
| `gaia_db_path` | (prazdne) | Interni | config.json | vyber srovnavacich hvezd (`comp_selection_per_target.py:151`) | Cesta k lokalni celoblohove databazi Gaia DR3 (40M+ hvezd) pro identifikaci hvezd a vyber srovnavacich hvezd. |
| `project_root` | VYVAR | Interni | pouze vychozi v kodu | hlavni UI aplikace (`app.py:2047`) | Koren instalace VYVAR; odvozuje se z umisteni kodu, nikdy needitovat. |
| `vsx_local_db_path` | (prazdne) | Interni | config.json | kalibrace a zpracovani snimku (`pipeline.py:8015`) | Cesta k lokalni databazi AAVSO VSX znamych promennych hvezd, pouzite k jejich identifikaci v poli. |

## Kalibrace

Odstraneni otisku kamery ze surovych snimku pomoci master darku a flatu, plus kontrolni brany, ktere odhali spatny nebo zastaraly master driv, nez potichu poskodi vedecky vysledek.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `bpm_dark_mad_sigma` | 5.0; rozsah 2 .. 12 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1645`) | Citlivost detekce vadnych pixelu na master darku: pixely odchylene o vice nez tolik robustnich sigma se oznaci jako vadne. |
| `cal_diag_autocorrect_enabled` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1930`) | Povoli kalibracni diagnostice automaticky opravit zjisteny nesoulad konvence master darku (napr. skladani SUM vs MEAN) misto preruseni. |
| `cal_diag_gate_enabled` | True | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:395`) | Hlavni vypinac kalibracni diagnosticke brany: kontroluje uroven oblohy po odectu darku a pri nesmyslu se bezpecne zastavi. |
| `cal_diag_hard_sigma` | 5.0; rozsah 3 .. 10 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2120`) | Tvrdy limit (v sigmach) kontroly oblohy po kalibraci; za nim se sada snimku odmitne. |
| `cal_diag_rel_tol` | 0.02; rozsah 0 .. 0.2 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2119`) | Relativni tolerance krizove kontroly urovni dark vs snimek pred odectem (vychozi 2 %). |
| `cal_diag_sat_warn_frac` | 0.9; rozsah 0.5 .. 1 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2121`) | Podil saturacniho limitu, od ktereho diagnostika varuje pred prilis jasnym masterem nebo snimkem. |
| `calibration_library_native_binning` | 1; rozsah 1 .. 16 | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:694`) | Binning, ve kterem byly postaveny mastery knihovny kalibraci; pri jinem binningu draftu se mastery prevzorkuji (s provenance priznakem). |
| `calibration_master_ccd_temp_tolerance_c` | 0.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:743`) | Maximalni povoleny rozdil teploty CCD (deg C) mezi master darkem a svetelnymi snimky, ktere kalibruje. |
| `dao_qc_in_calibrate` | True | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:14476`) | Spousti QC zalozene na detekci hvezd primo behem kalibrace, aby se spatne snimky oznacily co nejdrive. |
| `masterdark_validity_days` | 90 | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:1853`) | Kolik dni zustava master dark platny; starsi mastery vyvolaji vyzvu k nafoceni novych darku (aktualne termin ~21.7.). |
| `masterflat_validity_days` | 200 | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:1854`) | Kolik dni zustava master flat platny, nez si VYVAR rekne o novy. |

## Kontrola kvality snimku (QC)

Automaticke kontroly kazdeho snimku: ostrost hvezd (FWHM/HFR), protazeni, pozadi, minimalni pocty hvezd a volitelne brany, ktere spatne snimky vyradi z fotometrie. Patri sem i predzpracovani pozadi (sky-surface).

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `auto_fwhm_enabled` | True | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:408`) | Automaticky odvodi limit ostrosti (FWHM) pro kvalitu snimku ze statistiky seeingu dane noci misto pevneho cisla. |
| `auto_fwhm_k_factor` | 1.5 | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:418`) | Nasobitel medianu FWHM dane noci pri odvozovani automatickeho limitu kvality. |
| `auto_fwhm_k_max` | 4.0 | Nastaveni (config.json) | config.json | UI kvality snimku (`ui_quality_dashboard.py:586`) | Horni mez nasobitele automatickeho FWHM limitu. |
| `auto_fwhm_k_min` | 1.0 | Nastaveni (config.json) | config.json | UI kvality snimku (`ui_quality_dashboard.py:585`) | Dolni mez nasobitele automatickeho FWHM limitu. |
| `frame_align_residual_gate_enabled` | False | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1105`) | Volitelna brana vyrazujici snimky s neobvykle velkymi rezidui zarovnani (spatne sesazene snimky). |
| `frame_align_residual_max_frac` | 0.25; rozsah 0.05 .. 1 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:7321`) | Maximalni podil snimku, ktery smi brana rezidui zarovnani vyradit; chrani pred zahozenim cele noci. |
| `frame_align_residual_min_keep_frames` | 10; rozsah 3 .. 100000 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:7324`) | Minimalni pocet snimku, ktery musi branu rezidui zarovnani prezit. |
| `frame_quality_fwhm_factor` | 1.0; rozsah 0.8 .. 3 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:7143`) | Skalovani prahu FWHM pouzivaneho branou kvality snimku. |
| `frame_quality_gate_enabled` | False | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1101`) | Volitelna brana vylucujici snimky se zretelne horsi ostrosti, nez je typicky seeing noci. |
| `frame_quality_min_keep_frames` | 10; rozsah 3 .. 100000 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:7148`) | Minimalni pocet snimku, ktery musi prezit branu kvality snimku. |
| `frame_quality_ratio_k` | 5.0; rozsah 2 .. 20 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:7142`) | Kolikrat horsi nez typicky snimek musi snimek byt, aby ho brana kvality vyradila. |
| `preprocess_sky_surface_order` | 2; rozsah 0 .. 2 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1330`) | Rad polynomu modelu pozadi oblohy (sky-surface) odecitaneho pred fotometrii se zachovanim toku (2 = jemne odstraneni 2D gradientu; soucast anchoru draft_435). |
| `qc_after_calibrate_enabled` | True | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:14449`) | Spousti plne QC hned po kalibraci a uklada metriky kvality kazdeho snimku. |
| `qc_dao_detection_sigma` | 5.0 | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:14450`) | Citlivost detekce (sigma nad pozadim) hledace hvezd, kterym QC pocita hvezdy na snimku. |
| `qc_elong_limit` | 1.8 | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:16575`) | Maximalni prijatelne protazeni hvezd; vyssi hodnoty znamenaji carkovani (vedeni/vitr) a snimek se oznaci. |
| `qc_fwhm_limit` | 8.0 | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:16570`) | Absolutni horni limit FWHM (px), nez je snimek oznacen jako prilis neostry (pouzito pri vypnutem auto-FWHM). |
| `qc_max_background_rms` | None | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:14470`) | Volitelny strop sumu pozadi (RMS) na snimek; None kontrolu vypne. |
| `qc_max_hfr` | 5.0 | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:14468`) | Maximalni half-flux radius snimku - alternativni mira ostrosti znama ze snimaciho softwaru. |
| `qc_min_stars` | 10 | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:14469`) | Minimalni pocet detekovanych hvezd, aby byl snimek povazovan za pouzitelny. |

## Zarovnani snimku

Sesazeni vsech snimku serie na spolecnou pixelovou mriz, aby kazda hvezda zustala po celou noc na stejnych souradnicich.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `alignment_detection_sigma` | 5.0 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:122`) | Citlivost detekce hledace hvezd pro vyber kontrolnich hvezd zarovnani. |
| `alignment_max_control_points` | 80 | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:2163`) | Maximalni pocet kontrolnich bodu zarovnani na snimek; vice je robustnejsi, ale pomalejsi. |
| `alignment_max_stars` | 160; rozsah 10 .. 5000 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:122`) | Strop poctu hvezd uvazovanych parovacem pri zarovnani. |

## Detekce, reseni souradnic a masterstar

Hledani hvezd na snimcich, urceni oblohovych souradnic (plate solving vcetne slepeho reseni), stavba referencniho katalogu masterstar, parovani s katalogem Gaia a klasifikace hustoty pole a kandidatu promennosti.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `blind_img_select_mode` | per_cell | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:941`) | Strategie vyberu detekovanych hvezd pro slepy solver ('per_cell' je rovnomerne rozprostre po snimku). |
| `blind_img_star_budget` | 80 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:935`) | Maximalni pocet hvezd ze snimku predanych slepemu solveru; rozpocet drzi reseni rychle. |
| `blind_use_rig_prior` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:945`) | Pouziva zname meritko sestavy jako prior k brzkemu zamitnuti neverohodnych kandidatu slepeho reseni. |
| `blind_verify_early_accept` | 30 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:909`) | Pocet overenych shod hvezd, pri kterem se slepy solver predcasne zastavi a reseni prijme. |
| `blind_verify_early_floor` | 0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:916`) | Minimalni pocet shod, nez smi predcasne prijeti vubec nastat. |
| `blind_verify_early_fraction` | 0.2; rozsah 0 .. 0.95 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:923`) | Podil katalogovych hvezd, ktery se musi shodovat pro predcasne prijeti. |
| `blind_verify_enabled` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:873`) | Overuje kazdeho kandidata slepeho reseni proti katalogu Gaia, nez mu uveri - pojistka slepeho reseni. |
| `blind_verify_inmemory_catalog` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:900`) | Drzi overovaci katalog v pameti kvuli rychlosti misto opakovanych dotazu do databaze. |
| `blind_verify_match_tol_px` | 2.5; rozsah 0.5 .. 20 | Nastaveni (config.json) | config.json | slepe reseni souradnic (blind solve) (`vyvar_blind_series.py:215`) | Pixelova tolerance pri parovani hvezd snimku s katalogovymi pozicemi behem overeni. |
| `blind_verify_min_fraction` | 0.15; rozsah 0.05 .. 0.95 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:894`) | Minimalni podil shodujicich se hvezd, aby slepe reseni proslo overenim. |
| `blind_verify_min_matches` | 12 | Nastaveni (config.json) | config.json | slepe reseni souradnic (blind solve) (`vyvar_blind_series.py:217`) | Absolutni minimum sparovanych hvezd pro overene slepe reseni. |
| `blind_verify_top_n` | 15 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:875`) | Kolik nejlepsich slepych kandidatu projde plnym overenim. |
| `catalog_query_max_rows` | 15000; rozsah 1000 .. 500000 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1021`) | Strop poctu radku jednoho dotazu do katalogu Gaia; chrani pamet na velmi hustych polich. |
| `crowding_blend_tighten_threshold` | 0.04; rozsah 0 .. 1 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2076`) | Uroven podilu blendu, nad kterou by (experimentalni) klasifikator nahusteni zprisnil kriteria srovnavacich hvezd. |
| `crowding_classifier_enabled` | False | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2072`) | Vypinac experimentalniho klasifikatoru nahusteni; VYPNUTO do validace na hustem poli Newtonu. |
| `crowding_comp_availability_loosen_count` | 500.0; rozsah 0 .. 1000000 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2083`) | Uroven dostupnosti srovnavacich hvezd povazovana za dostatek, kdyz logika nahusteni rozhoduje o povolovani. |
| `crowding_tighten_min_fwhm_px` | 3.0; rozsah 0 .. 30 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2092`) | Minimalni FWHM (px), pod kterym se zprisneni kvuli nahusteni neaplikuje (podvzorkovane snimky). |
| `debug_platesolver` | False | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1666`) | Podrobne diagnosticke logovani plate solveru; jen pro ladeni problemu. |
| `epsf_min_stars` | 30 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1335`) | Minimalni pocet vhodnych hvezd pro stavbu empirickeho PSF modelu (ePSF) pro PSF fotometrii. |
| `exoplanet_match_max_sep_arcsec` | 3.0; rozsah 0.5 .. 30 | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:5026`) | Maximalni uhlova vzdalenost (arcsec) pro ztotozneni detekovane hvezdy se znamym hostitelem exoplanety. |
| `field_density_adaptive_enabled` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2069`) | Hlavni vypinac adaptace na hustotu: profily ridke/normalni/huste pole automaticky povoluji ci zprisnuji kriteria srovnavacich hvezd. |
| `field_density_dense_threshold` | 1000.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2060`) | Pocet sparovanych hvezd, nad kterym se pole povazuje za huste (prisnejsi kriteria). |
| `field_density_sparse_threshold` | 300.0; rozsah 1 .. 50000 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2053`) | Pocet sparovanych hvezd, pod kterym se pole povazuje za ridke (volnejsi kriteria, aby soubor vubec vznikl). |
| `frame_height_px` | 1397 | Interni | FITS hlavicka | fotometricke jadro (Phase 2A) (`photometry_core.py:14713`) | Vyska snimku v pixelech; za behu se meri z FITS NAXIS2. WAVE-B ji internalizovala (uz se neuklada do config.json). |
| `frame_width_px` | 2082 | Interni | FITS hlavicka | fotometricke jadro (Phase 2A) (`photometry_core.py:14712`) | Sirka snimku v pixelech; za behu se meri z FITS NAXIS1. WAVE-B ji internalizovala (uz se neuklada do config.json). |
| `masterstar_accept_mode` | odds | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1775`) | Strategie prijeti astrometrickeho reseni masterstar ('odds' = statisticky test pomeru sanci). |
| `masterstar_best_of_n` | 10; rozsah 1 .. 25 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1476`) | Kolik nejlepsich snimku se sklada/uvazuje pri stavbe reference masterstar. |
| `masterstar_catalog_recovery_min` | 0.65; rozsah 0.4 .. 0.95 | Nastaveni (config.json) | config.json | UI masterstar/DAO (`ui_dao_stars.py:353`) | Minimalni podil katalogovych hvezd, ktery musi masterstar znovu najit, aby bylo reseni duveryhodne. |
| `masterstar_centre_rms_max_px` | 1.2; rozsah 0.5 .. 5 | Nastaveni (config.json) | config.json | UI masterstar/DAO (`ui_dao_stars.py:355`) | Maximalni RMS (px) pozic hvezd u stredu snimku pro prijatelne astrometricke reseni. |
| `masterstar_dao_pass2_sigma` | 1.9 | Nastaveni (config.json) | pouze vychozi v kodu | kalibrace a zpracovani snimku (`pipeline.py:7411`) | Detekcni sigma druheho, hlubsiho pruchodu DAO na masterstar stacku. |
| `masterstar_dao_threshold_sigma` | 2.1; rozsah 0.1 .. 6 | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:13137`) | Hlavni detekcni prah DAO (sigma) na masterstar; nizsi najde slabsi hvezdy, ale i vice sumu. |
| `masterstar_detection_cap_adaptive` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1816`) | Prizpusobuje strop detekce hustote pole misto pevneho poctu. |
| `masterstar_detection_cap_k` | 0.08; rozsah 0.01 .. 1 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1837`) | Skalovaci konstanta adaptivniho stropu detekce. |
| `masterstar_detection_cap_max` | 800 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1827`) | Horni mez adaptivniho stropu detekce. |
| `masterstar_detection_cap_min` | 250 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1820`) | Dolni mez adaptivniho stropu detekce. |
| `masterstar_distortion_benign_ratio_max` | 3.2; rozsah 2 .. 5 | Nastaveni (config.json) | config.json | UI masterstar/DAO (`ui_dao_stars.py:356`) | Limit pomeru zkresleni okraj/stred, ktery se jeste povazuje za neskodny pro optiku. |
| `masterstar_min_matched_floor` | 40 | Nastaveni (config.json) | config.json | UI masterstar/DAO (`ui_dao_stars.py:354`) | Absolutni minimum sparovanych hvezd, ktereho musi reseni masterstar dosahnout. |
| `masterstar_platesolve_sip_max_order` | 4 | Nastaveni (config.json) | config.json | UI masterstar/DAO (`ui_dao_stars.py:351`) | Nejvyssi rad SIP polynomu zkresleni, ktery smi solver fitovat. |
| `masterstar_platesolve_sip_min_order` | 3 | Nastaveni (config.json) | config.json | UI masterstar/DAO (`ui_dao_stars.py:352`) | Nejnizsi rad SIP zkresleni, ktery solver zkousi. |
| `masterstar_prematch_peak_sigma_floor` | 1.8; rozsah 0.5 .. 6 | Nastaveni (config.json) | config.json | kalibrace a zpracovani snimku (`pipeline.py:14006`) | Minimalni vyznamnost piku hvezd pouzitych ve fazi predbezneho parovani. |
| `masterstar_quality_crowded_n_cat_min` | 800 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1808`) | Pocet katalogovych hvezd, nad kterym kontroly kvality masterstar prepnou do rezimu husteho pole. |
| `masterstar_sibling_min_matched` | 40 | Nastaveni (config.json) | config.json | UI masterstar/DAO (`ui_dao_stars.py:358`) | Minimalni pocet shod, ktery potrebuje zachranne (sibling) reseni. |
| `masterstar_sibling_min_quadrants` | 3 | Nastaveni (config.json) | config.json | UI masterstar/DAO (`ui_dao_stars.py:360`) | Pozadovane pokryti kvadrantu u sibling reseni. |
| `masterstar_sibling_recovery_enabled` | True | Nastaveni (config.json) | config.json | UI masterstar/DAO (`ui_dao_stars.py:357`) | Povoli zachrannou cestu pres sibling stack, kdyz hlavni reseni masterstar selze. |
| `masterstar_sibling_rms_max_px` | 2.0; rozsah 0.5 .. 10 | Nastaveni (config.json) | config.json | UI masterstar/DAO (`ui_dao_stars.py:359`) | Limit RMS pro prijeti zachranneho sibling reseni. |
| `masterstar_sibling_stack_n` | 10 | Nastaveni (config.json) | config.json | UI masterstar/DAO (`ui_dao_stars.py:361`) | Kolik snimku sklada zachranny sibling stack. |
| `masterstar_use_best_frame_fwhm` | True | Nastaveni (config.json) | pouze vychozi v kodu | kalibrace a zpracovani snimku (`pipeline.py:11889`) | Pouziva FWHM nejlepsiho snimku pro detekcni jadra masterstar misto prumeru. |
| `phase01_chip_interior_margin_px` | 50 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14732`) | Okraj (px) od hrany cipu, ve kterem se hvezdy vylucuji z vyberu srovnavacich (okrajove efekty). |
| `phase01_match_radius_arcsec` | 10.0; rozsah 3 .. 30 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:11998`) | Uhlovy polomer (arcsec) pro parovani detekovanych hvezd se zaznamy katalogu Gaia. |
| `phase01_plate_scale_arcsec_per_px` | 1.3; rozsah 0 .. 30 | Dynamicky (FITS / za behu) | vypocteno z WCS | fotometricke jadro (Phase 2A) (`photometry_core.py:10063`) | Meritko snimku pro Phase 1; za behu se vyhodnoti z WCS reseni. WAVE-B odstranila jeho zalohu v config.json - resolver je autoritativni. |
| `plate_scale_arcsec_per_px` | 1.3; rozsah 0.1 .. 30 | Dynamicky (FITS / za behu) | vypocteno z WCS | fotometricke jadro (Phase 2A) (`photometry_core.py:9934`) | Globalni meritko snimku (arcsec/px); za behu se vyhodnoti z WCS - cislo prevadejici pixely na uhly na obloze. WAVE-B odstranila jeho zalohu v config.json. |
| `plate_solve_fov_deg` | 1.0 | Dynamicky (FITS / za behu) | vypocteno (FITS + optika z DB) | hlavni UI aplikace (`app.py:2155`) | Odhad zorneho pole (stupne) predavany plate solveru; pocita se z rozmeru snimku a optiky. |
| `saturate_limit_fraction` | 0.85 | Nastaveni (config.json) | pouze vychozi v kodu | kalibrace a zpracovani snimku (`pipeline.py:6097`) | Podil saturacni urovne detektoru, nad kterym se hvezda povazuje za saturovanou a vylouci se z fotometrie. |
| `sips_dao_fwhm_px` | 2.5; rozsah 1 .. 8 | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:529`) | Predpokladane FWHM hvezd (px) pro detekcni preset DAO ve stylu SIPS. |
| `sips_dao_threshold_sigma` | 3.5 | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:530`) | Detekcni prah (sigma) presetu DAO ve stylu SIPS. |
| `variability_clip_ratio_min` | 0.8 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2420`) | Minimalni podil bodu prezivsich sigma clipping, aby hvezda zustala v analyze promennosti. |
| `variability_comp_floor_factor` | 1.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2415`) | Kolikrat nad sumovym dnem srovnavacich hvezd musi hvezda kolisat, aby se pocitala jako promenna. |
| `variability_mag_limit` | 14.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2417`) | Limit jasnosti (slaby konec) pro hledani promennosti. |
| `variability_min_amplitude_mag` | 0.01 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2419`) | Minimalni amplituda (mag) kandidata promennosti. |
| `variability_min_frames` | 30 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2010`) | Minimalni pocet snimku, na kterych se hvezda musi objevit, aby byla analyzovana na promennost. |
| `variability_min_frames_frac` | 0.5; rozsah 0.05 .. 0.99 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2016`) | Minimalni podil snimku noci, ktery musi hvezda pokryvat pro analyzu promennosti. |
| `variability_min_points_rms` | 20 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2422`) | Minimalni pocet bodu, nez se pocita RMS statistika promennosti. |
| `variability_min_rms_pct` | 1.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2418`) | Percentilove dno vztahu RMS vs jasnost pouzite k normalizaci skore promennosti. |
| `variability_p85_filter` | 85 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2412`) | Percentilovy filtr odstranujici nejsumovejsi konec pred statistikou promennosti. |
| `variability_sigma_clip` | 5.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2411`) | Uroven sigma clippingu aplikovana na svetelne krivky pred metrikami promennosti. |
| `variability_sigma_threshold` | 2.3 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:5519`) | Vyznamnost (sigma), kterou musi rozptyl hvezdy prekrocit, aby byla oznacena jako kandidat promennosti. |
| `variability_slope_floor` | 0.02 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2413`) | Minimalni sklon trendu nadmerneho rozptylu povazovany za vyznamny pri skorovani kandidatu. |
| `variability_smoothness_max` | 0.8 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2416`) | Maximalni skore hladkosti - velmi hladke krivky jsou spis trendy/artefakty nez hvezdna promennost. |
| `variability_vdi_z_threshold` | 3.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2421`) | Prah z-skore indexu detekce promennosti (VDI). |
| `verify_mag_limit` | 14.0; rozsah 8 .. 18 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:904`) | Limit jasnosti katalogovych hvezd pouzitych pro overeni slepeho reseni. |
| `vsx_variable_targets_mag_limit` | 14.5 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:5523`) | Limit jasnosti pro automaticke prevzeti znamych promennych z VSX v poli jako cilu mereni. |

## Fotometrie

Mereni jasnosti hvezd: velikost apertury, oblozne mezikruzi (annulus), model chyb, aperturni korekce, volitelna PSF fotometrie a odecet sousedu a volitelne metody detrendovani. Srdce pipeline.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `annulus_inner_fwhm` | 4.75; rozsah 1 .. 10 | Nastaveni + auto-uprava za behu | config.json | PSF fotometrie (`psf_photometry.py:1940`) | Vnitrni polomer obloznich mezikruzi v nasobcich FWHM; adaptace na hustotu ho na hustych polich muze zprisnit. |
| `annulus_outer_fwhm` | 9.0; rozsah 1.5 .. 12 | Nastaveni + auto-uprava za behu | config.json | PSF fotometrie (`psf_photometry.py:1941`) | Vnejsi polomer obloznich mezikruzi v nasobcich FWHM. |
| `aperture_comp_factor` | 1.1; rozsah 0.25 .. 3 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:7092`) | Nasobitel velikosti apertury pro srovnavaci hvezdy. |
| `aperture_correction_enabled` | True | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:8650`) | Zapina aperturni korekci: prevod toku zmereneho v male aperture na skalu celkoveho toku pomoci jasnych referencnich hvezd. |
| `aperture_correction_max_contamination` | 0.15; rozsah 0 .. 2 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:8656`) | Maximalni kontaminace sousedy, kterou smi mit referencni hvezda pro aperturni korekci. |
| `aperture_correction_max_scatter_mag` | 0.03; rozsah 0 .. 2 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:8657`) | Maximalni rozptyl (mag) povoleny mezi referencnimi hvezdami aperturni korekce. |
| `aperture_correction_min_ref_stars` | 3; rozsah 1 .. 50 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:8655`) | Minimalni pocet referencnich hvezd pro vypocet aperturni korekce. |
| `aperture_fwhm_factor` | 1.9; rozsah 0.5 .. 6 | Nastaveni + auto-uprava za behu | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:7393`) | Zakladni polomer apertury v nasobcich FWHM; SNR-optimalni sweep velikosti muze efektivni polomer prizpusobit po hvezdach. |
| `aperture_snr_sizing` | {small: 1.5, large: 4.0} | Nastaveni + auto-uprava za behu | config.json | pipeline (`pipeline.py`) | Meze SNR-optimalniho sweepu velikosti apertury v nasobcich FWHM: 'small' je minimalni polomer (nejlepsi pro slabe hvezdy), 'large' maximalni. WAVE-B slucila puvodni skalary aperture_fwhm_factor_small/_large do tohoto mapovani (stredni trida neexistuje). |
| `aperture_photometry_enabled` | True | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:7899`) | Hlavni vypinac aperturni fotometrie - produkcni merici metody VYVAR. |
| `aperture_variable_factor` | 1.0; rozsah 0.25 .. 3 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:7091`) | Nasobitel velikosti apertury pro promennou (cilovou) hvezdu. |
| `cog_ac_factor_max` | 5.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1439`) | Horni mez faktoru aperturni korekce metodou curve-of-growth. |
| `cog_aperture_correction_enabled` | False | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1405`) | Zapina variantu aperturni korekce curve-of-growth (experimentalni alternativa). |
| `cog_isolation_fwhm` | 6.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1419`) | Polomer izolace (FWHM), ktery hvezda potrebuje, aby slouzila jako reference curve-of-growth. |
| `cog_ladder_step_px` | 0.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1434`) | Krok polomeru (px) aperturniho zebriku curve-of-growth. |
| `cog_min_stars` | 8; rozsah 1 .. 500 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1415`) | Minimalni pocet hvezd pro fit krivky rustu (curve of growth). |
| `cog_ref_fwhm` | 4.5; rozsah 1.5 .. 10 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1409`) | Referencni FWHM pro normalizaci krivky rustu. |
| `cog_sat_frac` | 0.85 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1429`) | Odriznuti referencnich hvezd curve-of-growth podle podilu saturace. |
| `cog_snr_min` | 50.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1424`) | Minimalni SNR referencni hvezdy curve-of-growth. |
| `democratic_detrend_enabled` | False | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:323`) | Volitelny 'demokraticky' detrend svetelne krivky (medianovy trend pole); vychozi VYPNUTO - detrend muze pozrit skutecnou promennost. |
| `democratic_sg_window_frac` | 0.5; rozsah 0.05 .. 0.95 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:329`) | Sirka okna (podil noci) vyhlazovace demokratickeho detrendu. |
| `err_background_mode` | empirical | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1456`) | Jak se odhaduje clen sumu pozadi v modelu chyb ('empirical' = mereni z prazdnych apertur na snimku). |
| `err_empty_apertures_min` | 16 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1469`) | Minimalni pocet prazdnych apertur pro platny empiricky odhad pozadi. |
| `err_empty_apertures_n` | 64 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1462`) | Kolik prazdnych apertur se na snimek umisti pro empiricke mereni sumu pozadi. |
| `gain` | 1.0 | Dynamicky (FITS / za behu) | FITS hlavicka | kalibrace a zpracovani snimku (`pipeline.py:309`) | Gain detektoru (e-/ADU) prevadejici county na elektrony v modelu chyb; vyhodnocuje se z FITS hlavicky (s krizovou kontrolou proti DB). WAVE-B odstranila jeho zalohu v config.json - resolver je autoritativni. |
| `gs11_comp_max_dilution` | 0.9; rozsah 0.01 .. 1 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1944`) | Maximalni dilution (kontaminace toku), kterou smi mit srovnavaci hvezda v modelu GS11. |
| `gs11_comp_suspect_dilution` | 0.98; rozsah 0.01 .. 1 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1944`) | Uroven dilution, pri ktere je srovnavaci hvezda oznacena za podezrelou. |
| `gs11_dilution_aperture_arcsec` | 0.0; rozsah 0 .. 120 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:165`) | Apertura (arcsec) pro vypocet dilution z katalogu; 0 ji odvodi z fotometricke apertury. |
| `gs11_dilution_enabled` | False | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:144`) | Zapina odhad dilution z katalogu (kolik svetla sousedu protece do apertur). |
| `gs11_dilution_mag_limit_delta` | 5.0; rozsah 0.5 .. 15 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:183`) | O kolik magnitud slabsi nez hvezda jeste dilution scitani pocita sousedy. |
| `gs11_target_min_dilution` | 0.5; rozsah 0.01 .. 1 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:6096`) | Minimalni prijatelna dilution cile, nez je cil oznacen jako silne blendovany. |
| `neighbor_sub_centroid_max_fwhm` | 1.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2233`) | Maximalni posun centroidu (FWHM) povoleny po odectu souseda. |
| `neighbor_sub_chi2_max` | 120.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2230`) | Strop chi-kvadrat fitu modelu souseda. |
| `neighbor_sub_max_neighbor_overmag` | 0.3 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2235`) | Pojistka: fitovany soused nesmi vyjit jasnejsi, nez se ceka, o vic nez tolik (mag). |
| `neighbor_sub_max_target_undermag` | 0.2 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2236`) | Pojistka: cil nesmi odectem ztratit vic nez tolik (mag). |
| `neighbor_sub_min_recovered_snr` | 5.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2237`) | Minimalni SNR, ktere si cil musi po odectu souseda udrzet. |
| `neighbor_sub_nn_contam_dmag` | 2.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2234`) | Rozdil jasnosti, do ktereho se nejblizsi soused pocita jako kontaminujici. |
| `neighbor_sub_refuse_sep_fwhm` | 0.8 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2232`) | Pod touto vzdalenosti (FWHM) se odecet odmitne - dvojice je prilis slita. |
| `neighbor_sub_regime_dmag_min` | 2.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2238`) | Mez rozdilu jasnosti vymezujici rezim, kde se odecet souseda pouziva. |
| `neighbor_sub_regime_sep_max` | 1.1 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2239`) | Mez vzdalenosti (FWHM) rezimu odectu souseda. |
| `neighbor_sub_residual_rms_max` | 150.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2231`) | Strop RMS rezidua po odectu, aby byl vysledek prijat. |
| `nonlinearity_fwhm_ratio` | 1.25; rozsah 1.01 .. 3 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:713`) | Prah pomeru FWHM v diagnostice nelinearity detektoru (jasne hvezdy tloustnou vuci slabym). |
| `nonlinearity_peak_percentile` | 20.0; rozsah 0 .. 50 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:706`) | Percentil jasnosti piku, na kterem diagnostika nelinearity vzorkuje tvary hvezd. |
| `phase2a_airmass_before_outlier` | False | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:4411`) | Prepinac poradi: aplikovat airmass detrend pred (True) nebo po (False) vyrazeni odlehlych bodu v Phase 2A. |
| `photometry_mode` | both | Nastaveni (config.json) | config.json | UI fotometrie (`ui_photometry.py:51`) | Ktere rodiny mereni bezi: aperturni, PSF, nebo obe. |
| `psf_adaptive_enabled` | False | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1310`) | Adaptivni smerovani po hvezdach mezi PSF a aperturnim merenim (program PSF je VYPNUT do brany husteho pole Newtonu). |
| `psf_adaptive_resolve_fwhm` | 2.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1312`) | Vzdalenost (FWHM), pod kterou adaptivni smerovac preferuje PSF pro slite dvojice. |
| `psf_adaptive_snr_lo` | 15.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1318`) | SNR, pod kterym adaptivni smerovac preferuje mereni PSF. |
| `psf_chi2_threshold` | 50.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1264`) | Limit chi-kvadrat prijatelneho PSF fitu. |
| `psf_group_sep_fwhm` | 1.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1270`) | Vzdalenost (FWHM), do ktere se hvezdy fituji spolecne jako PSF skupina. |
| `psf_grouper_enabled` | False | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1268`) | Zapina soucasne skupinove fitovani blizkych hvezd v PSF fotometrii. |
| `psf_neighbor_include_fwhm` | 3.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1275`) | Polomer (FWHM), ve kterem se sousede zahrnuji do PSF fitu. |
| `psf_neighbor_sub_enabled` | False | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1279`) | Zapina odecet sousedu PSF modelem pred aperturnim merenim blendovanych cilu. |
| `psf_photometry_enabled` | False | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:8686`) | Hlavni vypinac PSF fotometrie (nyni VYPNUTO; zapnuti ceka na validaci na hustem poli Newtonu). |
| `psf_quality_fallback_enabled` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1307`) | Pri selhani kvality PSF fitu se vrati k aperturnimu mereni. |
| `psf_spatial_enabled` | False | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1299`) | Zapina prostorove promenne PSF modely pres snimek. |
| `psf_spatial_grid` | 3x3 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1300`) | Mrizka prostorovych PSF bunek (napr. 3x3). |
| `psf_spatial_min_stars_per_cell` | 25 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1303`) | Minimalni pocet hvezd na bunku pro stavbu lokalniho PSF modelu. |
| `psf_spatial_order` | 0; rozsah 0 .. 2 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1259`) | Rad polynomu prostorove zmeny PSF (0 = konstantni PSF). |
| `pytics_enabled` | True | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:125`) | Zapina iteracni kalibraci srovnavacich hvezd ve stylu PyTICS pri stavbe svetelne krivky. |
| `pytics_n_iter` | 5; rozsah 1 .. 20 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:124`) | Pocet iteraci kalibrace ve stylu PyTICS. |
| `read_noise` | 10.0 | Dynamicky (FITS / za behu) | databaze (EQUIPMENTS) | kalibrace a zpracovani snimku (`pipeline.py:310`) | Sumove cteni detektoru (elektrony) v modelu chyb; vyhodnocuje se nejdrive z DB (vlastnost kamery), pak z FITS. WAVE-B odstranila jeho zalohu v config.json. |
| `save_lightcurve_png` | False | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:7386`) | Behem behu uklada i PNG nahledy svetelnych krivek jednotlivych cilu. |
| `savgol_detrend_enabled` | False | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:309`) | Volitelny detrend vyhlazenim Savitzky-Golay; vychozi VYPNUTO ze stejneho duvodu jako ostatni detrendy. |
| `savgol_polyorder` | 2 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:315`) | Rad polynomu filtru Savitzky-Golay. |
| `savgol_window_frac` | 0.5 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:314`) | Sirka okna (podil serie) filtru Savitzky-Golay. |
| `sigma_sys_mag` | {} | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1519`) | Systematicke dno chyb (mag) po pasmech pridavane kvadraticky ke statistickym chybam (napr. {'4': 0.018} pro pasmo 4). |
| `sysrem_enabled` | False | Nastaveni (config.json) | config.json | rizeni nocniho behu (`night_run.py:527`) | Volitelne odstraneni systematik SysRem (Tamuz+ 2005); vychozi VYPNUTO - overeno jako rizikove pro zachovani skutecne promennosti. |
| `sysrem_n_iter` | 3 | Nastaveni (config.json) | config.json | rizeni nocniho behu (`night_run.py:529`) | Pocet iteraci SysRem, je-li zapnut. |
| `temporal_bin_window` | 0; rozsah 0 .. 51 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:106`) | Sirka casoveho binu pro casove binovani (0 = zadne). |
| `temporal_binning_enabled` | False | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:107`) | Casove binovani svetelnych krivek; VYPNUTO zamerne - injekcni testy prokazaly skodlivost (24/25 cilu horsich). |

## Vyber srovnavacich hvezd

Vyber souboru stalych hvezd, vuci nimz se cil meri (diferencialni fotometrie). Kriteria: shoda barvy (urovne dle Gaia BP-RP), rozdil jasnosti, vzdalenost, stabilita (RMS), izolovanost a pokryti snimky. Nektere limity se automaticky prizpusobuji hustote pole.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `comp_clip_sigma` | 5.0; rozsah 3 .. 10 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1132`) | Uroven sigma clippingu aplikovana na serie srovnavacich hvezd pred statistikou souboru. |
| `comp_contamination_penalty_k` | 3.0; rozsah 0 .. 20 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2359`) | Sila penalizace vahy kontaminovanych srovnavacich hvezd pri vazeni souboru. |
| `comp_iterative_clip_enabled` | False | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1129`) | Iteracni pre-clipovani souboru srovnavacich hvezd (smycka vyrad-a-prevazi); v produkci ZAPNUTO od brnenskeho fixu 14. 6. 2026. |
| `comp_max_delta_bprp` | 0.79; rozsah 0 .. 5 | Nastaveni + auto-uprava za behu | config.json | vyber srovnavacich hvezd (`comp_selection_per_target.py:239`) | Maximalni rozdil barvy Gaia BP-RP mezi srovnavaci hvezdou a cilem - hlavni obrana proti extinkcnim systematikam u nefiltrovanych dat; adaptace na hustotu ji upravuje. |
| `comp_max_slope_mmag_hr` | 5.0; rozsah 0 .. 500 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:116`) | Maximalni linearni trend (mmag/hod), ktery smi srovnavaci hvezda vykazovat, nez je vyrazena jako drifujici. |
| `comp_select_rms_floor` | 1e-06 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2211`) | Numericke dno RMS srovnavacich hvezd pri vazeni (brani numerickym explozim pri deleni). |
| `comp_slope_significance_k` | 3.0; rozsah 0 .. 10 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1092`) | Statisticka vyznamnost, od ktere se trend srovnavaci hvezdy pocita jako skutecny drift. |
| `comp_sparse_fallback_enabled` | True | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1128`) | Povoli zachrannou cestu pro ridka pole, kdyz prisna kriteria daji prilis malo srovnavacich hvezd. |
| `comp_sparse_fallback_min` | 0 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1131`) | Minimalni pocet srovnavacich hvezd, o ktery zachranna cesta usiluje (0 = vezmi co je). |
| `comp_color_tiers` | [{bprp: 0.15, w: 1.0}, {bprp: 0.3, w: 0.85}, {bprp: 0.55, w: 0.5}, {bprp: 1.1, w: 0.25}] | Nastaveni (config.json) | config.json | vyber srovnavacich hvezd (`comp_selection_per_target.py:258`) | Urovne shody barvy pro srovnavaci hvezdy jako seznam kroku: kazdy ma limit rozdilu barvy BP-RP (`bprp`) a vahu (`w`) v souboru. Prvni krok je nejtesnejsi shoda s plnou vahou, dalsi jsou volnejsi s nizsi vahou. WAVE-B slucila puvodnich 8 skalaru comp_tier{1..4}_bprp_limit/_weight do teto jedne struktury. |
| `global_comp_pool_enabled` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2098`) | Stavi jeden spolecny fond srovnavacich hvezd pro cele pole misto plne oddelenych fondu po cilech. |
| `phase01_comparison_exclude_gaia_extobj` | True | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:12656`) | Vylucuje objekty oznacene v Gaia jako rozlehle (galaxie) z kandidatu na srovnavaci hvezdy. |
| `phase01_comparison_exclude_gaia_nss` | True | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:12654`) | Vylucuje zdroje Gaia non-single-star (dvojhvezdy) z kandidatu na srovnavaci hvezdy. |
| `phase01_comparison_fov_fraction` | 0.75 | Nastaveni (config.json) | pouze vychozi v kodu | fotometricke jadro (Phase 2A) (`photometry_core.py:14739`) | Podil zorneho pole kolem cile, ve kterem se hledaji srovnavaci hvezdy. |
| `phase01_comparison_isolation_radius_px` | 25.0; rozsah 1 .. 200 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14762`) | Polomer izolace (px): srovnavaci hvezda v nem nesmi mit jasneho souseda. |
| `phase01_comparison_mag_bright_threshold` | 12.75; rozsah 6 .. 18 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14756`) | Jasnost, nad kterou plati prisnejsi limit rozdilu jasnosti pro jasne hvezdy. |
| `phase01_comparison_max_comp_rms` | 0.1; rozsah 0.01 .. 0.5 | Nastaveni + auto-uprava za behu | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14750`) | Maximalni nocni RMS serie srovnavaci hvezdy; adaptace na hustotu ji muze zprisnit. |
| `phase01_comparison_max_dist_deg` | 1.5; rozsah 0.05 .. 10 | Nastaveni + auto-uprava za behu | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14740`) | Maximalni uhlova vzdalenost (stupne) srovnavaci hvezdy od cile; adaptace na hustotu pridava k zakladu z FOV. |
| `phase01_comparison_max_fwhm_factor` | 1.5; rozsah 0.5 .. 5 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14761`) | Maximalni pomer FWHM vuci medianu pole u srovnavaci hvezdy (vyrazuje rozostrene/slite tvary). |
| `phase01_comparison_max_mag_diff` | 1.5; rozsah 0.05 .. 5 | Nastaveni + auto-uprava za behu | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14742`) | Zakladni maximalni rozdil jasnosti (mag) mezi srovnavaci hvezdou a cilem; upravovano profilem hustoty. |
| `phase01_comparison_max_mag_diff_absolute` | 3.0; rozsah 1 .. 10 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1113`) | Tvrdy strop rozdilu jasnosti, ktery zadna adaptace nesmi prekrocit. |
| `phase01_comparison_max_mag_diff_bright_floor` | 1.5; rozsah 0 .. 4 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14758`) | Limit rozdilu jasnosti pro jasne hvezdy bez ohledu na adaptaci. |
| `phase01_comparison_max_psf_chi2` | 50.0; rozsah 1 .. 500 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14760`) | Maximalni chi-kvadrat PSF fitu srovnavaci hvezdy (kontrola tvaru). |
| `phase01_comparison_min_dist_arcsec` | 60.0; rozsah 0 .. 600 | Nastaveni + auto-uprava za behu | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14751`) | Minimalni vzdalenost (arcsec) mezi srovnavaci hvezdou a cilem proti vzajemne kontaminaci; adaptovano hustotou. |
| `phase01_comparison_min_frames_frac` | 0.2; rozsah 0.05 .. 0.95 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14752`) | Minimalni podil snimku, na kterych musi byt srovnavaci hvezda zmerena. |
| `phase01_comparison_n_comp_max` | 8 | Nastaveni (config.json) | config.json | kontrolni hvezda (`check_star_kmag.py:537`) | Maximalni velikost souboru; dle literatury se prinos scintilace nasycuje kolem 6-8 srovnavacich hvezd. |
| `phase01_comparison_n_comp_min` | 3 | Nastaveni + auto-uprava za behu | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14748`) | Minimalni pocet srovnavacich hvezd, o ktery vyber usiluje; adaptace ho na ridkych polich muze snizit. |
| `phase01_comparison_rms_outlier_sigma` | 3.0; rozsah 1 .. 10 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14753`) | Uroven sigma pro oznaceni srovnavaci hvezdy za odlehlou v RMS vuci svemu binu jasnosti. |
| `phase01_ct_extrapolation_tol` | 0.0 | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:8927`) | Povolena extrapolace barevneho rozsahu vztahu barevneho clenu (0 = zadna extrapolace). |
| `phase01_ct_min_comp` | 7; rozsah 2 .. 30 | Nastaveni (config.json) | config.json | tvorba svetelne krivky (`method_lc_output.py:249`) | Minimalni pocet srovnavacich hvezd pro fit vztahu barevneho clenu. |
| `phase01_flux_col` | dao_flux | Nastaveni (config.json) | config.json | fotometricke jadro (Phase 2A) (`photometry_core.py:14763`) | Ktery sloupec toku syti statistiku srovnavacich hvezd Phase 1 (dao_flux = tok z detekcni faze). |
| `phase01_tiers` | [0.5, 1.0, 1.5, 2.0] | Nastaveni (config.json) | pouze vychozi v kodu | fotometricke jadro (Phase 2A) (`photometry_core.py:14744`) | Meze rozdilu jasnosti (mag) pro trideni kandidatu na srovnavaci hvezdy do urovni podle jasnosti, jako rostouci seznam. WAVE-B slucila puvodni skalary phase01_tier{1..4}_mag do tohoto jednoho seznamu. |
| `phase01_use_bprp_primary` | True | Nastaveni (config.json) | config.json | UI aperturni fotometrie (`ui_aperture_photometry.py:1701`) | Pouziva Gaia BP-RP primo jako hlavni barevne kriterium (misto pocitaneho B-V) - podlozena navrhova volba VYVAR. |

## Duveryhodnost a kvalita vysledku

Jak VYVAR znamkuje vlastni vysledky: duvera GREEN/YELLOW/RED srovnavaciho souboru a kontrolni hvezdy, minimalni pocty epoch a snimku a prahy kvality svetelne krivky. Od 06/2026 je hodnota min-comps prahem pro GREEN, ne tvrdym minimem - mensi pocet srovnavacich hvezd elegantne degraduje na YELLOW s odpovidajicim zvetsenim chyb.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `check_select_rms_floor` | 0.0001; rozsah 0 .. 0.01 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2210`) | Dno RMS pri skorovani vyberu kontrolni hvezdy. |
| `check_star_min_epochs` | 5 | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1100`) | Minimalni pocet epoch, ktere musi kontrolni hvezda pokryt, aby jeji verdikt kvality platil. |
| `comp_qa_enabled` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1182`) | Hodnoceni kvality souboru srovnavacich hvezd po epochach, ktere syti verdikt duvery. |
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
| `apply_color_term` | off | Nastaveni (config.json) | config.json | klasifikace fotometrickeho pasma (`band_classify.py:348`) | Zda se na magnitudy aplikuje transformace barevnym clenem ('off' ponechava instrumentalni system). |
| `k2_ceiling` | 0.1 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1532`) | Horni strop fitovaneho koeficientu extinkce druheho radu k2 (mag na airmass na jednotku barvy). |
| `k2_defaults_bprp` | {} | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1508`) | Literarni vychozi hodnoty k2 po pasmech vuci BP-RP, pouzite pri vypnutem nebo neuspesnem fitu. |
| `k2_fit_consistency_sigma` | 2.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1542`) | Pozadavek konzistence (sigma) mezi fitovanym k2 a ocekavanim, nez se fitu uveri. |
| `k2_fit_enabled` | False | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1530`) | Zapina nocni fitovani koeficientu extinkce druheho radu (v2 NIGHT_FIT; VYPNUTO do validace). |
| `k2_fit_lit_factor` | 4.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1548`) | Povoleny nasobek literarni hodnoty k2; fity mimo nej se odmitnou jako nefyzikalni. |
| `k2_fit_min_detectability` | 3.0 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1536`) | Minimalni detekovatelnost (sila signalu), kterou noc musi nabidnout, aby se fit k2 vubec zkusil. |
| `k2_mode` | literature | Nastaveni (config.json) | config.json | UI Nastaveni (`ui_settings.py:1091`) | Zdroj k2: 'literature' (vychozi hodnoty) nebo nocni fit, je-li program fitovani zapnut. |

## Reporty a HR diagram

Co se objevi v souhrnnem PDF reportu: barevny HR diagram pole, online obohaceni zajimavych objektu (SIMBAD, Gaia) a jeho vizualni ladeni.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `hrd_color_bg_box_px` | 96; rozsah 32 .. 512 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:838`) | Vzorkovaci box pozadi (px) pro barvy hvezd v barevnem HR diagramu. |
| `hrd_color_chroma_boost` | 2.2; rozsah 1 .. 3 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:831`) | Zesileni sytosti barev hvezd v HRD pro vizualni citelnost. |
| `hrd_color_chroma_snr` | 3.0; rozsah 0 .. 20 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:822`) | Minimalni SNR barvy, nez hvezda v HRD dostane sytou barvu. |
| `hrd_color_field_enabled` | True | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:809`) | Vykresli barevny HR diagram pozorovaneho pole v reportu. |
| `hrd_color_highlight_mode` | soft | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:819`) | Styl zvyrazneni zajimavych objektu v HRD ('soft' = jemne zvyrazneni). |
| `hrd_color_saturation` | 0.85; rozsah 0 .. 1 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:813`) | Zakladni sytost barev bodu HRD. |
| `hrd_color_white_point` | field_median | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:828`) | Reference bileho bodu barev HRD ('field_median' vyvazuje na medianovou barvu pole). |
| `hrd_dsc_confirm_prob` | 0.9; rozsah 0.5 .. 1 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:803`) | Prah pravdepodobnosti pro potvrzeni klasifikace kandidata na zaklade HRD. |
| `hrd_enrich_max_candidates` | 20; rozsah 1 .. 100 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:760`) | Strop poctu objektu odeslanych do online obohaceni na jeden report. |
| `hrd_enrich_tap_timeout_s` | 20.0; rozsah 5 .. 120 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:767`) | Casovy limit (s) TAP dotazu behem online obohaceni. |
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
| `export_arcsec_per_px` | 1.3 | Dynamicky (FITS / za behu) | vypocteno z WCS | sestaveni a validace konfigurace (`config.py:1253`) | Popisek meritka zapisovany do metadat exportu; vedecka hodnota pochazi z WCS. WAVE-B odstranila jeho zalohu v config.json. |
| `tess_enabled` | False | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:2100`) | Zapina blok krizove analyzy s TESS (porovnani vasi svetelne krivky s daty TESS). |

## System a vykon

Chovani na urovni stroje: pocet paralelnich pracovniku a rezerva RAM. Odvozuje se z vaseho hardwaru; prepisujte, jen kdyz musite.

| Parametr | Vychozi | Typ | Odkud se bere | Kde se pouziva | Vysvetleni |
|---|---|---|---|---|---|
| `per_frame_mp_reserve_ram_gb` | 1.5 | Nastaveni (config.json) | config.json | sestaveni a validace konfigurace (`config.py:1028`) | RAM (GB) drzena volna na pracovnika pri dimenzovani paralelniho zpracovani snimku. |
| `qc_preprocess_workers` | 1 | Interni | prostredi / stroj | kalibrace a zpracovani snimku (`pipeline.py:15384`) | Pocet paralelnich pracovniku predzpracovani; pocita se z CPU/RAM pri startu, lze prepsat promennou prostredi VYVAR_PARALLEL_WORKERS. |
| `skip_processed_directory` | False | Nastaveni (config.json) | config.json | hlavni UI aplikace (`app.py:209`) | Preskakuje drafty, jejichz zpracovany adresar uz existuje (chovani pri navazovani). |
