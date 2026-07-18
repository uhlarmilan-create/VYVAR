# VYVAR — automatizovaná pipeline pro diferenciální fotometrii

*Z noci surových FITS snímk? ud?lá d?v?ryhodné sv?telné k?ivky prom?nných hv?zd p?ipravené k odeslání.*

*Read this in English: [README.md](README.md) — English version.*

---

## Co to je

VYVAR je automatizovaná pipeline pro diferenciální fotometrii ur?ená amatérskému pozorovateli
prom?nných hv?zd, který chce výsledky na observatorní úrovni bez ru?ního lad?ní každého kroku.
Sta?í ukázat na složku se science snímky a kalibra?ními snímky a VYVAR provede kalibraci,
astrometrické ?ešení, sestaví soubor srovnávacích hv?zd, zm??í každou hv?zdu, posoudí, nakolik
lze každé sv?telné k?ivce v??it, a vytvo?í reporty i soubory pro odeslání do AAVSO / VarAstro.
Vše b?ží z jediné Streamlit aplikace a každý nastavitelný parametr je popsán srozumitelným
jazykem, takže pipeline m?žete ?ídit z UI nebo p?ímou úpravou souboru `config.json`.

Cílem je *poctivá* fotometrie. VYVAR nekreslí jen hezkou k?ivku — kvantifikuje šum reálným
modelem chyby CCD, vybírá srovnávací hv?zdy tak, jak zamýšlí algoritmus Broeg (2005), ov??uje
kandidáty proti Gaia DR3, VSX a TESS a p?ipojuje verdikt d?v?ry ZELENÁ / ŽLUTÁ / ?ERVENÁ,
takže víte, zda je detekovaná zm?na skute?ná, nebo jde o artefakt. Jeho extrakce byla k?ížov?
ov??ena proti ?ty?em nezávislým nástroj?m a výstupy fotometrie jsou hlídány byte po bytu proti
zmrazené referenci, takže refaktoring nem?že tiše posunout v?decká ?ísla.

---

## Pipeline podrobn?

Každá fáze níže je reálný, testovatelný krok produk?ní cesty.

### Kalibrace
Korekce bias / dark / flat sestaví a aplikuje master kalibra?ní snímky z kalibra?ní knihovny
párované podle teploty a stá?í. **Radiometrická brána CAL-DIAG** (výchozí ZAP) zkontroluje
rozumnou radiometrii kalibrovaných snímk? a zaznamená provenienci ješt? p?ed m??ením.
**Preprocess povrchu oblohy** modeluje a vyrovnává zbytkové gradienty pozadí (M?síc, sv?telné
zne?išt?ní) nízkostup?ovým fitem plochy p?ed detekcí.

### Kontrola kvality (QC)
Každý snímek projde QC, než p?isp?je: kontroly FWHM, pozadí a po?tu hv?zd vy?adí zamra?ené,
rozmazané nebo rozost?ené snímky, aby špatný snímek neotrávil soubor.

### Zarovnání
Snímky se registrují na spole?nou pixelovou m?ížku (astroalign) s m??eným, stabilním posunem
t?žišt? (medián ~0,4 px za noc na valida?ním poli), což nezávisle potvrdila k?ížová validace.

### Astrometrické ?ešení (slepý solver + ov??ení Gaia)
P?estav?ný **slepý plate-solver** (`vyvar_blind_solver.py`) používá density-matched Gaia DR3
trojúhelníkový index (8-NN trojúhelníkové hashe), DBSCAN vote clustering (haversinová metrika)
a ov??ení WCS na úrovni clusteru p?es RANSAC s geometrickým skóre match-fraction místo pouhého
po?tu hlas?. **Dvoutierový index** (fine pro dlouhé ohnisko, wide pro krátké) umož?uje
scale-aware orchestrátoru zvolit správný tier. ?ešení se ov??ují proti Gaia do
`verify_mag_limit = 14` (A/B ov??eno: stejn? spolehlivé jako mag 16 p?i ~28 % kratším b?hu).
Zápis WCS je fail-closed — chyba zápisu zablokuje fázi 2A místo odeslání zastaralé astrometrie.
Slepá astrometrická kalibrace vychází z filosofie hint-as-prior (Lang et al. 2010).

### MASTERSTAR
Hluboký, zarovnaný master snímek dává stabilní t?žišt?, WCS a FWHM pro celou noc, v??i nimž se
ukotvuje fotometrie jednotlivých snímk?.

### Krížové párování Gaia DR3 / VSX / exoplanety
Každá zm??ená hv?zda se páruje s lokálním katalogem Gaia DR3 (SQLite, celooblohov? sestaveným
`build_gaia_catalog.py`), ozna?í se p?íznaky známých prom?nných z AAVSO VSX (Watson, Henden &
Price 2006) a ov??í proti katalog?m hostitel? exoplanet. Barvy Gaia BP-RP a hv?zdné parametry
GSP-Phot (Andrae et al. 2023) vstupují do barevných ?len? a HR diagramu.

### Souborová diferenciální fotometrie
Srovnávací hv?zdy se vybírají a váží podle **Broeg, Fernández & Neuhäuser (2005)** — um?le
sestavená srovnávací hv?zda vážená variabilitou s iterativním potla?ováním prom?nných
referencí. Výb?r ?adí hv?zdy podle **barevných tier? BP-RP** (barevné transformace Gaia; Jordi
et al. 2010, Riello et al. 2021) a poté podle RMS, s `comp_select_rms_floor` (výchozí `1e-6`),
která odstraní artefakty izolovaných bin?. Cesta se **p?izp?sobuje hustot? pole**: plynulá
degradace na ?ídkých polích zachová poctivý výsledek místo selhání a na po?adí nezávislý locus
comp-QA (Sokolovského indexy) ?iní ozna?ování srovnávacích hv?zd reprodukovatelným bez ohledu
na po?adí cíl?. Kanonická kombinace souboru je flux-sum (ov??eno proti AstroImageJ; Collins,
Kielkopf & Stelzer 2017); inverzn?-varian?ní vážení podle Broega je k dispozici, ale
pozastavené, dokud se nevaliduje rozpo?et sigma na m??ení.

### Model chyby
Nejistoty jednotlivých hv?zd používají rovnici signál/šum pro CCD podle **Howella (1989)** s
aperturami optimálními na SNR, ?len pozadí z prázdné apertury (Labbé et al. 2003), potla?ení
spole?ného módu na zbytku souboru podle **Honeycutta (1992)** a systematickou podlahu na rig
(`sigma_sys`), aby formální chyby nepodhodnocovaly skute?ný rozptyl (Merline & Howell 1995).
Vzdušná hmota používá vzorec Kasten & Young (1989). Systematický detrending SysRem (Tamuz,
Mazeh & Zucker 2005) byl vyhodnocen a je k dispozici, ale ve výchozí produk?ní cest? není zapnut.

### Model d?v?ry
Každá sv?telná k?ivka dostane pás d?v?ry **ZELENÁ / ŽLUTÁ / ?ERVENÁ** podle kontrolních hv?zd
jako sv?dk?, po?tu srovnávacích hv?zd (`comp_trust_min_comps`) a diagnostiky rozptylu. ?ídká
pole se vrací k souboru kontrolních hv?zd p?i n>=2 s triangulací variance podle Howell, Warnock
& Mitchell (1988) a pásy d?v?ry z interval? spolehlivosti, takže tenké pole degraduje plynule
místo tichého p?ece?ování p?esnosti.

### Detekce variability + k?ížová analýza TESS
Kandidáti se ozna?ují robustními indexy variability (Sokolovsky et al. 2017; pom?r von Neumann
1941) a periodami z analýzy Lomb-Scargle (Lomb 1976; Scargle 1982; VanderPlas 2018). Každý
kandidát se automaticky ov??í proti TESS (p?es Lightkurve) s kontrolou blendu a klasifikací
spolehlivosti periody.

### Reporty
PDF report **Summary Measure** obsahuje sv?telné k?ivky jednotlivých hv?zd, HR diagram s
klasifikací podle Gaia a plnou **stránku provenience konfigurace** — p?esný snímek parametr?,
git head a ?asové razítko zape?ené v každém reportu, takže je libovolný obrázek reprodukovatelný.

### Export AAVSO / VarAstro
Soubory k odeslání se zapisují s plnými cita?ními hlavi?kami ?ízenými `CITATIONS.bib` (jediný
zdroj pravdy), citují pouze metody, které pro daná data skute?n? prob?hly.

---

## Validace

Fotometrie VYVAR byla k?ížov? ov??ena proti ?ty?em nezávislým profesionálním nástroj?m na poli
draft_310 (BO CVn). Každý ?ádek níže vychází ze záznamu projektu (`docs/VYVAR_JOURNAL.md`);
verze nástroj?, po?ty hv?zd a hodnoty shody jsou nam??ené, nikoli aspira?ní.

| Nástroj | Metoda | Hv?zd | Shoda |
|---------|--------|-------|-------|
| photutils 3.0 | Diferenciální LC vs VYVAR `dao_flux`, mag 8–13 | 67 | ? < 0,001 mag |
| Muniwin 2.1.36 (c-munipack) | Diferenciální LC, stejné srovnávací hv?zdy | 3 | ±5–15 % RMS |
| IRAF apphot (Community IRAF 2.17.1) | Tok z jednoho snímku na MASTERSTAR.fits | 48 | 2,2 % rozptyl (po ZP) |
| SExtractor 2.28 | Tok z jednoho snímku | 273 | 6 % offset (growth curve / k?ídla PSF) |

Nezávislá end-to-end studie na draft_000365 (V842 Her) si postavila vlastní katalog Gaia,
detekci, apertury i pozadí: pipeline t?ídy SExtractor s mesh pozadím (SEP; Barbary 2016)
reprodukuje aperturní extrakci VYVAR na **~0,2 % na snímek** a t?i nezávislé enginy (photutils
+ sep + VYVAR) reprodukují RMS v?decké k?ivky na ~1 % bez systematického offsetu.

**Kotevní disciplína.** Výstupy fotometrie jsou drženy na byte-identické **regresní referenci
SHA-256**, podpo?ené v?decky smysluplným numerickým
komparátorem na úrovni ~`1e-6`, takže refaktoring nebo zm?na konfigurace bu? p?esn? reprodukuje
zmrazená v?decká ?ísla, nebo je ozna?en.

---

## Reprodukovatelnost a inženýrství

- **963 test?** prochází (19 p?esko?eno) na aktuálním stromu.
- **Kotevní brány**: rychlá/plná kontrola baseline na za?átku session znovu ov??í pytest,
  cesty konfigurace a zmrazenou v?deckou kotvu, než se p?ijme nová práce.
- **Provenience v každém reportu**: p?esný snímek konfigurace + git head + ?asové razítko na
  stránce konfigurace reportu.
- **Ru?n? editovatelná konfigurace**: `config.json` je seskupený, okomentovaný a toleruje
  komentá?e `//`; `python dev/scripts/validate_config.py` zkontroluje ru?n? upravený soubor
  p?ed b?hem.
- **Zdokumentovaná plocha parametr?**: **269** registrovaných parametr? (config.json jich
  persistuje **249**); vše popsáno v `docs/VYVAR_PARAMS.md` a v pr?vodcích konfigurací.

---

## Snímky obrazovky

<!-- SNIMEK 1: Zachyt hlavni Streamlit dashboard po plnem behu - pohled RUN VYVAR s
     dokoncenymi fazemi pipeline a viditelnym dashboardem variability/duvery.
     Uloz jako img/readme_dashboard.png -->
![Streamlit dashboard VYVAR po dokon?eném b?hu](img/readme_dashboard.png)

<!-- SNIMEK 2: Zachyt reprezentativni stranku PDF reportu Summary Measure - idealne s HR
     diagramem nebo strankou provenience konfigurace. Uloz jako img/readme_report.png -->
![Stránka PDF reportu Summary Measure z VYVAR](img/readme_report.png)

<!-- SNIMEK 3: Zachyt jednu cistou svetelnou krivku promenne hvezdy (napr. zakrytove
     dvojhvezdy) se srovnavacimi hvezdami a chybovymi useckami. Uloz jako img/readme_lightcurve.png -->
![Ukázková sv?telná k?ivka prom?nné hv?zdy z VYVAR](img/readme_lightcurve.png)

---

## Na jakém hardwaru b?ží

VYVAR je desktopová Python aplikace, nikoli služba. B?ží na:

- **OS:** Windows 10/11 nebo Linux (vyvíjeno na obou).
- **Python:** 3.12 (vyvíjeno a testováno na 3.12).
- **RAM:** minimáln? 8 GB, doporu?eno 16 GB pro velké noci.
- **GPU:** volitelné NVIDIA GPU — urychluje pouze astrometrické ?ešení; není nutné.
- **Data:** jakýkoli dalekohled produkující FITS + monochromatická CMOS/CCD sestava, od
  širokoúhlého krátkého ohniska (~9,8?/px) po dlouhoohniskový Newton (~0,65?/px). VYVAR je
  scale-aware a podle toho volí cestu apertura vs PSF.

Pot?ebujete také lokální katalog Gaia DR3 (staví se jednou; viz Instalace).

---

## Instalace

Kompletní krokový instalátor a pr?vodce nastavením (**INSTALL.md**, EN + CZ) se p?ipravuje.
Zatím zkrácená verze:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Katalog Gaia DR3 postavte jednou (spoušt?jte zevnit? repozitá?e — buildery importují
`gaia_catalog_id.py` z `src_py/`; pomocí `--out` umístíte velkou databázi kamkoli):

```bash
python GAIA_DR3/build_gaia_catalog.py --mag-limit 16.5 --out <cesta-k-db>
python GAIA_DR3/build_blind_index.py --db <cesta-k-db> --tier both
```

Rychlý zkušební build (malý kousek oblohy):

```bash
python GAIA_DR3/build_gaia_catalog.py --dec-min 89 --dec-max 90 --mag-limit 10 --skip-vacuum --out tmp/smoke_gaia.db
```

Spušt?ní test? (nastaveno p?es `pyproject.toml`: `testpaths = dev/tests`,
`pythonpath = [".", "src_py", "dev"]`):

```bash
python -m pytest
```

---

## Dokumentace

| Téma | Dokument |
|------|----------|
| Pr?vodce konfigurací (anglicky) | `docs/VYVAR_CONFIG_GUIDE_EN.md` |
| Pr?vodce konfigurací (?esky) | `docs/VYVAR_CONFIG_GUIDE_CZ.md` |
| Všech 269 parametr? (reference) | `docs/VYVAR_PARAMS.md` |
| Prirucka parametru (cesky, PDF) | `docs/VYVAR_PARAMETER_HANDBOOK_CZ.pdf` |
| Instalacni a spousteci prirucka (cesky, PDF) | `docs/VYVAR_INSTALL_GUIDE_CZ.pdf` |
| Manuál pipeline (?esky) | `docs/VYVAR_PIPELINE_CZ.md` |
| Datový tok kalibrace magnitud (?esky) | `docs/VYVAR_CALIBRATION.md` |
| Provozní runbook | `docs/VYVAR_RUNBOOK.md` |
| Valida?ní harness a kotevní disciplína | `docs/VYVAR_VALIDATION.md` |
| Citace algoritm? a softwaru | `CITATIONS.bib` |

Ru?ní úprava `config.json` je pln? podporována — viz pr?vodci konfigurací a kontrolní skript
`validate_config.py`.

---

## Stav projektu a licence

VYVAR je v aktivním vývoji a používá se pro reálná odeslání prom?nných hv?zd. Aktuální strom:
963 test? zelených, 269 zdokumentovaných parametr?, byte-identická kotevní disciplína
fotometrie. Ve?ejný ?lánek se p?ipravuje.

VYVAR je **proprietární**. Copyright © 2026 Milan Uhlár. Všechna práva vyhrazena. Bez
p?edchozího písemného souhlasu není povoleno žádné použití, kopírování, úprava ani distribuce.
Software je poskytován bez jakékoli záruky. Viz [LICENSE](LICENSE).

## Citace

Pokud VYVAR p?isp?je k vaší práci, citujte prosím:

> VYVAR: An Automated Differential Photometry Pipeline for Amateur Variable Star
> Observers. (paper in preparation)
