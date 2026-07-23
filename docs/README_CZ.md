# VYVAR - automatizovana pipeline pro diferencialni fotometrii

*Z noci surovych FITS snimk? ud?la d?v?ryhodne sv?telne k?ivky prom?nnych hv?zd p?ipravene k odeslani.*

*Read this in English: [README_FULL.md](README_FULL.md) - English version.*

---

## Co to je

VYVAR je automatizovana pipeline pro diferencialni fotometrii ur?ena amaterskemu pozorovateli
prom?nnych hv?zd, ktery chce vysledky na observatorni urovni bez ru?niho lad?ni kazdeho kroku.
Sta?i ukazat na slozku se science snimky a kalibra?nimi snimky a VYVAR provede kalibraci,
astrometricke ?eseni, sestavi soubor srovnavacich hv?zd, zm??i kazdou hv?zdu, posoudi, nakolik
lze kazde sv?telne k?ivce v??it, a vytvo?i reporty i soubory pro odeslani do AAVSO / VarAstro.
Vse b?zi z jedine Streamlit aplikace a kazdy nastavitelny parametr je popsan srozumitelnym
jazykem, takze pipeline m?zete ?idit z UI nebo p?imou upravou souboru `config.json`.

Cilem je *poctiva* fotometrie. VYVAR nekresli jen hezkou k?ivku - kvantifikuje sum realnym
modelem chyby CCD, vybira srovnavaci hv?zdy tak, jak zamysli algoritmus Broeg (2005), ov??uje
kandidaty proti Gaia DR3, VSX a TESS a p?ipojuje verdikt d?v?ry ZELENA / ZLUTA / ?ERVENA,
takze vite, zda je detekovana zm?na skute?na, nebo jde o artefakt. Jeho extrakce byla k?izov?
ov??ena proti ?ty?em nezavislym nastroj?m a vystupy fotometrie jsou hlidany byte po bytu proti
zmrazene referenci, takze refaktoring nem?ze tise posunout v?decka ?isla.

---

## Pipeline podrobn?

Kazda faze nize je realny, testovatelny krok produk?ni cesty.

### Kalibrace
Korekce bias / dark / flat sestavi a aplikuje master kalibra?ni snimky z kalibra?ni knihovny
parovane podle teploty a sta?i. **Radiometricka brana CAL-DIAG** (vychozi ZAP) zkontroluje
rozumnou radiometrii kalibrovanych snimk? a zaznamena provenienci jest? p?ed m??enim.
**Preprocess povrchu oblohy** modeluje a vyrovnava zbytkove gradienty pozadi (M?sic, sv?telne
zne?ist?ni) nizkostup?ovym fitem plochy p?ed detekci.

### Kontrola kvality (QC)
Kazdy snimek projde QC, nez p?isp?je: kontroly FWHM, pozadi a po?tu hv?zd vy?adi zamra?ene,
rozmazane nebo rozost?ene snimky, aby spatny snimek neotravil soubor.

### Zarovnani
Snimky se registruji na spole?nou pixelovou m?izku (astroalign) s m??enym, stabilnim posunem
t?zist? (median ~0,4 px za noc na valida?nim poli), coz nezavisle potvrdila k?izova validace.

### Astrometricke ?eseni (slepy solver + ov??eni Gaia)
P?estav?ny **slepy plate-solver** (`vyvar_blind_solver.py`) pouziva density-matched Gaia DR3
trojuhelnikovy index (8-NN trojuhelnikove hashe), DBSCAN vote clustering (haversinova metrika)
a ov??eni WCS na urovni clusteru p?es RANSAC s geometrickym skore match-fraction misto pouheho
po?tu hlas?. **Dvoutierovy index** (fine pro dlouhe ohnisko, wide pro kratke) umoz?uje
scale-aware orchestratoru zvolit spravny tier. ?eseni se ov??uji proti Gaia do
`verify_mag_limit = 14` (A/B ov??eno: stejn? spolehlive jako mag 16 p?i ~28 % kratsim b?hu).
Zapis WCS je fail-closed - chyba zapisu zablokuje fazi 2A misto odeslani zastarale astrometrie.
Slepa astrometricka kalibrace vychazi z filosofie hint-as-prior (Lang et al. 2010).

### MASTERSTAR
Hluboky, zarovnany master snimek dava stabilni t?zist?, WCS a FWHM pro celou noc, v??i nimz se
ukotvuje fotometrie jednotlivych snimk?.

### Krizove parovani Gaia DR3 / VSX / exoplanety
Kazda zm??ena hv?zda se paruje s lokalnim katalogem Gaia DR3 (SQLite, celooblohov? sestavenym
`build_gaia_catalog.py`), ozna?i se p?iznaky znamych prom?nnych z AAVSO VSX (Watson, Henden &
Price 2006) a ov??i proti katalog?m hostitel? exoplanet. Barvy Gaia BP-RP a hv?zdne parametry
GSP-Phot (Andrae et al. 2023) vstupuji do barevnych ?len? a HR diagramu.

### Souborova diferencialni fotometrie
Srovnavaci hv?zdy se vybiraji a vazi podle **Broeg, Fernandez & Neuhauser (2005)** - um?le
sestavena srovnavaci hv?zda vazena variabilitou s iterativnim potla?ovanim prom?nnych
referenci. Vyb?r ?adi hv?zdy podle **barevnych tier? BP-RP** (barevne transformace Gaia; Jordi
et al. 2010, Riello et al. 2021) a pote podle RMS, s `comp_select_rms_floor` (vychozi `1e-6`),
ktera odstrani artefakty izolovanych bin?. Cesta se **p?izp?sobuje hustot? pole**: plynula
degradace na ?idkych polich zachova poctivy vysledek misto selhani a na po?adi nezavisly locus
comp-QA (Sokolovskeho indexy) ?ini ozna?ovani srovnavacich hv?zd reprodukovatelnym bez ohledu
na po?adi cil?. Kanonicka kombinace souboru je flux-sum (ov??eno proti AstroImageJ; Collins,
Kielkopf & Stelzer 2017); inverzn?-varian?ni vazeni podle Broega je k dispozici, ale
pozastavene, dokud se nevaliduje rozpo?et sigma na m??eni.

### Model chyby
Nejistoty jednotlivych hv?zd pouzivaji rovnici signal/sum pro CCD podle **Howella (1989)** s
aperturami optimalnimi na SNR, ?len pozadi z prazdne apertury (Labbe et al. 2003), potla?eni
spole?neho modu na zbytku souboru podle **Honeycutta (1992)** a systematickou podlahu na rig
(`sigma_sys`), aby formalni chyby nepodhodnocovaly skute?ny rozptyl (Merline & Howell 1995).
Vzdusna hmota pouziva vzorec Kasten & Young (1989). Systematicky detrending SysRem (Tamuz,
Mazeh & Zucker 2005) byl vyhodnocen a je k dispozici, ale ve vychozi produk?ni cest? neni zapnut.

### Model d?v?ry
Kazda sv?telna k?ivka dostane pas d?v?ry **ZELENA / ZLUTA / ?ERVENA** podle kontrolnich hv?zd
jako sv?dk?, po?tu srovnavacich hv?zd (`comp_trust_min_comps`) a diagnostiky rozptylu. ?idka
pole se vraci k souboru kontrolnich hv?zd p?i n>=2 s triangulaci variance podle Howell, Warnock
& Mitchell (1988) a pasy d?v?ry z interval? spolehlivosti, takze tenke pole degraduje plynule
misto ticheho p?ece?ovani p?esnosti.

### Detekce variability + k?izova analyza TESS
Kandidati se ozna?uji robustnimi indexy variability (Sokolovsky et al. 2017; pom?r von Neumann
1941) a periodami z analyzy Lomb-Scargle (Lomb 1976; Scargle 1982; VanderPlas 2018). Kazdy
kandidat se automaticky ov??i proti TESS (p?es Lightkurve) s kontrolou blendu a klasifikaci
spolehlivosti periody.

### Reporty
PDF report **Summary Measure** obsahuje sv?telne k?ivky jednotlivych hv?zd, HR diagram s
klasifikaci podle Gaia a plnou **stranku provenience konfigurace** - p?esny snimek parametr?,
git head a ?asove razitko zape?ene v kazdem reportu, takze je libovolny obrazek reprodukovatelny.

### Export AAVSO / VarAstro
Soubory k odeslani se zapisuji s plnymi cita?nimi hlavi?kami ?izenymi `CITATIONS.bib` (jediny
zdroj pravdy), cituji pouze metody, ktere pro dana data skute?n? prob?hly.

---

## Validace

Fotometrie VYVAR byla k?izov? ov??ena proti ?ty?em nezavislym profesionalnim nastroj?m na poli
draft_310 (BO CVn). Kazdy ?adek nize vychazi ze zaznamu projektu (`docs/VYVAR_JOURNAL.md`);
verze nastroj?, po?ty hv?zd a hodnoty shody jsou nam??ene, nikoli aspira?ni.

| Nastroj | Metoda | Hv?zd | Shoda |
|---------|--------|-------|-------|
| photutils 3.0 | Diferencialni LC vs VYVAR `dao_flux`, mag 8-13 | 67 | ? < 0,001 mag |
| Muniwin 2.1.36 (c-munipack) | Diferencialni LC, stejne srovnavaci hv?zdy | 3 | +-5-15 % RMS |
| IRAF apphot (Community IRAF 2.17.1) | Tok z jednoho snimku na MASTERSTAR.fits | 48 | 2,2 % rozptyl (po ZP) |
| SExtractor 2.28 | Tok z jednoho snimku | 273 | 6 % offset (growth curve / k?idla PSF) |

Nezavisla end-to-end studie na draft_000365 (V842 Her) si postavila vlastni katalog Gaia,
detekci, apertury i pozadi: pipeline t?idy SExtractor s mesh pozadim (SEP; Barbary 2016)
reprodukuje aperturni extrakci VYVAR na **~0,2 % na snimek** a t?i nezavisle enginy (photutils
+ sep + VYVAR) reprodukuji RMS v?decke k?ivky na ~1 % bez systematickeho offsetu.

**Kotevni disciplina.** Vystupy fotometrie jsou drzeny na byte-identicke **regresni referenci
SHA-256**, podpo?ene v?decky smysluplnym numerickym
komparatorem na urovni ~`1e-6`, takze refaktoring nebo zm?na konfigurace bu? p?esn? reprodukuje
zmrazena v?decka ?isla, nebo je ozna?en.

---

## Reprodukovatelnost a inzenyrstvi

- **963 test?** prochazi (19 p?esko?eno) na aktualnim stromu.
- **Kotevni brany**: rychla/plna kontrola baseline na za?atku session znovu ov??i pytest,
  cesty konfigurace a zmrazenou v?deckou kotvu, nez se p?ijme nova prace.
- **Provenience v kazdem reportu**: p?esny snimek konfigurace + git head + ?asove razitko na
  strance konfigurace reportu.
- **Ru?n? editovatelna konfigurace**: `config.json` je seskupeny, okomentovany a toleruje
  komenta?e `//`; `python dev/scripts/validate_config.py` zkontroluje ru?n? upraveny soubor
  p?ed b?hem.
- **Zdokumentovana plocha parametr?**: **269** registrovanych parametr? (config.json jich
  persistuje **249**); vse popsano v `docs/VYVAR_PARAMS.md` a v pr?vodcich konfiguraci.

---

## Snimky obrazovky

<!-- SNIMEK 1: Zachyt hlavni Streamlit dashboard po plnem behu - pohled RUN VYVAR s
     dokoncenymi fazemi pipeline a viditelnym dashboardem variability/duvery.
     Uloz jako img/readme_dashboard.png -->
![Streamlit dashboard VYVAR po dokon?enem b?hu](img/readme_dashboard.png)

<!-- SNIMEK 2: Zachyt reprezentativni stranku PDF reportu Summary Measure - idealne s HR
     diagramem nebo strankou provenience konfigurace. Uloz jako img/readme_report.png -->
![Stranka PDF reportu Summary Measure z VYVAR](img/readme_report.png)

<!-- SNIMEK 3: Zachyt jednu cistou svetelnou krivku promenne hvezdy (napr. zakrytove
     dvojhvezdy) se srovnavacimi hvezdami a chybovymi useckami. Uloz jako img/readme_lightcurve.png -->
![Ukazkova sv?telna k?ivka prom?nne hv?zdy z VYVAR](img/readme_lightcurve.png)

---

## Na jakem hardwaru b?zi

VYVAR je desktopova Python aplikace, nikoli sluzba. B?zi na:

- **OS:** Windows 10/11 nebo Linux (vyvijeno na obou).
- **Python:** 3.12 (vyvijeno a testovano na 3.12).
- **RAM:** minimaln? 8 GB, doporu?eno 16 GB pro velke noci.
- **GPU:** volitelne NVIDIA GPU - urychluje pouze astrometricke ?eseni; neni nutne.
- **Data:** jakykoli dalekohled produkujici FITS + monochromaticka CMOS/CCD sestava, od
  sirokouhleho kratkeho ohniska (~9,8?/px) po dlouhoohniskovy Newton (~0,65?/px). VYVAR je
  scale-aware a podle toho voli cestu apertura vs PSF.

Pot?ebujete take lokalni katalog Gaia DR3 (stavi se jednou; viz Instalace).

---

## Instalace

Kompletni krokovy instalator a pr?vodce nastavenim (**[INSTALL.md](INSTALL.md)**, EN + CZ) je k dispozici.
Zkracena verze pro git dev checkout:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Katalog Gaia DR3 postavte jednou (spoust?jte zevnit? repozita?e - buildery importuji
`gaia_catalog_id.py` z `src_py/`; pomoci `--out` umistite velkou databazi kamkoli):

```bash
python GAIA_DR3/build_gaia_catalog.py --mag-limit 16.5 --out <cesta-k-db>
python GAIA_DR3/build_blind_index.py --db <cesta-k-db> --tier both
```

Rychly zkusebni build (maly kousek oblohy):

```bash
python GAIA_DR3/build_gaia_catalog.py --dec-min 89 --dec-max 90 --mag-limit 10 --skip-vacuum --out tmp/smoke_gaia.db
```

Spust?ni test? (nastaveno p?es `pyproject.toml`: `testpaths = dev/tests`,
`pythonpath = [".", "src_py", "dev"]`):

```bash
python -m pytest
```

---

## Dokumentace

| Tema | Dokument |
|------|----------|
| Pr?vodce konfiguraci (anglicky) | `docs/VYVAR_CONFIG_GUIDE_EN.md` |
| Pr?vodce konfiguraci (?esky) | `docs/VYVAR_CONFIG_GUIDE_CZ.md` |
| Vsech 269 parametr? (reference) | `docs/VYVAR_PARAMS.md` |
| Prirucka parametru (cesky, PDF) | `docs/VYVAR_PARAMETER_HANDBOOK_CZ.pdf` |
| Instalacni a spousteci prirucka (cesky, PDF) | `docs/VYVAR_INSTALL_GUIDE_CZ.pdf` |
| Technicky popis pipeline (cesky, PDF) | `docs/VYVAR_FLOW_CZ.pdf` |
| Manual pipeline (?esky) | `docs/VYVAR_PIPELINE_CZ.md` |
| Datovy tok kalibrace magnitud (?esky) | `docs/VYVAR_CALIBRATION.md` |
| Provozni runbook | `docs/VYVAR_RUNBOOK.md` |
| Valida?ni harness a kotevni disciplina | `docs/VYVAR_VALIDATION.md` |
| Citace algoritm? a softwaru | `CITATIONS.bib` |

Ru?ni uprava `config.json` je pln? podporovana - viz pr?vodci konfiguraci a kontrolni skript
`validate_config.py`.

---

## Stav projektu a licence

VYVAR je v aktivnim vyvoji a pouziva se pro realna odeslani prom?nnych hv?zd. Aktualni strom:
963 test? zelenych, 269 zdokumentovanych parametr?, byte-identicka kotevni disciplina
fotometrie. Ve?ejny ?lanek se p?ipravuje.

VYVAR je **proprietarni**. Copyright (c) 2026 Milan Uhlar. Vsechna prava vyhrazena. Bez
p?edchoziho pisemneho souhlasu neni povoleno zadne pouziti, kopirovani, uprava ani distribuce.
Software je poskytovan bez jakekoli zaruky. Viz [LICENSE](LICENSE).

## Citace

Pokud VYVAR p?isp?je k vasi praci, citujte prosim:

> VYVAR: An Automated Differential Photometry Pipeline for Amateur Variable Star
> Observers. (paper in preparation)
