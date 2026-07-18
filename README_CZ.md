# VYVAR — automatizovaná pipeline pro diferenciální fotometrii

*Read this in English: [README.md](README.md) — English version.*

VYVAR prom?ní noc surových FITS snímk? na sv?telné k?ivky prom?nných hv?zd p?ipravené
k publikaci. Je ur?ený pro amatérského pozorovatele, který chce diferenciální fotometrii
na observatorní úrovni bez ru?ního lad?ní každého kroku: sta?í ukázat na složku se
science snímky a kalibra?ními snímky a VYVAR provede kalibraci, astrometrii, sestaví
soubor srovnávacích hv?zd, zm??í každou hv?zdu, posoudí, nakolik lze každé sv?telné
k?ivce v??it, a vytvo?í reporty i soubory pro odeslání do AAVSO/VAR.ASTRO.

Cílem je *poctivá* fotometrie. VYVAR nevytvá?í jen hezkou k?ivku — kvantifikuje šum,
vybírá srovnávací hv?zdy tak, jak zamýšlí algoritmus Broeg (2005), ov??uje kandidáty
proti Gaia DR3 a TESS a p?ipojuje verdikt d?v?ry, takže víte, zda je detekovaná zm?na
skute?ná, nebo jde o artefakt. Vše b?ží z jediné Streamlit aplikace a každý nastavitelný
parametr je popsán srozumitelným jazykem, takže pipeline m?žete ?ídit z UI nebo p?ímou
úpravou souboru `config.json`.

![Uživatelské rozhraní VYVAR (zástupný obrázek — img/vyvar_ui.png)](img/vyvar_ui.png)

## Co VYVAR umí

- **Kalibrace** — korekce bias/dark/flat, modelování pozadí oblohy a kontrola kvality
  každého snímku ješt? p?ed m??ením.
- **Detekce a Gaia** — detekce zdroj? ve stylu DAOPHOT, slepé astrometrické ?ešení a
  párování s katalogem Gaia DR3 v?etn? klasifikace v HR diagramu pro každou hv?zdu.
- **Souborová fotometrie** — diferenciální fotometrie vážená variabilitou podle Broeg,
  Fernández & Neuhäuser (2005), apertury optimální na SNR (Howell 1989) a automatický
  výb?r srovnávacích hv?zd s politikou minima a plynulou degradací na ?ídkých polích.
- **D?v?ra** — skóre d?v?ry pro každou k?ivku, kontrolní hv?zdy jako sv?dci, ú?tování
  sigma rozpo?tu a k?ížová validace proti TESS, takže sporné detekce jsou ozna?eny, ne skryty.
- **Reporty a odesílání** — PDF report „Summary Measure“ se sv?telnými k?ivkami každé
  hv?zdy a exporty pro AAVSO a VAR.ASTRO s plnými cita?ními hlavi?kami.

## Stav projektu (poctivá ?ísla)

- **963 test?** prochází (19 p?esko?eno) na aktuálním stromu.
- **269** registrovaných a zdokumentovaných konfigura?ních parametr? (viz `docs/VYVAR_PARAMS.md`).
- **Kotevní disciplína (anchor):** výstup fotometrie je hlídán byte po bytu proti
  zmrazenému referen?nímu b?hu, takže refaktoring ani zm?ny konfigurace nemohou tiše
  posunout v?decká ?ísla (viz `docs/VYVAR_VALIDATION.md`).
- Fotometrie byla k?ížov? ov??ena proti AstroImageJ, Muniwin, IRAF apphot a SExtractoru;
  podrobnosti a aktuální shodu najdete v `docs/VYVAR_VALIDATION.md`.

## Instalace

Kompletní krokový instalátor a pr?vodce nastavením (**INSTALL.md**, EN + CZ) vyjde
v p?íští verzi. Zatím zkrácená verze:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pot?ebujete také lokální katalog Gaia DR3. Sestavte jej zevnit? repozitá?e (buildery
importují `gaia_catalog_id.py` z `src_py/`); pomocí `--out` umístíte velkou databázi
kamkoli chcete:

```bash
python GAIA_DR3/build_gaia_catalog.py --mag-limit 16.5 --out <cesta-k-db>
python GAIA_DR3/build_blind_index.py --db <cesta-k-db> --tier both
```

Rychlý zkušební build (malý kousek oblohy):

```bash
python GAIA_DR3/build_gaia_catalog.py --dec-min 89 --dec-max 90 --mag-limit 10 --skip-vacuum --out tmp/smoke_gaia.db
```

Požadavky: Python 3.12 (vyvíjeno a testováno na 3.12), Windows 10/11 nebo Linux,
minimáln? 8 GB RAM (doporu?eno 16 GB). NVIDIA GPU je volitelné a urychluje pouze
astrometrické ?ešení.

## Dokumentace

| Téma | Dokument |
|------|----------|
| Pr?vodce konfigurací (anglicky) | `docs/VYVAR_CONFIG_GUIDE_EN.md` |
| Pr?vodce konfigurací (?esky) | `docs/VYVAR_CONFIG_GUIDE_CZ.md` |
| Všech 269 parametr? (reference) | `docs/VYVAR_PARAMS.md` |
| Manuál pipeline (?esky) | `docs/VYVAR_PIPELINE_CZ.md` |
| Provozní runbook | `docs/VYVAR_RUNBOOK.md` |
| Validace a kotevní disciplína | `docs/VYVAR_VALIDATION.md` |

Ru?ní úprava `config.json` je pln? podporována: soubor je seskupený, okomentovaný a
toleruje komentá?e `//`, a `python dev/scripts/validate_config.py` jej p?ed b?hem
zkontroluje. Podrobnosti najdete v pr?vodcích konfigurací.

## Odkazy na algoritmy

Úplný seznam viz `CITATIONS.bib`. Klí?ové algoritmy:

- **Diferenciální fotometrie:** Broeg, Fernández & Neuhäuser (2005) AN 326:134
- **Fotometrická chyba CCD:** Howell (1989) PASP 101:616
- **ZP sigma-clip / detekce DAOPHOT:** Stetson (1987) PASP 99:191
- **Katalog:** Gaia DR3 — Gaia Collaboration (2023) A&A 674, A1

## Pro vývojá?e

Testy jsou nastaveny p?es `pyproject.toml` (`testpaths = dev/tests`,
`pythonpath = [".", "src_py", "dev"]`), takže sta?í spustit:

```bash
python -m pytest            # celá sada
python -m pytest -q         # tiše
```

Produk?ní moduly jsou v `src_py/`; vývojá?ský materiál (testy, skripty, nástroje,
validace, výsledky) je pod `dev/`. Ko?enový `app.py` je tenký shim, který p?idá
`src_py/` na `sys.path`. Rozvržení repozitá?e a workflow viz `CLAUDE.md`.

## Licence

VYVAR je proprietární. Copyright © 2026 Milan Uhlár. Všechna práva vyhrazena. Bez
p?edchozího písemného souhlasu není povoleno žádné použití, kopírování, úprava ani
distribuce. Software je poskytován bez jakékoli záruky. Viz [LICENSE](LICENSE).

## Citace

Pokud VYVAR p?isp?je k vaší práci, citujte prosím:

> VYVAR: An Automated Differential Photometry Pipeline for Amateur Variable Star
> Observers. (paper in preparation)
