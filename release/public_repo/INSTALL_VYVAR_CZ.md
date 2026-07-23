# VYVAR release bundle - ceska instalacni prirucka (RELEASE-2)

Preview balicek obsahuje **vlastni embedded Python 3.12** (R1). Systemove Python
se nepouziva. Uzivatelska data (config, databaze, Archive, katalogy) jsou v oddelenem
**datovem adresari** mimo instalacni slozku.

## 1. Stazeni a rozbaleni

1. Stahnete `VYVAR-<tag>-win64.zip` nebo `VYVAR-<tag>-linux-x64.tar.gz` z GitHub
   Releases verejneho repozitare `VYVAR-release`.
2. Overte SHA256 proti souboru `SHA256SUMS`.
3. Rozbalte na trvale misto, napr.:
   - Windows: `C:\Program Files\VYVAR\`
   - Linux: `~/apps/vyvar/`
4. Cesty s mezerami jsou podporovany.

## 2. Prvni spusteni

**Windows:** `VYVAR.bat`

**Linux:** `chmod +x vyvar.sh && ./vyvar.sh`

Pri prvnim spusteni se vytvori datovy adresar:

| Platforma | Vychozi datovy adresar |
|-----------|------------------------|
| Windows   | `%LOCALAPPDATA%\VYVAR` |
| Linux     | `~/.local/share/vyvar` |

Prepsani: promenna prostredi `VYVAR_DATA_DIR`.

Initializer vytvori `Archive/Drafts/`, `CalibrationLibrary/`, slozky pro katalogy,
`logs/`, prazdnou `vyvar.sqlite3`, `config.json` ze sablony a `NEXT_STEPS.txt`.

Katalogy a pozorovani **nikdy** nepatri do instalacni slozky (design B1).

## 3. Kontrola instalace (selftest)

```
VYVAR.bat --selftest
./vyvar.sh --selftest
```

Vypise verze Pythonu, platformu, datovy adresar, stav datove kostry, klicove
zavislosti, kontrolu runtime souboru a import vsech zkompilovanych modulu.
Exit code 0 = instalace je v poradku.

## 4. Stavba katalogu (povinne - nikdy se nesifi)

Katalogy **vzdy stavi uzivatel** (R2). Release balicek obsahuje stavebni skripty v
`scripts/catalogs/` (stejny bundled Python; zadny systemovy Python).

### 4.1 Poradi (Gaia prvni)

1. **Gaia DR3 SQLite** (nejvetsi; stahovani z ESA Gaia TAP)
2. **Gaia blind indexy** (lokalni CPU; vyzaduje Gaia DB z kroku 1)
3. **VSX local DB** (VizieR)
4. **Exoplanet local DB** (NASA Exoplanet Archive TAP)

### 4.2 Priklady prikazu (launcher)

Pouzijte `--` pred flagy skriptu. Vystupy defaultne do **datoveho adresare** (sekce 2).

**Linux:**

```
./vyvar.sh --tool build_gaia -- --help
./vyvar.sh --tool build_gaia -- --mag-limit 16.5
./vyvar.sh --tool build_blind_index --
./vyvar.sh --tool build_vsx --
./vyvar.sh --tool build_exoplanets --
```

**Windows:**

```
VYVAR.bat --tool build_gaia -- --help
VYVAR.bat --tool build_gaia -- --mag-limit 16.5
VYVAR.bat --tool build_blind_index --
VYVAR.bat --tool build_vsx --
VYVAR.bat --tool build_exoplanets --
```

Viz `scripts/catalogs/README.md` v instalacni slozce.

### 4.3 Kam se soubory ulozi (datovy adresar)

| Krok | Vychozi cesta (v datovem adresari) |
|------|-------------------------------------|
| Gaia DR3 | `GAIA_DR3/vyvar_gaia_dr3.db` |
| Blind indexy | `GAIA_DR3/gaia_triangles_fine.pkl`, `gaia_triangles_wide.pkl` |
| VSX | `VSX/vyvar_vsx_local.db` |
| Exoplanety | `exoplanets/vyvar_exoplanet_local.db` |

Prepsani: flagy skriptu (`--out`, `--db`, ...) nebo Settings / `config.json`.

### 4.4 Cas a disk (typicky)

| Krok | Zdroj site | Stahovani | Velikost vystupu |
|------|------------|-----------|------------------|
| Gaia G<=16.5 cela obloha | esa.gaia.eu TAP | hodiny az dny | **~9-10 GB** SQLite |
| Blind indexy | (lokalni Gaia DB) | zadne | ~100-500 MB PKL |
| VSX | VizieR B/vsx/vsx | minuty | ~10-50 MB |
| Exoplanety | exoplanetarchive.ipac.caltech.edu | minuty | ~1-5 MB |

Gaia build je **obnovitelny** (tabulka `strip_progress`). Bezpecne restartovat.
Pro test pouzijte uzsi `--dec-min`/`--dec-max` pred celym nebem.

### 4.5 Overeni

```
./vyvar.sh --selftest
sqlite3 ~/.local/share/vyvar/GAIA_DR3/vyvar_gaia_dr3.db "SELECT COUNT(*) FROM gaia_dr3;"
```

Pocet radku zavisi na mag limitu a pokryti. Po stavbe overte cesty v Settings nebo
`config.json`.

### 4.6 Kdy rebuildovat

| Situace | Akce |
|---------|------|
| Prvni instalace | Cela sekvence (4.1) |
| Zvyseny Gaia mag limit | Gaia + blind indexy |
| Zvyseny VSX mag limit | Jen VSX (inkrementalne) |
| Aktualizace exoplanet archivu | Jen exoplanet builder |
| Nova instalace VYVAR | Datovy adresar ponechat; rebuild volitelny |

## 5. Vybaveni (DB Explorer)

V UI otev?ete **Database Explorer**:

1. **Location** (lat/lon/vyska)
2. **Telescope**
3. **Equipment** (kamera, filtry, gain)

U **OSC kamer** nastavte **BAYERMASK** u zaznamu equipment (povinne pro spravny debayer).

## 6. Upgrade

1. Ukoncete VYVAR.
2. Nahradte **instalacni adresar** novym balickem.
3. **Nemazejte** datovy adresar.

## 7. Odinstalace

Smazte instalacni adresar; volitelne i datovy adresar.

## 8. Reseni problemu

| Problem | Reseni |
|---------|--------|
| Aplikace nespusti | `--selftest`; antivirus / SmartScreen (nepodepsane binarky) |
| Zadne hvezdy/kompy | Katalogy chybi - sekce 4 |
| Linux chyby importu | **glibc >= 2.39** (Ubuntu 24.04) |
| Spatny datovy adresar | `VYVAR_DATA_DIR` pred spustenim |

## Licence

Soubor `LICENSE` v instalacni slozce (proprietary).
