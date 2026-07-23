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

Initializer vytvori `Archive/`, `CalibrationLibrary/`, slozky pro katalogy, prazdnou
`vyvar.sqlite3`, `config.json` ze sablony a `NEXT_STEPS.txt`.

## 3. Kontrola instalace (selftest)

```
VYVAR.bat --selftest
./vyvar.sh --selftest
```

Vypise verze Pythonu, platformu, datovy adresar, klicove zavislosti a import vsech
zkompilovanych modulu. Exit code 0 = instalace je v poradku.

## 4. Stavba katalogu (povinne - nikdy se nesifi)

Katalogy **vzdy stavi uzivatel** (R2). Skripty z dev checkoutu (nebo zkopirujte vystupy
do datoveho adresare):

| Katalog | Skript | Vystup v datovem adresari |
|---------|--------|---------------------------|
| Gaia DR3 SQLite | `GAIA_DR3/build_gaia_catalog.py` | `GAIA_DR3/vyvar_gaia_dr3.db` |
| Blind indexy | `GAIA_DR3/build_blind_index.py` | `GAIA_DR3/gaia_triangles_*.pkl` |
| VSX local | `VSX/vsx_make.py` | `VSX/vyvar_vsx_local_v2.db` |
| Exoplanety | `exoplanets/exoplanet_make.py` | `exoplanets/vyvar_exoplanet_local.db` |

Cesty nastavte v Settings nebo v `config.json` v **datovem adresari**.

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
