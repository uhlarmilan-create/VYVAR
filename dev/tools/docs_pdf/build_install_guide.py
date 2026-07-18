# Regenerates docs/VYVAR_INSTALL_GUIDE_CZ.pdf. Run from repo root:
# python dev/tools/docs_pdf/build_install_guide.py
import os
ROOT = os.getcwd()
# -*- coding: ascii -*-
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Table, TableStyle, Preformatted, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

S = getSampleStyleSheet()
T  = ParagraphStyle('T', parent=S['Title'], fontSize=22, spaceAfter=4)
H1 = ParagraphStyle('H1', parent=S['Heading1'], fontSize=14.5, spaceBefore=14, spaceAfter=5, textColor=colors.HexColor('#1a3a5c'))
H2 = ParagraphStyle('H2', parent=S['Heading2'], fontSize=11.5, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor('#0f2740'))
B  = ParagraphStyle('B', parent=S['Normal'], fontSize=9.6, leading=13, spaceAfter=4)
M  = ParagraphStyle('M', parent=B, textColor=colors.HexColor('#555555'), fontSize=8.8)
CODE = ParagraphStyle('C', parent=S['Code'], fontSize=8.6, leading=11, backColor=colors.HexColor('#f2f2f2'), borderPadding=5, leftIndent=4)

def esc(t): return t.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def mk(t):
    t = esc(t)
    for a,b in (('[b]','<b>'),('[/b]','</b>'),('[i]','<i>'),('[/i]','</i>')):
        t = t.replace(a,b)
    return t
def P(t, s=B): return Paragraph(mk(t), s)
def box(title, paras, bg='#fdf4e7', border='#d9a45b', tcol='#7a3b00'):
    ts = ParagraphStyle('bt', parent=S['Heading3'], fontSize=10.2, textColor=colors.HexColor(tcol), spaceAfter=3)
    inner = [Paragraph(mk(title), ts)] + [P(p, ParagraphStyle('bb', parent=B, fontSize=9))
                                          for p in paras]
    tb = Table([[inner]], colWidths=[168*mm])
    tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1), colors.HexColor(bg)),
        ('BOX',(0,0),(-1,-1),0.8,colors.HexColor(border)),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    return tb
def tab(rows, widths):
    t = Table([[P('[b]%s[/b]' % c, M) for c in rows[0]]] +
              [[P(c, M) for c in r] for r in rows[1:]], colWidths=[w*mm for w in widths])
    t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#bbbbbb')),
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
        ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
    return t

st = []
st.append(P("VYVAR - Instalacni prirucka a prvni spusteni", T))
st.append(P("Krok za krokem od stazeni po prvni zpracovanou noc. Verze 1.0 (2026-07-18), Windows. Cestina bez diakritiky dle konvence projektu.", M))
st.append(Spacer(1,8))
st.append(P("VYVAR je automatizovana pipeline diferencialni fotometrie promennych hvezd: ze surovych FITS snimku vyrobi kalibrovane, astrometricky vyresene a zmerene svetelne krivky s exportem pro AAVSO/VarAstro a podrobnym PDF reportem. Tato prirucka vas provede instalaci a uplne prvnim behem.", B))
st.append(Spacer(1,4))

st.append(P("1. Co budete potrebovat", H1))
st.append(tab([
 ["Polozka","Pozadavek","Poznamka"],
 ["Operacni system","Windows 10/11 (64-bit) NEBO Linux (x86-64)","Ubuntu/Debian/Fedora/Arch - navod pro oba"],
 ["Python","3.12","Windows: py launcher; Linux: balicek distribuce ci pyenv"],
 ["Git","volitelne","stazeni pres git clone NEBO ZIP z GitHubu"],
 ["Disk - kod + prostredi","~2 GB","repo + virtualni prostredi (.venv)"],
 ["Disk - katalogy","~12.5 GB","Gaia DR3 + VSX + exoplanety + blind indexy"],
 ["Disk - vase data","dle poctu noci","archiv snimku + knihovna kalibraci"],
 ["Cas","~30-60 min","z toho vetsinu kopirovani katalogu"],
 ["Pristup k repozitari","GitHub ucet + pozvanka","repozitar je privatni"],
], [42,44,82]))
st.append(Spacer(1,4))
st.append(box("Dulezite: katalogy", [
 "VYVAR identifikuje hvezdy proti LOKALNI kopii katalogu Gaia DR3 (40+ mil. hvezd). Bez katalogu se pipeline spusti, ale neumi hvezdy pojmenovat ani vybrat srovnavaci hvezdy - vedecky je nepouzitelna. Nejrychlejsi cesta je ZKOPIROVAT katalogovou sadu z existujici instalace (volba [1] v instalatoru); stavba ze zdroju trva hodiny a stahne ~50 GB."]))
st.append(PageBreak())

st.append(P("2. Ziskani VYVAR", H1))
st.append(P("Varianta A - git (doporucena; snadne aktualizace):", H2))
st.append(Preformatted("cd D:\\\ngit clone https://github.com/uhlarmilan-create/VYVAR.git\ncd VYVAR", CODE))
st.append(P("Pri vyzve zadejte GitHub jmeno a jako heslo osobni pristupovy token (PAT), ktery vam poskytl spravce projektu.", B))
st.append(P("Varianta B - ZIP:", H2))
st.append(P("Na strance repozitare zvolte Code -> Download ZIP, rozbalte napr. do D:\\VYVAR. Vsechny dalsi kroky jsou stejne.", B))

st.append(P("3. Spusteni instalatora", H1))
st.append(P("3A. Windows (PowerShell v korenove slozce VYVAR):", H2))
st.append(Preformatted("powershell -ExecutionPolicy Bypass -File .\\install_vyvar.ps1", CODE))
st.append(P("3B. Linux (terminal v korenove slozce VYVAR):", H2))
st.append(Preformatted("# predpoklady (Ubuntu/Debian; jine distribuce obdobne):\nsudo apt install python3.12 python3.12-venv git\nchmod +x install_vyvar.sh\n./install_vyvar.sh", CODE))
st.append(P("Neni-li Python 3.12 v repozitarich distribuce (starsi Ubuntu), pouzijte PPA deadsnakes nebo pyenv. Faze instalatoru jsou na obou systemech IDENTICKE - lisi se jen lomitka v cestach a aktivace prostredi.", B))
st.append(P("Instalator probehne sedmi fazemi; kazda konci radkem [OK] (nebo [FAIL] s vysvetlenim). Lze ho kdykoliv spustit znovu - hotove kroky preskoci.", B))
st.append(tab([
 ["Faze","Co se deje","Co uvidite / zadate"],
 ["1 PYTHON","kontrola Pythonu 3.12","pri absenci odkaz ke stazeni a konec"],
 ["2 VENV","virtualni prostredi + baliky","pip instalace (par minut, velke baliky)"],
 ["3 CATALOGS","katalogova sada","menu [1] kopie / [2] stavba / [3] preskocit"],
 ["4 PATHS","cesty k datum","archiv, kalibr. knihovna, databaze, katalogy"],
 ["5 VALIDATE","kontrola config.json","validate_config.py musi skoncit bez chyb"],
 ["6 SMOKE","zkusebni import aplikace","+ zalozeni prazdne databaze"],
 ["7 FINISH","souhrn a dalsi kroky","presne prikazy pro prvni spusteni"],
], [24,52,92]))
st.append(Spacer(1,4))
st.append(P("3.1 Faze 3 - katalogy podrobne", H2))
st.append(P("Volba [1] Kopie z existujici instalace (doporuceno): zadate korenovou slozku zdrojove instalace (napr. C:\\ASTRO\\python\\VYVAR na hlavnim stroji, pripojeny disk nebo sitova cesta). Instalator predem overi volne misto, zkopiruje nasledujici sadu a u kazdeho souboru overi velikost:", B))
st.append(tab([
 ["Slozka/soubor","Ucel","Priblizna velikost"],
 ["GAIA_DR3/ (sqlite)","identifikace hvezd, vyber kompu","~9.4 GB"],
 ["VSX/ (sqlite)","zname promenne hvezdy","~1-2 GB"],
 ["exoplanets/ (sqlite)","hostitele exoplanet","~10-100 MB"],
 ["blind indexy (fine+wide)","slepe reseni souradnic","~1-2 GB"],
], [52,62,54]))
st.append(P("Volba [2] Stavba ze zdroju: pro pokrocile; stahne desitky GB a trva hodiny. Volba [3] Preskocit: instalace dobehne v OMEZENEM REZIMU - doplnit katalogy lze pozdeji opetovnym spustenim instalatora.", B))
st.append(P("3.2 Faze 4 - cesty", H2))
st.append(P("Instalator nabidne rozumne vychozi hodnoty (pod korenem VYVAR nebo na zvolenem datovem disku) a zapise je do config.json. Zadne cesty z ciziho stroje se NEPRENASEJI - pokud jste repozitar klonovali vcetne cizich cest, zde se nahradi vasimi.", B))
st.append(P("3.3 Stavba katalogu ze zdroju (volba [2]) krok za krokem", H2))
st.append(P("Instalator umi kroky spustit za vas; zde je popsano, co se deje a jak to spustit rucne. PORADI JE ZAVAZNE: blind indexy se stavi z hotove lokalni Gaia databaze, takze Gaia musi byt prvni. Vsechny prikazy spoustejte z korene VYVAR s aktivovanym prostredim.", B))
st.append(P("Krok 1 - Gaia DR3 katalog (nejdelsi; hodiny az den, ~50 GB stazenych dat):", B))
st.append(Preformatted("python GAIA_DR3/build_gaia_catalog.py --help\npython GAIA_DR3/build_gaia_catalog.py", CODE))
st.append(P("Stahuje z archivu ESA Gaia (TAP) po deklinacnich pasech do lokalni SQLite. Stavba je RESUMOVATELNA - pri preruseni (sit, restart) ji spustte znovu a pokracuje od posledniho dokonceneho pasu (tabulka strip_progress). Vysledny soubor ma ~9.4 GB. Volby --dec-min/--dec-max a --mag-limit umozni zkusebni malou stavbu (napr. jen okoli polu).", B))
st.append(P("Krok 2 - blind indexy (z hotove Gaia DB; desitky minut):", B))
st.append(Preformatted("python GAIA_DR3/build_blind_index.py --tier fine\npython GAIA_DR3/build_blind_index.py --tier wide", CODE))
st.append(P("Vytvori dva soubory trojuhelnikovych hash indexu (PKL): fine pro dlouhoohniskove sestavy (Newton), wide pro sirokouhle (kratke ohnisko). Slepy solver si mezi nimi vybira automaticky podle zorneho pole sestavy. Bez techto souboru funguje reseni souradnic jen s napovedou pointingu z FITS hlavicky.", B))
st.append(P("Krok 3 - VSX (zname promenne hvezdy; minuty):", B))
st.append(Preformatted("python VSX/vsx_make.py", CODE))
st.append(P("Krok 4 - exoplanety (NASA Exoplanet Archive; minuty):", B))
st.append(Preformatted("python exoplanets/exoplanet_make.py", CODE))
st.append(P("Po dokonceni zapiste cesty k vyslednym souborum ve fazi 4 instalatora (nebo instalator spustte znovu - hotove kroky preskoci a cesty doplni). Overeni: validate_config.py a v aplikaci Database Explorer musi katalogy videt.", B))
st.append(PageBreak())

st.append(P("4. Prvni spusteni aplikace", H1))
st.append(Preformatted(".\\.venv\\Scripts\\Activate.ps1\nstreamlit run app.py", CODE))
st.append(P("Linux:", M))
st.append(Preformatted("source .venv/bin/activate\nstreamlit run app.py", CODE))
st.append(P("V prohlizeci se otevre http://localhost:8501 s hlavnim panelem VYVAR Dashboard. V leve navigaci najdete: Pipeline (zpracovani noci), Calibration Library (mastery), Database Explorer a Settings (nastaveni).", B))
st.append(P("5. Prvotni nastaveni observatore (JEDNORAZOVE, NUTNE)", H1))
st.append(P("VYVAR uklada fakta o observatori do databaze - bez nich neumi pocitat airmass ani parovat kalibrace. Cerstva databaze je PRAZDNA - observator patri vam, zalozte v Settings postupne:", B))
st.append(tab([
 ["Krok","Kde","Co vyplnit"],
 ["1 Stanoviste (Location)","Settings -> Observatory","nazev, zem. sirka a delka (stupne), nadm. vyska (m)"],
 ["2 Dalekohled (Telescope)","Settings -> Observatory","nazev, prumer (mm), ohniskova vzdalenost (mm)"],
 ["3 Kamera (Equipment)","Settings -> Observatory","model, rozmer pixelu (um), gain, saturacni strop (ADU, je-li znam)"],
], [42,46,80]))
st.append(P("Vyber aktualniho stanoviste se provadi prepinacem v zahlavi Settings; aplikace ho ulozi do config.json. Dokud vlastni stanoviste nevyberete, muze byt volba prazdna/nevyresena - to je v poradku a vyberem se srovna.", B))
st.append(Spacer(1,3))
st.append(P("6. Kalibrace: darky a flaty", H1))
st.append(P("Pred prvni noci nahrajte do Calibration Library sve kalibracni snimky (dark frames pro pouzivane expozice a teploty, flat fieldy pro pouzivane filtry). Knihovna si mastery postavi a paruje je automaticky podle binningu, expozice, teploty a filtru. Bez darku/flatu pipeline noc odmitne kalibrovat - to je zamer (ochrana dat), ne chyba.", B))
st.append(P("7. Prvni noc krok za krokem", H1))
st.append(tab([
 ["Krok","Akce"],
 ["1","Nakopirujte slozku s FITS snimky jedne noci do sveho archivu (cesta z faze 4)."],
 ["2","Pipeline -> Import: vyberte slozku noci; VYVAR nacte hlavicky a zalozi draft."],
 ["3","Spustte zpracovani (kalibrace -> zarovnani -> reseni souradnic -> fotometrie). Prvni beh trva desitky minut dle poctu snimku."],
 ["4","Vysledky: svetelne krivky v UI, SUMMARY MEASURE REPORT (PDF) vc. plneho snimku konfigurace, export pro AAVSO/VarAstro."],
], [12,156]))
st.append(Spacer(1,3))
st.append(box("Tip: prvni noc volte snadnou", [
 "Pro prvni beh zvolte noc s jasnou znamou promennou (napr. zakrytova dvojhvezda s amplitudou > 0.3 mag), 50+ snimky a dobrym pocasim. Uspesny prvni vysledek overi celou instalaci najednou; exoticke pripady nechte na pozdeji."]))
st.append(PageBreak())

st.append(P("8. Overeni instalace", H1))
st.append(Preformatted(".\\.venv\\Scripts\\Activate.ps1\npython dev\\scripts\\validate_config.py\npython dev\\scripts\\session_baseline_check.py --fast", CODE))
st.append(P("Linux:", M))
st.append(Preformatted("source .venv/bin/activate\npython dev/scripts/validate_config.py\npython dev/scripts/session_baseline_check.py --fast", CODE))
st.append(P("Prvni prikaz zkontroluje config.json (syntaxe, nezname klice, rozsahy). Druhy probehne rychlou kontrolu prostredi vc. testu; radek 'deps-outdated' je jen informativni.", B))
st.append(P("9. Reseni problemu", H1))
st.append(tab([
 ["Priznak","Pricina a reseni"],
 ["'python' nenalezen / spatna verze","Nainstalujte Python 3.12 z python.org (zaskrtnete Add to PATH) a spustte instalator znovu."],
 ["pip selze (SSL/proxy)","Firemni sit/antivir; zkuste jinou sit nebo nastavte proxy pro pip."],
 ["Malo mista pri kopirovani katalogu","Zvolte jiny cilovy disk ve fazi 4, instalator spustte znovu."],
 ["Aplikace bezi, ale hvezdy 'Unknown'","Katalogy preskoceny/nekompletni - spustte instalator, volba [1]."],
 ["Port 8501 obsazeny","streamlit run app.py --server.port 8502"],
 ["Import noci: 'no master dark/flat'","Nahrajte kalibrace do Calibration Library (kap. 6) - zamerna pojistka."],
 ["Databaze 'locked' / antivir","Vyjimka antiviru pro datovou slozku (sqlite soubory)."],
 ["Linux: ensurepip is not available","Chybi balicek python3.12-venv - doinstalujte a spustte instalator znovu."],
 ["Linux: Python 3.12 neni v distribuci","PPA deadsnakes (Ubuntu) nebo pyenv install 3.12."],
 ["Linux: prohlizec se neotevre sam","Otevrete rucne http://localhost:8501; na vzdalenem stroji pridejte --server.address 0.0.0.0."],
 ["Stavba Gaia katalogu prerusena","Spustte build_gaia_catalog.py znovu - pokracuje od posledniho pasu (resume)."],
], [52,116]))
st.append(P("10. Kam dal", H1))
st.append(P("[b]VYVAR_CONFIG_GUIDE_CZ.md[/b] - rychla reference vsech parametru. [b]VYVAR_PARAMETER_HANDBOOK_CZ.pdf[/b] - podrobny rozbor parametru s matematikou a literaturou. [b]README_CZ.md[/b] - prehled schopnosti. Config.json lze bezpecne editovat i rucne (ma komentare) a zkontrolovat prikazem validate_config.py.", B))
st.append(Spacer(1,6))
st.append(box("Poznamka pro uzivatele KStars/Ekos (Linux)", [
 "Snimate-li pres KStars/Ekos, VYVAR muze bezet primo na temze linuxovem stroji - odpada prenos dat. Archivni cestu ve fazi 4 nasmerujte na slozku, kam Ekos uklada snimky, a noci importujte primo z ni."]))
st.append(Spacer(1,4))
st.append(P("Tato prirucka odpovida instalatoru verze 1.0 (Windows + Linux, vc. stavby katalogu; v1.2 - cerstva databaze je prazdna, zaznamy zaklada uzivatel). Pripadne odchylky po prvnim ostrem testu (Lenovo) budou promitnuty do verze 1.1.", M))

doc = SimpleDocTemplate(os.path.join(ROOT,'docs','VYVAR_INSTALL_GUIDE_CZ.pdf'), pagesize=A4,
    leftMargin=20*mm, rightMargin=20*mm, topMargin=16*mm, bottomMargin=16*mm,
    title='VYVAR - Instalacni prirucka', author='VYVAR project')
doc.build(st)
print('ok')
