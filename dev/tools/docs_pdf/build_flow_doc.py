# -*- coding: ascii -*-
# Regenerates docs/VYVAR_FLOW_CZ.pdf. Run from repo root:
#   python dev/tools/docs_pdf/build_flow_doc.py
# Content policy: this builder holds the FULL static text of the technical
# pipeline documentation (v3, full-depth edition). Content changes are made
# HERE and the PDF is regenerated as part of the docs-revision ritual.
import os
import sys
ROOT = os.getcwd()
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from flow_doc_facts import DOC_CONFIG_FACTS, DOC_FUNCTIONS, ANCHOR_ID  # noqa: E402,F401
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Table, TableStyle, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

S = getSampleStyleSheet()
T  = ParagraphStyle('T', parent=S['Title'], fontSize=21, spaceAfter=4)
H1 = ParagraphStyle('H1', parent=S['Heading1'], fontSize=14, spaceBefore=13, spaceAfter=5, textColor=colors.HexColor('#1a3a5c'))
H2 = ParagraphStyle('H2', parent=S['Heading2'], fontSize=11, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor('#0f2740'))
H3 = ParagraphStyle('H3', parent=S['Heading3'], fontSize=10, spaceBefore=6, spaceAfter=2, textColor=colors.HexColor('#274a6d'))
B  = ParagraphStyle('B', parent=S['Normal'], fontSize=10.0, leading=13.8, spaceAfter=5)
M  = ParagraphStyle('M', parent=B, textColor=colors.HexColor('#555555'), fontSize=9.0, leading=12.2)
FN = ParagraphStyle('FN', parent=B, fontName='Courier', fontSize=8.5, leading=11.0, textColor=colors.HexColor('#333355'))
EQ = ParagraphStyle('EQ', parent=B, fontName='Courier', fontSize=9.2, leading=12.2, leftIndent=8, textColor=colors.HexColor('#1a2a4a'))

def esc(t): return t.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def mk(t):
    t = esc(t)
    for a,b in (('[b]','<b>'),('[/b]','</b>'),('[i]','<i>'),('[/i]','</i>'),
                ('[sup]','<super>'),('[/sup]','</super>'),('[sub]','<sub>'),('[/sub]','</sub>'),
                ('[c]','<font face="Courier" size="8.3" color="#333355">'),('[/c]','</font>')):
        t = t.replace(a,b)
    return t
def P(t, s=B): return Paragraph(mk(t), s)
def EQP(t): return Paragraph(mk(t), EQ)
def box(title, paras, bg='#fdf4e7', border='#d9a45b', tcol='#7a3b00'):
    ts = ParagraphStyle('bt', parent=S['Heading3'], fontSize=10, textColor=colors.HexColor(tcol), spaceAfter=3)
    inner = [Paragraph(mk(title), ts)] + [P(p, ParagraphStyle('bb', parent=B, fontSize=8.9, leading=11.6)) for p in paras]
    tb = Table([[inner]], colWidths=[168*mm])
    tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor(bg)),
        ('BOX',(0,0),(-1,-1),0.8,colors.HexColor(border)),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    return tb
def tab(rows, widths):
    t = Table([[P('[b]%s[/b]' % c, M) for c in rows[0]]] + [[P(c, M) for c in r] for r in rows[1:]],
              colWidths=[w*mm for w in widths])
    t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#bbbbbb')),
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
        ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
    return t
def fn(t): return P("[b]Moduly a funkce:[/b] " + t, FN)
def par(t): return P("[b]Parametry (config.json):[/b] " + t, FN)

st = []

# ============================================================ TITLE
st.append(P("VYVAR - Technicky popis pipeline (od zacatku do konce)", T))
st.append(P("Plna technicka dokumentace toku zpracovani a algoritmu: konkretni moduly, funkce, parametry s vychozimi hodnotami, matematika aperturni fotometrie, pripravena ePSF vetev, QA/trust vrstva a exporty. Verze 3.0 (2026-07-19, HEAD main po PUSH drzeneho stacku). Cestina bez diakritiky dle konvence projektu; dokument je generovan builderem (dev/tools/docs_pdf/build_flow_doc.py) a udrzuje se s kodem - obsahove zmeny se zapisuji do builderu a PDF se regeneruje.", M))
st.append(Spacer(1,4))
st.append(P("Patri do rodiny dokumentu VYVAR (STATE, DECISIONS, ROADMAP, PROCESS, PARAMS, JOURNAL). Rozhodnuti a jejich zduvodneni ziji v DECISIONS; otevrena prace v ROADMAP; parametry v komentovanem config.json a ve VYVAR_PARAMETER_HANDBOOK_CZ.pdf; instalace ve VYVAR_INSTALL_GUIDE_CZ.pdf. Nazvy funkci a vychozi hodnoty v tomto dokumentu byly overeny proti zivemu kodu a config.json k datu vydani; pri odchylce plati kod a dokument se regeneruje.", B))
st.append(box("Jak cist tento dokument", [
 "Kapitoly 1-2 vysvetluji filozofii a celkovou architekturu - doporucny start pro kazdeho.",
 "Kapitoly 3-9 sleduji data od FITS souboru po vyber srovnavacich hvezd (priprava mereni).",
 "Kapitoly 10-11 jsou jadro: aperturni fotometrie DO DETAILU (matematika, chybovy model) a pripravena ePSF vetev.",
 "Kapitoly 12-14 popisuji kontrolu kvality (trust gate), detekci promennych a vystupy.",
 "Kapitoly 15-17 pokryvaji katalogy, konfiguraci, reprodukovatelnost a vyvojove garance.",
 "Bloky [b]Moduly a funkce[/b] na konci sekci odkazuji do kodu; bloky [b]Parametry[/b] uvadeji klice config.json s vychozimi hodnotami. Ramecky jako tento shrnuji dulezite koncepty."]))


st.append(P("Obsah", H1))
for line in [
 "1. Uvod, poslani a filozofie navrhu (+ co VYVAR neni; ranni pruchod)",
 "2. Prehled architektury (moduly, UI, tok dat, vykon)",
 "3. Vstupni data: rigy, FITS, archiv a provenience parametru",
 "4. Kalibrace snimku (mastery, knihovna, dark_resample, CAL-DIAG, sky-surface)",
 "5. Kontrola kvality snimku (QC metriky, auto FWHM, brany)",
 "6. MASTERSTAR a astrometrie (hintovane + slepe reseni, SIP, verifikace)",
 "7. Zarovnani snimku",
 "8. Per-frame katalogy a detekce (sloupce, casy, DAO-RECONCILE)",
 "9. Faze 0 - priprava cilu",
 "10. Faze 1 - vyber srovnavacich hvezd (kaskada, tiery, pool, adaptace)",
 "11. Faze 2A - aperturni fotometrie DETAILNE (apertura, chyby, ensemble, k'', AC)",
 "12. Faze 2B - ePSF fotometrie DETAILNE (modely, mereni, audit, gate)",
 "13. Kontrola kvality a duvera (comp QA, check star, trust gate, sparse trust)",
 "14. Detekce promennych a overeni (hockey-stick, VDI, crossmatch, TESS)",
 "15. Vystupy: PDF report a exporty (AAVSO, VarAstro, HRD)",
 "16. Katalogy a databaze",
 "17. Konfigurace, parametry a reprodukovatelnost (anchor, ritual)",
 "18. Vyvojovy proces a garance kvality",
 "Omezeni a zname hranice",
 "Pruvodce: jedna noc od zacatku do konce",
 "Caste otazky (FAQ)",
 "19. Vedecke reference",
 "Prilohy: A struktura draftu, B slovnicek, C sloupce LC CSV, D diagnostika situaci, E rodina dokumentu",
]:
    st.append(P(line, ParagraphStyle('toc', parent=B, spaceAfter=1.5, leftIndent=4)))
st.append(PageBreak())
# ============================================================ CH 1
st.append(P("1. Uvod, poslani a filozofie navrhu", H1))
st.append(P("VYVAR je vysoce automatizovany pipeline diferencialni fotometrie promennych hvezd: od surovych FITS snimku po svetelne krivky a exporty (AAVSO Extended Format, B.R.N.O./VarAstro) s vycislenou duverou v kazde publikovane cislo. Cilem je, aby amatersky pozorovatel rano po noci nasel hotovy report a mohl se rozhodnout, co odeslat - bez rucniho klikani, ale take bez slepe viry v automat. Navrh stoji na sesti zasadach:", B))
st.append(P("[b]1. Duvera na prvnim miste (trust-first).[/b] Kazdy cil prochazi krizovou validaci a trojbarevnym trust gate (GREEN/YELLOW/RED). Report neukazuje jen krivku, ale i to, JAK moc ji verit a proc. Mereni s malo srovnavacimi hvezdami se nezahazuje binarne - degraduje plynule do YELLOW s vetsi chybovou usecckou (kap. 12).", B))
st.append(P("[b]2. Aperturni fotometrie jako overeny zaklad; ePSF jako opt-in.[/b] Na sirokem poli (~9.77 arcsec/px) je hvezdny profil dobre navzorkovany a apertura vitezi jednoduchosti i robustnosti; aperturni cesta je validovana proti AstroImageJ (Delta < 0.001 mag na 67 hvezdach). Plna ePSF vetev (kap. 11) je pripravena a kodove auditovana, ale globalne vypnuta do validace na hustem poli Newtonu (~0.65 arcsec/px). Zasada: zadna metoda se nezapina bez validace na realnych datech.", B))
st.append(P("[b]3. Gaia DR3 nativne.[/b] Paterni katalog je lokalni Gaia DR3 (SQLite, 40+ mil. hvezd, ~9.4 GB) - identity hvezd, astrometrie, fotometrie i barvy. Barva se pouziva primo jako BP-RP; historicka konverze pres Johnsonovo B-V byla kompletne odstranena (DECISIONS 2026-06-25). Cil bez Gaia protejsku padne do nejnizsiho barevneho tieru (neznama barva), nikoli do vymyslene hodnoty.", B))
st.append(P("[b]4. Nocni davkove zpracovani.[/b] Pipeline bezi po skonceni pozorovani, zatimco pozorovatel spi. Vazanou velicinou neni rychlost behu, ale spravnost a duvera raniho reportu. Proto si VYVAR muze dovolit plnou detekci hvezd na kazdem snimku (kap. 8) - CPU cas se vymeni za robustni QC, ktere by 'odecet na pevnych pixelech' ztratil.", B))
st.append(P("[b]5. Vysvetlitelna statistika, ne cerna skrinka.[/b] Detekce promennych a QA stoji na transparentnich, citovatelnych indexech (Sokolovsky, von Neumann, RMS obalka pole, T-statistika kontrolnich hvezd), ne na neuronove siti. Kazdy prah ma jmeno v config.json a lze jej dohledat i zduvodnit.", B))
st.append(P("[b]6. Reprodukovatelnost jako vlastnost.[/b] Kazdy vysledek nese provenance (plny config snapshot, git hash, resolved facts). Vedecke jadro je jisteno bajtove identickym anchorem (draft_435) a sessionovym ritualem --fast/--full (kap. 16-17). Zmena, ktera meni cisla, musi byt ohranicena a vysvetlitelna; zmena, ktera cisla menit nema, musi byt bitove identicka.", B))
st.append(P("Dulezity dusledek techto zasad: kdyz nejaka 'vylepsovaci' metoda v testech poskozuje signal, VYVAR ji vypne a vysledek zdokumentuje (negativni vysledky v DECISIONS). Priklady: casove binovani komparaci (zhorsilo 24 z 25 cilu), SysRem (riziko pozirani skutecne variability), proximitni tie-break vyberu komparaci (obetoval stabilitu za blizkost). Filozofie: [i]nersit symptom - hledat pricinu[/i].", B))



st.append(box("Miniprimer: co je diferencialni fotometrie (pro nove uzivatele)", [
 "Absolutni jasnost hvezdy ze zeme merit temer nejde - atmosfera 'dycha' (pruhlednost, seeing, extinkce se meni minutu po minute). Trik: merit ROZDIL jasnosti cile a blizkych srovnavacich hvezd na TOMTEZ snimku. Vse spolecne (mrak, opar, rosa na optice) postihne cil i reference stejne a v rozdilu zmizi.",
 "Co v rozdilu NEzmizi: skutecna zmena cile (to chceme!) a systematiky, kterymi se cil od referenci LISI - jina barva (kap. 10-11: k''), jina pozice na cipu (flat), jina jasnost (nelinearita). Cely navrh VYVARu je o tom, aby tyto rozdily byly male (vyber kompu) a zbytky korigovane ci aspon vycislene (chybovy model, trust).",
 "Vysledkem je krivka delta_mag s presnosti radove jednotek mmag i z prumerneho mesta - o 2-3 rady lepsi, nez by dala absolutni fotometrie za tychz podminek."]))
st.append(P("1.1 Co VYVAR je - a co zamerne neni", H2))
st.append(P("VYVAR JE: automatizovany nocni pipeline diferencialni fotometrie s vycislenou duverou, stavejici na lokalnich katalozich (offline-first) a na reprodukovatelnem vedeckem jadru. VYVAR NENI: interaktivni fotometricky editor (jako AIJ/SIPS - v nich se klika, VYVAR bezi davkove), neni to nastroj periodove analyzy vlastnich dat (rozsah produktu je 'svetelne krivky dovnitr, periodova veda ven' - periody resi specializovane nastroje nad exportovanymi LC; TESS periodova analyza v kap. 14 slouzi jen k OVERENI kandidatu, ne jako produkt) a neni to multi-night platforma (kanonickou publikovatelnou jednotkou je JEDNA noc; vicenocni globalni zeropoint je vedome odlozen jako nice-to-have - DECISIONS 2026-06-25 a 2026-06-09).", B))
st.append(P("Dlouhodoba vize (ROADMAP): plne autonomni uzavrena smycka observatore PLANNER -> EXECUTOR (KStars/Ekos) -> VYVAR -> REPORT -> PLANNER, kde VYVAR je fotometricko-analyticky clanek. Soucasny produkt je stredni cast: od surovych FITS po report a export.", B))
st.append(P("1.2 Ranni pruchod ocima uzivatele", H2))
st.append(P("Typicke rano vypada takto: [b](1)[/b] otevrete SUMMARY MEASURE REPORT draftu; titulni strana ukaze pocet zmerenych cilu, RMS obalku pole a pripadne kandidaty. [b](2)[/b] U kazdeho cile vidite krivku s trust badge - GREEN berete, u YELLOW ctete duvod (tenky ensemble? check star 0.02-0.05 mag?), RED znamena 'necist krivku, cist diagnostiku'. [b](3)[/b] Konfiguracni strana + Resolved Facts odpovi na kazde 'odkud se vzalo tohle cislo'. [b](4)[/b] Export AAVSO/VarAstro odesilate az po lidske kontrole - trust je inform-only, nikdy nemaze data za vas. Cely zbytek dokumentu vysvetluje, co se mezi FITS a timto ranem deje.", B))
# ============================================================ CH 2
st.append(P("2. Prehled architektury", H1))
st.append(P("Zpracovava se jeden [b]draft[/b] = jedna pozorovaci rada jednoho pole v jednom filtru (jedna noc, jedna sestava). Hlavni tok:", B))
st.append(P("[c]import -> kalibrace (+CAL-DIAG) -> QC -> zarovnani -> plate solve -> MASTERSTAR -> per-frame katalogy -> Faze 0 (cile) -> Faze 1 (kompy) -> Faze 2A (aperturni fotometrie) [-> Faze 2B ePSF, vypnuto] -> trust/QA -> detekce promennosti -> report + export[/c]", B))
st.append(P("Produkcnim vstupem cele fotometrie je jedina funkce [c]run_full_photometry_pipeline[/c] (photometry_core.py) - vsechny validace a regresni testy jdou pres ni, nikdy ne pres prime volani vnitrnich funkci (lekce z retrahovaneho 'Brno fixu': vysledek overeny mimo produkcni cestu neplati). Po REPO-REORG zije produkci kod v [c]src_py/[/c] (96 modulu), vyvojove nastroje v [c]dev/[/c] (tests, tools, validation, scripts, sandbox), korenovy [c]app.py[/c] je tenky Streamlit shim.", B))
st.append(tab([
 ["Modul (src_py/)","Ucel"],
 ["pipeline.py","orchestrace behu draftu: kalibrace, zarovnani, masterstar, per-frame katalogy, PSF sloupce"],
 ["importer.py","nacteni noci, parovani masteru z kalibracni knihovny (vc. teplotni tolerance)"],
 ["calibration.py","stavba/aplikace masteru, stari masteru, prevzorkovani na jiny binning, normalizace flatu"],
 ["cal_diag.py","CAL-DIAG radiometricka brana kalibrace (fail-closed, autocorrect SUM/MEAN)"],
 ["vyvar_alignment_frame.py","zarovnani serie (astroalign, kontrolni body, rezidua)"],
 ["vyvar_platesolver.py + vyvar_blind_solver.py","WCS: hintovane i slepe reseni, SIP, RANSAC, overeni proti Gaia"],
 ["photometry_core.py","jadro (~15300 radku): Faze 0/1/2A, chybovy model, ensemble, trust vstupy"],
 ["photometry_phase2a.py","Faze 2A pomocna vrstva (mereni per snimek)"],
 ["psf_photometry.py + psf_runner.py + psf_neighbor_sub.py","ePSF vetev: modely, mereni, odecet sousedu (pripraveno, vypnuto)"],
 ["comp_qa_core.py","LOO QA komparaci (Sokolovsky indexy + magnitudovy locus)"],
 ["trust_flag_core.py + sparse_trust_core.py","trust gate; T/X2 statistiky kontrolni hvezdy (Howell 1988)"],
 ["check_star_kmag.py","vyber check star a jeji merena KMAG pro AAVSO"],
 ["k2_extinction.py + band_classify.py","extinkce 2. radu k'' (BP-RP-nativni) + klasifikace pasma"],
 ["variability_detector.py + tess_verify.py","detekce kandidatu (RMS obalka, VDI) + TESS overeni period"],
 ["dilution.py + crowding_index.py + lunar_context.py","volitelna QA diagnostika: redeni toku, crowding, Mesic"],
 ["report_methods.py + pdf_report.py + photometry_report.py","SUMMARY MEASURE REPORT, exporty AAVSO/VarAstro"],
 ["hrd_analysis.py + hrd_colorfield.py + hrd_enrich.py","HR diagram pole (GSP-Phot), barevne pole, online obohaceni"],
 ["database.py","SQLite: observator, knihovna kalibraci, drafty, katalogy"],
 ["param_resolver.py + params_registry.py","provenience parametru (DB/FITS/config) + registr 269 klicu"],
 ["dao_reconcile.py","Gaia<->DAO uplnost pole (Fleming 1995, miss@G90)"],
], [52,116]))
st.append(P("Sestavy projektu (referencni sada autora): wide Carl-Zeiss 200 mm + QHY294MM (~9.77 arcsec/px, Jirny), Newton 300/1200 + C3-26000/IMX571 (~0.65 arcsec/px bin1, Dablice/Zdanice), Brno AZ800 80 cm + C5A-150M/IMX411 (~0.566 arcsec/px). [b]Univerzalita:[/b] novy uzivatel zaklada vlastni sestavy do prazdne databaze (referencni sada autora je pouze harness-only seed v dev/tools/reference_seed.py); pipeline se adaptuje na binning, meritko i hustotu pole automaticky - odvozene veliciny se ctou z WCS a FITS, ne z natvrdo zapsane konfigurace.", B))


st.append(P("2.1 Uzivatelske rozhrani (Streamlit) a headless rezim", H2))
st.append(P("UI (app.py -> src_py/vyvar_app.py) je organizovane do zalozek podle toku: Import (noc, parovani masteru), Calibrate (kalibrace + QC + CAL-DIAG vysledky), Align, Platesolve/MASTERSTAR, Photometry (Faze 0/1/2A, badges cilu, tabulky kompu), Variability (hockey-stick, kandidati), Reports/Export a Settings (Observatory, config editace s napovedou z registru). Kazda faze jde spustit i HEADLESS (bez UI) - session_baseline_check --full presne takto spousti celou pipeline; UI je jen tenka vrstva nad tymiz funkcemi, takze co vidite v UI, je tocitelne i skriptem (zadna 'UI-only' veda).", B))
st.append(P("2.2 Tok dat v kostce (co z ceho vznika)", H2))
st.append(P("[c]FITS lights + mastery -> calibrated/*.fits (vy_* QC hlavicky) -> zarovnane FITS -> MASTERSTAR.fits + masterstars_full_match.csv -> proc_*.csv (per snimek) -> active_targets.csv -> comparison_stars_per_target.csv -> lightcurves/*.csv + photometry_summary.csv -> trust + report PDF -> export/[/c]", B))
st.append(P("Kazdy soubor vpravo vznika VYHRADNE ze souboru vlevo od sebe - zadne skryte stavy. Diky tomu lze pipeline restartovat od libovolneho checkpointu a --full komparator umi porovnavat vedecky obsah po souborech.", B))

st.append(P("2.3 Vykon a paralelismus", H2))
st.append(P("Per-frame kroky (kalibrace, detekce, mereni) bezi v multiprocessing poolu; pocet workeru se dimenzuje z dostupne RAM (rezerva per_frame_mp_reserve_ram_gb=1.5 GB na workera) - pipeline se tak sam prizpusobi od notebooku po pracovni stanici. Zarovnane snimky se pri dostatku pameti predavaji v RAM (handoff), zapis na disk zustava checkpointem. Behove parametry (workery, RAM) NEJSOU vedecke - jejich zmena nesmi zmenit vysledek, coz hlida bajtova reprodukovatelnost (17.3): tentyz draft na 2 i 16 jadrech da identicke CSV.", B))
# ============================================================ CH 3
st.append(P("3. Vstupni data: rigy, FITS, archiv a provenience parametru", H1))
st.append(P("3.1 Pozorovaci sestavy (rigy) a odvozovani meritka", H2))

st.append(tab([
 ["Sestava (referencni sada autora)","Meritko","Charakter"],
 ["Carl-Zeiss 200 mm + QHY294MM (Jirny)","~9.77 arcsec/px (bin2)","wide field, NoFilter; jasne/stredni cile, velke FOV, undersampling PSF"],
 ["Newton 300/1200 + C3-26000 (IMX571)","~0.65 / ~1.30 arcsec/px (bin1/bin2)","filtrovana fotometrie, husta pole; cilovy rig pro PSF validaci"],
 ["Brno AZ800 80 cm + C5A-150M (IMX411)","~0.566 arcsec/px","velka apertura, slabe cile; pozorovatel Zejda"],
], [64,30,74]))
st.append(P("Sestava (LOCATION + TELESCOPE + EQUIPMENTS) je statickym faktem v databazi - novy uzivatel ji zada jednou v Settings -> Observatory. Meritko snimku (arcsec/px) se ale NIKDY neprebira slepe z konfigurace: primarnim zdrojem je vyresene WCS (CD matice, proj_plane_pixel_scales), konfiguracni hodnota je az posledni zachrana. Duvod: placeholder v konfiguraci uz jednou zpusobil omyl (historicka hodnota '1.3 arcsec/px' pro Newton); WCS lze naopak overit proti katalogu. Pixelova geometrie (aperturni a anulove polomery, SNR tabulka) se pocita v pixelech, takze je vuci zamene meritka imunni.", B))
st.append(P("3.2 FITS metadata a resolver provenience", H2))

st.append(tab([
 ["FITS klic(e)","Vyznam a uziti"],
 ["EXPTIME / EXPOSURE","expozicni cas [s]; obs_group, casova osa, parovani darku"],
 ["XBINNING / YBINNING","binning; parovani/prevzorkovani masteru, RN_eff = RN x bin"],
 ["GAIN / EGAIN","zisk [e-/ADU]; chybovy model (resolver: DB vybaveni ma prednost, hlavicka varuje pri neshode)"],
 ["CCD-TEMP","teplota cipu; vyber darku v toleranci +-0.5 C"],
 ["FILTER","filtr; obs_group, band_classify (k'', CT, AAVSO kod)"],
 ["DATE-OBS","cas zacatku expozice (UTC); JD -> HJD/BJD"],
 ["RA / DEC / OBJCTRA / OBJCTDEC","zamereni montaze; hint pro plate solve (chybi-li, nastupuje blind)"],
 ["SITELAT / SITELONG / SITEELEV","stanoviste (fallback; primarni je DB LOCATION)"],
 ["vy_* (zapisuje VYVAR)","QC metriky (vy_fwhm, vy_hfr, vy_elong, vy_nstars, vy_qc status, vy_algn priznak zarovnani...)"],
], [42,126]))
st.append(P("Z FITS hlavicek se za behu ctou: expozicni cas, binning, rozmer pixelu, ohnisko, filtr, cas expozice, teplota cipu, pripadne souradnice ze zamereni montaze. Kazdy parametr ma definovany zdrojovy retezec (param_resolver.py): [b]vybaveni[/b] = DB (validni) -> hlavicka (varovani pri neshode) -> config; [b]pozorovani[/b] = hlavicka -> DB -> config; [b]stanoviste[/b] = per-draft -> hlavicka -> config. Nikdy se nic nepouzije potichu - kazda vyresena hodnota se zapisuje do sekce Resolved Facts v reportu, takze BJD, airmass i gain jsou zpetne dohledatelne k svemu zdroji.", B))
st.append(P("Tri druhy hodnot v systemu (viz tez kap. 16): [b]nastaveni[/b] (config.json - politika, prahy, prepinace), [b]staticka fakta[/b] (databaze - observator, kamery, katalogy) a [b]dynamicke hodnoty[/b] (FITS za behu - gain, read noise, rozmery, filtr, expozice). Toto deleni je zamerne: geometrie/FWHM/saturace jsou ODVOZENE veliciny (konfigurace je jen override), zatimco vedecke prahy (limity VSX, kriteria vyberu komparaci, QC prahy) jsou POLITIKA a ziji v konfiguraci.", B))
st.append(P("3.3 Archiv a draft", H2))
st.append(P("Data jsou organizovana do draftu v archivu ([c]archive_root[/c], vychozi Archive/ v koreni projektu). Jeden draft ma podadresare calibrated/, platesolve/<obs_group>/ (vc. photometry/), reports/ a export/ (plna struktura v Priloze A). Observacni skupina (obs_group, napr. NoFilter_60_2) koduje filtr, expozici a binning. Spojovacim klicem mezi vsemi tabulkami je Gaia DR3 source_id drzeny jako 19misty RETEZEC - nikdy jako float, aby nedoslo ke ztrate presnosti.", B))
st.append(fn("param_resolver.py (provenience), database.py (LOCATION/TELESCOPE/EQUIPMENTS), pipeline.py: observation_group_key_from_metadata, _summarize_lights_binning_from_headers (preflight diagnostika binningu z hlavicek)"))
st.append(par("archive_root, calibration_library_root, database_path, gaia_db_path, vsx_local_db_path, exoplanet_local_db_path, blind_index_fine_path / blind_index_wide_path (cesty; per-machine)"))

# ============================================================ CH 4
st.append(P("4. Kalibrace snimku", H1))
st.append(P("Cil: odstranit otisk kamery (temny proud, nerovnomerne osvetleni pole) a NEZNICIT pritom data. Kalibrace je prvni misto, kde muze cela noc potichu zhavarovat - proto ji VYVAR obklopuje diagnostickymi branami (4.4).", B))

st.append(P("[b]Proc vubec kalibrovat:[/b] surovy snimek = scena x odezva optiky/cipu + temny proud + offset + sum. Fotometrie potrebuje SCENU: dark odstranuje aditivni termalni signal (a s nim horke pixely), flat multiplikativni otisk cesty (vinetace az desitky procent na okraji, prachove 'donuty' jednotky procent, pixel-to-pixel citlivost pod procento). Bez kalibrace by hvezda putujici driftem po poli menila jasnost s mistem - falesna 'variabilita' o radech vetsi nez hledane signaly.", B))
st.append(P("4.1 Master dark a master flat", H2))

st.append(P("[b]Fyzika v pozadi (proc to funguje):[/b] temny proud je termalni generace elektronu - roste ~exponencialne s teplotou (zhruba zdvojnasobeni na kazdych 6-7 C u typickych CMOS), proto se dark paruje na teplotu s tolerance +-0.5 C a expozici. Flat popisuje multiplikativni odezvu cesty (vinetace, prach, pixel-to-pixel citlivost) - proto se jim DELI, zatimco dark se ODECITA (aditivni signal). Poradi je zavazne: (RAW - dark) / flat_norm; prohozeni by flatem delilo i temny proud.", B))
st.append(P("[b]Pocetni priklad:[/b] RAW pixel 12 400 ADU, master dark tehoz mista 350 ADU, master flat normalizovany 0.94 (vinetovany roh): kalibrovana hodnota = (12400 - 350) / 0.94 = 12 819 ADU. Roh se 'dorovnal' na uroven stredu - fotometrie pak nevidi vinetaci jako falesny gradient jasnosti pres pole.", B))
st.append(P("[b]Master dark[/b] vznika medianovou kombinaci sady darku o stejne expozici a teplote cipu; median potlacuje nahodne artefakty (kosmicke zareni) lepe nez prumer. U CMOS senzoru se NEPOUZIVAJI bias ramce - senzor aktivne drzi offset a kazda dalsi aditivni operace jen zvysuje sum o faktor sqrt(2). Plati jednoduche: Kalibrovany = (RAW - MasterDark) / MasterFlat_norm. [b]Master flat[/b] se kombinuje medianem s vyrovnanim strednich hodnot jednotlivych ramcu (sky flaty behem soumraku rychle meni jas); normalize_flat_master resi i flaty ulozene nenormalizovane a rozpozna OSC/Bayer vzor. Z master darku se zaroven odvozuje mapa vadnych pixelu (BPM): pixely odchylene o vice nez bpm_dark_mad_sigma robustnich sigma (default 5.0) se mapuji jako defektni.", B))
st.append(P("Mastery se staveji RUCNE v kalibracni knihovne (zadny auto-stack pri importu - DECISIONS 2026-07-07); knihovna ma prednost pred syrovymi kalibracnimi snimky v sessionu. Registrace master darku VYZADUJE konecnou CCD_TEMP v hlavicce (databaze jinak registraci odmitne).", B))

st.append(box("Dobra praxe pri stavbe masteru (doporuceni)", [
 "Darky: 20-30 ramcu na kombinaci (expozice x teplota); vice uz zlepsuje malo (chyba medianu klesa ~1/sqrt(N)). Fotit se zakrytym dalekohledem, stejna expozice a chlazeni jako lights.",
 "Flaty: 20+ ramcu, cilova uroven ~1/3 az 1/2 plne studny (linearita!); sky flaty za soumraku ci flatfield panel. Po kazde manipulaci s optikou/kamerou (prach se pohnul) fotit nove.",
 "Registrujte do knihovny prubezne - parovani si vzdy bere teplotne nejblizsi validni master; stare mastery nechavejte (historie pro reanalyzu starych noci)."]))
st.append(P("4.2 Kalibracni knihovna a vyber masteru", H2))
st.append(P("Knihovna (Calibration Library) uklada mastery organizovane dle kamery, binningu, expozice, teploty a filtru. Importer pro kazdou noc vybira nejlepsi master (find_best_calibration_library_path): dark s |dT| <= calibration_master_ccd_temp_tolerance_c (default 0.5 C) a odpovidajici expozici, flat dle filtru. Stari masteru se hlida: masterdark_validity_days (default 90) a masterflat_validity_days (default 200) - prosly master vyvola varovani, ze je cas nafotit nove kalibracni snimky.", B))
st.append(P("4.3 Prevzorkovani masteru na jiny binning (dark_resample)", H2))
st.append(P("Uzivatele mohou stavet mastery v libovolnem binningu (calibration_library_native_binning, default 1). Kdyz svetelne snimky prijdou v jinem binningu, master se prevzorkuje se ZACHOVANIM TOKU: blokovy soucet pri zmensovani (bin1 -> bin2 scita 2x2 bloky, protoze binovany pixel fyzicky nasbiral soucet nabooju), rovnomerne rozprostreni pri zvetsovani. Operace se zapisuje do provenance (priznak dark_resample) - zadna ticha adaptace. Filozofie: [i]adaptovat, ne predpokladat[/i] - pipeline musi konvenci dat diagnostikovat a prizpusobit se, nebo beh bezpecne zastavit.", B))
st.append(P("4.4 CAL-DIAG: radiometricke brany kalibrace (fail-closed)", H2))
st.append(P("Trida chyb 'spatny master' (zamena konvence stacku SUM vs MEAN, master z jine kamery, spatna expozice) drive dokazala POTICHU znehodnotit celou noc - snimky vypadaly normalne, ale fotometrie byla posunuta. CAL-DIAG (cal_diag.py, spec VYVAR_CAL_DIAG_SPEC v1.1, master prepinac cal_diag_gate_enabled=true) zavadi dve kontroly:", B))
st.append(P("[b](a) Krizova kontrola urovni PRED odectem:[/b] median darku se porovna s medianem svetelneho snimku (relativni tolerance cal_diag_rel_tol, default 2 %). Kdyz dark vychazi ~N-krat vyssi, nez odpovida expozici (typicky priznak SUM stacku tam, kde se ceka MEAN), diagnostika konvenci rozpozna a bud ji bezpecne prepocita (cal_diag_autocorrect_enabled=true; korekce se zapise do provenance), nebo beh zastavi (fail-closed).", B))
st.append(P("[b](b) Sanity check PO odectu:[/b] median oblohy po odectu darku musi zustat fyzikalne smysluplny; odchylka nad cal_diag_hard_sigma (default 5.0 sigma) beh zastavi. K tomu varovani na prilis jasny master ci snimek nad cal_diag_sat_warn_frac (default 0.9) saturacniho limitu.", B))

st.append(box("Jak se pozna spatny master (a proc na tom zalezi)", [
 "SUM misto MEAN stacku: dark vyjde N-krat vyssi (N = pocet ramcu) -> po odectu zaporna obloha nebo nesmyslne nizky median -> CAL-DIAG (a) chyti pomer urovni, (b) chyti nesmyslnou oblohu; autocorrect umi konvenci prepocitat.",
 "Dark z jine expozice/teploty: rezidualni temny proud vytvori falesny 'signal' zavisly na teple noci - fotometrie se posune systematicky a NEODHALITELNE pouhym pohledem. Proto fail-closed: radeji noc zastavit nez tise znehodnotit.",
 "Stary flat (prach se pohnul): 'donut' artefakty se objevi/zmizi -> lokalni fotometricke chyby hvezd, ktere na artefakt padnou. Hlida se stari flatu (200 dnu) a QC per snimek."]))
st.append(P("4.5 Sky-surface preprocess (mono anchor draft_435; OSC post-extraction only)", H2))
st.append(P("Po kalibraci se na cely snimek fituje robustni polynomialni plocha radu 2 (6 clenu; preprocess_sky_surface_order=2) a odecita se CELA fitovana plocha vcetne konstantniho clenu (pedestal konvence dle T3 rozhodnuti; median snimku se posouva o fitovany pedestal, na referencnim snimku ~ -96 ADU). Odstranuje velkoplosny gradient (Mesic nizko, svitani) i zbytkove zakriveni po flatu; tvar i uroven pozadi jsou pak konzistentni pro DAO detekci a Labbe empiricke pozadi. Vyssi rad je ZAMERNE zakazan: polynom radu 3+ by zacal pozirat plosne objekty (mlhoviny) a lokalni struktury pozadi. Krok je soucasti anchoru draft_435 pro mono cestu. [b]OSC vetev (2026-07):[/b] pri nastavenem EQUIPMENTS.BAYERMASK se sky-surface na mozaice NIKDY nespousti (checkerboard by fit pokazil); kalibrace probiha na CFA, pak extrakce kanalu oneRGGB/R/G/B (plane split, prumer, bez interpolace) a sky-surface az na extrahovanych kanalech per obs-group. Viz osc_extract.py a OSC-01. [b]OSC-2 (2026-07):[/b] plate-solve jednou na oneRGGB MASTERSTAR; WCS + registracni transformace (osc_registration_handoff.json) se prenasi na R/G/B; QC verdict jednotny (qc_source=oneRGGB); kazdy kanal ma vlastni DAO katalog a Phase 0/1/2A; OSC-02 hlida shodnou sadu snimku.", B))
st.append(fn("calibration.py: get_processed_master, normalize_flat_master, resample_master_to_light_binning, infer_spatial_block_factor / infer_spatial_upscale_factor, resolve_master_age, get_master_age_days; cal_diag.py: CalDiagSession a brany; importer.py: find_best_calibration_library_path; database.py: registrace masteru (povinna CCD_TEMP)"))
st.append(par("cal_diag_gate_enabled=true, cal_diag_autocorrect_enabled=true, cal_diag_rel_tol=0.02, cal_diag_hard_sigma=5.0, cal_diag_sat_warn_frac=0.9, calibration_master_ccd_temp_tolerance_c=0.5, masterdark_validity_days=90, masterflat_validity_days=200, calibration_library_native_binning=1, bpm_dark_mad_sigma=5.0, dao_qc_in_calibrate=true, preprocess_sky_surface_order=2, osc_channel_binning=2, EQUIPMENTS.BAYERMASK (OSC authority)"))

# ============================================================ CH 5
st.append(P("5. Kontrola kvality snimku (QC)", H1))
st.append(P("Hned po kalibraci bezi per-frame QC (qc_after_calibrate_enabled=true): na kazdem snimku se detekuji hvezdy (DAO, prah qc_dao_detection_sigma=5.0) a meri se FWHM, HFR (half-flux radius), elongace, pocet hvezd a RMS pozadi. Metriky se zapisuji do hlavicek (vy_* klice) a slouzi jak branam, tak pozdejsim krokum (vyber referencniho ramce, vyber MASTERSTARu).", B))

st.append(tab([
 ["Metrika","Co meri a co diagnostikuje"],
 ["FWHM [px]","sirka hvezdneho profilu v polovine maxima; seeing + zaostreni. Skok FWHM = rozostreni/teplotni drift ohniska; postupny rust = zhorsujici se seeing"],
 ["HFR [px]","polomer, v nemz je polovina toku; alternativni ostrost zname z capture softwaru (NINA/SGP) - srovnatelnost s tim, co uzivatel videl v noci"],
 ["Elongace","pomer os elipsy profilu; > 1.8 znaci trailing (vedeni montaze, vitr, flexe)"],
 ["Pocet hvezd","nahly pokles = mrak/rosa; < 10 = snimek nepouzitelny pro parovani"],
 ["RMS pozadi","sum oblohy; roste s Mesicem, svitanim, oparem - koreluje s lunarnim kontextem (13.6)"],
], [30,138]))
st.append(P("5.1 Automaticky FWHM limit", H2))
st.append(P("Misto pevneho prahu ostrosti se limit odvozuje z nocnich statistik (auto_fwhm_enabled=true): limit = median_FWHM_noci x k, kde k = auto_fwhm_k_factor (default 1.5) sevreny do [auto_fwhm_k_min=1.0, auto_fwhm_k_max=4.0]. Dalsi meze: qc_max_hfr=5.0, qc_min_stars=10, volitelny strop qc_max_background_rms. Elongace v QC tabulce je diagnostika (pomer os); samostatny persistovany strop elongace v config.json neni.", B))
st.append(P("5.2 Volitelne brany vyrazovani snimku (default OFF)", H2))
st.append(P("Dve brany umi snimky aktivne vyradit, obe jsou vychozi VYPNUTE a maji pojistky proti zahozeni noci: [b]frame_quality_gate[/b] (frame_quality_gate_enabled=false) vyrazuje snimky vyrazne horsi nez typicke seeing noci (pomer frame_quality_ratio_k=5.0, minimum ponechanych frame_quality_min_keep_frames=10); [b]frame_align_residual_gate[/b] (frame_align_residual_gate_enabled=false) vyrazuje snimky s vysokymi rezidui zarovnani - zachyti mrak ci vitr v casti pole, i kdyz snimek jinak vypada ostre (max. podil vyrazenych frame_align_residual_max_frac=0.25, minimum ponechanych 10). Rozhodnuti drzet brany default-OFF je zaznam v DECISIONS (2026-06-18: brana je rig-agnosticka a pricinne spravna, ale zapina se az po overeni na konkretnim rigu); pro promenlivou pruhlednost byla adoptovana transparency frame-quality gate, rovnez default-OFF (DECISIONS 2026-06-17).", B))
st.append(fn("pipeline.py: QC vetev po kalibraci (vy_* hlavicky), _get_vy_qc_status; photometry_core.py: compute_auto_fwhm_limit, _frame_quality_gate_select, _frame_align_residual_gate_select, _compute_frame_align_residuals"))
st.append(par("qc_after_calibrate_enabled=true, qc_dao_detection_sigma=5.0, auto_fwhm_enabled=true (k=1.5, clamp 1.0..4.0), qc_max_hfr=5.0, qc_min_stars=10, qc_max_background_rms=null, frame_quality_gate_enabled=false (ratio_k=5.0, min_keep=10), frame_align_residual_gate_enabled=false (max_frac=0.25, min_keep=10)"))

# ============================================================ CH 6
st.append(P("6. MASTERSTAR a astrometrie (plate solving)", H1))
st.append(P("MASTERSTAR je astrometricka 'pravda' cele rady: referencni obraz s vyresenym WCS a hlubokym katalogem hvezd pole. Vybira se z nejlepsich snimku rady (masterstar_best_of_n=10 kandidatu dle FWHM a poctu kvalitnich detekci); pro pripad selhani primarniho reseni existuje zachranna cesta pres stack sourozencu (masterstar_sibling_recovery_enabled=true: stack masterstar_sibling_stack_n=10 snimku, prijeti vyzaduje >= masterstar_sibling_min_matched=40 shod, pokryti >= 3 kvadrantu a RMS <= 2.0 px).", B))

st.append(P("[b]Co je WCS:[/b] World Coordinate System - matematicke zobrazeni pixel (x, y) <-> obloha (RA, Dec): referencni bod (CRVAL/CRPIX), linearni cast (CD matice: meritko + rotace + zrcadleni) a projekce (TAN = gnomonicka) + polynomialni distorze (SIP). 'Vyresit snimek' znamena najit tyto koeficienty tak, aby detekovane hvezdy padly na katalogove pozice. Jakmile WCS existuje, kazdy pixel ma souradnice a kazda hvezda katalogovou identitu - na tom stoji vse dalsi.", B))
st.append(P("6.1 Hluboka detekce a katalog pole", H2))
st.append(P("Na MASTERSTARu bezi dvoupruchodova DAO detekce: bezny prah (masterstar_dao_threshold_sigma=3.8, prekalibrovano proti gradient-imunnemu sigma_pp z adjacent-difference MAD - reprodukuje stejny ~175 ADU prah, ktery drive dal 2.1 proti sigma_clipped_stats; hloubka detekce se nezmenila, zmenil se odhad sumu) a hlubsi druhy pruchod na stacku zarovnane serie s nizsim sumem - slaby konec katalogu se tak prohloubi bez zaplaveni falesnymi detekcemi. Strop poctu detekci je adaptivni k hustote pole (masterstar_detection_cap_adaptive=true, k=0.08, sevreno do 250..800). Detekce se krizove ztotoznuje s Gaia DR3 pri exportu katalogu, s VSX (zname promenne) a s lokalnim exoplanetovym katalogem (exoplanet_match_max_sep_arcsec=3.0). Kazda hvezda dostava stabilni identitu (Gaia source_id jako string), barvu BP-RP a vlajky: znama promenna, NSS (non-single star), exoplaneta, zona saturace/sumu (linear / noisy1-3 / saturated s priznaky is_saturated, is_noisy, is_usable dle peak_max_adu). Vystup: masterstars_full_match.csv. Gaia pozice se pred matchem posouvaji o vlastni pohyb k epose pozorovani (_apply_proper_motion z obs roku hlavicky).", B))
st.append(P("6.2 Hintovana cesta WCS", H2))
st.append(P("Kdyz FITS hlavicka nese pouzitelne zamereni (pointing_hint_from_header; RA/Dec z montaze), resi se WCS lokalne: detekce jasnych hvezd, parovani na Gaia vyrez, robustni fit TAN + SIP distorze (rad masterstar_platesolve_sip_min_order=3 az max_order=4, adaptivni ridge regularizace dle poctu shod). Prijeti reseni je statisticke (MASTERSTAR odds-ratio test; jedina politika, CONSOLIDATE-01D) s tvrdymi zavorami: obnoveni >= masterstar_catalog_recovery_min=0.65 podilu katalogovych hvezd, absolutni dno masterstar_min_matched_floor=40 shod, RMS stredu <= masterstar_centre_rms_max_px=1.2, benigni pomer distorze okraj/stred <= 3.2.", B))
st.append(P("6.3 Slepa cesta (blind solve)", H2))
st.append(P("Bez pouzitelneho zamereni (po Meridian Flipu, rucni presun) nastupuje slepe reseni nezavisle na rotaci: [b](1)[/b] vyber obrazovych hvezd rozprostreny po snimku (blind_img_select_mode='per_cell', rozpocet blind_img_star_budget=80); [b](2)[/b] geometricke invarianty - trojuhelniky/kvady pres 8 nejblizsich sousedu, porovnavane proti predpocitanemu indexu (fine/wide dle zorneho pole, blind_index_select_mode='auto'; stavba indexu offline: GAIA_DR3/build_blind_index.py); [b](3)[/b] kandidatni shody se sdruzuji hlasovanim (DBSCAN clustering korespondenci) a kazdy cluster dava WCS seed; [b](4)[/b] robustni fit RANSAC (_fit_cluster_ransac_wcs, _ransac_fit_wcs_tan). Znalost meritka rigu slouzi jako prior k rychlemu zavrzeni nesmyslnych kandidatu (blind_use_rig_prior=true; DECISIONS 2026-06-04).", B))
st.append(P("[b]Overeni proti Gaia (bezpecnostni sit):[/b] KAZDY kandidat musi projit verifikaci (blind_verify_enabled=true): tolerance shody blind_verify_match_tol_px=2.5, minimalni podil blind_verify_min_fraction=0.15, absolutni minimum blind_verify_min_matches=12; plnou verifikaci dostava blind_verify_top_n=15 nejlepsich kandidatu, predcasne prijeti pri 30 shodach (blind_verify_early_accept). Empiricky vysledek: verify_mag_limit=14 je stejne spolehlive jako 16 a o ~28 % rychlejsi. Zapis WCS je fail-closed: bez platneho WCS se Faze 2A nespusti. F-428 pridal branu invertibility WCS (wcs_invertibility.py): reseni, jehoz transformace neni spolehlive obousmerne, se odmitne - chrani konzistenci pixel<->obloha souradnic v celem downstream.", B))


st.append(P("[b]Odds test prijeti (jedina MASTERSTAR politika):[/b] misto pevneho prahu 'pocet shod' se porovnava pravdepodobnost pozorovaneho poctu shod za hypotezy spravneho reseni proti hypoteze nahodnych koincidenci (pomer sanci). Vyhoda: automaticky se skaluje s hustotou pole - 40 shod na ridkem poli je silny dukaz, na poli s 2000 hvezdami slabsi; pevny prah by jedno z toho posuzoval spatne.", B))
st.append(P("6.4 Jak funguji geometricke invarianty (pro pochopeni)", H2))
st.append(P("Trojuhelnik tri hvezd ma pomery stran nezavisle na posunu, rotaci i meritku - to je invariant. Slepy solver proto neporovnava POZICE (nezna je), ale TVARY: pomery stran trojuhelniku z obrazku proti predpocitanym pomerum trojuhelniku z Gaia (index fine/wide). Kazda shoda tvaru je HYPOTEZA korespondence tri hvezd; jednotliva shoda muze byt nahodna, proto se hypotezy sdruzuji hlasovanim (DBSCAN nad parametry transformace) - skutecne reseni se projevi jako husty shluk konzistentnich hypotez. Z nejlepsiho shluku se RANSACem (nahodne minimalni vzorky, pocitani inlieru, iterace) vytahne robustni TAN WCS odolne vuci falesnym param. Nakonec statisticka verifikace proti Gaia (6.3) rozhodne, zda reseni prijmout.", B))
st.append(P("[b]SIP distorze:[/b] realna optika neni idealni gnomonicka projekce - SIP (Simple Imaging Polynomial) pridava k TAN polynomialni korekce u,v -> x,y radu 3-4. Rad se voli adaptivne (try-orders) a fit ma ridge regularizaci rostouci pri malem poctu shod - vyssi rad s malo hvezdami by se 'prohnul' pres sum. Benigni pomer distorze okraj/stred <= 3.2 je sanity brana na nefyzikalni reseni. [b]Parita/zrcadleni:[/b] nektera FITS maji prohozene poradi radku ci zrcadleny senzor; solver testuje obe parity (_mirror_detections_xy, _fits_roworder_yflip_applied) a parita reseni se zapisuje do MASTERSTAR hlavicky.", B))
st.append(fn("pipeline.py: build_masterstar_from_detrended; vyvar_platesolver.py: pointing_hint_from_header, resolve_pointing_for_vyvar, _fit_sip_on_matches(_masterstar_try_orders), _ransac_fit_wcs_tan, _verify_blind_candidates, _pool_cluster_correspondences (DBSCAN), _apply_proper_motion; vyvar_blind_solver.py + vyvar_blind_series.py; wcs_invertibility.py; GAIA_DR3/build_blind_index.py (offline index)"))
st.append(par("masterstar_dao_threshold_sigma=3.8, masterstar_best_of_n=10, MASTERSTAR odds gate (hard-wired), masterstar_catalog_recovery_min=0.65, masterstar_min_matched_floor=40, masterstar_centre_rms_max_px=1.2, masterstar_platesolve_sip_min/max_order=3/4, sibling recovery (stack 10, matched>=40, quadrants>=3, rms<=2.0), detection cap adaptive (k=0.08, 250..800), blind_* (budget 80, verify frac 0.15, matches 12, tol 2.5 px, top 15, early 30), verify_mag_limit=14.0, exoplanet_match_max_sep_arcsec=3.0"))

# ============================================================ CH 7
st.append(P("7. Zarovnani snimku (alignment)", H1))
st.append(P("Serie se sesazuje na spolecnou pixelovou mriz, aby taz hvezda lezela na stejnych pixelech napric noci. Metoda: astroalign-style hledani podobnych trojuhelniku mezi kontrolnimi body referencniho a zarovnavaneho snimku, z nich afinni transformace (posun + rotace + meritko; 6 stupnu volnosti). Zvlada drift pole i rotaci vcetne ~180 stupnu po Meridian Flipu. Kontrolni body dava stredne prisna DAO detekce (alignment_detection_sigma=5.0), matcher uvazuje max alignment_max_stars=160 hvezd a pouzije nejvyse alignment_max_control_points=80 bodu na snimek.", B))

st.append(P("Prevzorkovani pri aplikaci transformace je interpolacni (bilinearni/spline) a NENI dokonale flux-conserving na urovni pixelu - vznika drobna kovariance sousednich pixelu. To je presne duvod, proc chybovy model NEMERI sum pozadi teoreticky, ale empiricky prazdnymi aperturami na JIZ ZAROVNANEM snimku (kap. 11.4, vrstva 2): zmerena sigma_bkg korelovany sum resamplingu prirozene obsahuje. Souhra techto dvou rozhodnuti je zamerna.", B))
st.append(P("Referencni ramec se vybira podle poctu a kvality detekovanych hvezd (idealne dobre zaostreny snimek z prostredka rady). Geometrickym cilem zarovnani je referencni SVETELNY snimek - WCS z MASTERSTARu slouzi jako astrometricky zdroj, ale parovani geometrie je hvezdne. Kvalita se meri rezidui transformace per snimek; volitelna brana (kap. 5.2) umi snimky s vysokymi rezidui vyradit. Typicky vysledek na wide rigu: median driftu centroidu pres noc ~0.4 px (zmereno na 127 snimcich). Pri dostatku pameti se zarovnane snimky drzi v RAM (handoff do per-frame kroku bez opakovaneho cteni disku); zapis zarovnanych FITS na disk zustava checkpointem pro obnovu behu a paralelismus.", B))
st.append(fn("vyvar_alignment_frame.py: _alignment_detect_xy, _alignment_run_astroalign_points, _alignment_compute_one_frame, _alignment_load_masterstar_catalog_points_for_frame, multiprocessing vetev _astrometry_align_mp_*; photometry_core.py: _compute_frame_align_residuals"))
st.append(par("alignment_detection_sigma=5.0, alignment_max_stars=160, alignment_max_control_points=80"))

# ============================================================ CH 8
st.append(P("8. Per-frame katalogy a detekce hvezd", H1))
st.append(P("8.1 Proc plna detekce na kazdem snimku", H2))
st.append(P("Na KAZDEM zarovnanem snimku bezi plna DAO detekce celeho pole (photutils DAOStarFinder; prahy per ucel: QC stabilita vs masterstar uplnost) a vysledek se paruje na pevny seznam masterstars_full_match.csv (nearest-neighbour v rovine oblohy pres WCS). To je vedome architektonicke rozhodnuti (DECISIONS): plna detekce per frame 'kupuje' QC, ktere by proste odecitani na pevnych pixelech ztratilo - lokalni centroidy pohlti drift a zbytky WCS chyb, tvarove filtry odmitnou kosmiky/horke pixely/sloupce, pocet shod proti katalogu okamzite prozradi spatne reseni a nove zdroje jsou detekovatelne. Cena (CPU) je prijatelna diky nocnimu davkovemu modelu (kap. 1, zasada 4).", B))
st.append(P("8.2 Co se pocita na snimek", H2))
st.append(P("Per-frame katalog (proc_*.csv) obsahuje: subpixelove centroidy (DAO, intenzitou vazeny prumer) a prirazeni na master pres WCS; flux a dao_flux; peak_max_adu (saturace); FWHM a elongaci; casy jd/hjd/bjd (heliocentricka/barycentricka korekce dle Eastman et al. 2010, stanoviste z resolveru provenience); a per-epoch trust vlajku [b]catalog_match_mode[/b] - jak dobre snimek sedi na katalog (vstup trust gate, kap. 12). Staticke sloupce (Gaia ID, katalogove magnitudy, BP-RP) se prebiraji z master radku; flux/peak/saturace jsou per-frame, protoze na jine expozici a case muze byt hvezda jina.", B))

st.append(tab([
 ["Sloupce proc_*.csv (vyber)","Vyznam"],
 ["catalog_id, gaia_source_id","identita hvezdy (string; DET_* pro cizi detekce bez Gaia)"],
 ["x, y, ra, dec","per-frame centroid a WCS souradnice"],
 ["flux, dao_flux, flux_err","aperturni tok, detekcni tok, chyba (empiricky model 11.4)"],
 ["peak_max_adu, is_saturated / noisy / usable","saturacni diagnostika per snimek"],
 ["fwhm, elongation","tvar profilu per snimek"],
 ["jd, hjd, bjd, airmass","casy a vzdusna hmota (Kasten & Young 1989)"],
 ["catalog_match_mode","per-epoch trust vlajka shody snimku s katalogem"],
 ["psf_flux, psf_flux_err, psf_chi2, psf_quality_fallback, psf_ac_factor, psf_ac_n_used, psf_ac_applied","PSF sloupce (plni se pri zapnute vetvi; PROC_STORE_COLS guard)"],
], [58,110]))
st.append(P("[b]Casove systemy:[/b] JD je 'cas hodinek observatore'; HJD koriguje na pohyb Zeme vuci Slunci (az +-8.3 min svetelneho casu pres rok!) a BJD_TDB na barycentrum Slunecni soustavy (dalsi ~+-4 s + relativisticka skala). Pro periody kratsi nez hodiny je HJD/BJD korekce ROZDIL mezi pouzitelnou a rozbitou fazovou krivkou. VYVAR pocita vse tri (Eastman et al. 2010) per snimek a per cil (souradnice cile vstupuji do korekce).", B))
st.append(P("8.3 Uplnost slabeho konce: DAO-RECONCILE", H2))

st.append(P("Uplnost detekce se nehlida sigmou 'od oka', ale reconciliaci vuci Gaia (dao_reconcile.py): referencni populace pole se cte primym dotazem do lokalni Gaia DB pres bounding box MASTERSTAR WCS (bez stropu radku), krivka uplnosti se fituje error-function modelem dle Fleming et al. (1995) a odvozuji se G_lim_50 / G_lim_90 (magnitudy 50% a 90% uplnosti) plus metrika miss@G90 (podil Gaia hvezd jasnejsich nez G_lim_90, ktere DAO nenaslo). Blend radius 1.5 x FWHM px sdili s crowding_index. Workstream DAO-RECONCILE byl uzavren 2026-07-09: uplnost 89.7-98.3 % napric rigy, G_lim charakterizovano (wide ~15.0, Newton V ~16.7, B/R censored >= 17.5) a completeness_50 + missed_below_g90 bezi dal jako trvale health signaly QA dashboardu.", B))
st.append(fn("photutils.detection.DAOStarFinder (49 vyskytu napric QC/alignment/masterstar/SIPS presety); pipeline.py: per-frame katalogova cesta, _per_frame_noise_error_map; photometry_core.py: measure_fwhm_from_masterstar; dao_reconcile.py (Fleming 1995, G_lim_50/90, miss@G90); catalog_match_trust.py (per-epoch vlajka); time_utils.py (jd/hjd/bjd)"))
st.append(par("sips_dao_fwhm_px=2.5, sips_dao_threshold_sigma=3.5 (SIPS-styl preset), qc_dao_detection_sigma=5.0, masterstar_dao_threshold_sigma=3.8"))

# ============================================================ CH 9
st.append(P("9. Faze 0 - priprava cilu (active targets)", H1))
st.append(P("Cile mereni vznikaji tremi cestami: [b](1)[/b] automaticky ze VSX - detection-limited: VSX target adopted when it has a DAO detection with Gaia cross-match on MASTERSTAR; nevyresene VSX hvezdy se dohledavaji primym Gaia DR3 lookupem (VSX -> Gaia fallback), takze podil neprirazenych se blizi nule; [b](2)[/b] automaticky z lokalniho exoplanetoveho katalogu (hostitele tranzitu; match 3.0 arcsec); [b](3)[/b] rucne zadane cile uzivatele v UI. Kazdy cil se pinuje na Gaia source_id - identita je katalogova, ne pixelova, takze prezije zmenu pole i sestavy.", B))
st.append(P("Volitelny filtr vsx_out_of_scope_types=[] (default prazdny = vypnuto): VSX auto-vybrane cile, jejichz typovy retezec po tokenizaci na oddelovacich | / + a mezerach (tokeny uppercase, koncove ':' VSX nejistoty se orezava) ma ALESPON jeden token shodny s konfiguraci, zustanou v active_targets.csv se skip_photometry=True a skip_reason=vsx_type_out_of_scope (mask-first; UI badge). Podretezecovy match se NIKDY nepouziva. Rucne pridane cile se NEfiltruji. Out-of-scope hvezdy zustavaji znamymi promennymi a dale se cisti z komparacniho poolu.", B))
st.append(P("Predfiltr saturacnich zon: hvezdy v zone linear se ponechavaji, saturated se vylucuji, mezistavy dostanou badge. Zony z MASTERSTARu jsou predfiltr z jedne referencni expozice; KRITICKA rozhodnuti o vyrazeni z finalni fotometrie se opiraji o per-frame saturacni priznaky (kap. 8.2). Vyber cilu je prostorove-prvni (frame bbox) a zname promenne se zaroven PROPLACHUJI z komparacniho poolu (DECISIONS 2026-06-17) - promenna hvezda nikdy nesmi slouzit jako srovnavaci. Vystup: active_targets.csv s priznaky zone_flag a skip_photometry (v UI badges).", B))

st.append(tab([
 ["Zona (z MASTERSTARu)","Vyznam a chovani"],
 ["linear","peak hluboko pod saturaci - plnohodnotne mereni"],
 ["noisy1-3","slaby signal (odstupnovane) - meri se, badge v UI, vstup do trustu"],
 ["likely_saturated / saturated","peak u/za limitem - cil se z fotometrie vyrazuje (skip_photometry), komp nikdy"],
], [44,124]))
st.append(P("Pripravena per-frame varianta (per_frame_saturation_enabled, default OFF): misto celohvezdneho vyrazeni rozhoduje podil cistych snimku (prah per_frame_sat_min_clean_frac); validace ceka na dataset se saturovanymi hvezdami.", B))
st.append(P("Zivotni cyklus cile: navrzen (VSX/exoplanet/rucne) -> obohacen (Gaia ID, BP-RP, zone) -> aktivni (active_targets.csv) -> zmeren (LC CSV) -> znamkovan (trust) -> reportovan/exportovan. V kazdem kroku je stav viditelny v UI (badges) a v CSV - zadny cil nemizi tise; i skip_photometry ma zapsany duvod.", B))
st.append(fn("photometry_core.py: select_active_targets, _active_target_zone_flag, _enrich_active_targets_bp_rp, stamp_vsx_known_variable_on_masterstars, resolve_variable_targets_csv; VSX/vsx_make.py, exoplanets/exoplanet_make.py (offline stavba katalogu)"))
st.append(par("exoplanet_match_max_sep_arcsec=3.0"))

# ============================================================ CH 10
st.append(P("10. Faze 1 - vyber srovnavacich hvezd", H1))
st.append(P("Pro kazdy cil se stavi soubor srovnavacich hvezd (kompu). Je to jeden z nejpropracovanejsich kroku pipeline: kaskada TVRDYCH filtru (vyrad/ponechej), nasledovana RMS stabilitou s barevnymi tiery a finalnim vyberem. Klicova funkce: select_comparison_stars_per_target (volana z run_phase0_and_phase1), vystup comparison_stars_per_target.csv.", B))
st.append(P("10.1 Tvrde filtry kandidatu", H2))
st.append(tab([
 ["Kriterium","Podminka (vychozi hodnoty)"],
 ["Geometrie","vzdalenost od cile >= phase01_comparison_min_dist_arcsec=60 (PSF/blend) a <= max_dist_deg=1.5; orez vnitrniho okraje cipu phase01_chip_interior_margin_px=50 px; max_dist se odvozuje i z FOV (_compute_fov_max_dist)"],
 ["Zona / pouzitelnost","is_usable=true; nikoli is_saturated / is_noisy / likely_saturated; nikoli samotny cil; nikoli znama promenna (VSX purge, kap. 9)"],
 ["Magnituda","|Delta mag| <= phase01_comparison_max_mag_diff=1.5 od cile; absolutni strop 3.0, ktery zadna adaptace neprekroci; pro jasne cile (mag < 12.75) plati bright floor 1.5"],
 ["Barva (primarni!)","|Delta(BP-RP)| <= comp_max_delta_bprp=0.79; BP-RP je first-order kriterium (a jediny color display basis): u NoFilter/sirokopasmovych dat neexistuje filtr, ktery by barevny clen vyrusil, takze barevna shoda kompu je HLAVNI obrana proti extinkci 2. radu"],
 ["Gaia priznaky","vylouceni NSS dvojhvezd (exclude_gaia_nss=true) a rozsahlych objektu/galaxii (exclude_gaia_extobj=true)"],
 ["Izolace","zadny jasny soused do phase01_comparison_isolation_radius_px=25 px; filtr je 'bezpecny' - neaplikuje se, pokud by snizil pocet pod n_comp_min"],
 ["Tvar","FWHM <= phase01_comparison_max_fwhm_factor=1.5 x median snimku; psf_chi2 <= 50 (tvarova prijatelnost)"],
 ["Pokryti","pritomnost na >= phase01_comparison_min_frames_frac=0.2 podilu snimku; saturace na > 10 % snimku vyrazuje"],
], [34,134]))
st.append(P("10.2 RMS stabilita, barevne tiery a driftovy test", H2))
st.append(P("Z casove rady fluxu kandidata (phase01_flux_col='dao_flux') se pocita nocni RMS; tvrdy limit phase01_comparison_max_comp_rms=0.1 mag. Kandidati se radi do barevnych tieru dle shody BP-RP s cilem - comp_color_tiers: |Delta BP-RP| <= 0.15 (vaha 1.0), <= 0.30 (0.85), <= 0.55 (0.50), <= 1.10 (0.25); kandidat bez zname barvy padne do nejnizsiho tieru, nevylucuje se. Iterativni odstranovani RMS outlieru: robustni prah median(rms) + phase01_comparison_rms_outlier_sigma=3.0 x MAD/0.6745, max ~10 iteraci, nikdy pod n_comp_min. Driftovy test: linearne trendujici komp (|slope| > comp_max_slope_mmag_hr=5.0 mmag/h) se vyradi, ale jen pri statisticke vyznamnosti comp_slope_significance_k=3.0 - a slope se meri na common-mode-ocistenem rezidualu (DECISIONS 2026-06-11), aby spolecny atmosfericky trend nevyrazoval nevinne kompy.", B))
st.append(P("10.3 Skore, finalni vyber a poradi kriterii", H2))
st.append(P("Poradi dulezitosti: barva (tier) -> stabilita (RMS) -> vzdalenost (jen brana, NE radici kriterium). Dulezity negativni vysledek (DECISIONS): proximitni tie-break byl vyzkousen a REVERTOVAN - zmenil sadu u 143/143 cilu a obetoval stabilitu za blizkost; Broegovy vahy (w ~ 1/sigma^2) jsou nezavisle na poradi kandidatu, takze proximita patri jako brana. Cilovy pocet: phase01_comparison_n_comp_min=3 az n_comp_max=8 - literatura ukazuje, ze zisk ze scintilace saturuje kolem 6-8 kompu (Broeg 2005, Osborn 2015 aj.). Hvezdy s Gaia ID maji prednost pred cistymi detekcemi (DET_*). Sparse fallback (comp_sparse_fallback_enabled=true) povoli v ridkem poli i 1 komp - vysledek pak nese YELLOW (plynula degradace, kap. 12).", B))

st.append(P("Vystupni comparison_stars_per_target.csv nese per komp: catalog_id/gaia_source_id, souradnice, katalogovou mag a BP-RP, tier, vzdalenost od cile, comp_rms (nocni), p2p_rms, pocet snimku, priznaky kvality (good/suspect/excluded + duvod), vahu a roli (comp/check). Schema je vynucovane (_require_comparison_stars_per_target_schema) - downstream se na sloupce smi spolehnout.", B))
st.append(P("10.4 Globalni comp pool a RMS mapa (default ON)", H2))
st.append(P("Filtry a RMS se v praxi nepocitaji opakovane pro kazdy cil: produkce vzdy stavi JEDEN sdileny pool pole (COMP-POOL-01; build_global_comp_pool + compute_global_pool_rms_map, comp_pool_rms.py). Pool vznika z masterstars_full_match.csv aplikaci statickych filtru a pro kazdeho kandidata se napric snimky spocita stejny flux->RMS retezec jako ve Fazi 1 (bez per-target ensemble). Per-target vyber pak z poolu jen cerpa a pouziva hotovou RMS mapu. Prinos: konzistence (taz komparace ma stejne RMS u vsech cilu) a usetreny vypocet. Deduplikace dle Gaia klice zabranuje dvojim zaznamum.", B))
st.append(P("10.5 Adaptace na hustotu pole (default ON)", H2))

st.append(P("field_density_adaptive_enabled=true: plosna hustota se odvozuje z poctu Gaia-matchovanych hvezd a rozmeru cipu (_read_field_density_inputs; fallback vy_ndao z hlavicky MASTERSTARu). Prahy: pod field_density_sparse_threshold=300 hvezd je pole 'sparse' (geometricke brany a tolerance se UVOLNI, jinak by nebylo dost kandidatu), nad field_density_dense_threshold=1000 je 'dense' (kriteria se UTAHNOU kvuli blendu). Zdroj techto vstupu je odvozeny (FITS/katalog), ne pevna konfigurace - v souladu s principem 'odvoditelnost vs. politika' (kap. 3.2). Pro velmi husta pole existuje spatial-grid vyber (pipeline.py: select_comparison_stars_spatial_grid), ktery kompy rozprostre po poli.", B))

st.append(tab([
 ["Profil","Smer uprav (priklady)"],
 ["sparse (< 300)","uvolnit: vetsi max_dist, tolerantnejsi mag diff (pod absolutnim stropem 3.0), mekci izolace, nizsi n_comp_min; sparse fallback smi vzit i 1 komp (-> YELLOW)"],
 ["normal (300-1000)","zakladni hodnoty z config.json beze zmen"],
 ["dense (> 1000)","utahnout: tesnejsi barva a mag diff, prisnejsi izolace a tvar, uzsi annulus - obrana proti blendu; kandidatu je dost"],
], [30,138]))
st.append(box("Priklad: jak kaskadou projde konkretni cil (ilustracni cisla)", [
 "Cil V* o G=12.1, BP-RP=0.95 na wide poli s 620 Gaia hvezdami (profil normal). Kandidatu v FOV: 480.",
 "Geometrie (60 arcsec..FOV, okraj 50 px): -70. Zony/pouzitelnost + VSX purge: -55. Magnituda (10.6..13.6): -190. Barva |dBP-RP| <= 0.79: -95. NSS/extobj: -8. Izolace 25 px + tvar: -22. Pokryti >= 20 % snimku: -6. Zbytek: 34 kandidatu.",
 "RMS limit 0.1 mag + iterativni MAD outliery: 27. Driftovy test (5 mmag/h, k=3 na common-mode rezidualu): 26. Tiery: 6x T1, 11x T2, 7x T3, 2x T4.",
 "Vyber: 8 kompu (max) - nejprve dle tieru, uvnitr tieru dle RMS. Vahy w ~ tier_w / sigma^2. Check star = nejstabilnejsi z vyberu, vyrazena z ensemble -> finalni ensemble 7 kompu, n_clean po comp_qa treba 6 -> GREEN."]))
st.append(P("10.6 Iterativni cisteni a PyTICS", H2))
st.append(P("Po prvnim sestaveni krivky bezi iterativni cisteni ensemble (comp_sparse_fallback_enabled; v produkci od 'Brno fixu' 2026-06-14; stary JSON alias comp_iterative_clip_enabled se stale nacte): kompy se preveri proti souboru a odlehle se vyradi ci prevazi (drop-and-reweigh smycka). Volitelne nad tim bezi PyTICS-styl iterativni interkalibrace vah (pytics_enabled=true, pytics_n_iter=5; Marconi et al. 2026, RASTI): kazdy komp se docasne stane cilem a jeho vaha se zpresni z rozptylu vuci zbytku souboru. Obe smycky zpresnuji per-frame zeropoint jeste PRED vypoctem delta_mag.", B))
st.append(fn("photometry_comp.py: select_comparison_stars_per_target, _select_comps_tiered, _select_comps_by_color_then_rms, _bprp_tier_ladder_for_selection, build_global_comp_pool, _count_gate_passing_comps; phase01_run.py: run_phase0_and_phase1; photometry_core.py: check_comparison_stability, pytics_iterative_weights, _common_mode_detrend_comp_lc; comp_pool_rms.py; comp_selection_per_target.py; pipeline.py: select_comparison_stars_spatial_grid"))
st.append(par("comp_max_delta_bprp=0.79, comp_color_tiers=[0.15/1.0, 0.30/0.85, 0.55/0.50, 1.10/0.25], min_dist=60 arcsec, max_dist_deg=1.5 (+FOV), chip_margin=50 px, max_mag_diff=1.5 (abs 3.0, bright floor 1.5 pod 12.75 mag), max_comp_rms=0.1, rms_outlier_sigma=3.0, max_slope=5.0 mmag/h (significance k=3.0), isolation=25 px, max_fwhm_factor=1.5, max_psf_chi2=50, min_frames_frac=0.2, n_comp=3..8, sparse fallback ON, global pool ON, density adaptive ON (sparse<300, dense>1000), iterative clip ON, pytics ON (n_iter=5), comp_clip_sigma=5.0, comp_contamination_penalty_k=3.0, comp_select_rms_floor=1e-06"))
st.append(box("Proc je barva na prvnim miste", [
 "Atmosfericka extinkce zavisi na vlnove delce: modra hvezda slabne s rostouci vzdusnou hmotou rychleji nez cervena. U diferencialni fotometrie se extinkce 1. radu vyrusi v rozdilu cil-komp, ale ROZDIL BAREV ponechava zbytkovy clen 2. radu k'' x Delta(BP-RP) x X (kap. 10.8 v ramci Faze 2A).",
 "Zadny nastroj z literatury (AIJ, SysRem/TFA, EPD) neresi airmassove systematiky vyberovou chirurgii kompu - vsechny pouzivaji korekci ci dekorelaci. VYVAR proto drzi tesnou barevnou shodu jako PRIMARNI obranu (u nefiltrovanych dat jedinou preventivni) a k'' korekci jako lecbu priciny.",
 "Negativni vysledek k zapamatovani: redesign vyberu kompu 'measure-all -> greedy grow' (Broeg inverse-variance) vyhraval na rucne vybranych cilech, ale v populacnim testu explicitne zhorsil 45 % cilu - byl odmitnut a zdokumentovan (DECISIONS). Airmass je korekci, ne vyberovy problem."]))

# ============================================================ CH 11 (Phase 2A) - core chapter
st.append(PageBreak())
st.append(P("11. Faze 2A - aperturni fotometrie (hlavni cesta) - DETAILNE", H1))
st.append(P("Vstup: zarovnane kalibrovane snimky + per-frame katalogy + soubor kompu. Prehled kroku pro kazdou hvezdu na kazdem snimku (detaily v podsekcich):", B))
st.append(tab([
 ["Krok","Co se deje"],
 ["1 Centroid","pozice z per-frame katalogu (DAO centroid); zadne refitovani v aperture"],
 ["2 Apertura","polomer = f x nocni QC FWHM (APERTURE-01; SNR tabulka je diagnostika)"],
 ["3 Pozadi","mezikruzi annulus 2.7..5.2 x FWHM (AIJ 14/27 px na 516); robustni odhad (median / sigma-clip, MAD)"],
 ["4 Tok","suma pixelu v aperture minus pozadi x plocha; saturacni a dilucni vlajky"],
 ["5 Chyba","Howellova CCD rovnice + EMPIRICKY sum pozadi (prazdne apertury) + SEM ensemble + sigma_sys dno (11.4)"],
 ["6 Aperturni korekce","preskalovani na spolecnou skalu z jasnych referencnich kompu - Metoda B (11.5)"],
 ["7 k'' korekce","extinkce 2. radu na comp mag_inst (literaturni k'' v BP-RP jednotkach; 11.8)"],
 ["8 Diferencial","cil minus vazeny soubor kompu (flux-sum kanonicka kombinace, Broegovy vahy; 11.6)"],
 ["9 Ansambl QA","spolecne odchylky, epochove korekce (Honeycutt-style), comp QA statistiky"],
 ["10 Postprocess","outlier maskovani (mask-first), volitelne CT, mag_calib_final (11.9-11.10)"],
], [34,134]))

st.append(P("11.1 Apertura per hvezda: growth curve a SNR optimum", H2))
st.append(P("Polomer apertury se odvozuje z FWHM. Souvislost s Gaussovou sirkou: FWHM = 2 sqrt(2 ln 2) x sigma = 2.3548 sigma. VYVAR pracuje ve dvou krocich:", B))
st.append(P("[b]Krok 1 - globalni pevna apertura z FWHM:[/b] produkcni polomer r = aperture_fwhm_factor=1.35 x nocni QC FWHM (APERTURE-01d, rezim f_fixed_night; jeden r pro cil i kompy). SNR-tabulka a scatter ladder jsou diagnostika. Role-aware skalovani: cil dostava faktor aperture_variable_factor=1.0, kompy aperture_comp_factor=1.1.", B))
st.append(P("[b]Krok 2 - per-hvezda SNR-optimalni apertura[/b] (compute_snr_optimal_aperture_table): pro magnitudove biny (rozsah 7..18 mag, krok 0.5) se z modelu Gaussova enclosed flux prohleda polomer r od r_min=0.8 x FWHM do r_max=2.5 x FWHM s krokem 0.05 px a vybere se r maximalizujici SNR dle CCD rovnice (Howell 1989; Merline & Howell 1995):", B))
st.append(EQP("SNR(r) = F(r)/g / sqrt( F(r)/g + N_pix(r) x bkg_var/g^2 )    [vse v elektronech]"))
st.append(P("kde F(r) je flux v aperture [ADU], g zisk [e-/ADU], N_pix(r) = pi r^2 pocet pixelu apertury a bkg_var rozptyl pozadi na pixel. Kdyz je k dispozici zmereny rozptyl pozadi z prazdnych pixelu tehoz snimku (bkg_var_adu2_per_px), pouzije se misto teoretickeho sky/g + (RN/g)^2. Sazba per trida jasnosti: aperture_snr_sizing = {small: 1.5, large: 4.0} x FWHM jsou meze prohledavaneho rozsahu pro slabe/jasne hvezdy - optimum pro slabe hvezdy lezi ~0.7 x FWHM (dominuje sum pozadi), pro jasne prevazi robustnost vetsi apertury. Pri FWHM = 3.7 px tak apertura ~5-6 px obsahuje ~94-98 % svetla a lezi blizko SNR optima. Tabulka se predpocitava per draft (precompute_and_save_snr_aperture_table_for_draft) a apertura per hvezda je pak KONZISTENTNE aplikovana napric snimky - pro diferencialni fotometrii se konstantni apertura rusi v rozdilu.", B))

st.append(box("Miniprimer: magnitudy a proc 1.0857", [
 "Pogsonova skala: m = -2.5 log10(F) + ZP. Rozdil 5 mag = faktor 100 v toku; 1 mag ~ faktor 2.512.",
 "Prevod relativni chyby toku na magnitudy: sigma_mag = |dm/dF| sigma_F = (2.5 / ln 10) x sigma_F/F = 1.0857 x sigma_F/F. Proto se v cele dokumentaci objevuje konstanta 1.0857.",
 "Diferencialni fotometrie meri ROZDIL magnitud cil-reference: cokoli spolecneho (pruhlednost, extinkce 1. radu, drobna rozostreni) se v rozdilu vyrusi - zbyva jen to, cim se cil od reference LISI (barva -> k'', pozice -> flat/vinetace, jasnost -> nelinearita)."]))
st.append(P("[b]Enclosed flux Gaussova profilu[/b] (zaklad SNR tabulky): E(r) = 1 - exp(-r^2 / (2 sigma^2)), sigma = FWHM/2.3548. Odtud E(1.0 FWHM) ~ 93.7 %, E(1.5 FWHM) ~ 99.8 %, E(1.75 FWHM) ~ 99.99 %. Maly polomer ztraci svetlo (ale i sum pozadi ~ r^2), velky polomer sbira sum - odtud existence SNR optima zavisleho na jasnosti hvezdy.", B))
st.append(P("11.2 Pozadi: sky annulus", H2))
st.append(P("Pozadi se meri v mezikruzi okolo hvezdy, mimo kridla PSF: vnitrni polomer annulus_inner_fwhm=2.7 x FWHM, vnejsi annulus_outer_fwhm=5.2 x FWHM (APERTURE-01d; na 516 = AIJ Sky_Inner/Outer 14/27 px). Howell: annulus tesne za kridly apertury. Odhad na pixel je robustni: median (pripadne sigma-clipped median) hodnot v mezikruzi, rozptyl pres MAD - potlaci kontaminaci sousednimi hvezdami a horkymi pixely. Odectena hodnota v aperture = sky_per_pixel x N_pix(r_ap). Hustotni adaptace muze vnitrni polomer na hustych polich posunout v Phase 1 effective cfg; stamping r_in/r_out na katalogu cte surove AppConfig. Tradice DAOPHOT (Stetson 1987); optimalni apertura dle Howella (1989).", B))
st.append(P("11.3 Tok a vlajky", H2))
st.append(P("Tok = suma pixelu v kruzne aperture (photutils CircularAperture / aperture_photometry, presne pocitani zlomkovych pixelu) minus pozadi. K bodu se pripisuji vlajky: saturace (peak_max_adu vs limit), nelinearita (diagnostika nonlinearity_fwhm_ratio=1.25 na percentilu 20 nejjasnejsich), pripadne dilucni vlajky (11.11) a bad-column/BPM zasahy (enhance_catalog_dataframe_aperture_bpm, bad_columns_for_light_frame).", B))

st.append(P("11.4 Chybovy model - tri vrstvy", H2))
st.append(P("[b]Vrstva 1 - fotonova statistika (Howell 1989):[/b] relativni chyba fluxu v elektronech:", B))
st.append(EQP("sigma_F^2 = F/g + N_pix x ( N_sky/g + N_dark/g + (RN/g)^2 )    ->    sigma_mag = 1.0857 x sigma_F / F"))
st.append(P("Tri cleny: fotonovy sum hvezdy, sum pozadi (obloha x plocha apertury) a cteci sum (RN^2 x plocha). Gain a read noise pochazeji z hlavicek / DB vybaveni (resolver provenience); pri softwarovem binningu plati RN_eff = RN_db x bin (READNOISE_E je per-pixel pri bin1; DECISIONS 2026-07-07).", B))
st.append(P("[b]Vrstva 2 - empiricke pozadi (Labbe et al. 2003; F-BINGAIN-1, vzdy empiricke):[/b] teoreticky vypocet sumu pozadi PODCENUJE korelovany sum (resampling pri zarovnani, ploche gradienty, drobne artefakty flatu). VYVAR proto sigma_bkg MERI: na kazdem snimku se rozmisti err_empty_apertures_n=64 prazdnych apertur (minimum validnich 16) mimo hvezdy, kazde umisteni pouziva TENTYZ anulovy odecet pozadi jako produkci mereni, a robustni rozptyl cistych souctu = sigma_bkg_ap [ADU]. Tato hodnota uz obsahuje Poissonuv sum pozadi, cteci sum, kovarianci resamplingu i clen odhadu oblohy dle Merline & Howell (1995) - NIC z toho se nesmi pridavat podruhe (double-counting). Determinismus (LABBE-DET): seznam hvezd se kanonizuje a RNG se odvozuje ze SeedSequence (content seed + r_ap), takze opakovana fotometrie tehoz draftu je bajtove stabilni. Kdyz sigma_bkg_ap chybi, Howellova variance je datovy fallback (howell_fallback); klic err_background_mode byl odstraneny. Pro chybejici mereni existuje i hybridni fallback: pomer empiricke/Howellovy sigmy per setup (bkg_scale_ratio_empirical_over_howell) preskaluje teoretickou hodnotu.", B))
st.append(P("[b]Vrstva 3 - produkci kombinace (sigma_floor_core.py):[/b] celkova chyba bodu svetelne krivky v relativnim fluxovem prostoru:", B))
st.append(EQP("err_total^2 = err_photon_bkg^2 + sem_ens_rel^2 + sigma_sys_rel^2"))
st.append(P("kde sem_ens_rel je SEM (standard error of the mean) REZIDUALU ensemble zeropointu - nikoli std instrumentalnich magnitud kompu (DECISIONS 2026-06-18: std kompu meri jejich jasovy rozptyl, ne nejistotu zeropointu; zamena nafukovala chyby ~10x). Pro maly pocet kompu se SEM koriguje faktorem c4 (nestrannost odhadu smerodatne odchylky pri malem n). sigma_sys je per-pasmove chybove dno kalibrovane z rozptylu kontrolnich hvezd - sigma_sys_mag = {'4': 0.018} (18 mmag pro wide/B4 pasmo); pricita se kvadraticky a JE soucasti sigma v exportech. Diagnosticky (mimo produkci) existuje sigma_budget.py: plny rozpocet Howell + Young/Osborn scintilace pro porovnani ocekavani vs mereni.", B))


st.append(box("Pocetni priklad: rozpocet chyby jednoho bodu (ilustracni, wide rig)", [
 "Hvezda G ~ 12.5, expozice 60 s, gain 1.0 e-/ADU, FWHM 3.7 px, apertura r = 5.6 px (N_pix ~ 98), obloha 210 ADU/px, RN 3.5 e-, tok hvezdy F = 48 000 ADU.",
 "Fotony hvezdy: sqrt(48000) ~ 219 e- -> 219/48000 = 0.46 %. Teoreticke pozadi: sqrt(98 x (210 + 12.25)) ~ 148 e- -> 0.31 %. Empiricka sigma_bkg_ap (prazdne apertury) vsak vyjde treba 190 ADU (korelace resamplingu!) -> 0.40 % - pouzije se TATO hodnota.",
 "Photon (+) empiricke pozadi: sqrt(0.46^2 + 0.40^2) ~ 0.61 % ~ 6.6 mmag. SEM ensemble rezidualu (7 kompu, c4 korekce): ~2.5 mmag. sigma_sys (pasmo 4): 18 mmag.",
 "err_total = sqrt(6.6^2 + 2.5^2 + 18^2) ~ 19.3 mmag. Pouceni: na wide rigu casto DOMINUJE systematicke dno - proto ma smysl kalibrovat sigma_sys z check hvezd a proc 'jeste vetsi apertura' nic nezachrani."]))
st.append(P("11.5 Aperturni korekce - 'Metoda B'", H2))
st.append(P("SNR-optimalni apertura je mensi nez 'totalni', takze ztraci cast svetla - a ruzne jasne hvezdy maji ruzne optimum, coz by vnaselo systematiku. Korekce (compute_aperture_correction, aperture_correction_enabled=true): z referencnich kompu (preferencne Tier 1, pri nedostatku doplni Tier 2; filtr kontaminace <= aperture_correction_max_contamination=0.15 a platne comp_rms) se pres snimky spocita median rozdilu velke a male apertury Delta M_corr = mag_large - mag_small. Korekce se aplikuje jen pri rozptylu referencnich hodnot <= aperture_correction_max_scatter_mag=0.03 mag a poctu >= aperture_correction_min_ref_stars=3 (jinak ok=False, beze zmeny - fail-safe). Vraci svetlo 'ztracene' malou aperturou, aniz by obetovala jeji SNR. Reference: krivka rustu (Stetson 1990; Howell 1989). Volitelna alternativa COG (curve-of-growth; cog_aperture_correction_enabled=false) je implementovana, ale vypnuta; mixed-frame riziko odstraneno all-or-nothing nocni pojistkou (DECISIONS), zapnuti ceka na validaci.", B))

st.append(P("11.6 Ensemble normalizace (Broeg / AIJ flux-sum)", H2))
st.append(P("ensemble_normalize provadi skutecnou ansamblovou fotometrii (ne jedinou komparaci). Per snimek:", B))
st.append(EQP("mag_ensemble = -2.5 log10( SUM_j 10^(-0.4 m_j) )          [soucet fluxu, AIJ tot_C_cnts]"))
st.append(EQP("delta_mag    = mag_inst(cil) - mag_ensemble               [diferencialni krivka]"))
st.append(EQP("mag_calib    = mag_inst(cil) + median_j( cat_mag_j - mag_inst_j )   [kalibrovana]"))
st.append(P("Pozor na zeropoint kalibrace: soucet fluxu dava m_ensemble ~ m_i - 2.5 log10(n) pri podobnych m_i, takze pricteni pouheho median(cat) ke krivce delta_mag by ji posunulo o ~2.5 log10(n) mag. Proto mag_calib pouziva klasicky diferencialni posun median(cat - inst), zatimco delta_mag zustava vuci AIJ souctu (kanonicka kombinace A vs B vyresena v DECISIONS 2026-06-15; validovano proti AstroImageJ: Delta < 0.001 mag, 67 hvezd). Vyber clenu ensemble: kompy kvality good i suspect (excluded ne), serazene dle comp_rms z Faze 1; prvnich n_comp_min vzdy, dalsi jen pri rms_p2p pod prahem stability, max n_comp_max. Vazeni w ~ 1/sigma^2 dle zmerene variability kompu (Broeg 2005) s tierovymi vahami barvy (10.2) a penalizaci kontaminace (comp_contamination_penalty_k=3.0). Ensemble scatter (rozptyl kompu okolo medianu) se propaguje do chyboveho modelu (11.4, vrstva 3).", B))


st.append(P("[b]Epochove korekce (Honeycutt 1992) v ansamblovem retezci:[/b] klasicka ansamblova fotometrie resi soustavu m_ij = m_i + e_j (hvezda i, epocha j) - kazdy snimek dostane svou epochovou korekci e_j (spolecny posun pruhlednosti/citlivosti) a kazda hvezda svou stredni magnitudu m_i. VYVAR tuto logiku realizuje per-frame ensemble zeropointem (median cat - inst pres kompy daneho snimku) + iterativnim cistenim; rezidua zeropointu davaji SEM slozku chyby (11.4). Vysledek je ekvivalentni Honeycuttove principu 'kazdy snimek ma svuj offset', jen pocitany robustne po snimcich.", B))

st.append(P("11.6b Saturace, nelinearita a smerovani chyb per bod", H2))
st.append(P("Saturace se posuzuje per bod (peak_max_adu vs limit kamery z DB/hlavicky): saturovany bod dostava flag a nevstupuje do ensemble statistik; cil saturovany na vetsine snimku je vyrazen uz ve Fazi 0. Nelinearita CMOS pod plnou studnou se diagnostikuje tvarem: jasne hvezdy 'tloustnou' (nonlinearity_fwhm_ratio=1.25 proti percentilu 20) - varovani, ze i nesaturovane spicky lzou. Chyba bodu se smeruje dle metody (_route_lc_per_frame_err): aperturni bod nese aperturni chybu, PSF bod PSF chybu (sandwich) - zadne micheni skal (T1 nalez auditu, 12.4). BJD/HJD se prepocitava PER CIL (souradnice cile vstupuji do korekce; _recompute_bjd_hjd_per_target) se statusem time_base.", B))
st.append(P("11.7 FWHM v pipeline - ctyri cesty (a proc)", H2))
st.append(P("FWHM se objevuje na vice mistech a pocita se ruzne - je dulezite je rozlisovat: [b](1)[/b] QC na MASTERSTARu (measure_fwhm_from_masterstar) zapisuje do hlavicek DAO-styl hodnotu (momentova sirka detekce; rychla, robustni) a 2D Gaussuv fit jasnych hvezd (presnejsi jadro). [b](2)[/b] Faze 2A voli zdroj v poradi 2D fit -> DAO -> fallback (resolve_fwhm_px_for_snr_aperture_table pro SNR tabulku). [b](3)[/b] Model-free druhe momenty (_fwhm_moment_at): FWHM z druhych momentu intenzity ve vyrezu bez predpokladu profilu - pouziva se tam, kde Gaussuv fit s pevnym vyrezem zkresluje (lekce ze STATE: pro elongaci/asymetrii davaji momenty nezkresleny vysledek, fit je systematicky nadhodnoceny). [b](4)[/b] Moffat FWHM v PSF ceste (kap. 11B): FWHM = 2 gamma sqrt(2^(1/alpha) - 1).", B))

st.append(P("11.8 Extinkce 2. radu k'' (band-aware, ZIVA v produkci)", H2))
st.append(P("Extinkce 1. radu se v diferencialu rusi; zbyva clen 2. radu zavisly na ROZDILU barev cile a kompu: delta_m = k'' x Delta C x X (X = vzdusna hmota, Kasten & Young 1989). VYVAR jej resi korekci na strane kompu (apply_k2_to_comp_mag_inst v ensemble retezci): instrumentalni magnitudy kompu se pred kombinaci opravi k''-clenem, takze zbytkovy barevny trend se odstrani u zdroje. Rezim k2_mode='literature': literaturni koeficienty (Smith 2002 pro Sloan, Henden & Kaitchuck pro Johnson B...) se prevadeji do BP-RP jednotek pres barevne sklony Jordi et al. 2010 (k2_bprp = k''_nativni x dC_nativni/dC_bprp v FGK kotve); pasmo urcuje band_classify.py (jediny zdroj pravdy pro klasifikaci filtru; fail-safe: neznamy filtr = CLEAR/UNFILTERED -> tesna barva, zadne nespolehlive k''). Pojistky: strop k2_ceiling=0.1 mag/airmass/barvu; per-nocni FIT k'' (NIGHT_FIT v2) je IMPLEMENTOVAN a synteticky validovan (recovery sweep + REFUSE brany vc. monotonni airmass); aktivace (k2_fit_enabled) ceka jen na noc s dostatecnou detekovatelnosti (K2-DATA-BLOCKER). Merene poradi dulezitosti na wide rigu: k'' clen je SUBDOMINANTNI (K2-STATS-FIX: bootstrap CI, rho ~ -0.013) - tesna barevna shoda kompu zustava primarni obranou, k'' je korektni lecba zbytku.", B))
st.append(P("Design spec: dev/results/specs/VYVAR_K2_DESIGN_SPEC.md v1.1. Poznamka k historii: starsi per-target airmass detrend (fit linearniho trendu na 'normal' bodech) byl z produkce ODSTRANEN - detrend cilove krivky je symptom-fix, ktery umi sezrat realnou variabilitu; reporting postprocess dnes airmass NEdetrenduje (mask-first outliery, zadny per-target trend).", B))


st.append(box("Pocetni priklad k''", [
 "NoFilter (CV pasmo), literaturni k''_bprp ~ 0.03 mag/airmass/mag(BP-RP). Cil BP-RP = 1.30, median kompu 0.85 -> Delta C = 0.45. Noc: X od 1.15 (kulminace) do 1.95 (konec rady).",
 "Zbytkovy clen: k'' x Delta C x X = 0.03 x 0.45 x (1.95-1.15) ~ 10.8 mmag ROZDILU pres noc - falesny 'trend', ktery by neopatrny pozorovatel precetl jako pomalou variabilitu.",
 "Obrana 1 (prevence): tesna barva kompu - pri Delta C = 0.1 klesne efekt na ~2.4 mmag, pod sumove dno. Obrana 2 (lecba): apply_k2_to_comp_mag_inst koriguje zbytek. Poradi dulezitosti presne v tomto smyslu potvrdila i mereni na wide rigu (k'' subdominantni)."]))
st.append(P("11.9 Barevny clen (transformace; default OFF)", H2))
st.append(P("Infrastruktura barevne transformace do standardniho systemu je pripravena (fit_color_term_c1, apply_color_term, should_apply_color_term): fit vztahu (cat - inst) vs BP-RP z kompu (minimum phase01_ct_min_comp=7, extrapolace mimo rozsah barev kompu zakazana, phase01_ct_extrapolation_tol=0.0) a aplikace na cil. Vychozi stav apply_color_term='off' - krivky zustavaji v instrumentalnim systemu s diferencialnim zeropointem; pro NoFilter/Clear se CT neaplikuje NIKDY (neni definovan cilovy system), pro V/B/R/I/Sloan jen je-li spolehlivy. Sloupce: mag_calib_ct, ct_correction, ct_c1, ct_bp_rp_target, ct_ok.", B))
st.append(P("11.10 Kanonicka publikovana magnituda", H2))
st.append(EQP("mag_calib_final = mag_calib + CT (je-li ct_ok) + Delta M_corr (je-li ac_ok)      [Path A, DECISIONS 2026-06-22]"))
st.append(P("CT a AC jsou aditivni per-target/nocni konstanty nad ensemble-kalibrovanou bazi; pri CT off je vysledek bajtove identicky s mag_calib_ac. Svetelna krivka (CSV) nese vsechny reprezentace: mag_inst, mag_calib, delta_mag_ensemble, mag_calib_final, err, aperture_r_px, method - prepinac mag/delta_mag v UI meni jen osu grafu, nikdy data.", B))


st.append(P("11.10b Outliery a reporting postprocess (mask-first)", H2))
st.append(P("detect_outliers znaci body odlehle od lokalniho prubehu krivky (robustni MAD kriterium s ochranou tvaru zakrytu - maskovani nesmi 'ukousnout' minimum zakrytu); empirical_feature_mask_mag chrani oblasti se skutecnym signalem. Zasada MASK-FIRST: outlier se NIKDY nemaze z CSV - dostane flag a grafy/statistiky ho tlumi; data zustavaji kompletni pro reanalyzu. apply_reporting_postprocess je posledni krok pred zapisem: aplikuje vlajky, pocita lc_rms/kvalitu (classify_lc_quality: good/noisy/short dle poctu snimku a podilu normalnich bodu - min 20 snimku pro plny verdikt, kratky track od 3) a NIC nedetrenduje (zadny per-target airmass fit - 11.8).", B))
st.append(P("11.11 Redeni toku (dilution, GS11) - volitelne, default OFF", H2))
st.append(P("Merena apertura muze obsahovat svetlo nerozlisenych sousedu (blend), ktere 'redi' merenou amplitudu a posouva magnitudu. Modul dilution.py (gs11_dilution_enabled=false) odhaduje redeni KATALOGOVE z Gaia DR3 (predikce, ne mereni): pro cil se najdou Gaia sousede uvnitr apertury (search_radius = aperture_arcsec; gs11_dilution_aperture_arcsec=0 = odvod z fotometricke apertury), zahrnou se sousede az o gs11_dilution_mag_limit_delta=5.0 mag slabsi (slabsi prispivaji < 1 % toku) a spocita se:", B))
st.append(EQP("D = F_star / (F_star + SUM F_neighbors);   delta_mag = -2.5 log10(D) > 0 pri D < 1;   mag_corr = mag_obs + delta_mag"))
st.append(P("Vystupy: dilution_factor, dilution_delta_mag, n_neighbors, neighbor_flux_sum. Uzitecne jako QA priznak blendu a korekce na hustych polich; kompy s D < gs11_comp_max_dilution=0.9 se vyrazuji, D < 0.98 znaci suspect. Reference: Seager & Mallen-Ornelas (2003); Howell (2006). Skutecne PSF odecteni souseda je v PSF vetvi (kap. 11B).", B))

st.append(P("11.12 Co je zamerne VYPNUTO a proc (negativni vysledky)", H2))
st.append(tab([
 ["Metoda","Stav a duvod"],
 ["ALG-3 casove binovani kompu","temporal_binning_enabled=false. Injekcni testy PROKAZALY skodlivost: klouzavy median vyhladil serie kompu a vtiskl artefakty pruhlednosti do diferencialnich magnitud; populacni sweep 24/25 cilu horsich, 0 lepsich. Kod (temporal_bin_comp_lc) zustava pro reprodukci testu."],
 ["ALG-5 PyTICS","pytics_enabled=TRUE - jedina zapnuta 'pokrocila' metoda; iterativni interkalibrace vah kompu (10.6)."],
 ["ALG-2 Savitzky-Golay detrend","savgol_detrend_enabled=false (opt-in). Vyhlazeni SG filtrem umi odstranit pomaly nelinearni trend, ale detrend cilove krivky je rizikem pro realnou variabilitu."],
 ["ALG-4 Democratic Detrender","democratic_detrend_enabled=false (opt-in). Marginalizace pres 3 modely trendu (polynom v airmass, polynom v BJD, SG v case) s poctivou inflaci chyb pri nesouhlasu modelu (err_inflation = MAD modelu); sloupce delta_mag_democratic, err_inflation."],
 ["SysRem","sysrem_enabled=false. run_sysrem_field (Tamuz+ 2005) iterativne odecita nejsilnejsi spolecny trend R_ij = c_i x a_j (n_iter=3); riziko: umi 'odecist' i skutecnou variabilitu sdilenou vice hvezdami. Vyhodnoceno, drzeno OFF."],
 ["Per-target airmass detrend","ODSTRANEN z produkce (11.8) - nahrazen pricinami: barevna shoda + k''."],
 ["COG aperturni korekce","cog_aperture_correction_enabled=false; mixed-frame riziko odstraneno all-or-nothing nocni pojistkou (11.5 / DECISIONS); zapnuti ceka na validaci."],
], [42,126]))
st.append(P("Spolecny jmenovatel: kazda z metod byla implementovana, otestovana injekcnimi ci populacnimi testy a rozhodnuti je zdokumentovano v DECISIONS. Citace se v reportu emituji JEN pro metody, ktere skutecne bezely (citations.py; gating priznak = citacni priznak).", B))
st.append(fn("photometry_core.py: run_full_photometry_pipeline (PRODUKCNI VSTUP), run_phase2a, compute_snr_optimal_aperture_table, precompute_and_save_snr_aperture_table_for_draft, _photometric_error(_with_bkg_mode), compute_aperture_correction, ensemble_normalize, compute_mag_calib_final, apply_reporting_postprocess, detect_outliers, save_lightcurve_csv; photometry_gate_helpers.py: measure_empty_aperture_sigma_bkg (re-exported from photometry_core); sigma_floor_core.py: combine_production_err_rel, c4_small_sample, ensemble_sem_mag_from_residuals; k2_extinction.py: resolve_k2_bprp_value, apply_k2_to_comp_mag_inst, computed_k2_bprp_for_token; band_classify.py: classify_photometric_band, effective_band_for_extinction; dilution.py; photutils.aperture: CircularAperture, CircularAnnulus, aperture_photometry"))
st.append(par("aperture_fwhm_factor=1.35, aperture_variable_factor=1.0, aperture_comp_factor=1.1, aperture_snr_sizing={small:1.5, large:4.0}, annulus 2.7..5.2 FWHM, empirical empty-aperture ERR (n=64, min 16; F-BINGAIN-1), sigma_sys_mag={'4':0.018}, aperture_correction ON (min_ref 3, contamination 0.15, scatter 0.03), k2_mode='literature' (ceiling 0.1, fit OFF), apply_color_term='off' (ct_min_comp 7, extrapolation 0.0), gs11 OFF, sysrem OFF (n_iter 3), temporal binning OFF, savgol OFF (polyorder 2, window 0.5), democratic OFF (window 0.5), pytics ON (n_iter 5), phase2a_airmass_before_outlier=false, photometry_mode='both', save_lightcurve_png=false"))

# ============================================================ CH 12 (Phase 2B ePSF)
st.append(PageBreak())
st.append(P("12. Faze 2B - ePSF fotometrie (pripravena, vypnuta) - DETAILNE", H1))
st.append(P("PSF fotometrie nefotometruje kruh, ale FITUJE MODEL hvezdneho profilu na data. Prinos je dvoji: [b](a)[/b] vazi centralni pixely, kde je signal, coz zlepsi SNR slabych hvezd; [b](b)[/b] matematicky rozdeluje tok mezi prekryvajici se profily (deblending) - v hustych polich, kde apertura konci. VYVAR ma plnou vetev (psf_photometry.py ~3100 radku + psf_runner.py + psf_neighbor_sub.py) pripravenou a globalne vypnutou (psf_photometry_enabled=false) do validace na hustem poli Newtonu.", B))

st.append(P("[b]Koncept pro nefotometriky:[/b] PSF (Point Spread Function) je 'otisk' bodoveho zdroje - jak dalekohled + atmosfera + cip rozmaznou matematicky bod hvezdy do skvrny. Aperturni fotometrie skvrnu SCITA (kruh a soucet - zadny predpoklad tvaru); PSF fotometrie skvrnu MODELUJE a hleda, jaka amplituda modelu nejlepe sedi na data. Modelovani vynasi dve schopnosti navic: vahovani pixelu podle informace (stred skvrny vs okraj) a rozklad prekryvajicich se skvrn na jednotlive hvezdy. Cena: vysledek je jen tak dobry, jak dobry je model - proto tolik pece o stavbu ePSF a kontrolu kvality fitu nize.", B))
st.append(P("12.1 Dva modely profilu: Moffat a ePSF", H2))
st.append(P("[b]Moffat (analyticky):[/b] radialni profil I(r) = I0 x (1 + (r/gamma)^2)^(-alpha), kde gamma je sirka jadra a alpha mocninovy index (v klasickem znaceni beta). Moffat modeluje kridla PSF lepe nez Gauss - atmosfericky seeing ma mocninna, ne gaussovska kridla. Souvislost s FWHM (_moffat_fwhm_px):", B))
st.append(EQP("FWHM = 2 gamma sqrt( 2^(1/alpha) - 1 )"))
st.append(P("[b]ePSF (empiricka 'effective PSF', Anderson & King 2000):[/b] misto predpokladu tvaru se PSF ZMERI z dat: jasne izolovane nesaturovane hvezdy se vyrezou, subpixelove zarovnaji a iterativne slozi do prevzorkovaneho (oversampled) rastru. Vyhoda: zachyti realny tvar (seeing, optika, vedeni montaze) vcetne asymetrii, ktere zadny analyticky model nepopise. Nevyhoda: potrebuje dost hvezd (epsf_min_stars=30) a stabilni profil pres noc. Moffat slouzi jako fallback pri nedostatku hvezd (parametry gamma/alpha z medianu FWHM).", B))
st.append(P("12.2 Stavba modelu", H2))
st.append(P("[b]build_epsf_model:[/b] kandidati se vybiraji z per-frame katalogu (izolace, SNR, nesaturovane; _epsf_prepare_stars), pripadne se doplni z detekovaneho poolu (_epsf_augment_candidates_from_detected_pool). photutils EPSFBuilder iterativne stavi model: extract_stars -> fit -> re-centrace -> smoothing, opakovane do konvergence; vysledkem je ImagePSF (rastrovy model s oversamplingem), ulozeny jako masterstar_epsf.fits. Vstupni FWHM se resi kaskadou hlavicka -> DB median -> fallback 4.5 px; vyrez je lichy, min 5 px. FWHM native modelu se meri z radialniho profilu ePSF (_epsf_fwhm_native_from_profile) - kontrola konzistence proti ocekavani.", B))
st.append(P("[b]build_epsf_grid_model:[/b] volitelna mrizka lokalnich ImagePSF modelu zachycuje prostorovou promennost PSF pres pole (koma v rozich, naklon ohniskove roviny). Mereni pak interpoluje model pro konkretni pozici (interp_gridded_epsf_array). Prepinac psf_spatial_enabled=false; psf_spatial_order=0 znamena konstantni PSF. Persistovane klice mrizky (psf_spatial_grid / min stars per cell) byly odstraneny - nikdy nebyly cteny.", B))

st.append(P("[b]K cemu je oversampling:[/b] ePSF se stavi na jemnejsi mrizce nez pixel (typicky 4x) - kazda hvezda dopada na pixely s jinym subpixelovym posunem a slozenim mnoha hvezd do jemne mrizky se profil rekonstruuje NAD pixelovym rozlisenim. Presne proto ePSF funguje i na undersamplovanych datech (wide rig, FWHM ~ 2-3 px), kde by fit hrubeho modelu selhaval na interpolacnich artefaktech.", B))
st.append(P("[b]Kvalita fitu chi2:[/b] chi^2 = SUM ( (data - model)^2 / sigma_pix^2 ) pres pixely fitovaciho okna, s per-pixel sigma z chybove mapy (Poisson + sky + RN). Redukovane chi^2 ~ 1 znaci konzistentni fit; strop psf_chi2_threshold=50 je zamerne benevolentni (odmita jen evidentni havarie - kosmik v okne, spatny soused), jemnejsi selekci dela assess_psf_quality.", B))
st.append(P("12.3 Mereni", H2))
st.append(P("SourceGrouper(min_separation = psf_group_sep_fwhm=1.5 x FWHM) sdruzi prekryvajici se hvezdy do skupin fitovanych SPOLECNE (_grouped_psf_fit; psf_grouper_enabled=false do validace) - to je podstata deblendingu: model soucasne resi pozice a amplitudy vsech clenu skupiny. PSFPhotometry fituje model v okne fit_shape odvozenem z FWHM (_fit_shape_for_cutout); IterativePSFPhotometry volitelne pridava smycku najdi-odecti-najdi pro slabe sousedy. Pozadi je lokalni per hvezda: anulovy median s kaskadou fallbacku (_annulus_median_per_px -> rezidualni anulus po odectu modelu sousedu -> full-frame odhad), okrajovy median vyrezu pro cutout cesty. Kvalita fitu se propisuje do vlajek: psf_chi2 (strop psf_chi2_threshold=50), konvergence, assess_psf_quality; pri selhani kvality nastupuje fallback na aperturu (psf_quality_fallback_enabled=true; priznak psf_quality_fallback).", B))
st.append(P("12.4 Chybovy model PSF a kalibrace na aperturni skalu", H2))
st.append(P("[b]Chyba PSF toku je ODDELENA od aperturni:[/b] psf_flux_err se pocita 'sandwich' odhadem z fitu (_psf_sandwich_flux_err) s per-cutout chybovou mapou (Poisson + sky RMS + RN; _per_cutout_error_map). Toto je vysledek T1 nalezu PSF auditu: driv se psf_flux_err nikde necetl a PSF-routovane snimky nesly aperturni chybu do LC, Honeycutt vazeni i AAVSO exportu - opraveno pri vypnute vetvi. [b]AC faktor (aperture correction):[/b] PSF tok neni ve stejne skale jako aperturni; faktor se pocita z >= 5 referencnich hvezd merenych OBEMA cestami (median pomeru; sloupce psf_ac_factor, psf_ac_n_used, psf_ac_applied v proc CSV). Pod 5 referenci se PSF vysledek NEPOUZIJE (fallback na aperturu) - to je T2 nalez auditu (driv tichy default 1.0). [b]Adaptivni brana (router):[/b] psf_adaptive_enabled=false; az bude ON, PSF se routuje jen tam, kde ma smysl: SNR pod psf_adaptive_snr_lo=15 nebo blend se separaci pod psf_adaptive_resolve_fwhm=2.0 x FWHM; chi2 strop odmitne spatne fity. Vyber metody per hvezda se zapisuje (method, _resolve_star_flux_method) - zadne tiche michani.", B))
st.append(P("12.5 Odecet souseda (psf_neighbor_sub) - pripraveno", H2))
st.append(P("Pro rezim 'jasny soused kontaminuje slaby cil' je pripravena cesta: ePSF model jasneho souseda se odecte a cil se zmeri aperturou v reziduu. Prisne guardy (vse psf_neighbor_sub_enabled=false): rezim definovan separaci <= 1.1 FWHM a rozdilem jasnosti >= 2.5 mag; odmitnuti pod 0.8 FWHM (par prilis slity); fit souseda nesmi vyjit jasnejsi nez ocekavani o > 0.3 mag; cil nesmi ztratit > 0.2 mag; rezidualni RMS strop 150; minimalni obnovene SNR 5; centroid se nesmi posunout o > 1.0 FWHM.", B))

st.append(box("Kdy PSF fotometrie realne vyhrava (a kdy ne)", [
 "VYHRAVA: (a) blend se separaci 0.8-2 FWHM - apertura obou hvezd se prekryva a zadna volba polomeru to neresi; skupinovy fit toky rozdeli. (b) Slabe hvezdy na jasnem pozadi - vazeni centralnich pixelu zvedne SNR o desitky procent. (c) Husta pole (h/chi Per trida) - izolovanych hvezd pro cistou aperturu je malo.",
 "NEVYHRAVA: izolovane, dobre exponovane hvezdy na ridkem poli - apertura je jednodussi, bez modelove systematiky a s validaci AIJ. Presne proto je vychozi produkce aperturni a PSF je router-only pro rezimy (a)-(c).",
 "Poucka z auditu: PSF prinasi NOVE tride chyb (model mismatch, AC skala, dekorelovane chyby) - kazda z nich uz byla v kodu nalezena a osetrena, ale bez validace na realnem hustem poli zustava vetev OFF."]))
st.append(P("12.6 Proc je vetev vypnuta a co ji zapne", H2))
st.append(P("Aperturni cesta je VALIDOVANA (AIJ Delta < 0.001 mag); PSF cesta je kodove AUDITOVANA (4 nalezy opraveny pri vypnute vetvi: T1 chybova dekorelace, T2 tichy AC fallback, T3 mrtva vlajka, T4 odstraneny mrtvy per-frame mixer), ale NEVALIDOVANA na realnem hustem poli. Enablement checklist ma jedinou zbyvajici polozku: draft z husteho pole Newtonu (trida h/chi Per) + krizova validace vuci aperture na izolovanych hvezdach. Do te doby plati: kazde zapnuti bez validace = neznama systematika. (Kompletni zduvodneni: DECISIONS, PSF audit.)", B))

st.append(P("[b]Headless runner:[/b] psf_runner.py umoznuje spustit PSF mereni mimo UI (validacni behy): resi gain kaskadou hlavicka EGAIN -> DB vybaveni, cte vy_fwhm/vy_qcrms z hlavicek, pocita per-cutout chybove mapy a tiskne distribuci chi2 - presne nastroj, kterym probehne budouci validace na Newtonu. [b]Checklist zapnuti (zbyva 1 polozka):[/b] (1) hotovo - audit kodu (4 nalezy opraveny); (2) hotovo - psf_* sloupce v proc CSV + PROC_STORE_COLS guard; (3) hotovo - chybova dekorelace a AC pravidlo >= 5 referenci; (4) OTEVRENO - draft husteho pole Newtonu + krizova validace vuci aperture na izolovanych hvezdach (cil: shoda stredu < nekolik mmag, zadny trend s jasnosti).", B))
st.append(fn("psf_photometry.py: build_epsf_model, build_epsf_grid_model, interp_gridded_epsf_array, fit_moffat_psf_stars, _moffat_fwhm_px, psf_photometry_stars, _grouped_psf_fit, assess_psf_quality, _psf_sandwich_flux_err, _compute_aperture_correction; psf_runner.py (headless beh, gain reseni, chi2 distribuce); psf_neighbor_sub.py; pipeline.py: _fill_psf_catalog_columns (psf_* sloupce v proc CSV); photutils.psf: EPSFBuilder, ImagePSF, SourceGrouper, PSFPhotometry, IterativePSFPhotometry"))
st.append(par("psf_photometry_enabled=false, epsf_min_stars=30, psf_chi2_threshold=50.0, psf_quality_fallback_enabled=true, psf_adaptive_enabled=false (snr_lo=15, resolve=2.0 FWHM), psf_grouper_enabled=false (sep=1.5 FWHM), psf_spatial_enabled=false (grid 3x3, min 25/bunka, order 0), psf_neighbor_include_fwhm=3.0, psf_neighbor_sub_enabled=false (guardy viz 12.5), photometry_mode='both'"))

# ============================================================ CH 13 QA & trust
st.append(P("13. Kontrola kvality a duvera (comp QA, check star, trust gate)", H1))
st.append(P("Tato vrstva je jadrem trust-first filozofie. Bezi jako READ-ONLY faze po 2A: nemeni fotometrii (overeno bitovou identitou vystupu), jen ji znamkuje.", B))
st.append(P("13.1 comp_qa - Sokolovsky leave-one-out", H2))
st.append(P("Pro kazdou komparaci se sestavi jeji LOO (leave-one-out) diferencialni krivka - komp se meri jako cil vuci zbytku souboru - a spocitaji se indexy variability (comp_qa_core.py, Sokolovsky et al. 2017): robustni amplituda G_IQR = (P75-P25)/1.349, von Neumannuv pomer eta = s^2/sigma^2 (citlivy na pomaly drift - koreluje po sobe jdouci body), podil vypadku. Vyhodnoceni NENI proti plochemu prahu, ale proti magnitudove zavislemu locusu (biny po 0.5 mag, median + 4 x MAD) - opravuje nadmerne flagovani slabych komparaci, jejichz sum je prirozene vyssi. Vysledek: pocet 'cistych' komparaci per cil (n_clean v photometry_summary.csv, sidecar lightcurves/comp_qa_{id}.json).", B))

st.append(P("[b]Formule indexu:[/b] G_IQR = IQR/1.349 je robustni odhad sigmy (IQR = P75-P25; delitel 1.349 normalizuje na gaussovskou sigmu - outliery na nej nemaji vliv, na rozdil od proste std). Von Neumann eta = s^2/sigma^2, kde s^2 = mean( (x_{i+1} - x_i)^2 ) je stredni kvadraticka naslednost: bily sum ma eta ~ 2, pomaly drift/koherentni signal eta << 2 - index tedy chyti komp, ktery 'tece', i kdyz jeho celkova RMS vypada nenapadne. Locus: v binech po 0.5 mag se z populace kompu spocita median a MAD indexu; komp je vlajkovany az nad median + 4 x MAD SVEHO binu - slaby komp se posuzuje mezi slabymi, jasny mezi jasnymi.", B))
st.append(P("13.2 Check star a KMAG", H2))
st.append(P("Z komparaci se vybere jedna kontrolni hvezda (check star) - nejstabilnejsi (nejnizsi p2p_rms, fallback comp_rms) - a to jen zustane-li po jejim odebrani dost komparaci (select_check_star). Check star se MERI jako cil, ale je VYLOUCENA z vlastniho ensemble; jeji rozptyl je tak nezavislym ukazatelem kvality noci. Jeji ensemble-standardizovana kalibrovana magnituda KMAG (compute_check_ensemble_mag_calib) se uklada do sidecaru lightcurves/check_kmag_{id}.csv a exportuje do AAVSO (kap. 15.2) - AAVSO vyzaduje MERENOU magnitudu check star, ne katalogovou.", B))
st.append(P("13.3 Trust gate - hard/soft model", H2))
st.append(P("trust_flag_core.py cte photometry_summary.csv a vydava trojbarevny verdikt per cil (a per epocha pres vstupy jako catalog_match_mode, saturace, dilucni vlajky):", B))
st.append(tab([
 ["Uroven","Pravidlo"],
 ["RED","n_clean == 0 NEBO zadna check star NEBO jakekoli HARD varovani: lc_quality mimo {good, noisy}, check-star rozptyl >= 0.05 mag NEBO >= 3 SOFT priznaky"],
 ["YELLOW","1-2 SOFT priznaky: tenky ensemble (n_clean mezi min a strong-1, kde strong = min(min+2, max) = 5), check-star rozptyl 0.02-0.05 mag"],
 ["GREEN","0 priznaku: n_clean >= strong prah a check star sedi"],
], [24,144]))
st.append(P("Prahy se odvozuji z uzivatelskeho n_comp_min/max (3/8). Vychozi comp_trust_min_comps=3 vs literaturni 5 je INTENCIONALNI rozhodnuti (DECISIONS 2026-07-17): mereni s mene kompy PLATI, jen s chybou skalovanou ~1/sqrt(N) - plynula degradace misto binarniho zahazovani. lc_rms je pouze informacni (vychazi z variability cile, ne z chyby mereni: napr. V0349 Dra ma lc_rms 0.155 pri check-star 0.0054 -> GREEN). Trust je inform-only: RED se zobrazi, ale automaticky se z exportu nevyrazuje - rozhoduje clovek. Priznak se propisuje do PDF (barevny badge 'TRUST: UROVEN - duvod'), do AAVSO NOTES (trust=UROVEN..., < 100 znaku) a do VarAstro (# Trust: ...).", B))

st.append(box("Trust gate na prikladech", [
 "GREEN: n_clean = 6 (>= strong 5), check-star scatter 0.008 mag, zadny hard priznak - krivka i chyby jsou duveryhodne tak, jak jsou.",
 "YELLOW: n_clean = 3 (mezi min a strong) NEBO check 0.031 mag - mereni plati, ale chyba je vetsi/hure overitelna; v AAVSO NOTES bude trust=YELLOW s duvodem.",
 "RED: n_clean = 0 (comp QA vsechno vylucila), NEBO check 0.07 mag, NEBO lc_quality mimo {good, noisy}, NEBO >= 3 soft priznaky - krivku necist jako vedu, cist diagnostiku (co se te noci stalo?).",
 "Zapamatujte: vysoky lc_rms SAM O SOBE trust nesnizuje - V0349 Dra ma lc_rms 0.155 mag (skutecna zakrytova amplituda!) pri check-star 0.0054 mag -> GREEN. Trust meri kvalitu MERENI, ne klidnost hvezdy."]))
st.append(P("13.4 Sparse trust (Howell 1988) a kontrolni statistiky", H2))
st.append(P("Na ridkych polich, kde klasicky ensemble-based trust nema dost vstupu, nastupuje sparse_trust_core.py dle Howell, Warnock & Mitchell (1988): [b]T-statistika[/b] = pomer merene a predpovezene sigmy check star (GREEN do sparse_trust_T_green=1.5; RED nad T_red=4.0) a [b]X2 nadmerny rozptyl[/b] (excess variance; RED nad sparse_trust_X2_RED=4e-4). CI-based pasma zohlednuji maly pocet epoch (minimum check_star_min_epochs=5).", B))

st.append(P("[b]catalog_match_mode (per epocha):[/b] vlajka jak dobre snimek 'sedi' na katalog pole - typicky good (plna shoda), degraded (malo shod / velka rezidua: vitr, mrak, okraj) az failed. Trust gate epochy s degradovanou shodou zapocitava do soft priznaku; body zustavaji v CSV (mask-first). [b]Sidecar comp_qa_{id}.json:[/b] per komp G_IQR, eta (von Neumann), podil vypadku, locus prahy binu, verdikt clean/flagged + duvod - kompletni podklad, proc n_clean vyslo, jak vyslo.", B))
st.append(P("13.5 Crowding index a limitni magnituda (volitelne)", H2))
st.append(P("crowding_index.py (crowding_classifier_enabled=false; standalone, side-effect-free - cte jen existujici artefakty) pocita: [b](a)[/b] metriky blendu per hvezda - vzdalenost nejblizsiho souseda nn_dist; hvezda je is_blended pri nn_dist < 1.5 x FWHM (vstup pro budouci adaptivni PSF/apertura routing); [b](b)[/b] limitni magnitudu pole - kde medianove SNR klesa na 5 (faint-side crossing, interpolace v log10(SNR) z merene krivky, pripadne analyticky z tehoz SNR modelu jako optimalni apertura). Limitni magnituda rika, kam az ma smysl merit a hledat promenne; od teto verze report porovnava limit s merenou hloubkou pole (G_lim_90, SNR5) a varuje pri nesouladu; plna automatika limitu zustava FUTURE. Zapnuti klasifikatoru ceka na data z husteho pole Newtonu (stejny gate jako PSF).", B))
st.append(P("13.6 Lunarni kontext (volitelne QA)", H2))
st.append(P("lunar_context.py (astropy efemeridy, offline) pro stred noci pocita pozici Mesice (vyska/azimut), fazi (% osvetleni) a uhlovou separaci Mesic-pole. Z toho 'lunar risk' (prvni platne pravidlo vyhrava): Mesic pod obzorem -> LOW; nov (faze < 10 %) -> LOW; velka faze a mala separace -> HIGH (mesicni svit silne zveda pozadi a sum zejmena u slabych cilu); jinak stredni. Slouzi jako QA metadata noci v reportu.", B))
st.append(fn("comp_qa_core.py: compute_comp_qa, build_locus, locus_at (IQR norm 1.349, locus median + 4 MAD); check_star_kmag.py: select_check_star, compute_check_ensemble_mag_calib; trust_flag_core.py: compute_trust_for_photometry_dir, CompTrustThresholds (soft 0.02, hard 0.05); sparse_trust_core.py; catalog_match_trust.py; crowding_index.py: compute_crowding_index; lunar_context.py: get_jd_midpoint a risk pravidla"))
st.append(par("comp_qa_enabled=true, comp_trust_min_comps=3 (strong=5), check_star_min_epochs=5, check_select_rms_floor=1e-4, sparse_trust_T_green=1.5, sparse_trust_T_red=4.0, sparse_trust_X2_RED=4e-4, trust_flag_enabled=true, lc_quality_min_frames=20 (short track 3), lc_quality_min_normal_frac=0.5, crowding_classifier_enabled=false (blend 1.5 FWHM, tighten threshold 0.04)"))

# ============================================================ CH 14 variability + TESS
st.append(P("14. Detekce promennych a overeni kandidatu", H1))
st.append(P("Ucel: najit v poli nove/podezrele promenne a overit je proti katalogum a TESS datum. Filozofie: konzervativni prahy, clovek potvrzuje - kandidat jde do reportu s plnym kontextem, ne do automatickeho ohlaseni.", B))
st.append(P("14.1 Detekce kandidatu (RMS obalka + VDI)", H2))
st.append(P("[b](1) Matice fluxu pole[/b] (load_field_flux_matrix): normalizace per snimek, filtry pokryti (variability_min_frames=30 a min_frames_frac=0.5), sigma clipping (5.0), percentilovy filtr nejhlucnejsiho chvostu (p85), deduplikace. [b](2) RMS obalka ('hockey-stick'):[/b] RMS jako funkce log-magnitudy se prolozi robustnim modelem (build_rms_mag_model) - obalka prirozeneho sumu pole; kandidat = hvezda s RMS vyznamne nad lokalnim medianem sve jasnosti: prebytek >= variability_sigma_threshold=2.3 sigma A zaroven >= variability_comp_floor_factor=1.5 x sum komparacniho dna. [b](3) Doplnkove filtry:[/b] minimalni amplituda 0.01 mag, smoothness strop 0.8 (velmi hladke krivky jsou spis trendy/artefakty nez hvezdna variabilita), slope floor 0.02, podil prezivsich bodu po clippingu >= 0.8. [b](4) VDI[/b] (compute_vdi; von Neumann 1941): z-skore pomeru stredni kvadraticke naslednosti - koreluje po sobe jdouci body a odlisi koherentni krivku od bileho sumu; prah variability_vdi_z_threshold=3.0; citlive na rychle promenne. Kandidat nesmi byt VSX ani aktivni cil. Vystup: catalog_id, RA, Dec, mag, RMS, n_frames; v UI interaktivni hockey-stick (Plotly), tabulka kandidatu s checkboxy a 'Pridat do VAR'.", B))
st.append(P("14.2 Crossmatch katalogu a TESS overeni", H2))
st.append(P("Pro kandidaty bezi crossmatch (VSX, Gaia DR3 variable, SIMBAD, ASAS-SN, ZTF, ATLAS, CSS...; crossmatch_runner.py). Kandidat bez shody v zadnem variabilitnim katalogu jde do TESS overeni (tess_verify.py pres lightkurve; tess_enabled=false default - vyzaduje internet): stazeni dostupnych sektoru a hledani periody az ctyrmi metodami s konsenzem:", B))
st.append(tab([
 ["Metoda","Princip"],
 ["Lomb-Scargle","periodogram nerovnomerne vzorkovane rady (Lomb 1976; Scargle 1982); zakladni stopa + astropy varianta jako volitelna 4. stopa"],
 ["PDM","Phase Dispersion Minimization (Stellingwerf 1978) - minimalizace rozptylu ve fazovych binech; robustni pro nesinusove tvary (zakryty)"],
 ["BLS","Box Least Squares (Kovacs, Zucker & Mazeh 2002) - obdelnikove zakryvy/tranzity"],
 ["Konsenzus","perioda, kde se shodnou >= 2 metody do 5 % tolerance (priorita LS+PDM > LS+BLS > PDM+BLS); u velmi kratkych period (< 0.15 d) mirnejsi prah 12 % pro PDM+BLS; harmonic refine doladi nasobky; jinak fallback na L-S"],
], [30,138]))

st.append(P("Mechanika TESS vstupu: pro cil se stahnou dostupne sektory (27denni pozorovaci useky; kadence 2 min ci FFI), krivka se ocisti (orez okraju sektoru _delete_error, iterativni sigma-clip, detrend s dynamickou delkou okna dle hledane periody - okno se voli tak, aby NEVYHLADILO samotnou periodu: _get_optimal_window / _dynamic_window_length) a kazdy sektor se analyzuje zvlast (TessSectorResult) pred slucenim. Aperturni parametry na TPF se voli dle jasnosti (_get_aperture_params). Konsenzus pres sektory + pres metody je dvojita pojistka proti aliasum (1denni aliasy pozemnich dat TESS nema, ale ma vlastni - orbitalni ~13.7 d systematiku).", B))
st.append(P("Vystup: period_consensus + period_method_used, fazovane grafy z konsenzualni periody (P a 2P) a hodnoceni spolehlivosti (_assess_period_reliability). [b]Vyhrada blendingu:[/b] TESS pixel ma 21 arcsec - blend check (_generate_tess_blend_check_png) vykresli okoli cile z MASTERSTARu s TESS aperturou, aby bylo videt, kdo vsechno do ni svitit.", B))

st.append(P("[b]Crossmatch zdroje:[/b] lokalne VSX a Gaia DR3 (vc. priznaku variability phot_variable_flag), online (je-li povoleno) SIMBAD, ASAS-SN, ZTF, ATLAS, CSS/CRTS pres crossmatch_runner. Vysledek per kandidat: nejblizsi protejsek, separace, typ, perioda z katalogu (je-li) - vse do strany kandidata v reportu. Kandidat s katalogovou shodou NENI 'objev', ale uzitecna detekce zname promenne (potvrzuje citlivost pipeline).", B))
st.append(fn("variability_detector.py: load_field_flux_matrix, compute_rms_variability, compute_vdi; photometry_core.py: build_rms_mag_model, expected_rms_from_model, auto_export_variability_candidates_csv; ui_variability.py (hockey-stick UI); crossmatch_runner.py, catalog_crossmatch.py; tess_verify.py: run_tess_analysis, _find_period(_pdm/_bls/_anova), _period_consensus, _harmonic_refine_period, _generate_tess_blend_check_png"))
st.append(par("variability_sigma_threshold=2.3, variability_comp_floor_factor=1.5, variability_min_amplitude_mag=0.01, variability_min_frames=30 (frac 0.5), variability_min_points_rms=20, variability_sigma_clip=5.0, variability_p85_filter=85, variability_smoothness_max=0.8, variability_vdi_z_threshold=3.0, variability_slope_floor=0.02, variability_clip_ratio_min=0.8, variability_mag_limit=14.5, tess_enabled=false"))

# ============================================================ CH 15 outputs
st.append(P("15. Vystupy: PDF report a exporty (AAVSO, VarAstro)", H1))
st.append(P("15.1 SUMMARY MEASURE REPORT (PDF)", H2))
st.append(P("pdf_report.py + photometry_report.py (reportlab). Struktura: [b]titulni strana[/b] - souhrnne metriky noci (pocet cilu, sestava, nejlepsi/nejhorsi lc_rms, prumerne BP-RP, RMS hockey-stick s vyznacenim kandidatu a znamych VSX, lunarni kontext); [b]strana na cil[/b] (A4 na sirku) - svetelna krivka s chybami a trust barvami, vyrez pole, tabulka komparaci (BP-RP, vzdalenost, pocet snimku, kvalita), poznamka k barve a trust badge; vyska grafu je dynamicka dle poctu radku komparaci (garance nepreteceni); [b]strana kandidata[/b] (bez shody ve variabilitnim katalogu) - crossmatch souhrn, raw LC, pripadna TESS sekce (sektory, perioda, metoda, fazovane grafy P a 2P, blend check); [b]HRD pole[/b] - barevny HR diagram (15.3); [b]konfiguracni strana[/b] - PLNY config snapshot + Resolved Facts (hodnoty vyresene z DB/FITS za behu vc. zdroje) - kazde cislo v reportu je zpetne dohledatelne; [b]zaverecna strana[/b] - tabulka vsech hvezd.", B))
st.append(P("15.2 Export AAVSO", H2))
st.append(P("AAVSO Extended Format: TYPE=EXTENDED, ensemble-standardizovana mereni, observer code (aavso_observer_code; napr. UMIA - varovani se emituje JEN kdyz je kod prazdny), identity check a comp hvezd. KMAG = MERENA ensemble-standardizovana magnituda check star (check star vyloucena z vlastniho ensemble; kap. 13.2). Mapovani filtru je tabulkove pres resolve_aavso_filt_from_obs_group (OSC kanaly R/G/B -> TR/TG/TB; oneRGGB internal-only bez exportu; mono NoFilter -> CV pres aavso_filter_map): nezname filtry davaji #WARNING, zadne tiche 'CV'. [b]OSC-3 (2026-07):[/b] comp/check katalogove magnitudy pro TG/TB/TR exporty jsou Gaia G+BP-RP -> Johnson V/B/Cousins R_C (gaia_johnson.py, koeficienty Gaia DR3 CU5 Table 5.9; validace Ruelas-Mayorga et al. 2025 RASTI); mimo validity se comp vylouci s logem. NOTES obsahuji trust=UROVEN... (< 100 znaku). Cas: HJD/BJD s heliocentrickou/barycentrickou korekci (Eastman et al. 2010). Sigma v exportech OBSAHUJE sigma_sys dno (11.4). Validator scripts/validate_aavso_export.py kontroluje format pred odeslanim.", B))
st.append(P("Citacni hlavicka AAVSO je SLIM (DECISIONS EXPORT-HEADER-SLIM): bez nepodminenych bloku [CORE]/[CATALOGS & TIME]/[SOFTWARE]/[FIELD ASTROPHYSICS]. Misto nich: (1) ASCII METHODS MATRIX (this run) ON/OFF radky ze stejneho flag->method mapovani jako citations.py; (2) `# [METHODS - this run]` jen pro metody ktere jsou ON; (3) pointer `# Full algorithm references: SUMMARY MEASURE REPORT (PDF)`. Plny citacni blok + stejna matice zustavaji v SUMMARY MEASURE REPORT PDF.", B))

st.append(tab([
 ["AAVSO pole","Hodnota ve VYVAR"],
 ["NAME","nazev cile (VSX/AUID preferencne)"],
 ["DATE","HJD (pripadne BJD_TDB dle nastaveni; Eastman 2010)"],
 ["MAG / MERR","mag_calib_final / err_total (VC. sigma_sys dna)"],
 ["FILT","AAVSO kod z resolve_aavso_filt_from_obs_group (OSC R/G/B -> TR/TG/TB; oneRGGB neexportovano; mono NoFilter -> CV; neznamy -> #WARNING)"],
 ["TRANS","NO (apply_color_term='off'; pri CT by bylo YES)"],
 ["MTYPE","STD (ensemble-standardizovane)"],
 ["CNAME / KNAME","'ENSEMBLE' / identifikator check star"],
 ["KMAG","MERENA ensemble-standardizovana magnituda check star (13.2)"],
 ["AIRMASS","per bod (Kasten & Young 1989)"],
 ["NOTES","trust=UROVEN[;duvody] + poznamky; < 100 znaku"],
], [34,134]))
st.append(P("15.3 HRD pole (Gaia GSP-Phot)", H2))
st.append(P("hrd_analysis.py (build_hrd_dataframe): x = BP-RP, y = absolutni magnituda M_G = G + 5 - 5 log10(d) (vzdalenost z distance_gspphot / paralaxy; filtr paralaxy >= 0.15 mas a SNR >= 5), barva bodu = teff (Gaia GSP-Phot, Andrae et al. 2023). Prirazuje spektralni tridu (z teff, fallback z BP-RP) a tridu svitivosti (z logg). hrd_colorfield.py vykresluje barevne pole s chroma boostem (2.2) a white-pointem na medianu pole; hrd_enrich.py volitelne obohacuje zajimave objekty online (SIMBAD, Gaia TAP; hrd_online_enrich_enabled=true, max 20 kandidatu, timeout 20 s). Zvyraznuji se kandidati, VSX, exoplanety, pripadne NSS (hrd_nss_category_enabled=false).", B))

st.append(P("Soucasti reportu je i metodicka sekce: strucny popis pouzitych metod s citacemi generovany PODMINENE dle behu (citations.py) - bezel-li PyTICS, cituje se Marconi et al.; nebezel-li SysRem, Tamuz se neobjevi. Report je tak sam o sobe korektni metodickou prilohou pro pripadnou publikaci.", B))
st.append(P("15.4 Export VarAstro / B.R.N.O. a smerovani", H2))
st.append(P("CSV/TXT format pro ceskou sekci; obsahuje # Trust: ... hlavicku. Smerovani vysledku: zakrytove dvojhvezdy -> VarAstro (LC), pulzujici a ostatni -> AAVSO. Citacni hlavicka je stejne SLIM jako u AAVSO (METHODS MATRIX + podminene [METHODS - this run] + pointer na SUMMARY MEASURE REPORT PDF); plne reference zustavaji v PDF. CITATIONS.bib je kanonicky zdroj; emise podminene dle metod ktere skutecne bezely.", B))
st.append(fn("report_methods.py: aavso_export_path a export cesta; pdf_report.py, photometry_report.py (reportlab); gaia_johnson.py (OSC Gaia->Johnson comps); band_classify.py: guess_aavso_code_from_obs_group; hrd_analysis.py: build_hrd_dataframe; hrd_colorfield.py, hrd_enrich.py; citations.py; dev/scripts + scripts: validate_aavso_export.py; time_utils.py (HJD/BJD)"))
st.append(par("aavso_observer_code='UMIA', observer_code, observer_name, aavso_filter_map={}, hrd_color_field_enabled=true (chroma_boost 2.2, saturation 0.85, white_point 'field_median', parallax >= 0.15 mas, SNR >= 5), hrd_online_enrich_enabled=true (max 20, timeout 20 s), hrd_simbad_enrich_enabled=true"))

# ============================================================ CH 16 catalogs & DB
st.append(P("16. Katalogy a databaze", H1))
st.append(P("VYVAR je navrzen pro OFFLINE provoz - vsechny katalogy jsou lokalni: [b]Gaia DR3 SQLite[/b] (~9.4 GB, 40+ mil. hvezd; stavba GAIA_DR3/build_gaia_catalog.py, resumovatelne stahovani z ESA TAP; sloupce vc. parallax, parallax_error, teff/logg/mh/distance_gspphot pro HRD), [b]blind indexy[/b] fine + wide (build_blind_index.py z Gaia DB; geometricke invarianty pro slepe reseni), [b]VSX[/b] (vsx_make.py; index znamych promennych; Watson, Henden & Price 2006), [b]exoplanety[/b] (exoplanet_make.py; lokalni NASA Exoplanet Archive, 14k+ radku). Barva je ciste Gaia BP-RP: po dokoncenem vyrazeni Johnsonova B-V (DECISIONS 2026-06-25) byly konverze bp_rp_to_bv / teff_to_bv i tabulky APASS/Tycho odstraneny; vyber komparaci je BP-RP-nativni. Cile bez Gaia protejsku padaji do tieru 'neznama barva'; Gaia DR4 je planovane zlepseni pokryti.", B))

st.append(P("[b]Stavba Gaia DB (jednorazove, offline potom):[/b] build_gaia_catalog.py stahuje po deklinacnich pasech z ESA TAP (resumovatelne - preruseni neztraci praci), sklada SQLite s prostorovym indexem (dlazdice RA/Dec) pro rychle kuzelove dotazy (catalog_query_max_rows=15000 strop na dotaz chrani pamet na velmi hustych polich). Sloupce zahrnuji astrometrii (ra, dec, pmra, pmdec, parallax + chyby), fotometrii (G, BP, RP, BP-RP), GSP-Phot (teff, logg, mh, distance) a priznaky (NSS, extended object, variability). Analogicky VSX (vsx_make.py z AAVSO exportu) a exoplanety (exoplanet_make.py z NASA Exoplanet Archive). Blind indexy (fine/wide) se predpocitavaji z Gaia DB jednou per FOV trida.", B))
st.append(P("Provozni SQLite databaze (database.py): observator (LOCATION / TELESCOPE / EQUIPMENTS - u NOVEHO uzivatele PRAZDNA; referencni sada autora je harness-only seed v dev/tools/reference_seed.py, do produkcni DB se nikdy nesadi), knihovna kalibraci (registrace masteru; dark bez konecne CCD_TEMP se odmitne), drafty a jejich stav, katalogove cache. Vestigialni tabulka SETTINGS byla odstranena (WAVE-B): jedinym zdrojem nastaveni je config.json + DB referencni tabulky + FITS resolved hodnoty.", B))
st.append(fn("database.py; GAIA_DR3/build_gaia_catalog.py, GAIA_DR3/build_blind_index.py; VSX/vsx_make.py; exoplanets/exoplanet_make.py; dev/tools/reference_seed.py (harness-only)"))

# ============================================================ CH 17 config & repro
st.append(P("17. Konfigurace, parametry a reprodukovatelnost", H1))
st.append(P("17.1 Tri druhy hodnot a registr parametru", H2))
st.append(P("(Opakovani z kap. 3.2, protoze je to pater systemu:) [b]nastaveni[/b] v config.json, [b]staticka fakta[/b] v DB, [b]dynamicke hodnoty[/b] z FITS za behu. config.json je generovany, skupinovany a KOMENTOVANY soubor (JSONC-lite: '//' radkove komentare povoleny), editovatelny rucne bez UI; loader toleruje komentare a varuje na nezname klice s navrhem nejblizsiho spravneho (difflib); validator dev/scripts/validate_config.py hlasi syntaxi, nezname klice, typy a rozsahy. Persistuje 249 klicu. Registr parametru (dev/validation/params_registry.json, 269 zaznamu po WAVE-B redukci z 304) je JEDINY zdroj metadat: napoveda per klic (prenesena z CONFIG_GUIDE), tier (basic/advanced/expert/dev), clampy, parita config-UI-registr je vynucovana testy. Nove parametry musi paritu respektovat (VYVAR_PROCESS.md).", B))
st.append(P("17.2 Gating a citace", H2))
st.append(P("Nove chovani se VZDY zavadi za config priznakem s konzervativnim defaultem (OFF, dokud neni prokazano, ze je lepsi). Tyz priznak ridi citace: report cituje jen metody, ktere skutecne bezely. Machine-enforced invarianty (VYVAR-INVARIANTS, DECISIONS 2026-07-16) hlida testova sada - napr. ze read-only faze nemeni fotometrii, ze validacni ledger nelze tise smazat, ze pocitadla except-fixu jsou nulova.", B))
st.append(P("17.3 Anchor, ritual a reprodukovatelnost", H2))
st.append(P("Vedecke jadro jisti [b]bajtove identicky anchor draft_435[/b]: SHA-256 core 03d8fb64... (n=333 souboru) a extended bbfcc92e... (n=499) nad lightcurves / comp_quality / comparison_stars / exporty. Pravidla: read-only zmeny MUSI byt bitove identicke; zmeny menici vedu vyzaduji ohraniceny, vysvetlitelny diff (vedecky komparator s numerickou toleranci ~1e-6, mimo provenance sloupce). Anchor se nikdy nelockuje bez potvrzene reprodukovatelnosti (dva nezavisle cerstve behy bajtove identicke - DECISIONS 2026-06-11). Sessionovy ritual: [b]--fast[/b] pri kazde sessi (git stav, sanity cest, plny pytest ~907 testu, ledger hinty) a [b]--full[/b] (~45 min: headless beh pipeline + vedecky komparator proti anchor SHA + zero-assert except pocitadel) pred pushem vedeckych zmen - hned prvni ostry beh --full odhalil produkci bug, coz je presne jeho prace. Zavislosti drzi DEPS_POLICY.md (ctvrtletni gated cyklus; pinovane photutils 3.0.0, astropy 8.0.x, numpy 2.4.4+).", B))


st.append(P("[b]Prepinani anchoru (kdyz se veda zmeni zamerne):[/b] (1) zmena projde --full komparatorem s vysvetlitelnym diffem (ktere sloupce, proc, o kolik); (2) dva nezavisle cerstve behy noveho stavu byte-identicke; (3) teprve pak se re-cutne anchor SHA (draft_435 vznikl presne timto postupem koherentne pres run_full_photometry_pipeline). [b]Zavislosti (DEPS_POLICY):[/b] ctvrtletni gated cyklus - upgrade knihoven se testuje proti anchoru jako kazda vedecka zmena; pinovane verze (photutils 3.0.0, astropy 8.0.x, numpy 2.4.4+) jsou soucasti reprodukovatelnosti. Behove prostredi je tak drzeno stejne prisne jako kod.", B))
st.append(P("[b]Prakticky postup zmeny parametru:[/b] (1) najdete klic v config.json - komentar nad nim rika, co dela (generovano z registru; plne vysvetleni v CONFIG_GUIDE / parameter handbooku); (2) upravte hodnotu (radkove '//' komentare jsou povoleny, trailing carky ne); (3) [c]python dev/scripts/validate_config.py[/c] - chyti preklepy klicu (navrhne nejblizsi spravny), typy i rozsahy; (4) spustte draft; (5) config snapshot v reportu dokumentuje, s cim beh probehl. Ulozeni z UI soubor regeneruje - vlastni komentare se neprezivaji.", B))
st.append(P("[b]Priklady machine-enforced invariantu:[/b] trust/QA faze nemeni fotometrii (bitova identita vystupu); validacni ledger nelze zmensit bez explicitniho zaznamu (guard proti tichemu mazani); except-fix pocitadla == 0 v --full; config-UI-registr parita (zadny klic bez napovedy a tieru); LC schema drzi povinne sloupce (Priloha C). Poruseni kterehokoli = cerveny test, ne 'warning'.", B))
st.append(fn("dev/scripts/validate_config.py; dev/validation/params_registry.json (269), VYVAR_VALIDATION_LEDGER.json (+ guard test); dev/tools: session_baseline_check.py (--fast/--full); config.py (JSONC-lite loader/writer); params_registry.py"))

# ============================================================ CH 18 dev process
st.append(P("18. Vyvojovy proces a garance kvality (pro duveru uzivatele)", H1))
st.append(P("Tato kapitola neni o algoritmech, ale o tom, PROC verit cislum: jak se VYVAR vyviji, aby se vedecka spravnost nerozpadla pod zmenami.", B))
st.append(P("[b]Testy:[/b] 900+ testovych funkci (142 souboru) vcetne regresnich guardu na tridy chyb, ktere se uz jednou staly. Priklad tridy PROC_STORE_COLS: sloupec se spocita, ale nezapise do proc CSV - nalezeny dva nezavisle pripady (catalog_match_mode; petice psf_* sloupcu), oba opraveny a trida dostala systemovy guard (--full komparator + testy). Pravidlo dvou oprav: opakovana chyba stejne tridy dostava systemovou pojistku, ne dalsi per-instance zaplatu.", B))
st.append(P("[b]Except triaz:[/b] ~625 tichych exception handleru proslo radkovou triazi s tier klasifikaci (T1-SCIENCE az T4-LEGIT); vedecky relevantni umlcovani chyb bylo odstraneno, pocitadla except-fixu jsou soucasti --full zero-assertu. Grounded fakt pouzivany pri triazi: log_event je interne guardovany a nikdy nevyhazuje - ciste log_event guardy jsou prokazatelne mrtvy kod.", B))
st.append(P("[b]Retrakcni disciplina:[/b] negativni vysledky se explicitne vlastni a dokumentuji (retrahovany '83.1% Brno fix' - nevalidovan na produkci ceste; odmitnuty comp redesign se 45 % regresi; skodlive binovani ALG-3). Kalibracni pravidlo: fyzika/literatura = spolehlive; kod/runtime = overit a znacit jako hypotezu, dokud neprojde produkcni cestou.", B))
st.append(P("[b]Dokumentacni rodina:[/b] STATE (aktualni stav), DECISIONS (rozhodnuti + zduvodneni + negativni vysledky), ROADMAP (otevrena prace), PROCESS (jak se pracuje), PARAMS/handbook (parametry), JOURNAL (denik). Tento dokument (FLOW) je referencni popis toku a regeneruje se builderem pri kazde vetsi zmene pipeline (docs-revision ritual).", B))



# ============================================================ CH 18c limits
st.append(P("Omezeni a zname hranice", H1))
st.append(P("Poctivy popis toho, co VYVAR nedela nebo kde ma zname meze - vetsina je vedomym rozhodnutim (viz DECISIONS):", B))
st.append(P("[b]Jedna noc jako jednotka.[/b] Multi-night globalni zeropoint (spojovani noci na spolecnou skalu) je odlozen; noci se porovnavaji pres katalogovou kalibraci, coz pro dlouhoperiodicke zmeny s amplitudou pod ~2 sigma_sys nemusi stacit. [b]Periodova analyza vlastnich dat[/b] neni produkt (TESS periody slouzi jen overeni kandidatu). [b]Astrometrie neni veda vystupu[/b] - WCS slouzi identifikaci, ne mereni poloh. [b]Jasna mez:[/b] saturovane cile se nemeri (zadny 'saturation repair'); prakticka mez wide rigu ~G 9-10 pri 60 s. [b]Slaba mez:[/b] G_lim_90 pole (DAO-RECONCILE) - typicky ~15 na wide, hloubeji na Newtonu; pod ni rychle roste nekompletnost i sum. [b]NoFilter:[/b] bez transformace do standardniho systemu (CV pseudopasmo); k'' korekce je literaturni, per-nocni fit ceka na data. [b]PSF vetev:[/b] pripravena, nevalidovana - OFF (kap. 12.6). [b]Husta pole:[/b] do validace PSF/crowding klasifikatoru plati aperturni cesta s dilucnimi vlajkami - extremni pole (stred kulove hvezdokupy) jsou za hranici soucasne produkce. [b]Internet:[/b] TESS overeni a HRD/SIMBAD obohaceni vyzaduji sit (jedine online kroky; oba volitelne).", B))
# ============================================================ CH 18b worked night
st.append(PageBreak())
st.append(P("Pruvodce: jedna noc od zacatku do konce (ilustracni pruchod)", H1))
st.append(P("Nasledujici pruchod sleduje realisticky scenar na wide rigu (Zeiss 200 mm, QHY294MM, NoFilter, bin2, 60 s expozice) - cisla jsou ilustracni, ale radove odpovidaji produkci. Ukazuje, ktera kapitola 'pracuje' v kterem okamziku.", B))
st.append(P("[b]21:40 - Import.[/b] 214 lights (NoFilter_60_2) + knihovna masteru. Importer paruje master dark (60 s, -10 C, |dT| = 0.3 C < 0.5) a flat (NoFilter, stari 41 dnu < 200). Draft zalozen, obs_group NoFilter_60_2. (Kap. 3, 4.2)", B))
st.append(P("[b]Kalibrace.[/b] CAL-DIAG (a): median darku vs svetla v toleranci 2 % - OK. Odecet + deleni flatem; (b) obloha po odectu 208 ADU/px, v mezich. BPM z darku: 1 912 pixelu (5 sigma MAD). Sky-surface rad 2 odstranil gradient ~6 ADU pres pole (Mesic 38 % nizko na JZ). (Kap. 4)", B))
st.append(P("[b]QC.[/b] Median FWHM noci 3.6 px -> auto limit 5.4 px (k=1.5). 3 snimky elongace > 1.8 (naraz vetru, vy_qc=warn), 1 snimek 4 hvezdy (mrak, fail). Zadna brana nevyrazuje (default OFF) - vlajky nesou snimky do fotometrie, kde se s nimi pracuje bod po bodu. (Kap. 5)", B))
st.append(P("[b]Zarovnani + MASTERSTAR.[/b] Referencni ramec #96 (nejvic hvezd, FWHM 3.3). Astroalign: median rezidua 0.24 px, max drift pres noc 0.4 px. MASTERSTAR z best-of-10; hintovane WCS: 496 shod s Gaia, recovery 0.81, RMS stredu 0.6 px, SIP rad 3, odds test PASS. Katalog pole: 1 108 detekci / 934 Gaia matchu -> hustota 'normal'. VSX: 3 zname promenne v poli; exoplanety: 0. (Kap. 6, 7)", B))
st.append(P("[b]Per-frame katalogy.[/b] 213 pouzitelnych snimku x plna DAO detekce; parovani na master; jd/hjd/bjd + airmass (X 1.12 -> 1.87 pres noc); catalog_match_mode: 211x good, 2x degraded (vetrne snimky). DAO-RECONCILE: G_lim_50 = 15.8, G_lim_90 = 15.1, miss@G90 = 1.4 %. (Kap. 8)", B))
st.append(P("[b]Faze 0 + 1.[/b] Cile: 3 VSX + 1 rucni = 4 aktivni. Globalni pool: 156 kandidatu po statickych filtrech, RMS mapa hotova. Per cil kaskada (10.1-10.3) da 6-8 kompu; u nejjasnejsiho cile (G=10.9) jen 4 kompy (bright floor) -> ceka YELLOW. Check stars vybrany. (Kap. 9, 10)", B))
st.append(P("[b]Faze 2A.[/b] SNR tabulka (FWHM 3.6, obloha 208): slabe hvezdy r ~ 3.1 px, jasne r ~ 6.8 px. Empiricke pozadi: 64 prazdnych apertur/snimek, sigma_bkg_ap ~ 1.12x Howell (kovariance resamplingu). Aperturni korekce: Delta M_corr = -0.021 mag (5 T1 referenci, scatter 0.011 < 0.03, ok). k'': CV pasmo, k2 = 0.030, korekce kompu aplikovana. Ensemble: PyTICS 5 iteraci, iterativni clip vyradil 1 komp u cile #2 (eta = 1.1, drift). CT: off. Vystup: 4x LC CSV + summary. (Kap. 11)", B))
st.append(P("[b]Trust + variabilita.[/b] comp_qa n_clean: 6/5/7/3 -> trusty GREEN/GREEN/GREEN/YELLOW(tenky ensemble). Check scattery 6-9 mmag. Hockey-stick: 1 kandidat 2.9 sigma nad obalkou, VDI z=3.4, neni ve VSX -> crossmatch (nic) -> TESS: 2 sektory, LS+PDM konsenzus P = 0.3121 d (EW?); blend check cisty. (Kap. 13, 14)", B))
st.append(P("[b]Rano.[/b] Report: 4 cile + 1 kandidat + HRD; export AAVSO (3 cile GREEN; YELLOW cil s trust poznamkou v NOTES - rozhodnuti na cloveku), zakrytovy cil -> VarAstro. (Kap. 15)", B))


# first-draft guide
st.append(P("Prvni draft na vlastnich datech: co zkontrolovat", H1))
st.append(P("[b](1) Kalibrace:[/b] CAL-DIAG bez varovani; obloha po odectu kladna a rozumna (stovky ADU dle Mesice). [b](2) WCS:[/b] pocet Gaia shod a recovery v reportu; RMS stredu ~1 px a mene. [b](3) QC:[/b] median FWHM odpovida ocekavani rigu; elongace pod 1.3-1.4 u dobre noci. [b](4) Check star:[/b] NEJDULEZITEJSI JEDNO CISLO - nocni rozptyl check star je realna presnost vaseho rigu (wide: ocekavejte ~5-15 mmag na jasnych; sigma_sys se casem kalibruje prave z tohoto). [b](5) Trust:[/b] GREEN na znamych stabilnich cilech; YELLOW ctete s duvodem. [b](6) G_lim_90:[/b] hloubka pole (8.3) - diagnostika dosaznosti detekce; VSX scope je automaticky (DAO+Gaia). [b](7) Srovnani:[/b] mate-li AIJ/SIPS mereni tehoz cile, porovnejte TVAR krivky (shoda < ~1 mmag ve strednich rozdilech je ocekavana; velikost chybovych usecek se lisit BUDE - viz FAQ).", B))
# ============================================================ FAQ
st.append(P("Caste otazky (FAQ)", H1))
st.append(P("[b]Proc jsou chyby VYVARu vetsi nez v AIJ/SIPS na stejnych datech?[/b] Protoze VYVAR pocita POCTIVEJI: empiricke pozadi (zachyti korelovany sum resamplingu, ktery teoreticky vzorec nevidi), SEM ensemble zeropointu a hlavne sigma_sys dno kalibrovane z check hvezd. Bodova chyba 6 mmag pri nocnim check-star rozptylu 18 mmag je sebeklam - VYVAR ho odmita. Srovnavejte rozptyl check star, ne velikost chybovych usecek.", B))
st.append(P("[b]Proc mi vybral 'horsi' kompy nez bych vybral rucne?[/b] Nejcasteji kvuli barve: krasne stabilni, ale barevne vzdaleny komp prohrava s mirne sumnejsim barevne bliznim - ochrana proti k'' systematice ma prednost (10.2, box). Druhy duvod: VSX purge - hvezda, kterou znate jako 'stabilni', muze byt katalogova promenna.", B))
st.append(P("[b]Proc je casove binovani vypnute, kdyz 'vyhlazuje krivku'?[/b] Protoze vyhlazuje i REFERENCI: injekci testy prokazaly, ze binovani kompu vtiskne do diferencialni krivky artefakty pruhlednosti (24/25 cilu horsich). Vyhlazeni na pohled neni totez co lepsi data. (11.12)", B))
st.append(P("[b]Muzu verit YELLOW mereni?[/b] Ano - YELLOW znamena 'plati, s vetsi/hure overitelnou chybou' (typicky tenky ensemble: sigma ~ 1/sqrt(N)). RED znamena 'necist krivku, cist diagnostiku'. Trust nikdy nemaze data; export je vase rozhodnuti. (13.3)", B))
st.append(P("[b]Proc nevidim transformovane (standardni) magnitudy?[/b] apply_color_term='off': NoFilter nema definovany cilovy system a u filtrovanych dat se CT zapina az pri spolehlivem fitu (>= 7 kompu, bez extrapolace). Ensemble-standardizovane magnitudy s MTYPE=STD jsou pro AAVSO korektni vystup. (11.9, 15.2)", B))
st.append(P("[b]Muj cil nema Gaia BP-RP - co to znamena?[/b] Padne do nejnizsiho barevneho tieru (neznama barva, vaha 0.25 pri vyberu kompu) a k'' korekce pouzije median kompu jako referenci. Mereni probehne; barevna ochrana je slabsi - v reportu to uvidite v poznamce k barve.", B))
st.append(P("[b]Proc beh trva tak dlouho?[/b] Plna DAO detekce na kazdem snimku + empiricke pozadi 64 apertur/snimek + PyTICS iterace. To je vedoma cena za QC a poctive chyby v nocnim davkovem modelu (1, zasada 4). Paralelizace se ridi RAM (per_frame_mp_reserve_ram_gb=1.5).", B))
st.append(P("[b]Co mam delat, kdyz report hlasi prosly master?[/b] Nafotit novou sadu darku/flatu a zaregistrovat do knihovny (Settings). Stare mastery zustavaji - knihovna je historicka; parovani si vzdy bere nejblizsi validni. (4.2)", B))
# ============================================================ CH 19 references
st.append(P("19. Vedecke reference", H1))
st.append(P("Kanonicky a vzdy aktualni zdroj citaci je CITATIONS.bib v repozitari; nize uvedeny seznam je s nim sesouladen. Citace v reportech se emituji podminene dle skutecne pouzitych metod.", B))
st.append(P("[b]Fotometrie a detekce:[/b]", B))
for r in [
 "Stetson P. B. 1987, PASP 99, 191 - DAOPHOT (detekce, tradice anulu).",
 "Stetson P. B. 1990, PASP 102, 932 - aperturni korekce / krivka rustu.",
 "Howell S. B. 1989, PASP 101, 616 - CCD rovnice, SNR, optimalni apertura.",
 "Merline W. J., Howell S. B. 1995, Exp. Astron. 6, 163 - realisticky sumovy model CCD (clen odhadu oblohy).",
 "Labbe I. et al. 2003, AJ 125, 1107 - empiricky sum pozadi z prazdnych apertur.",
 "Howell S. B. 2006, Handbook of CCD Astronomy, 2. vyd. - kontaminace apertury.",
 "Broeg C., Fernandez M., Neuhauser R. 2005, AN 326, 134 - vazeny soubor kompu (umela srovnavaci hvezda).",
 "Honeycutt R. K. 1992, PASP 104, 435 - ansamblova fotometrie, epochove korekce.",
 "Anderson J., King I. R. 2000, PASP 112, 1360 - empiricka (efektivni) PSF.",
 "Moffat A. F. J. 1969, A&A 3, 455 - Moffatuv profil.",
 "Fleming T. A. et al. 1995 - error-function model uplnosti detekce (DAO-RECONCILE).",
]: st.append(P(r, M))
st.append(P("[b]Statistika variability a kontrola kvality:[/b]", B))
for r in [
 "Sokolovsky K. V. et al. 2017, MNRAS 464, 274 - srovnani indexu variability.",
 "von Neumann J. 1941, Ann. Math. Stat. 12, 367 - pomer stredni kvadraticke naslednosti.",
 "Howell S. B., Warnock A., Mitchell K. J. 1988, AJ 95, 247 - kontrolni hvezdy, T-statistika, excess variance.",
 "Osborn J. et al. 2015, MNRAS 452, 1707 - scintilace (saturace zisku ensemble).",
]: st.append(P(r, M))
st.append(P("[b]Analyza period (TESS overeni):[/b]", B))
for r in [
 "Lomb N. R. 1976, Ap&SS 39, 447; Scargle J. D. 1982, ApJ 263, 835 - Lomb-Scargle.",
 "Stellingwerf R. F. 1978, ApJ 224, 953 - PDM.",
 "Kovacs G., Zucker S., Mazeh T. 2002, A&A 391, 369 - BLS.",
]: st.append(P(r, M))
st.append(P("[b]Extinkce, barvy a cas:[/b]", B))
for r in [
 "Kasten F., Young A. T. 1989, Appl. Opt. 28, 4735 - airmass model.",
 "Hardie R. H. 1962, in Astronomical Techniques - extinkce 1. radu (kontext).",
 "Henden A. A., Kaitchuck R. H. 1982, Astronomical Photometry - barevny clen, vyber komparaci, k'' Johnson B.",
 "Smith J. A. et al. 2002, AJ 123, 2121 - Sloan fotometrie (nativni k'' koeficienty).",
 "Jordi C. et al. 2010, A&A 523, A48 - Gaia barevne transformace (konverze k'' do BP-RP).",
 "Riello M. et al. 2021, A&A 649, A3 - Gaia EDR3 fotometrie, BP-RP.",
 "Eastman J., Siverd R., Gaudi B. S. 2010, PASP 122, 935 - HJD/BJD korekce.",
 "Gaia Collaboration, Vallenari A., et al. 2023, A&A 674, A1 - Gaia DR3.",
 "Andrae R. et al. 2023, A&A 674, A27 - Gaia GSP-Phot (HRD).",
 "Watson C., Henden A., Price A. 2006, SASS 25, 47 - VSX.",
]: st.append(P(r, M))
st.append(P("[b]Systematiky a detrendy (vyhodnocene; vetsina vypnuta):[/b]", B))
for r in [
 "Tamuz O., Mazeh T., Zucker S. 2005, MNRAS 356, 1466 - SysRem (vyhodnoceno, OFF).",
 "Savitzky A., Golay M. J. E. 1964, Anal. Chem. 36, 1627 - SG filtr (ALG-2, opt-in OFF).",
 "Marconi et al. 2026, RASTI - PyTICS iterativni interkalibrace (ALG-5, ON).",
 "Democratic detrender, arXiv:2411.09753 - marginalizace pres modely (ALG-4, opt-in OFF).",
 "Seager S., Mallen-Ornelas G. 2003, ApJ 585, 1038 - faktor redeni (GS11, OFF).",
]: st.append(P(r, M))
st.append(P("[b]Software:[/b]", B))
for r in [
 "Astropy Collaboration 2013/2018/2022 - astropy (8.0.x).",
 "Bradley L. et al. - photutils (3.0.0; DAOStarFinder, aperture, EPSFBuilder, PSFPhotometry).",
 "Beroiz M., Cabral J. B., Sanchez B. 2020, A&C 32, 100384 - astroalign (princip zarovnani).",
 "Lightkurve Collaboration 2018 - TESS/Kepler casove rady.",
 "Pejcha O., Cagas P. et al. 2022, A&A 667, A53 - SIPS algoritmy (srovnavaci reference, neprebirano).",
]: st.append(P(r, M))

# ============================================================ Appendices

st.append(P("Rychla reference: parametry, ktere uzivatele ladi nejcasteji", H1))
st.append(P("(Vyber; plny popis vsech 249 klicu: parameter handbook. Sloupec 'Kdy sahat' je doporuceni, ne pravidlo.)", M))
st.append(tab([
 ["Klic (default)","Kdy sahat"],
 ["comp_max_delta_bprp (0.79)","uzsi na presnost pri dostatku kandidatu; sirsi na ridkych polich (radeji nechat density adaptaci)"],
 ["phase01_comparison_n_comp_max (8)","zvysovat nema smysl (saturace zisku ~6-8); snizeni jen pro experimenty"],
 ["aperture_fwhm_factor (1.35)","APERTURE-01d produkcni f; SNR tabulka je diagnostika"],
 ["masterdark/flat_validity_days (90/200)","dle discipliny fotografovani kalibraci"],
 ["auto_fwhm_k_factor (1.5)","prisnejsi (1.2-1.3) pro vyber jen spickovych snimku pri prebytku dat"],
 ["VSX scope","automaticky (DAO+Gaia detekce); parametr odstranen 2026-07"],
 ["frame_quality_gate_enabled (false)","zapnout az po overeni na vlastnim rigu; drzet min_keep_frames"],
 ["tess_enabled (false)","zapnout pri overovani kandidatu (vyzaduje internet)"],
 ["sigma_sys_mag ({'4': 0.018})","NEsnizovat 'aby chyby vypadaly lepe' - kalibruje se z check hvezd"],
 ["apply_color_term ('off')","'auto' jen pro filtrovana data s dostatkem kompu (>= 7)"],
], [58,110]))
st.append(P("Priloha A - struktura draftu na disku", H1))
st.append(P("[c]Archive/<pole>/<draft>/[/c] s podadresari: [c]calibrated/[/c] (kalibrovane a zarovnane FITS s vy_* QC hlavickami), [c]platesolve/<obs_group>/[/c] (MASTERSTAR.fits, masterstars_full_match.csv, masterstar_epsf.fits je-li stavena, per-frame proc_*.csv s toky/chybami/vlajkami vc. psf_* sloupcu a catalog_match_mode), [c]platesolve/<obs_group>/photometry/[/c] (active_targets.csv, comparison_stars_per_target.csv, photometry_summary.csv, lightcurves/ s LC CSV per cil + sidecary comp_qa_{id}.json a check_kmag_{id}.csv), [c]reports/[/c] (SUMMARY MEASURE REPORT PDF), [c]export/[/c] (AAVSO, VarAstro).", B))

st.append(P("Priloha C - sloupce svetelne krivky (lightcurves/*.csv)", H1))
st.append(P("Presny vycet sloupcu produkcniho LC CSV (save_lightcurve_csv); poradi orientacni, seskupeno dle vyznamu:", B))
st.append(tab([
 ["Skupina","Sloupce a vyznam"],
 ["Cas","bjd, hjd, jd (Eastman 2010), time_base (BJD_TDB vs JD_FALLBACK - znaci cestu prepoctu, hodnoty nemeni), airmass"],
 ["Magnitudy","mag_inst (instrumentalni), mag_calib (ensemble kalibrace), mag_calib_raw, mag_calib_ac (+AC), mag_calib_ct (+CT), mag_calib_final (KANONICKA publikovana, 11.10), delta_mag (vuci AIJ flux-sum ensemble), delta_mag_democratic (jen pri ALG-4)"],
 ["Chyby","err (err_total, 11.4), err_method, err_scatter_unmatched, err_inflation (ALG-4), sigma_sys_mag (pouzite dno)"],
 ["Aperturni korekce","ac_ok, ac_correction, ac_n_ref, ac_scatter"],
 ["Barevny clen","ct_ok, ct_correction, ct_c1, ct_n_comp, ct_bp_rp_target, ct_bp_rp_comp_med"],
 ["k''","k2_source (literature/fit/none), k2_value, k2_colour_ref (referencni barva kompu)"],
 ["Kvalita bodu","flag (normal/outlier/saturated...), method (aperture/psf per bod), catalog_match_mode, wcs_untrusted, alignment_failed, is_flipped (strana montaze - diagnostika meridian-flip kroku), dilution_factor, lunar_phase_pct, lunar_separation_deg, lunar_risk"],
 ["Provenience","source_file (zdrojovy snimek bodu), aperture_r_px"],
], [34,134]))
st.append(P("photometry_summary.csv (per cil): n_frames, n_saturated, lc_rms, lc_quality_flag (good/noisy/short/...; classify_lc_quality), n_clean (po comp QA), zone_flag, trust vstupy, lunar kontext, koeficienty RMS-mag modelu pole. Sidecary: comp_qa_{id}.json (indexy per komp), check_kmag_{id}.csv (KMAG rada check star).", B))
st.append(P("Priloha D - diagnostika castych situaci", H1))
st.append(tab([
 ["Priznak","Pravdepodobna pricina a kroky"],
 ["Plate solve selhal (hinted)","spatny hint z montaze -> pobezi blind; zkontrolujte RA/DEC v hlavicce. Blind selhal? Zkontrolujte cesty indexu (blind_index_*) a FOV rezim (auto)."],
 ["CAL-DIAG abort","zamena SUM/MEAN konvence ci cizi master: ctete hlaseni brany - autocorrect zapsal prepocet do provenance, nebo fail-closed uvadi pomer urovni. Overte expozici/teplotu masteru. (4.4)"],
 ["Malo kompu / YELLOW vsude","ridke pole ci prilis jasny cil (bright floor). Zkontrolujte density profil v reportu; zvazte comp_sparse_fallback (ON) a vedome sirsi mag diff - absolutni strop 3.0 plati vzdy."],
 ["Krivka ma 'schod' uprostred noci","meridian flip: sloupec is_flipped v LC CSV; schod = pozicne zavisla systematika (flat/vinetace). Viz DECISIONS V0454 CrA - egress + flip step. Nezamenovat s variabilitou."],
 ["Pomaly 'trend' pres noc koreluje s airmass","barevny rozdil cil-kompy (k'' zbytek): zkontrolujte ct_bp_rp_target vs komp median a k2_value v LC CSV; tesnejsi barevny vyber kompu je prvni krok. (11.8 box)"],
 ["Vsechny body outlier na 2 snimcich","vetrne/mracne snimky - flag + catalog_match_mode degraded; mask-first je nechava v CSV s vlajkou, grafy je tlumi. Nic delat netreba."],
 ["Check star 'skace' jen u jednoho cile","kontaminace check star (soused, sloupec BPM?) - viz cutout v reportu; vyber jine check star probehne sam pri pristi selekci, pripadne uzsi isolation."],
 ["TESS perioda nesedi s mou","blend 21 arcsec pixelu (blend check PNG!), alias, ci polovicni/dvojnasobna perioda (EW vs EA) - proto report kresli faze pro P i 2P."],
 ["Report cituje metodu, kterou jsem nezapnul","necituje - citace jsou gatovane behem (17.2). Pokud ji vidite, metoda skutecne bezela (zkontrolujte config snapshot na konfiguracni strane)."],
 ["Cisla se lisi od vcerejsiho behu","zkontrolujte config snapshot diff (report) a git provenience; vedecke jadro je pri stejnem vstupu bajtove reprodukovatelne (17.3) - zmena znamena zmenu vstupu ci konfigurace."],
], [40,128]))
st.append(P("Priloha B - slovnicek", H1))
for r in [
 "[b]draft[/b] - jedna zpracovavana pozorovaci rada (jedno pole, jeden filtr, sada snimku jedne noci).",
 "[b]obs_group[/b] - observacni skupina v draftu (filtr_expozice_binning, napr. NoFilter_60_2).",
 "[b]MASTERSTAR[/b] - referencni snimek rady s vyresenym WCS + hluboky katalog pole (astrometricka pravda).",
 "[b]komp / ensemble[/b] - srovnavaci hvezda / vazeny soubor komparaci, jehoz kombinovany flux tvori referenci diferencialni magnitudy.",
 "[b]check star[/b] - kontrolni hvezda merena jako cil, vyloucena z ensemble; nezavisly ukazatel kvality noci.",
 "[b]KMAG[/b] - merena ensemble-standardizovana magnituda check star (AAVSO export).",
 "[b]growth curve[/b] - krivka kumulativniho fluxu vs polomer apertury; zdroj SNR-optimalni apertury.",
 "[b]comp_qa / n_clean[/b] - leave-one-out QA komparaci (Sokolovsky indexy + magnitudovy locus) / pocet cistych komparaci po QA.",
 "[b]trust gate[/b] - trojbarevny GREEN/YELLOW/RED priznak duvery per cil (hard/soft model).",
 "[b]catalog_match_mode[/b] - per-epoch vlajka, jak dobre snimek sedi na katalog (vstup trustu).",
 "[b]BP-RP[/b] - Gaia barevny index (nahrada B-V; nativni barva VYVARu).",
 "[b]k''[/b] - koeficient extinkce 2. radu (mag / airmass / jednotka barvy); ve VYVARu v BP-RP jednotkach.",
 "[b]sigma_sys[/b] - per-pasmove systematicke chybove dno pricitane kvadraticky (kalibrovano z check hvezd).",
 "[b]HJD/BJD[/b] - helio-/barycentricke julianske datum (korekce na pohyb Zeme).",
 "[b]master / BPM[/b] - kalibracni ramec (dark/flat) / mapa vadnych pixelu z master darku.",
 "[b]anchor[/b] - bajtove identicka referencni sada vystupu (draft_435) jistici vedecke jadro.",
 "[b]--fast / --full[/b] - sessionovy ritual: rychla kontrola stavu a testu / plny headless beh s vedeckym komparatorem.",
 "[b]ePSF / ImagePSF[/b] - empiricka PSF (Anderson & King 2000) / jeji rastrova reprezentace v photutils.",
 "[b]AC faktor[/b] - aperture correction: prevod PSF toku na aperturni skalu z referencnich hvezd merenych obema cestami.",
 "[b]dilution (D)[/b] - katalogovy faktor redeni toku sousedy v aperture; delta_mag = -2.5 log10(D).",
 "[b]G_lim_50/90, miss@G90[/b] - magnitudy 50/90% uplnosti detekce a podil minutych jasnych Gaia hvezd (DAO-RECONCILE).",
]: st.append(P(r, B))



st.append(P("Priloha G - vystupni soubory draftu (kdo je pise a kdo cte)", H1))
st.append(tab([
 ["Soubor","Pise -> ctou"],
 ["calibrated/*.fits (vy_* hlavicky)","kalibrace+QC -> zarovnani, per-frame katalogy"],
 ["MASTERSTAR.fits (+WCS, parita)","platesolve -> vse downstream (astrometricka pravda)"],
 ["masterstars_full_match.csv","masterstar katalog -> Faze 0/1, per-frame parovani, HRD, TESS blend check"],
 ["proc_*.csv (per snimek)","per-frame mereni -> Faze 2A (read_flux_from_csv), variabilita, k2 airmass"],
 ["snr_aperture_table (per draft)","precompute -> aperturni sizing per hvezda"],
 ["active_targets.csv","Faze 0 -> Faze 1/2A, UI badges"],
 ["comparison_stars_per_target.csv","Faze 1 -> Faze 2A ensemble, comp QA, report tabulky"],
 ["lightcurves/*.csv + sidecary","Faze 2A -> trust, report, exporty (Priloha C)"],
 ["photometry_summary.csv","Faze 2A -> trust gate, report titulka, hockey-stick model"],
 ["reports/*.pdf, export/*","report/export -> clovek, AAVSO, VarAstro"],
], [58,110]))
st.append(P("Priloha F - matematicky dodatek (vzorce na jednom miste)", H1))
st.append(P("[b]Magnitudy (Pogson):[/b]", B))
st.append(EQP("m = -2.5 log10(F) + ZP;   Delta m = -2.5 log10(F1/F2);   sigma_mag = 1.0857 sigma_F / F"))
st.append(P("[b]CCD rovnice (Howell 1989; vse v elektronech):[/b]", B))
st.append(EQP("SNR = N_* / sqrt( N_* + n_pix (N_sky + N_dark + RN^2) );   N_* = F g;   n_pix = pi r^2"))
st.append(P("[b]Enclosed flux Gaussova profilu:[/b]", B))
st.append(EQP("E(r) = 1 - exp( -r^2 / (2 sigma^2) );   sigma = FWHM / 2.3548"))
st.append(P("[b]Moffat:[/b]", B))
st.append(EQP("I(r) = I0 (1 + (r/gamma)^2)^(-alpha);   FWHM = 2 gamma sqrt(2^(1/alpha) - 1)"))
st.append(P("[b]Robustni odhady sigmy:[/b]", B))
st.append(EQP("sigma_MAD = MAD / 0.6745;   sigma_IQR = (P75 - P25) / 1.349    [necitlive na outliery]"))
st.append(P("[b]Produkci chyba bodu (11.4):[/b]", B))
st.append(EQP("err_total^2 = err_photon_bkg^2 + sem_ens_rel^2 + sigma_sys_rel^2;   sem = s / (c4(n) sqrt(n))"))
st.append(P("kde c4(n) = sqrt(2/(n-1)) Gamma(n/2) / Gamma((n-1)/2) koriguje podhodnoceni smerodatne odchylky pri malem n (c4(3) ~ 0.886, c4(8) ~ 0.965).", B))
st.append(P("[b]Ensemble (11.6):[/b]", B))
st.append(EQP("m_ens = -2.5 log10( SUM_j w_j 10^(-0.4 m_j) );   delta_mag = m_inst - m_ens;   m_calib = m_inst + med_j(cat_j - inst_j)"))
st.append(P("[b]Extinkce (Kasten & Young 1989; k'' clen):[/b]", B))
st.append(EQP("X ~ 1/(cos z + 0.50572 (96.07995 - z)^(-1.6364));   delta_m_k2 = k'' x Delta(BP-RP) x X"))
st.append(P("[b]Von Neumann / VDI (14.1):[/b]", B))
st.append(EQP("eta = mean((x_{i+1}-x_i)^2) / var(x);   bily sum: eta ~ 2; koherentni signal: eta << 2"))
st.append(P("[b]Kontrolni hvezda (Howell 1988; 13.4):[/b]", B))
st.append(EQP("T = sigma_measured / sigma_predicted;   X2_excess = sigma_meas^2 - sigma_pred^2   [RED: T > 4, X2 > 4e-4]"))
st.append(P("[b]Dilution (11.11):[/b]", B))
st.append(EQP("D = F_star / (F_star + SUM F_nbr);   delta_mag = -2.5 log10(D);   mag_corr = mag_obs + delta_mag"))
st.append(P("[b]Uplnost detekce (Fleming 1995; 8.3):[/b]", B))
st.append(EQP("C(m) = 0.5 [ 1 - erf( (m - m_50) / (sqrt(2) s) ) ]   ->  G_lim_50, G_lim_90"))
st.append(P("Priloha E - rodina dokumentu (kde co hledat)", H1))
st.append(tab([
 ["Dokument","Obsah"],
 ["VYVAR_FLOW_CZ.pdf (tento)","referencni popis toku a algoritmu s parametry"],
 ["VYVAR_STATE.md","aktualni stav projektu (co je hotovo, co bezi)"],
 ["VYVAR_DECISIONS.md","rozhodnuti + zduvodneni + NEGATIVNI vysledky (proc neco NEdelame)"],
 ["VYVAR_ROADMAP.md","otevrena prace a poradi"],
 ["VYVAR_PROCESS.md","jak se vyviji (ritual, gating, parita parametru)"],
 ["VYVAR_INVARIANTS.md","strojove vynucovane kontrakty (flux, WCS, DAG, RNG, provenance, config)"],
 ["VYVAR_CONFIG_GUIDE_EN/CZ.md + VYVAR_PARAMETER_HANDBOOK_CZ.pdf","plne vysvetleni kazdeho parametru"],
 ["VYVAR_INSTALL_GUIDE_CZ.pdf","instalace a prvni spusteni"],
 ["CITATIONS.bib","kanonicky seznam citaci (gating dle behu)"],
 ["dev/results/specs/*_SPEC.md (CAL_DIAG, K2_DESIGN, SIGMA_FLOOR, SPARSE_TRUST, SIGMA_BUDGET...) - vyvojove specifikace subsystemu","technicke specifikace jednotlivych subsystemu"],
], [70,98]))
st.append(Spacer(1,6))
st.append(P("Overeni proti kodu: nazvy funkci a vychozi hodnoty v tomto dokumentu byly overeny grepem proti HEAD a config.json ke dni vydani. Pri kazde vetsi zmene pipeline patri regenerace tohoto PDF do docs-revision ritualu (builder cte jen staticky text - obsahove zmeny se pisi do builderu).", M))

doc = SimpleDocTemplate(os.path.join(ROOT,'docs','VYVAR_FLOW_CZ.pdf') if os.path.isdir(os.path.join(ROOT,'docs')) else '/home/claude/flowdoc/VYVAR_FLOW_CZ.pdf',
    pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=16*mm, bottomMargin=16*mm,
    title='VYVAR - Technicky popis pipeline', author='VYVAR project')
doc.build(st)
print('ok')
