# -*- coding: ascii -*-
# Regenerates docs/VYVAR_FLOW_CZ.pdf. Run from repo root:
#   python dev/tools/docs_pdf/build_flow_doc.py
import os
ROOT = os.getcwd()
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
B  = ParagraphStyle('B', parent=S['Normal'], fontSize=9.4, leading=12.6, spaceAfter=4)
M  = ParagraphStyle('M', parent=B, textColor=colors.HexColor('#555555'), fontSize=8.6)
FN = ParagraphStyle('FN', parent=B, fontName='Courier', fontSize=8.2, leading=10.5, textColor=colors.HexColor('#333355'))

def esc(t): return t.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def mk(t):
    t = esc(t)
    for a,b in (('[b]','<b>'),('[/b]','</b>'),('[i]','<i>'),('[/i]','</i>'),
                ('[sup]','<super>'),('[/sup]','</super>'),('[sub]','<sub>'),('[/sub]','</sub>'),
                ('[c]','<font face="Courier" size="8.3" color="#333355">'),('[/c]','</font>')):
        t = t.replace(a,b)
    return t
def P(t, s=B): return Paragraph(mk(t), s)
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

st = []
st.append(P("VYVAR - Technicky popis pipeline (od zacatku do konce)", T))
st.append(P("Referencni popis toku zpracovani a algoritmu vcetne konkretnich modulu a funkci, aperturni fotometrie do detailu a pripravene ePSF vetve. Verze 2.0 (2026-07-18, HEAD 7e88c3b + drzene stacky). Cestina bez diakritiky dle konvence projektu; dokument je generovan builderem (dev/tools/docs_pdf/build_flow_doc.py) a udrzuje se s kodem.", M))
st.append(Spacer(1,4))
st.append(P("Patri do rodiny dokumentu VYVAR (STATE, DECISIONS, ROADMAP, PROCESS, PARAMS, JOURNAL). Rozhodnuti a zduvodneni ziji v DECISIONS; otevrena prace v ROADMAP; parametry v config.json (komentovany) a VYVAR_PARAMETER_HANDBOOK_CZ.pdf. Nazvy funkci v tomto dokumentu byly overeny proti zivemu kodu k datu vydani; pri odchylce plati kod a dokument se regeneruje.", B))

st.append(P("1. Uvod, poslani a filozofie navrhu", H1))
st.append(P("VYVAR je vysoce automatizovany pipeline diferencialni fotometrie promennych hvezd: od surovych FITS po svetelne krivky a exporty (AAVSO, B.R.N.O./VarAstro) s vycislenou duverou v kazde cislo. Zasady navrhu:", B))
st.append(P("[b]Duvera na prvnim miste (trust-first).[/b] Kazdy cil prochazi krizovou validaci a trojbarevnym trust gate (GREEN/YELLOW/RED); uzivatel rano vidi krivku I miru jeji spolehlivosti.", B))
st.append(P("[b]Aperturni fotometrie jako overeny zaklad; ePSF jako opt-in.[/b] Na sirokem poli (~9.77 arcsec/px) je profil dobre navzorkovany a apertura vitezi; ePSF vetev je plne pripravena, ale vypnuta do validace na hustem poli Newtonu (~0.65 arcsec/px).", B))
st.append(P("[b]Gaia DR3 nativne.[/b] Paterni katalog je lokalni Gaia DR3 (SQLite, 40+ mil. hvezd); barvy primo z BP-RP, nikoli pres Johnsonovo B-V.", B))
st.append(P("[b]Nocni davkove zpracovani.[/b] Pipeline bezi po skonceni pozorovani; vazanou velicinou neni rychlost, ale spravnost a duvera raniho reportu. Proto si muzeme dovolit plnou detekci hvezd na kazdem snimku.", B))
st.append(P("[b]Vysvetlitelna statistika, ne cerna skrinka.[/b] Detekce promennych a QA stoji na transparentnich indexech (Sokolovsky, von Neumann, RMS obalka), ne na neuronove siti.", B))
st.append(P("[b]Reprodukovatelnost jako vlastnost.[/b] Kazdy vysledek nese provenance (config snapshot, git hash); vedecke jadro je jisteno bajtove identickym anchorem (draft_435) a ritualem --fast/--full.", B))

st.append(P("2. Prehled architektury", H1))
st.append(P("Zpracovava se jeden draft = jedna pozorovaci rada jednoho pole v jednom filtru. Hlavni tok:", B))
st.append(P("[c]import -> kalibrace -> QC -> zarovnani -> plate solve -> MASTERSTAR -> per-frame katalogy -> Faze 0 (cile) -> Faze 1 (kompy) -> Faze 2A (fotometrie) -> trust/QA -> detekce promennosti -> report + export[/c]", B))
st.append(tab([
 ["Modul","Ucel"],
 ["pipeline.py","orchestrace behu, masterstar, spatial-grid vyber, per-frame katalogy"],
 ["importer.py","nacteni noci, parovani masteru z knihovny (vc. teplotni tolerance)"],
 ["calibration.py","stavba a aplikace masteru, stari masteru, CAL-DIAG brany"],
 ["vyvar_alignment_frame.py","zarovnani serie (afinni transformace, rezidua)"],
 ["vyvar_platesolver.py","WCS: hintovane i slepe reseni, overeni proti Gaia"],
 ["photometry_core.py","run_full_photometry_pipeline, Faze 1+2A, chybovy model, trust vstupy"],
 ["psf_photometry.py","ePSF vetev: stavba modelu, PSF mereni (vypnuto, pripraveno)"],
 ["trust_flag_core.py","trust gate, T/X2 statistiky kontrolnich hvezd"],
 ["report_methods.py + pdf_report.py","SUMMARY MEASURE REPORT, exporty AAVSO/VarAstro"],
 ["database.py","SQLite: observator, knihovna kalibraci, drafty, katalogy"],
], [58,110]))
st.append(P("Sestavy projektu (referencni sada): wide Carl-Zeiss 200 mm + QHY294MM (~9.77 arcsec/px, Jirny), Newton 300/1200 + C3-26000 (~0.65 arcsec/px bin1, Dablice/Zdanice), Brno AZ800 80 cm + C5A-150M (~0.566 arcsec/px). Univerzalita: novy uzivatel zaklada vlastni sestavy do prazdne databaze; pipeline se adaptuje (binning, meritko, hustota pole).", B))
st.append(PageBreak())

st.append(P("3. Kalibrace snimku", H1))
st.append(P("Cil: odstranit otisk kamery a NEZNICIT pri tom data. Knihovna kalibraci (Calibration Library) uklada mastery organizovane dle kamery, binningu, expozice, teploty a filtru; importer pro kazdou noc vybira nejlepsi master (find_best_calibration_library_path) - dark s |dT| <= tolerance (vychozi 0.5 C), flat dle filtru. Pri jinem binningu se master prevzorkuje se zachovanim toku (provenance priznak dark_resample).", B))
st.append(P("[b]CAL-DIAG brany (fail-closed):[/b] po odectu darku musi median oblohy zustat fyzikalne smysluplny; krizova kontrola urovni odhali zamenu konvence stacku (SUM vs MEAN) a umi ji bezpecne prepocitat (autocorrect, zapsano do provenance) nebo beh zastavit. Trida chyb, ktera by jinak POTICHU znehodnotila celou noc.", B))
st.append(P("[b]Sky-surface preprocess:[/b] robustni fit polynomialni plochy radu 2 (6 clenu) na cely snimek, odecteni TVARU pozadi se zachovanim urovne (flux-conserving). Odstranuje gradient (Mesic, svitani) a zbytkove zakriveni po flatu; vyssi rad je zamerne zakazan (pojidal by mlhoviny). Soucast anchoru draft_435.", B))
st.append(fn("importer.py: find_best_calibration_library_path (volani), calibration.py: resolve_master_age, get_master_age_days; database.py: registrace masteru (odmitne dark bez konecne CCD_TEMP)"))

st.append(P("4. MASTERSTAR a astrometrie (plate solving)", H1))
st.append(P("MASTERSTAR je referencni katalog pole: hluboka detekce na stacku zarovnane serie (dvoupruchodova DAO: bezny prah + hlubsi pass2 na stacku s nizsim sumem), krizove ztotozneni s Gaia DR3, VSX a exoplanetovym katalogem. Kazda hvezda dostava stabilni identitu (Gaia source_id), barvu BP-RP a vlajky (znama promenna, NSS, exoplaneta).", B))
st.append(P("[b]WCS:[/b] hintovana cesta (pointing z FITS hlavicky + lokalni index) a SLEPA cesta: geometricke invarianty (trojuhelniky/kvady pres 8 nejblizsich sousedu) proti predpocitanemu indexu (fine/wide dle zorneho pole), kandidati sdruzeni hlasovanim (DBSCAN), WCS fit robustne (RANSAC). KAZDY kandidat musi projit overenim proti Gaia (min. podil a pocet shod; verify_mag_limit=14 je stejne spolehlive jako 16 a o ~28 % rychlejsi). Zapis WCS je fail-closed: bez platneho WCS se Faze 2A nespusti.", B))
st.append(fn("pipeline.py: build_masterstar_from_detrended; vyvar_platesolver.py: hintovane + slepe reseni (DAO na jasnych hvezdach, invarianty, DBSCAN, RANSAC), GAIA_DR3/build_blind_index.py (offline stavba indexu)"))

st.append(P("5. Zarovnani snimku (alignment)", H1))
st.append(P("Serie se sesazuje na spolecnou pixelovou mriz afinni transformaci (6 stupnu volnosti) z kontrolnich bodu (stredne prisna DAO detekce, strop poctu bodu ~ desitky az stovka). Kvalita se meri rezidui transformace; volitelna brana vyrazuje snimky s vysokymi rezidui (mrak, vitr v casti pole) i kdyz jinak vypadaji ostre. Median driftu centroidu pres noc ~0.4 px (zmereno na 127 snimcich).", B))
st.append(fn("vyvar_alignment_frame.py (detekce bodu, transformace, rezidua); QC brany ve frame_quality_* a frame_align_residual_*"))

st.append(P("6. Per-frame katalogy a detekce hvezd", H1))
st.append(P("Na KAZDEM kalibrovanem snimku bezi plna DAO detekce (photutils DAOStarFinder; prahy per ucel: QC stabilita vs masterstar uplnost - viz handbook, box o detekcni sigme). Detekce se ztotoznuje s masterstar/Gaia; vysledkem je per-frame katalog s centroidem, FWHM, elongaci a per-epoch trust vlajkou catalog_match_mode (jak dobre snimek sedi na katalog). Uplnost slabeho konce se nehlida sigmou, ale reconciliaci vuci Gaia (miss@G90 metrika, workstream DAO-RECONCILE).", B))
st.append(fn("photutils.detection.DAOStarFinder (13 mist volani napric QC/alignment/masterstar/SIPS); photometry_core.py: measure_fwhm_from_masterstar; pipeline.py: per-frame katalogova cesta"))
st.append(PageBreak())

st.append(P("7. Faze 0 - priprava cilu (active targets)", H1))
st.append(P("Cile mereni vznikaji tremi cestami: (1) automaticky ze VSX (zname promenne v poli do mag limitu), (2) automaticky z exoplanetoveho katalogu (hostitele tranzitu), (3) rucne zadane cile uzivatele. Kazdy cil se pinuje na Gaia source_id - identita je katalogova, ne pixelova.", B))
st.append(fn("database.py: active targets; VSX/vsx_make.py, exoplanets/exoplanet_make.py (offline stavba katalogu)"))

st.append(P("8. Faze 1 - vyber srovnavacich hvezd", H1))
st.append(P("Pro kazdy cil se stavi soubor srovnavacich hvezd (kompu) z masterstar kandidatu. Kriteria v poradi dulezitosti: [b]barva[/b] (comp_color_tiers: 4 urovne shody BP-RP s klesajici vahou - hlavni obrana proti extinkci 2. radu u nefiltrovanych dat), jasnostni blizkost (max_mag_diff s absolutnim stropem), prostorova blizkost a izolace (min_dist, gs11 dilucni filtr z Gaia predpovedi kontaminace), stabilita (max_comp_rms nad obalkou pole, comp_max_slope driftovy test se statistickou vyznamnosti), vylouceni znamych promennych/NSS. Cilovy pocet 3-8 (Broegova saturace); hustotni profil pole (sparse/dense) kriteria adaptuje, sparse fallback povoli i 1 komp - vysledek pak nese YELLOW.", B))
st.append(P("Po prvnim sestaveni krivky bezi iterativni cisteni (comp_iterative_clip): kompy se preveri proti souboru a odlehle se vyradi ci prevazi. Volitelny PyTICS rezim rekalibruje kompy iteracne (kazdy komp docasne cilem).", B))
st.append(fn("photometry_core.py: select_comparison_stars_per_target; pipeline.py: select_comparison_stars_spatial_grid (husta pole); comp_selection_per_target logika + trust vstupy"))

st.append(P("9. Faze 2A - aperturni fotometrie (hlavni cesta) - DETAILNE", H1))
st.append(P("Vstup: zarovnane kalibrovane snimky + per-frame katalogy + soubor kompu. Krok za krokem pro kazdou hvezdu na kazdem snimku:", B))
st.append(tab([
 ["Krok","Co se deje"],
 ["1 Centroid","pozice z per-frame katalogu (DAO centroid); zadne refitovani v aperture"],
 ["2 Apertura","polomer = trida jasnosti x FWHM (aperture_snr_sizing; jasne vetsi, slabe mensi)"],
 ["3 Pozadi","mezikruzi annulus_inner..outer x FWHM; robustni odhad (median/sigma-clip)"],
 ["4 Tok","suma pixelu v aperture minus pozadi x plocha; saturacni a dilucni vlajky"],
 ["5 Chyba","Howellova CCD rovnice + EMPIRICKY sum pozadi (prazdne apertury) + sigma_sys dno"],
 ["6 Aperturni korekce","preskalovani na spolecnou skalu z jasnych izolovanych referenci (median pomeru)"],
 ["7 Diferencial","cil minus vazeny soubor kompu (flux-sum kanonicka kombinace, Broegovy vahy)"],
 ["8 Ansambl QA","Honeycutt-style spolecne odchylky, epochove korekce, comp QA statistiky"],
], [34,134]))
st.append(P("[b]Matematika toku a sumu:[/b] SNR = N[sub]*[/sub]/sqrt(N[sub]*[/sub] + n[sub]pix[/sub](N[sub]sky[/sub]+N[sub]dark[/sub]+RN[sup]2[/sup])) (Howell 1989). Optimalni polomer pro slabe hvezdy ~0.7x FWHM, pro jasne robustnost prevazuje - proto tridy. Empiricke pozadi: sigma_bkg se MERI ~50-100 prazdnymi aperturami primo na snimku (measure_empty_aperture_sigma_bkg) - zachyti korelovany sum, ktery by teorie podcenila. Chybove dno sigma_sys (per pasmo, napr. 18 mmag wide/B4) je kalibrovane z rozptylu kontrolnich hvezd. Vazeny soubor: w ~ 1/sigma[sup]2[/sup] dle zmerene variability kompu (Broeg 2005); kombinace flux-sum je validovana proti AstroImageJ (Delta < 0.001 mag, 67 hvezd).", B))
st.append(P("[b]Co je zamerne VYPNUTO:[/b] SysRem, democratic/SavGol detrend, casove binovani vstupu - injekcni testy prokazaly poskozovani signalu na malych polich (binning zhorsil 24/25 cilu). Airmassove systematiky se resi pricinou: barevnou shodou kompu a (planovane) k'' korekci 2. radu.", B))
st.append(fn("photometry_core.py: run_full_photometry_pipeline (produkci vstup), run_phase2a, measure_empty_aperture_sigma_bkg; photutils.aperture: CircularAperture, CircularAnnulus, aperture_photometry"))
st.append(PageBreak())

st.append(P("10. Faze 2B - ePSF fotometrie (pripravena, vypnuta) - DETAILNE", H1))
st.append(P("PSF fotometrie nefotometruje kruh, ale FITUJE MODEL hvezdneho profilu na data - v hustych polich umi rozdelit prekryvajici se hvezdy, kde apertura koncem. VYVAR ma plnou ePSF vetev (psf_photometry.py, ~2900 radku) pripravenou a globalne vypnutou (psf_photometry_enabled=false) do validace na hustem poli Newtonu. Architektura:", B))
st.append(P("10.1 Co je ePSF (empiricka PSF)", H2))
st.append(P("Misto analytickeho profilu (Gauss, Moffat) se PSF ZMERI z dat: jasne izolovane hvezdy se vyrezou, subpixelove zarovnaji a slozi do prevzorkovaneho (oversampled) rastru - efektivni PSF dle Anderson & King (2000). Vyhoda: zachyti realny tvar (seeing, optika, vedeni) vcetne asymetrii; nevyhoda: potrebuje dost hvezd (epsf_min_stars >= 10) a stabilni profil.", B))
st.append(P("10.2 Stavba modelu", H2))
st.append(P("[c]build_epsf_model[/c]: vybere kandidaty z per-frame katalogu (izolace, SNR, nesaturovane), photutils [c]EPSFBuilder[/c] iterativne stavi prevzorkovany model (extract_stars -> fit -> re-centrace -> smoothing), vysledkem [c]ImagePSF[/c] (rastrovy model s oversamplingem). [c]build_epsf_grid_model[/c]: pole se rozdeli na bunky a v kazde se stavi lokalni ImagePSF - MRIZKA modelu zachycuje prostorovou promennost PSF pres pole (koma v rozich, naklon ohniskove roviny). Fallback: pri nedostatku hvezd analyticky Moffat (parametry gamma/alpha z medianu FWHM; _moffat_fwhm_px).", B))
st.append(P("10.3 Mereni", H2))
st.append(P("[c]SourceGrouper(min_separation)[/c] sdruzi prekryvajici se hvezdy do skupin fitovanych SPOLECNE (deblending); [c]PSFPhotometry[/c] fituje model (pozice + amplituda) v okne fit_shape odvozenem z FWHM; [c]IterativePSFPhotometry[/c] volitelne pridava iterace najdi-odecti-najdi pro slabe sousedy. Lokalni pozadi per hvezda; kvalita fitu (chi2, konvergence) se propisuje do vlajek.", B))
st.append(P("10.4 Kalibrace na aperturni skalu a brany", H2))
st.append(P("PSF tok neni ve stejne skale jako aperturni: [b]AC faktor[/b] (aperture correction) se pocita z >= 5 referencnich hvezd merenych OBEMA cestami (psf_ac_factor, psf_ac_n_used, psf_ac_applied v proc CSV); pod 5 referenci se PSF vysledek nepouzije (fallback na aperturu, psf_quality_fallback). Chybova cesta je oddelena (psf_flux_err - T1 fix z PSF auditu: driv se PSF frame nesl aperturni chybu). Adaptivni brana: PSF se routuje jen tam, kde ma smysl (SNR pod prahem, husta okoli); chi2 strop odmitne spatne fity.", B))
st.append(P("10.5 Proc je vetev vypnuta a co ji zapne", H2))
st.append(P("Aperturni cesta je validovana (AIJ Delta < 0.001 mag); PSF cesta je kodove auditovana (4 nalezy opraveny pri vypnute vetvi: chybova dekorelace, tichy AC fallback, mrtve vlajky), ale NEVALIDOVANA na realnem hustem poli. Enablement checklist ma jedinou zbyvajici polozku: draft z husteho pole Newtonu (h/chi Per trida) + krizova validace vuci aperture na izolovanych hvezdach. Do te doby: kazde zapnuti bez validace = neznama systematika.", B))
st.append(fn("psf_photometry.py: build_epsf_model, build_epsf_grid_model, get_epsf_fwhm_from_context, _moffat_fwhm_px; photutils.psf: EPSFBuilder, ImagePSF, SourceGrouper, PSFPhotometry, IterativePSFPhotometry"))
st.append(PageBreak())

st.append(P("11. Kontrola kvality a duvera (trust gate)", H1))
st.append(P("Trojbarevny verdikt per epocha i per cil: [b]GREEN[/b] = dost dobrych kompu (prah comp_trust_min_comps) + kontrolni hvezda sedi; [b]YELLOW[/b] = mene kompu, mereni PLATI s vetsi chybou (sigma skalovana ~1/sqrt(N)) - plynula degradace, ne binarni zahazovani; [b]RED[/b] = chybovy model neplati. Kontrolni hvezdy: T-statistika (pomer merena/predpovezena sigma; GREEN do 1.5) a X2 nadmerny rozptyl (RED nad prahem) dle Howell, Warnock & Mitchell (1988). Per-epoch vstupy: catalog_match_mode, saturace, dilucni vlajky, comp QA.", B))
st.append(fn("trust_flag_core.py: compute_trust_for_photometry_dir; photometry_core.py: comp QA statistiky, check-star ensemble (n >= 2)"))

st.append(P("12. Detekce promennych a overeni kandidatu", H1))
st.append(P("Vicekriterialni sito: prebytek RMS nad obalkou pole (sigma_threshold, comp_floor_factor), von Neumannuv pomer / VDI z-skore (koreluje po sobe jdouci body - odlisi krivku od bileho sumu), smoothness strop (proti cistym trendum), amplitudova minima. Kandidat jde do reportu s kontextem; TESS krizova kontrola (Lightkurve) overi periodu/amplitudu z vesmiru s vyhradou blendingu 21 arcsec pixelu. Filozofie: konzervativni prahy, clovek potvrzuje.", B))
st.append(fn("photometry_core.py: variability indexy (Sokolovsky 2017, von Neumann 1941); TESS most pres lightkurve (tess_enabled)"))

st.append(P("13. Vystupy: PDF report a exporty", H1))
st.append(P("SUMMARY MEASURE REPORT (pdf_report.py, reportlab): svetelne krivky s chybami a trust barvami, comp panel, HRD (Gaia GSP-Phot, paralaxni filtr SNR >= 5), konfiguracni stranka s PLNYM config snapshotem a Resolved Facts (hodnoty rozresene z DB/FITS za behu) - kazde cislo v reportu je zpetne dohledatelne. Exporty: AAVSO Extended Format (observer code, check/comp identity, citacni hlavicky) a VarAstro; sigma v exportech obsahuje sigma_sys dno.", B))
st.append(fn("report_methods.py: aavso_export_path a export cesta; pdf_report.py, photometry_report.py (reportlab rendering)"))

st.append(P("14. Katalogy a databaze", H1))
st.append(P("Lokalni katalogy (offline provoz): Gaia DR3 SQLite (~9.4 GB; GAIA_DR3/build_gaia_catalog.py, resumovatelna stavba z ESA TAP), blind indexy fine+wide (build_blind_index.py, z Gaia DB), VSX (vsx_make.py), exoplanety (exoplanet_make.py). Provozni SQLite databaze: observator (LOCATION/TELESCOPE/EQUIPMENTS - PRAZDNA u noveho uzivatele, referencni sada autora je harness-only seed v dev/tools/reference_seed.py), knihovna kalibraci (registrace masteru s povinnou CCD_TEMP u darku), drafty a jejich stav.", B))

st.append(P("15. Konfigurace, parametry a reprodukovatelnost", H1))
st.append(P("Tri druhy hodnot: [b]nastaveni[/b] v config.json (komentovany, editovatelny bez UI, validator dev/scripts/validate_config.py), [b]staticka fakta[/b] v databazi, [b]dynamicke hodnoty[/b] z FITS za behu. Registr parametru (dev/validation/params_registry.json, 269 zaznamu) je jediny zdroj metadat i napovedy. Reprodukovatelnost: bajtove identicky anchor (draft_435; SHA core 3d26f469, extended 6420f1da), sessionovy ritual --fast (testy, stav) a --full (headless beh + vedecky komparator ~1e-6), 970+ testu, provenance v kazdem vystupu. Zavislosti drzi DEPS_POLICY.md (ctvrtletni gated cyklus).", B))

st.append(P("16. Vedecke reference", H1))
for r in [
 "Anderson J., King I. R. 2000, PASP 112, 1360 - empiricka (efektivni) PSF.",
 "Broeg C., Fernandez M., Neuhauser R. 2005, AN 326, 134 - vazeny soubor kompu.",
 "Honeycutt R. K. 1992, PASP 104, 435 - ansamblova fotometrie, epochove korekce.",
 "Howell S. B. 1989, PASP 101, 616 - CCD rovnice, optimalni apertura.",
 "Howell S. B., Warnock A., Mitchell K. J. 1988, AJ 95, 247 - kontrolni hvezdy, T-statistika.",
 "Kasten F., Young A. T. 1989, Appl. Opt. 28, 4735 - airmass.",
 "Jordi C. et al. 2010, A&A 523, A48; Riello M. et al. 2021, A&A 649, A3 - Gaia fotometrie, BP-RP.",
 "Tamuz O., Mazeh T., Zucker S. 2005, MNRAS 356, 1466 - SysRem (vyhodnoceno, vypnuto).",
 "Sokolovsky K. V. et al. 2017, MNRAS 464, 274 - indexy promennosti.",
 "von Neumann J. 1941, Ann. Math. Stat. 12, 367 - pomer stredni kvadraticke naslednosti.",
 "Osborn J. et al. 2015, MNRAS 452, 1707 - scintilace.",
 "Watson C., Henden A., Price A. 2006, SASS 25, 47 - VSX.",
 "Andrae R. et al. 2023, A&A 674, A27 - Gaia GSP-Phot (HRD).",
 "Lightkurve Collaboration 2018 - TESS/Kepler casove rady.",
]: st.append(P(r, M))
st.append(Spacer(1,4))
st.append(P("Priloha A - struktura draftu: Archive/<pole>/<draft>/ s podadresari calibrated/, platesolve/, photometry/ (proc CSV per cil: toky, chyby, vlajky vc. psf_* sloupcu), reports/, export/. Priloha B - slovnicek: draft (pozorovaci rada), komp (srovnavaci hvezda), check (kontrolni hvezda), master (kalibracni ramec), trust gate (GREEN/YELLOW/RED), anchor (bajtove identicka referencni sada).", B))
st.append(P("Overeni proti kodu: nazvy funkci v tomto dokumentu byly overeny grepem proti HEAD ke dni vydani. Pri kazde vetsi zmene pipeline patri regenerace tohoto PDF do docs-revision ritualu (builder cte jen staticky text - obsahove zmeny se pisi do builderu).", M))

doc = SimpleDocTemplate(os.path.join(ROOT,'docs','VYVAR_FLOW_CZ.pdf') if os.path.isdir(os.path.join(ROOT,'docs')) else '/home/claude/flowdoc/VYVAR_FLOW_CZ.pdf',
    pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=16*mm, bottomMargin=16*mm,
    title='VYVAR - Technicky popis pipeline', author='VYVAR project')
doc.build(st)
print('ok')
