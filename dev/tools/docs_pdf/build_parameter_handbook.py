# -*- coding: ascii -*-
# Regenerates docs/VYVAR_PARAMETER_HANDBOOK_CZ.pdf from repo sources.
# Run from repo root: python dev/tools/docs_pdf/build_parameter_handbook.py
import json, re, os, sys
ROOT = os.getcwd()
def _strip_comments(txt):
    out, i, in_str, esc = [], 0, False, False
    while i < len(txt):
        c = txt[i]
        if in_str:
            out.append(c)
            if esc: esc = False
            elif c == chr(92): esc = True
            elif c == '"': in_str = False
            i += 1
        else:
            if c == '"': in_str = True; out.append(c); i += 1
            elif c == '/' and i+1 < len(txt) and txt[i+1] == '/':
                while i < len(txt) and txt[i] != chr(10): i += 1
            else: out.append(c); i += 1
    return ''.join(out)
_cfgtxt = open(os.path.join(ROOT,'config.json'), encoding='utf-8', errors='replace').read()
CFG = json.loads(_strip_comments(_cfgtxt))
_g = open(os.path.join(ROOT,'docs','VYVAR_CONFIG_GUIDE_CZ.md'), encoding='utf-8', errors='replace').read()
_rows = re.findall(r'^\| `([A-Za-z0-9_]+)` \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \|$', _g, re.M)
GUIDE = {r[0]: {'default': r[1], 'typ': r[2], 'zdroj': r[3], 'kde': r[4], 'vysv': r[5]} for r in _rows}
import json, re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Table, TableStyle, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


S = getSampleStyleSheet()
st_title = ParagraphStyle('T', parent=S['Title'], fontSize=22, spaceAfter=6)
st_h1 = ParagraphStyle('H1', parent=S['Heading1'], fontSize=15, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#1a3a5c'))
st_h2 = ParagraphStyle('H2', parent=S['Heading2'], fontSize=11.5, spaceBefore=10, spaceAfter=2, fontName='Courier-Bold', textColor=colors.HexColor('#0f2740'))
st_body = ParagraphStyle('B', parent=S['Normal'], fontSize=9.2, leading=12.2, spaceAfter=3)
st_meta = ParagraphStyle('M', parent=st_body, textColor=colors.HexColor('#444444'), fontSize=8.6)
st_eff = ParagraphStyle('E', parent=st_body, leftIndent=10)
st_deep_t = ParagraphStyle('DT', parent=S['Heading3'], fontSize=10.5, textColor=colors.HexColor('#7a3b00'), spaceBefore=2, spaceAfter=3)
st_deep_b = ParagraphStyle('DB', parent=st_body, fontSize=8.9, leading=11.8)
st_intro = ParagraphStyle('I', parent=st_body, fontSize=9.6, leading=13)

def esc(t):
    return t.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def mk(t):
    # allow our own markup: [sup]..[/sup], [sub]..[/sub], [b]..[/b], [i]..[/i]
    t = esc(t)
    for a,b in (('[sup]','<super>'),('[/sup]','</super>'),('[sub]','<sub>'),('[/sub]','</sub>'),
                ('[b]','<b>'),('[/b]','</b>'),('[i]','<i>'),('[/i]','</i>')):
        t = t.replace(esc(a) if '<' in a else a, b)
    return t

def deepbox(title, paras):
    inner = [Paragraph(mk(title), st_deep_t)] + [Paragraph(mk(p), st_deep_b) for p in paras]
    tb = Table([[inner]], colWidths=[168*mm])
    tb.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1), colors.HexColor('#fdf4e7')),
        ('BOX',(0,0),(-1,-1), 0.8, colors.HexColor('#d9a45b')),
        ('LEFTPADDING',(0,0),(-1,-1), 8), ('RIGHTPADDING',(0,0),(-1,-1), 8),
        ('TOPPADDING',(0,0),(-1,-1), 6), ('BOTTOMPADDING',(0,0),(-1,-1), 6),
    ]))
    return tb

def fmt_default(v):
    s = json.dumps(v, ensure_ascii=True)
    return s if len(s) <= 60 else s[:57] + '...'

DETAIL = {}   # key -> dict(mensi=, vetsi=, proc=, rozsah=)  (any subset)
DEEP = {}     # anchor_key -> (title, [paras])  rendered before that key's entry
SEC_INTRO = {} # section title -> intro paragraph
LIT = []      # literature list (filled later)

D = DETAIL
# ================= KALIBRACE =================
D['masterdark_validity_days'] = dict(
 proc="30 dni je kompromis: temny proud kamery se meni hlavne s teplotou (zdvojnasobuje se zhruba kazdych 5-7 C), ale pri chlazene kamere drzene na cilove teplote se master meni pomalu - hlavni riziko jsou nove horke pixely a drift elektroniky.",
 mensi="Castejsi foceni darku; vyssi jistota, vic prace. Pod ~14 dni uz zisk nemeritelny, pokud se nemeni teplota ci gain.",
 vetsi="Roste riziko, ze mapa horkych pixelu nesedi (nove horke pixely zustanou v datech jako falesne hvezdy nebo poskodi fotometrii pixelu pod hvezdou). Nad ~60 dni u chlazene CMOS znatelne; u nechlazene mnohem driv.")
D['masterflat_validity_days'] = dict(
 proc="Flat se meni s prachem a mechanikou (rotace kamery, zaostreni). 30 dni sedi pro stabilni sestavu.",
 mensi="Bezpecnejsi pri caste manipulaci se sestavou.",
 vetsi="Prachova zrnka se posunou nebo pribudou - deleni starym flatem pak zrnko NEODSTRANI, ale vytvori falesnou strukturu (dvojice tmavy/svetly flicek). Na fotometrii cile to ma vliv, jen kdyz hvezda prejde pres flicek - o to zaludnejsi.")
D['calibration_master_ccd_temp_tolerance_c'] = dict(
 rozsah="0.1-5 C; vychozi 0.5 C.",
 mensi="Prisnejsi parovani darku dle teploty - nejcistsi kalibrace, ale nemusi se najit zadny master.",
 vetsi="Povoli dark z jine teploty; temny proud pak nesedi (~15 %/C u typicke CMOS) a po odectu zustane zbytek nebo prekompenzace.",
 proc="0.5 C odpovida beznemu kolisani regulace chlazeni; vetsi rozdil uz je meritelny v pozadi.")

# ================= QC =================
D['auto_fwhm_enabled'] = dict(
 proc="Pevny limit netusi, jaka byla noc. Auto rezim odvodi prah z medianu FWHM teto noci (x k-faktor) - snimky se posuzuji vuci vlastnimu seeingu, ne vuci idealu.")
D['auto_fwhm_k_factor'] = dict(
 rozsah="mezi k_min a k_max.",
 mensi="Prisnejsi (blize medianu) - vyrazuje i mirne zhorsene snimky.",
 vetsi="Tolerantnejsi; propusti chvosty seeingu.",
 proc="Nasobitel medianu: median x k = prah. Typicky 1.3-1.8: drzi jadro noci, rezne zjevne zhorseni (vitr, mrak, rozostreni).")
D['preprocess_sky_surface_order'] = dict(
 rozsah="0 (vypnuto), 1 (rovina), 2 (kvadraticka plocha; vychozi a soucast anchoru).",
 mensi="0/1: gradienty oblohy (Mesic, svitani, vinetace zbytkova po flatu) zustanou v datech a prosaknou do mezikruzi -> systematicky posun pozadi zavisly na poloze.",
 vetsi="Vyssi rad neni povolen zamerne: flexibilnejsi plocha uz by zacala pojidat realne velkoskalove struktury (mlhoviny!) a lokalni pozadi hvezd - viz box.",
 proc="Rad 2 odstrani presne to, co je systematicke (gradient + zakriveni), a nic vic. Fit zachovava tok (flux-conserving).")
D['osc_channel_binning'] = dict(
 rozsah="1-4; vychozi 2.",
 proc="Pouze OSC: NxN prumer po extrakci kanalu (R/G/B/oneRGGB). Celkova linearni mira vuci raw mozaice = 2 (superpixel) x N. S n=2 je vysledne rozliseni 4x hrubsi nez raw pixel.",
 mensi="Jemnejsi vzorkovani (blize superpixelu 2x2); vice sumu, pomalejsi beh.",
 vetsi="Hrubsi kanaly; mensi soubory, ale ztrata detailu a horsi match na MASTERSTAR pri OSC-2.")

# ================= ALIGNMENT =================
D['alignment_max_control_points'] = dict(
 mensi="Rychlejsi, ale mene redundance vuci spatnym parum.",
 vetsi="Robustnejsi odhad transformace, pomalejsi; nad ~100 bodu uz presnost afinni transformace nerostie.",
 proc="Afinni transformace ma 6 stupnu volnosti - desitky kvalitnich bodu bohate staci, zbytek je pojistka proti outlierum.")
D['alignment_max_stars'] = dict(
 proc="Strop kandidatu pro parovac (vykonovy limit kombinatoriky trojuhelnikoveho matchingu).")

# ================= DETEKCE / MASTERSTAR / VARIABILITA / BLIND =================
D['epsf_min_stars'] = dict(
 proc="ePSF model z mene nez ~10 hvezd je nestabilni (sum jednotlivych hvezd se propise do modelu); PSF program je stejne vypnut do validace.")
D['exoplanet_match_max_sep_arcsec'] = dict(
 proc="Ztotozneni s hostiteli exoplanet; mala tolerance staci, katalogove pozice jsou presne.")

# ================= FOTOMETRIE =================
D['annulus_inner_fwhm'] = dict(
 mensi="Mezikruzi blize hvezde - kontaminace kridly PSF samotne hvezdy (pozadi nadhodnocene, tok podhodnoceny).",
 vetsi="Cistejsi od kridel, ale pozadi se meri dal od hvezdy (gradienty).",
 proc="Vychozi ~3x FWHM je za >99 % toku Gaussovskeho profilu; density adaptace smi na hustem poli zprisnit.")
D['annulus_outer_fwhm'] = dict(
 proc="Sirka mezikruzi urcuje pocet pixelu pro odhad pozadi - potrebujeme stovky pixelu, aby sum medianu pozadi byl zanedbatelny vuci sumu apertury.")
D['gs11_dilution_enabled'] = dict(
 proc="Dilucia = cizi svetlo v aperture. Z Gaia kataloga vime, kdo je pobliz a jak jasny - kontaminaci lze predpovedet bez mereni. Kandidat s velkou diluci nesmi byt komparacni hvezda.")
D['err_empty_apertures_n'] = dict(
 mensi="Rychlejsi, ale odhad sumu pozadi ma vetsi rozptyl.",
 vetsi="Presnejsi odhad, drazsi; ~50-100 apertur je optimum (chyba odhadu sigma ~ 1/sqrt(2N)).")

# ================= COMP SELECTION / TRUST / K2 / OSTATNI =================
D['comp_color_tiers'] = dict(
 proc="Ctyri urovne shody barvy (Gaia BP-RP) s klesajici vahou: |dBP-RP| do 0.15 / 0.30 / 0.55 / 1.10 s vahami 1.0 / 0.85 / 0.5 / 0.25. Barva je NEJDULEZITEJSI kriterium u nefiltrovaneho pozorovani - viz box o extinkci 2. radu. Prahy jsou nastavene tak, aby uroven 1 drzela barevny clen pod ~1-2 mmag na jednotku airmass a dalsi urovne degradovaly plynule.",
 mensi="(prahy) Cistsi barevna shoda, ale mene kandidatu - na ridkem poli nemusi vzniknout soubor.",
 vetsi="(prahy) Vic kandidatu, ale roste barevny clen: rozdilna extinkce behem noci se propise do krivky jako falesny trend korelovany s airmass.")
D['phase01_comparison_n_comp_min'] = dict(
 proc="Cilova velikost souboru; density adaptace smi na ridkem poli snizit. Viz box o Broegove souboru - proc vic neznamena lepe.",
 mensi="Rychlejsi sestaveni, ale sum souboru klesa jako ~1/sqrt(N) - pod 3 uz je soubor krehky (jeden spatny komp = tretina vahy).",
 vetsi="Nad ~8 uz sum souboru nedominuje - limitem je scintilace a systematiky; dalsi kompy jen redi barevnou shodu.")
D['phase01_comparison_n_comp_max'] = dict(
 proc="Strop 8: literatura (Broeg 2005; scintilacni saturace ~6-8) i nase mereni ukazuji, ze dalsi kompy uz nesnizuji sum, ale nuti brat horsi kandidaty (barva, vzdalenost).")
D['phase01_comparison_min_dist_arcsec'] = dict(
 proc="Prilis blizky komp sdileli s cilem mezikruzi a kontaminuje; minimum drzi mereni nezavisla.",
 mensi="Riziko vzajemne kontaminace pozadi/kridel.",
 vetsi="Kompy dal od cile - roste vliv gradientu pruzracnosti pres pole (flat-field/mraky).")
D['phase01_comparison_max_comp_rms'] = dict(
 mensi="Jen velmi stabilni kandidati - malo kompu na horsich nocich.",
 vetsi="Do souboru proniknou sumive/promenne hvezdy; Broegovo vazeni je sice potlaci, ale nevynuluje.",
 proc="Prah nad RMS-mag obalkou pole; dense/tighten adaptace smi zprisnit.")
D['comp_max_slope_mmag_hr'] = dict(
 proc="Linearni drift kompu (mmag/hod) = bud skutecna pomala promennost, nebo barevny extinkci clen. Oboje je pro komp diskvalifikace. Prah spolupracuje se significance testem - drift musi byt STATISTICKY prokazany, ne jen sumovy.")
D['comp_trust_min_comps'] = dict(
 proc="Prah pro GREEN duveru (od 06/2026 uz NE tvrde minimum!): >= prah dobrych kompu = GREEN; 1 az prah-1 = YELLOW se sigma skalovanym dle N; degradace je plynula, ne binarni. Kodovy vychozi 5 = literaturni doporuceni; produkce bezi validovane na 3 (DECISIONS zaznam) - mereni s 3 kompy JE korektni, jen nese vetsi chybu.",
 mensi="GREEN i pro male soubory - odznak ztraci vypovidaci hodnotu.",
 vetsi="GREEN vzacny; exporty prevazne YELLOW i pri kvalitnich datech.")
D['sparse_trust_T_green'] = dict(
 proc="T-statistika kontrolni hvezdy (pomer namereneho rozptylu k ocekavane chybe): T<=1.5 znamena, ze chybovy model sedi - viz box o trust modelu.")
D['sparse_trust_X2_RED'] = dict(
 proc="Nadmerny rozptyl (excess variance) kontrolni hvezdy nad prahem = chybovy model NEplati (systematika, spatny komp) - vysledek RED bez ohledu na pocet kompu.")
D['lc_quality_min_frames'] = dict(
 proc="Pod ~30 bodu jsou statistiky krivky (RMS, trend) nespolehlive; kratke serie maji vlastni short-baseline drahu s mekcimi pozadavky a explicitni znackou.")
D['check_star_min_epochs'] = dict(
 proc="Verdikt kontrolni hvezdy z par bodu je nahoda; minimum epoch zajistuje statistickou vahu T/X2 testu.")

# K2
D['k2_mode'] = dict(
 proc="'literature' = koeficienty 2. radu z literatury dle pasma a barvy (bezpecny vychozi stav). 'off' = bez korekce. Nocni fit k'' (NIGHT_FIT v2) byl odstranen (CFG-REVIEW-2026-10) - zadna noc nemela dost rozsahu airmass.")
D['k2_defaults_bprp'] = dict(
 proc="Literaturni k'' po pasmech vuci BP-RP; typicky -0.03 az -0.06 mag/airmass/mag(barvy) pro modra pasma, ~0 pro cervena. Viz box o extinkci.")
D['apply_color_term'] = dict(
 proc="'off' = instrumentalni system (CV pro AAVSO). Transformace na standardni system vyzaduje filtrovana data a stabilni barevny clen - budouci rozsireni.")

# HRD / EXPORT / SYSTEM / PATHS (kratke)
D['hrd_parallax_snr_min'] = dict(
 proc="Paralaxa s SNR<5 dava pri prevodu na vzdalenost silne vychylene absolutni magnitudy (Lutz-Kelkerova trida efektu) - takove hvezdy do HRD nepatri.")
D['hrd_online_enrich_enabled'] = dict(
 proc="Online obohaceni (Gaia/SIMBAD) bezi jen pro report a je omezene caps/timeouty - vypadek site nikdy neblokuje fotometrii.")
D['transient_search_enabled'] = dict(
 proc="Hledani novych zdroju (novy, supernovy) srovnanim detekci s Gaia na kazdem snimku noci (Gaia Science Alerts / NMW); planetky a komety z drah MPC (dvoutelesovy vypocet). Bez odecitani snimku: supernova na jadru galaxie se nenajde.")
D['tess_enabled'] = dict(
 proc="Krizova kontrola kandidatu promennosti proti TESS (Lightkurve): perioda/amplituda z vesmiru je nejsilnejsi nezavisle potvrzeni; pozor na blending 21'' pixelu TESS.")

# ================= DEEP DIVE BOXY =================

DEEP['phase01_comparison_n_comp_min'] = ("BOX: Kolik srovnavacich hvezd? (Broeg 2005, Honeycutt 1992, scintilace)", [
 "Diferencialni magnituda cile vuci vazenemu souboru kompu ma sum: [b]sigma[sub]dif[/sub][sup]2[/sup] = sigma[sub]cil[/sub][sup]2[/sup] + sigma[sub]soubor[/sub][sup]2[/sup][/b], kde pro N srovnatelnych kompu [b]sigma[sub]soubor[/sub] ~ sigma[sub]komp[/sub]/sqrt(N)[/b]. Prechod z 1 na 4 kompy tedy snizi prispevek souboru 2x; z 4 na 8 uz jen 1.4x; z 8 na 16 dalsich 1.4x - vynosy klesaji.",
 "Zaroven existuje spodni mez, kterou zadny pocet kompu neprorazi: scintilace atmosfery (Young 1967, Osborn+2015) je pro vsechny hvezdy v poli castecne KORELOVANA, a systematiky (flat, barva) se prumerovanim neodstrani. Empiricky se prinos nasycuje kolem 6-8 kompu - proto n_comp_max=8.",
 "Broeg (2005): vahy kompu se voli podle jejich zmerene variability (w ~ 1/sigma[sup]2[/sup]) - 'spatny' komp se sam potlaci. Honeycutt (1992): spolecne nocni odchylky (mraky, pruzracnost) se resi soustavou epochovych korekci - VYVAR pouziva flux-sum kanonickou variantu validovanou proti AIJ.",
 "Dusledek pro uzivatele: 3 dobre BAREVNE SHODNE kompy jsou lepsi nez 10 spatnych. Proto je barva (comp_color_tiers) prisnejsi kriterium nez pocet."])

DEEP['comp_color_tiers'] = ("BOX: Barva kompu a extinkce druheho radu (Jordi 2010, Riello 2021, Kasten-Young 1989)", [
 "Atmosfera zeslabuje modre svetlo vic nez cervene. Pro hvezdy RUZNE barvy proto extinkce roste s airmass X ruzne rychle: [b]dm = k' X + k'' C X[/b], kde C je barva (u nas Gaia BP-RP) a k'' koeficient druheho radu, typicky -0.02 az -0.06 mag/airmass/mag pro siroka modra pasma.",
 "Diferencialni fotometrie prvni clen (k'X) dokonale vyrusi - je stejny pro cil i komp. Druhy clen se vyrusi JEN pri shode barev: zbytek je [b]d(dm) = k'' (C[sub]cil[/sub] - C[sub]komp[/sub]) X[/b]. Priklad: k''=-0.04, rozdil barev 0.5 mag, zmena airmass behem noci dX=0.5 -> falesny trend 10 mmag - vic nez amplituda mnoha promennych!",
 "Urovne 0.15/0.30/0.55/1.10 s vahami 1.0/0.85/0.5/0.25 drzi vazeny prispevek pod par mmag pro typicke k''. U nefiltrovaneho (CV) pozorovani, kde je efektivni pasmo siroke a k'' nejvetsi, je tesna barevna shoda HLAVNI obranou - zadny pocet kompu ji nenahradi.",
 "Airmass se pocita dle Kasten & Young (1989); k'' je vzdy literaturni (nocni fit byl odstranen, CFG-REVIEW-2026-10)."])

DEEP['comp_trust_min_comps'] = ("BOX: Trust model GREEN/YELLOW/RED a kontrolni hvezda (Howell, Warnock & Mitchell 1988)", [
 "VYVAR znamkuje kazdou epochu: GREEN = plna duvera (dost dobrych kompu + kontrolni hvezda sedi), YELLOW = mereni platne s vetsi chybou (malo kompu; sigma skalovana ~1/sqrt(N)), RED = chybovy model neplati.",
 "Kontrolni hvezda je 'promenna, o ktere vime, ze je stala': meri se stejne jako cil. T-statistika porovnava jeji namereny rozptyl s predpovezenou chybou: [b]T = sigma[sub]merena[/sub]/sigma[sub]predpoved[/sub][/b]. T~1 = model chyb sedi (GREEN do 1.5); T>>1 = neco systematickeho (RED nad 4). Nadmerny rozptyl X[sup]2[/sup] doplnuje test o absolutni miru.",
 "Od 06/2026 plati plynula degradace (COMP_DEGRADATION_SPEC): mene kompu uz NEZNAMENA zahozeni mereni - jen poctive vetsi chybu a YELLOW znacku. AAVSO minimum je 1 komp + check; nas GREEN prah je prisnejsi prave proto, aby znacka mela vahu."])


DEEP['preprocess_sky_surface_order'] = ("BOX: Proc sky-surface prave rad 2", [
 "Model pozadi je polynomialni plocha radu 2 (6 clenu: 1, x, y, x[sup]2[/sup], xy, y[sup]2[/sup]) fitovana robustne na cely snimek a odectena se zachovanim toku (median plochy se vrati zpet - odstranuje se jen TVAR, ne uroven).",
 "Rad 2 pokryva presne systematicke tvary: linearni gradient (Mesic nizko, svitani) a zakriveni (zbytkova vinetace po flatu). Rad 3+ by zacal sledovat realne struktury - mlhoviny, Mlecnou drahu, hala jasnych hvezd - a POTICHU by je odecital z dat. U promennych hvezd v mlhovinach (nase cile v h/chi Per!) by to byla vedecka chyba.",
 "Lokalni pozadi kazde hvezdy se stejne meri mezikruzim - sky-surface resi jen velkoskalovy tvar, ktery by mezikruzi vzdalenejsich kompu posouval ruzne. Zavedeno s anchorem draft_435; zmena radu = zmena vedy = novy anchor."])


DEEP['err_empty_apertures_n'] = ("BOX: Chybovy rozpocet - odkud se bere systematicke dno", [
 "Uplna chyba bodu: [b]sigma[sup]2[/sup] = sigma[sub]foton[/sub][sup]2[/sup] + sigma[sub]pozadi[/sub][sup]2[/sup] + sigma[sub]soubor[/sub][sup]2[/sup] + sigma[sub]scint[/sub][sup]2[/sup] + sigma[sub]sys[/sub][sup]2[/sup][/b]. Prvni cleny umime spocitat (Howell + empiricke pozadi); scintilaci odhadnout (Osborn+2015: zavisi na aperture dalekohledu, airmass a rychlosti vetru); sigma_sys je poctive priznani zbytku (flat, barva, PSF variace).",
 "Systematicke dno se meri kazdou noc na jasnych kompech (rozptyl minus fotonovy sum, SIGMA-SYS-NIGHT-2026-10). Kdyz noc nema dost kompu, pouzije se hodnota kamery EQUIPMENTS.SIGMA_SYS_MAG v databazi (napr. C5A-150M: 18 mmag); prazdna = 0.",
 "Kontrola spravnosti bezi neustale: T-statistika kontrolni hvezdy ma vychazet ~1. T<1 znamena chyby nadhodnocene (dno prilis velke), T>1.5 podhodnocene."])

DEEP['gs11_dilution_enabled'] = ("BOX: Dilucia - cizi svetlo v aperture", [
 "Podil kontaminace: [b]d = F[sub]sousede[/sub] / (F[sub]hvezda[/sub] + F[sub]sousede[/sub])[/b], kde toky sousedu v aperture odhadneme z Gaia magnitud a profilu PSF - jeste PRED merenim.",
 "Proc je to pro kompy fatalni: kontaminovany komp vypada stabilne, dokud je seeing stabilni. Zmena FWHM zmeni podil sousedova svetla v aperture -> 'konstantni' komp se pohne synchronne se seeingem a vyrobi falesnou anti-korelaci v cili. Par procent dilucie = mmag systematiky.",
 "Pro cil naopak dilucia znamena PODCENENOU amplitudu (promenny signal je redeny konstantnim svetlem) - amplitudova korekce 1/(1-d) je mozna jen se znamym d."])


DEEP['masterdark_validity_days'] = ("BOX: Stari masteru - fyzika za 30 dny", [
 "Temny proud: I[sub]dark[/sub](T) ~ exp(T/T[sub]0[/sub]) s T[sub]0[/sub] ~ 5-7 C (zdvojnasobeni na ~6 C). Pri stabilni cilove teplote chlazeni se master nemeni proudem, ale POPULACI horkych pixelu - ta roste s casem (kosmicke zareni degraduje pixely) radove o jednotky az desitky pixelu za mesic.",
 "Novy horky pixel, ktery ve starem masteru neni, zustane v kalibrovanem snimku jako 'hvezdicka' - detekce ho odfiltruje tvarem, ale padne-li pod skutecnou hvezdu, tise pokrivi jeji tok. 30 dni drzi ocekavany pocet nezachycenych novych horkych pixelu zanedbatelny; ledger hlida expiraci a pipeline si o nove darky rekne.",
 "Flat: zadna fyzika starnuti, jen mechanika (prach, rotace kamery, refokus). Po kazde manipulaci se sestavou focte novy flat bez ohledu na pocitadlo dni."])

SEC_INTRO.update({
 'Observer & export identity': "Identita pozorovatele a stanoviste. Vetsina hodnot jsou fakta (autoritativni v databazi LOCATION); zde ziji kopie pro export a UI.",
 'Calibration': "Odstraneni otisku kamery (dark, flat) a pojistky, ktere chytnou spatny master driv, nez znici noc. Zde se rozhoduje o samotne pouzitelnosti dat.",
 'Frame quality control (QC)': "Automaticke znamkovani kazdeho snimku a brany vyrazujici zjevne vadne snimky. Filozofie: radeji poctive vyradit, nez tise prumerovat odpad.",
 'Alignment': "Sesazeni serie na spolecnou pixelovou mriz. Tri parametry, ktere temer nikdy nemusite menit.",
 'Detection, plate solving & masterstar': "Od pixelu ke hvezdam se jmeny: detekce, slepe reseni souradnic s overenim proti Gaia, referencni katalog masterstar a sito kandidatu promennosti.",
 'Photometry': "Srdce pipeline: apertury, pozadi, model chyb, korekce. Zde kazda hodnota primo hybe vysledkem - detrendy (SysRem, SavGol, Democratic, binovani) byly z kodu odstraneny.",
 'Comparison-star selection': "Vyber souboru stalych hvezd. Nejdulezitejsi kapitola pro kvalitu krivky: barva > pocet.",
 'Trust & quality flags': "Jak VYVAR znamkuje sam sebe: GREEN/YELLOW/RED, kontrolni hvezda, plynula degradace.",
 'Atmospheric extinction & color': "Extinkce druheho radu - najvetsi zbyvajici systematika nefiltrovanych dat a plan jejiho fitovani.",
 'Reports & HRD': "Vizualni ladeni reportu a HR diagramu; na cisla fotometrie nema vliv.",
 'Export': "AAVSO/VarAstro export a TESS krizova kontrola.",
 'System & performance': "Vykon stroje; hodnoty se pocitaji automaticky.",
 'File & catalog paths': "Cesty k datum a katalogum - nastavuje instalace, benze editace neni potreba. Jedina rodina, kde chyba znamena 'nenajdu data', ne 'spatna veda'.",
})

LIT.extend([
 "Broeg C., Fernandez M., Neuhauser R. 2005, AN 326, 134 - A new algorithm for differential photometry (vazeny soubor kompu).",
 "Honeycutt R. K. 1992, PASP 104, 435 - CCD ensemble photometry on an inhomogeneous set of exposures.",
 "Howell S. B. 1989, PASP 101, 616 - Two-dimensional aperture photometry (CCD rovnice, optimalni apertura).",
 "Howell S. B., Warnock A., Mitchell K. J. 1988, AJ 95, 247 - Statisticka analiza promennosti s kontrolnimi hvezdami.",
 "Kasten F., Young A. T. 1989, Applied Optics 28, 4735 - Revised optical air mass tables.",
 "Jordi C. et al. 2010, A&A 523, A48 - Gaia broad band photometry (barevne transformace).",
 "Riello M. et al. 2021, A&A 649, A3 - Gaia EDR3 photometric passbands (BP-RP).",
 "Sokolovsky K. V. et al. 2017, MNRAS 464, 274 - Comparative performance of variability indices.",
 "von Neumann J. 1941, Ann. Math. Stat. 12, 367 - Ratio of mean square successive difference.",
 "Osborn J. et al. 2015, MNRAS 452, 1707 - Scintillation noise in astronomical photometry.",
 "Young A. T. 1967, AJ 72, 747 - Photometric error analysis (scintilace).",
 "Watson C., Henden A., Price A. 2006, SASS 25, 47 - AAVSO VSX katalog.",
 "Andrae R. et al. 2023, A&A 674, A27 - Gaia DR3 GSP-Phot (parametry hvezd pro HRD).",
])

# ================= ASSEMBLY =================
# section membership + key order from the commented config.json
raw = _cfgtxt
sections, cur = [], None
for line in raw.split('\n'):
    m = re.match(r'\s*// === (.+?) ===', line)
    if m:
        cur = (m.group(1), []); sections.append(cur); continue
    m = re.match(r'\s*"([A-Za-z0-9_]+)"\s*:', line)
    if m and cur is not None:
        cur[1].append(m.group(1))



story = []
story.append(Paragraph("VYVAR - Referencni prirucka parametru", st_title))
story.append(Paragraph("Detailni rozbor vsech %d parametru config.json: vyznam, rozsahy, duvody vychozich hodnot, dusledky zmen, matematicke pozadi a literatura." % len(CFG), st_intro))
story.append(Spacer(1, 8))
story.append(Paragraph("Stav: 2026-07-22, VYVAR HEAD (OSC-3). Registrovanych parametru: 266. Doprovodne dokumenty: VYVAR_CONFIG_GUIDE_CZ/EN.md (rychla reference), docs/VYVAR_PARAMS.md (strojovy index). Cestina bez diakritiky dle konvence projektu.", st_meta))
story.append(Spacer(1, 10))
story.append(Paragraph("Jak cist tuto prirucku", st_h1))
story.append(Paragraph("Kazdy parametr ma: vyznam, aktualni hodnotu (z vaseho config.json) a tam, kde na hodnote zalezi, rozbor [b]Mensi / Vetsi / Proc vychozi[/b]. Vedecky nabite skupiny maji oranzove BOXY s matematikou a odkazy. Parametry bez rozboru jsou bezpecne 'jasne' - jejich vyznam plne popisuje prvni veta.", st_body))
story.append(Paragraph("Tri druhy parametru (viz hlavicka config.json): [b]Nastaveni[/b] ziji zde; [b]staticka fakta[/b] (stanoviste, kamera, dalekohled) ziji v databazi; [b]dynamicke hodnoty[/b] (gain, meritko, rozmery) se ctou z FITS za behu. Menit kopie faktu zde nema na beh vliv.", st_body))
story.append(PageBreak())

n_detail = n_deep = 0
for sec_title, keys in sections:
    story.append(Paragraph(sec_title, st_h1))
    if sec_title in SEC_INTRO:
        story.append(Paragraph(SEC_INTRO[sec_title], st_intro))
    for k in keys:
        if k not in CFG:  # safety
            continue
        block = []
        if k in DEEP:
            t, paras = DEEP[k]; n_deep += 1
            story.append(Spacer(1, 4)); story.append(deepbox(t, paras)); story.append(Spacer(1, 4))
        g = GUIDE.get(k, {})
        block.append(Paragraph(k, st_h2))
        meta = "Hodnota: [b]%s[/b]" % fmt_default(CFG[k])
        d = DETAIL.get(k, {})
        if d.get('rozsah'): meta += "   |   Rozsah: %s" % d['rozsah']
        elif g.get('default') and g['default'].strip() not in ('', '-'):
            pass
        block.append(Paragraph(meta, st_meta))
        vysv = g.get('vysv', '').replace('\\|','|')
        if vysv: block.append(Paragraph(vysv, st_body))
        if d.get('proc'):  block.append(Paragraph("[b]Proc vychozi:[/b] " + d['proc'], st_eff))
        if d.get('mensi'): block.append(Paragraph("[b]Mensi hodnota:[/b] " + d['mensi'], st_eff))
        if d.get('vetsi'): block.append(Paragraph("[b]Vetsi hodnota:[/b] " + d['vetsi'], st_eff))
        if d: n_detail += 1
        story.append(KeepTogether(block))
        story.append(Spacer(1, 3))
    story.append(PageBreak())

story.append(Paragraph("Literatura", st_h1))
for ref in LIT:
    story.append(Paragraph(ref, st_body))

doc = SimpleDocTemplate(os.path.join(ROOT,'docs','VYVAR_PARAMETER_HANDBOOK_CZ.pdf'), pagesize=A4,
                        leftMargin=20*mm, rightMargin=20*mm, topMargin=16*mm, bottomMargin=16*mm,
                        title='VYVAR - Referencni prirucka parametru', author='VYVAR project')
doc.build(story)
print('built: params=%d, detailed=%d, deep boxes=%d' % (len(CFG), n_detail, n_deep))
