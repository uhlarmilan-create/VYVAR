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
D['bpm_dark_mad_sigma'] = dict(
 rozsah="Typicky 5-10; nizsi = agresivnejsi mapa vadnych pixelu.",
 mensi="Vic pixelu oznacenych za vadne - bezpecnejsi vuci horkym pixelum, ale roste sance vyradit zdrave pixely (ztrata plochy, interpolace).",
 vetsi="Mene vadnych pixelu zachyceno; horke pixely mohou prezit a chovat se jako mikro-hvezdy.",
 proc="MAD (median absolute deviation) je robustni vuci samotnym vadnym pixelum; volba v jednotkach robustni sigmy dava stabilni prah nezavisly na urovni signalu.")
D['cal_diag_gate_enabled'] = dict(
 proc="Fail-closed pojistka: po odectu darku musi median oblohy zustat fyzikalne smysluplny. Chrani pred zamenou konvence masteru (SUM vs MEAN) a spatnym parovanim masteru - trida chyb, ktera jinak POTICHU znici celou noc.",
 mensi="(vypnuto) Pipeline zpracuje i nesmyslne kalibrovana data - nedoporucujeme vypinat.",
 vetsi="(zapnuto, vychozi) Pri anomalii se beh zastavi s jasnou hlaskou.")
D['cal_diag_rel_tol'] = dict(
 rozsah="0.01-0.10; vychozi 0.02 (2 %).",
 mensi="Prisnejsi krizova kontrola urovni dark vs snimek; vice falesnych poplachu pri promenlivem pozadi (Mesic, svitani).",
 vetsi="Volnejsi kontrola; skutecny nesoulad konvence muze proklouznout.")
D['cal_diag_hard_sigma'] = dict(
 mensi="Citlivejsi tvrda brana - vic zastaveni.",
 vetsi="Brana zachyti jen extremni pripady.",
 proc="Dvoustupnova logika: mekka tolerance (rel_tol) varuje/koriguje, tvrda sigma zastavi.")
D['cal_diag_autocorrect_enabled'] = dict(
 proc="Kdyz diagnostika bezpecne rozpozna znamou zamenu (SUM misto MEAN stack), umi ji prepocitat misto zastaveni. Oprava je zapsana do provenance.",
 mensi="(vypnuto) Kazdy nesoulad = zastaveni; vhodne pri ladeni knihovny.",
 vetsi="(zapnuto) Pohodlnejsi provoz; oprava je vzdy dohledatelna v reportu.")
D['calibration_master_ccd_temp_tolerance_c'] = dict(
 rozsah="0.1-5 C; vychozi 0.5 C.",
 mensi="Prisnejsi parovani darku dle teploty - nejcistsi kalibrace, ale nemusi se najit zadny master.",
 vetsi="Povoli dark z jine teploty; temny proud pak nesedi (~15 %/C u typicke CMOS) a po odectu zustane zbytek nebo prekompenzace.",
 proc="0.5 C odpovida beznemu kolisani regulace chlazeni; vetsi rozdil uz je meritelny v pozadi.")
D['calibration_library_native_binning'] = dict(
 proc="Knihovna je postavena v jednom binningu; pri behu s jinym binningem se master prevzorkuje (se zachovanim toku) a beh dostane provenance priznak dark_resample. Filozofie: prizpusob se, nepredpokladej.")
D['dao_qc_in_calibrate'] = dict(
 proc="Detekce hvezd uz behem kalibrace = metriky kvality jsou k dispozici okamzite a spatne snimky lze vyradit pred drahymi kroky.")

# ================= QC =================
D['qc_dao_detection_sigma'] = dict(
 rozsah="2.5-6; vychozi hodnota viz config.",
 mensi="Citlivejsi detekce: vice slabych hvezd, ale prudce roste pocet falesnych detekci ze sumu (viz box o detekcni sigme). QC pocty hvezd pak nadhodnocene.",
 vetsi="Robustnejsi pocty, ale slabe hvezdy zmizi; na ridkych polich muze snimek neprojit qc_min_stars.",
 proc="QC potrebuje STABILNI pocet hvezd napric snimky, ne uplnost - proto byva o neco prisnejsi nez science detekce.")
D['qc_fwhm_limit'] = dict(
 mensi="Prisnejsi limit ostrosti - vyradi vic snimku; dobra noc prezije, spatna se smrskne.",
 vetsi="Projdou i mekke snimky; sirsi PSF = vice prekryvu sousedu a horsi SNR v aperture.",
 proc="Pouziva se jen pri vypnutem auto-FWHM; absolutni pojistka pro noci bez spolehlive statistiky.")
D['auto_fwhm_enabled'] = dict(
 proc="Pevny limit netusi, jaka byla noc. Auto rezim odvodi prah z medianu FWHM teto noci (x k-faktor) - snimky se posuzuji vuci vlastnimu seeingu, ne vuci idealu.")
D['auto_fwhm_k_factor'] = dict(
 rozsah="mezi k_min a k_max.",
 mensi="Prisnejsi (blize medianu) - vyrazuje i mirne zhorsene snimky.",
 vetsi="Tolerantnejsi; propusti chvosty seeingu.",
 proc="Nasobitel medianu: median x k = prah. Typicky 1.3-1.8: drzi jadro noci, rezne zjevne zhorseni (vitr, mrak, rozostreni).")
D['qc_elong_limit'] = dict(
 rozsah="1.0 = dokonaly kruh; vychozi ~1.5-1.8.",
 mensi="Prisnejsi na carkovani (vedeni/vitr); vic vyrazenych snimku.",
 vetsi="Protazene hvezdy projdou; aperturni fotometrie kruhem pak ztraci tok smerem podel carky a zvysuje rozptyl.",
 proc="Elongace a/b je citlivy indikator problemu montaze drive, nez je videt okem.")
D['qc_min_stars'] = dict(
 mensi="I temer prazdny snimek projde; alignment a fotometrie pak mohou selhat pozdeji a osklive.",
 vetsi="Ridka pole (wide rig, kratke expozice) nemusi projit vubec.",
 proc="Minimalni pocet pro spolehlivy alignment (kontrolni body) a smysluplne QC statistiky.")
D['qc_max_hfr'] = dict(
 proc="HFR (half-flux radius) je mira ostrosti znama ze snimaciho SW (NINA/Ekos) - stejna metrika umozni porovnat, co videl capture a co vidi VYVAR.")
D['preprocess_sky_surface_order'] = dict(
 rozsah="0 (vypnuto), 1 (rovina), 2 (kvadraticka plocha; vychozi a soucast anchoru).",
 mensi="0/1: gradienty oblohy (Mesic, svitani, vinetace zbytkova po flatu) zustanou v datech a prosaknou do mezikruzi -> systematicky posun pozadi zavisly na poloze.",
 vetsi="Vyssi rad neni povolen zamerne: flexibilnejsi plocha uz by zacala pojidat realne velkoskalove struktury (mlhoviny!) a lokalni pozadi hvezd - viz box.",
 proc="Rad 2 odstrani presne to, co je systematicke (gradient + zakriveni), a nic vic. Fit zachovava tok (flux-conserving).")
D['frame_quality_gate_enabled'] = dict(
 proc="Relativni brana: vyrazuje snimky vyrazne horsi nez typicky snimek TE NOCI (pomer ratio_k). Doplnuje absolutni limity.")
D['frame_quality_ratio_k'] = dict(
 mensi="Prisnejsi - i mirne horsi snimky leti.",
 vetsi="Jen katastrofy jsou vyrazeny.",
 proc="Pomer vuci medianu noci; typicky 1.5-2.5. Chranen min_keep_frames, aby brana nikdy nesnedla noc.")
D['frame_align_residual_gate_enabled'] = dict(
 proc="Snimek se muze zdat ostry, a presto se spatne sesadit (oblacnost, carkovani v casti pole). Rezidua zarovnani jsou nezavisly signal kvality.")
D['qc_max_background_rms'] = dict(
 proc="None = kontrola vypnuta (vychozi). Uzitecne zapnout pri honbe za mraky: sum pozadi roste s oblacnosti drive nez klesa pocet hvezd.")

# ================= ALIGNMENT =================
D['alignment_detection_sigma'] = dict(
 mensi="Vice kontrolnich bodu vc. slabych - parovac ma vic prace a slabe body pridavaji sum do transformace.",
 vetsi="Mene, ale kvalitnich bodu; na ridkych polich muze byt bodu malo.",
 proc="Alignment potrebuje JASNE a STABILNI hvezdy; stredne prisna detekce je optimum.")
D['alignment_max_control_points'] = dict(
 mensi="Rychlejsi, ale mene redundance vuci spatnym parum.",
 vetsi="Robustnejsi odhad transformace, pomalejsi; nad ~100 bodu uz presnost afinni transformace nerostie.",
 proc="Afinni transformace ma 6 stupnu volnosti - desitky kvalitnich bodu bohate staci, zbytek je pojistka proti outlierum.")
D['alignment_max_stars'] = dict(
 proc="Strop kandidatu pro parovac (vykonovy limit kombinatoriky trojuhelnikoveho matchingu).")

# ================= DETEKCE / MASTERSTAR / VARIABILITA / BLIND =================
D['masterstar_dao_threshold_sigma'] = dict(
 mensi="Hlubsi katalog (slabsi hvezdy), ale falesne detekce rostou exponencialne (box o sigme); DAO-RECONCILE hlida completeness vuci Gaia.",
 vetsi="Cistsi katalog, ale ztrata slabeho konce - miss@G90 metrika by rostla.",
 proc="Vychozi vyvazuje completeness ~90 % pri zanedbatelnem poctu falesnych; overeno reconciliaci vuci Gaia DR3 (fadezone analyza).")
D['masterstar_dao_pass2_sigma'] = dict(
 proc="Druhy, hlubsi pruchod na stacku: stack ma nizsi sum, takze si muze dovolit nizsi prah nez jednotlive snimky.")
D['saturate_limit_fraction'] = dict(
 rozsah="0.5-1.0; vychozi 0.85.",
 mensi="Konzervativnejsi - vic jasnych hvezd vylouceno; prijdete o jasne komparacni kandidaty.",
 vetsi="Riziko mereni v nelinearni oblasti: tok jasnych hvezd je systematicky PODCENEN, cimz se pokrivi diferencialni magnitudy vuci jasnym kompum - viz box o saturaci.",
 proc="Detektory ztraceji linearitu pod plnou studni; 15% rezerva kryje typicky nastup nelinearity CMOS. Absolutni strop (ADU) se bere z FITS/DB per kamera.")
D['phase01_match_radius_arcsec'] = dict(
 mensi="Mene falesnych ztotozneni s Gaia, ale pri horsim WCS/seeingu prijdete o prava sparovani.",
 vetsi="Vic sparovani, ale v hustych polich roste riziko zameny souseda (spatna identita = fatalni pro fotometrii promenne!).",
 proc="Volba ~ nekolik nasobku astrometricke chyby (WCS RMS + centroid) - typicky 2-3 arcsec na wide, mene na Newtonu.")
D['field_density_sparse_threshold'] = dict(
 mensi="Adaptace na ridka pole nastane pozdeji - na wide poli hrozi prazdne comp pooly.",
 vetsi="I normalni pole dostanou uvolnena kriteria - zbytecne slabsi vyber.",
 proc="Prah v poctu sparovanych hvezd; kalibrovan na realnych polich vsech tri sestav (DENSITY_OVERRIDES).")
D['variability_sigma_threshold'] = dict(
 mensi="Citlivejsi vyhledavani promennosti - vice kandidatu, vice falesnych (kazda systematika je 'promennost').",
 vetsi="Jen vyrazne promenne; male amplitudy uniknou.",
 proc="Prah nad sumovym modelem RMS(mag); spolu s VDI a smoothness tvori vicekriterialni sito (Sokolovsky 2017).")
D['variability_vdi_z_threshold'] = dict(
 proc="VDI kombinuje von Neumannuv pomer (1941) - citlivy na KORELOVANOU zmenu bod po bodu, tedy skutecnou krivku, ne bily sum - se z-skorovanim vuci poli.")
D['variability_comp_floor_factor'] = dict(
 mensi="Kandidati tesne nad sumem kompu - hodne falesnych.",
 vetsi="Jen jasne nadprahove; konzervativni.",
 proc="Hvezda musi kolisat NASOBNE vic nez srovnavaci hvezdy stejne jasnosti - pojistka proti spolecnym systematikam.")
D['variability_smoothness_max'] = dict(
 proc="Prilis hladka 'krivka' je typicky trend (airmass, teplota), ne hvezda; skutecna promennost ma strukturu na vice casovych skalach.")
D['blind_verify_min_fraction'] = dict(
 mensi="Overeni pusti i slabsi shody - riziko prijeti spatneho pole je male, ale nenulove.",
 vetsi="Prisne; na chudych polich se spravne reseni nemusi overit.",
 proc="Podil katalogovych hvezd, ktere se musi najit v obraze. Slepe reseni bez overeni proti Gaia NIKDY neverime - to je zakladni pojistka.")
D['blind_scale_tol_frac'] = dict(
 proc="Meritko sestavy zname presne (optika se nemeni); uzka tolerance okamzite zabiji 99 % falesnych kandidatu jeste pred drahym overenim.")
D['blind_use_rig_prior'] = dict(
 proc="Prior meritka = obrovske zrychleni; vypnout jen pri zcela nezname optice.")
D['blind_img_star_budget'] = dict(
 mensi="Rychlejsi, ale na chudych snimcich nemusi stacit na kvadove shody.",
 vetsi="Pomalejsi kombinatorika (roste ~N^4 u ctveric).",
 proc="Rozpocet drzi reseni v sekundach; per_cell vyber zaruci pokryti celeho pole.")
D['verify_mag_limit'] = dict(
 proc="Overovaci katalog do mag 14 je stejne spolehlivy jako 16 a o ~28 % rychlejsi (zmereno) - slabsi hvezdy overeni nepridavaji, jen zpomaluji.")
D['epsf_min_stars'] = dict(
 proc="ePSF model z mene nez ~10 hvezd je nestabilni (sum jednotlivych hvezd se propise do modelu); PSF program je stejne vypnut do validace.")
D['exoplanet_match_max_sep_arcsec'] = dict(
 proc="Ztotozneni s hostiteli exoplanet; mala tolerance staci, katalogove pozice jsou presne.")

# ================= FOTOMETRIE =================
D['aperture_snr_sizing'] = dict(
 proc="Mapa trid jasnosti -> nasobek FWHM. Jasne hvezdy snesou vetsi aperturu (zachyti kridla PSF), slabe potrebuji malou (sum pozadi roste s plochou ~r[sup]2[/sup]) - viz box o Howellove rovnici.",
 mensi="(faktory) Mensi apertury: lepsi SNR slabych, ale vetsi citlivost na chyby centrovani a variace PSF; aperturni korekce musi vic pracovat.",
 vetsi="Vic zachyceneho toku a robustnost, ale sum pozadi a sousede; slabe hvezdy ztraceji.")
D['annulus_inner_fwhm'] = dict(
 mensi="Mezikruzi blize hvezde - kontaminace kridly PSF samotne hvezdy (pozadi nadhodnocene, tok podhodnoceny).",
 vetsi="Cistejsi od kridel, ale pozadi se meri dal od hvezdy (gradienty).",
 proc="Vychozi ~3x FWHM je za >99 % toku Gaussovskeho profilu; density adaptace smi na hustem poli zprisnit.")
D['annulus_outer_fwhm'] = dict(
 proc="Sirka mezikruzi urcuje pocet pixelu pro odhad pozadi - potrebujeme stovky pixelu, aby sum medianu pozadi byl zanedbatelny vuci sumu apertury.")
D['aperture_correction_enabled'] = dict(
 proc="Mala apertura ztraci definovany podil toku; korekce ho vraci pomoci jasnych izolovanych hvezd (pomer velka/mala apertura). Nutna, kdykoliv se apertury lisi mezi hvezdami.")
D['aperture_correction_min_ref_stars'] = dict(
 mensi="Korekce z mala hvezd = sum korekce se propise vsem;",
 vetsi="Robustni, ale na ridkem poli nemusi byt dost referenci a korekce se nepouzije.",
 proc="Median z >=5 referenci potlaci vliv jedne spatne.")
D['gs11_dilution_enabled'] = dict(
 proc="Dilucia = cizi svetlo v aperture. Z Gaia kataloga vime, kdo je pobliz a jak jasny - kontaminaci lze predpovedet bez mereni. Kandidat s velkou diluci nesmi byt komparacni hvezda.")
D['gs11_comp_max_dilution'] = dict(
 mensi="Prisnejsi cistota kompu; mene kandidatu.",
 vetsi="Kontaminovane kompy: jejich 'konstantnost' je iluze - soused muze byt promenny nebo se meni seeing a s nim podil kontaminace.",
 proc="Par procent kontaminace uz vytvari mmag systematiky korelovane se seeingem.")
D['err_background_mode'] = dict(
 proc="'empirical': sum pozadi se MERI prazdnymi aperturami primo na snimku - zachyti i korelovany sum (vzory cteni, zbytky flatu), ktery by teoreticky vypocet z RN+sky podcenil.")
D['err_empty_apertures_n'] = dict(
 mensi="Rychlejsi, ale odhad sumu pozadi ma vetsi rozptyl.",
 vetsi="Presnejsi odhad, drazsi; ~50-100 apertur je optimum (chyba odhadu sigma ~ 1/sqrt(2N)).")
D['sigma_sys_mag'] = dict(
 proc="Systematicke dno chyb po pasmech (pridava se kvadraticky). Zadna pipeline nedosahne ciste Poissonovske chyby - scintilace, flat, barva. Hodnota {'4': 0.018} = 18 mmag pro pasmo 4 je EMPIRICKA: zmerena z rozptylu kontrolnich hvezd po odectu formalni chyby (Honeycutt rezidua). Bez dna by exporty tvrdily nerealne male chyby.",
 mensi="Chyby v exportech podcenene -> prilis sebevedome body v AAVSO.",
 vetsi="Chyby nadhodnocene -> realna promennost muze byt statisticky 'neviditelna'.")
D['neighbor_sub_refuse_sep_fwhm'] = dict(
 proc="Pod ~0.8 FWHM uz dva profily nelze spolehlive rozdelit ani fitem - odecet by vnesl vic chyby, nez odstrani. Radeji poctive odmitnout.")
D['neighbor_sub_min_recovered_snr'] = dict(
 proc="Po odectu souseda musi cili zbyt pouzitelny signal; jinak je vysledek numericky sum a mereni se zahodi.")
D['psf_photometry_enabled'] = dict(
 proc="VYPNUTO zamerne: aperturni fotometrie je na nasich polich validovana (AIJ, Delta<0.001 mag); PSF ceka na validaci na hustem poli Newtonu. Zapnuti bez validace = neznama systematika.")
D['sysrem_enabled'] = dict(
 proc="VYPNUTO zamerne. SysRem (Tamuz+2005) odstranuje spolecne trendy - ale nase injekcni testy ukazaly, ze pri malem poctu hvezd POJIDA I SKUTECNOU PROMENNOST (nerozezna ji od systematiky). Bezpecne az pri stovkach hvezd a vice nocich.")
D['temporal_binning_enabled'] = dict(
 proc="VYPNUTO na zaklade tvrdych dat: populacni test 25 cilu - binning zhorsil 24, nepomohl zadnemu. Vyhlazene kompy prestanou sledovat rychle zmeny pruzracnosti a chyba se INJEKTUJE do diferencialni krivky. Chcete-li hladsi krivku, binujte az VYSLEDEK, ne vstup.")
D['democratic_detrend_enabled'] = dict(
 proc="VYPNUTO zamerne (stejna rodina rizik jako SysRem): medianovy trend pole obsahuje i prispevek cile, jeho odecteni deformuje amplitudu.")
D['savgol_detrend_enabled'] = dict(
 proc="VYPNUTO zamerne; viz democratic/sysrem. Vyhlazovaci detrendy patri do analyzy, ne do produkce dat.")
D['nonlinearity_fwhm_ratio'] = dict(
 proc="Diagnostika: v nelinearite jasne hvezdy 'tloustnou' (spicka se orizne, profil se rozsiri). Pomer FWHM jasne/slabe je levny detektor problemu bez fotonove-transferove krivky.")
D['sky_adu_fallback'] = dict(
 proc="Nouzova hodnota pozadi, kdyz selze mereni (prazdny roh pole). Nastavte na typicke pozadi sve oblohy; pouziti je vzdy zapsano v provenance.")
D['pytics_enabled'] = dict(
 proc="Iteracni rekalibrace kompu (v duchu PyTICS): kazdy komp je docasne 'cil' vuci ostatnim - odhali nestabilni kompy, ktere prosly statickym vyberem.")

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
D['phase01_comparison_max_mag_diff'] = dict(
 mensi="Kompy jasnostne blize cili - podobny SNR rezim (dobre), ale mene kandidatu.",
 vetsi="Velky rozdil jasnosti = jiny rezim chyb (slaby komp pridava sum, jasny riskuje nelinearitu/saturaci).",
 proc="Adaptovano profilem hustoty; absolutni strop max_mag_diff_absolute nikdy neprekrocitelny.")
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
D['comp_iterative_clip_enabled'] = dict(
 proc="ZAPNUTO v produkci (od brnenskeho fixu 06/2026): po prvnim sestaveni krivky se kompy preveri proti souboru a odlehle se vyradi/prevazi - iterace 'vyrad a prevaz' konverguje k cistemu souboru.")
D['comp_sparse_fallback_enabled'] = dict(
 proc="Na extremne ridkem poli je 1 slabsi komp lepsi nez zadny - fallback to umozni, ale vysledek nese YELLOW/RED trust, nikdy GREEN.")
D['phase01_comparison_exclude_gaia_nss'] = dict(
 proc="Gaia non-single-star = znama dvojhvezda; potencialne promenna (zakryty, elipsoidalni) - jako komp nepripustna.")
D['phase01_use_bprp_primary'] = dict(
 proc="BP-RP primo z Gaia je homogenni, presna barva pro cele nebe - lepsi nez pocitane B-V pres transformace (Jordi 2010, Riello 2021), ktere pridavaji chybu transformace.")
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
 proc="'literature' = koeficienty 2. radu z literatury dle pasma a barvy (bezpecny vychozi stav). 'fit' bude az po validaci NIGHT_FIT v2 (vyzaduje noc s dX>=0.3).")
D['k2_defaults_bprp'] = dict(
 proc="Literaturni k'' po pasmech vuci BP-RP; typicky -0.03 az -0.06 mag/airmass/mag(barvy) pro modra pasma, ~0 pro cervena. Viz box o extinkci.")
D['k2_fit_enabled'] = dict(
 proc="VYPNUTO do validace: fit k'' z jedne noci je snadno degenerovany s trendy (monotonni airmass!). Brana vyzaduje dX>=0.3, zlom monotonie a residual floor << 15 mmag - presne to hleda TOI-1131 gate scoring.")
D['k2_fit_lit_factor'] = dict(
 proc="Fit smi vyjit jen v nasobku literaturni hodnoty - k''=+0.5 je fyzikalne nesmysl a znamena, ze fit chytil systematiku, ne extinkci.")
D['apply_color_term'] = dict(
 proc="'off' = instrumentalni system (CV pro AAVSO). Transformace na standardni system vyzaduje filtrovana data a stabilni barevny clen - budouci rozsireni.")

# HRD / EXPORT / SYSTEM / PATHS (kratke)
D['hrd_parallax_snr_min'] = dict(
 proc="Paralaxa s SNR<5 dava pri prevodu na vzdalenost silne vychylene absolutni magnitudy (Lutz-Kelkerova trida efektu) - takove hvezdy do HRD nepatri.")
D['hrd_online_enrich_enabled'] = dict(
 proc="Online obohaceni (Gaia/SIMBAD) bezi jen pro report a je omezene caps/timeouty - vypadek site nikdy neblokuje fotometrii.")
D['tess_enabled'] = dict(
 proc="Krizova kontrola kandidatu promennosti proti TESS (Lightkurve): perioda/amplituda z vesmiru je nejsilnejsi nezavisle potvrzeni; pozor na blending 21'' pixelu TESS.")
D['qc_preprocess_workers'] = dict(
 proc="Auto z CPU/RAM; rucni prepis jen pri sdilenem stroji. Vic workeru nez jader nic nezrychli; malo RAM na worker vede k swapovani (per_frame_mp_reserve_ram_gb drzi rezervu).")

# ================= DEEP DIVE BOXY =================
DEEP['aperture_snr_sizing'] = ("BOX: Optimalni apertura a Howellova CCD rovnice (Howell 1989)", [
 "Pomer signal/sum aperturni fotometrie: [b]SNR = N[sub]*[/sub] / sqrt(N[sub]*[/sub] + n[sub]pix[/sub](N[sub]sky[/sub] + N[sub]dark[/sub] + RN[sup]2[/sup]))[/b], kde N[sub]*[/sub] je tok hvezdy v elektronech, n[sub]pix[/sub] pocet pixelu apertury, N[sub]sky[/sub] pozadi na pixel a RN sumove cteni.",
 "Rust polomeru r ma dva protichudne efekty: N[sub]*[/sub](r) roste k plnemu toku (u Gaussova profilu je v r=1x FWHM ~93 % toku, v 1.5x FWHM ~99 %), ale sumove cleny rostou s plochou n[sub]pix[/sub] ~ pi r[sup]2[/sup]. Pro slabe hvezdy (N[sub]*[/sub] << n[sub]pix[/sub]N[sub]sky[/sub]) vychazi maximum SNR kolem r ~ 0.7x FWHM; pro jasne hvezdy (fotonovy sum hvezdy dominuje) SNR s r uz temer neroste a vetsi apertura kupuje robustnost vuci centrovani a variacim PSF za zanedbatelnou cenu.",
 "Presne proto ma aperture_snr_sizing tridy: slabe hvezdy male faktory, jasne velke. Rozdilne apertury mezi hvezdami pak NUTNE vyzaduji aperturni korekci (aperture_correction_*), ktera preskaluje toky na spolecnou skalu pomoci jasnych izolovanych referenci."])

DEEP['phase01_comparison_n_comp_min'] = ("BOX: Kolik srovnavacich hvezd? (Broeg 2005, Honeycutt 1992, scintilace)", [
 "Diferencialni magnituda cile vuci vazenemu souboru kompu ma sum: [b]sigma[sub]dif[/sub][sup]2[/sup] = sigma[sub]cil[/sub][sup]2[/sup] + sigma[sub]soubor[/sub][sup]2[/sup][/b], kde pro N srovnatelnych kompu [b]sigma[sub]soubor[/sub] ~ sigma[sub]komp[/sub]/sqrt(N)[/b]. Prechod z 1 na 4 kompy tedy snizi prispevek souboru 2x; z 4 na 8 uz jen 1.4x; z 8 na 16 dalsich 1.4x - vynosy klesaji.",
 "Zaroven existuje spodni mez, kterou zadny pocet kompu neprorazi: scintilace atmosfery (Young 1967, Osborn+2015) je pro vsechny hvezdy v poli castecne KORELOVANA, a systematiky (flat, barva) se prumerovanim neodstrani. Empiricky se prinos nasycuje kolem 6-8 kompu - proto n_comp_max=8.",
 "Broeg (2005): vahy kompu se voli podle jejich zmerene variability (w ~ 1/sigma[sup]2[/sup]) - 'spatny' komp se sam potlaci. Honeycutt (1992): spolecne nocni odchylky (mraky, pruzracnost) se resi soustavou epochovych korekci - VYVAR pouziva flux-sum kanonickou variantu validovanou proti AIJ.",
 "Dusledek pro uzivatele: 3 dobre BAREVNE SHODNE kompy jsou lepsi nez 10 spatnych. Proto je barva (comp_color_tiers) prisnejsi kriterium nez pocet."])

DEEP['comp_color_tiers'] = ("BOX: Barva kompu a extinkce druheho radu (Jordi 2010, Riello 2021, Kasten-Young 1989)", [
 "Atmosfera zeslabuje modre svetlo vic nez cervene. Pro hvezdy RUZNE barvy proto extinkce roste s airmass X ruzne rychle: [b]dm = k' X + k'' C X[/b], kde C je barva (u nas Gaia BP-RP) a k'' koeficient druheho radu, typicky -0.02 az -0.06 mag/airmass/mag pro siroka modra pasma.",
 "Diferencialni fotometrie prvni clen (k'X) dokonale vyrusi - je stejny pro cil i komp. Druhy clen se vyrusi JEN pri shode barev: zbytek je [b]d(dm) = k'' (C[sub]cil[/sub] - C[sub]komp[/sub]) X[/b]. Priklad: k''=-0.04, rozdil barev 0.5 mag, zmena airmass behem noci dX=0.5 -> falesny trend 10 mmag - vic nez amplituda mnoha promennych!",
 "Urovne 0.15/0.30/0.55/1.10 s vahami 1.0/0.85/0.5/0.25 drzi vazeny prispevek pod par mmag pro typicke k''. U nefiltrovaneho (CV) pozorovani, kde je efektivni pasmo siroke a k'' nejvetsi, je tesna barevna shoda HLAVNI obranou - zadny pocet kompu ji nenahradi.",
 "Airmass se pocita dle Kasten & Young (1989); planovany NIGHT_FIT v2 (k2_fit_*) umozni k'' fitovat z vlastnich dat, jakmile bude k dispozici noc s dostatecnym rozsahem airmass (dX>=0.3)."])

DEEP['comp_trust_min_comps'] = ("BOX: Trust model GREEN/YELLOW/RED a kontrolni hvezda (Howell, Warnock & Mitchell 1988)", [
 "VYVAR znamkuje kazdou epochu: GREEN = plna duvera (dost dobrych kompu + kontrolni hvezda sedi), YELLOW = mereni platne s vetsi chybou (malo kompu; sigma skalovana ~1/sqrt(N)), RED = chybovy model neplati.",
 "Kontrolni hvezda je 'promenna, o ktere vime, ze je stala': meri se stejne jako cil. T-statistika porovnava jeji namereny rozptyl s predpovezenou chybou: [b]T = sigma[sub]merena[/sub]/sigma[sub]predpoved[/sub][/b]. T~1 = model chyb sedi (GREEN do 1.5); T>>1 = neco systematickeho (RED nad 4). Nadmerny rozptyl X[sup]2[/sup] doplnuje test o absolutni miru.",
 "Od 06/2026 plati plynula degradace (COMP_DEGRADATION_SPEC): mene kompu uz NEZNAMENA zahozeni mereni - jen poctive vetsi chybu a YELLOW znacku. AAVSO minimum je 1 komp + check; nas GREEN prah je prisnejsi prave proto, aby znacka mela vahu."])

DEEP['qc_dao_detection_sigma'] = ("BOX: Detekcni prah a falesne detekce (statistika Gaussova chvostu)", [
 "Prah n sigma nad pozadim znamena pravdepodobnost falesneho piku na pixel ~ P(z>n): pro 3 sigma 1.3e-3, pro 3.5 sigma 2.3e-4, pro 4 sigma 3.2e-5, pro 5 sigma 2.9e-7.",
 "Snimek 6248x4176 (C3-26000) ma 26 mil. pixelu: prah 3 sigma da radove TISICE falesnych piku (nez je srazi tvarove filtry), 4 sigma stovky, 5 sigma jednotky. Proto detekcni sigmy nikdy nestavime pod ~3.5 a QC/masterstar pouzivaji ruzne prisne prahy dle ucelu (stabilita poctu vs uplnost katalogu).",
 "Skutecnou uplnost slabeho konce nehlidame sigmou, ale reconciliaci vuci Gaia DR3 (completeness ~90 %, miss@G90 metrika) - to je nezavisla kontrola, ze prah nesezral realne hvezdy."])

DEEP['preprocess_sky_surface_order'] = ("BOX: Proc sky-surface prave rad 2", [
 "Model pozadi je polynomialni plocha radu 2 (6 clenu: 1, x, y, x[sup]2[/sup], xy, y[sup]2[/sup]) fitovana robustne na cely snimek a odectena se zachovanim toku (median plochy se vrati zpet - odstranuje se jen TVAR, ne uroven).",
 "Rad 2 pokryva presne systematicke tvary: linearni gradient (Mesic nizko, svitani) a zakriveni (zbytkova vinetace po flatu). Rad 3+ by zacal sledovat realne struktury - mlhoviny, Mlecnou drahu, hala jasnych hvezd - a POTICHU by je odecital z dat. U promennych hvezd v mlhovinach (nase cile v h/chi Per!) by to byla vedecka chyba.",
 "Lokalni pozadi kazde hvezdy se stejne meri mezikruzim - sky-surface resi jen velkoskalovy tvar, ktery by mezikruzi vzdalenejsich kompu posouval ruzne. Zavedeno s anchorem draft_435; zmena radu = zmena vedy = novy anchor."])

DEEP['saturate_limit_fraction'] = ("BOX: Saturace, linearita a proc 0.85", [
 "CMOS/CCD pixel sbira elektrony do plne studne (full well). Blizko ni odezva prestava byt linearni: dalsi fotony pridavaji mene ADU. Fotometrie predpoklada linearitu - tok hvezdy s orezanou spickou je systematicky PODCENEN, a to vic pro jasnejsi hvezdy -> pokriveny jasovy zebricek.",
 "Absolutni strop se resi za behu: FITS klice (SATURATE/MAXLIN/...) maji prednost, pak EQUIPMENTS.SATURATE_ADU z databaze, nakonec bitova hloubka. Pouzitelny limit = strop x 0.85. Patnactiprocentni rezerva kryje typicky nastup nelinearity CMOS (poslednich ~10-15 % studne) a chybu odhadu stropu.",
 "Chcete-li presnejsi hranici pro svou kameru: zmerte fotonovou-transferovou krivku (variance vs signal) a zapiste skutecny linearni strop do EQUIPMENTS.SATURATE_ADU - frakce pak zustava jako bezpecnostni rezerva navrch. To je cistsi cesta nez menit frakci."])

DEEP['sysrem_enabled'] = ("BOX: Proc jsou detrendy v produkci vypnute (negativni vysledky maji cenu)", [
 "SysRem (Tamuz, Mazeh & Zucker 2005) a pribuzne metody (TFA, democratic/SavGol detrend, casove binovani) odstranuji SPOLECNE trendy mnoha hvezd. Funguji skvele na prehlidkach s tisici hvezd - a prave tam je jejich matematika doma: cil je zanedbatelna cast statistiky.",
 "Nase testy s injektovanym signalem (zname umele promenne) ukazaly opak pro mala pole: casove binovani zhorsilo 24 z 25 cilu (vyhlazene kompy prestanou sledovat rychle zmeny pruzracnosti - chyba se INJEKTUJE do krivky); SysRem pri malem poctu hvezd absorboval cast skutecne amplitudy (nerozezna promennost od systematiky).",
 "Zaver zapsany v DECISIONS: airmassove systematiky se resi PRICINOU (barevna shoda kompu, k'' korekce), ne kosmetikou signalu. Detrendovaci nastroje zustavaji dostupne pro ANALYZU, ale produkce dat je nepouziva. Tyto vypinace nechte vypnute, pokud presne nevite, proc je zapinate."])

DEEP['blind_verify_min_fraction'] = ("BOX: Slepe reseni souradnic a jeho overeni", [
 "Slepy solver nezna pointing: z jasnych hvezd obrazu stavi geometricke invarianty (trojuhelniky/kvady pres 8 nejblizsich sousedu), hleda je v predpocitanem indexu, kandidatni pozice sdruzuje hlasovanim (DBSCAN clustery) a WCS fituje robustne (RANSAC).",
 "Kazdy kandidat MUSI projit overenim proti Gaia: min. podil a pocet katalogovych hvezd nalezenych v obraze (verify_min_fraction/min_matches). Bez tohoto kroku by obcasna falesna shoda invariantu prosla - a spatna identifikace pole je nejhorsi mozna chyba (vsechna mereni pak patri jinym hvezdam).",
 "Meritko sestavy jako prior (scale_tol_frac) zabiji falesne kandidaty brzy a levne. Empirie: overovaci limit mag 14 je stejne spolehlivy jako 16 a o ~28 % rychlejsi - slabe hvezdy overeni nezpresnuji."])

DEEP['sigma_sys_mag'] = ("BOX: Chybovy rozpocet - odkud se bere 18 mmag", [
 "Uplna chyba bodu: [b]sigma[sup]2[/sup] = sigma[sub]foton[/sub][sup]2[/sup] + sigma[sub]pozadi[/sub][sup]2[/sup] + sigma[sub]soubor[/sub][sup]2[/sup] + sigma[sub]scint[/sub][sup]2[/sup] + sigma[sub]sys[/sub][sup]2[/sup][/b]. Prvni cleny umime spocitat (Howell + empiricke pozadi); scintilaci odhadnout (Osborn+2015: zavisi na aperture dalekohledu, airmass a rychlosti vetru); sigma_sys je poctive priznani zbytku (flat, barva, PSF variace).",
 "Kalibrace: na dobre noci se vezme rozptyl KONTROLNICH hvezd a odecte se kvadraticky formalni chyba - co zbyde, je systematicke dno. Pro wide sestavu v pasmu 4 vyslo ~18 mmag; jine sestavy/pasma budou mit jine dno (proto je klic slovnik po pasmech).",
 "Kontrola spravnosti bezi neustale: T-statistika kontrolni hvezdy ma vychazet ~1. T<1 znamena chyby nadhodnocene (dno prilis velke), T>1.5 podhodnocene."])

DEEP['gs11_dilution_enabled'] = ("BOX: Dilucia - cizi svetlo v aperture", [
 "Podil kontaminace: [b]d = F[sub]sousede[/sub] / (F[sub]hvezda[/sub] + F[sub]sousede[/sub])[/b], kde toky sousedu v aperture odhadneme z Gaia magnitud a profilu PSF - jeste PRED merenim.",
 "Proc je to pro kompy fatalni: kontaminovany komp vypada stabilne, dokud je seeing stabilni. Zmena FWHM zmeni podil sousedova svetla v aperture -> 'konstantni' komp se pohne synchronne se seeingem a vyrobi falesnou anti-korelaci v cili. Par procent dilucie = mmag systematiky.",
 "Pro cil naopak dilucia znamena PODCENENOU amplitudu (promenny signal je redeny konstantnim svetlem) - amplitudova korekce 1/(1-d) je mozna jen se znamym d."])

DEEP['variability_sigma_threshold'] = ("BOX: Hledani promennosti - vicekriterialni sito (Sokolovsky 2017, von Neumann 1941)", [
 "Jedina statistika nestaci: vysoke RMS ma i hvezda se spatnym pixelem. VYVAR kombinuje: (1) prebytek RMS nad obalkou RMS-mag pole (sigma_threshold, comp_floor_factor), (2) von Neumannuv pomer eta = mean((m[sub]i+1[/sub]-m[sub]i[/sub])[sup]2[/sup])/var(m) - skutecna krivka ma po sobe jdouci body KORELOVANE (eta male), bily sum ne (eta~2), (3) VDI z-skore vuci poli, (4) smoothness strop proti cistym trendum, (5) amplitudove a pokryti minima.",
 "Sokolovsky+2017 srovnal 18 indexu promennosti: kombinace korelacniho indexu (typ eta/VDI) s amplitudovym je nejrobustnejsi - presne tato dvojice je jadrem nasi kaskady.",
 "Prahy jsou zamerne konzervativni: kandidat projde do reportu, kde ho potvrdi/vyvrati clovek + TESS krizova kontrola - falesny kandidat stoji minutu, prehlednuta nova promenna je skoda navzdy."])

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
 'Photometry': "Srdce pipeline: apertury, pozadi, model chyb, korekce. Zde kazda hodnota primo hybe vysledkem - a zde jsou zamerne vypnute detrendy.",
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
 "Tamuz O., Mazeh T., Zucker S. 2005, MNRAS 356, 1466 - SysRem (vyhodnoceno, v produkci vypnuto).",
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


# inject important code-default-only keys into their sections
for sec_title, keys in sections:
    if sec_title.startswith('Detection'):
        keys.insert(keys.index('masterstar_dao_threshold_sigma') if 'masterstar_dao_threshold_sigma' in keys else 0, 'saturate_limit_fraction')
CFG['saturate_limit_fraction'] = 0.85  # code default (not persisted in config.json)
GUIDE.setdefault('saturate_limit_fraction', {})['vysv'] = GUIDE.get('saturate_limit_fraction',{}).get('vysv', 'Podil saturacni urovne detektoru, nad kterym se hvezda povazuje za saturovanou a vylouci se z fotometrie. POZOR: tento klic NENI v config.json - bezi na kodovem vychozim 0.85; zmenit ho lze pridanim radku do config.json nebo v UI (Expert / detection).')

story = []
story.append(Paragraph("VYVAR - Referencni prirucka parametru", st_title))
story.append(Paragraph("Detailni rozbor vsech %d parametru config.json: vyznam, rozsahy, duvody vychozich hodnot, dusledky zmen, matematicke pozadi a literatura." % len(CFG), st_intro))
story.append(Spacer(1, 8))
story.append(Paragraph("Stav: 2026-07-22, VYVAR HEAD 2c520c6. Registrovanych parametru: 270. Doprovodne dokumenty: VYVAR_CONFIG_GUIDE_CZ/EN.md (rychla reference), docs/VYVAR_PARAMS.md (strojovy index). Cestina bez diakritiky dle konvence projektu.", st_meta))
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
