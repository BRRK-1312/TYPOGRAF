---
 glyph-sheet-to-font
 
Glifo-orri baten irudia (eskuz edo IAz sortutako letra, zenbaki eta ikurren sareta edo antolaera) OTF letra-tipo funtzional bihurtzea. Erabili hau erabiltzaileak alfabeto/karaktere-multzo oso bat erakusten duen irudi bat igotzen duenean eta "bihurtu hau letra-tipo", "sortu tipografia bat honetatik", "egin hau instalagarri" edo antzekorik eskatzen duenean — nahiz eta "letra-tipo" hitza esplizituki erabili ez. Erabili, halaber, pipeline honekin aurrez eraikitako letra-tipo batek oinarri-lerro, eskala edo lerrokatze arazoak dituenean eta araztu behar denean.
---

# Glifo-orria → Letra-tipoa: pipeline-a

Marraztutako/sortutako glifoen irudi lau bat `.otf` fitxategi instalagarri bihurtzen du. Bi exekuzio errealetatik eraikia: antolaera librekoa (sareta-marrarik gabe) eta saretadun espezimen-orri bat ("KONTROL TYPE — GLYPH SET"). Bi exekuzioek benetako erroreak topatu zituzten; konponketak dokumentu honetan txertatuta daude, berriro ez errepikatzeko.

## Pipeline-aren ikuspegi orokorra

1. Aztertu jatorrizko irudia (polaritatea, antolaera, saretaduna edo librea)
2. Erauzi glifo bakoitzaren muga-kutxa (bounding box) modu sendoan (hemen bizi dira errore gehienak)
3. Moztu + alderantzikatu behar bada, bektorizatu glifo bakoitza SVGra `vtracer`-ekin
4. Analizatu SVG bideak (paths) eta marraztu CFF/OTF letra-tipo batean `fontTools`-ekin
5. Zuzendu eskala eta oinarri-lerroa — lerroz lerro / taldez talde, ez globalki
6. Errendatu aurrebista osoa oinarri-lerroaren gidalerroarekin eta egiaztatu bisualki entregatu aurretik

Beharrezko tresnak: `opencv-python-headless`, `numpy`, `vtracer`, `fonttools`, `cairosvg` (aurrebista errendatzeko soilik). Instalatu honela: `pip install --break-system-packages <pkg>`.

---

## 1. urratsa — Aztertu irudia

Egiaztatu polaritatea lehenik — honek geroagoko urrats bat baldintzatzen du, ahaztea erraza dena:

```python
import cv2
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray[5,5])      # izkinako/hondoko lagina
print(gray.max())     # pixel distiratsuena (glifoaren tinta, argia-ilunaren-gainean bada)
```

- **Glifo ilunak hondo argiaren gainean** (inprimatze-konbentzio arrunta) → ez da alderantzikatzerik behar bektorizatu aurretik.
- **Glifo argiak hondo ilunaren gainean** (stencil/grunge estiloko espezimen-orriak) → alderantzikatu behar da `vtracer` erabili aurretik, ikus 3. urratsa.

Gero, bilatu sareta bat. Orriak gelaxka banatzeko marra ikusgaiak baditu, ez saiatu marra horiek zuzenean detektatzen mozketarako egia-iturri gisa (ikus 2. urratseko tranpa) — lerro/zutabe kopuruak egiaztatzeko baino ez dira baliagarriak.

---

## 2. urratsa — Erauzi glifoen muga-kutxak (urrats kritikoa)

Hurbilketa xumea — irudia "distira"/"tinta" maskara batera atalasetu, eta gero *edozein* tinta-pixel duen *edozein* lerro bilatu — ez da sendoa eta errore isilak eta diagnostikatzen zailak sortzen ditu. Erabili hau haren ordez:

```python
import cv2, numpy as np
from collections import Counter

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ink = (gray > 140).astype(np.uint8) * 255   # alderantzikatu atalasearen noranzkoa iluna-argiaren-gainean bada

def significant_components_bbox(mask, min_area=30):
    """min_area gainditzen duten osagai konexu guztien muga-kutxen bilketa.
    Pixel bakarreko/gutxiko zarata iragazten du (sareta-marren isuria,
    JPEG artefaktuak), zati anitzeko glifoak (i, j, !, ?, :, ;) mantenduz."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    boxes = [stats[i] for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] >= min_area]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0]+b[2] for b in boxes); y1 = max(b[1]+b[3] for b in boxes)
    return x0, y0, x1, y1
```

**Zergatik den garrantzitsua — topatutako benetako errorea:** lehen saiakera batek `np.where(h_proj_local > 0)` erabili zuen (edozein pixel distiratsu "tinta" gisa zenbatzen da). KONTROL TYPE orrian, 1. eta 2. lerroak banatzen zituen sareta-marrak 146ko gris-balioan egin zuen gailurra — 140ko atalasea baino sei puntu gorago — antialiasing hutsagatik. Nahikoa izan zen metodo xumeak sareta-marra 1. lerroko letra guztien behealde gisa hartzeko, haien muga-kutxak ~170px puztuz eta lerro oso horren maiuskula-altuera neurtua zein oinarri-lerroa hondatuz. *Beste* lerroetako letrak ez ziren kaltetu, haien ondoko sareta-marrek 140tik justu *azpitik* egin zutelako gailurra (adib. 135) — zorte hutsa, ez seinale sendoa. Osagai konexuen hurbilketak hau konpontzen du: 1-2 pixeleko sareta-marra baten printzak ez du inoiz `min_area=30` lortzen; benetako letra-tintak, beti.

Prozesua:

```python
# 1. Proiekzio horizontala lerro-bandak aurkitzeko
h_proj = np.sum(ink, axis=1)
in_row = h_proj > h_proj.max() * 0.01
# → lerro-bandak True eskualde jarraituetatik, altuera >= ~40px goiburuko testua saltatzeko

# 2. Lerro-banda bakoitzaren barruan, proiekzio bertikala glifo-zutabeak aurkitzeko
v_proj = np.sum(ink[y0:y1, :], axis=0)
in_col = v_proj > 0
# → zutabe-bandak True eskualde jarraituetatik, zabalera >= ~12px zarata saltatzeko

# 3. Detektatutako zutabe bakoitzeko, exekutatu significant_components_bbox
#    zutabe-xerra horretan (lerroaren altuera osoa) glifoaren BENETAKO y0/y1
#    lortzeko, sareta-marren isuria eta beste edozein artefaktu txiki alde batera utzita.
```

Mapatu detektatutako glifoak espero diren karaktereetara **lerro-hurrenkeran** (`list('ABCDEFGHI')` etab.) eta egiaztatu kopurua bat datorrela aurrera jarraitu aurretik — vtracer-ek/erauzketak 10 zutabe aurkitzen baditu 9 espero zirenean, iragazi beharreko zarata dago (igo gutxieneko zutabe-zabalera), ez asmatu beharreko karaktererik.

---

## 3. urratsa — Moztu, alderantzikatu behar bada, bektorizatu

Moztu glifo bakoitza tarte txiki batekin (~6px), egiaztatutako muga-kutxa erabiliz. **Gorde mozketaren desplazamendu absolutua jatorrizko irudian** (`crop_x0, crop_y0`) — 5. urratsean behar da glifoen arteko oinarri-lerroaren lerrokatze zuzenerako. Ez baztertu moztu ondoren.

```python
PADDING = 6
cy0, cy1 = max(0, y0-PADDING), min(img.shape[0], y1+PADDING)
cx0, cx1 = max(0, x0-PADDING), min(img.shape[1], x1+PADDING)
crop = img[cy0:cy1, cx0:cx1]
# gorde cx0, cy0 manifestu batean mozketaren bidearekin batera
```

Jatorria argia-ilunaren-gainean bada (1. urratsa), alderantzikatu bektorizatu aurretik:

```python
inv = 255 - crop
cv2.imwrite(inv_path, inv)
```

**Zergatik:** `vtracer`-en modu binarioak tinta-iluna-paper-argiaren-gainean konbentzioa suposatzen du. Argia-ilunaren-gainean irudi bat zuzenean emanda, *hondoa* trazatzen du aurreko plano gisa, letraren negatibo fotografikoa den ingerada bat sortuz (silueta trinko bat, letraren forma zulo gisa duena — ondo dirudi jatorrizko koloreekin errendatuta, baina alderantziz dago letra-tipoaren ingerada gisa, honek tinta trinkoa behar baitu letra dagoen tokian, hain zuzen). Errendatu beti proba-glifo bat bektorizatu ondoren eta egiaztatu bisualki betegarria letraren forma dela, ez haren alderantzizkoa, glifo guztiak sortan prozesatu aurretik.

Bektorizatu:

```python
import vtracer
vtracer.convert_image_to_svg_py(
    inv_path, svg_path,
    colormode='binary', hierarchical='stacked', mode='spline',
    filter_speckle=3, corner_threshold=60, length_threshold=4.0,
    max_iterations=10, splice_threshold=45, path_precision=3,
)
```

---

## 4. urratsa — Analizatu SVGa eta eraiki letra-tipoa

Erabili `fontTools.fontBuilder.FontBuilder` CFF (OTF) irteerarekin — Bézier kubikoak zuzenean mapatzen dira SVGtik, kurba-bihurketarik gabe (TTFrekin ez bezala, hark koadratikoak behar baititu).

```python
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
import xml.etree.ElementTree as ET, re

def parse_svg(svg_file):
    root = ET.parse(svg_file).getroot()
    w, h = float(root.get('width', 100)), float(root.get('height', 100))
    paths = []
    for path in root.iter('{http://www.w3.org/2000/svg}path'):
        d = path.get('d', '')
        tx, ty = 0.0, 0.0
        m = re.search(r'translate\(([^,]+),([^)]+)\)', path.get('transform', ''))
        if m: tx, ty = float(m.group(1)), float(m.group(2))
        if d.strip(): paths.append((d, tx, ty))
    return w, h, paths

def tokenize(d):
    return re.findall(r'[MmLlCcZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d)
```

M/L/C/Z bideak pen batera marrazteko gutxieneko funtzioa (hedatu Q/H/V/S komandoekin bektorizatzaileak igortzen baditu — vtracer-ek `mode='spline'` moduan M/L/C/Z baino ez du igortzen):

```python
def draw_to_pen(pen, d, tx, ty, F):
    """F(x, y) -> (font_x, font_y) funtzioak egiten du koordenatu-eraldaketa — ikus 5. urratsa."""
    tokens = tokenize(d); i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not (len(tok) == 1 and tok.isalpha()): i += 1; continue
        cmd = tok; i += 1
        if cmd == 'M':
            x, y = float(tokens[i]), float(tokens[i+1]); i += 2
            pen.moveTo(F(x, y))
            while i+1 < len(tokens) and not tokens[i][0].isalpha():
                x, y = float(tokens[i]), float(tokens[i+1]); i += 2
                pen.lineTo(F(x, y))
        elif cmd == 'L':
            while i+1 < len(tokens) and not tokens[i][0].isalpha():
                x, y = float(tokens[i]), float(tokens[i+1]); i += 2
                pen.lineTo(F(x, y))
        elif cmd == 'C':
            while i+5 < len(tokens) and not tokens[i][0].isalpha():
                x1,y1 = float(tokens[i]),float(tokens[i+1])
                x2,y2 = float(tokens[i+2]),float(tokens[i+3])
                x,y   = float(tokens[i+4]),float(tokens[i+5])
                i += 6
                pen.curveTo(F(x1,y1), F(x2,y2), F(x,y))
        elif cmd in ('Z','z'):
            pen.closePath()
```

Letra-tipoaren muntaia (CFF konfigurazioa, karaktere-mapa, metrika-taulak):

```python
fb = FontBuilder(1000, isTTF=False)   # unitsPerEm=1000, CFF/OTF
fb.setupGlyphOrder(['.notdef'] + glyph_names)
fb.setupCharacterMap({ord(ch): name for ch, name in char_to_name.items()})

charstrings, metrics = {}, {}
for ch, name in ...:
    pen = T2CharStringPen(advance_width, None)
    draw_to_pen(pen, d, tx, ty, F)         # F, 5. urratsaren arabera
    charstrings[name] = pen.getCharString()
    metrics[name] = (advance_width, left_side_bearing)

fb.setupCFF(psName='MyFont-Regular',
            fontInfo={'version':'1.000','FullName':'My Font Regular',
                      'FamilyName':'My Font','Weight':'Regular'},
            charStringsDict=charstrings,
            privateDict={'defaultWidthX':0,'nominalWidthX':0})
fb.setupHorizontalMetrics(metrics)
fb.setupHorizontalHeader(ascent=780, descent=-280)
fb.setupNameTable({'familyName':'My Font','styleName':'Regular'})
fb.setupOS2(sTypoAscender=780, sTypoDescender=-280, sTypoLineGap=100,
            usWinAscent=780, usWinDescent=280,
            sxHeight=500, sCapHeight=700, fsType=0, achVendID='ABCD')
fb.setupPost()
fb.setupHead(unitsPerEm=1000)
fb.font.save(out_path)
```

Kontuan hartu `FontBuilder.setupCFF`-ren sinadura zehatza `(psName, fontInfo, charStringsDict, privateDict)` dela — parametroen posizioak/izenak garrantzitsuak dira eta ez datoz bat asmatuko zenukeenarekin (ez dago `nameStrings=` kwarg-ik).

---

## 5. urratsa — Eskala eta oinarri-lerroa (bigarren urrats kritikoa)

**Ez** normalizatu glifo bakoitza bere kabuz, bere mozketaren goia/behea xede-altuera finko batera mapatuz. Hori probatu zen eta bi huts-modu bereizi sortzen ditu:

1. **Lerroen arteko tamaina-inkoherentzia.** Jatorrizko orriko lerro desberdinetako letrek pixel-altuera desberdinak badituzte (oso ohikoa — eskuz edo IAz sortutako espezimen-orriak nekez dira pixelez koherenteak lerroen artean, begi-kolpe batean uniformeak diruditen arren), erreferentzia-letra bakar batetik eskala-faktore global bat ateratzeak eta lerro guztiei aplikatzeak beste lerroak txikiegi edo handiegi errendarazten ditu. *Atera beti eskala bereizi bat lerroko* (edo glifo-talde logikoko), guztiek letra-tipoaren unitateetan maiuskula-altuera bera xede hartuta.

2. **Beherakoak hautsita eta glifo flotatzaileak.** "Glifo honen beraren muga-kutxaren behealdea" oinarri-lerroari ainguratzea ondo dabil letra arruntekin, baina hausten da benetako beherakoa duen edozerekin (Q-ren isatsa, J-ren kakoa, koma, puntu eta koma) — haien muga-kutxaren behealdea benetako oinarri-lerroaren *azpitik* dago berez, beraz oinarri-lerrora behartzeak glifo osoa gorantz tiratzen du eta beherakoa desagertu egiten da. Gainera, ezin du bereizi benetako beherako bat eta diseinuz besterik gabe auzokideak baino laburragoa den glifo bat.

Eredu zuzena: **oinarri-lerroaren pixel-koordenatu partekatu bat eta eskala partekatu bat lerroko, irudiaren koordenatu absolutuen bidez aplikatuta**, ez glifoz glifoko muga-kutxaren normalizazioa.

```python
# Lerro bakoitzeko: aurkitu oinarri-lerro partekatua (lerroko glifo gehienen
# behe-y balioaren moda, benetako beherakoak dituzten letra bat edo birekiko sendoa)
# eta erreferentzia-altuera bat (oinarri-lerroa ken goi-y balioen moda).
from collections import Counter
baseline = Counter(bottom_ys).most_common(1)[0][0]
top_ref  = Counter(top_ys).most_common(1)[0][0]
ref_height = baseline - top_ref
row_scale = TARGET_CAP_HEIGHT / ref_height     # adib. TARGET_CAP_HEIGHT = 700

# Puntu bakoitzeko, bihurtu irudiaren pixel-koordenatu ABSOLUTUAK
# (ez mozketa-lokalak) letra-tipoaren unitateetara:
def F(x_local, y_local):
    abs_x = crop_x0 + tx + x_local
    abs_y = crop_y0 + ty + y_local
    font_x = (abs_x - crop_x0) * row_scale + LSB_UNITS
    font_y = (baseline - abs_y) * row_scale     # 0 oinarri-lerroan, positiboa gorantz
    return font_x, font_y
```

Formula bakar honek, lerro bateko glifo guztiei uniformeki aplikatuta, emaitza zuzenak sortzen ditu automatikoki kasu guztietarako, kasu berezirik gabe:

- Tinta-behealdea lerroaren oinarri-lerroaren berdina duen letra arrunt bat → bere behealdea zehazki `font_y = 0`-n kokatzen da.
- Q-ren isatsa, komaren isatsa, puntu eta komaren isatsa → haien punturik baxuenak `abs_y > baseline` du, beraz `font_y` negatiboa ateratzen da — benetako beherako bat, tamaina zuzenekoa, oinarri-lerroaren azpian. Kasu berezirik behar ez duena.
- Altuera ertaineko puntuazioa (marratxoa) edo oinarri-lerro azpikoa (azpimarra) → zehazki haien pixel absolutuek dioten tokian kokatuta, oinarri-lerroarekiko. **Ez** behartu hauek oinarri-lerroa "ukitzera" — ez dute hori egin behar.
- Auzokideak baino benetan laburragoa den letra bat **eta jatorrizko artean oinarri-lerroaren gainetik flotatzen duena** (hau proiektu honetako lehen letra-tipoko "I" batekin gertatu zen), formula honekin, flotatzen errendatuko da hemen ere — jatorria fidelki erreproduzituz. Horrelako kasu bakan batek praktikan gaizki ematen badu (normalean hala ematen du, jendeak maiuskula guztiak oinarri-lerroan pausatzea espero baitu idaztean), erabaki gardena eta ikusgarria da *glifo horri bakarrik* kasu berezia egitea, neurtutako pixel-desplazamendu bat gehituz bihurketaren aurretik — ez da arrazoia glifoz glifoko muga-kutxaren aingura-eredura globalki aldatzeko (horrek beherako guztiak hausten ditu, goiko 2. huts-moduan bezala). Egiaztatu berriro glifo "flotatzailea" benetan oinarri-lerroaren detekzio-artefaktu bat ez ote den (2. urratseko sareta-marren errorea) diseinu-berezitasun bat dela suposatu aurretik — proiektu honetan, hasieran horrelako bat zirudienak, egiaz, lerro bereko *beste* zortzi letrak zeuden gaizki, ez letra bakarra benetako salbuespena. Egiaztatu pixel-neurketa gordinekin nahita egindakotzat hartu aurretik.

**Ikur/puntuazio-lerroek eskuz aukeratutako erreferentzia-glifoa behar dute.** Modan oinarritutako erreferentzia-altueraren detekzio automatikoak (goikoak) suposatzen du lerroko glifo gehienek altuera bera dutela — egia alfabetiko/zenbakizko lerroetan, gezurra `! ? . , : ; - _ /` nahasten dituen lerro batean. Hor, behe-y ohikoena ziurrenik glifo txikiena izango da (puntua, koma), `ref_height` txiki eta oker bat emanez. Aukeratu eskuz altuera osoko glifo bat lerro horretan (`!` edo `?`) erreferentzia gisa:

```python
ref_height = baseline - (excl_mark_top_y)   # erabili '!' zehazki, ez Counter.most_common
```

---

## 6. urratsa — Balioztatu entregatu aurretik

Errendatu beti letra-tipoko glifo guztiak, lerroka taldekatuta, oinarri-lerroaren gidalerro batekin, emaitza bidali aurretik. Urrats hau saltatzea da goiko bi erroreak hasieran entregatzera iritsi izanaren arrazoia.

```python
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen

font = TTFont(otf_path)
cs = font['CFF '].cff.topDictIndex[0].CharStrings
hmtx = font['hmtx'].metrics
cmap = font.getBestCmap()

# lerro bakoitzaren testu-katerako, marraztu karaktere bakoitza SVGPathPen bidez,
# antolatu ezkerretik eskuinera hmtx-ren aurrerapen-zabalerak erabiliz, pilatu
# lerroak bertikalki, marraztu marra gorri bat lerroko oinarri-lerroaren y
# desplazamenduan, errendatu muntatutako SVGa cairosvg.svg2png()-rekin, eta ikusi.
```

Ondoren, egiaztatu zenbakiz `BoundsPen`-ekin — lerro bateko maiuskula arrunt guztiek ia `y0` bera eman behar dute (~0, antialiasing-etik datorren jitter negatibo/positibo txikia onargarria da) eta ia `y1` bera (lerroaren maiuskula-altueraren xedea). Beherakoak dituzten letrek (Q, J) nabarmen negatiboagoa den `y0` bat erakutsi behar dute; beste ezerk ez.

```python
from fontTools.pens.boundsPen import BoundsPen
for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    bp = BoundsPen(None); cs[ch].draw(bp)
    print(ch, bp.bounds)   # begiz: y0 0tik hurbil, y1 xede maiuskula-altueratik hurbil, Q/J beherakodunetan izan ezik
```

Zerbait okerra badirudi, bounds zerrenda hau da glifo edo lerro zehatza okerra zein den lokalizatzeko biderik azkarrena, errendatutako iruditik soilik asmatzen ibili beharrean.

---

## Ohiko tranpak — egiaztapen-zerrenda azkarra

- [ ] Polaritatea egiaztatuta (iluna-argiaren-gainean vs argia-ilunaren-gainean) eta alderantzikatuta bektorizatu aurretik, behar bazen
- [ ] Proba-glifo bat errendatuta bektorizazioaren ondoren, betegarria letra dela egiaztatzeko, ez haren negatibo fotografikoa
- [ ] Osagai konexuen muga-kutxak erabilita (ez atalase-gaineko-edozein-pixel modu xumea), sareta-marren / zarataren kutsadura saihesteko
- [ ] Eskala **lerroko/taldeko** kalkulatuta, ez eskala global bakarra erreferentzia-glifo bakar batetik
- [ ] Lerroko oinarri-lerro partekatua + pixel-koordenatu absolutuak erabilita kokapen bertikalerako, ez glifoz glifoko muga-kutxaren ainguraketa
- [ ] Altuera osoko erreferentzia-glifoa eskuz aukeratuta tamaina nahasiko puntuazio-lerroetarako
- [ ] Glifo-multzo osoa errendatuta oinarri-lerroaren gidalerroarekin eta `BoundsPen` egiaztapenak eginda entregatu aurretik
