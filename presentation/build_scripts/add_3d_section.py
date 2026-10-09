# Archival: this script edited an unpacked copy of presentation/elastica_snap_surface.pptx
# (./unpacked), in places via helpers from the Claude pptx skill (hard-coded paths).
# It is kept as a record of how the slides were built, not as a reusable tool.

"""Insert the "Snap surface in 3D" section (divider + 3 slides) into the unpacked deck.

Run from /home/goraksh/elastica_deck with the deck unpacked in ./unpacked:
    tools/bin/python add_3d_section.py
"""
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

SKILL = Path("/home/goraksh/.claude/skills/synced/6b76838e-25d3-4069-b11d-75f398bcbcbd_39040c30-50d1-4d41-910a-741dc4d038ea/pptx/scripts")
U = Path("unpacked")
EMU = 914400
SECTION = "The snap surface in 3D"


def add_slide(src, after):
    out = subprocess.run(["python", str(SKILL / "add_slide.py"), str(U), src, "--after", after],
                         check=True, capture_output=True, text=True).stdout
    return re.search(r"(slide\d+\.xml)", out.split("Created")[1]).group(1)


# ---------- XML builders (same styling as the pptxgenjs-built slides) ----------
_id = [100]


def nid():
    _id[0] += 1
    return _id[0]


def xfrm(x, y, w, h):
    return f'<a:xfrm><a:off x="{int(x*EMU)}" y="{int(y*EMU)}"/><a:ext cx="{int(w*EMU)}" cy="{int(h*EMU)}"/></a:xfrm>'


def runs(text, sz, color="tx1", b=False, i=False):
    """'f_x' -> f with subscript x."""
    out, pieces, last = [], [], 0
    for m in re.finditer(r"(?<![A-Za-z])([A-Za-zθφ])_([A-Za-z0-9]{1,3})(?![A-Za-z_])", text):
        pieces.append((text[last:m.start()] + m.group(1), False)); pieces.append((m.group(2), True)); last = m.end()
    pieces.append((text[last:], False))
    for t, sub in pieces:
        if not t:
            continue
        attrs = f'lang="en-US" sz="{sz}"' + (' b="1"' if b else '') + (' i="1"' if i else '') + (' baseline="-25000"' if sub else '')
        out.append(f'<a:r><a:rPr {attrs} dirty="0"><a:solidFill><a:schemeClr val="{color}"/></a:solidFill></a:rPr><a:t>{escape(t)}</a:t></a:r>')
    return "".join(out)


def para(text, sz, color="tx1", b=False, i=False, align=None, after=200):
    al = f' algn="{align}"' if align else ''
    return (f'<a:p><a:pPr{al} indent="0" marL="0"><a:spcAft><a:spcPts val="{after}"/></a:spcAft><a:buNone/></a:pPr>'
            f'{runs(text, sz, color, b, i)}<a:endParaRPr lang="en-US" sz="{sz}" dirty="0"/></a:p>')


def textbox(x, y, w, h, paras, name="text", anchor="t"):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="{name}-{_id[0]}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="{anchor}"/><a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')


def shape(x, y, w, h, prst, fill, name):
    av = '<a:gd name="adj" fmla="val 6000"/>' if prst == "roundRect" else ''
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="{name}-{_id[0]}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{xfrm(x, y, w, h)}<a:prstGeom prst="{prst}"><a:avLst>{av}</a:avLst></a:prstGeom>'
            f'<a:solidFill><a:schemeClr val="{fill}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr></p:sp>')


def badge(label, x, y, d=0.42):
    return shape(x, y, d, d, "ellipse", "accent1", "badge") + textbox(
        x, y, d, d, [para(label, 1400, "bg1", b=True, align="ctr", after=0)], "badge-label", "ctr")


def picture(rid, x, y, w, h, descr):
    return (f'<p:pic><p:nvPicPr><p:cNvPr id="{nid()}" name="figure-{_id[0]}" descr="{escape(descr)}"/>'
            f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr>{xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')


def steps(items, x, y, w, row, sz=1400):
    out = []
    for k, (head, body) in enumerate(items):
        yy = y + k * row
        out.append(badge(str(k + 1), x, yy))
        out.append(textbox(x + 0.6, yy - 0.03, w - 0.6, row - 0.1,
                           [para(head, sz, b=True), para(body, sz)], "step"))
    return "".join(out)


def caption(text, x, y, w, h=0.4):
    return textbox(x, y, w, h, [para(text, 1100, "accent5", i=True)], "caption")


TITLE_RE = re.compile(r'(<p:sp><p:nvSpPr><p:cNvPr id="2" name="Text 0">.*?</p:sp>)', re.S)


def fill_slide(slide, title, body, image, notes):
    p = U / "ppt/slides" / slide
    x = p.read_text()
    title_sp = TITLE_RE.search(x).group(1)
    title_sp = re.sub(r"<a:t>.*?</a:t>", f"<a:t>{escape(title)}</a:t>", title_sp, count=1, flags=re.S)
    slide_num = re.search(r'<p:sp>\s*<p:nvSpPr>\s*<p:cNvPr id="\d+" name="Slide Number Placeholder 0"/>.*?</p:sp>', x, re.S).group(0)
    head = x[: x.index("<p:sp>")]
    tail = x[x.index("</p:spTree>"):]
    p.write_text(head + title_sp + body + slide_num + tail)
    # point the copied image relationship at the new render
    media = f"snap3d-{slide.replace('.xml', '')}.png"
    shutil.copy(image, U / "ppt/media" / media)
    rels = U / "ppt/slides/_rels" / f"{slide}.rels"
    r = rels.read_text()
    r = re.sub(r'(Id="rId1" Type="[^"]*/image" Target=")[^"]*"', rf'\1../media/{media}"', r)
    rels.write_text(r)
    add_notes(slide, notes)


def add_notes(slide, text):
    n = int(re.search(r"\d+", slide).group())
    src = U / "ppt/notesSlides/notesSlide15.xml"
    dst = U / f"ppt/notesSlides/notesSlide{n}.xml"
    x = src.read_text()
    # the notes body is the longest run pptxgenjs wrote; swap its text
    bodies = re.findall(r"<a:t>(.*?)</a:t>", x, re.S)
    note_old = max(bodies, key=len)
    x = x.replace(f"<a:t>{note_old}</a:t>", f"<a:t>{escape(text)}</a:t>", 1)
    dst.write_text(x)
    rel_src = (U / "ppt/notesSlides/_rels/notesSlide15.xml.rels").read_text()
    rel = re.sub(r'Target="\.\./slides/slide\d+\.xml"', f'Target="../slides/{slide}"', rel_src)
    (U / f"ppt/notesSlides/_rels/notesSlide{n}.xml.rels").write_text(rel)
    srels = U / "ppt/slides/_rels" / f"{slide}.rels"
    s = srels.read_text()
    s = s.replace("</Relationships>", f'<Relationship Id="rId99" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide{n}.xml"/></Relationships>')
    srels.write_text(s)
    ct = U / "[Content_Types].xml"
    c = ct.read_text()
    part = f"/ppt/notesSlides/notesSlide{n}.xml"
    if part not in c:
        c = c.replace("</Types>", f'<Override PartName="{part}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/></Types>')
        ct.write_text(c)


def fit(image, h):
    im = Image.open(image)
    return h * im.width / im.height


# ---------- build ----------
div = add_slide("slide11.xml", "slide10.xml")
sA = add_slide("slide15.xml", div)
sB = add_slide("slide15.xml", sA)
sC = add_slide("slide15.xml", sB)
print("new slides:", div, sA, sB, sC)

# divider: reuse the section-divider slide, new text
p = U / "ppt/slides" / div
x = p.read_text()
x = x.replace("<a:t>Continuation strategies</a:t>", f"<a:t>{SECTION}</a:t>")
x = x.replace("<a:t>How each code walks the parameter space and guards against branch switches</a:t>",
              "<a:t>The MAC indicator as an isosurface over (d, φ₁, φ₂), extracted and rendered in ParaView</a:t>")
p.write_text(x)

IMG_Y, IMG_H = 1.45, 4.95

# A: method + oblique view
img = "renders/snap3d_oblique_crop.png"; w = fit(img, IMG_H); rx = 0.5 + w + 0.45
body = picture("rId1", 0.5, IMG_Y, w, IMG_H, "ParaView isosurface of the neighbour-MAC indicator over (phi1, phi2, d), coloured by d")
body += caption("CURRENTv4 Δ = 0.02 run. d axis stretched ×8; colour = d. The surface is a shell about one mesh cell thick around the jump.", 0.6, 6.5, w - 0.1, 0.45)
body += steps([
    ("Read θ from the VTU", "98,120 points and 559,751 tets, with θ stored at 51 beam nodes per point."),
    ("MAC on every tet edge", "675,318 edges; each node keeps the largest 1 − MAC among its edges."),
    ("Contour in ParaView", "Isosurface at 1 − MAC = 0.1, i.e. the deck's MAC = 0.90 threshold; then Smooth and Surface Normals."),
    ("Cross-check with energy", "88 % of edges with MAC < 0.90 also jump in U by more than 0.5."),
], rx, 1.6, 12.75 - rx, 1.25, 1400)
fill_slide(sA, "Extracting the snap surface: a MAC isosurface", body, img,
           "The indicator is computed from the theta field stored in solution_space_results.vtu of the CURRENTv4 0.02 run "
           "(make_mac_vtu.py); ParaView does the contouring, smoothing and rendering (render_snap.py, state saved as snap_surface.pvsm). "
           "Why MAC and not energy: a relative energy-jump indicator on the denser 0.01 run also picked up steep but smooth regions near d -> 1 "
           "and produced a false sheet at the top of the box. 1 - MAC has a clear gap between smooth edges (below 0.03) and jump edges (above 0.3).")

# B: top view
img = "renders/snap3d_top_crop.png"; w = fit(img, IMG_H); rx = 0.5 + w + 0.5
body = picture("rId1", 0.5, IMG_Y, w, IMG_H, "Snap surface seen down the d axis, coloured by d")
body += caption("Same surface, looking straight down the d axis (parallel projection).", 0.6, 6.5, w - 0.1, 0.4)
cards = [
    ("The 2D L-shape, stacked", "Every d-level of the sheet is the L-shaped locus from the 2D MAC map, coloured by d."),
    ("Two codes agree", "At d = 0.90, φ₂ = 0 the jump sits at φ₁ ≈ −0.55 in the v4 data and at −0.565 in the v3 run."),
    ("Closing in as d → 1", "Dark (d = 0.80) hugs the box edge; by d = 0.97 the jump at φ₂ = 0 has moved to φ₁ ≈ −0.31."),
]
cw = 12.75 - rx
for k, (h, t) in enumerate(cards):
    y = 1.6 + k * 1.65
    body += shape(rx, y, cw, 1.45, "roundRect", "bg2", "card")
    body += textbox(rx + 0.25, y + 0.18, cw - 0.5, 0.4, [para(h, 1600, "accent1", b=True)], "card-head")
    body += textbox(rx + 0.25, y + 0.6, cw - 0.5, 0.8, [para(t, 1400)], "card-body")
fill_slide(sB, "Seen down the d axis, the L-shaped locus closes in", body, img,
           "Looking down the d axis turns the sheet back into the family of L-shaped curves from the 2D slides. "
           "The v4 run (irregular tet mesh, distance-ordered traversal) and the v3 run (structured grid, adaptive MAC continuation) "
           "put the jump in the same place at d = 0.90, which is a useful independent check on both codes.")

# C: cut-away with fx
img = "renders/snap3d_slices_crop.png"; w = fit(img, IMG_H); rx = 0.5 + w + 0.5
body = picture("rId1", 0.5, IMG_Y, w, IMG_H, "Snap surface clipped at d = 0.90 above a slice coloured by the axial force fx")
body += caption("Plane at d = 0.90 coloured by f_x; the sheet above it is translucent.", 0.6, 6.5, w - 0.1, 0.4)
body += textbox(rx, 1.6, 12.75 - rx, 0.5, [para("Across the line, f_x roughly doubles at every d (φ₂ = 0):", 1500, "tx2")], "lead")
stats = [("28.9 → 58.0", "d = 0.86, jump at φ₁ ≈ −0.67"), ("27.3 → 57.0", "d = 0.92, jump at φ₁ ≈ −0.52"), ("26.6 → 53.7", "d = 0.97, jump at φ₁ ≈ −0.31")]
cw = 12.75 - rx
for k, (big, lab) in enumerate(stats):
    y = 2.25 + k * 1.2
    body += shape(rx, y, cw, 1.0, "roundRect", "bg2", "card")
    body += textbox(rx + 0.25, y + 0.12, 2.6, 0.75, [para(big, 2600, "accent1", b=True)], "stat", "ctr")
    body += textbox(rx + 2.95, y + 0.12, cw - 3.15, 0.75, [para(lab, 1300, "accent5")], "stat-label", "ctr")
body += textbox(rx, 5.75, cw, 0.7, [para("Built from θ alone, the surface still lands on the step the forces see, to within about one mesh cell.", 1400, "tx2")], "takeaway")
fill_slide(sC, "The sheet rises from the step in axial force", body, img,
           "Cut-away at d = 0.90: the plane is the volume data sliced at d = 0.90 and coloured by f_x on a fixed 0-70 scale; "
           "the MAC isosurface is clipped to d >= 0.90 and drawn translucent, so its base line and the f_x step can be compared directly. "
           "The values on the right are the largest f_x step along phi2 = 0 at three d levels.")

# ---------- section list ----------
pres = U / "ppt/presentation.xml"
x = pres.read_text()
rels = (U / "ppt/_rels/presentation.xml.rels").read_text()
ids = []
for s in (div, sA, sB, sC):
    rid = re.search(rf'Id="(rId\d+)"[^>]*Target="slides/{s}"', rels).group(1)
    ids.append(re.search(rf'<p:sldId id="(\d+)" r:id="{rid}"/>', x).group(1))
new_sec = f'<p14:section name="{SECTION}" id="{{{str(uuid.uuid4()).upper()}}}"><p14:sldIdLst>' + "".join(
    f'<p14:sldId id="{i}"/>' for i in ids) + "</p14:sldIdLst></p14:section>"
x = x.replace('<p14:section name="Continuation strategies"', new_sec + '<p14:section name="Continuation strategies"', 1)
pres.write_text(x)
print("section ids", ids)
