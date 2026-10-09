# Archival: this script edited an unpacked copy of presentation/elastica_snap_surface.pptx
# (./unpacked), in places via helpers from the Claude pptx skill (hard-coded paths).
# It is kept as a record of how the slides were built, not as a reusable tool.

"""Insert the "Towards a data-driven controller" section before the Takeaways slide.

Run from /home/goraksh/elastica_deck with the deck unpacked in ./unpacked:
    PYTHONPATH=<pptx skill>/scripts python controller_section.py
"""
import re
import subprocess
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

SKILL = Path("/home/goraksh/.claude/skills/synced/6b76838e-25d3-4069-b11d-75f398bcbcbd_39040c30-50d1-4d41-910a-741dc4d038ea/pptx/scripts")
U = Path("unpacked")
E = 914400
SECTION = "Towards a data-driven controller"
SUB_RE = re.compile(r"(?<![A-Za-z])([A-Za-zθφλ])_([A-Za-z0-9]{1,3})(?![A-Za-z_])")


def slide_with_title(title):
    for p in (U / "ppt/slides").glob("slide*.xml"):
        if f"<a:t>{title}</a:t>" in p.read_text():
            return p.name
    raise KeyError(title)


def add_slide(src, after):
    out = subprocess.run(["python", str(SKILL / "add_slide.py"), str(U), src, "--after", after],
                         check=True, capture_output=True, text=True).stdout
    return re.search(r"(slide\d+\.xml)", out.split("Created")[1]).group(1)


# ---------------- XML builders ----------------
_id = [200]


def nid():
    _id[0] += 1
    return _id[0]


def xfrm(x, y, w, h):
    return f'<a:xfrm><a:off x="{int(x*E)}" y="{int(y*E)}"/><a:ext cx="{int(w*E)}" cy="{int(h*E)}"/></a:xfrm>'


def runs(text, sz, color="tx1", b=False, i=False):
    pieces, last = [], 0
    for m in SUB_RE.finditer(text):
        pieces += [(text[last:m.start()] + m.group(1), False), (m.group(2), True)]
        last = m.end()
    pieces.append((text[last:], False))
    out = []
    for t, sub in pieces:
        if t:
            a = f'lang="en-US" sz="{sz}"' + (' b="1"' if b else '') + (' i="1"' if i else '') + (' baseline="-25000"' if sub else '')
            out.append(f'<a:r><a:rPr {a} dirty="0"><a:solidFill><a:schemeClr val="{color}"/></a:solidFill></a:rPr><a:t>{escape(t)}</a:t></a:r>')
    return "".join(out)


def para(text, sz, color="tx1", b=False, i=False, align=None, after=200):
    al = f' algn="{align}"' if align else ''
    return (f'<a:p><a:pPr{al} indent="0" marL="0"><a:spcAft><a:spcPts val="{after}"/></a:spcAft><a:buNone/></a:pPr>'
            f'{runs(text, sz, color, b, i)}<a:endParaRPr lang="en-US" sz="{sz}" dirty="0"/></a:p>')


def textbox(x, y, w, h, paras, name="text", anchor="t"):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="{name}-{_id[0]}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="{anchor}"/><a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')


def shape(x, y, w, h, prst, fill, name, line=None):
    av = '<a:gd name="adj" fmla="val 6000"/>' if prst == "roundRect" else ''
    fill_xml = f'<a:solidFill><a:schemeClr val="{fill}"/></a:solidFill>' if fill else '<a:noFill/>'
    ln = (f'<a:ln w="19050"><a:solidFill><a:schemeClr val="{line}"/></a:solidFill></a:ln>' if line else '<a:ln><a:noFill/></a:ln>')
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="{name}-{_id[0]}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{xfrm(x, y, w, h)}<a:prstGeom prst="{prst}"><a:avLst>{av}</a:avLst></a:prstGeom>{fill_xml}{ln}</p:spPr></p:sp>')


def card(x, y, w, h, fill="bg2", line=None):
    return shape(x, y, w, h, "roundRect", fill, "card", line)


def arrow(x, y, w, h=0.0, color="accent5"):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="flow-arrow-{_id[0]}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{xfrm(x, y, w, h)}<a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:noFill/>'
            f'<a:ln w="22225"><a:solidFill><a:schemeClr val="{color}"/></a:solidFill><a:prstDash val="solid"/><a:tailEnd type="triangle"/></a:ln></p:spPr></p:sp>')


def badge(label, x, y, d=0.45, fill="accent1", sz=1500):
    return shape(x, y, d, d, "ellipse", fill, "badge") + textbox(
        x, y, d, d, [para(label, sz, "bg1", b=True, align="ctr", after=0)], "badge-label", "ctr")


def box_text(x, y, w, h, head, body, sz=1300, head_color="accent1", fill="bg2", body_color="tx1"):
    out = card(x, y, w, h, fill)
    out += textbox(x + 0.22, y + 0.18, w - 0.44, 0.4, [para(head, sz + 300, head_color, b=True)], "card-head")
    out += textbox(x + 0.22, y + 0.62, w - 0.44, h - 0.75, [para(body, sz, body_color)], "card-body")
    return out


_BORDER = ''.join(f'<a:{s} w="9525" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:srgbClr val="D5DCE2"/></a:solidFill>'
                  f'<a:prstDash val="solid"/><a:round/><a:headEnd type="none" w="med" len="med"/><a:tailEnd type="none" w="med" len="med"/></a:{s}>'
                  for s in ("lnL", "lnR", "lnT", "lnB"))


def table(x, y, col_w, rows, row_h, sz=1200, first_col_color="accent1"):
    """rows[0] is the header. Cells are strings."""
    grid = "".join(f'<a:gridCol w="{int(w*E)}"/>' for w in col_w)
    trs = []
    for r, row in enumerate(rows):
        tcs = []
        for c, cell in enumerate(row):
            if r == 0:
                p, fill = para(cell, sz, "bg1", b=True, after=0), '<a:solidFill><a:schemeClr val="tx1"/></a:solidFill>'
            else:
                p = para(cell, sz, first_col_color if c == 0 else "tx1", b=(c == 0), after=0)
                fill = '<a:noFill/>'
            tcs.append(f'<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>{p}</a:txBody>'
                       f'<a:tcPr marL="91440" marR="91440" marT="45720" marB="45720" anchor="ctr">{_BORDER}{fill}</a:tcPr></a:tc>')
        trs.append(f'<a:tr h="{int(row_h[r]*E)}">{"".join(tcs)}</a:tr>')
    return (f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{nid()}" name="table-{_id[0]}"/>'
            f'<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>'
            f'<p:xfrm><a:off x="{int(x*E)}" y="{int(y*E)}"/><a:ext cx="{int(sum(col_w)*E)}" cy="{int(sum(row_h)*E)}"/></p:xfrm>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table"><a:tbl><a:tblPr/>'
            f'<a:tblGrid>{grid}</a:tblGrid>{"".join(trs)}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>')


TITLE_RE = re.compile(r'<p:sp><p:nvSpPr><p:cNvPr id="2" name="Text 0"\s*/?>.*?</p:sp>', re.S)
NUM_RE = re.compile(r'<p:sp>\s*<p:nvSpPr>\s*<p:cNvPr id="\d+" name="Slide Number Placeholder 0"\s*/?>.*?</p:sp>', re.S)


def fill_slide(slide, title, body, notes):
    p = U / "ppt/slides" / slide
    x = p.read_text()
    title_sp = re.sub(r"<a:t>.*?</a:t>", f"<a:t>{escape(title)}</a:t>", TITLE_RE.search(x).group(0), count=1, flags=re.S)
    head, tail = x[: x.index("<p:sp>")], x[x.index("</p:spTree>"):]
    p.write_text(head + title_sp + body + NUM_RE.search(x).group(0) + tail)
    add_notes(slide, notes)


def add_notes(slide, text):
    n = int(re.search(r"\d+", slide).group())
    src = sorted((U / "ppt/notesSlides").glob("notesSlide*.xml"))[0]
    x = src.read_text()
    old = max(re.findall(r"<a:t>(.*?)</a:t>", x, re.S), key=len)
    (U / f"ppt/notesSlides/notesSlide{n}.xml").write_text(x.replace(f"<a:t>{old}</a:t>", f"<a:t>{escape(text)}</a:t>", 1))
    rel = (U / f"ppt/notesSlides/_rels/{src.name}.rels").read_text()
    (U / f"ppt/notesSlides/_rels/notesSlide{n}.xml.rels").write_text(
        re.sub(r'Target="\.\./slides/slide\d+\.xml"', f'Target="../slides/{slide}"', rel))
    srels = U / "ppt/slides/_rels" / f"{slide}.rels"
    s = re.sub(r'<Relationship Id="rId\d+" Type="[^"]*/notesSlide"[^>]*/>', "", srels.read_text())
    srels.write_text(s.replace("</Relationships>", f'<Relationship Id="rId99" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide{n}.xml"/></Relationships>'))
    ct = U / "[Content_Types].xml"
    c = ct.read_text()
    part = f"/ppt/notesSlides/notesSlide{n}.xml"
    if part not in c:
        ct.write_text(c.replace("</Types>", f'<Override PartName="{part}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/></Types>'))


# ---------------- slides ----------------
divider_src = slide_with_title("Continuation strategies")
content_src = slide_with_title("Open issues worth checking")
takeaways = slide_with_title("Takeaways")

new = []
after = content_src
div = add_slide(divider_src, after); new.append(div); after = div
for _ in range(7):
    s = add_slide(content_src, after); new.append(s); after = s
print("new slides:", new)
for s in new:   # the source slide is hidden in the deck; the new slides must not be
    q = U / "ppt/slides" / s
    q.write_text(q.read_text().replace(' show="0"', "", 1))

# divider text
p = U / "ppt/slides" / div
x = p.read_text().replace("<a:t>Continuation strategies</a:t>", f"<a:t>{SECTION}</a:t>")
x = x.replace("<a:t>How each code walks the parameter space and guards against branch switches</a:t>",
              "<a:t>How to run the production simulation that trains it, and the options for doing so</a:t>")
p.write_text(x)

# 1. What the controller needs
b = ""
flow = [("Actuation", "d, φ₁, φ₂"), ("+ branch / history", "which equilibrium the rod is on"),
        ("Neural network", "surrogate, inverse model or policy"), ("Predicted state", "θ(s), f_x, f_y, M₀, snap risk")]
for k, (h, t) in enumerate(flow):
    x0 = 0.6 + k * 3.12
    nn = k == 2
    b += card(x0, 1.55, 2.75, 1.15, "accent1" if nn else "bg2")
    b += textbox(x0 + 0.15, 1.62, 2.45, 1.0, [para(h, 1600, "bg1" if nn else "tx1", b=True, align="ctr", after=300),
                                              para(t, 1200, "bg1" if nn else "accent5", align="ctr", after=0)], "flow", "ctr")
    if k < 3:
        b += arrow(x0 + 2.78, 2.125, 0.31)
cards = [("Forward surrogate (MPC)", "Learns (d, φ₁, φ₂, branch) → shape and forces. An optimiser searches actuations online, with the network standing in for the FE solve."),
         ("Inverse model", "Learns target shape or force → actuation. Ill-posed where two branches overlap, so the branch has to be an input."),
         ("Policy (RL or imitation)", "Learns state → next actuation inside a surrogate environment. Needs transitions, including the snaps.")]
for k, (h, t) in enumerate(cards):
    b += box_text(0.6 + k * 4.14, 3.0, 3.83, 2.35, h, t, 1350)
b += card(0.6, 5.6, 12.1, 1.2, "bg1", line="accent3")
b += badge("!", 0.85, 5.97, 0.45, "accent3")
b += textbox(1.55, 5.7, 10.95, 1.0, [para("Inside the hysteresis zone one actuation has two equilibria. The current studies keep whichever branch the "
                                          "continuation reached first, so a dataset built the same way would teach the network a path-dependent artefact.", 1450)],
             "callout", "ctr")
fill_slide(new[1], "A controller needs branch-aware data", b,
           "Whatever the network is, the data has to say which equilibrium each sample sits on. Across the fold the forward map "
           "from actuation to state is multi-valued: f_x roughly doubles across the snap surface, so averaging the two branches is "
           "meaningless. The choice between surrogate, inverse model and policy is the main decision that shapes the campaign.")

# 2. Options overview
rows = [["Option", "How it runs", "What it adds for the controller", "Cost and new code"],
        ["A  Scaled-up grid", "The v3 driver at production resolution over the full box", "One equilibrium per point; the snap surface located", "1× (7–28 h on one core); no new code"],
        ["B  Branch-aware sweeps", "Sweep compression and release from several seeds; keep every distinct equilibrium; add a stability eigenvalue", "Both branches in the hysteresis zone, labelled stable or unstable", "2–4×; multi-seed driver and a small eigen-solve"],
        ["C  Fold tracking", "Pseudo-arclength continuation through the folds; solve for the fold surface directly", "The exact snap boundary and the unstable branch between", "≈ A per point; substantial new code"],
        ["D  Active learning", "Train an ensemble on a coarse set; solve next where the ensemble disagrees", "Samples where the network needs them, mostly near the snap", "Fewest solves; needs the ML loop in place"],
        ["E  Actuation paths", "March random quasi-static actuation paths with MAC-guided continuation; log every snap", "Sequences with history: (state, action) → next state", "Scales with number × length of paths; one path per core"]]
b = table(0.6, 1.5, [2.45, 3.85, 3.35, 2.45], rows, [0.5] + [0.9] * 5, 1250)
fill_slide(new[2], "Five ways to run the production campaign", b,
           "A is what the existing code already does, scaled up. B and E change what is recorded, C changes the numerics, "
           "D changes where the solves go. They combine: the recommended plan later in this section uses B, E and D.")

# 3. Options A and B
def detail(x0, w, letter, name, items, sz=1300, row=1.1):
    out = card(x0, 1.5, w, 5.3)
    out += badge(letter, x0 + 0.25, 1.68, 0.5, "accent1", 1700)
    out += textbox(x0 + 0.9, 1.72, w - 1.1, 0.45, [para(name, 1800, b=True)], "detail-head", "ctr")
    for k, (lab, t) in enumerate(items):
        y = 2.4 + k * row
        out += textbox(x0 + 0.25, y, 1.25, 0.4, [para(lab, sz, "accent1", b=True)], "detail-label")
        out += textbox(x0 + 1.5, y, w - 1.75, row - 0.1, [para(t, sz)], "detail-text")
    return out

b = detail(0.6, 5.9, "A", "Scaled-up grid sweep", [
    ("How", "The v3 driver on its config grid: d 0.70–0.9999 (Δ 0.005), φ ±π/2 (Δ π/144). About 1.26 M points, half solved via symmetry, plus gradient refinement."),
    ("Gives", "One equilibrium with all outputs per point; the snap surface falls out of the MAC jumps."),
    ("Cost", "7–28 h on one core at the measured 0.02–0.08 s per point; about 0.6 GB of float32 data."),
    ("Watch out", "In the hysteresis zone the stored branch depends on the sweep path.")])
b += detail(6.8, 5.9, "B", "Branch-aware sweeps", [
    ("How", "Repeat the sweep with d rising and falling, from 2–4 seeds across (φ₁, φ₂). Merge the results: MAC > 0.99 means the same equilibrium."),
    ("Gives", "Every equilibrium continuation reaches, a branch id, and λ_min of the constrained Hessian (stable or not)."),
    ("Cost", "2–4× option A. The ~100 × 100 dense eigen-solve per point is small next to the Newton solve."),
    ("Watch out", "Keeping branch labels consistent across the box, and choosing the merge threshold.")])
fill_slide(new[3], "Options A and B: grid sweeps", b,
           "Option A cost: 60 d-layers × 145 × 145 phi values ≈ 1.26 M stored points. Measured throughput from file timestamps: "
           "v3 sol4 stored 155,526 points in 44 min (0.017 s/point); the v4 runs stored about 13 points per second (0.076 s/point). "
           "Stability in B: the smallest eigenvalue of the Hessian of the Lagrangian restricted to the tangent space of the two constraints; "
           "negative means the equilibrium is unstable.")

# 4. Options C, D, E
b = ""
for k, (letter, name, items) in enumerate([
    ("C", "Fold tracking", [("How", "Pseudo-arclength continuation that passes through folds; a bordered system pins the fold where J is singular."),
                            ("Gives", "The fold surface itself, plus the unstable branch linking the two stable ones."),
                            ("Cost", "Similar per point, but significant development."),
                            ("Watch out", "Two-parameter fold continuation is research-grade code.")]),
    ("D", "Active learning", [("How", "Train an ensemble on a coarse option-A grid; solve next where the ensemble disagrees most."),
                              ("Gives", "Samples concentrated near the snap surface, where the network errs."),
                              ("Cost", "Fewest solves for a target accuracy; training runs between batches."),
                              ("Watch out", "Needs the network design fixed first; keep an untouched test set.")]),
    ("E", "Actuation paths", [("How", "Draw smooth random paths in (d, φ₁, φ₂) and march along each with MAC-guided continuation."),
                              ("Gives", "(state, action) → next state, snaps included: direct data for a policy or MPC."),
                              ("Cost", "Proportional to number × length of paths; one path per core."),
                              ("Watch out", "Coverage needs many paths; quasi-static only, no snap dynamics.")])]):
    x0 = 0.6 + k * 4.14
    w = 3.83
    b += card(x0, 1.5, w, 5.3)
    b += badge(letter, x0 + 0.22, 1.66, 0.5, "accent1", 1700)
    b += textbox(x0 + 0.85, 1.7, w - 1.0, 0.45, [para(name, 1700, b=True)], "detail-head", "ctr")
    for j, (lab, t) in enumerate(items):
        y = 2.35 + j * 1.1
        b += textbox(x0 + 0.22, y, w - 0.44, 1.05, [para(lab, 1250, "accent1", b=True, after=100), para(t, 1250)], "detail-text")
fill_slide(new[4], "Options C–E: change the numerics, sampling or data", b,
           "C is the most rigorous view of the snap but the most code. D needs a first network to exist, so it is a second stage. "
           "E is the closest match to how a controller is actually used: it sees the rod's history and the snaps along realistic actuation paths.")

# 5. Scale and running it
b = ""
tiles = [("≈ 1.26 M", "grid points", "v3 config: 60 × 145 × 145"),
         ("0.02–0.08 s", "per point, one core", "from v3 sol4 and v4 run timestamps"),
         ("7–28 h", "serial, option A", "before any parallelism"),
         ("≈ 0.6 GB", "per 1.26 M points", "θ in float32 plus scalar outputs")]
for k, (big, lab, sub) in enumerate(tiles):
    x0 = 0.6 + k * 3.1
    b += card(x0, 1.5, 2.8, 1.95)
    b += textbox(x0 + 0.22, 1.62, 2.4, 0.75, [para(big, 3000, "accent1", b=True, after=0)], "tile-big", "ctr")
    b += textbox(x0 + 0.22, 2.42, 2.4, 0.35, [para(lab, 1400, b=True, after=0)], "tile-label")
    b += textbox(x0 + 0.22, 2.8, 2.4, 0.55, [para(sub, 1150, "accent5", after=0)], "tile-sub")
for k, (h, t) in enumerate([
        ("Seed, then fan out", "A coarse serial pass gives one converged solution per block of d-layers or per path; each block then continues on its own."),
        ("One rank per block", "Each MPI rank builds its own 100-element beam problem on COMM_SELF. The problem is tiny, so throughput scales with core count."),
        ("Restart and audit", "Checkpoint per block (v3 and v4 already do), keep solver metadata, and re-solve a random 1 % at 200 elements as a convergence audit.")]):
    b += box_text(0.6 + k * 4.14, 3.75, 3.83, 2.0, h, t, 1350)
fill_slide(new[5], "Scale: a day on one core, hours on a workstation", b,
           "Throughput: v3 sol4 stored 155,526 points in 44 min; v4 sol1 stored 740,741 in 15.6 h and v4 sol2 98,120 in 2.1 h. "
           "These are wall-clock times per stored point including I/O and refinement, so treat them as a range. "
           "Option B multiplies the work by 2–4 and E adds whatever path budget is chosen; both parallelise the same way.")

# 6. Dataset schema
rows = [["Field", "Size", "Why the controller needs it"],
        ["d, φ₁, φ₂", "3", "Actuation: the network's inputs"],
        ["branch_id", "1 (int)", "Separates the two equilibria inside the hysteresis zone"],
        ["θ(s)", "101", "Shape; compress with POD before training"],
        ["f_x, f_y, M₀, M_L", "4", "Forces and moments at both clamps"],
        ["U", "1", "Energy; also a check on the branch label"],
        ["λ_min", "1", "Stability: negative means an unstable equilibrium"],
        ["MAC to source, refinement count", "2", "Flags samples right next to a snap"],
        ["path_id, step", "2", "Sequence context for option E"],
        ["Newton iterations, residual", "2", "Data quality; filter suspicious solves"]]
b = table(0.6, 1.5, [3.6, 1.6, 6.9], rows, [0.5] + [0.5] * 9, 1300)
fill_slide(new[6], "One record per equilibrium: what to store", b,
           "Storing one record per equilibrium (not per parameter point) is what makes the dataset branch-aware: a point inside the "
           "hysteresis zone gets two records with different branch_id. M_L is the clamp moment at s = L, which v4 already computes as right_moment.")

# 7. Recommended plan
b = ""
phases = [("0", "Fix and verify", "Mirrored-moment sign; mesh convergence near d → 1; add the stability eigenvalue."),
          ("1", "Branch-aware grid (B)", "Production run on the v3 driver: the main training set."),
          ("2", "Actuation paths (E)", "Sequence data for a policy or MPC, plus an independent test set."),
          ("3", "Active learning (D)", "Once a first network exists, top up where it is least certain.")]
for k, (num, h, t) in enumerate(phases):
    x0 = 0.6 + k * 3.12
    b += card(x0, 1.55, 2.75, 2.75)
    b += badge(num, x0 + 0.22, 1.72, 0.5, "accent1", 1700)
    b += textbox(x0 + 0.22, 2.35, 2.35, 0.5, [para(h, 1600, b=True)], "phase-head")
    b += textbox(x0 + 0.22, 2.9, 2.35, 1.3, [para(t, 1350)], "phase-body")
    if k < 3:
        b += arrow(x0 + 2.78, 2.925, 0.31)
b += box_text(0.6, 4.6, 7.0, 1.5, "Decision needed: which controller?",
              "A forward surrogate for MPC leans the plan towards B. A learned policy leans it towards E. "
              "The choice sets how the budget is split between phases 1 and 2.", 1400)
b += box_text(7.9, 4.6, 4.8, 1.5, "Where C fits",
              "Only if the controller must work right at the fold boundary, where the exact snap surface matters.", 1400)
fill_slide(new[7], "Recommended plan: fix, sweep both branches, then add paths", b,
           "Phase 0 removes known errors before spending compute: the v3 mirrored-moment sign and the large f_x values near d → 1 "
           "in the v4 data. Phases 1 and 2 can run in parallel once the driver supports multiple seeds.")

# ---------------- sections ----------------
pres = U / "ppt/presentation.xml"
x = pres.read_text()
rels = (U / "ppt/_rels/presentation.xml.rels").read_text()


def sld_id(slide):
    rid = re.search(rf'Id="(rId\d+)"[^>]*Target="slides/{slide}"', rels).group(1)
    return re.search(rf'<p:sldId id="(\d+)" r:id="{rid}"/>', x).group(1)


ids = [sld_id(s) for s in new] + [sld_id(takeaways)]
x = x.replace(f'<p14:sldId id="{ids[-1]}"/>', "", 1)            # takeaways moves to the new last section
sec = f'<p14:section name="{SECTION}" id="{{{str(uuid.uuid4()).upper()}}}"><p14:sldIdLst>' + "".join(
    f'<p14:sldId id="{i}"/>' for i in ids) + "</p14:sldIdLst></p14:section>"
x = x.replace("</p14:sectionLst>", sec + "</p14:sectionLst>", 1)
pres.write_text(x)

# takeaways: point the "next" line at the production campaign
p = U / "ppt/slides" / takeaways
t = p.read_text()
t = re.sub(r"<a:t>Next: [^<]*</a:t>",
           "<a:t>Next: fix the mirrored-moment sign · mesh-convergence check near d → 1 · branch-aware production run for the controller dataset</a:t>", t)
p.write_text(t)
print("section ids", ids)
