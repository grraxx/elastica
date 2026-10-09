# Archival: this script edited an unpacked copy of presentation/elastica_snap_surface.pptx
# (./unpacked), in places via helpers from the Claude pptx skill (hard-coded paths).
# It is kept as a record of how the slides were built, not as a reusable tool.

"""Replace every equation image in the unpacked deck with a text box holding its LaTeX source.

Run from /home/goraksh/elastica_deck with the deck unpacked in ./unpacked.
"""
import re
from pathlib import Path
from xml.sax.saxutils import escape

U = Path("unpacked")
EMU = 914400

LATEX = {
    "energy": r"\Pi[\theta]=\frac{1}{2}\int_0^L \theta'(s)^2\,ds",
    "constraints": r"\int_0^L \cos\theta\,ds=d,\quad \int_0^L \sin\theta\,ds=0,\quad \theta(0)=\varphi_1,\quad \theta(L)=\varphi_2",
    "lagrangian": r"\mathcal{L}[\theta,f_x,f_y]=\frac{1}{2}\int_0^L \theta'^2\,ds+f_x\left(\int_0^L \cos\theta\,ds-d\right)+f_y\int_0^L \sin\theta\,ds",
    "strong": r"\theta''+f_x\sin\theta-f_y\cos\theta=0",
    "w0": r"F_0=\int_0^L \left(\theta' v'-f_x\sin\theta\,v+f_y\cos\theta\,v\right)ds=0\quad \forall v\in V_0",
    "w1": r"F_1=\int_0^L \left(\cos\theta-d\right)q_x\,ds=0",
    "w2": r"F_2=\int_0^L \sin\theta\,q_y\,ds=0",
    "outputs": r"M_0=\theta'(0)\approx\frac{\theta_1-\theta_0}{h},\qquad U=\frac{1}{2}\int_0^L \theta'^2\,ds",
    "newton": r"J(\mathbf{u}^k)\,\delta\mathbf{u}=-\mathbf{F}(\mathbf{u}^k),\qquad \mathbf{u}^{k+1}=\mathbf{u}^k+\alpha_k\,\delta\mathbf{u}",
    "mac": r"\mathrm{MAC}(\theta_a,\theta_b)=\frac{\left(\theta_a^T\theta_b\right)^2}{\left(\theta_a^T\theta_a\right)\left(\theta_b^T\theta_b\right)}\in[0,1]",
    "cold": r"\theta_0(s)=\varphi_1+(\varphi_2-\varphi_1)\frac{s}{L}+A\sin\frac{\pi s}{L},\qquad A=2\sqrt{\max\left(0,\,1-d/L\right)}",
    "amp": r"\frac{d}{L}\approx\frac{1}{L}\int_0^L \left(1-\frac{\theta^2}{2}\right)ds\;\Rightarrow\;\frac{d}{L}\approx 1-\frac{A^2}{4}",
    "substep": r"\mathbf{p}^{(m)}=\mathbf{p}_0+\frac{m}{2^r}\,\Delta p_i\,\mathbf{e}_i,\quad m=1,\dots,2^r,\quad r=0,1,\dots,5",
    "grad": r"G=\Delta\Phi^{-1}\,\Delta\Theta,\quad \Delta\Phi=\left(\Delta\varphi_a,\Delta\varphi_b\right)^T\in\mathbb{R}^{2\times 2},\quad \mathrm{score}_k=\sqrt{\mathrm{mean}_s\,G_k(s)^2}",
    "sym": r"(d,\varphi_1,\varphi_2)\mapsto(d,-\varphi_2,-\varphi_1):\quad \tilde{\theta}(s)=-\theta(L-s),\quad \tilde{f}_x=f_x,\quad \tilde{f}_y=-f_y,\quad \tilde{M}_0=\theta'(L)",
}

PIC_RE = re.compile(r"<p:pic>(?:(?!</p:pic>).)*?name=\"equation-\d+\" descr=\"(\w+)\".*?</p:pic>", re.S)
XFRM_RE = re.compile(r'<a:off x="(\d+)" y="(\d+)"/>\s*<a:ext cx="(\d+)" cy="(\d+)"/>')
CARD_RE = re.compile(r'<p:sp>(?:(?!</p:sp>).)*?<a:xfrm><a:off x="(\d+)" y="(\d+)"/><a:ext cx="(\d+)" cy="(\d+)"/></a:xfrm><a:prstGeom prst="roundRect">', re.S)
SLIDE_W = int(13.333 * EMU)


def textbox(sid, key, x, y, w, h):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="latex-{key}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="ctr"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
            f'<a:p><a:pPr indent="0" marL="0"><a:buNone/></a:pPr><a:r><a:rPr lang="en-US" sz="1200" dirty="0">'
            f'<a:solidFill><a:schemeClr val="tx1"/></a:solidFill><a:latin typeface="Courier New"/><a:cs typeface="Courier New"/></a:rPr>'
            f'<a:t>{escape(LATEX[key])}</a:t></a:r><a:endParaRPr lang="en-US" sz="1200" dirty="0"/></a:p></p:txBody></p:sp>')


done = []
for slide in sorted((U / "ppt/slides").glob("slide*.xml")):
    x = slide.read_text()
    if "equation-" not in x:
        continue
    cards = [tuple(map(int, m.groups())) for m in CARD_RE.finditer(x)]
    rels_path = U / "ppt/slides/_rels" / f"{slide.name}.rels"
    rels = rels_path.read_text()
    for m in list(PIC_RE.finditer(x)):
        pic, key = m.group(0), m.group(1)
        px, py, pw, ph = map(int, XFRM_RE.search(pic).groups())
        sid = re.search(r'<p:cNvPr id="(\d+)"', pic).group(1)
        rid = re.search(r'r:embed="(rId\d+)"', pic).group(1)
        # widen to the enclosing card (or to the right margin) so long LaTeX fits
        right = SLIDE_W - int(0.6 * EMU)
        for cx, cy, cw, ch in cards:
            if cx <= px <= cx + cw and cy <= py <= cy + ch:
                right = cx + cw - int(0.2 * EMU)
        w = max(pw, right - px)
        h = max(ph, int(0.45 * EMU))
        x = x.replace(pic, textbox(sid, key, px, py, w, h), 1)
        rels = re.sub(rf'<Relationship Id="{rid}"[^>]*/>', "", rels)
        done.append((slide.name, key, round(w / EMU, 2), len(LATEX[key])))
    slide.write_text(x)
    rels_path.write_text(rels)

for d in done:
    print(*d)
print(len(done), "equations replaced")
