"""The surrogate section's equations: LaTeX -> PowerPoint OMML (via pandoc's docx writer)
and a fallback PNG (matplotlib mathtext), as the deck's other equations are.

usage: python make_surrogate_equations.py omml   (any env with pandoc reachable at PANDOC)
       python make_surrogate_equations.py png    (elastica-ml: needs matplotlib)
Writes presentation/figures/equations/omml.json (one OMML paragraph per key, in the deck's run
style), <key>.png (the fallback, 300 dpi) and png.json (their sizes in inches).
"""
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures" / "equations"
OUT.mkdir(parents=True, exist_ok=True)
PANDOC = "/home/goraksh/miniforge3/envs/elastica-env/bin/pandoc"
SIZE = 14                                           # pt, every equation

EQUATIONS = {
    # why ln U
    "log_rel": r"\ln(U + \delta U) - \ln U \approx \dfrac{\delta U}{U}",
    "log_weight": r"(\delta U)^2 = (0.01\,U)^2 \qquad \mathrm{against} \qquad (\delta \ln U)^2 = 0.01^2",
    "log_pos": r"U = e^{z} > 0 \quad \mathrm{for\ every\ } z",
    # why asinh M
    "asinh_def": r"\operatorname{asinh} M = \ln\left(M + \sqrt{M^2 + 1}\right)",
    "asinh_lim": r"\operatorname{asinh} M \approx M \quad (|M| \ll 1), \qquad \operatorname{asinh} M \approx \pm \ln 2|M| \quad (|M| \gg 1)",
    "asinh_err": r"\delta M \approx \sqrt{1 + M^2}\;\delta(\operatorname{asinh} M), \qquad \operatorname{asinh}(-M) = -\operatorname{asinh} M",
    # the loss
    "loss_u": r"\mathcal{L}_U = \left\langle (\ln U - \ln U^{*})^2 \right\rangle",
    "loss_m": r"\mathcal{L}_M = \left\langle \sum_{i=1,2} (\operatorname{asinh} M_i - \operatorname{asinh} M_i^{*})^2 \right\rangle",
    "loss_slope": r"\mathcal{L}_{\partial} = \left\langle \left(\operatorname{asinh}\dfrac{-\partial U}{\partial \varphi_1} - \operatorname{asinh} M_1^{*}\right)^2 + \left(\operatorname{asinh}\dfrac{\partial U}{\partial \varphi_2} - \operatorname{asinh} M_2^{*}\right)^2 \right\rangle",
    "loss_f": r"\mathcal{L}_f = \left\langle \left(\operatorname{asinh}\dfrac{-\partial U/\partial d}{f_0} - \operatorname{asinh}\dfrac{f_x^{*}}{f_0}\right)^2 \right\rangle,\quad f_0 = 10",
    "loss_e": r"\mathcal{L}_{\exists} = -\left\langle e \ln P + (1 - e) \ln(1 - P) \right\rangle",
    "loss": r"\mathcal{L} = \mathcal{L}_U + \mathcal{L}_M + \mathcal{L}_{\partial} + \mathcal{L}_f + \mathcal{L}_{\exists}",
}

RPR = ('<a:rPr lang="en-US" sz="{sz}" i="{i}" dirty="0"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill>'
       '<a:latin typeface="Cambria Math" panose="02040503050406030204" pitchFamily="18" charset="0"/></a:rPr>')


def to_pptx(omml):
    """pandoc's docx OMML -> the deck's: drawingml run properties instead of Word's."""
    omml = re.sub(r"<w:rPr>.*?</w:rPr>|<w:rPr/>", "", omml, flags=re.S)

    def run(m):
        body = m.group(1)
        italic = "0" if 'm:val="p"' in body else "1"
        return "<m:r>" + body.replace("<m:t", RPR.format(sz=SIZE * 100, i=italic) + "<m:t", 1) + "</m:r>"

    omml = re.sub(r"<m:r>(.*?)</m:r>", run, omml, flags=re.S)
    omml = re.sub(r"<m:ctrlPr>.*?</m:ctrlPr>|<m:ctrlPr/>",
                  "<m:ctrlPr>" + RPR.format(sz=SIZE * 100, i="1") + "</m:ctrlPr>", omml, flags=re.S)
    assert "w:" not in omml, omml[:300]
    omml = re.sub(r"<m:oMathParaPr>.*?</m:oMathParaPr>", "", omml, flags=re.S)
    omml = omml.replace("<m:oMathPara>", '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/'
                        'officeDocument/2006/math"><m:oMathParaPr><m:jc m:val="left"/></m:oMathParaPr>', 1)
    return omml


def omml():
    md = OUT / "eq.md"
    md.write_text("\n\n".join(f"$${tex}$$" for tex in EQUATIONS.values()) + "\n")
    subprocess.run([PANDOC, str(md), "-o", str(OUT / "eq.docx")], check=True)
    doc = zipfile.ZipFile(OUT / "eq.docx").read("word/document.xml").decode()
    paras = re.findall(r"<m:oMathPara>.*?</m:oMathPara>", doc, flags=re.S)
    assert len(paras) == len(EQUATIONS), (len(paras), len(EQUATIONS))
    (OUT / "omml.json").write_text(json.dumps({k: to_pptx(p) for k, p in zip(EQUATIONS, paras)},
                                              ensure_ascii=False))
    print(f"{len(paras)} equations -> {OUT / 'omml.json'}")


def png():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["mathtext.fontset"] = "stix"
    dpi, sizes = 300, {}
    for key, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontsize=SIZE, color="#16222E")
        path = OUT / f"{key}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
        from PIL import Image
        w, h = Image.open(path).size
        sizes[key] = {"w_in": w / dpi, "h_in": h / dpi}
    (OUT / "png.json").write_text(json.dumps(sizes, indent=1))
    print(f"{len(sizes)} PNGs -> {OUT}")


if __name__ == "__main__":
    {"omml": omml, "png": png}[sys.argv[1]]()
