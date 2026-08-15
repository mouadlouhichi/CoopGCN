#!/usr/bin/env python3
"""
Render coopgcn_array.tex to a preview PDF without a TeX installation.

This is a *preview* renderer for environments with no LaTeX toolchain. The
authoritative artefact is coopgcn_array.tex, compiled with:
    pdflatex coopgcn_array && bibtex coopgcn_array && pdflatex coopgcn_array x2

The renderer parses the .tex directly (rather than duplicating its content)
so the preview cannot drift from the source. Display equations are rendered
with matplotlib mathtext; tables and figures are laid out with ReportLab.
"""
import os
import re
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                FrameBreak, ListFlowable, ListItem, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "coopgcn_array.tex")
BIB = os.path.join(HERE, "references.bib")
OUT = os.path.join(HERE, "CoopGCN_Array_preview.pdf")
EQD = os.path.join(HERE, ".eqcache")
os.makedirs(EQD, exist_ok=True)

PAGE_W, PAGE_H = A4
MARGIN = 1.9 * cm
BODY_W = PAGE_W - 2 * MARGIN
GUTTER = 0.7 * cm
COL_W_G = (BODY_W - GUTTER) / 2.0
corn = 0
algn = 0

# ----------------------------------------------------------------- styles ---
ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=ss["Title"], fontName="Times-Bold",
                            fontSize=17, leading=21, spaceAfter=10),
    "author": ParagraphStyle("author", parent=ss["Normal"], fontName="Times-Roman",
                             fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=2),
    "affil": ParagraphStyle("affil", parent=ss["Normal"], fontName="Times-Italic",
                            fontSize=9, leading=12, alignment=TA_CENTER, spaceAfter=2),
    "absthead": ParagraphStyle("absthead", parent=ss["Normal"], fontName="Times-Bold",
                               fontSize=10.5, leading=13, spaceBefore=10, spaceAfter=4),
    "abstract": ParagraphStyle("abstract", parent=ss["Normal"], fontName="Times-Roman",
                               fontSize=9.5, leading=12.4, alignment=TA_JUSTIFY),
    "kw": ParagraphStyle("kw", parent=ss["Normal"], fontName="Times-Roman",
                         fontSize=9.5, leading=12.4, alignment=TA_JUSTIFY, spaceBefore=6),
    "h1": ParagraphStyle("h1", parent=ss["Normal"], fontName="Times-Bold",
                         fontSize=10.8, leading=13, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=ss["Normal"], fontName="Times-Bold",
                         fontSize=9.8, leading=12, spaceBefore=10, spaceAfter=4),
    "h1s": ParagraphStyle("h1s", parent=ss["Normal"], fontName="Times-Bold",
                          fontSize=10.2, leading=12.5, spaceBefore=13, spaceAfter=5),
    "body": ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman",
                           fontSize=9.2, leading=11.6, alignment=TA_JUSTIFY,
                           spaceAfter=6),
    "cap": ParagraphStyle("cap", parent=ss["Normal"], fontName="Times-Roman",
                          fontSize=8.2, leading=10.2, alignment=TA_JUSTIFY,
                          spaceBefore=4, spaceAfter=8),
    "prop": ParagraphStyle("prop", parent=ss["Normal"], fontName="Times-Roman",
                           fontSize=9.2, leading=11.6, alignment=TA_JUSTIFY,
                           leftIndent=10, rightIndent=6, spaceAfter=6,
                           borderPadding=0),
    "ref": ParagraphStyle("ref", parent=ss["Normal"], fontName="Times-Roman",
                          fontSize=7.9, leading=9.9, alignment=TA_JUSTIFY,
                          leftIndent=12, firstLineIndent=-16, spaceAfter=3),
    "tbl": ParagraphStyle("tbl", parent=ss["Normal"], fontName="Times-Roman",
                          fontSize=7.4, leading=9.2),
    "tblb": ParagraphStyle("tblb", parent=ss["Normal"], fontName="Times-Bold",
                           fontSize=7.4, leading=9.2),
    "tblh": ParagraphStyle("tblh", parent=ss["Normal"], fontName="Times-Bold",
                           fontSize=7.4, leading=9.2, alignment=TA_CENTER),
    "tblw": ParagraphStyle("tblw", parent=ss["Normal"], fontName="Times-Roman",
                           fontSize=5.9, leading=7.2),
    "tblwb": ParagraphStyle("tblwb", parent=ss["Normal"], fontName="Times-Bold",
                            fontSize=5.9, leading=7.2),
    "tblwh": ParagraphStyle("tblwh", parent=ss["Normal"], fontName="Times-Bold",
                            fontSize=5.9, leading=7.2, alignment=TA_CENTER),
    "alg": ParagraphStyle("alg", parent=ss["Normal"], fontName="Times-Roman",
                          fontSize=7.8, leading=9.8),
    "algn": ParagraphStyle("algn", parent=ss["Normal"], fontName="Times-Roman",
                           fontSize=7.0, leading=9.8, alignment=TA_CENTER),
    "eqn": ParagraphStyle("eqn", parent=ss["Normal"], fontName="Times-Roman",
                          fontSize=8.5, leading=10),
    "hl": ParagraphStyle("hl", parent=ss["Normal"], fontName="Times-Roman",
                         fontSize=9.0, leading=11.4, alignment=TA_JUSTIFY,
                         leftIndent=12, bulletIndent=2, spaceAfter=2),
}

# ------------------------------------------------------------- tex helpers ---
GREEK = {
    r"\gets": "\u2190", r"\emptyset": "\u2205", r"\varnothing": "\u2205",
    r"\equiv": "\u2261",
    r"\alpha": "\u03b1", r"\beta": "\u03b2", r"\gamma": "\u03b3",
    r"\delta": "\u03b4", r"\epsilon": "\u03b5", r"\eta": "\u03b7",
    r"\kappa": "\u03ba", r"\lambda": "\u03bb", r"\mu": "\u03bc",
    r"\tau": "\u03c4", r"\sigma": "\u03c3", r"\phi": "\u03c6",
    r"\Theta": "\u0398", r"\Gamma": "\u0393", r"\Delta": "\u0394",
    r"\times": "\u00d7", r"\le": "\u2264", r"\ge": "\u2265",
    r"\to": "\u2192", r"\infty": "\u221e", r"\ast": "*",
    r"\cdot": "\u00b7", r"\dagger": "\u2020", r"\rightarrow": "\u2192",
    r"\downarrow": "\u2193", r"\subseteq": "\u2286", r"\in": "\u2208",
    r"\setminus": "\\", r"\forall": "\u2200", r"\Rightarrow": "\u21d2",
    r"\sum": "\u03a3", r"\sim": "~", r"\approx": "\u2248", r"\pm": "\u00b1",
    r"\cup": "\u222a", r"\cap": "\u2229",
}

CITEKEYS = []


def load_bib():
    txt = open(BIB).read()
    entries = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", txt, re.S):
        key, body = m.group(1).strip(), m.group(2)
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.+?)[}\"],?\s*\n", body, re.S):
            f[fm.group(1).lower()] = re.sub(r"\s+", " ", fm.group(2)).strip()
        # nested braces in title
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n\s*(?=\w+\s*=|$)", body, re.S):
            k = fm.group(1).lower()
            if k not in f or len(fm.group(2)) > len(f.get(k, "")):
                f[k] = re.sub(r"\s+", " ", fm.group(2)).strip()
        entries[key] = f
    return entries


BIBE = load_bib()


def clean_braces(s):
    s = s.replace("{", "").replace("}", "")
    s = s.replace("$n$", "n").replace("$", "")
    return s.replace("--", "\u2013")


def fmt_authors(a):
    if not a:
        return ""
    parts = [p.strip() for p in a.split(" and ")]
    out = []
    for p in parts:
        if "," in p:
            last, first = [x.strip() for x in p.split(",", 1)]
            ini = " ".join(w[0] + "." for w in first.replace(".", " ").split() if w)
            out.append(f"{last}, {ini}")
        else:
            out.append(p)
    if len(out) > 6:
        return ", ".join(out[:6]) + ", et al."
    return ", ".join(out)


def inline(s, refmap=None):
    """LaTeX inline -> ReportLab markup."""
    s = re.sub(r"(?<!\\)%.*?$", "", s, flags=re.M)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = s.replace("\n", " ")

    # citations -> [n]
    def _cite(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        nums = []
        for k in keys:
            if k not in CITEKEYS:
                CITEKEYS.append(k)
            nums.append(str(CITEKEYS.index(k) + 1))
        return "[" + ",".join(nums) + "]"
    s = re.sub(r"\\citep?\{([^}]+)\}", _cite, s)

    # cross references
    def _ref(m):
        return (refmap or {}).get(m.group(1), "?")
    s = re.sub(r"\\(?:eq)?ref\{([^}]+)\}", _ref, s)
    s = re.sub(r"Eq\.~\\?\(?", "Eq.&nbsp;(", s) if False else s

    s = re.sub(r"\\url\{([^}]+)\}", r"\1", s)
    # normalise subscripts inside \mathbf{...} first so tags cannot interleave
    s = re.sub(r"\\mathbf\{([A-Za-z]+)_\{(\w+)\}\}", r"<b>\1<sub>\2</sub></b>", s)
    s = re.sub(r"\\mathbf\{([A-Za-z]+)_(\w)\}", r"<b>\1<sub>\2</sub></b>", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"<i>\1</i>", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"<i>\1</i>", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\lVert", "||").replace("\\rVert", "||")
    s = s.replace("\\Vert", "|")
    s = s.replace("\\|", "||").replace("\\ast", "*")
    s = re.sub(r"\\sqrt\{([^{}]*)\}", "\u221a(\\1)", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\tfrac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\hat\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\bar\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\{", "\x01").replace("\\}", "\x02")

    # superscripts / subscripts
    s = re.sub(r"\^\{([^{}]*)\}", r"<super>\1</super>", s)
    s = re.sub(r"\^(\w)", r"<super>\1</super>", s)
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"_(\w)", r"<sub>\1</sub>", s)

    for k, v in sorted(GREEK.items(), key=lambda x: -len(x[0])):
        s = s.replace(k, v)

    s = re.sub(r"\\!|\\;|\\:", "", s)
    s = s.replace("$", "").replace("\\,", " ").replace("\\ ", " ")
    s = s.replace("\\%", "%").replace("\\&", "&amp;").replace("\\_", "_")
    s = s.replace("\\#", "#").replace("\\$", "$")
    s = s.replace("~", " ").replace("``", "\u201c").replace("''", "\u201d")
    s = s.replace("---", "\u2014")
    s = s.replace("--", "\u2013")
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\x01", "{").replace("\x02", "}")
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def render_eq(latex, width_pt=380):
    """Render a display equation via matplotlib mathtext."""
    key = hashlib.md5(latex.encode()).hexdigest()[:16]
    path = os.path.join(EQD, key + ".png")
    body = latex.strip()
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = re.sub(r"\s+", " ", body).strip().rstrip(",.")
    body = re.sub(r"\\qquad|\\quad", r"\\ \\ ", body)
    body = body.replace("\\mathrm", "\\mathregular")
    body = re.sub(r"\\bigl|\\bigr|\\Bigl|\\Bigr", "", body)
    body = body.replace("\\lVert", "\\|").replace("\\rVert", "\\|")
    body = body.replace("\\ast", "*")
    body = re.sub(r"\\!|\\,|\\;|\\:", "", body)
    body = re.sub(r"\\textstyle|\\displaystyle|\\limits", "", body)
    body = body.replace("\\Vert", "|")
    body = re.sub(r"\\mathregular\{([^{}]*)\}", r"\\mathrm{\1}", body)
    body = body.replace("&", "")
    body = body.replace("\\tfrac", "\\frac")
    body = re.sub(r"\\left\\?\||\\right\\?\|", "|", body)
    if not os.path.exists(path):
        for size in (13, 11, 9):
            try:
                fig = plt.figure(figsize=(0.01, 0.01))
                fig.text(0, 0, f"${body}$", fontsize=size)
                fig.savefig(path, dpi=260, bbox_inches="tight",
                            pad_inches=0.06, transparent=False, facecolor="white")
                plt.close(fig)
                break
            except Exception:
                plt.close("all")
                continue
        else:
            return None
    if not os.path.exists(path):
        return None
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    scale = min(width_pt / (w / 260 * 72), 1.0)
    iw = w / 260 * 72 * scale
    ih = h / 260 * 72 * scale
    if iw > BODY_W * 0.94:
        r = BODY_W * 0.94 / iw
        iw, ih = iw * r, ih * r
    return Image(path, width=iw, height=ih)


# ------------------------------------------------------------------ parse ---
tex = open(TEX).read()

# number sections/tables/figures/equations for cross-refs
refmap = {}
sec_n = 0
sub_n = 0
for line in tex.splitlines():
    m = re.match(r"\\section\{", line)
    if m:
        sec_n += 1
        sub_n = 0
    m2 = re.match(r"\\subsection\{", line)
    if m2:
        sub_n += 1
    lm = re.search(r"\\label\{(sec:[^}]+)\}", line)
    if lm:
        refmap[lm.group(1)] = f"{sec_n}.{sub_n}" if sub_n else f"{sec_n}"

for i, m in enumerate(re.finditer(r"\\label\{(tab:[^}]+)\}", tex), 1):
    refmap[m.group(1)] = str(i)
for i, m in enumerate(re.finditer(r"\\label\{(fig:[^}]+)\}", tex), 1):
    refmap[m.group(1)] = str(i)
for i, m in enumerate(re.finditer(r"\\label\{(eq:[^}]+)\}", tex), 1):
    refmap[m.group(1)] = f"({i})"
for i, m in enumerate(re.finditer(r"\\begin\{proposition\}\s*(?:\[[^\]]*\])?\s*\\label\{(prop:[^}]+)\}", tex), 1):
    refmap[m.group(1)] = str(i)
for i, m in enumerate(re.finditer(r"\\begin\{definition\}\s*(?:\[[^\]]*\])?\s*\\label\{(def:[^}]+)\}", tex), 1):
    refmap[m.group(1)] = str(i)
for i, m in enumerate(re.finditer(r"\\begin\{algorithm\}.*?\\label\{(alg:[^}]+)\}", tex, re.S), 1):
    refmap[m.group(1)] = str(i)

body_start = tex.index("\\section{Introduction}")
body_end = tex.index("\\bibliographystyle")
body = tex[body_start:body_end]

story = []

# ---- front matter
title = re.search(r"\\title\{(.*?)\n?\}", tex, re.S).group(1)
story.append(Paragraph(inline(title), S["title"]))
story.append(Paragraph("Mouad Louhichi<super>a,*</super> &nbsp; Redwane Nesmaoui<super>a</super> &nbsp; Mohamed Lazaar<super>a</super>", S["author"]))
story.append(Paragraph("<super>a</super> National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco", S["affil"]))
story.append(Paragraph("* Corresponding author: mouad_louhichi@um5.ac.ma", S["affil"]))
story.append(Spacer(1, 4))
story.append(Table([[""]], colWidths=[BODY_W],
                   style=TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor("#999999"))])))

abst = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
story.append(Paragraph("Abstract", S["absthead"]))
story.append(Paragraph(inline(abst, refmap), S["abstract"]))
hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", tex, re.S)
if hl:
    story.append(Paragraph("Highlights", S["absthead"]))
    items = [x.strip() for x in re.split(r"\\item", hl.group(1)) if x.strip()]
    story.append(ListFlowable(
        [ListItem(Paragraph(inline(i, refmap), S["hl"]), leftIndent=14)
         for i in items],
        bulletType="bullet", leftIndent=12,
        bulletFontName="Times-Roman", bulletFontSize=8))

kw = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", tex, re.S).group(1)
kw = " · ".join(x.strip() for x in kw.replace("\\sep", "|").split("|") if x.strip())
story.append(Paragraph("<b>Keywords:</b> " + inline(kw), S["kw"]))
story.append(Spacer(1, 6))
story.append(Table([[""]], colWidths=[BODY_W],
                   style=TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor("#999999"))])))
story.append(NextPageTemplate("two"))
story.append(FrameBreak())


def emit_table_env(block, avail=None, wide=False):
    tsty = ("tblw", "tblwb", "tblwh") if wide else ("tbl", "tblb", "tblh")
    cap = re.search(r"\\caption\{(.*?)\}\s*\n\s*\\label", block, re.S)
    lab = re.search(r"\\label\{(tab:[^}]+)\}", block)
    capt = cap.group(1) if cap else ""
    num = refmap.get(lab.group(1), "?") if lab else "?"

    tstart = block.find("\\begin{tabular}")
    if tstart < 0:
        return
    i = block.index("{", tstart + len("\\begin{tabular}") - 1)
    depth, j = 0, i
    while j < len(block):
        if block[j] == "{":
            depth += 1
        elif block[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    colspec = block[i + 1:j]
    tend = block.find("\\end{tabular}", j)
    content = block[j + 1:tend]
    ncol = len(re.findall(r"[lcrp]", re.sub(r"p\{[^}]*\}", "p", colspec)))

    content = re.sub(r"\\(toprule|midrule|bottomrule|addlinespace)", "@@RULE@@\\1", content)
    rows, styles = [], []
    rowi = 0
    pending_multirow = None
    for raw in content.split("\\\\"):
        rules = re.findall(r"@@RULE@@(\w+)", raw)
        line = re.sub(r"@@RULE@@\w+", "", raw).strip()
        for r in rules:
            if r == "toprule":
                styles.append(("LINEABOVE", (0, rowi), (-1, rowi), 1.0, colors.black))
            elif r == "midrule":
                styles.append(("LINEABOVE", (0, rowi), (-1, rowi), 0.5, colors.black))
            elif r == "bottomrule":
                styles.append(("LINEBELOW", (0, rowi - 1), (-1, rowi - 1), 1.0, colors.black))
        if not line:
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)&", line)]
        out = []
        for c in cells:
            mr = re.match(r"\\multirow\{\d+\}\{[^}]*\}\{(.*)\}$", c)
            if mr:
                c = mr.group(1)
            mc = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", c.strip())
            if mc:
                c = mc.group(2)
                styles.append(("SPAN", (0, rowi), (-1, rowi)))
            st = S[tsty[1]] if ("\\textbf" in c or "\\multicolumn" in c) else S[tsty[0]]
            if rowi == 0:
                st = S[tsty[2]]
            out.append(Paragraph(inline(c, refmap), st))
        while len(out) < ncol:
            out.append(Paragraph("", S["tbl"]))
        rows.append(out[:ncol])
        rowi += 1
    if not rows:
        return

    if avail is None:
        avail = BODY_W
    if ncol >= 8:
        w = [avail * 0.09, avail * 0.09] + [(avail * 0.82) / (ncol - 2)] * (ncol - 2)
    elif ncol == 7:
        w = [avail * 0.145, avail * 0.175] + [(avail * 0.68) / 5] * 5
    elif ncol == 6:
        w = [avail * 0.22] + [(avail * 0.78) / 5] * 5
    elif ncol == 5:
        w = [avail * 0.32] + [(avail * 0.68) / 4] * 4
    elif ncol == 3:
        w = [avail * 0.17, avail * 0.32, avail * 0.51]
    else:
        w = [avail / ncol] * ncol

    t = Table(rows, colWidths=w, repeatRows=1, hAlign="CENTER")
    base = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 1.4 if wide else 2.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4 if wide else 2.6),
            ("LEFTPADDING", (0, 0), (-1, -1), 1.2 if wide else 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 1.2 if wide else 3),
            ("ALIGN", (1, 1), (-1, -1), "CENTER")]
    t.setStyle(TableStyle(base + styles))
    story.append(KeepTogether([
        Paragraph(f"<b>Table {num}.</b> " + inline(capt, refmap), S["cap"]),
        t, Spacer(1, 12)]))


def emit_figure_env(block, avail=None, wide=False):
    img = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", block)
    cap = re.search(r"\\caption\{(.*?)\}\s*\n\s*\\label", block, re.S)
    lab = re.search(r"\\label\{(fig:[^}]+)\}", block)
    if img:
        path = os.path.join(HERE, img.group(1))
    elif "\\begin{tikzpicture}" in block:
        # TikZ figures are authoritative in the .tex; the preview build has no
        # LaTeX, so fall back to the equivalent monochrome PNG if one exists.
        path = os.path.join(HERE, "figures", "Figure_1.png")
    else:
        return
    if not os.path.exists(path):
        return
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    iw = avail if avail is not None else BODY_W
    ih = h * (iw / w)
    maxh = PAGE_H - 2 * MARGIN - 150
    if ih > maxh:
        ih = maxh
        iw = w * (ih / h)
    num = refmap.get(lab.group(1), "?") if lab else "?"
    capt = cap.group(1) if cap else ""
    story.append(KeepTogether([
        Image(path, width=iw, height=ih),
        Paragraph(f"<b>Fig. {num}.</b> " + inline(capt, refmap), S["cap"])]))


def emit_algorithm(block):
    global algn
    algn += 1
    cap = re.search(r"\\caption\{(.*?)\}", block, re.S)
    body = re.search(r"\\begin\{algorithmic\}(?:\[\d+\])?(.*?)\\end\{algorithmic\}",
                     block, re.S)
    if not body:
        return
    lines = []
    depth = 0
    for raw in body.group(1).split("\n"):
        t = raw.strip()
        if not t:
            continue
        if re.match(r"\\(EndFor|EndIf|EndWhile|EndProcedure)", t):
            depth = max(0, depth - 1)
            continue
        kw = None
        m = re.match(r"\\(State|For|If|ElsIf|Else|While|Return|Ensure|Require|Procedure)\b(.*)",
                     t, re.S)
        if m:
            kw, rest = m.group(1), m.group(2)
        else:
            kw, rest = "State", t
        rest = re.sub(r"^\{(.*)\}$", r"\1", rest.strip())
        cm_ = re.search(r"\\Comment\{(.*?)\}", rest)
        comment = cm_.group(1) if cm_ else None
        rest = re.sub(r"\\Comment\{.*?\}", "", rest)
        prefix = {"For": "for ", "If": "if ", "ElsIf": "else if ",
                  "Else": "else", "While": "while ", "Return": "return ",
                  "Procedure": "procedure "}.get(kw, "")
        txt = inline(prefix + rest, refmap)
        if comment:
            txt += f' <font color="#777777"><i>// {inline(comment, refmap)}</i></font>'
        lines.append(("&nbsp;" * (4 * depth)) + txt)
        if kw in ("For", "If", "While", "Procedure"):
            depth += 1
    rows = [[Paragraph(f"<font color='#999999'>{i}</font>", S["algn"]),
             Paragraph(l, S["alg"])] for i, l in enumerate(lines, 1)]
    t = Table(rows, colWidths=[14, COL_W_G - 14])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 0.6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    head = Paragraph(f"<b>Algorithm {algn}.</b> " +
                     inline(cap.group(1) if cap else "", refmap), S["cap"])
    story.append(KeepTogether([
        Table([[""]], colWidths=[COL_W_G],
              style=TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black)])),
        head, t,
        Table([[""]], colWidths=[COL_W_G],
              style=TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black)])),
        Spacer(1, 9)]))


_wide_open = [False]


def start_wide():
    return


def end_wide():
    """Nothing to do immediately; the band stays open for adjacent floats."""
    return


def close_wide():
    return


def emit_paragraphs(txt):
    txt = re.sub(r"(?<!\\)%.*?$", "", txt, flags=re.M)
    txt = re.sub(r"\\label\{[^}]*\}", "", txt)
    for chunk in re.split(r"\n\s*\n", txt):
        c = chunk.strip()
        if not c:
            continue
        close_wide()
        story.append(Paragraph(inline(c, refmap), S["body"]))


# walk the body
pos = 0
pattern = re.compile(
    r"\\section(?P<secstar>\*?)\{(?P<sec>(?:[^{}]|\{[^{}]*\})*)\}"
    r"|\\subsection\*?\{(?P<sub>(?:[^{}]|\{[^{}]*\})*)\}"
    r"|\\begin\{table\*\}(?P<tbls>.*?)\\end\{table\*\}"
    r"|\\begin\{table\}(?P<tbl>.*?)\\end\{table\}"
    r"|\\begin\{figure\*\}(?P<figs>.*?)\\end\{figure\*\}"
    r"|\\begin\{figure\}(?P<fig>.*?)\\end\{figure\}"
    r"|\\begin\{algorithm\}(?P<alg>.*?)\\end\{algorithm\}"
    r"|\\begin\{corollary\}(?P<cor>.*?)\\end\{corollary\}"
    r"|\\begin\{align\}(?P<align>.*?)\\end\{align\}"
    r"|\\begin\{equation\}(?P<eq>.*?)\\end\{equation\}"
    r"|\\begin\{(?P<lstype>itemize|enumerate)\}(?P<lst>.*?)\\end\{(?P=lstype)\}"
    r"|\\begin\{proposition\}(?P<prop>.*?)\\end\{proposition\}"
    r"|\\begin\{definition\}(?P<defn>.*?)\\end\{definition\}"
    r"|\\begin\{proof\}(?P<prf>.*?)\\end\{proof\}",
    re.S)

secn = 0
subn = 0
propn = 0
defn_n = 0
eqn = 0
for m in pattern.finditer(body):
    emit_paragraphs(body[pos:m.start()])
    pos = m.end()
    g = m.groupdict()
    if g["sec"] is not None:
        close_wide()
        raw = m.group(0)
        if "\\section*" in raw:
            story.append(Paragraph(inline(g["sec"]), S["h1s"]))
        else:
            secn += 1
            subn = 0
            story.append(Paragraph(f"{secn}. " + inline(g["sec"]), S["h1"]))
    elif g["sub"] is not None:
        close_wide()
        subn += 1
        story.append(Paragraph(f"{secn}.{subn}. " + inline(g["sub"]), S["h2"]))
    elif g["tbl"] is not None:
        emit_table_env(g["tbl"], avail=COL_W_G)
    elif g["tbls"] is not None:
        emit_table_env(g["tbls"], avail=COL_W_G, wide=True)
    elif g["fig"] is not None:
        emit_figure_env(g["fig"], avail=COL_W_G)
    elif g["figs"] is not None:
        emit_figure_env(g["figs"], avail=COL_W_G, wide=True)
    elif g["alg"] is not None:
        emit_algorithm(g["alg"])
    elif g["cor"] is not None:
        corn += 1
        t = re.sub(r"\\label\{[^}]*\}", "", g["cor"])
        story.append(Paragraph(f"<b>Corollary {corn}</b>. <i>" +
                               inline(t, refmap) + "</i>", S["prop"]))
    elif g["align"] is not None:
        eqn += 1
        im = render_eq(g["align"].replace("\\\\", " \\quad "), width_pt=COL_W_G - 26)
        if im:
            story.append(im)
    elif g["eq"] is not None:
        eqn += 1
        im = render_eq(g["eq"], width_pt=COL_W_G - 26)
        if im:
            et = Table([[im, Paragraph(f"({eqn})", S["eqn"])]],
                       colWidths=[COL_W_G - 22, 22])
            et.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                    ("ALIGN", (0, 0), (0, 0), "CENTER"),
                                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            story.append(et)
        else:
            story.append(Paragraph(inline(g["eq"], refmap), S["body"]))
    elif g["lst"] is not None:
        items = [x.strip() for x in re.split(r"\\item", g["lst"]) if x.strip()]
        bt = "1" if g["lstype"] == "enumerate" else "bullet"
        story.append(ListFlowable(
            [ListItem(Paragraph(inline(i, refmap), S["body"]), leftIndent=16)
             for i in items],
            bulletType=bt, start="1" if bt == "1" else None,
            leftIndent=14, bulletFontName="Times-Roman", bulletFontSize=10))
        story.append(Spacer(1, 3))
    elif g["prop"] is not None:
        propn += 1
        t = g["prop"]
        name = re.search(r"^\s*\[(.*?)\]", t)
        t = re.sub(r"^\s*\[.*?\]", "", t, count=1)
        t = re.sub(r"\\label\{[^}]*\}", "", t)
        head = f"<b>Proposition {propn}"
        head += f" ({inline(name.group(1))})</b>. " if name else "</b>. "
        # split embedded display equations
        sub = re.split(r"\\begin\{equation\}(.*?)\\end\{equation\}", t, flags=re.S)
        buf = [Paragraph(head + "<i>" + inline(sub[0], refmap) + "</i>", S["prop"])]
        for j in range(1, len(sub), 2):
            eqn += 1
            im = render_eq(sub[j], width_pt=COL_W_G - 30)
            if im:
                buf.append(im)
            if j + 1 < len(sub) and sub[j + 1].strip():
                buf.append(Paragraph("<i>" + inline(sub[j + 1], refmap) + "</i>", S["prop"]))
        story.append(KeepTogether(buf))
    elif g["defn"] is not None:
        defn_n += 1
        t = g["defn"]
        name = re.search(r"^\s*\[(.*?)\]", t)
        t = re.sub(r"^\s*\[.*?\]", "", t, count=1)
        head = f"<b>Definition {defn_n}"
        head += f" ({inline(name.group(1))})</b>. " if name else "</b>. "
        story.append(Paragraph(head + "<i>" + inline(t, refmap) + "</i>", S["prop"]))
    elif g["prf"] is not None:
        story.append(Paragraph("<b>Proof.</b> " + inline(g["prf"], refmap) + " \u25a1",
                               S["prop"]))

emit_paragraphs(body[pos:])

# ---- references
story.append(Paragraph("References", S["h1s"]))
for i, k in enumerate(CITEKEYS, 1):
    e = BIBE.get(k, {})
    au = fmt_authors(clean_braces(e.get("author", "")))
    ti = clean_braces(e.get("title", k))
    venue = e.get("journal") or e.get("booktitle") or ""
    venue = clean_braces(venue)
    yr = e.get("year", "")
    bits = [b for b in [au, ti, f"<i>{venue}</i>" if venue else "", yr] if b]
    line = f"[{i}] " + ", ".join(bits) + "."
    if e.get("pages"):
        line = line[:-1] + f", pp. {e['pages'].replace(chr(45)*2, chr(8211))}."
    if e.get("doi"):
        line += f" doi:{e['doi']}."
    story.append(Paragraph(line, S["ref"]))


# ------------------------------------------------------------------ build ---
def on_page(canv, doc):
    canv.saveState()
    canv.setFont("Times-Roman", 8)
    canv.setFillColor(colors.HexColor("#666666"))
    canv.drawString(MARGIN, PAGE_H - MARGIN + 12,
                    "Array (Elsevier) — submitted manuscript")
    canv.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 12,
                         "Louhichi, Nesmaoui & Lazaar")
    canv.setLineWidth(0.4)
    canv.line(MARGIN, PAGE_H - MARGIN + 8, PAGE_W - MARGIN, PAGE_H - MARGIN + 8)
    canv.drawCentredString(PAGE_W / 2, MARGIN - 16, str(doc.page))
    canv.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=MARGIN, rightMargin=MARGIN,
                      topMargin=MARGIN, bottomMargin=MARGIN,
                      title="CoopGCN — Array submission",
                      author="Louhichi, Nesmaoui, Lazaar")
H = PAGE_H - 2 * MARGIN
fl = Frame(MARGIN, MARGIN, COL_W_G, H, id="c1",
           leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
fr = Frame(MARGIN + COL_W_G + GUTTER, MARGIN, COL_W_G, H, id="c2",
           leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
fw = Frame(MARGIN, MARGIN, BODY_W, H, id="w",
           leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
TBAND = 0.545 * H          # title + abstract + highlights + keywords
ftitle = Frame(MARGIN, MARGIN + H - TBAND, BODY_W, TBAND, id="t",
               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
ftl = Frame(MARGIN, MARGIN, COL_W_G, H - TBAND - 8, id="tc1",
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
ftr = Frame(MARGIN + COL_W_G + GUTTER, MARGIN, COL_W_G, H - TBAND - 8, id="tc2",
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
doc.addPageTemplates([
    PageTemplate(id="title", frames=[ftitle, ftl, ftr], onPage=on_page),
    PageTemplate(id="two", frames=[fl, fr], onPage=on_page),
    PageTemplate(id="wide", frames=[fw], onPage=on_page),
])
doc.build(story)
print(f"wrote {OUT}")
