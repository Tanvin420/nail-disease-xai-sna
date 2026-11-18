"""Stage 16 - figures of the revised manuscript, each written as
   figures/manuscript/<main|si>/<name>.pdf (LaTeX), .png (preview), .tiff (submission, 300/600 dpi) and .drawio (editable).

Every figure is described once as a list of primitives (boxes, text, ellipses, lines/arrows, images) on a pixel canvas;
the same description is rendered with matplotlib (paper) and written as native draw.io XML, so both versions match.
All numbers are read from results/ and data/ (no hand-typed values).
Main figures: 1 pictorial study overview, 2 external per source, 3 shared-label cross-dataset matrix, 4 robustness
decomposition (s17), 5 XAI examples, 6 confusion networks + model similarity. (The partitioning-unit comparison is a table.) SI: representative images per class, and draw.io wrappers
(embedded image) for the stage-13 SI figures.
"""
import base64
import io
import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from PIL import Image  # noqa: E402

from nailrev.config import load_config  # noqa: E402
from nailrev.labels import CLASSES as CLASS_CODES, DATASETS  # noqa: E402

PX2PT = 0.72                      # canvas: 100 px per inch -> 1 px = 0.72 pt
MODEL_COLORS = ["#2a6fdb", "#e8633a", "#1aaa7a", "#e8a23a", "#d6609b"]
CLASS_SHORT = {"beau_s": "Beau's\nlines", "black_line": "Black\nline", "clubbing": "Clubbing", "healthy_nail": "Healthy\nnail",
               "onychomycosis": "Onycho-\nmycosis", "onycholysis": "Onycho-\nlysis", "white_spot": "White\nspot"}
DS_NAME = {"kopin": "NailGroup-7 (development)", "han": "Han A1+A2+B-E (hospital)", "han_A1": "Han A1 (hospital)",
           "han_A2": "Han A2 (hospital)", "han_BCDE": "Han B-E (hospital)", "han_virtualE": "Han Virtual-E (web)",
           "thai": "Thai Kaggle sets", "arthitaya": "Kaggle arthitaya", "arthitaya2": "Kaggle arthitaya-2",
           "nuttida": "Kaggle nuttidalapthanachai", "josephrasanjana": "Kaggle josephrasanjana",
           "nikhilgurav21": "Kaggle nikhilgurav21", "kaggle17": "Kaggle 17-class", "farisfaikar": "Kaggle nail-melanoma-350",
           "isratjahanmitu": "Kaggle isratjahanmitu", "legacy2272": "Combined Roboflow/KNM set", "own_test": "own test"}
GALLERY_BOXES = [(84, 162, 461, 538), (583, 162, 960, 538), (1081, 162, 1458, 538), (1580, 162, 1957, 538)]


# ---------------------------------------------------------------------------------------------------------------- canvas
class Canvas:
    def __init__(self, w: int, h: int, name: str):
        self.w, self.h, self.name, self.items, self.n = w, h, name, [], 1

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def rect(self, x, y, w, h, fill="#ffffff", stroke="#333333", sw=1.0, rounded=False, text="", fs=12, fc="#000000",
             bold=False, align="center", valign="middle", dashed=False, opacity=100):
        i = self._id()
        self.items.append(dict(k="rect", id=i, x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, rounded=rounded, text=text,
                               fs=fs, fc=fc, bold=bold, align=align, valign=valign, dashed=dashed, opacity=opacity, rot=0, italic=False))
        return i

    def text(self, x, y, w, h, text, fs=12, fc="#000000", bold=False, align="center", valign="middle", rot=0, italic=False):
        i = self._id()
        self.items.append(dict(k="rect", id=i, x=x, y=y, w=w, h=h, fill="none", stroke="none", sw=0, rounded=False, text=text,
                               fs=fs, fc=fc, bold=bold, align=align, valign=valign, dashed=False, opacity=100, rot=rot, italic=italic))
        return i

    def ellipse(self, cx, cy, rx, ry, fill="#ffffff", stroke="#333333", sw=1.0, text="", fs=10, fc="#000000", marker=False):
        i = self._id()
        self.items.append(dict(k="ellipse", id=i, cx=cx, cy=cy, rx=rx, ry=ry, fill=fill, stroke=stroke, sw=sw, text=text, fs=fs, fc=fc))
        return i

    def line(self, pts, color="#333333", sw=1.0, arrow=False, dashed=False, ctrl=None, opacity=100, head=8):
        i = self._id()
        self.items.append(dict(k="line", id=i, pts=[tuple(map(float, p)) for p in pts], color=color, sw=sw, arrow=arrow,
                               dashed=dashed, ctrl=ctrl, opacity=opacity, head=head))
        return i

    def image(self, x, y, w, h, img: Image.Image, fmt="JPEG"):
        i = self._id()
        self.items.append(dict(k="image", id=i, x=x, y=y, w=w, h=h, img=img.convert("RGB"), fmt=fmt))
        return i

    # ------------------------------------------------------------ matplotlib
    def render(self, out_base: Path, dpi_tiff=600):
        fig = plt.figure(figsize=(self.w / 100, self.h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.axis("off")
        z = 1
        for it in self.items:
            z += 1
            k = it["k"]
            if k == "rect":
                if it["fill"] != "none" or it["stroke"] != "none":
                    style = "round,pad=0,rounding_size=8" if it["rounded"] else "square,pad=0"
                    ax.add_patch(FancyBboxPatch((it["x"], it["y"]), it["w"], it["h"], boxstyle=style,
                                                facecolor=it["fill"] if it["fill"] != "none" else "none",
                                                edgecolor=it["stroke"] if it["stroke"] != "none" else "none",
                                                linewidth=it["sw"] * PX2PT, linestyle="--" if it["dashed"] else "-",
                                                alpha=it["opacity"] / 100, zorder=z))
                if it["text"] and it["rot"] == 90:          # vertical label reading bottom-to-top, anchored at the box bottom
                    ax.text(it["x"] + it["w"] / 2, it["y"] + it["h"] - 2, it["text"], ha="center", va="bottom", rotation=90,
                            fontsize=it["fs"] * PX2PT, color=it["fc"], fontweight="bold" if it["bold"] else "normal", zorder=z + 0.5)
                elif it["text"]:
                    ha = {"center": "center", "left": "left", "right": "right"}[it["align"]]
                    va = {"middle": "center", "top": "top", "bottom": "bottom", "baseline": "baseline"}[it["valign"]]
                    tx = it["x"] + (it["w"] / 2 if ha == "center" else (4 if ha == "left" else it["w"] - 4))
                    ty = it["y"] + {"center": it["h"] / 2, "top": 3, "bottom": it["h"] - 3, "baseline": it["h"] * 0.74}[va]
                    ax.text(tx, ty, it["text"], ha=ha, va=va, fontsize=it["fs"] * PX2PT, color=it["fc"],
                            fontweight="bold" if it["bold"] else "normal", fontstyle="italic" if it["italic"] else "normal",
                            rotation=it["rot"], rotation_mode="anchor" if it["rot"] else "default", zorder=z + 0.5,
                            linespacing=1.15)
            elif k == "ellipse":
                ax.add_patch(Ellipse((it["cx"], it["cy"]), 2 * it["rx"], 2 * it["ry"], facecolor=it["fill"], edgecolor=it["stroke"],
                                     linewidth=it["sw"] * PX2PT, zorder=z))
                if it["text"]:
                    ax.text(it["cx"], it["cy"], it["text"], ha="center", va="center", fontsize=it["fs"] * PX2PT, color=it["fc"], zorder=z + 0.5)
            elif k == "line":
                p = it["pts"]
                if it["ctrl"] is not None and len(p) == 2:
                    path = MPath([p[0], it["ctrl"], p[1]], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
                else:
                    path = MPath(p, [MPath.MOVETO] + [MPath.LINETO] * (len(p) - 1))
                ar = FancyArrowPatch(path=path, arrowstyle=f"-|>,head_length={it['head'] * 0.06:.2f},head_width={it['head'] * 0.035:.2f}"
                                     if it["arrow"] else "-", color=it["color"], linewidth=it["sw"] * PX2PT,
                                     linestyle="--" if it["dashed"] else "-", alpha=it["opacity"] / 100, zorder=z,
                                     mutation_scale=10, shrinkA=0, shrinkB=0)
                ax.add_patch(ar)
            elif k == "image":
                ax.imshow(np.asarray(it["img"]), extent=(it["x"], it["x"] + it["w"], it["y"] + it["h"], it["y"]), zorder=z,
                          interpolation="lanczos")
                ax.add_patch(Rectangle((it["x"], it["y"]), it["w"], it["h"], fill=False, edgecolor="#444444", linewidth=0.5, zorder=z + 0.1))
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_base.with_suffix(".pdf"))
        fig.savefig(out_base.with_suffix(".png"), dpi=200)
        fig.savefig(out_base.with_suffix(".tiff"), dpi=dpi_tiff, pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)

    # ------------------------------------------------------------ draw.io
    def drawio(self, out_path: Path):
        def font(it):
            return (1 if it.get("bold") else 0) + (2 if it.get("italic") else 0)

        def html(t):
            return escape(t).replace("\n", "&lt;br&gt;")
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        for it in self.items:
            k = it["k"]
            if k == "rect":
                al = {"center": "center", "left": "left", "right": "right"}[it["align"]]
                va = {"middle": "middle", "top": "top", "bottom": "bottom", "baseline": "middle"}[it["valign"]]
                st = (f"rounded={1 if it['rounded'] else 0};whiteSpace=wrap;html=1;fillColor={it['fill']};strokeColor={it['stroke']};"
                      f"strokeWidth={it['sw']};fontSize={it['fs']};fontColor={it['fc']};fontStyle={font(it)};align={al};"
                      f"verticalAlign={va};opacity={it['opacity']};spacing=2;" + ("dashed=1;" if it["dashed"] else ""))
                x, y, w, h = it["x"], it["y"], it["w"], it["h"]
                if it["rot"]:
                    cx, cy = x + w / 2, y + h / 2
                    if it["rot"] in (90, -90, 270):
                        w, h = h, w
                        x, y = cx - w / 2, cy - h / 2
                    st += f"rotation={-it['rot']};"
                cells.append(f'<mxCell id="{it["id"]}" value="{html(it["text"])}" style="{st}" vertex="1" parent="1">'
                             f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')
            elif k == "ellipse":
                st = (f"ellipse;whiteSpace=wrap;html=1;fillColor={it['fill']};strokeColor={it['stroke']};strokeWidth={it['sw']};"
                      f"fontSize={it['fs']};fontColor={it['fc']};")
                cells.append(f'<mxCell id="{it["id"]}" value="{html(it["text"])}" style="{st}" vertex="1" parent="1">'
                             f'<mxGeometry x="{it["cx"] - it["rx"]:.1f}" y="{it["cy"] - it["ry"]:.1f}" width="{2 * it["rx"]:.1f}" '
                             f'height="{2 * it["ry"]:.1f}" as="geometry"/></mxCell>')
            elif k == "line":
                p = it["pts"]
                st = (f"html=1;endArrow={'classic' if it['arrow'] else 'none'};endFill=1;endSize={max(3, int(it['head']))};"
                      f"strokeColor={it['color']};strokeWidth={it['sw']:.2f};opacity={it['opacity']};"
                      + ("dashed=1;" if it["dashed"] else "") + ("curved=1;" if it["ctrl"] is not None else "rounded=0;"))
                way = [it["ctrl"]] if it["ctrl"] is not None else p[1:-1]
                pts = "".join(f'<mxPoint x="{q[0]:.1f}" y="{q[1]:.1f}"/>' for q in way)
                arr = f'<Array as="points">{pts}</Array>' if way else ""
                cells.append(f'<mxCell id="{it["id"]}" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                             f'<mxPoint x="{p[0][0]:.1f}" y="{p[0][1]:.1f}" as="sourcePoint"/>'
                             f'<mxPoint x="{p[-1][0]:.1f}" y="{p[-1][1]:.1f}" as="targetPoint"/>{arr}</mxGeometry></mxCell>')
            elif k == "image":
                buf = io.BytesIO()
                im = it["img"]
                lim = it.get("maxpx", 600)                 # small photo panels; full-size SI figures keep up to 2400 px
                if max(im.size) > lim:
                    im = im.copy()
                    im.thumbnail((lim, lim))
                im.save(buf, format=it["fmt"], quality=90)
                mime = "image/jpeg" if it["fmt"] == "JPEG" else "image/png"
                b64 = base64.b64encode(buf.getvalue()).decode()
                st = f"shape=image;html=1;imageAspect=0;aspect=fixed;image=data:{mime},{b64};"
                cells.append(f'<mxCell id="{it["id"]}" value="" style="{st}" vertex="1" parent="1">'
                             f'<mxGeometry x="{it["x"]:.1f}" y="{it["y"]:.1f}" width="{it["w"]:.1f}" height="{it["h"]:.1f}" as="geometry"/></mxCell>')
        xml = (f'<mxfile host="drawio" type="device"><diagram id="{self.name}" name="{escape(self.name)}">'
               f'<mxGraphModel dx="{self.w}" dy="{self.h}" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
               f'page="1" pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" math="0" shadow="0"><root>'
               + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>\n")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(xml, encoding="utf-8")

    def save(self, out_dir: Path, dpi_tiff=600):
        base = out_dir / self.name
        self.render(base, dpi_tiff)
        self.drawio(base.with_suffix(".drawio"))
        print("figure", base.name)


def hexcol(v, cmap="Blues", lo=0.0, hi=1.0):
    c = plt.get_cmap(cmap)(0.12 + 0.88 * min(max((v - lo) / (hi - lo), 0), 1))
    return "#%02x%02x%02x" % tuple(int(255 * t) for t in c[:3])


def axis_x(cv: Canvas, x0, x1, y, ticks, label, fs=12):
    cv.line([(x0, y), (x1, y)], "#666666", 1)
    for t in ticks:
        x = x0 + (x1 - x0) * t
        cv.line([(x, y), (x, y + 5)], "#666666", 1)
        cv.text(x - 20, y + 6, 40, 16, f"{t:.1f}", fs=fs - 1, fc="#444444")
    cv.text(x0, y + 24, x1 - x0, 18, label, fs=fs)


def code_legend(cv: Canvas, x, y, w, items, cols, fs=11, title=None):
    """Aligned grid of 'CODE  full name' entries (codes bold in a fixed-width column); returns the y below the legend."""
    if title:
        cv.text(x, y, w, 18, title, fs=fs, bold=True, align="left")
        y += 20
    colw = w / cols
    codew = max(len(c) for c, _ in items) * fs * 0.66 + 14
    per_col = math.ceil(len(items) / cols)                  # filled column by column
    for i, (c, name) in enumerate(items):
        lx, ly = x + (i // per_col) * colw, y + (i % per_col) * 18
        cv.text(lx, ly, codew, 18, c, fs=fs, bold=True, align="left", valign="baseline")
        cv.text(lx + codew, ly, colw - codew - 4, 18, name, fs=fs, align="left", valign="baseline")
    return y + math.ceil(len(items) / cols) * 18


# ---------------------------------------------------------------------------------------------------------------- figures
def read_macros(res):
    import re
    return {k: v for f in sorted(res.glob("macros*.tex")) for k, v in re.findall(r"\\newcommand\{\\(R[A-Za-z]+)\}\{([^}]*)\}", f.read_text())}


def thumb(gdir, gal, image_id, k=0, model="efficientnet_b2"):
    f = gal[(gal.model == model) & (gal.image_id == image_id)].file.iloc[0]
    return crop_panels(gdir / f)[k]


def fig_overview(cfg, res, out, models, label):
    """Fig 1: pictorial overview of the study (data -> preparation -> training -> tests -> explanation/SNA -> outcome)."""
    M = read_macros(res)
    flow = json.loads((cfg.path("dedup") / "flow.json").read_text())["dev"]
    gal = pd.read_csv(res / "xai" / "gallery.csv")
    gdir = cfg.path("figures") / "gallery"
    sel = pd.read_csv(out / "fig5_xai_examples_selection.csv")
    classes = list(cfg.classes)
    corr = {r.row: r.image_id for r in sel[sel.kind == "correct"].itertuples()}
    es = pd.read_csv(res / "metrics" / "external_summary.csv")
    ext = sorted(set(es[(es.protocol == "cv") & es.source.str.startswith("ext_")].source.str[4:]))
    n_han = sum(s.startswith("han") for s in ext)
    W, H = 1200, 772
    cv = Canvas(W, H, "fig1_overview")
    PW, PH, xs, ys = 380, 352, [10, 410, 810], [8, 400]
    INK, GREY, ORA, BLUE = "#222222", "#666666", "#e8633a", "#2a6fdb"

    def panel(i, x, y, title, sub):
        cv.rect(x, y, PW, PH, "#f7f7f4", "#c9c9c4", 1.2, True)
        cv.ellipse(x + 26, y + 27, 15, 15, "#ffffff", INK, 1.6, str(i), fs=17, fc=INK)
        cv.text(x + 48, y + 12, PW - 56, 28, title, fs=21, bold=True, align="left", fc=INK)
        cv.text(x + 12, y + 44, PW - 24, 22, sub, fs=15, align="left", fc=GREY)

    def arrow(p, q):
        cv.line([p, q], "#a9a9a9", 3.0, arrow=True, head=13)

    # ---------------- 1 collect
    x, y = xs[0], ys[0]
    panel(1, x, y, "Collect", f"{1 + len(ext)} public nail datasets, 7 classes")
    tw, gap = 70, 12
    for i, c in enumerate(classes):
        r_, k_ = divmod(i, 4)
        n_row = 4 if r_ == 0 else 3
        x0 = x + (PW - (n_row * tw + (n_row - 1) * gap)) / 2 + k_ * (tw + gap)
        y0 = y + 76 + r_ * 98
        cv.image(x0, y0, tw, tw, thumb(gdir, gal, corr[c]))
        cv.text(x0 - 6, y0 + tw + 1, tw + 12, 20, CLASS_CODES[c][0], fs=14, bold=True, fc=INK)
    chips = [(f"Roboflow export\n{int(M['RKopinFiles']):,} files", "#dae8fc", "#6c8ebf"),
             (f"Han et al.\nhospital, web ({n_han})", "#d5e8d4", "#82b366"),
             (f"Kaggle sets\n({len(ext) - n_han})", "#fff2cc", "#d6b656")]
    for j, (txt, f_, s_) in enumerate(chips):
        cv.rect(x + 10 + j * 122, y + 278, 118, 58, f_, s_, 1.2, True, txt, fs=12.5, fc=INK)

    # ---------------- 2 prepare
    x, y = xs[1], ys[0]
    panel(2, x, y, "Prepare", "group copies, align labels")
    base = thumb(gdir, gal, corr["beau_s"])
    copies = [base, base.rotate(90), base.transpose(Image.FLIP_LEFT_RIGHT), base.rotate(180)]
    cx0, cy0, s = x + 92, y + 140, 50
    pts = [(cx0 - 30, cy0 - 30), (cx0 + 30, cy0 - 30), (cx0 - 30, cy0 + 30), (cx0 + 30, cy0 + 30)]
    for a in range(4):
        for b in range(a + 1, 4):
            cv.line([pts[a], pts[b]], "#7a7a7a", 1.4)
    for (px_, py_), im in zip(pts, copies):
        cv.image(px_ - s / 2, py_ - s / 2, s, s, im)
    cv.ellipse(cx0, cy0, 84, 76, "none", BLUE, 1.6)
    cv.text(x + 8, y + 222, 170, 20, "near-duplicate group", fs=14, fc=BLUE, bold=True)
    arrow((x + 184, cy0), (x + 236, cy0))
    cv.image(x + 246, cy0 - 40, 80, 80, base)
    cv.text(x + 220, y + 222, 136, 20, "1 independent unit", fs=14, fc=INK, bold=True)
    cv.text(x + 12, y + 244, PW - 24, 22, f"{int(M['RDevComponents']):,} groups  →  NailGroup-7: {int(M['RDevIncluded']):,} used",
            fs=14, fc=INK, bold=True)
    cv.line([(x + 12, y + 270), (x + PW - 12, y + 270)], "#d0d0cc", 1)
    cv.text(x + 12, y + 274, PW - 24, 20, "label mapping: label in other datasets → class used here", fs=11.5, fc=GREY,
            align="left", italic=True)
    for j, (a_, b_) in enumerate([("melanonychia", "Black line"), ("leukonychia", "White spot"), ("normal nail", "Healthy nail")]):
        cv.text(x + 8 + j * 122, y + 296, 122, 20, a_, fs=13, fc=INK)
        cv.text(x + 8 + j * 122, y + 316, 122, 22, "→ " + b_, fs=13.5, bold=True, fc=BLUE)

    # ---------------- 3 train
    x, y = xs[2], ys[0]
    panel(3, x, y, "Train", "5 architectures × 3 set-ups")
    short = {"vgg16": "VGG-16", "densenet121": "DenseNet\n-121", "efficientnet_b2": "EfficientNet\n-B2", "mobilenetv3": "MobileNet\nV3-Large",
             "vit_b16": "ViT-B/16"}
    for i, m in enumerate(models):
        mx = x + 14 + i * 72
        col = MODEL_COLORS[i]
        if m == "vit_b16":
            for a in range(3):
                for b in range(3):
                    cv.rect(mx + 14 + a * 13, y + 78 + b * 13, 11, 11, col, "#ffffff", 0.8)
        else:
            for k_ in range(3):
                cv.rect(mx + 10 + k_ * 9, y + 78 + k_ * 5, 18, 34 - k_ * 6, col, "#ffffff", 0.8, True)
        cv.text(mx - 4, y + 120, 72, 38, short[m], fs=12, fc=INK)
    cy_ = y + 182
    for k_ in range(5):                                    # (a) 5-fold CV
        cv.rect(x + 18 + k_ * 21, cy_, 18, 30, ORA if k_ == 2 else ("#f4d03f" if k_ == 3 else "#bcd3f5"), "#ffffff", 0.8)
    cv.text(x + 8, cy_ + 36, 120, 40, "within dataset\n5-fold × 3", fs=13, fc=INK)
    for a in range(4):                                     # (b) cross-dataset matrix
        for b in range(4):
            cv.rect(x + 152 + b * 17, cy_ - 8 + a * 12, 15, 10, "#d9d9d9" if a == b else "#9ec5fe", "#ffffff", 0.6)
    cv.text(x + 132, cy_ + 36, 112, 40, "train on one,\ntest on others", fs=13, fc=INK)
    for k_, col in enumerate(["#9ec5fe", "#b7e1cd", "#fbe3a6"]):   # (c) pooled
        cv.rect(x + 268 + k_ * 26, cy_ - 6, 20, 16, col, "#888888", 0.8)
        cv.line([(x + 278 + k_ * 26, cy_ + 10), (x + 304, cy_ + 24)], "#888888", 1.0)
    cv.rect(x + 290, cy_ + 24, 28, 12, "#c9b6e4", "#888888", 0.8)
    cv.text(x + 252, cy_ + 36, 120, 40, "pooled sources,\nhold one out", fs=13, fc=INK)
    cv.text(x + 12, y + 300, PW - 24, 30, f"{int(M['RNRunsTotal'])} trained models", fs=17, bold=True, fc=INK)
    arrow((xs[0] + PW + 2, ys[0] + PH / 2), (xs[1] - 2, ys[0] + PH / 2))
    arrow((xs[1] + PW + 2, ys[0] + PH / 2), (xs[2] - 2, ys[0] + PH / 2))
    cv.line([(xs[2] + PW / 2, ys[0] + PH + 1), (xs[2] + PW / 2, ys[0] + PH + 20), (xs[0] + PW / 2, ys[0] + PH + 20),
             (xs[0] + PW / 2, ys[1] - 2)], "#a9a9a9", 3.0, arrow=True, head=13)

    # ---------------- 4 test
    x, y = xs[0], ys[1]
    panel(4, x, y, "Test", "accuracy, robustness, cost")
    qx, qy, qw, qh = [x + 12, x + 194], [y + 74, y + 212], 174, 128
    for (i, j, title, txt) in [(0, 0, "Internal", "unseen pictures,\nsame dataset"), (1, 0, "External", f"{len(ext)} independent\nsources"),
                               (0, 1, "Cross-data", "train on one,\ntest on rest"), (1, 1, "Cost", "GPU, CPU,\nONNX runtime")]:
        bx, by = qx[i], qy[j]
        cv.rect(bx, by, qw, qh, "#ffffff", "#d0d0cc", 1.0, True)
        cv.text(bx + 62, by + 10, qw - 64, 22, title, fs=15, bold=True, align="left", fc=INK)
        cv.text(bx + 62, by + 40, qw - 64, 44, txt, fs=12.5, align="left", fc=GREY)
        ix, iy = bx + 10, by + 22
        if title == "Internal":
            for k_ in range(5):
                cv.rect(ix + k_ * 9, iy + 8, 8, 34, ORA if k_ == 2 else "#bcd3f5", "#ffffff", 0.5)
        elif title == "External":
            cv.ellipse(ix + 24, iy + 26, 22, 22, "#eaf2fd", BLUE, 1.4)
            cv.ellipse(ix + 24, iy + 26, 9, 22, "none", BLUE, 1.0)
            cv.line([(ix + 2, iy + 26), (ix + 46, iy + 26)], BLUE, 1.0)
        elif title == "Cross-data":
            sl = pd.read_csv(res / "metrics" / "shared_label_matrix.csv")
            sl = sl[sl.acc.notna() & (sl.n_runs == 1)].groupby(["train_set", "target"]).recall.mean()
            rows_, cols_ = ["kopin", "han", "han_virtualE", "thai"], ["own_test", "kopin", "han_A1", "han_virtualE", "arthitaya"]
            for a, r_ in enumerate(rows_):
                for b, c_ in enumerate(cols_):
                    v = sl.get((r_, c_), np.nan)
                    cv.rect(ix + b * 10, iy + 6 + a * 10, 9, 9, "#dddddd" if pd.isna(v) else hexcol(float(v)), "#ffffff", 0.4)
        else:
            cv.ellipse(ix + 24, iy + 28, 20, 20, "#ffffff", INK, 1.6)
            cv.rect(ix + 20, iy + 2, 8, 5, INK, INK, 1)
            cv.line([(ix + 24, iy + 28), (ix + 24, iy + 14)], INK, 1.8)
            cv.line([(ix + 24, iy + 28), (ix + 34, iy + 32)], INK, 1.8)
        cv.text(bx + 6, by + qh - 34, qw - 12, 30, {"Internal": "macro-F1, CI, tests", "External": "per-source balanced acc.",
                                                     "Cross-data": "variance decomposition", "Cost": "latency, memory"}[title],
                fs=12.5, italic=True, fc=INK)

    # ---------------- 5 explain & relate
    x, y = xs[1], ys[1]
    panel(5, x, y, "Explain & relate", "XAI faithfulness, error network")
    ex_id = corr["onycholysis"]
    cv.image(x + 14, y + 76, 76, 76, thumb(gdir, gal, ex_id, 0))
    cv.image(x + 96, y + 76, 76, 76, thumb(gdir, gal, ex_id, 1))
    cv.text(x + 10, y + 154, 170, 20, "image + Grad-CAM", fs=13, fc=GREY)
    ax0, ay0, aw, ah = x + 210, y + 80, 150, 70                       # deletion-curve icon (schematic)
    cv.line([(ax0, ay0), (ax0, ay0 + ah), (ax0 + aw, ay0 + ah)], GREY, 1.2)
    cv.line([(ax0 + 2, ay0 + 4), (ax0 + 30, ay0 + 40), (ax0 + 70, ay0 + 60), (ax0 + aw, ay0 + 66)], ORA, 2.4)
    cv.line([(ax0 + 2, ay0 + 4), (ax0 + 60, ay0 + 22), (ax0 + 110, ay0 + 42), (ax0 + aw, ay0 + 58)], GREY, 1.8, dashed=True)
    cv.text(ax0 - 4, y + 154, aw + 10, 20, "deletion test vs random", fs=13, fc=GREY)
    Rm = np.mean([pd.read_csv(res / "sna" / f"rates_{m}.csv", index_col=0).loc[classes, classes].to_numpy() for m in models], axis=0)
    np.fill_diagonal(Rm, 0)
    ncx, ncy, nr = x + PW / 2, y + 258, 60
    pos = {c: (ncx + 1.9 * nr * math.cos(-math.pi / 2 + 2 * math.pi * i / 7), ncy + nr * math.sin(-math.pi / 2 + 2 * math.pi * i / 7))
           for i, c in enumerate(classes)}
    top = np.dstack(np.unravel_index(np.argsort(-Rm, axis=None)[:6], Rm.shape))[0]
    sink = classes[int(Rm.sum(0).argmax())]
    for i, j in top:
        (x1, y1), (x2, y2) = pos[classes[i]], pos[classes[j]]
        L = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        cv.line([(x1 + ux * 17, y1 + uy * 17), (x2 - ux * 19, y2 - uy * 19)], "#555555", 1 + 25 * Rm[i, j], arrow=True, head=8,
                ctrl=((x1 + x2) / 2 - uy * 8, (y1 + y2) / 2 + ux * 8), opacity=75)
    for c in classes:
        cv.ellipse(*pos[c], 16, 16, ORA if c == sink else "#9ec5fe", "#ffffff", 1.2, CLASS_CODES[c][0], fs=11, fc=INK)
    cv.text(x + 8, y + 322, PW - 16, 22, "nodes = classes, arrows = confusions", fs=13, fc=GREY)

    # ---------------- 6 outcome
    x, y = xs[2], ys[1]
    panel(6, x, y, "Outcome", "what limits prediction")
    sh = [(float(M["RVarXdDatasetPct"]), BLUE, "data"), (float(M["RVarXdArchPct"]), ORA, "architecture"),
          (float(M["RVarXdInterPct"]), "#b0b0b0", "interaction")]
    cv.text(x + 12, y + 72, PW - 24, 20, "variance of cross-dataset performance", fs=14, bold=True, align="left", fc=INK)
    bx = x + 14
    for v, col, nm in sh:
        w_ = (PW - 28) * v / 100
        cv.rect(bx, y + 96, w_, 24, col, "#ffffff", 0.8)
        bx += w_
    for (v, col, nm), lx, lw_ in zip(sh, (x + 14, x + 106, x + 250), (88, 140, 120)):
        cv.rect(lx, y + 127, 12, 12, col, "#ffffff", 0.6)
        cv.text(lx + 15, y + 122, lw_, 20, f"{nm} {v:.0f}%" if v >= 10 else f"{nm} {v:.1f}%", fs=12, align="left", fc=INK,
                bold=(nm != "interaction"))
    cv.text(x + 12, y + 148, PW - 24, 20, "macro-F1 (5 architectures)", fs=14, bold=True, align="left", fc=INK)
    a0, a1 = x + 150, x + PW - 16
    for k_, (lab_, lo, hi, col) in enumerate([("same dataset", M["RCvMacroFOneMin"], M["RCvMacroFOneMax"], "#2e8b57"),
                                              ("hospital set", M["RHanAoneMacroFOneMin"], M["RHanAoneMacroFOneMax"], "#c0392b")]):
        yy = y + 176 + k_ * 30
        cv.text(x + 12, yy - 2, 136, 22, lab_, fs=13, align="left", fc=INK)
        cv.line([(a0, yy + 9), (a1, yy + 9)], "#dddddd", 6)
        cv.rect(a0 + (a1 - a0) * float(lo), yy + 2, max((a1 - a0) * (float(hi) - float(lo)), 4), 14, col, col, 0.5)
        cv.text(a0 + (a1 - a0) * float(hi) + 2 if float(hi) < 0.7 else a0 + (a1 - a0) * float(lo) - 82, yy - 2, 80, 22,
                f"{float(lo):.2f}–{float(hi):.2f}", fs=12.5, fc=col, bold=True, align="left" if float(hi) < 0.7 else "right")
    cv.ellipse(x + 26, y + 252, 10, 10, ORA, "#ffffff", 1)
    cv.text(x + 42, y + 241, PW - 50, 22, f"Onycholysis = top error sink ({M['RSnaOnycholysisSinkModels']}/5 models)", fs=13,
            align="left", fc=INK)
    cv.text(x + 14, y + 266, PW - 24, 22, "Grad-CAM ✓ sanity check    IG ✗", fs=13, align="left", fc=INK)
    cv.rect(x + 14, y + 296, PW - 28, 46, "#ffffff", INK, 1.4, True, "Robustness depends on the data\nmore than on the architecture",
            fs=14.5, bold=True, fc=INK)
    arrow((xs[0] + PW + 2, ys[1] + PH / 2), (xs[1] - 2, ys[1] + PH / 2))
    arrow((xs[1] + PW + 2, ys[1] + PH / 2), (xs[2] - 2, ys[1] + PH / 2))
    cv.save(out)


def fig_variance(cfg, res, out, models, label):
    """Fig 5: dataset robustness vs model robustness (variance decomposition, scripts/s17_robustness_decomposition.py)."""
    M = read_macros(res)
    an = pd.read_csv(res / "metrics" / "robustness_anova.csv")
    mg = pd.read_csv(res / "metrics" / "robustness_marginal_means.csv")
    W, H = 1040, 470
    cv = Canvas(W, H, "fig4_robustness_decomposition")
    INK = "#222222"
    col = {"Training dataset": "#1f4e9c", "Test dataset": "#4f86d6", "Training x test dataset (specific pair)": "#a9c6ef",
           "External source": "#2f6fd0", "Architecture": "#e8633a", "Architecture x dataset pair": "#f3b49b",
           "Architecture x external source": "#f3b49b", "Residual": "#d7d7d7"}
    nice = {"Training x test dataset (specific pair)": "train × test pair", "Architecture x dataset pair": "architecture × data",
            "Architecture x external source": "architecture × data", "Residual": "run-to-run (folds, seeds)",
            "Training dataset": "training dataset", "Test dataset": "test dataset", "External source": "test source",
            "Architecture": "architecture"}
    arch_int = float(M["RVarIntArchPct"])
    bars = [("Same dataset\n(internal test, 15 runs)", [("Architecture", arch_int), ("Residual", 100 - arch_int)]),
            ("Development models\non 10 external sources",
             [(r.term, 100 * r.eta2) for r in an[an.analysis == "B_external_sources"].itertuples()]),
            ("Cross-dataset matrix\n(7 training datasets)",
             [(r.term, 100 * r.eta2) for r in an[(an.analysis == "A_cross_dataset") & (an.score == "cc")].itertuples()])]
    cv.text(10, 8, 560, 26, "(a) Share of performance variance (%)", fs=16, bold=True, align="left", fc=INK)
    x0, x1 = 190, 580
    for i, (name, parts) in enumerate(bars):
        y = 52 + i * 78
        cv.text(4, y - 4, x0 - 12, 46, name, fs=13, align="right", fc=INK)
        xx = x0
        for term, v in parts:
            w_ = (x1 - x0) * v / 100
            cv.rect(xx, y, w_, 38, col[term], "#ffffff", 1.0)
            if v >= 7:
                cv.text(xx, y, w_, 38, f"{v:.0f}", fs=13, bold=True, fc="#ffffff" if term not in ("Residual", "Training x test dataset (specific pair)",
                                                                                                "Architecture x dataset pair", "Architecture x external source") else INK)
            elif v >= 2:
                cv.text(xx - 14, y - 20, w_ + 28, 18, f"{v:.1f}", fs=12, bold=True, fc=col[term])
            xx += w_
    for t_ in np.linspace(0, 1, 6):
        cv.line([(x0 + (x1 - x0) * t_, 284), (x0 + (x1 - x0) * t_, 290)], "#666666", 1)
        cv.text(x0 + (x1 - x0) * t_ - 20, 290, 40, 18, f"{int(100 * t_)}", fs=12, fc="#444444")
    cv.line([(x0, 284), (x1, 284)], "#666666", 1)
    leg = [("Training dataset", "training dataset"), ("Test dataset", "test dataset / source"),
           ("Training x test dataset (specific pair)", "specific dataset pair"), ("Architecture", "architecture"),
           ("Architecture x dataset pair", "architecture × data"), ("Residual", "run-to-run (folds, seeds)")]
    for k_, (term, txt) in enumerate(leg):
        lx, ly = 20 + (k_ % 3) * 190, 330 + (k_ // 3) * 26
        cv.rect(lx, ly, 18, 16, col[term], "#ffffff", 0.8)
        cv.text(lx + 24, ly - 3, 172, 22, txt, fs=12.5, align="left", fc=INK)
    cv.text(10, 392, 600, 70, "dataset-related parts in blue, architecture-related parts in orange;\n(a) cross-dataset: chance-corrected "
            "balanced accuracy on the shared classes", fs=12, italic=True, align="left", fc="#555555")
    # (b) marginal means
    bx0, bx1 = 770, 1015
    cv.text(630, 8, 400, 26, "(b) Mean transfer by factor level", fs=16, bold=True, align="left", fc=INK)
    a_ = mg[mg.analysis == "A_cross_dataset"]
    groups = [("Training dataset", "#1f4e9c", lambda lv: DATASETS[lv][0]), ("Architecture", "#e8633a", lambda lv: label[lv])]
    y = 52
    for fac, c_, nm in groups:
        d = a_[a_.factor == fac].sort_values("mean_cc", ascending=False)
        cv.text(630, y - 6, 370, 22, "training dataset" if fac == "Training dataset" else "architecture", fs=13.5, bold=True, align="left", fc=c_)
        y += 32
        for r in d.itertuples():
            cv.text(610, y - 9, bx0 - 616, 18, nm(r.level), fs=12, align="right", fc=INK)
            cv.line([(bx0, y), (bx1, y)], "#eeeeee", 1)
            cv.line([(bx0, y), (bx0 + (bx1 - bx0) * r.mean_cc, y)], c_, 3, opacity=60)
            cv.ellipse(bx0 + (bx1 - bx0) * r.mean_cc, y, 5.5, 5.5, c_, "#ffffff", 1)
            y += 21
        y += 10
    for t_ in np.linspace(0, 1, 6):
        cv.line([(bx0 + (bx1 - bx0) * t_, y), (bx0 + (bx1 - bx0) * t_, y + 6)], "#666666", 1)
        cv.text(bx0 + (bx1 - bx0) * t_ - 20, y + 6, 40, 18, f"{t_:.1f}", fs=12, fc="#444444")
    cv.line([(bx0, y), (bx1, y)], "#666666", 1)
    cv.text(bx0 - 40, y + 24, bx1 - bx0 + 80, 36, "chance-corrected balanced accuracy\n(mean over all transfer pairs)", fs=12,
            fc="#444444")
    cv.save(out)


def fig_external(cfg, res, out, models, label):
    es = pd.read_csv(res / "metrics" / "external_summary.csv")
    e_ = es[(es.protocol == "cv") & es.source.str.startswith("ext_")]
    order = e_.drop_duplicates("source").sort_values("n", ascending=False).source.tolist()
    rowh, sub = 54, 8
    leg_rows = math.ceil(len(order) / 3)
    H = 70 + len(order) * rowh + 70 + 30 + leg_rows * 18
    cv = Canvas(900, H, "fig2_external_by_source")
    x0, x1 = 200, 870
    y0 = 60
    for t in np.linspace(0, 1, 6):
        cv.line([(x0 + (x1 - x0) * t, y0 - 8), (x0 + (x1 - x0) * t, y0 + len(order) * rowh)], "#dddddd", 1)
    for i, m in enumerate(models):
        cv.ellipse(150 + i * 148 + 8, 18, 6, 6, MODEL_COLORS[i], "#ffffff", 1)
        cv.text(150 + i * 148 + 18, 9, 130, 18, label[m], fs=12, align="left")
    cv.text(150, 30, 720, 18, "filled: strict (any of the 7 classes may be predicted); open: restricted to the classes present in the source",
            fs=11, align="left", italic=True)
    for r, s in enumerate(order):
        d = e_[e_.source == s].set_index("model")
        yb = y0 + r * rowh
        if r % 2 == 0:
            cv.rect(20, yb, x1 - 20, rowh, "#f6f6f6", "none", 0)
        n = int(d.n.iloc[0])
        k = len(str(d.classes_present.iloc[0]).split(";"))
        cv.text(25, yb + 6, x0 - 35, 22, DATASETS[s[4:]][0], fs=14, bold=True, align="left")
        cv.text(25, yb + 27, x0 - 35, 20, f"n = {n:,}; {k} class{'es' if k > 1 else ''}", fs=11, fc="#444444", align="left")
        for i, m in enumerate(models):
            if m not in d.index:
                continue
            y = yb + 12 + i * sub
            lo, hi, v = float(d.loc[m, "bal_acc_lo"]), float(d.loc[m, "bal_acc_hi"]), float(d.loc[m, "bal_acc"])
            cv.line([(x0 + (x1 - x0) * lo, y), (x0 + (x1 - x0) * hi, y)], MODEL_COLORS[i], 2)
            cv.ellipse(x0 + (x1 - x0) * v, y, 4.5, 4.5, MODEL_COLORS[i], "#ffffff", 0.8)
            rv = float(d.loc[m, "restricted_bal_acc"])
            cv.ellipse(x0 + (x1 - x0) * rv, y, 4, 4, "#ffffff", MODEL_COLORS[i], 1.4)
    axis_x(cv, x0, x1, y0 + len(order) * rowh + 4, list(np.linspace(0, 1, 6)),
           "Balanced accuracy (mean of 15 CV models; bars: 95% bootstrap CI)", fs=12)
    code_legend(cv, 25, y0 + len(order) * rowh + 60, 850, [DATASETS[k] for k in DATASETS if "ext_" + k in order], cols=3, fs=11,
                title="Dataset codes")
    cv.save(out)


def fig_shared(cfg, res, out):
    sl = pd.read_csv(res / "metrics" / "shared_label_matrix.csv")
    ok = sl[sl.acc.notna() & (sl.n_runs == 1)]          # cross-dataset protocol: one model per architecture and training dataset
    rows = [t for t in ["kopin", "han", "han_virtualE", "thai", "nikhilgurav21", "kaggle17", "josephrasanjana"] if t in set(ok.train_set)]
    cols = ["own_test"] + [c for c in ["kopin", "han_A1", "han_A2", "han_virtualE", "arthitaya", "arthitaya2", "nuttida",
                                       "josephrasanjana", "nikhilgurav21", "kaggle17", "legacy2272"] if c in set(ok.target)]
    g = ok.groupby(["train_set", "target"]).agg(recall=("recall", "mean"), n=("n", "first"), k=("n_shared_classes", "first"))
    cw, rh, lw, top = 64, 46, 96, 62
    used = set(rows) | set(cols)
    legend_keys = [k for k in DATASETS if k in used and k != "own_test"] + ["own_test"]
    W = lw + cw * len(cols) + 20
    H = top + rh * len(rows) + 75 + 24 + math.ceil(len(legend_keys) / 3) * 18
    cv = Canvas(W, H, "fig3_shared_label_matrix")
    for j, c in enumerate(cols):
        cv.text(lw + j * cw, top - 28, cw, 24, DATASETS[c][0], fs=12, bold=True)
    for i, r in enumerate(rows):
        cv.text(5, top + i * rh, lw - 12, rh, DATASETS[r][0], fs=12, bold=True, align="right")
        for j, c in enumerate(cols):
            x, y = lw + j * cw, top + i * rh
            if (r, c) in g.index:
                v, n, k = g.loc[(r, c), "recall"], g.loc[(r, c), "n"], int(g.loc[(r, c), "k"])
                cv.rect(x, y, cw, rh, hexcol(v), "#ffffff", 1)
                fc = "#ffffff" if v > 0.62 else "#111111"
                cv.text(x, y + 4, cw, 22, f"{v:.2f}" + ("*" if n < 30 else ""), fs=13, fc=fc, bold=True)
                cv.text(x, y + 25, cw, 16, f"{k} cls", fs=10, fc=fc)
            else:
                cv.rect(x, y, cw, rh, "#efeee8", "#ffffff", 1, text="-", fs=12, fc="#888888")
    cv.text(lw, 6, cw * len(cols), 20, "tested on →", fs=12, italic=True)
    cv.text(5, top - 28, lw - 12, 24, "trained on ↓", fs=12, italic=True, align="right")
    yl = top + rh * len(rows) + 18
    cv.text(lw, yl, 60, 18, "0.0", fs=11, align="right")
    for t in range(11):
        cv.rect(lw + 66 + t * 26, yl, 26, 16, hexcol(t / 10), "none", 0)
    cv.text(lw + 66 + 11 * 26 + 4, yl, 40, 18, "1.0", fs=11, align="left")
    cv.text(lw + 66 + 11 * 26 + 44, yl - 2, W - (lw + 66 + 11 * 26 + 44) - 10, 40,
            "mean per-class recall on the shared classes (mean of 5 architectures);\n* fewer than 30 test images; '-' fewer than 2 shared classes",
            fs=11, align="left", valign="top")
    code_legend(cv, 20, yl + 44, W - 40, [DATASETS[k] for k in legend_keys], cols=3, fs=11, title="Dataset codes")
    cv.save(out)


def fig_sna(cfg, res, out, models, label):
    sna = res / "sna"
    nm = pd.read_csv(sna / "node_metrics.csv")
    gs = pd.read_csv(sna / "graph_summary.csv")
    classes = list(cfg.classes)
    W, H = 1020, 770
    cv = Canvas(W, H, "fig6_confusion_networks")
    pw, ph = 340, 330
    letters = "abcdef"
    for p, m in enumerate(models):
        ox, oy = (p % 3) * pw, (p // 3) * ph
        R = pd.read_csv(sna / f"rates_{m}.csv", index_col=0).loc[classes, classes].to_numpy()
        n1 = nm[(nm.model == m) & (nm["filter"] == "w>=0.01")].set_index("class")
        sink = gs[(gs.model == m) & (gs["filter"] == "w>=0.01")].top_sink.iloc[0]
        freq = gs[(gs.model == m) & (gs["filter"] == "w>=0.01")].top_sink_bootstrap_freq.iloc[0]
        cx, cy, rad = ox + pw / 2, oy + 192, 92
        cv.text(ox + 5, oy + 6, pw - 10, 22, f"({letters[p]}) {label[m]}", fs=14, bold=True)
        cv.text(ox + 5, oy + 28, pw - 10, 18, f"top sink: {cfg.display_names[sink]} (bootstrap {freq:.2f})", fs=11, italic=True)
        pos, rr = {}, {}
        for i, c in enumerate(classes):
            a = -math.pi / 2 + 2 * math.pi * i / len(classes)
            pos[c] = (cx + rad * math.cos(a), cy + rad * math.sin(a))
            rr[c] = 4 + 40 * math.sqrt(float(n1.loc[c, "in_strength"]))   # area ~ in-strength
        for i, a_ in enumerate(classes):
            for j, b_ in enumerate(classes):
                w = R[i, j]
                if i == j or w < 0.01:
                    continue
                (x1, y1), (x2, y2) = pos[a_], pos[b_]
                dx, dy = x2 - x1, y2 - y1
                L = math.hypot(dx, dy)
                ux, uy = dx / L, dy / L
                px, py = -uy, ux
                s = (x1 + ux * (rr[a_] + 2), y1 + uy * (rr[a_] + 2))
                t = (x2 - ux * (rr[b_] + 4), y2 - uy * (rr[b_] + 4))
                ctrl = ((s[0] + t[0]) / 2 + px * 14, (s[1] + t[1]) / 2 + py * 14)
                cv.line([s, t], "#3b3b3b", 0.6 + 45 * w, arrow=True, ctrl=ctrl, opacity=55, head=7)
        for c in classes:
            col = "#f4a261" if c == sink else "#9ec5fe"
            cv.ellipse(*pos[c], rr[c], rr[c], col, "#2b4a7a" if c != sink else "#b5651d", 1.2)
            a = math.atan2(pos[c][1] - cy, pos[c][0] - cx)
            lx, ly = cx + (rad + rr[c] + 17) * math.cos(a), cy + (rad + rr[c] + 13) * math.sin(a)
            cv.text(lx - 22, ly - 10, 44, 20, CLASS_CODES[c][0], fs=12, bold=True)
    # (f) model similarity graph
    ox, oy = 2 * pw, ph
    K = pd.read_csv(sna / "similarity_error_consistency_kappa.csv", index_col=0)
    P = pd.read_csv(sna / "similarity_pearson_confusion.csv", index_col=0)
    ms = pd.read_csv(sna / "mst_support.csv")
    cv.text(ox + 5, oy + 6, pw - 10, 22, "(f) Model similarity", fs=14, bold=True)
    cv.text(ox + 5, oy + 28, pw - 10, 34, "grey width ∝ error-consistency κ; blue = minimum\nspanning tree on 1 − Pearson r (s = bootstrap support)",
            fs=11, italic=True)
    cx, cy, rad = ox + pw / 2, oy + 200, 88
    pos = {m: (cx + rad * math.cos(-math.pi / 2 + 2 * math.pi * i / len(models)), cy + rad * math.sin(-math.pi / 2 + 2 * math.pi * i / len(models)))
           for i, m in enumerate(models)}
    for i, a_ in enumerate(models):
        for b_ in models[i + 1:]:
            k = float(K.loc[a_, b_])
            cv.line([pos[a_], pos[b_]], "#9a9a9a", 1 + 14 * max(k - 0.35, 0), opacity=60)
    for r in ms[ms.in_full_mst == 1].itertuples():
        (x1, y1), (x2, y2) = pos[r.model_a], pos[r.model_b]
        cv.line([pos[r.model_a], pos[r.model_b]], "#1f5fbf", 2.2, dashed=False)
        tpos = 0.5 if abs(y2 - y1) < 20 else 0.36                 # off-centre so labels of crossing edges do not collide
        mx, my = x1 + (x2 - x1) * tpos, y1 + (y2 - y1) * tpos
        L_ = math.hypot(x2 - x1, y2 - y1)
        nx_, ny_ = -(y2 - y1) / L_, (x2 - x1) / L_
        if nx_ * (mx - cx) + ny_ * (my - cy) > 0:          # place the label on the inner side of the edge
            nx_, ny_ = -nx_, -ny_
        cv.rect(mx + nx_ * 16 - 40, my + ny_ * 16 - 9, 80, 18, "#ffffff", "#1f5fbf", 0.8, False,
                f"r={float(P.loc[r.model_a, r.model_b]):.2f}  s={r.bootstrap_support:.2f}", fs=10, fc="#1f5fbf", bold=True, opacity=90)
    for i, m in enumerate(models):
        cv.ellipse(*pos[m], 12, 12, MODEL_COLORS[i], "#ffffff", 1.2)
        a = math.atan2(pos[m][1] - cy, pos[m][0] - cx)
        lx_ = min(max(cx + (rad + 40) * math.cos(a) - 60, ox + 2), ox + pw - 124)
        cv.text(lx_, cy + (rad + 28) * math.sin(a) - 9, 120, 18, label[m], fs=11, bold=True)
    code_legend(cv, 20, H - 98, W - 40, [CLASS_CODES[c] for c in classes], cols=4, fs=11, title="Class codes")
    cv.text(0, H - 22, W, 18, "Nodes = classes (area ∝ weighted in-strength; orange = top error sink); arrows i→j weighted by "
            "P(predicted j | true i), shown if ≥ 0.01; pooled group-disjoint CV test predictions.", fs=11, italic=True)
    cv.save(out)


def crop_panels(path):
    im = Image.open(path).convert("RGB")
    return [im.crop(b) for b in GALLERY_BOXES]


XAI_PICK = {  # (model, image_id) chosen by visual inspection: clean, no watermark, representative; misclassified examples follow the
    # consensus confusion edges. EfficientNet-B2 (r0_f0) made no healthy-nail error, so that example comes from ViT-B/16.
    "correct": {"beau_s": 11, "black_line": 7, "clubbing": 6, "healthy_nail": 5, "onychomycosis": 7, "onycholysis": 7, "white_spot": 7},
    "incorrect": {"beau_s": 0, "black_line": 1, "clubbing": 0, "healthy_nail": ("vit_b16", "kopin_4ad8f5e5573b"), "onychomycosis": 0,
                  "onycholysis": 1, "white_spot": 0},
}


def fig_xai(cfg, res, out, label):
    gal = pd.read_csv(res / "xai" / "gallery.csv")
    gdir = cfg.path("figures") / "gallery"
    classes = list(cfg.classes)
    im_w, gap, lw, top = 132, 6, 112, 56
    rowh = im_w + 30
    W = lw + 6 * im_w + 5 * gap + 26 + 10
    H = top + len(classes) * rowh + 34
    cv = Canvas(W, H, "fig5_xai_examples")
    xs = [lw + i * (im_w + gap) for i in range(4)] + [lw + 4 * (im_w + gap) + 26 + i * (im_w + gap) for i in range(2)]
    cv.text(xs[0], 2, 4 * im_w + 3 * gap, 22, "Correctly classified (EfficientNet-B2)", fs=14, bold=True)
    cv.text(xs[4], 2, 2 * im_w + gap, 22, "Misclassified", fs=14, bold=True)
    for x, t in zip(xs, ["Original", "Grad-CAM", "Integrated Gradients", "LIME", "Original", "Grad-CAM"]):
        cv.text(x, 28, im_w, 20, t, fs=12)
    cv.line([(xs[4] - 15, 26), (xs[4] - 15, H - 30)], "#999999", 1, dashed=True)
    sel_rows = []
    for r, c in enumerate(classes):
        y = top + r * rowh
        cv.text(2, y, lw - 8, im_w, CLASS_SHORT[c], fs=13, bold=True, align="right")
        sub = gal[(gal.model == "efficientnet_b2") & (gal["class"] == c) & (gal.correct == 1)].reset_index(drop=True)
        rc = sub.iloc[XAI_PICK["correct"][c]]
        panels = crop_panels(gdir / rc.file)
        for k in range(4):
            cv.image(xs[k], y, im_w, im_w, panels[k])
        cv.text(xs[0], y + im_w + 1, 4 * im_w + 3 * gap, 18, f"p = {rc.confidence:.2f}", fs=11, fc="#333333")
        pk = XAI_PICK["incorrect"][c]
        if isinstance(pk, tuple):
            ri = gal[(gal.model == pk[0]) & (gal.image_id == pk[1])].iloc[0]
        else:
            ri = gal[(gal.model == "efficientnet_b2") & (gal["class"] == c) & (gal.correct == 0)].reset_index(drop=True).iloc[pk]
        pi = crop_panels(gdir / ri.file)
        cv.image(xs[4], y, im_w, im_w, pi[0])
        cv.image(xs[5], y, im_w, im_w, pi[1])
        who = "" if ri.model == "efficientnet_b2" else f" [{label[ri.model]}]"
        cv.text(xs[4] - 6, y + im_w + 1, 2 * im_w + gap + 12, 18, f"→ {cfg.display_names[ri.pred]} (p = {ri.confidence:.2f}){who}",
                fs=11, fc="#b03a2e")
        sel_rows += [dict(row=c, kind="correct", model=rc.model, image_id=rc.image_id, pred=rc.pred, p=rc.confidence),
                     dict(row=c, kind="misclassified", model=ri.model, image_id=ri.image_id, pred=ri.pred, p=ri.confidence)]
    cv.text(0, H - 26, W, 20, "Maps are for the predicted class; Grad-CAM on the last convolutional stage; IG: black baseline, 64 steps; "
            "LIME: quickshift superpixels, 1,000 samples, top 10 segments.", fs=11, italic=True)
    cv.save(out, dpi_tiff=300)
    pd.DataFrame(sel_rows).to_csv(out / "fig5_xai_examples_selection.csv", index=False)
    return sel_rows


def fig_samples(cfg, res, out, sel_rows):
    gdir = cfg.path("figures") / "gallery"
    gal = pd.read_csv(res / "xai" / "gallery.csv")
    classes = list(cfg.classes)
    im_w, gap = 130, 8
    cv = Canvas(20 + len(classes) * (im_w + gap), im_w + 60, "figS1_class_examples")
    for i, c in enumerate(classes):
        rid = [r for r in sel_rows if r["row"] == c and r["kind"] == "correct"][0]["image_id"]
        f = gal[(gal.model == "efficientnet_b2") & (gal.image_id == rid)].file.iloc[0]
        x = 10 + i * (im_w + gap)
        cv.image(x, 10, im_w, im_w, crop_panels(gdir / f)[0])
        cv.text(x, im_w + 14, im_w, 36, cfg.display_names[c], fs=13, bold=True)
    cv.save(out, dpi_tiff=300)


def wrap_existing(src_png: Path, out_dir: Path, name: str):
    """draw.io wrapper (embedded PNG) for a raster/matplotlib SI figure from stage 13."""
    im = Image.open(src_png).convert("RGB")
    w = 1000
    h = int(im.size[1] * w / im.size[0])
    cv = Canvas(w, h, name)
    cv.items.append(dict(k="image", id="c2", x=0, y=0, w=w, h=h, img=im, fmt="PNG", maxpx=2400))
    out_dir.mkdir(parents=True, exist_ok=True)
    big = im.copy()
    cv.items[0]["img"] = big
    cv.drawio(out_dir / f"{name}.drawio")


def main():
    cfg = load_config()
    res = cfg.path("results")
    figs = cfg.path("figures")
    out_m, out_s = figs / "manuscript" / "main", figs / "manuscript" / "si"
    models = list(cfg.models)
    label = {m: cfg.models[m]["label"] for m in models}
    fig_external(cfg, res, out_m, models, label)
    fig_shared(cfg, res, out_m)
    fig_sna(cfg, res, out_m, models, label)
    sel = fig_xai(cfg, res, out_m, label)
    fig_samples(cfg, res, out_s, sel)
    fig_overview(cfg, res, out_m, models, label)
    fig_variance(cfg, res, out_m, models, label)
    si_src = {"figS2_dedup_calibration": figs / "si" / "fig_dedup_calibration.png",
              "figS3_learning_curves": figs / "si" / "fig_learning_curves.png",
              "figS4_confusion_matrices": figs / "si" / "fig_confusion_matrices.png",
              "figS5_per_class_recall": figs / "main" / "fig_per_class_recall.png",
              "figS6_reliability": figs / "si" / "fig_reliability.png",
              "figS7_risk_coverage": figs / "si" / "fig_risk_coverage.png",
              "figS8_cross_dataset_strict": figs / "main" / "fig_cross_dataset_matrix.png",
              "figS9_cross_dataset_restricted": figs / "main" / "fig_cross_dataset_matrix_restricted.png",
              "figS10_cross_dataset_by_model": figs / "si" / "fig_cross_dataset_matrix_by_model.png",
              "figS11_similarity_kappa": figs / "main" / "fig_similarity_error_consistency_kappa.png",
              "figS12_similarity_pearson": figs / "si" / "fig_similarity_pearson_confusion.png",
              "figS13_accuracy_efficiency_tradeoff": figs / "si" / "fig_accuracy_efficiency_tradeoff.png",
              "figS14_xai_vgg16": figs / "si" / "xai_vgg16.png", "figS15_xai_densenet121": figs / "si" / "xai_densenet121.png",
              "figS16_xai_efficientnet_b2": figs / "si" / "xai_efficientnet_b2.png", "figS17_xai_mobilenetv3": figs / "si" / "xai_mobilenetv3.png",
              "figS18_xai_vit_b16": figs / "si" / "xai_vit_b16.png", "figS19_xai_all_models": figs / "main" / "xai_gradcam_all_models.png"}
    for name, src in si_src.items():
        if src.exists():
            wrap_existing(src, out_s, name)
            im = Image.open(src).convert("RGB")
            im.save(out_s / f"{name}.png")
            im.save(out_s / f"{name}.tiff", compression="tiff_lzw", dpi=(300, 300))
            pdf = src.with_suffix(".pdf")
            if pdf.exists():
                (out_s / f"{name}.pdf").write_bytes(pdf.read_bytes())
            print("si", name)
    (figs / "manuscript" / "SI_SOURCES.json").write_text(json.dumps({k: str(v.relative_to(figs)) for k, v in si_src.items()}, indent=1))


if __name__ == "__main__":
    main()
