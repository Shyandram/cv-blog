"""Figures for DLCV 11A - Text in Images: OCR and Document Understanding.

Run:  .venv/bin/python scripts/figures/dl_11a_text_in_images.py
Writes PNGs to src/assets/deep-learning/dl-11a-text-in-images/ only.
Data: synthetic text rendered with matplotlib's default font (DejaVu Sans) and synthetic
probability matrices; nothing is downloaded and no network weights are used. Each experiment
repeats the code of the matching snippet in the chapter with the same seeds, so the plotted
values equal the printed ones. Runs in well under a minute on a CPU.
"""
import itertools
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage.draw import polygon as draw_polygon  # noqa: E402
from skimage.filters import threshold_otsu, threshold_sauvola  # noqa: E402
from skimage.transform import resize  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-11a-text-in-images"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"
STYLE = {"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
         "axes.facecolor": "white"}


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    w, h = Image.open(path).size
    assert w <= 1600, f"{name} is {w} px wide"
    assert path.stat().st_size < 1024 * 1024, f"{name} is larger than 1 MB"
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB, {w}x{h} px)")


# ----------------------------------------------------------------- Figure 1: the pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    ax.set_xlim(-1, 107)
    ax.set_ylim(-1, 47)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc="white", ec=color, lw=2))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=11, weight="bold", color=color)
        ax.text(x + w / 2, y + 1.4, body, ha="center", va="bottom", fontsize=10)

    def arrow(x0, y0, x1, y1, color="black", conn="arc3"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.6,
                                     color=color, connectionstyle=conn))

    y, h, w = 25, 17, 18.5
    xs = [1, 22.5, 44, 65.5, 87]
    titles = ["Image", "Detection", "Recognition", "Layout and\nunderstanding", "Output"]
    bodies = ["photo, scan,\nscreenshot",
              "where is text?\nboxes, quadrilaterals\nor polygons",
              "what does it say?\ncrop -> characters",
              "reading order,\nfields, tables",
              "plain text,\nkey: value pairs,\nan answer"]
    colors = [GRAY, BLUE, ORANGE, PURPLE, GREEN]
    for x, t, b, c in zip(xs, titles, bodies, colors):
        box(x, y, w, h, t, b, c)
    for i in range(4):
        arrow(xs[i] + w + 0.5, y + h / 2, xs[i + 1] - 0.5, y + h / 2)

    fails = ["blur, glare,\nlow resolution",
             "missed small words,\nmerged neighboring\nwords, curved lines",
             "look-alike glyphs (0/O, 1/l),\nunseen fonts, scripts,\nvertical text",
             "wrong column order,\nsplit tables,\nwrong key-value links",
             "errors from all\nearlier stages\nadd up"]
    for x, f in zip(xs, fails):
        ax.text(x + w / 2, y - 1.4, f, ha="center", va="top", fontsize=9.5, color=RED)

    # The OCR-free route: one image-to-text model replaces the three middle stages.
    ax.add_patch(FancyBboxPatch((22.5, 0.5), 61.5, 7.0, boxstyle="round,pad=0.4", fc="#f4f4f4",
                                ec=GRAY, lw=1.5, ls="--"))
    ax.text(53.25, 5.6, "OCR-free route: one image-to-text model (Donut, Pix2Struct, a VLM)",
            ha="center", va="center", fontsize=10)
    ax.text(53.25, 2.4, "writes the output directly; it can also write text that is not in the image",
            ha="center", va="center", fontsize=9.5, color=RED)
    arrow(xs[0] + 1.5, y - 0.6, 22.0, 4.0, GRAY, "angle,angleA=-90,angleB=0,rad=0")
    arrow(84.5, 4.0, xs[4] + w - 0.6, y - 0.6, GRAY, "angle,angleA=0,angleB=90,rad=0")
    ax.text(-0.5, 46.5, "Failure points (red) are listed under the stage where they start.",
            fontsize=9.5, color=RED, va="top")
    save(fig, "pipeline.png")


# ------------------------------------------------------- Figure 2: a detection toy (snippet 1)
H, W = 200, 560
WORDS = [("INVOICE", 20, 18, 22), ("No.", 200, 24, 14), ("1046", 245, 24, 14),
         ("Date:", 330, 24, 14), ("2026-11-01", 385, 24, 14),
         ("Item", 20, 80, 13), ("Qty", 260, 80, 13), ("Price", 400, 80, 13),
         ("Paper", 20, 110, 13), ("12", 265, 110, 13), ("3.50", 405, 110, 13),
         ("Total", 20, 160, 15), ("42.00", 395, 160, 15)]


def render(words):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    texts = [fig.text(x / W, 1 - y / H, s, fontsize=fs, va="top") for s, x, y, fs in words]
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].mean(axis=2) / 255.0
    r = fig.canvas.get_renderer()
    boxes = [[b.x0, H - b.y1, b.x1, H - b.y0] for b in (t.get_window_extent(r) for t in texts)]
    plt.close(fig)
    return img, np.array(boxes)


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)

    def area(r):
        return (r[2] - r[0]) * (r[3] - r[1])

    return inter / (area(a) + area(b) - inter)


def detect(img, thresh, gap=9):
    ink = img < thresh
    ink = ndi.binary_opening(ink, np.ones((2, 2)))
    words = ndi.binary_dilation(ink, np.ones((3, gap)))
    lab, _ = ndi.label(words)
    out = []
    for sl in ndi.find_objects(lab):
        y, x = sl
        if (y.stop - y.start) * (x.stop - x.start) > 40:
            out.append([x.start, y.start, x.stop, y.stop])
    return np.array(out, float), ink


def match(pred, gt, thr=0.5):
    used, tp_pred = set(), []
    for p in pred:
        cand = [(iou(p, g), j) for j, g in enumerate(gt) if j not in used]
        best = max(cand, default=(0, -1))
        if best[0] >= thr:
            used.add(best[1])
            tp_pred.append(True)
        else:
            tp_pred.append(False)
    tp = sum(tp_pred)
    P, R = tp / max(len(pred), 1), tp / len(gt)
    return P, R, 2 * P * R / max(P + R, 1e-9), np.array(tp_pred), used


def detection_data():
    rng = np.random.default_rng(0)
    clean, gt = render(WORDS)
    shade = 0.45 + 0.55 * np.linspace(0, 1, W)[None, :]
    img = np.clip(clean * shade + rng.normal(0, 0.04, clean.shape), 0, 1)
    return img, gt


def fig_detection():
    img, gt = detection_data()
    t_otsu, t_sau = threshold_otsu(img), threshold_sauvola(img, window_size=31)
    fig = plt.figure(figsize=(10.4, 7.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=0.38, wspace=0.08)

    def show(ax, im, title):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    def draw_boxes(ax, pred, gt_boxes, tp, used):
        for g_i, g in enumerate(gt_boxes):
            if g_i not in used:
                ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec=ORANGE,
                                       lw=1.5, ls="--"))
        for p, ok in zip(pred, tp):
            ax.add_patch(Rectangle((p[0], p[1]), p[2] - p[0], p[3] - p[1], fill=False,
                                   ec=GREEN if ok else RED, lw=1.5))

    ax = fig.add_subplot(gs[0, 0])
    show(ax, img, "(a) input: uneven light + noise; ground-truth boxes")
    for g in gt:
        ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec=BLUE, lw=1.2))
    ax = fig.add_subplot(gs[0, 1])
    pred, ink = detect(img, t_otsu)
    P, R, F, tp, used = match(pred, gt)
    show(ax, 1 - ink, f"(b) Otsu global threshold: P={P:.2f} R={R:.2f} H={F:.2f}")
    draw_boxes(ax, pred, gt, tp, used)
    ax = fig.add_subplot(gs[1, 0])
    pred, ink = detect(img, t_sau)
    P, R, F, tp, used = match(pred, gt)
    show(ax, 1 - ink, f"(c) Sauvola local threshold, gap 9: P={P:.2f} R={R:.2f} H={F:.2f}")
    draw_boxes(ax, pred, gt, tp, used)
    ax = fig.add_subplot(gs[1, 1])
    pred, ink = detect(img, t_sau, 25)
    P, R, F, tp, used = match(pred, gt)
    show(ax, 1 - ink, f"(d) Sauvola, gap 25 (words merge): P={P:.2f} R={R:.2f} H={F:.2f}")
    draw_boxes(ax, pred, gt, tp, used)

    ax = fig.add_subplot(gs[2, :])
    gaps = np.arange(1, 46, 2)
    stats = np.array([match(detect(img, t_sau, g)[0], gt)[:3] for g in gaps])
    ax.plot(gaps, stats[:, 0], "o-", color=BLUE, ms=4, label="precision")
    ax.plot(gaps, stats[:, 1], "s-", color=ORANGE, ms=4, label="recall")
    ax.plot(gaps, stats[:, 2], "^-", color=GREEN, ms=4, lw=2, label="H-mean")
    ax.set_xlabel("dilation width that bridges gaps between letters (pixels)")
    ax.set_ylabel("score at IoU 0.5")
    ax.set_title("(e) one hand-set gap: too small splits words into letters, too large merges words",
                 fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", ncol=3, fontsize=9)
    fig.text(0.5, 0.355, "green = true positive, red = false positive, dashed orange = missed word",
             ha="center", fontsize=9.5)
    save(fig, "detection-toy.png")


# --------------------------------------------------- Figure 3: upright box vs quadrilateral
def rotated_word(cx, cy, w, h, deg):
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
    return corners @ R.T + [cx, cy]


def poly_mask(quad, shape=(300, 300)):
    m = np.zeros(shape, bool)
    rr, cc = draw_polygon(quad[:, 1], quad[:, 0], shape)
    m[rr, cc] = True
    return m


def fig_quad():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    deg = 20
    quad = rotated_word(150, 150, 160, 24, deg)
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    ax.add_patch(Polygon(quad, closed=True, fc="#dbe9f6", ec=BLUE, lw=2, label="word quadrilateral"))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=RED, lw=2, ls="--",
                           label="tightest upright box"))
    ax.text(150, 150, "OPEN DAILY", rotation=-deg, ha="center", va="center", fontsize=13,
            weight="bold", rotation_mode="anchor")
    ax.set_xlim(40, 260)
    ax.set_ylim(225, 75)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"(a) a 160 x 24 word rotated {deg} degrees", fontsize=10)

    ax = axes[1]
    angles = np.linspace(0, 90, 181)
    w, h = 160, 24
    t = np.deg2rad(angles)
    aabb = (w * np.cos(t) + h * np.sin(t)) * (w * np.sin(t) + h * np.cos(t))
    ax.plot(angles, w * h / aabb, color=BLUE, lw=2, label="exact (area ratio)")
    pts = [0, 10, 20, 30, 45]
    vals = []
    for d in pts:
        q = rotated_word(150, 150, w, h, d)
        a0, b0 = q.min(axis=0)
        a1, b1 = q.max(axis=0)
        box = np.array([[a0, b0], [a1, b0], [a1, b1], [a0, b1]])
        m1, m2 = poly_mask(q), poly_mask(box)
        vals.append((m1 & m2).sum() / (m1 | m2).sum())
    ax.plot(pts, vals, "o", color=ORANGE, ms=7, label="rasterized (snippet)")
    ax.axhline(0.5, color=RED, ls="--", lw=1.2)
    ax.text(30, 0.53, "IoU 0.5 matching threshold", color=RED, fontsize=9.5)
    ax.set_xlabel("rotation angle (degrees)")
    ax.set_ylabel("IoU(word, its upright box)")
    ax.set_title("(b) even a perfect upright box fails IoU 0.5 beyond a few degrees", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "quad-iou.png")


# ------------------------------------------------- Figure 4: differentiable binarization
def db(P, T, k=50.0):
    return 1.0 / (1.0 + np.exp(-k * (P - T)))


def fig_db():
    fig = plt.figure(figsize=(10.4, 5.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 0.75], hspace=0.5, wspace=0.4)
    x = np.linspace(-0.3, 0.3, 601)
    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(x, (x >= 0).astype(float), color="black", lw=2, label="step (standard)")
    for k, c in [(1, GRAY), (10, ORANGE), (50, BLUE)]:
        ax.plot(x, db(x, 0, k), color=c, lw=1.8, label=f"DB, k={k}")
    ax.set_xlabel("x = P - T")
    ax.set_ylabel("binary value")
    ax.set_title("(a) the approximate step sharpens as k grows", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[0, 2:4])
    for k, c in [(1, GRAY), (10, ORANGE), (50, BLUE)]:
        ax.plot(x, k * (1 - db(x, 0, k)), color=c, lw=1.8, label=f"k={k}")
    ax.set_xlabel("x = P - T for a pixel labeled text")
    ax.set_ylabel("|d loss / d x|")
    ax.set_title("(b) gradient is large where the pixel is on the wrong side", fontsize=10)
    ax.axvspan(-0.3, 0, color=RED, alpha=0.07)
    ax.text(-0.28, 45, "wrong side\n(x below 0)", fontsize=9, color=RED, va="top")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # Two words separated by a 4-pixel gap; the blurry probability map does not drop to 0 there.
    words = np.zeros((60, 160))
    words[20:40, 12:78] = 1
    words[20:40, 82:148] = 1
    P = np.clip(ndi.gaussian_filter(words, 3.0) * 1.05, 0, 1)
    inside = words > 0
    dist_edge = np.where(inside, ndi.distance_transform_edt(inside), ndi.distance_transform_edt(~inside))
    T = 0.3 + 0.4 * np.exp(-dist_edge ** 2 / (2 * 3.0 ** 2))   # high near every word border
    fixed = P >= 0.3
    learned = db(P, T) > 0.5
    n_fixed = ndi.label(fixed)[1]
    n_db = ndi.label(learned)[1]
    panels = [(P, "(c) probability map P", "viridis"),
              (T, "(d) threshold map T", "viridis"),
              (fixed, f"(e) P >= 0.3: {n_fixed} region", "gray_r"),
              (learned, f"(f) DB(P, T) > 0.5: {n_db} regions", "gray_r")]
    for c, (im, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "db-binarization.png")


# ----------------------------------------------------------------- Figure 5: CTC decoding
ALPHABET = ["-", "c", "a", "t"]
PROBS = np.array([[0.10, 0.80, 0.05, 0.05],
                  [0.20, 0.05, 0.70, 0.05],
                  [0.60, 0.05, 0.30, 0.05],
                  [0.55, 0.02, 0.03, 0.40],
                  [0.55, 0.02, 0.03, 0.40]])


def collapse(path):
    merged = [s for i, s in enumerate(path) if i == 0 or s != path[i - 1]]
    return "".join(ALPHABET[s] for s in merged if s != 0)


def fig_ctc():
    total = {}
    for path in itertools.product(range(4), repeat=len(PROBS)):
        p = np.prod(PROBS[np.arange(len(PROBS)), path])
        total[collapse(path)] = total.get(collapse(path), 0.0) + p
    greedy = PROBS.argmax(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    im = ax.imshow(PROBS.T, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for t in range(PROBS.shape[0]):
        for s in range(PROBS.shape[1]):
            ax.text(t, s, f"{PROBS[t, s]:.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if PROBS[t, s] > 0.5 else "black")
    ax.plot(range(5), greedy, "o", ms=26, mfc="none", mec=RED, mew=2)
    ax.set_xticks(range(5), [f"frame {t + 1}" for t in range(5)])
    ax.set_yticks(range(4), ["blank -", "c", "a", "t"])
    ax.set_title("(a) per-frame probabilities; red = greedy choice\n"
                 "greedy path 'ca---' collapses to 'ca'", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax = axes[1]
    top = sorted(total.items(), key=lambda kv: -kv[1])[:6]
    labels = [repr(k) for k, _ in top]
    vals = [v for _, v in top]
    colors = [GREEN if k == "cat" else (RED if k == "ca" else GRAY) for k, _ in top]
    ax.barh(range(len(top)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9.5)
    ax.set_yticks(range(len(top)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("p(text | image), summed over all alignments")
    ax.set_title("(b) exact text probabilities: 'cat' wins although\n"
                 "no single path through it beats the greedy path", fontsize=10)
    ax.set_xlim(0, 0.42)
    fig.tight_layout()
    save(fig, "ctc-decoding.png")


# --------------------------------------------------------------- Figure 6: CER and WER
def edit_matrix(ref, hyp):
    n, m = len(ref), len(hyp)
    E = np.zeros((n + 1, m + 1), int)
    E[:, 0], E[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E[i, j] = min(E[i - 1, j] + 1, E[i, j - 1] + 1, E[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    path, ops = [(n, m)], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and E[i, j] == E[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append("S" if ref[i - 1] != hyp[j - 1] else "=")
            i, j = i - 1, j - 1
        elif i > 0 and E[i, j] == E[i - 1, j] + 1:
            ops.append("D")
            i -= 1
        else:
            ops.append("I")
            j -= 1
        path.append((i, j))
    return E, path[::-1], ops[::-1]


def counts(ref, hyp):
    _, _, ops = edit_matrix(ref, hyp)
    return ops.count("S"), ops.count("D"), ops.count("I")


def fig_cer():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})
    ref, hyp = "Total 25.80", "Tota1 25.8O"
    E, path, _ = edit_matrix(ref, hyp)
    ax = axes[0]
    ax.imshow(E, cmap="Greys", vmin=0, vmax=E.max() * 2.2)
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            ax.text(j, i, str(E[i, j]), ha="center", va="center", fontsize=8)
    pi, pj = zip(*path)
    ax.plot(pj, pi, "-", color=RED, lw=2, alpha=0.8)
    for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
        if i1 == i0 + 1 and j1 == j0 + 1 and ref[i0] != hyp[j0]:
            ax.plot(j1, i1, "s", ms=13, mfc="none", mec=RED, mew=2)
    ax.set_xticks(range(len(hyp) + 1), [""] + [c if c != " " else "_" for c in hyp])
    ax.set_yticks(range(len(ref) + 1), [""] + [c if c != " " else "_" for c in ref])
    ax.set_xlabel("recognized (hypothesis)")
    ax.set_ylabel("reference")
    ax.set_title("(a) edit-distance table; red path = optimal alignment,\n"
                 "squares = the 2 substitutions (CER = 2/11)", fontsize=10)
    ax = axes[1]
    full = "Total due 25.80 EUR by 1 May"
    hyps = [("look-alike glyphs", "Tota1 due 25.8O EUR by l May"),
            ("missing space", "Total due25.80 EUR by 1 May"),
            ("last words lost", "Total due 25.80 EUR")]
    cer = [sum(counts(full, h)) / len(full) for _, h in hyps]
    wer = [sum(counts(full.split(), h.split())) / len(full.split()) for _, h in hyps]
    yb = np.arange(len(hyps))
    ax.barh(yb - 0.18, cer, height=0.34, color=BLUE, label="CER")
    ax.barh(yb + 0.18, wer, height=0.34, color=ORANGE, label="WER")
    for i in range(len(hyps)):
        ax.text(cer[i] + 0.01, i - 0.18, f"{cer[i]:.3f}", va="center", fontsize=9)
        ax.text(wer[i] + 0.01, i + 0.18, f"{wer[i]:.3f}", va="center", fontsize=9)
    ax.set_yticks(yb, [f"{n}\n'{h}'" for n, h in hyps], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.55)
    ax.set_xlabel("error rate")
    ax.set_title(f"(b) reference: '{full}'", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, "cer-wer.png")


# ------------------------------------------------------- Figure 7: look-alike glyphs
CHARS = "0O8B5S1lIZ2aekx"


def glyph(c):
    fig = plt.figure(figsize=(0.48, 0.48), dpi=100)
    fig.text(0.5, 0.5, c, fontsize=26, ha="center", va="center")
    fig.canvas.draw()
    img = 1 - np.asarray(fig.canvas.buffer_rgba())[..., :3].mean(axis=2) / 255.0
    plt.close(fig)
    return img


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())


def glyph_accuracy(heights):
    templates = {c: glyph(c) for c in CHARS}
    rng = np.random.default_rng(0)
    acc = []
    for h in heights:
        wrong, total = 0, 0
        for c, img in templates.items():
            for _ in range(20):
                low = resize(img, (h, h), anti_aliasing=True)
                low = low + rng.normal(0, 0.05, low.shape)
                back = resize(low, (48, 48), order=1)
                guess = max(CHARS, key=lambda t: ncc(back, templates[t]))
                total += 1
                wrong += guess != c
        acc.append(1 - wrong / total)
    return templates, np.array(acc)


def fig_glyphs():
    heights = [48, 12, 8, 6]
    templates, acc_snip = glyph_accuracy(heights)
    sweep = [4, 5, 6, 7, 8, 10, 12, 16, 24, 48]
    _, acc_sweep = glyph_accuracy(sweep)
    fig = plt.figure(figsize=(10.4, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.25)
    sub = gs[0, 0].subgridspec(len(heights), 7, hspace=0.12, wspace=0.08)
    shown = "0O8B1lI"
    for r, h in enumerate(heights):
        for c, ch in enumerate(shown):
            ax = fig.add_subplot(sub[r, c])
            low = resize(templates[ch], (h, h), anti_aliasing=True)
            ax.imshow(1 - low, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(ch, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{h} px", fontsize=10)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(sweep, acc_sweep, "o-", color=BLUE, lw=2)
    ax.plot(heights, acc_snip, "s", color=ORANGE, ms=8, mfc="none", mew=2, label="values printed by the snippet")
    ax.set_xscale("log")
    ax.set_xticks(sweep, [str(s) for s in sweep])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("pixels per character (log scale)")
    ax.set_ylabel("accuracy of a template reader")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("(b) 15 characters, 20 noisy copies each", fontsize=10)
    fig.text(0.29, 0.94, "(a) the same glyphs seen with fewer pixels\n(column 6 is lowercase L, column 7 is uppercase i)",
             ha="center", fontsize=10)
    save(fig, "confusable-glyphs.png")


# ---------------------------------------------------------- Figure 8: reading order
def page_boxes():
    left = "Pay the invoice within thirty days of receipt".split()
    right = "Late payments carry a two percent monthly fee".split()
    words, boxes = [], []
    for col, text in [(0, left), (1, right)]:
        for k, w in enumerate(text):
            row, pos = divmod(k, 3)
            x0 = 40 + 300 * col + 80 * pos
            y0 = 100 + 30 * row + 3 * col
            words.append(w)
            boxes.append([x0, y0, x0 + 70, y0 + 20])
    return words, np.array(boxes, float)


def naive_order(boxes, line_tol=10):
    line = np.round(boxes[:, 1] / line_tol)
    return np.lexsort((boxes[:, 0], line))


def column_order(boxes, line_tol=10):
    x_min, x_max = int(boxes[:, 0].min()), int(boxes[:, 2].max())
    covered = np.zeros(x_max - x_min, bool)
    for x0, _, x1, _ in boxes.astype(int):
        covered[x0 - x_min:x1 - x_min] = True
    edges = np.flatnonzero(np.diff(np.r_[1, covered.astype(int), 1]))
    starts, ends = edges[::2], edges[1::2]
    widest = np.argmax(ends - starts)
    split = x_min + (starts[widest] + ends[widest]) / 2
    col = (boxes[:, 0] > split).astype(int)
    line = np.round(boxes[:, 1] / line_tol)
    return np.lexsort((boxes[:, 0], line, col))


def fig_order():
    words, boxes = page_boxes()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
    for ax, order, title in [(axes[0], naive_order(boxes), "(a) naive: top to bottom, left to right"),
                             (axes[1], column_order(boxes), "(b) split at the widest empty strip first")]:
        for i, (w, b) in enumerate(zip(words, boxes)):
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fc="#eef3f9", ec=BLUE, lw=1))
            ax.text((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w, ha="center", va="center", fontsize=9)
        centers = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
        column = (boxes[:, 0] > 300).astype(int)
        for a, b in zip(order[:-1], order[1:]):
            jump = column[a] != column[b]
            ax.add_patch(FancyArrowPatch(centers[a] + [0, 6], centers[b] + [0, 6], arrowstyle="-|>",
                                         mutation_scale=9, lw=1.2, color=RED if jump else GREEN,
                                         alpha=0.85, connectionstyle="arc3,rad=0.15"))
        for k, i in enumerate(order):
            ax.text(boxes[i, 0] + 2, boxes[i, 1] - 1, str(k + 1), fontsize=8, color=PURPLE, va="bottom")
        ax.set_xlim(20, 640)
        ax.set_ylim(225, 85)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        jumps = int(sum(column[a] != column[b] for a, b in zip(order[:-1], order[1:])))
        ax.text(330, 222, f"column changes along the reading path: {jumps}", ha="center", fontsize=9.5,
                color=RED if jumps > 1 else GREEN)
    fig.tight_layout()
    save(fig, "reading-order.png")


if __name__ == "__main__":
    plt.rcParams.update(STYLE)
    fig_pipeline()
    fig_detection()
    fig_quad()
    fig_db()
    fig_ctc()
    fig_cer()
    fig_glyphs()
    fig_order()
