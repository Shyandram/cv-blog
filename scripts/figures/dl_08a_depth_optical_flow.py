"""Figures for DLCV 08A (depth estimation and optical flow); synthetic data only, CPU, a few seconds.

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/dl_08a_depth_optical_flow.py
Writes PNGs to src/assets/deep-learning/dl-08a-depth-optical-flow/. The stereo scene is ray-cast from
analytic planes with a random texture; the flow pair is built from skimage.data images with a known,
analytic motion. Nothing is downloaded.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import ndimage as ndi
from skimage import color, data, transform
import cv2

OUT = Path(__file__).resolve().parents[2] / "src/assets/deep-learning/dl-08a-depth-optical-flow"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, RED, GREY = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------------------------
# The synthetic rectified stereo scene (same as the tutorial's block-matching snippet)
# ----------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
H, W, F, B = 180, 280, 320.0, 0.25
tex = ndi.gaussian_filter(rng.random((700, 700)), 1.2)
tex = (tex - tex.min()) / np.ptp(tex)


def render(cx):
    v, u = np.mgrid[0:H, 0:W].astype(float)
    dx, dy = (u - W / 2) / F, (v - H / 2) / F
    Z = (9 + 0.15 * cx) / (1 - 0.15 * dx)
    lab = np.zeros((H, W), int)
    for k, z, inside in [(2, 6.0, lambda X, Y: (X - 1.2) ** 2 + (Y - 0.3) ** 2 < 0.36),
                         (3, 5.0, lambda X, Y: np.abs(X - 0.2) < 0.04),
                         (1, 4.0, lambda X, Y: (np.abs(X + 1.0) < 0.6) & (np.abs(Y) < 0.6))]:
        hit = inside(cx + z * dx, z * dy) & (z < Z)
        Z, lab = np.where(hit, z, Z), np.where(hit, k, lab)
    X, Y = cx + Z * dx, Z * dy
    img = ndi.map_coordinates(tex, [350 + 40 * Y + 60 * lab, 350 + 40 * X], order=1)
    flat = (lab == 0) & (X > 2.2) & (X < 3.6) & (Y > -2.0) & (Y < -0.9)
    rep = (lab == 0) & (X > -3.6) & (X < -2.2) & (Y > 0.9) & (Y < 2.0)
    img[flat] = 0.5
    img[rep] = 0.5 + 0.4 * np.sin(2 * np.pi * X[rep] / 0.15)
    lab[flat], lab[rep] = 4, 5
    return img + rng.normal(0, 0.01, img.shape), Z, lab


left, Z_left, lab = render(0.0)
right, Z_right, _ = render(B)
d_true = F * B / Z_left
rows, cols = np.mgrid[0:H, 0:W]
xr = np.round(cols - d_true).astype(int)
occluded = (xr < 0) | (np.abs(Z_right[rows, np.clip(xr, 0, W - 1)] - Z_left) > 0.05)


def sad_volume(ref, other, D, win, sign):
    cost = np.full((D, H, W), np.inf)
    for d in range(D):
        c = ndi.uniform_filter(np.abs(ref - np.roll(other, sign * d, axis=1)), win) * win ** 2
        if sign > 0:
            c[:, :d] = np.inf
        else:
            c[:, W - d:] = np.inf
        cost[d] = c
    return cost


D, WIN = 32, 7
cost_L = sad_volume(left, right, D, WIN, +1)
d_left = cost_L.argmin(0)
d_right = sad_volume(right, left, D, WIN, -1).argmin(0)
lr_ok = np.abs(d_left - d_right[rows, np.clip(cols - d_left, 0, W - 1)]) <= 1
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=D, blockSize=5, P1=8 * 25, P2=32 * 25,
                             uniquenessRatio=0, disp12MaxDiff=-1)
u8 = lambda a: np.clip(a * 255, 0, 255).astype(np.uint8)
d_sgbm = sgbm.compute(u8(left), u8(right)).astype(float) / 16
valid = ~occluded & (cols >= D)
regions = {"textured": valid & np.isin(lab, [0, 1, 2]), "textureless": valid & (lab == 4),
           "stripes": valid & (lab == 5), "thin pole": valid & (lab == 3)}
bad = lambda d, m: 100 * np.mean(np.abs(d - d_true)[m] > 1)


# ----------------------------------------------------------------------------------------------
# Figure 1: rectified stereo geometry and depth uncertainty
# ----------------------------------------------------------------------------------------------
def fig_geometry():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) top view: two cameras at x = 0 and x = B, image planes at z = f, point P at depth Z
    f, Bs, Px, Pz = 1.0, 2.0, 0.7, 5.0
    for cxx, name in [(0.0, "left camera"), (Bs, "right camera")]:
        a.plot([cxx - 0.9, cxx + 0.9], [f, f], color="k", lw=2)
        a.plot([cxx], [0], "o", color="k", ms=6)
        a.text(cxx, -0.45, name, ha="center", fontsize=10)
        a.plot([cxx, cxx], [0, 5.6], ls=":", color=GREY, lw=1)
        xi = cxx + (Px - cxx) * f / Pz
        a.plot([cxx, Px], [0, Pz], color=BLUE, lw=1.2)
        a.plot([xi], [f], "o", color=ORANGE, ms=6)
    a.plot([Px], [Pz], "*", color=RED, ms=14)
    a.text(Px + 0.15, Pz + 0.1, "P", fontsize=11, color=RED)
    a.annotate("", xy=(Bs, -0.15), xytext=(0, -0.15), arrowprops=dict(arrowstyle="<->", color=GREEN))
    a.text(Bs / 2, -0.05, "baseline B", ha="center", va="bottom", color=GREEN)
    a.annotate("", xy=(-1.25, Pz), xytext=(-1.25, 0), arrowprops=dict(arrowstyle="<->", color=GREY))
    a.text(-1.35, Pz / 2, "depth Z", rotation=90, ha="right", va="center", color=GREY)
    a.annotate("", xy=(-1.0, f), xytext=(-1.0, 0), arrowprops=dict(arrowstyle="<->", color=GREY))
    a.text(-0.95, f / 2, "f", ha="left", va="center", color=GREY)
    xl, xr_ = Px * f / Pz, (Px - Bs) * f / Pz
    a.text(0.0 + xl, f + 0.25, "$x_L$", ha="center", color=ORANGE)
    a.text(Bs + xr_, f + 0.25, "$x_R$", ha="center", color=ORANGE)
    a.text(1.0, 6.0, "disparity  $d = x_L - x_R = fB/Z$", ha="center", fontsize=11)
    a.set_xlim(-1.9, 3.2); a.set_ylim(-0.7, 6.4); a.set_aspect("equal"); a.axis("off")
    a.set_title("(a) Rectified pair, seen from above", fontsize=11)
    # (b) depth error for a 0.25 px disparity error, f = 700 px
    Z = np.linspace(1, 60, 300)
    for Bb, col in [(0.12, ORANGE), (0.5, BLUE)]:
        b.plot(Z, Z ** 2 * 0.25 / (700 * Bb), color=col, lw=2, label=f"B = {Bb} m")
    b.set_yscale("log"); b.set_xlabel("depth Z (m)"); b.set_ylabel("depth error (m)")
    b.set_title("(b) Error for a 0.25 px disparity error (f = 700 px)", fontsize=11)
    b.grid(True, which="both", alpha=0.3); b.legend(frameon=False)
    fig.tight_layout()
    save(fig, "stereo-geometry.png")


# ----------------------------------------------------------------------------------------------
# Figure 2: four families of dense correspondence models
# ----------------------------------------------------------------------------------------------
def fig_families():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.6); ax.axis("off")
    rows_ = [
        ("Classical stereo", ["matching cost\n(SAD, census)", "aggregation\n(window, SGM)",
                              "winner-take-all", "refinement\n(L-R check, sub-pixel)"], GREY),
        ("Cost-volume network", ["shared CNN\nfeatures", "cost volume\n(concat / correlation)",
                                 "3D CNN\nregularization", "soft argmin\n-> disparity"], BLUE),
        ("Iterative refinement", ["features +\ncontext", "all-pairs\ncorrelation", "look up around\ncurrent estimate",
                                  "GRU update\n(repeat N times)"], GREEN),
        ("Monocular network", ["one image", "encoder\n(CNN / ViT)", "dense decoder",
                               "relative or metric\ndepth"], ORANGE),
    ]
    for i, (name, boxes, col) in enumerate(rows_):
        y = 3.85 - i * 1.1
        ax.text(0.05, y, name, fontsize=10.5, va="center", weight="bold", color=col)
        for j, t in enumerate(boxes):
            x = 2.25 + j * 2.07
            ax.add_patch(FancyBboxPatch((x, y - 0.38), 1.75, 0.76, boxstyle="round,pad=0.02",
                                        fc="white", ec=col, lw=1.6))
            ax.text(x + 0.875, y, t, ha="center", va="center", fontsize=9.5)
            if j < 3:
                ax.annotate("", xy=(x + 2.07, y), xytext=(x + 1.77, y),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
    ax.annotate("", xy=(8.48, 1.9), xytext=(9.6, 1.9), arrowprops=dict(arrowstyle="->", color=GREEN,
                connectionstyle="arc3,rad=0.9", lw=1.2))
    fig.tight_layout()
    save(fig, "families.png")


# ----------------------------------------------------------------------------------------------
# Figure 3: the cost volume and four cost curves
# ----------------------------------------------------------------------------------------------
def fig_costs():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    r = 90
    sl = np.where(np.isfinite(cost_L[:, r, :]), cost_L[:, r, :], np.nan)
    im = a.imshow(sl, aspect="auto", origin="lower", cmap="viridis",
                  vmax=np.nanpercentile(sl, 95))
    a.plot(np.arange(W), d_true[r], color="w", lw=1.2, ls="--", label="true disparity")
    a.plot(np.arange(W), d_left[r], ".", color=RED, ms=2.5, label="lowest cost (WTA)")
    a.set_xlabel("column x (row 90)"); a.set_ylabel("disparity hypothesis d (px)")
    a.set_title("(a) One row of the SAD cost volume", fontsize=11)
    a.legend(loc="upper right", fontsize=9, framealpha=0.8)
    fig.colorbar(im, ax=a, fraction=0.04, pad=0.02, label="cost")
    pts = {"textured wall": None, "textureless patch": None, "stripes": None, "occluded": None}
    pick = lambda m: tuple(np.argwhere(m)[len(np.argwhere(m)) // 2])
    pts["textured wall"] = pick(valid & (lab == 0) & (rows > 120) & (cols > 120) & (cols < 160))
    pts["textureless patch"] = pick(valid & (lab == 4))
    pts["stripes"] = pick(valid & (lab == 5))
    pts["occluded"] = pick(occluded & (cols > 60) & (lab == 0))
    for (name, (y, x)), col in zip(pts.items(), [BLUE, GREY, ORANGE, RED]):
        c = cost_L[:, y, x]
        b.plot(np.arange(D), c / np.nanmax(c[np.isfinite(c)]), color=col, lw=1.8, label=name)
        b.axvline(d_true[y, x], color=col, ls=":", lw=1)
    b.set_xlabel("disparity hypothesis d (px)"); b.set_ylabel("normalized SAD cost")
    b.set_title("(b) Cost curves of four pixels", fontsize=11)
    b.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    b.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "cost-curves.png")


# ----------------------------------------------------------------------------------------------
# Figure 4: block matching, left-right check and SGBM against the ground truth
# ----------------------------------------------------------------------------------------------
def fig_block_matching():
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.6))
    vmin, vmax = 6, 21
    axs[0, 0].imshow(left, cmap="gray", vmin=0, vmax=1)
    for k, col, name in [(4, ORANGE, "textureless"), (5, GREEN, "stripes"), (3, RED, "thin pole")]:
        axs[0, 0].contour(lab == k, levels=[0.5], colors=col, linewidths=1.2)
    axs[0, 0].set_title("(a) Left image (regions outlined)", fontsize=10)
    axs[0, 1].imshow(d_true, cmap="magma", vmin=vmin, vmax=vmax)
    axs[0, 1].set_title("(b) True disparity", fontsize=10)
    axs[0, 2].imshow(np.where(cols >= D, d_left, np.nan), cmap="magma", vmin=vmin, vmax=vmax)
    axs[0, 2].set_title("(c) SAD 7x7 + winner-take-all", fontsize=10)
    shown = np.where((cols >= D) & lr_ok, d_left, np.nan)
    cm = plt.get_cmap("magma").copy(); cm.set_bad("#9ae6b4")
    axs[1, 0].imshow(shown, cmap=cm, vmin=vmin, vmax=vmax)
    axs[1, 0].add_patch(Rectangle((-0.5, -0.5), D, H, fc="white", ec="none"))   # strip not evaluated
    axs[1, 0].set_title("(d) After left-right check (green: rejected)", fontsize=10)
    im = axs[1, 1].imshow(np.where(cols >= D, d_sgbm, np.nan), cmap="magma", vmin=vmin, vmax=vmax)
    axs[1, 1].set_title("(e) OpenCV SGBM", fontsize=10)
    for a in axs.flat[:5]:
        a.axis("off")
    fig.colorbar(im, ax=axs[1, 1], fraction=0.045, pad=0.02, label="disparity (px)")
    a = axs[1, 2]
    names = list(regions)
    x = np.arange(len(names))
    a.bar(x - 0.2, [bad(d_left, regions[n]) for n in names], 0.4, color=BLUE, label="SAD + WTA")
    a.bar(x + 0.2, [bad(d_sgbm, regions[n]) for n in names], 0.4, color=ORANGE, label="SGBM")
    a.set_xticks(x); a.set_xticklabels(names, rotation=20); a.set_ylabel("bad-1 (%)")
    a.set_title("(f) Error rate by region", fontsize=10); a.legend(frameon=False, fontsize=9)
    a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "block-matching.png")


# ----------------------------------------------------------------------------------------------
# Figure 5: scale ambiguity and scale-and-shift alignment
# ----------------------------------------------------------------------------------------------
def fig_scale():
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    a = axs[0]
    a.plot([0], [0], "o", color="k", ms=6); a.text(0, -0.5, "camera", ha="center")
    for s, col, lbl in [(1.0, BLUE, "small box, near"), (2.5, ORANGE, "box 2.5x larger, 2.5x farther")]:
        zc, hh = 2.0 * s, 0.5 * s
        a.add_patch(Rectangle((-hh, zc), 2 * hh, 0.3 * s, fc=col, alpha=0.35, ec=col, lw=1.5))
        for xs_ in [-hh, hh]:
            a.plot([0, xs_], [0, zc], color=col, lw=1, ls="--")
        a.text(hh + 0.15, zc + 0.15 * s, lbl, color=col, fontsize=9, va="center")
    a.plot([-0.5, 0.5], [1.0, 1.0], color="k", lw=2); a.text(-0.55, 1.0, "image", ha="right", va="center")
    a.set_xlim(-2.2, 4.2); a.set_ylim(-0.8, 6.0); a.set_aspect("equal"); a.axis("off")
    a.set_title("(a) Same image, different scenes", fontsize=10.5)
    # (b) photometric loss as a function of a global depth scale s
    valid_b = valid & ~np.isin(lab, [4, 5])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ss = np.linspace(0.5, 2.0, 61)
    known, mono = [], []
    for s in ss:
        disp = F * B / (s * Z_left)                         # known baseline B, depth guess s * Z
        warped = cv2.remap(right.astype(np.float32), xx - disp.astype(np.float32), yy, cv2.INTER_LINEAR)
        known.append(np.abs(warped - left)[valid_b].mean())
        disp_m = F * (s * B) / (s * Z_left)                 # pose network also scales the translation
        warped = cv2.remap(right.astype(np.float32), xx - disp_m.astype(np.float32), yy, cv2.INTER_LINEAR)
        mono.append(np.abs(warped - left)[valid_b].mean())
    axs[1].plot(ss, known, color=BLUE, lw=2, label="stereo pair, baseline known")
    axs[1].plot(ss, mono, color=ORANGE, lw=2, ls="--", label="video, translation also learned")
    axs[1].axvline(1.0, color=GREY, ls=":")
    axs[1].set_xlabel("global depth scale s"); axs[1].set_ylabel("photometric L1 error")
    axs[1].set_title("(b) Self-supervised loss vs depth scale", fontsize=10.5)
    axs[1].legend(frameon=False, fontsize=8.5, loc="upper right"); axs[1].grid(alpha=0.3)
    # (c) affine-invariant prediction in inverse depth, before and after least-squares alignment
    r2 = np.random.default_rng(1)
    Hs, Ws, fs, cam_h = 120, 160, 150.0, 1.5
    v, u = np.mgrid[0:Hs, 0:Ws].astype(float)
    Z = np.full((Hs, Ws), 30.0)
    below = v > Hs / 2 + 1
    Z[below] = np.minimum(30.0, fs * cam_h / (v[below] - Hs / 2))
    Z[(np.abs(u - 50) < 15) & (v > 45) & (v < 80)] = 5.0
    pred = 0.37 / Z + 0.05 + r2.normal(0, 0.001, Z.shape)
    A = np.c_[pred.ravel(), np.ones(pred.size)]
    s_, t_ = np.linalg.lstsq(A, (1 / Z).ravel(), rcond=None)[0]
    idx = r2.choice(Z.size, 1500, replace=False)
    axs[2].plot((1 / Z).ravel()[idx], pred.ravel()[idx], ".", ms=3, color=GREY, label="raw prediction")
    axs[2].plot((1 / Z).ravel()[idx], (s_ * pred + t_).ravel()[idx], ".", ms=3, color=GREEN,
                label="after scale + shift")
    lim = [0, 0.22]
    axs[2].plot(lim, lim, color="k", lw=1, ls=":")
    axs[2].set_xlim(lim); axs[2].set_ylim(lim)
    axs[2].set_xlabel("true inverse depth 1/Z (1/m)"); axs[2].set_ylabel("predicted inverse depth")
    axs[2].set_title("(c) Least-squares alignment", fontsize=10.5)
    axs[2].legend(frameon=False, fontsize=8.5, loc="upper left", markerscale=3); axs[2].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "scale-ambiguity.png")


# ----------------------------------------------------------------------------------------------
# Figure 6: optical flow, Farneback estimate and its errors (same scene as the flow snippet)
# ----------------------------------------------------------------------------------------------
def flow_to_rgb(fl, maxmag):
    mag = np.linalg.norm(fl, axis=-1)
    ang = (np.arctan2(fl[..., 1], fl[..., 0]) + np.pi) / (2 * np.pi)
    hsv = np.stack([ang, np.clip(mag / maxmag, 0, 1), np.ones_like(mag)], -1)
    return hsv_to_rgb(hsv)


def fig_flow():
    I0 = transform.resize(data.camera() / 255.0, (240, 320), anti_aliasing=True)
    Hf, Wf = I0.shape
    rr, cc = np.mgrid[0:Hf, 0:Wf].astype(float)
    cy, cx = Hf / 2, Wf / 2
    a_, s_ = np.deg2rad(2.0), 1.03
    bx = cx + s_ * (np.cos(a_) * (cc - cx) - np.sin(a_) * (rr - cy))
    by = cy + s_ * (np.sin(a_) * (cc - cx) + np.cos(a_) * (rr - cy))
    flow = np.stack([bx - cc, by - rr], -1)
    patch = transform.resize(color.rgb2gray(data.astronaut())[30:230, 150:350], (50, 50), anti_aliasing=True)
    obj0 = (rr >= 90) & (rr < 140) & (cc >= 120) & (cc < 170)
    flow[obj0] = (14.0, 6.0)
    xs = cx + (np.cos(a_) * (cc - cx) + np.sin(a_) * (rr - cy)) / s_
    ys = cy + (-np.sin(a_) * (cc - cx) + np.cos(a_) * (rr - cy)) / s_
    I1 = ndi.map_coordinates(I0, [ys, xs], order=3, mode="reflect")
    I0 = I0.copy(); I0[90:140, 120:170] = patch
    I1[96:146, 134:184] = patch
    occ = ~obj0 & (by >= 96) & (by < 146) & (bx >= 134) & (bx < 184)
    q8 = lambda im: (np.clip(im, 0, 1) * 255).round().astype(np.uint8)
    est = cv2.calcOpticalFlowFarneback(q8(I0), q8(I1), None, pyr_scale=0.5, levels=4, winsize=15,
                                       iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
    epe = np.linalg.norm(est - flow, axis=-1)
    mag = np.linalg.norm(flow, axis=-1)
    outl = (epe > 3) & (epe > 0.05 * mag)
    inner = np.zeros((Hf, Wf), bool); inner[10:-10, 10:-10] = True     # the snippet ignores a 10 px border
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.4))
    axs[0, 0].imshow(I0, cmap="gray"); axs[0, 0].set_title("(a) Frame 0", fontsize=10)
    axs[0, 1].imshow(I1, cmap="gray"); axs[0, 1].set_title("(b) Frame 1", fontsize=10)
    axs[0, 2].imshow(flow_to_rgb(flow, 16)); axs[0, 2].set_title("(c) True flow", fontsize=10)
    # small colour-wheel key in the corner of (c)
    k = 41
    ky, kx = np.mgrid[-1:1:k * 1j, -1:1:k * 1j]
    wheel = flow_to_rgb(np.stack([kx, ky], -1) * 16, 16)
    wheel[np.hypot(kx, ky) > 1] = 1
    axs[0, 2].imshow(wheel, extent=[Wf - 58, Wf - 6, Hf - 6, Hf - 58])
    axs[0, 2].set_xlim(0, Wf); axs[0, 2].set_ylim(Hf, 0)
    axs[1, 0].imshow(flow_to_rgb(est, 16)); axs[1, 0].set_title("(d) Farneback estimate", fontsize=10)
    im = axs[1, 1].imshow(np.where(inner, epe, np.nan), cmap="inferno", vmin=0, vmax=10)
    axs[1, 1].contour(occ, levels=[0.5], colors="cyan", linewidths=1.0)
    axs[1, 1].set_title("(e) End-point error (cyan: occluded)", fontsize=10)
    fig.colorbar(im, ax=axs[1, 1], fraction=0.04, pad=0.02, label="EPE (px)")
    rgb = np.where(outl[..., None], np.array([0.77, 0.19, 0.19]), 0.85 * np.ones(3))
    rgb[~inner] = 1.0
    axs[1, 2].imshow(rgb)
    axs[1, 2].set_title(f"(f) Outliers, > 3 px and > 5% ({100 * outl[inner].mean():.0f}%)", fontsize=10)
    for a in axs.flat:
        a.axis("off")
    fig.tight_layout()
    save(fig, "flow-error.png")


if __name__ == "__main__":
    fig_geometry()
    fig_families()
    fig_costs()
    fig_block_matching()
    fig_scale()
    fig_flow()
