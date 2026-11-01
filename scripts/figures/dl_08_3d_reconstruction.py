"""Figures for DLCV 08 (3D reconstruction); synthetic data only, CPU, about 15 seconds.

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/dl_08_3d_reconstruction.py
Writes PNGs to src/assets/deep-learning/dl-08-3d-reconstruction/. Every scene is an analytic
shape (a torus, planes, ellipses) or random texture; nothing is downloaded.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.draw import disk, ellipse, polygon
from skimage.measure import marching_cubes
import cv2

OUT = Path(__file__).resolve().parents[2] / "src/assets/deep-learning/dl-08-3d-reconstruction"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white"})

R_T, r_T = 0.6, 0.25  # torus ring and tube radii


def sdf_torus(x, y, z):
    return np.hypot(np.hypot(x, y) - R_T, z) - r_T


def torus_mesh(n):
    ax = np.linspace(-1, 1, n)
    vol = sdf_torus(*np.meshgrid(ax, ax, ax, indexing="ij"))
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=(ax[1] - ax[0],) * 3)
    return v - 1.0, f


def look_at(C, target, up=(0.0, 0.0, 1.0)):
    z = (target - C) / np.linalg.norm(target - C)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def sphere_trace(C, R, f, H, W):
    """Ray-cast the torus SDF: returns depth along the optical axis and the 3D hit points."""
    v, u = np.mgrid[0:H, 0:W].astype(float)
    d_cam = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u)], -1)
    d = d_cam @ R  # camera -> world directions
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    t = np.zeros((H, W))
    for _ in range(80):
        p = C + t[..., None] * d
        t += sdf_torus(*np.moveaxis(p, -1, 0))
    p = C + t[..., None] * d
    hit = np.abs(sdf_torus(*np.moveaxis(p, -1, 0))) < 1e-3
    depth = np.where(hit, ((p - C) @ R[2]), np.nan)
    return depth, np.where(hit[..., None], p, np.nan), hit


def fig_representations():
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(10.5, 5.4))
    C = np.array([0.0, -2.2, 1.4]); Rc = look_at(C, np.zeros(3))
    depth, pts, hit = sphere_trace(C, Rc, 90.0, 110, 140)

    # (a) point cloud
    ax = fig.add_subplot(2, 4, 1, projection="3d")
    u, v = rng.uniform(0, 2 * np.pi, (2, 1500))
    P = np.c_[(R_T + r_T * np.cos(v)) * np.cos(u), (R_T + r_T * np.cos(v)) * np.sin(u), r_T * np.sin(v)]
    ax.scatter(*P.T, s=1.5, c=P[:, 2], cmap="viridis")
    ax.set_title("(a) Point cloud")
    # (b) depth map
    ax = fig.add_subplot(2, 4, 2)
    im = ax.imshow(depth, cmap="magma_r"); ax.set_title("(b) Depth map")
    fig.colorbar(im, ax=ax, fraction=0.04, label="depth")
    # (c) voxels
    ax = fig.add_subplot(2, 4, 3, projection="3d")
    n = 14; g = (np.arange(n) + 0.5) / n * 2 - 1
    occ = sdf_torus(*np.meshgrid(g, g, g, indexing="ij")) < 0
    edges = np.linspace(-1, 1, n + 1)
    ax.voxels(*np.meshgrid(edges, edges, edges, indexing="ij"), occ, facecolors="#7fa7d9",
              edgecolor="#3b5b8c", linewidth=0.2)
    ax.set_title("(c) Voxel grid (14³)")
    # (d) mesh
    ax = fig.add_subplot(2, 4, 4, projection="3d")
    vts, fcs = torus_mesh(16)
    ax.add_collection3d(Poly3DCollection(vts[fcs], facecolor="#c9d6e8", edgecolor="#40506a", linewidth=0.3))
    ax.set_title("(d) Triangle mesh")
    # (e) SDF slice
    ax = fig.add_subplot(2, 4, 5)
    xs = np.linspace(-1, 1, 200); Xg, Zg = np.meshgrid(xs, xs)
    sd = sdf_torus(Xg, 0 * Xg, Zg)
    im = ax.imshow(sd, extent=(-1, 1, -1, 1), origin="lower", cmap="RdBu", vmin=-0.6, vmax=0.6)
    ax.contour(Xg, Zg, sd, levels=[0], colors="k", linewidths=1.2)
    ax.set_title("(e) Signed distance (slice y=0)"); ax.set_xlabel("x"); ax.set_ylabel("z")
    fig.colorbar(im, ax=ax, fraction=0.04)
    # (f) radiance field: density slice and one ray with sample points
    ax = fig.add_subplot(2, 4, 6)
    sigma = 30 / (1 + np.exp(40 * sd))
    ax.imshow(sigma, extent=(-1, 1, -1, 1), origin="lower", cmap="Greys")
    tt = np.linspace(0, 2.2, 23); o = np.array([-1.0, -0.85]); dvec = np.array([0.85, 0.5]); dvec /= np.linalg.norm(dvec)
    q = o + tt[:, None] * dvec
    dens = 30 / (1 + np.exp(40 * sdf_torus(q[:, 0], 0 * q[:, 0], q[:, 1])))
    ax.plot(q[:, 0], q[:, 1], "-", color="tab:orange", lw=1)
    ax.scatter(q[:, 0], q[:, 1], s=10 + dens, color="tab:orange", zorder=3)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_title("(f) Radiance field: density\n+ samples on one ray"); ax.set_xlabel("x"); ax.set_ylabel("z")
    # (g) 3D Gaussians, top view
    ax = fig.add_subplot(2, 4, 7)
    for k in range(110):
        a = 2 * np.pi * k / 110 + rng.normal(0, 0.03)
        rad = R_T + rng.choice([-1, 1]) * r_T * rng.uniform(0.6, 1.0)
        e = Ellipse((rad * np.cos(a), rad * np.sin(a)), width=0.06, height=0.16, angle=np.degrees(a),
                    facecolor=plt.cm.viridis(k / 110), alpha=0.55, edgecolor="none")
        ax.add_patch(e)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    ax.set_title("(g) 3D Gaussians (top view)"); ax.set_xlabel("x"); ax.set_ylabel("y")
    # (h) pointmap: per-pixel XYZ shown as RGB
    ax = fig.add_subplot(2, 4, 8)
    rgb = np.clip((np.nan_to_num(pts, nan=0.0) + 1) / 2, 0, 1); rgb[~hit] = 1.0
    ax.imshow(rgb); ax.set_title("(h) Pointmap\n(XYZ shown as RGB)")
    for a in fig.axes:
        if hasattr(a, "zaxis"):
            a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_zlim(-1, 1); a.set_box_aspect((1, 1, 0.6), zoom=1.6)
            a.set_xticks([]); a.set_yticks([]); a.set_zticks([]); a.view_init(28, -60)
            a.set_proj_type("ortho"); a.set_axis_off()
        elif a.get_label() != "<colorbar>" and a.images and a.get_title().startswith(("(b)", "(h)")):
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "representations.png", dpi=150)
    plt.close(fig)


def two_view_setup():
    rng = np.random.default_rng(0)
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])

    def look(C, target):
        z = (target - C) / np.linalg.norm(target - C)
        x = np.cross([0.0, 1.0, 0.0], z); x /= np.linalg.norm(x)
        return np.stack([x, np.cross(z, x), z])

    def project(R, t, X):
        x = (X @ R.T + t) @ K.T
        return x[:, :2] / x[:, 2:]

    X = rng.uniform([-3, -2, 4], [3, 2, 8], size=(300, 3))
    C2 = np.array([2.0, 0.0, 0.0]); R2 = look(C2, np.array([0.0, 0.0, 6.0])); t2 = -R2 @ C2
    x1 = project(np.eye(3), np.zeros(3), X) + rng.normal(0, 0.5, (300, 2))
    x2 = project(R2, t2, X) + rng.normal(0, 0.5, (300, 2))
    x2[:45] = rng.uniform([0, 0], [640, 480], (45, 2))
    return K, x1, x2


def fig_two_view():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    C1, C2 = np.array([0.0, 0.0]), np.array([4.4, 0.0])
    planes = [np.array([[-0.9, 1.0], [0.9, 1.25], [0.9, 2.05], [-0.9, 1.8]]),
              np.array([[2.2, 1.3], [4.1, 1.0], [4.1, 1.8], [2.2, 2.1]])]
    for pl in planes:
        ax.add_patch(Polygon(pl, closed=True, facecolor="#e8eef7", edgecolor="#40506a"))

    def hit(C, P, pl):  # where the segment C->P crosses the plane's mid-line (enough for a drawing)
        a, b = (pl[0] + pl[3]) / 2, (pl[1] + pl[2]) / 2
        s = np.linalg.solve(np.array([P - C, a - b]).T, a - C)[0]
        return C + s * (P - C)
    x1 = np.array([0.45, 1.55])
    Xp, Xq = C1 + 2.4 * (x1 - C1), C1 + 1.75 * (x1 - C1)        # two points on the same ray of camera 1
    x2, x2q = hit(C2, Xp, planes[1]), hit(C2, Xq, planes[1])
    ax.plot(*np.c_[C1, Xp], color="tab:blue")
    ax.plot(*np.c_[C2, Xp], color="tab:blue"); ax.plot(*np.c_[C2, Xq], ":", color="tab:purple")
    ax.plot(*np.c_[C1, C2], "--", color="0.4"); ax.text(2.2, -0.3, "baseline", ha="center", color="0.3")
    ax.plot(*Xp, "o", color="tab:red"); ax.text(Xp[0] + 0.12, Xp[1], "X", color="tab:red")
    ax.plot(*Xq, "o", color="tab:purple"); ax.text(Xq[0] - 0.42, Xq[1] + 0.05, "X'", color="tab:purple")
    d = (x2q - x2) / np.linalg.norm(x2q - x2)
    ax.plot(*np.c_[x2 - 0.35 * d, x2q + 0.45 * d], color="tab:green", lw=2.5)
    ax.plot(*x1, "o", color="k", ms=5); ax.text(x1[0] - 0.42, x1[1] + 0.08, "x")
    ax.plot(*x2, "o", color="tab:red", ms=5); ax.plot(*x2q, "o", color="tab:purple", ms=5)
    ax.text(3.25, 2.5, "epipolar line of x:\nX or X' must land on it", color="tab:green", fontsize=10)
    for C, lab in [(C1, "camera 1\ncentre"), (C2, "camera 2\ncentre")]:
        ax.plot(*C, "ks"); ax.text(C[0], C[1] - 0.62, lab, ha="center")
    ax.text(-0.85, 2.2, "image 1"); ax.text(2.45, 2.25, "image 2")
    ax.set_xlim(-1.3, 6.0); ax.set_ylim(-1.1, 4.1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) Two views of one point: the epipolar constraint")

    K, x1p, x2p = two_view_setup()
    E, mask = cv2.findEssentialMat(x1p, x2p, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    inl = mask.ravel() > 0
    ax = axs[1]
    ax.set_xlim(0, 640); ax.set_ylim(480, 0)
    sel = np.where(inl)[0][:12]
    xs = np.array([0, 640])
    for i in sel:
        l = F @ np.r_[x1p[i], 1]
        ax.plot(xs, -(l[0] * xs + l[2]) / l[1], color="tab:green", lw=0.8, alpha=0.8)
    ax.plot(*x2p[inl].T, ".", color="tab:blue", ms=3, label="inlier matches")
    ax.plot(*x2p[~inl].T, "x", color="tab:red", ms=5, label="rejected by RANSAC")
    ax.plot(*x2p[sel].T, "o", mfc="none", color="k", ms=6)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("(b) Image 2 of the code example:\nepipolar lines of 12 points from image 1")
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    fig.tight_layout()
    fig.savefig(OUT / "two-view.png", dpi=150)
    plt.close(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 23); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", fc=fc, ec="#40506a"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x0, y0, x1, y1, style="-|>", color="#40506a", ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.4, linestyle=ls))
    box(0.2, 4.2, 2.4, 1.6, "Images", "#f2f2f2")
    box(3.4, 4.2, 3.6, 1.6, "Features,\nmatching,\nRANSAC", "#e3ecf7")
    box(7.8, 4.2, 3.6, 1.6, "Incremental SfM\n+ bundle\nadjustment", "#e3ecf7")
    box(12.2, 4.2, 2.8, 1.6, "Cameras +\nsparse points", "#fff2cc")
    box(16.0, 7.4, 2.6, 1.4, "MVS depth\nmaps, fusion", "#e2f0d9")
    box(16.0, 4.3, 2.6, 1.4, "NeRF / NeuS\n(neural field)", "#e2f0d9")
    box(16.0, 1.2, 2.6, 1.4, "3D Gaussian\nSplatting", "#e2f0d9")
    box(21.0, 4.3, 1.8, 1.4, "Mesh", "#fde2e2")
    arrow(2.7, 5.0, 3.3, 5.0); arrow(7.1, 5.0, 7.7, 5.0); arrow(11.5, 5.0, 12.1, 5.0)
    for y in (8.1, 5.0, 1.9):
        arrow(15.1, 5.0, 15.9, y)
    arrow(18.75, 8.1, 21.4, 5.8); arrow(18.75, 5.0, 20.9, 5.0); arrow(18.75, 1.9, 21.4, 4.2)
    ax.text(20.3, 7.4, "Poisson /\nTSDF fusion", fontsize=10, ha="left")
    ax.text(19.8, 5.15, "marching\ncubes", fontsize=10, ha="center", va="bottom")
    ax.text(20.3, 2.1, "SuGaR /\n2DGS", fontsize=10, ha="left")
    ax.annotate("", xy=(13.6, 4.15), xytext=(1.4, 4.15),
                arrowprops=dict(arrowstyle="-|>", color="tab:purple", lw=1.6, ls="--",
                                connectionstyle="arc3,rad=0.35"))
    ax.text(7.5, 1.0, "feed-forward models (DUSt3R, MASt3R, VGGT): images -> cameras + dense points\n"
            "in one network pass", color="tab:purple", ha="center", fontsize=10)
    ax.text(5.3, 6.2, "classical geometry", ha="center", color="#40506a", fontsize=10)
    ax.text(17.3, 9.3, "dense reconstruction", ha="center", color="#40506a", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", dpi=150)
    plt.close(fig)


def plane_sweep():
    rng = np.random.default_rng(0)
    H, W, f, B = 120, 160, 150.0, 0.5
    tex = ndi.gaussian_filter(rng.random((400, 400)), 1.5)
    tex = (tex - tex.min()) / np.ptp(tex)

    def render(cx):
        v, u = np.mgrid[0:H, 0:W].astype(float)
        dx, dy = (u - W / 2) / f, (v - H / 2) / f
        Xs, Ys = cx + 5 * dx, 5 * dy
        front = (np.abs(Xs) < 1) & (np.abs(Ys) < 1)
        Z = np.where(front, 5.0, 8.0)
        X, Y = cx + Z * dx, Z * dy
        img = ndi.map_coordinates(tex, [200 + 40 * Y + 100 * front, 200 + 40 * X], order=1)
        img[~front & (X > 1.5) & (Y < -0.5)] = 0.5
        return img, Z
    ref, Zt = render(0.0); src, _ = render(B)
    depths = 1 / np.linspace(1 / 10, 1 / 3.5, 80)
    cost = np.empty((len(depths), H, W))
    for k, Z in enumerate(depths):
        warped = ndi.shift(src, (0, f * B / Z), order=1, mode="nearest")
        cost[k] = ndi.uniform_filter((ref - warped) ** 2, 7)
    return ref, src, Zt, depths, cost


def fig_plane_sweep():
    ref, src, Zt, depths, cost = plane_sweep()
    Ze = depths[cost.argmin(0)]
    fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.9), gridspec_kw={"width_ratios": [1, 1, 1.2, 1.25]})
    axs[0].imshow(ref, cmap="gray"); axs[0].set_title("(a) Reference image")
    p_tex, p_flat = (60, 80), (20, 140)
    for (r, c), col in [(p_tex, "tab:orange"), (p_flat, "tab:cyan")]:
        axs[0].plot(c, r, "o", mfc="none", mec=col, ms=8, mew=2)
    for a, img, title in [(axs[1], Zt, "(b) True depth"), (axs[2], Ze, "(c) Estimated depth")]:
        im = a.imshow(img, cmap="viridis", vmin=3.5, vmax=10); a.set_title(title)
    for a in axs[:3]:
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=axs[2], fraction=0.045)
    for (r, c), col, lab in [(p_tex, "tab:orange", "textured pixel"), (p_flat, "tab:cyan", "textureless pixel")]:
        axs[3].plot(depths, cost[:, r, c], color=col, label=lab)
        axs[3].axvline(Zt[r, c], color=col, ls=":")
    axs[3].set_xlabel("depth hypothesis Z"); axs[3].set_ylabel("matching cost (SSD)")
    axs[3].set_title("(d) Cost along the sweep"); axs[3].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "plane-sweep.png", dpi=150)
    plt.close(fig)


def fig_ray_weights():
    slabs = [(1.0, 2.0, 0.5), (3.0, 3.5, 4.0)]
    edges = np.linspace(0, 6, 257); t = (edges[:-1] + edges[1:]) / 2; delta = np.diff(edges)
    sigma = np.zeros_like(t)
    for a, b, s in slabs:
        sigma[(t >= a) & (t < b)] = s
    alpha = 1 - np.exp(-sigma * delta)
    T = np.concatenate([[1.0], np.cumprod(1 - alpha)[:-1]])
    w = T * alpha
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
    axs[0].step(t, sigma, where="mid", color="k"); axs[0].set_title("(a) Density σ(t)")
    axs[0].axvspan(1, 2, color="#6680ff", alpha=0.25, label="blue fog"); axs[0].axvspan(3, 3.5, color="#ff4d4d", alpha=0.25, label="red surface")
    axs[0].legend(fontsize=10)
    axs[1].plot(t, T, color="tab:green"); axs[1].set_ylim(0, 1.05); axs[1].set_title("(b) Transmittance T(t)")
    axs[2].bar(t, w / delta, width=delta, color="tab:purple"); axs[2].set_title("(c) Weight density w(t)")
    d = np.sum(w * t)
    axs[2].axvline(d, color="k", ls="--"); axs[2].text(d + 0.1, axs[2].get_ylim()[1] * 0.8, f"depth Σ w t = {d:.2f}")
    for a in axs:
        a.set_xlabel("distance t along the ray")
    fig.tight_layout()
    fig.savefig(OUT / "ray-weights.png", dpi=150)
    plt.close(fig)


def splat_experiment():
    H = W = 48
    target = np.zeros((H, W, 3))
    target[disk((16, 16), 9)] = [0.9, 0.2, 0.2]
    target[ellipse(30, 30, 6, 14, rotation=np.deg2rad(30))] = [0.2, 0.4, 0.9]
    target[polygon([40, 44, 34], [6, 22, 14])] = [0.2, 0.8, 0.3]
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    sig = lambda z: 1 / (1 + np.exp(-z))

    def render(p, tgt=None):
        c, s, th = sig(p["col"]), np.exp(p["log_s"]), p["theta"]
        dx = xs[None] - p["mu"][:, 0, None, None]; dy = ys[None] - p["mu"][:, 1, None, None]
        ct, st = np.cos(th)[:, None, None], np.sin(th)[:, None, None]
        u1, u2 = ct * dx + st * dy, -st * dx + ct * dy
        sx2, sy2 = (s[:, 0] ** 2)[:, None, None], (s[:, 1] ** 2)[:, None, None]
        g = np.exp(-0.5 * (u1 ** 2 / sx2 + u2 ** 2 / sy2))
        o = sig(p["op"])[:, None, None]
        a = np.minimum(o * g, 0.99)
        T = np.cumprod(np.concatenate([np.ones((1, H, W)), 1 - a[:-1]]), axis=0)
        w = a * T
        img = np.einsum("khw,kc->hwc", w, c)
        if tgt is None:
            return img
        G = 2 * (img - tgt) / tgt.size
        contrib = w[..., None] * c[:, None, None, :]
        behind = np.cumsum(contrib[::-1], axis=0)[::-1] - contrib
        dC_da = T[..., None] * c[:, None, None, :] - behind / (1 - a[..., None])
        da = np.einsum("khwc,hwc->khw", dC_da, G) * (o * g < 0.99)
        dq = da * o * g * -0.5
        grads = {
            "col": np.einsum("khw,hwc->kc", w, G) * c * (1 - c),
            "op": (da * g).sum((1, 2)) * sig(p["op"]) * (1 - sig(p["op"])),
            "log_s": np.stack([(dq * -2 * u1 ** 2 / sx2).sum((1, 2)), (dq * -2 * u2 ** 2 / sy2).sum((1, 2))], 1),
            "theta": (dq * 2 * u1 * u2 * (1 / sx2 - 1 / sy2)).sum((1, 2)),
            "mu": -np.stack([(dq * (2 * u1 * ct / sx2 - 2 * u2 * st / sy2)).sum((1, 2)),
                             (dq * (2 * u1 * st / sx2 + 2 * u2 * ct / sy2)).sum((1, 2))], 1),
        }
        return img, grads

    rng = np.random.default_rng(0)
    K = 40
    p = {"mu": rng.uniform(4, 44, (K, 2)), "log_s": np.full((K, 2), np.log(3.0)), "theta": rng.uniform(0, np.pi, K),
         "op": np.zeros(K), "col": np.zeros((K, 3))}
    lr = {"mu": 0.5, "log_s": 0.05, "theta": 0.05, "op": 0.1, "col": 0.1}
    m = {k: np.zeros_like(v) for k, v in p.items()}; v2 = {k: np.zeros_like(v) for k, v in p.items()}
    curve, snaps = [], {}
    for it in range(1, 501):
        img, gr = render(p, target)
        curve.append(10 * np.log10(1 / np.mean((img - target) ** 2)))
        if it in (1, 50):
            snaps[it] = img
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * gr[k]; v2[k] = 0.999 * v2[k] + 0.001 * gr[k] ** 2
            p[k] -= lr[k] * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v2[k] / (1 - 0.999 ** it)) + 1e-8)
    snaps[500] = img
    return target, snaps, curve, p, sig


def fig_splat():
    target, snaps, curve, p, sig = splat_experiment()
    fig = plt.figure(figsize=(10.5, 3.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.5])
    for i, (img, title) in enumerate([(target, "(a) Target"), (snaps[1], "(b) Start (iter. 1)"),
                                      (snaps[50], "(c) Iteration 50"), (snaps[500], "(d) Iteration 500")]):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(np.clip(img, 0, 1)); ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 3:
            s = np.exp(p["log_s"])
            for k in range(len(s)):
                ax.add_patch(Ellipse(p["mu"][k], 2 * s[k, 0], 2 * s[k, 1], angle=np.degrees(p["theta"][k]),
                                     fill=False, ec="w", lw=0.5, alpha=float(sig(p["op"][k]))))
    ax = fig.add_subplot(gs[0, 4])
    ax.plot(np.arange(1, len(curve) + 1), curve, color="tab:blue")
    ax.set_xlabel("iteration"); ax.set_ylabel("PSNR (dB)"); ax.set_title("(e) Fit quality"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "splat-toy.png", dpi=150)
    plt.close(fig)


def fig_marching_cubes():
    rng = np.random.default_rng(0)

    def sample_torus(n):
        u, v = rng.uniform(0, 2 * np.pi, (2, 3 * n))
        ok = rng.random(3 * n) * (R_T + r_T) < R_T + r_T * np.cos(v)
        u, v = u[ok][:n], v[ok][:n]
        return np.c_[(R_T + r_T * np.cos(v)) * np.cos(u), (R_T + r_T * np.cos(v)) * np.sin(u), r_T * np.sin(v)]

    def sample_mesh(verts, faces, n):
        tri = verts[faces]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        k = rng.choice(len(faces), n, p=area / area.sum())
        a, b = rng.random((2, n, 1)); flip = a + b > 1; a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
        return tri[k, 0] + a * (tri[k, 1] - tri[k, 0]) + b * (tri[k, 2] - tri[k, 0])

    G = sample_torus(100_000)
    grids, cds, fss = [8, 12, 16, 24, 32, 48, 64, 96], [], []
    for n in grids:
        v, fc = torus_mesh(n)
        P = sample_mesh(v, fc, 100_000)
        d1, d2 = cKDTree(G).query(P)[0], cKDTree(P).query(G)[0]
        cds.append((d1.mean() + d2.mean()) / 2)
        pr, rc = (d1 < 0.01).mean(), (d2 < 0.01).mean(); fss.append(2 * pr * rc / (pr + rc))
    d1, d2 = cKDTree(G).query(sample_torus(100_000))[0], cKDTree(sample_torus(100_000)).query(G)[0]
    floor = (d1.mean() + d2.mean()) / 2
    fig = plt.figure(figsize=(10.5, 3.4))
    for i, n in enumerate([12, 48]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        v, fc = torus_mesh(n)
        ax.add_collection3d(Poly3DCollection(v[fc], facecolor="#c9d6e8", edgecolor="#40506a", linewidth=0.15))
        ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_zlim(-0.5, 0.5); ax.set_box_aspect((1, 1, 0.55))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(35, -60); ax.set_axis_off()
        ax.set_title(f"({'ab'[i]}) Marching cubes, grid {n}³\n{len(fc)} triangles")
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(grids, cds, "o-", color="tab:blue", label="Chamfer distance")
    ax.axhline(floor, color="tab:blue", ls=":", label="sampling floor")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("grid resolution n (n³ samples)")
    ax.set_ylabel("Chamfer distance"); ax.set_xticks(grids); ax.set_xticklabels([str(g) for g in grids])
    ax2 = ax.twinx(); ax2.plot(grids, fss, "s--", color="tab:orange", label="F-score @ 0.01")
    ax2.set_ylabel("F-score @ 0.01"); ax2.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=10, loc="center right"); ax.set_title("(c) Error vs grid resolution")
    fig.tight_layout()
    fig.savefig(OUT / "marching-cubes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_representations()
    fig_two_view()
    fig_pipeline()
    fig_plane_sweep()
    fig_ray_weights()
    fig_splat()
    fig_marching_cubes()
    for p in sorted(OUT.glob("*.png")):
        print(p.name, f"{p.stat().st_size / 1024:.0f} KB")
