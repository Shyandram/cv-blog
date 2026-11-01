import numpy as np

def mosaic(images, boxes, size=416, rng=None, min_keep=0.25):
    """Place four images around a random split point (xc, yc) of a size x size canvas.
    images: four (H, W, 3) arrays; boxes: four (n, 4) arrays of (x1, y1, x2, y2) in pixels.
    Each image touches the split point with one corner; whatever falls outside its quadrant is cut.
    A box is kept if at least `min_keep` of its area survives the cut."""
    xc, yc = (int(v) for v in rng.uniform(0.3, 0.7, 2) * size)
    canvas = np.full((size, size, 3), 114, np.uint8)
    out, n_cut, n_drop = [], 0, 0
    for k, (img, bx) in enumerate(zip(images, boxes)):
        h, w = img.shape[:2]
        ox = xc - w if k in (0, 2) else xc          # where the image's top-left corner lands
        oy = yc - h if k in (0, 1) else yc
        rx0, rx1 = (0, xc) if k in (0, 2) else (xc, size)   # the quadrant it may fill
        ry0, ry1 = (0, yc) if k in (0, 1) else (yc, size)
        x0, x1, y0, y1 = max(rx0, ox), min(rx1, ox + w), max(ry0, oy), min(ry1, oy + h)
        canvas[y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        shifted = bx + [ox, oy, ox, oy]
        clipped = np.clip(shifted, [x0, y0, x0, y0], [x1, y1, x1, y1])
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = area(clipped) >= min_keep * area(shifted)
        n_cut += int((keep & (area(clipped) < area(shifted))).sum()); n_drop += int((~keep).sum())
        out.append(clipped[keep])
    return canvas, np.concatenate(out), (xc, yc), n_cut, n_drop

rng = np.random.default_rng(0)
imgs, bxs = [], []
for k in range(4):                                   # four synthetic 256 x 256 "photos"
    img = np.full((256, 256, 3), 60 * k + 40, np.uint8)
    b = []
    for _ in range(3):                               # three rectangles = three objects each
        x, y = rng.integers(0, 200, 2); w, h = rng.integers(30, 90, 2)
        x2, y2 = min(x + w, 256), min(y + h, 256)
        img[y:y2, x:x2] = rng.integers(0, 256, 3); b.append([x, y, x2, y2])
    imgs.append(img); bxs.append(np.array(b, float))

canvas, boxes, split, n_cut, n_drop = mosaic(imgs, bxs, rng=rng)
print("canvas", canvas.shape, "| split point", split)
print(f"objects in: {sum(len(b) for b in bxs)} | kept: {len(boxes)} (of which cut: {n_cut}) | dropped: {n_drop}")
