import numpy as np

sigmoid = lambda t: 1 / (1 + np.exp(-t))
logit = lambda p: np.log(p / (1 - p))

def decode(t, cell, anchor, stride, s=1.0):
    """One YOLOv3/YOLOv4 prediction t = (tx, ty, tw, th) -> box centre and size in pixels.
    s = 1 is the YOLOv3 formula; s > 1 is YOLOv4's grid-sensitivity fix."""
    tx, ty, tw, th = t
    bx = s * sigmoid(tx) - (s - 1) / 2 + cell[0]        # centre in grid units
    by = s * sigmoid(ty) - (s - 1) / 2 + cell[1]
    return bx * stride, by * stride, anchor[0] * np.exp(tw), anchor[1] * np.exp(th)

# cell (10, 7) of the stride-8 grid, anchor 36 x 75 pixels, raw outputs all zero
print("box (cx, cy, w, h):", np.round(decode((0, 0, 0, 0), (10, 7), (36, 75), 8), 1))

# which raw output tx puts the centre at a given position inside its cell?
for pos in (0.5, 0.9, 0.99, 1.0):
    row = []
    for s in (1.0, 1.2):
        p = (pos + (s - 1) / 2) / s                     # the sigmoid value that is needed
        row.append(f"s={s}: " + (f"{logit(p):5.2f}" if p < 1 else "  inf"))
    print(f"centre at {pos:4.2f} of the cell -> tx = " + " | ".join(row))
