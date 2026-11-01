import numpy as np

def iou_terms(a, b):
    """a, b: boxes (x1, y1, x2, y2). Returns IoU, the DIoU penalty rho^2/c^2 and the CIoU term alpha*v."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    wa, ha, wb, hb = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    iou = inter / (wa * ha + wb * hb - inter)
    cw = max(a[2], b[2]) - min(a[0], b[0]); ch = max(a[3], b[3]) - min(a[1], b[1])
    rho2 = ((a[0] + a[2] - b[0] - b[2]) ** 2 + (a[1] + a[3] - b[1] - b[3]) ** 2) / 4
    v = 4 / np.pi ** 2 * (np.arctan(wb / hb) - np.arctan(wa / ha)) ** 2
    alpha = v / ((1 - iou) + v) if v > 0 else 0.0
    return iou, rho2 / (cw ** 2 + ch ** 2), alpha * v

gt = (100, 100, 200, 300)                                  # 100 x 200 pixels, tall
preds = {"centred, lying flat":       (50, 150, 250, 250),
         "shifted, right shape":      (125, 125, 225, 325),
         "far away":                  (300, 100, 400, 300)}
for name, p in preds.items():
    iou, d, av = iou_terms(p, gt)
    print(f"{name:26s} IoU {iou:.3f} | L_IoU {1 - iou:.3f} | L_DIoU {1 - iou + d:.3f} | L_CIoU {1 - iou + d + av:.3f}")

def nms(boxes, scores, thr=0.5, diou=False):
    order, keep = list(np.argsort(-scores)), []
    while order:
        m = order.pop(0); keep.append(int(m))
        crit = lambda j: iou_terms(boxes[m], boxes[j])[0] - (iou_terms(boxes[m], boxes[j])[1] if diou else 0)
        order = [j for j in order if crit(j) < thr]
    return keep

# two people side by side (centres 32 px apart) and one duplicate of the first person
boxes = np.array([[100, 100, 200, 200], [132, 100, 232, 200], [103, 100, 203, 200]], float)
scores = np.array([0.9, 0.8, 0.7])
print("greedy NMS keeps:", nms(boxes, scores), "| DIoU-NMS keeps:", nms(boxes, scores, diou=True))
