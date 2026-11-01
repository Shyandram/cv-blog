import numpy as np

# the nine anchors (width, height in pixels) of yolov4.cfg; three per output stride
anchors = np.array([[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146],
                    [142, 110], [192, 243], [459, 401]], float)
stride = np.repeat([8, 16, 32], 3)

def shape_iou(wh, anchors):
    """IoU of boxes that share a centre: only width and height matter."""
    inter = np.minimum(wh[0], anchors[:, 0]) * np.minimum(wh[1], anchors[:, 1])
    return inter / (wh[0] * wh[1] + anchors.prod(1) - inter)

for wh in [(30, 60), (100, 90), (60, 120)]:
    iou = shape_iou(np.array(wh, float), anchors)
    pos = np.flatnonzero(iou > 0.213)
    print(f"object {wh[0]:3d}x{wh[1]:<3d} | best anchor {iou.argmax()} (IoU {iou.max():.2f}) | "
          f"IoU > 0.213: anchors {pos.tolist()} on strides {sorted(set(stride[pos].tolist()))}")
