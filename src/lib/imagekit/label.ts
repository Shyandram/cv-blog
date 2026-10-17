// Connected-component labeling of binary images. Labels 1 … count are assigned in raster order of each component's first
// pixel, which is how skimage.measure.label numbers them, so the arrays can be compared exactly.

export interface Labeling { labels: Int32Array; count: number; areas: number[] }

/** `connectivity` 4 (edge neighbours) or 8 (edge and corner neighbours); 8 is scikit-image's default for 2-D images. */
export function labelComponents(mask: ArrayLike<number>, w: number, h: number, connectivity: 4 | 8 = 8): Labeling {
  const labels = new Int32Array(w * h), areas: number[] = [], stack = new Int32Array(w * h);
  let count = 0;
  const nb = connectivity === 8 ? [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]] : [[0, -1], [-1, 0], [1, 0], [0, 1]];
  for (let start = 0; start < w * h; start++) {
    if (!mask[start] || labels[start]) continue;
    count++;
    let top = 0, area = 0;
    stack[top++] = start; labels[start] = count;
    while (top) {
      const i = stack[--top], x = i % w, y = (i - x) / w;
      area++;
      for (const [dx, dy] of nb) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const j = yy * w + xx;
        if (mask[j] && !labels[j]) { labels[j] = count; stack[top++] = j; }
      }
    }
    areas.push(area);
  }
  return { labels, count, areas };
}
/**
 * Holes: background components that do not touch the image border. The background uses the dual connectivity (4 when the
 * foreground is 8-connected, and the reverse), so a one-pixel diagonal gap does not leak.
 */
export function countHoles(mask: ArrayLike<number>, w: number, h: number, foreground: 4 | 8 = 8): number {
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < inv.length; i++) inv[i] = mask[i] ? 0 : 1;
  const { labels, count } = labelComponents(inv, w, h, foreground === 8 ? 4 : 8), touches = new Uint8Array(count + 1);
  for (let x = 0; x < w; x++) { touches[labels[x]] = 1; touches[labels[(h - 1) * w + x]] = 1; }
  for (let y = 0; y < h; y++) { touches[labels[y * w]] = 1; touches[labels[y * w + w - 1]] = 1; }
  let holes = 0;
  for (let k = 1; k <= count; k++) if (!touches[k]) holes++;
  return holes;
}
