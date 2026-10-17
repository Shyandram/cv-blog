// Binary morphology with flat structuring elements. Pixels outside the image count as background (0) in every step, which is
// scipy.ndimage's default (border_value = 0) for binary_erosion, binary_dilation, binary_opening and binary_closing.
// Every element used here is a stack of centred horizontal runs, so each pass costs one prefix-sum lookup per row of the element.

export type ElementShape = 'disk' | 'square' | 'hline' | 'vline';
/** A flat structuring element as centred runs: row `dy` covers columns −hw … +hw. Symmetric about its centre. */
export interface Element { shape: ElementShape; r: number; rows: { dy: number; hw: number }[] }

/**
 * Element of "size" r. disk: all (dx, dy) with dx² + dy² ≤ r² (skimage.morphology.disk(r)); square: (2r+1) × (2r+1);
 * hline: 1 row of 2r+1 pixels; vline: 1 column of 2r+1 pixels.
 */
export function structuringElement(shape: ElementShape, r: number): Element {
  const rows: { dy: number; hw: number }[] = [];
  if (shape === 'hline') rows.push({ dy: 0, hw: r });
  else for (let dy = -r; dy <= r; dy++) rows.push({ dy, hw: shape === 'square' ? r : shape === 'vline' ? 0 : Math.floor(Math.sqrt(r * r - dy * dy) + 1e-9) });
  return { shape, r, rows };
}
/** The element as a (2r+1) × (2r+1) 0/1 array, row-major (for drawing and for comparison with skimage). */
export function elementGrid(se: Element): Uint8Array {
  const n = 2 * se.r + 1, out = new Uint8Array(n * n);
  for (const { dy, hw } of se.rows) for (let dx = -hw; dx <= hw; dx++) out[(dy + se.r) * n + dx + se.r] = 1;
  return out;
}
export const elementSize = (se: Element) => se.rows.reduce((s, row) => s + 2 * row.hw + 1, 0);

function rowPrefix(mask: ArrayLike<number>, w: number, h: number): Int32Array {
  const p = new Int32Array(h * (w + 1));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) p[y * (w + 1) + x + 1] = p[y * (w + 1) + x] + (mask[y * w + x] ? 1 : 0);
  return p;
}
/** Erosion: a pixel stays on only if the whole element, placed on it, lies inside the foreground. */
export function erode(mask: ArrayLike<number>, w: number, h: number, se: Element): Uint8Array {
  const p = rowPrefix(mask, w, h), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let ok = true;
    for (const { dy, hw } of se.rows) {
      const yy = y + dy, x0 = x - hw, x1 = x + hw;
      if (yy < 0 || yy >= h || x0 < 0 || x1 >= w || p[yy * (w + 1) + x1 + 1] - p[yy * (w + 1) + x0] !== 2 * hw + 1) { ok = false; break; }
    }
    out[y * w + x] = ok ? 1 : 0;
  }
  return out;
}
/** Dilation: a pixel turns on if the element, placed on it, touches any foreground pixel (the element is symmetric). */
export function dilate(mask: ArrayLike<number>, w: number, h: number, se: Element): Uint8Array {
  const p = rowPrefix(mask, w, h), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let hit = false;
    for (const { dy, hw } of se.rows) {
      const yy = y + dy;
      if (yy < 0 || yy >= h) continue;
      const x0 = Math.max(0, x - hw), x1 = Math.min(w - 1, x + hw);
      if (x0 <= x1 && p[yy * (w + 1) + x1 + 1] - p[yy * (w + 1) + x0] > 0) { hit = true; break; }
    }
    out[y * w + x] = hit ? 1 : 0;
  }
  return out;
}
export const open = (m: ArrayLike<number>, w: number, h: number, se: Element) => dilate(erode(m, w, h, se), w, h, se);
export const close = (m: ArrayLike<number>, w: number, h: number, se: Element) => erode(dilate(m, w, h, se), w, h, se);
/** Morphological gradient of a binary image: dilation minus erosion (the band around every boundary). */
export function gradient(m: ArrayLike<number>, w: number, h: number, se: Element): Uint8Array {
  const d = dilate(m, w, h, se), e = erode(m, w, h, se), out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = d[i] && !e[i] ? 1 : 0;
  return out;
}
export type Operation = 'erosion' | 'dilation' | 'opening' | 'closing' | 'gradient';
export function applyOperation(op: Operation, m: ArrayLike<number>, w: number, h: number, se: Element): Uint8Array {
  switch (op) {
    case 'erosion': return erode(m, w, h, se);
    case 'dilation': return dilate(m, w, h, se);
    case 'opening': return open(m, w, h, se);
    case 'closing': return close(m, w, h, se);
    case 'gradient': return gradient(m, w, h, se);
  }
}
