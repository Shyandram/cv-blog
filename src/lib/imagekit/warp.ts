// Warp a gray image by a 2 x 2 matrix about a chosen origin, by inverse mapping with bilinear interpolation.
//
// Conventions (the same as scipy.ndimage and skimage.transform.warp): pixel (col, row) has its centre at the integer coordinates
// (col, row), so the image occupies [0, w - 1] x [0, h - 1]. A `Frame` places the plane origin at pixel (cx, cy) and measures
// plane lengths in units of `unit` pixels; with `yUp` the plane y axis points up the screen (rows decrease), as in a math figure.
// The forward map is q = A p for plane points; every OUTPUT pixel looks up the source at p = A^{-1} q, so there are no holes.
//
// Boundary modes for samples outside the source:
//   'constant'      the value is 0 for any sample outside [0, w - 1] x [0, h - 1]; no blending across the edge (scipy mode 'constant').
//   'grid-constant' neighbours outside the grid count as 0, so the picture fades over one pixel at its border
//                   (scipy mode 'grid-constant', and also skimage.transform.warp with mode 'constant').
// Results are premultiplied: `data` is the interpolated value with outside neighbours counted as 0, and `alpha` is the share of the
// interpolation weight that came from inside the picture. `data + (1 - alpha) * background` composites over any background.
import type { Gray } from './types.ts';
import { inv2, svd2 } from './matrix2.ts';
import type { Mat2 } from './matrix2.ts';

export interface Frame { cx: number; cy: number; unit: number; yUp: boolean }
export type Boundary = 'constant' | 'grid-constant';
export interface Warped { w: number; h: number; data: Float32Array; alpha: Float32Array }
export interface WarpOptions {
  /** Frame of the source image (default: centred, unit 1 pixel, y down). */
  frame?: Frame;
  /** Output size (default: the source size) and output frame (default: the same centring rule as the source frame). */
  outW?: number; outH?: number; outFrame?: Frame;
  mode?: Boundary;
  /** Average sub-samples over the footprint of each output pixel along the two output directions of the SVD (box filter). */
  antialias?: boolean;
}

/** Plane origin at the centre of a w x h image (the middle of the middle pixels). */
export const centredFrame = (w: number, h: number, unit = 1, yUp = false): Frame => ({ cx: (w - 1) / 2, cy: (h - 1) / 2, unit, yUp });
export const planeToPixel = (f: Frame, x: number, y: number): [number, number] => [f.cx + f.unit * x, f.cy + (f.yUp ? -f.unit * y : f.unit * y)];
export const pixelToPlane = (f: Frame, col: number, row: number): [number, number] => [(col - f.cx) / f.unit, (f.yUp ? -(row - f.cy) : row - f.cy) / f.unit];

let lastAlpha = 0;
const EDGE_TOL = 1e-9;
/** Bilinear sample of `src` at fractional pixel coordinates; returns the premultiplied value and leaves its alpha in `lastAlpha`. */
function sample(src: Gray, col: number, row: number, mode: Boundary): number {
  const w = src.w, h = src.h, d = src.data;
  if (mode === 'constant') {
    // A sample within EDGE_TOL of the edge counts as on the edge (a coordinate that is exactly 0 in exact arithmetic can come out as -1e-16).
    if (!(col >= -EDGE_TOL && col <= w - 1 + EDGE_TOL && row >= -EDGE_TOL && row <= h - 1 + EDGE_TOL)) { lastAlpha = 0; return 0; }
    col = Math.min(w - 1, Math.max(0, col)); row = Math.min(h - 1, Math.max(0, row));
    const x0 = Math.floor(col), y0 = Math.floor(row), x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1), fx = col - x0, fy = row - y0;
    lastAlpha = 1;
    return (1 - fy) * ((1 - fx) * d[y0 * w + x0] + fx * d[y0 * w + x1]) + fy * ((1 - fx) * d[y1 * w + x0] + fx * d[y1 * w + x1]);
  }
  if (!(col > -1 && col < w && row > -1 && row < h)) { lastAlpha = 0; return 0; }
  const x0 = Math.floor(col), y0 = Math.floor(row), fx = col - x0, fy = row - y0;
  const wx0 = x0 >= 0 ? 1 - fx : 0, wx1 = x0 + 1 <= w - 1 ? fx : 0, wy0 = y0 >= 0 ? 1 - fy : 0, wy1 = y0 + 1 <= h - 1 ? fy : 0;
  let v = 0;
  if (wy0 > 0) v += wy0 * ((wx0 > 0 ? wx0 * d[y0 * w + x0] : 0) + (wx1 > 0 ? wx1 * d[y0 * w + x0 + 1] : 0));
  if (wy1 > 0) v += wy1 * ((wx0 > 0 ? wx0 * d[(y0 + 1) * w + x0] : 0) + (wx1 > 0 ? wx1 * d[(y0 + 1) * w + x0 + 1] : 0));
  lastAlpha = (wx0 + wx1) * (wy0 + wy1);
  return v;
}
/** Bilinear value at fractional pixel coordinates (alpha is the second return value). Exposed for tests. */
export function bilinearAt(src: Gray, col: number, row: number, mode: Boundary = 'grid-constant'): { value: number; alpha: number } {
  const value = sample(src, col, row, mode);
  return { value, alpha: lastAlpha };
}

/**
 * Inverse-mapping warp of `src` by the invertible matrix `A`. With `antialias` false it takes one bilinear sample per output pixel
 * (this is what the golden test compares with scipy.ndimage.affine_transform). With `antialias` true it averages
 * ceil(1 / s1) x ceil(1 / s2) samples across the pixel along the output singular directions, which removes the jagged aliasing
 * when the picture is squeezed (singular values below 1). Throws if A is singular: use `warpMatrix` for any matrix.
 */
export function warpBilinear(src: Gray, A: Mat2, o: WarpOptions = {}): Warped {
  const Ai = inv2(A);
  if (!Ai) throw new Error('warpBilinear needs an invertible matrix; use warpMatrix for singular ones');
  const w = o.outW ?? src.w, h = o.outH ?? src.h, mode = o.mode ?? 'grid-constant';
  const frame = o.frame ?? centredFrame(src.w, src.h), out = o.outFrame ?? (w === src.w && h === src.h ? frame : centredFrame(w, h, frame.unit, frame.yUp));
  const data = new Float32Array(w * h), alpha = new Float32Array(w * h);
  const sv = svd2(A), n1 = o.antialias ? Math.max(1, Math.ceil(1 / sv.s1 - 1e-9)) : 1, n2 = o.antialias ? Math.max(1, Math.ceil(1 / sv.s2 - 1e-9)) : 1;
  const reach = o.antialias ? 0.5 * (1 / sv.s1 + 1 / sv.s2) + 1 : 1;
  const multi = n1 > 1 || n2 > 1, inv = 1 / (n1 * n2);
  // Offsets of the sub-samples across one output pixel, in plane coordinates.
  const off: number[] = [];
  if (multi) for (let i = 0; i < n1; i++) for (let j = 0; j < n2; j++) {
    const a = (i + 0.5) / n1 - 0.5, b = (j + 0.5) / n2 - 0.5;
    off.push((a * sv.u1[0] + b * sv.u2[0]) / out.unit, (a * sv.u1[1] + b * sv.u2[1]) / out.unit);
  }
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const [x, y] = pixelToPlane(out, c, r);
    const [pc, pr] = planeToPixel(frame, Ai[0] * x + Ai[1] * y, Ai[2] * x + Ai[3] * y);
    if (pc < -reach || pc > src.w - 1 + reach || pr < -reach || pr > src.h - 1 + reach) continue;
    if (!multi) { data[r * w + c] = sample(src, pc, pr, mode); alpha[r * w + c] = lastAlpha; continue; }
    let sv_ = 0, sa = 0;
    for (let k = 0; k < off.length; k += 2) {
      const qx = x + off[k], qy = y + off[k + 1];
      const [sc, sr] = planeToPixel(frame, Ai[0] * qx + Ai[1] * qy, Ai[2] * qx + Ai[3] * qy);
      sv_ += sample(src, sc, sr, mode); sa += lastAlpha;
    }
    data[r * w + c] = sv_ * inv; alpha[r * w + c] = sa * inv;
  }
  return { w, h, data, alpha };
}

/** Below this singular value (in pixels per pixel) a matrix flattens the picture to less than a pixel across and is drawn by `warpMatrix` as a line or a point. */
export const COLLAPSE_BELOW = 1e-3;
/** Half-thickness of the drawn line or point, in output pixels. */
export const LINE_HALF = 1.25;
const softBand = (distancePx: number) => Math.max(0, Math.min(1, LINE_HALF - distancePx));

/**
 * Rank-1 matrix (or almost): the whole plane lands on the line through the origin along u1, and many source points land on the same
 * spot. The point t u1 of that line shows the MEAN of the picture along the source line {p : v1 . p = t / s1}, which is how a
 * projection loses information. The line is drawn about 2.5 pixels thick.
 */
export function warpLine(src: Gray, A: Mat2, o: WarpOptions = {}): Warped {
  const w = o.outW ?? src.w, h = o.outH ?? src.h, mode = o.mode ?? 'grid-constant';
  const frame = o.frame ?? centredFrame(src.w, src.h), out = o.outFrame ?? (w === src.w && h === src.h ? frame : centredFrame(w, h, frame.unit, frame.yUp));
  const data = new Float32Array(w * h), alpha = new Float32Array(w * h), sv = svd2(A);
  const [ux, uy] = sv.u1, [vx, vy] = sv.v1, half = Math.ceil(Math.hypot(src.w, src.h) / 2) + 2, step = 1 / frame.unit;
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const [x, y] = pixelToPlane(out, c, r), perp = (-uy * x + ux * y) * out.unit, band = softBand(Math.abs(perp));
    if (band <= 0) continue;
    const t = (ux * x + uy * y) / sv.s1;
    let sum = 0, weight = 0;
    for (let s = -half; s <= half; s++) {
      const px = t * vx - vy * s * step, py = t * vy + vx * s * step, [sc, sr] = planeToPixel(frame, px, py);
      sum += sample(src, sc, sr, mode); weight += lastAlpha;
    }
    if (weight < 0.5) continue;
    data[r * w + c] = band * (sum / weight); alpha[r * w + c] = band;
  }
  return { w, h, data, alpha };
}
/** Zero matrix (or almost): everything lands on the origin, drawn as a dot that shows the mean of the picture. */
export function warpPoint(src: Gray, o: WarpOptions = {}): Warped {
  const w = o.outW ?? src.w, h = o.outH ?? src.h, frame = o.frame ?? centredFrame(src.w, src.h);
  const out = o.outFrame ?? (w === src.w && h === src.h ? frame : centredFrame(w, h, frame.unit, frame.yUp));
  let mean = 0;
  for (let i = 0; i < src.data.length; i++) mean += src.data[i];
  mean /= src.data.length;
  const data = new Float32Array(w * h), alpha = new Float32Array(w * h);
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const [x, y] = pixelToPlane(out, c, r), band = softBand(Math.hypot(x, y) * out.unit - 1);
    if (band > 0) { data[r * w + c] = band * mean; alpha[r * w + c] = band; }
  }
  return { w, h, data, alpha };
}
/** Warp by any 2 x 2 matrix: bilinear inverse mapping (antialiased) when it is invertible enough, a line when it has rank 1, a dot when it is zero. */
export function warpMatrix(src: Gray, A: Mat2, o: WarpOptions = {}): Warped {
  const sv = svd2(A);
  if (sv.s1 < COLLAPSE_BELOW) return warpPoint(src, o);
  if (sv.s2 < COLLAPSE_BELOW) return warpLine(src, A, o);
  return warpBilinear(src, A, { antialias: true, ...o });
}
/** Composite a warp over a constant background value. */
export function overBackground(wp: Warped, background: number): Float32Array {
  const out = new Float32Array(wp.data.length);
  for (let i = 0; i < out.length; i++) out[i] = wp.data[i] + (1 - wp.alpha[i]) * background;
  return out;
}
