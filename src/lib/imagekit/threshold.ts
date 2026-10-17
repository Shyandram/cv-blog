import { gaussianBlur } from './filter.ts';

/**
 * Otsu's threshold, following scikit-image's `threshold_otsu`: a 256-bin histogram over [min, max],
 * the threshold is the centre of the bin that maximises the between-class variance.
 */
export function otsuThreshold(values: ArrayLike<number>, nbins = 256): number {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < values.length; i++) { const v = values[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  if (lo === hi) return lo;
  const counts = new Float64Array(nbins), width = (hi - lo) / nbins;
  for (let i = 0; i < values.length; i++) counts[Math.min(nbins - 1, Math.floor((values[i] - lo) / width))]++;
  const centers = Float64Array.from({ length: nbins }, (_, i) => lo + (i + 0.5) * width);
  const w1 = new Float64Array(nbins), w2 = new Float64Array(nbins), m1 = new Float64Array(nbins), m2 = new Float64Array(nbins);
  let c = 0, s = 0;
  for (let i = 0; i < nbins; i++) { c += counts[i]; s += counts[i] * centers[i]; w1[i] = c; m1[i] = s / c; }
  c = 0; s = 0;
  for (let i = nbins - 1; i >= 0; i--) { c += counts[i]; s += counts[i] * centers[i]; w2[i] = c; m2[i] = s / c; }
  let best = -1, idx = 0;
  for (let i = 0; i < nbins - 1; i++) {
    const v = w1[i] * w2[i + 1] * (m1[i] - m2[i + 1]) ** 2;
    if (v > best) { best = v; idx = i; }
  }
  return centers[idx];
}

/** Local threshold image, following `threshold_local(method='gaussian')`: blurred image with σ = (block − 1) / 6, minus `offset`. */
export function localThreshold(src: Float32Array, w: number, h: number, block: number, offset = 0): Float32Array {
  const t = gaussianBlur(src, w, h, (block - 1) / 6);
  for (let i = 0; i < t.length; i++) t[i] -= offset;
  return t;
}

export function histogram(values: ArrayLike<number>, bins = 64, lo = 0, hi = 1): Float64Array {
  const out = new Float64Array(bins);
  for (let i = 0; i < values.length; i++) {
    const b = Math.floor(((values[i] - lo) / (hi - lo)) * bins);
    out[Math.min(bins - 1, Math.max(0, b))]++;
  }
  return out;
}
