import { reflectIndex } from './filter.ts';

/**
 * Structure tensor and Harris response, as defined in the DIP 12 chapter (Harris and Stephens, 1988):
 * Gaussian derivatives Ix, Iy, a Gaussian-weighted window w, M = sum w [Ix^2, IxIy; IxIy, Iy^2] and R = det M - k tr(M)^2.
 * Borders are reflected (scipy's 'reflect'), so every pixel has a full window.
 */

/** Normalised Gaussian weights for offsets -radius..radius (the window of a structure tensor). */
export function windowWeights(radius: number, sigma: number): Float64Array {
  const k = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) { k[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += k[i + radius]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return k;
}
/** Correlation weights of a first-order Gaussian derivative, w[i] = i g(i) / sigma^2 (scipy gaussian_filter, order 1). */
export function derivativeWeights(sigma: number, truncate = 4): Float64Array {
  const radius = Math.max(1, Math.floor(truncate * sigma + 0.5)), g = windowWeights(radius, sigma), k = new Float64Array(2 * radius + 1);
  for (let i = -radius; i <= radius; i++) k[i + radius] = (i * g[i + radius]) / (sigma * sigma);
  return k;
}
/** Separable correlation: `kx` along x (columns), `ky` along y (rows), reflected borders. */
export function correlateSeparable(src: Float32Array, w: number, h: number, kx: Float64Array, ky: Float64Array): Float32Array {
  const rx = (kx.length - 1) >> 1, ry = (ky.length - 1) >> 1, tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let i = -rx; i <= rx; i++) s += kx[i + rx] * src[y * w + reflectIndex(x + i, w)];
    tmp[y * w + x] = s;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = -ry; j <= ry; j++) s += ky[j + ry] * tmp[reflectIndex(y + j, h) * w + x];
    out[y * w + x] = s;
  }
  return out;
}
/** Gaussian derivatives of scale sigmaD: ix along x (columns), iy along y (rows), in units of intensity per pixel. */
export function gaussianGradient(src: Float32Array, w: number, h: number, sigmaD: number): { ix: Float32Array; iy: Float32Array } {
  const d = derivativeWeights(sigmaD), g = windowWeights(d.length >> 1, sigmaD);
  return { ix: correlateSeparable(src, w, h, d, g), iy: correlateSeparable(src, w, h, g, d) };
}
/** Windowed entries of the structure tensor at every pixel: xx = sum w Ix^2, yy = sum w Iy^2, xy = sum w IxIy. */
export function structureTensorMaps(ix: Float32Array, iy: Float32Array, w: number, h: number, weights: Float64Array) {
  const n = w * h, axx = new Float32Array(n), ayy = new Float32Array(n), axy = new Float32Array(n);
  for (let i = 0; i < n; i++) { axx[i] = ix[i] * ix[i]; ayy[i] = iy[i] * iy[i]; axy[i] = ix[i] * iy[i]; }
  return {
    xx: correlateSeparable(axx, w, h, weights, weights),
    yy: correlateSeparable(ayy, w, h, weights, weights),
    xy: correlateSeparable(axy, w, h, weights, weights),
  };
}
/** R = det(M) - k tr(M)^2 at every pixel. */
export function harrisResponse(xx: Float32Array, yy: Float32Array, xy: Float32Array, k: number): Float32Array {
  const out = new Float32Array(xx.length);
  for (let i = 0; i < out.length; i++) { const tr = xx[i] + yy[i]; out[i] = xx[i] * yy[i] - xy[i] * xy[i] - k * tr * tr; }
  return out;
}
/** Eigenvalues (l1 >= l2 >= 0) of [[a, c], [c, b]] and the angle (radians) of the eigenvector of l1, measured from the x axis. */
export function eigenSym2(a: number, b: number, c: number): { l1: number; l2: number; angle: number } {
  const tr = a + b, d = Math.hypot(a - b, 2 * c);
  return { l1: (tr + d) / 2, l2: Math.max(0, (tr - d) / 2), angle: 0.5 * Math.atan2(2 * c, a - b) };
}
/** Smaller root t of k t^2 + (2k - 1) t + k = 0: R > 0 exactly when l2 / l1 > t (for l1 > 0). */
export function harrisRatioBoundary(k: number): number {
  const b = 1 - 2 * k;
  return (b - Math.sqrt(b * b - 4 * k * k)) / (2 * k);
}
