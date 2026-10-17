// Radix-2 FFT (Cooley-Tukey) on power-of-two sizes. Own implementation: no eval, no dependencies.
const twiddles = new Map<number, { cos: Float64Array; sin: Float64Array }>();
function table(n: number) {
  let t = twiddles.get(n);
  if (!t) {
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let k = 0; k < n / 2; k++) { cos[k] = Math.cos((2 * Math.PI * k) / n); sin[k] = -Math.sin((2 * Math.PI * k) / n); }
    t = { cos, sin }; twiddles.set(n, t);
  }
  return t;
}
export const isPow2 = (n: number) => n > 0 && (n & (n - 1)) === 0;

/** In-place 1-D FFT of `n` points starting at `offset` with `stride` between samples. Unnormalized forward, 1/n inverse. */
export function fft1d(re: Float64Array, im: Float64Array, n: number, inverse = false, offset = 0, stride = 1) {
  if (!isPow2(n)) throw new Error(`FFT length must be a power of two, got ${n}`);
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const a = offset + i * stride, b = offset + j * stride;
      [re[a], re[b]] = [re[b], re[a]]; [im[a], im[b]] = [im[b], im[a]];
    }
  }
  const { cos, sin } = table(n);
  const sign = inverse ? -1 : 1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step], wi = sign * sin[k * step];
        const a = offset + (i + k) * stride, b = offset + (i + k + half) * stride;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[offset + i * stride] /= n; im[offset + i * stride] /= n; }
}

/** In-place 2-D FFT of a w×h array stored row-major. */
export function fft2d(re: Float64Array, im: Float64Array, w: number, h: number, inverse = false) {
  for (let y = 0; y < h; y++) fft1d(re, im, w, inverse, y * w, 1);
  for (let x = 0; x < w; x++) fft1d(re, im, h, inverse, x, w);
}

/** Signed frequency index for bin `k` of `n` (cycles per image): 0, 1, …, n/2, −n/2+1, …, −1. */
export const freqIndex = (k: number, n: number) => (k <= n / 2 ? k : k - n);

export type RadialKind = 'ideal' | 'butterworth' | 'gaussian';
/** Radial low-pass transfer function on the unshifted FFT grid. `d0` is the cutoff in cycles per image. */
export function radialLowpass(w: number, h: number, kind: RadialKind, d0: number, order = 2): Float64Array {
  const H = new Float64Array(w * h);
  for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) {
    const d = Math.hypot(freqIndex(u, w), freqIndex(v, h));
    H[v * w + u] = kind === 'ideal' ? (d <= d0 ? 1 : 0)
      : kind === 'butterworth' ? 1 / (1 + (d / d0) ** (2 * order))
      : Math.exp(-(d * d) / (2 * d0 * d0));
  }
  return H;
}

/** Filter a real image in the frequency domain. Returns the real part of the inverse transform (not clipped). */
export function filterFrequency(src: Float32Array, w: number, h: number, H: Float64Array): Float32Array {
  const re = Float64Array.from(src), im = new Float64Array(w * h);
  fft2d(re, im, w, h);
  for (let i = 0; i < re.length; i++) { re[i] *= H[i]; im[i] *= H[i]; }
  fft2d(re, im, w, h, true);
  return Float32Array.from(re);
}

/** Spectrum for display: log(1 + |F|), centered, scaled to [0, 1]. */
export function logSpectrum(src: Float32Array, w: number, h: number): Float32Array {
  const re = Float64Array.from(src), im = new Float64Array(w * h);
  fft2d(re, im, w, h);
  const out = new Float32Array(w * h);
  let max = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const sx = (x + w / 2) % w, sy = (y + h / 2) % h, i = y * w + x, s = sy * w + sx;
    out[i] = Math.log1p(Math.hypot(re[s], im[s]));
    if (out[i] > max) max = out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= max || 1;
  return out;
}
