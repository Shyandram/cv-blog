// Block DCT (8×8, orthonormal DCT-II, as scipy.fft.dctn(norm='ortho')) and the orthonormal Haar wavelet.
const C = (() => {
  const m = new Float64Array(64);
  for (let k = 0; k < 8; k++) for (let n = 0; n < 8; n++) m[k * 8 + n] = (k === 0 ? Math.sqrt(1 / 8) : Math.sqrt(2 / 8)) * Math.cos(((2 * n + 1) * k * Math.PI) / 16);
  return m;
})();

/** Forward 8×8 DCT of `block` (row-major, 64 values). Returns a new array. */
export function dct8(block: ArrayLike<number>): Float64Array {
  const tmp = new Float64Array(64), out = new Float64Array(64);
  for (let k = 0; k < 8; k++) for (let n = 0; n < 8; n++) { let s = 0; for (let j = 0; j < 8; j++) s += C[k * 8 + j] * block[j * 8 + n]; tmp[k * 8 + n] = s; }
  for (let k = 0; k < 8; k++) for (let l = 0; l < 8; l++) { let s = 0; for (let n = 0; n < 8; n++) s += tmp[k * 8 + n] * C[l * 8 + n]; out[k * 8 + l] = s; }
  return out;
}
/** Inverse 8×8 DCT. */
export function idct8(coef: ArrayLike<number>): Float64Array {
  const tmp = new Float64Array(64), out = new Float64Array(64);
  for (let n = 0; n < 8; n++) for (let l = 0; l < 8; l++) { let s = 0; for (let k = 0; k < 8; k++) s += C[k * 8 + n] * coef[k * 8 + l]; tmp[n * 8 + l] = s; }
  for (let n = 0; n < 8; n++) for (let m = 0; m < 8; m++) { let s = 0; for (let l = 0; l < 8; l++) s += tmp[n * 8 + l] * C[l * 8 + m]; out[n * 8 + m] = s; }
  return out;
}

/** DCT of every 8×8 block; coefficients stay in the block's own position. `w` and `h` must be multiples of 8. */
export function blockDct(src: Float32Array, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h), blk = new Float64Array(64);
  for (let by = 0; by < h; by += 8) for (let bx = 0; bx < w; bx += 8) {
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) blk[y * 8 + x] = src[(by + y) * w + bx + x];
    const c = dct8(blk);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) out[(by + y) * w + bx + x] = c[y * 8 + x];
  }
  return out;
}
export function blockIdct(coef: Float32Array, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h), blk = new Float64Array(64);
  for (let by = 0; by < h; by += 8) for (let bx = 0; bx < w; bx += 8) {
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) blk[y * 8 + x] = coef[(by + y) * w + bx + x];
    const p = idct8(blk);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) out[(by + y) * w + bx + x] = p[y * 8 + x];
  }
  return out;
}

/** Uniform mid-tread quantizer: round(c / step) * step. Returns the quantized array and the fraction of nonzero entries. */
export function quantizeUniform(coef: Float32Array, step: number): { q: Float32Array; nonzero: number } {
  const q = new Float32Array(coef.length);
  let nz = 0;
  for (let i = 0; i < q.length; i++) { const v = Math.round(coef[i] / step) * step; q[i] = v; if (v !== 0) nz++; }
  return { q, nonzero: nz / q.length };
}
/** Keep the `fraction` largest-magnitude entries and zero the rest. */
export function keepLargest(coef: Float32Array, fraction: number): { q: Float32Array; nonzero: number } {
  const n = Math.max(0, Math.min(coef.length, Math.round(fraction * coef.length)));
  const order = Array.from(coef.keys()).sort((a, b) => Math.abs(coef[b]) - Math.abs(coef[a]));
  const q = new Float32Array(coef.length);
  for (let i = 0; i < n; i++) q[order[i]] = coef[order[i]];
  let nz = 0; for (const v of q) if (v !== 0) nz++;
  return { q, nonzero: nz / q.length };
}

const SQ = Math.SQRT1_2;
/** One level of the 2-D orthonormal Haar transform on the top-left `w`×`h` region (Mallat layout: LL, HL / LH, HH). */
function haarStep(a: Float32Array, stride: number, w: number, h: number, inverse: boolean) {
  const tmp = new Float32Array(Math.max(w, h));
  const pass = (len: number, get: (i: number) => number, set: (i: number, v: number) => void) => {
    if (!inverse) {
      for (let i = 0; i < len / 2; i++) { const p = get(2 * i), q = get(2 * i + 1); tmp[i] = (p + q) * SQ; tmp[len / 2 + i] = (p - q) * SQ; }
    } else {
      for (let i = 0; i < len / 2; i++) { const s = get(i), d = get(len / 2 + i); tmp[2 * i] = (s + d) * SQ; tmp[2 * i + 1] = (s - d) * SQ; }
    }
    for (let i = 0; i < len; i++) set(i, tmp[i]);
  };
  const rows = () => { for (let y = 0; y < h; y++) pass(w, i => a[y * stride + i], (i, v) => { a[y * stride + i] = v; }); };
  const cols = () => { for (let x = 0; x < w; x++) pass(h, i => a[i * stride + x], (i, v) => { a[i * stride + x] = v; }); };
  if (!inverse) { rows(); cols(); } else { cols(); rows(); }
}
/** Multi-level 2-D Haar transform (sizes must be divisible by 2^levels). Returns a new array. */
export function haar2d(src: Float32Array, w: number, h: number, levels: number): Float32Array {
  const a = Float32Array.from(src);
  for (let l = 0; l < levels; l++) haarStep(a, w, w >> l, h >> l, false);
  return a;
}
export function ihaar2d(coef: Float32Array, w: number, h: number, levels: number): Float32Array {
  const a = Float32Array.from(coef);
  for (let l = levels - 1; l >= 0; l--) haarStep(a, w, w >> l, h >> l, true);
  return a;
}
