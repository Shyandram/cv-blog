// Seeded noise with fixed fields: the same seed always gives the same pixels, so a slider never makes noise flicker.
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Uniform field on (0, 1). */
export function uniformField(n: number, seed: number): Float32Array {
  const r = mulberry32(seed), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.min(1 - 1e-9, Math.max(1e-9, r()));
  return out;
}
/** Standard normal field (Box-Muller). */
export function normalField(n: number, seed: number): Float32Array {
  const r = mulberry32(seed), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sqrt(-2 * Math.log(Math.max(1e-12, r()))) * Math.cos(2 * Math.PI * r());
  return out;
}
/** Inverse normal CDF (Acklam's rational approximation, relative error about 1e-9). */
export function invNormal(p: number): number {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const lo = 0.02425;
  if (p < lo) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - lo) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
/**
 * Poisson sample with mean `lambda` from a uniform `u` (inverse CDF). Large means use the normal approximation,
 * which keeps the loop short and is accurate to well under a photon at these levels.
 */
export function poissonFromUniform(lambda: number, u: number): number {
  if (lambda <= 0) return 0;
  if (lambda > 30) return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * invNormal(u)));
  let k = 0, p = Math.exp(-lambda), cdf = p;
  while (u > cdf && k < 200) { k++; p *= lambda / k; cdf += p; }
  return k;
}
