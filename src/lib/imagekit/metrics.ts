/** Mean squared error between two equally sized arrays. */
export function mse(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return s / a.length;
}
/** PSNR in dB for data in [0, 1] (peak 1), as scikit-image's `peak_signal_noise_ratio(data_range=1)`. */
export function psnr(ref: ArrayLike<number>, test: ArrayLike<number>): number {
  const m = mse(ref, test);
  return m === 0 ? Infinity : 10 * Math.log10(1 / m);
}
/** Intersection over union of two boolean masks. */
export function iou(a: ArrayLike<boolean | number>, b: ArrayLike<boolean | number>): number {
  let i = 0, u = 0;
  for (let k = 0; k < a.length; k++) { const x = !!a[k], y = !!b[k]; if (x && y) i++; if (x || y) u++; }
  return u === 0 ? 1 : i / u;
}
export const meanOf = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
export const stdOf = (a: ArrayLike<number>) => { const m = meanOf(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2; return Math.sqrt(s / a.length); };
