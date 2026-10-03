// Small image-processing helpers shared by the in-browser demos.
// Images are single-channel Float32Array in [0, 1], row-major, n×n.

/** Procedural test image: ramp, disc, square, checkerboard and chirp stripes. */
export function testImage(n: number): Float32Array {
  const g = new Float32Array(n * n);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const u = x / n, v = y / n;
      let val = 0.25 + 0.5 * u;
      const dx = u - 0.33, dy = v - 0.36;
      if (dx * dx + dy * dy < 0.045) val = 0.92;
      if (u > 0.58 && u < 0.9 && v > 0.14 && v < 0.46) val = 0.1;
      if (v > 0.6 && u < 0.5) val = (Math.floor(u * 24) + Math.floor(v * 24)) % 2 ? 0.85 : 0.15;
      if (v > 0.6 && u >= 0.5) val = 0.5 + 0.45 * Math.sin(u * n * 0.9 * (0.4 + v));
      g[y * n + x] = val;
    }
  return g;
}

/** Cross-correlation with a ks×ks kernel, edge-replicated borders. */
export function convolve(src: Float32Array, n: number, K: ArrayLike<number>, ks: number): Float32Array {
  const out = new Float32Array(n * n), r = ks >> 1;
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      let s = 0;
      for (let j = -r; j <= r; j++) {
        const yy = Math.min(n - 1, Math.max(0, y + j));
        for (let i = -r; i <= r; i++) {
          const xx = Math.min(n - 1, Math.max(0, x + i));
          s += K[(j + r) * ks + (i + r)] * src[yy * n + xx];
        }
      }
      out[y * n + x] = s;
    }
  return out;
}

export function gaussianKernel(size: number, sigma: number): number[] {
  const r = size >> 1, k: number[] = [];
  let sum = 0;
  for (let j = -r; j <= r; j++)
    for (let i = -r; i <= r; i++) {
      const w = Math.exp(-(i * i + j * j) / (2 * sigma * sigma));
      k.push(w);
      sum += w;
    }
  return k.map((w) => w / sum);
}

/** Draw a grayscale array to a canvas. `signed` maps [-1, 1] to [0, 1] for zero-sum kernels. */
export function drawGray(canvas: HTMLCanvasElement, g: Float32Array, n: number, signed = false) {
  const ctx = canvas.getContext('2d')!;
  const im = ctx.createImageData(n, n);
  for (let p = 0; p < n * n; p++) {
    const v = Math.max(0, Math.min(255, Math.round((signed ? 0.5 + 0.5 * g[p] : g[p]) * 255)));
    im.data[p * 4] = im.data[p * 4 + 1] = im.data[p * 4 + 2] = v;
    im.data[p * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}
