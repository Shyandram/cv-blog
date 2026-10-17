import type { Gray, Rgb } from './types.ts';

/** Decode a same-origin image into a canvas ImageData (no tainting because the URL is same-origin). */
async function decode(url: string): Promise<ImageData> {
  const img = new Image();
  img.decoding = 'async';
  img.src = url;
  await img.decode();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}
export async function loadGray(url: string): Promise<Gray> {
  const d = await decode(url), data = new Float32Array(d.width * d.height);
  for (let i = 0; i < data.length; i++) data[i] = d.data[4 * i] / 255;
  return { w: d.width, h: d.height, data };
}
export async function loadRgb(url: string): Promise<Rgb> {
  const d = await decode(url), data = new Float32Array(d.width * d.height * 3);
  for (let i = 0; i < d.width * d.height; i++) for (let c = 0; c < 3; c++) data[3 * i + c] = d.data[4 * i + c] / 255;
  return { w: d.width, h: d.height, data };
}
