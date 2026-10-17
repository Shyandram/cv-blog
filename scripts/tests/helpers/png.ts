// Minimal PNG reader for tests: 8-bit gray (type 0) and RGB (type 2), non-interlaced. Lets Node tests run a lab's real
// pipeline on the real asset in src/assets/playground/, so every number quoted in a lab's text can be asserted.
import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import type { Gray, Rgb } from '../../../src/lib/imagekit/types.ts';

function decode(path: string) {
  const buf = readFileSync(path);
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error(`${path} is not a PNG`);
  let pos = 8, w = 0, h = 0, depth = 0, type = 0, interlace = 0;
  const idat: Buffer[] = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos), tag = buf.toString('ascii', pos + 4, pos + 8), data = buf.subarray(pos + 8, pos + 8 + len);
    if (tag === 'IHDR') { w = data.readUInt32BE(0); h = data.readUInt32BE(4); depth = data[8]; type = data[9]; interlace = data[12]; }
    else if (tag === 'IDAT') idat.push(data);
    else if (tag === 'IEND') break;
    pos += 12 + len;
  }
  if (depth !== 8 || interlace !== 0 || (type !== 0 && type !== 2)) throw new Error(`unsupported PNG (depth ${depth}, type ${type}, interlace ${interlace})`);
  const bpp = type === 0 ? 1 : 3, stride = w * bpp, raw = inflateSync(Buffer.concat(idat)), out = new Uint8Array(h * stride);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)], line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? out[y * stride + x - bpp] : 0, b = y > 0 ? out[(y - 1) * stride + x] : 0, c = x >= bpp && y > 0 ? out[(y - 1) * stride + x - bpp] : 0;
      let v = line[x];
      if (f === 1) v += a; else if (f === 2) v += b; else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      out[y * stride + x] = v & 255;
    }
  }
  return { w, h, bpp, bytes: out };
}
/** Decode a gray PNG to a Gray image in [0, 1] (as the browser's loadGray does). */
export function readGray(path: string): Gray {
  const { w, h, bpp, bytes } = decode(path), data = new Float32Array(w * h);
  for (let i = 0; i < data.length; i++) data[i] = bytes[i * bpp] / 255;
  return { w, h, data };
}
/** Decode an RGB PNG to an Rgb image in [0, 1] (as loadRgb does). */
export function readRgb(path: string): Rgb {
  const { w, h, bpp, bytes } = decode(path);
  if (bpp !== 3) throw new Error('not an RGB PNG');
  return { w, h, data: Float32Array.from(bytes, v => v / 255) };
}
export const asset = (name: string) => new URL(`../../../src/assets/playground/${name}`, import.meta.url).pathname;
