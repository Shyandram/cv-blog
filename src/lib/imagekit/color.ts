/** RGB (each in [0, 1]) to HSV as scikit-image's `rgb2hsv`, with hue in degrees [0, 360). Gray pixels (and black) get hue 0 and saturation 0 by convention, so test saturation before trusting a hue. */
export function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  const s = max === 0 ? 0 : d / max;
  if (d === 0) return [0, s, max];
  let h: number;
  if (max === r) h = ((g - b) / d) % 6;
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  h *= 60;
  return [h < 0 ? h + 360 : h, s, max];
}
/** Shortest angular distance between two hues in degrees, in [0, 180]. */
export function hueDistance(a: number, b: number): number {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}
