// Resolved at build time by Astro; the client receives plain URLs through a data attribute.
import camera256 from '../../../assets/playground/camera-256.png';
import cameraCrop128 from '../../../assets/playground/camera-crop-128.png';
import doc from '../../../assets/playground/doc.png';
import docTruth from '../../../assets/playground/doc-truth.png';
import coins from '../../../assets/playground/coins.png';
import astronaut256 from '../../../assets/playground/astronaut-256.png';

const table: Record<string, { src: string }> = {
  'camera-256': camera256, 'camera-crop-128': cameraCrop128, doc, 'doc-truth': docTruth, coins, 'astronaut-256': astronaut256,
};
export function labAssetUrls(keys: string[]): Record<string, string> {
  return Object.fromEntries(keys.map(k => { if (!table[k]) throw new Error(`Unknown lab asset: ${k}`); return [k, table[k].src]; }));
}
