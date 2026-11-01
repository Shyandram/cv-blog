// Resolved at build time by Astro; the client receives plain URLs through a data attribute.
import camera256 from '../../../assets/playground/camera-256.png';
import cameraCrop128 from '../../../assets/playground/camera-crop-128.png';
import doc from '../../../assets/playground/doc.png';
import docTruth from '../../../assets/playground/doc-truth.png';
import coins from '../../../assets/playground/coins.png';
import astronaut256 from '../../../assets/playground/astronaut-256.png';
import coinsMask from '../../../assets/playground/coins-mask.png';
import zoneplate256 from '../../../assets/playground/zoneplate-256.png';
import coffee256 from '../../../assets/playground/coffee-256.png';
import coinCrop144 from '../../../assets/playground/coin-crop-144.png';
import fusionA from '../../../assets/playground/image-fusion-a.png';
import fusionB from '../../../assets/playground/image-fusion-b.png';
import fusionTruth from '../../../assets/playground/image-fusion-truth.png';
import fusionFocus from '../../../assets/playground/image-fusion-focus.png';

const table: Record<string, { src: string }> = {
  'camera-256': camera256, 'camera-crop-128': cameraCrop128, doc, 'doc-truth': docTruth, coins, 'astronaut-256': astronaut256, 'coins-mask': coinsMask, 'zoneplate-256': zoneplate256, 'coffee-256': coffee256, 'coin-crop-144': coinCrop144, 'image-fusion-a': fusionA, 'image-fusion-b': fusionB, 'image-fusion-truth': fusionTruth, 'image-fusion-focus': fusionFocus,
};
export function labAssetUrls(keys: string[]): Record<string, string> {
  return Object.fromEntries(keys.map(k => { if (!table[k]) throw new Error(`Unknown lab asset: ${k}`); return [k, table[k].src]; }));
}
