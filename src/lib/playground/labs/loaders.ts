// Browser-only: each lab is a separate chunk, fetched when its card scrolls into view.
import type { LabDef } from './types.ts';

export const labLoaders: Record<string, () => Promise<{ lab: LabDef }>> = {
  'frequency-filter': () => import('./frequency-filter.ts'),
  'segmentation-threshold': () => import('./segmentation-threshold.ts'),
  'lowlight-photons': () => import('./lowlight-photons.ts'),
};
