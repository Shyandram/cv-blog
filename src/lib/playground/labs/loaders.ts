// Browser-only: each lab is a separate chunk, fetched when its card scrolls into view.
import type { LabDef } from './types.ts';

export const labLoaders: Record<string, () => Promise<{ lab: LabDef }>> = {
  'frequency-filter': () => import('./frequency-filter.ts'),
  'segmentation-threshold': () => import('./segmentation-threshold.ts'),
  'lowlight-photons': () => import('./lowlight-photons.ts'),
  'transform-coding': () => import('./transform-coding.ts'),
  'inverse-regularization': () => import('./inverse-regularization.ts'),
  'morphology-shapes': () => import('./morphology-shapes.ts'),
  'array-dtype': () => import('./array-dtype.ts'),
  'pixel-sampling': () => import('./pixel-sampling.ts'),
  'color-clipping': () => import('./color-clipping.ts'),
  'corner-tensor': () => import('./corner-tensor.ts'),
  'receptive-field': () => import('./receptive-field.ts'),
  'neighbor-geometry': () => import('./neighbor-geometry.ts'),
  'diffusion-targets': () => import('./diffusion-targets.ts'),
};
