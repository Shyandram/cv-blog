// Browser-only: importing this table does not load the experiment implementations.
import type { Experiment } from './types.ts';
import type { ExperimentGroup } from './catalog.ts';

export const experimentLoaders: Record<ExperimentGroup, () => Promise<Experiment[]>> = {
  vision: () => import('./vision.ts').then(m => m.visionExperiments),
  language: () => import('./language.ts').then(m => m.languageExperiments),
  math: () => import('./math.ts').then(m => m.mathExperiments),
  'chapters-optimization': () => import('./chapters-optimization.ts').then(m => m.chapterOptimizationExperiments),
  'chapters-information': () => import('./chapters-information.ts').then(m => m.chapterInformationExperiments),
  'chapters-recognition': () => import('./chapters-recognition.ts').then(m => m.chapterRecognitionExperiments),
  'chapters-vision': () => import('./chapters-vision.ts').then(m => m.chapterVisionExperiments),
  'chapters-language': () => import('./chapters-language.ts').then(m => m.chapterLanguageExperiments),
};
