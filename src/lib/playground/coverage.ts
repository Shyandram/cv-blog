import { experiments } from './catalog';
import { labs } from './labs/registry';
import { L, type Localized } from './types';
import type { Track } from '../taxonomy';
/** Chapter routes are the stable keys. Titles and chapter order come from the content collection. */
export interface PlaygroundEntry {
    id: string;
    track: Track;
    title: Localized;
    description: Localized;
    related: string[];
}
const legacy: PlaygroundEntry[] = [
    { id: 'kernel-filter', track: 'image-processing', title: L('Edit a filter kernel', '編輯濾波核'), description: L('Predict the response of flat regions and edges as you change the kernel weights.', '改變核權重，預測平坦區域與邊緣的回應。'), related: ['image-processing/dip-03-intensity-spatial-filtering', 'ml/ml-03-convolutional-networks'] },
];
export const playgroundEntries: PlaygroundEntry[] = [...experiments, ...labs, ...legacy];
/**
 * The plan is the single source of truth for which experiments a chapter shows.
 * At most two per chapter; the first is the suggested starting point. A chapter without an entry
 * is reading-only (see `readingOnlyChapters`), so every demo has to earn its place.
 */
export const MAX_EXPERIMENTS_PER_CHAPTER = 2;
export const chapterPlan: Record<string, string[]> = {
    'foundations/images-as-arrays': ['array-dtype', 'pixel-sampling'],
    'image-processing/dip-02-fundamentals': ['pixel-sampling'],
    'image-processing/dip-03-intensity-spatial-filtering': ['kernel-filter'],
    'image-processing/dip-04-frequency-domain': ['frequency-filter'],
    'image-processing/dip-05-restoration-reconstruction': ['inverse-regularization'],
    'image-processing/dip-06-color': ['color-clipping'],
    'image-processing/dip-08-compression-watermarking': ['transform-coding'],
    'image-processing/dip-09-morphology': ['morphology-shapes'],
    'image-processing/dip-10-segmentation': ['segmentation-threshold'],
    'image-processing/dip-11-active-contours': ['contour-energy'],
    'image-processing/dip-12-feature-extraction': ['corner-tensor'],
    'image-processing/dip-13-pattern-classification': ['neighbor-geometry'],
    'pattern-recognition/pr-01-bayes-decisions': ['bayes-decision'],
    'pattern-recognition/pr-02-density-estimation': ['density-bandwidth'],
    'pattern-recognition/pr-03-linear-models': ['ridge-polynomial'],
    'pattern-recognition/pr-04-kernels-nonlinear-classifiers': ['neighbor-geometry'],
    'pattern-recognition/pr-05-features-dimensionality': ['projection-information'],
    'pattern-recognition/pr-06-templates-structured-models': ['sequence-alignment'],
    'pattern-recognition/pr-07-clustering-mixtures': ['mixture-em'],
    'pattern-recognition/pr-08-approximate-inference': ['variational-kl'],
    'pattern-recognition/pr-09-ensembles-validation': ['bagging-trees'],
    'pattern-recognition/pr-10-recognition-capstone': ['rejection-shift'],
    'deep-learning/dl-01-deep-learning-for-image-processing': ['receptive-field'],
    'deep-learning/dl-02-image-classification': ['classification-threshold', 'temperature-calibration'],
    'deep-learning/dl-03-object-detection': ['detection-nms'],
    'deep-learning/dl-03a-real-time-detection': ['detection-nms'],
    'deep-learning/dl-04-image-restoration': ['restoration-ambiguity'],
    'deep-learning/dl-04a-super-resolution': ['restoration-ambiguity', 'pixel-sampling'],
    'deep-learning/dl-04b-video-super-resolution': ['pixel-sampling'],
    'deep-learning/dl-04c-generative-restoration': ['restoration-ambiguity'],
    'deep-learning/dl-05-low-light-enhancement': ['lowlight-photons'],
    'deep-learning/dl-05a-lightweight-enhancement': ['lowlight-photons'],
    'deep-learning/dl-06-object-tracking': ['tracking-association'],
    'deep-learning/dl-07-world-models': ['world-model-rollout'],
    'ml/ml-01a-linear-algebra': ['linear-transform'],
    'ml/ml-01b-probability-information': ['probability-mismatch', 'bayes-decision'],
    'ml/ml-01c-numerical-computation': ['gradient-check'],
    'ml/ml-02-neural-networks': ['mlp-training', 'gradient-check'],
    'ml/ml-03-convolutional-networks': ['kernel-filter', 'receptive-field'],
    'ml/ml-03a-recurrent-recursive-networks': ['selective-memory'],
    'ml/ml-04-vision-transformers': ['attention'],
    'ml/ml-05-representation-learning': ['contrastive-alignment'],
    'ml/ml-05a-factor-models-autoencoders': ['transform-coding', 'vae-tradeoff'],
    'ml/ml-06-generative-models': ['vae-tradeoff', 'flow-density'],
    'ml/ml-07-diffusion-flow-matching': ['diffusion-targets', 'transport-steps'],
    'ml/ml-08-autoregressive-vision': ['autoregressive-pixels', 'decoding-controls'],
    'ml/ml-09-foundation-models': ['contrastive-alignment'],
    'ml/ml-10-evaluation-deployment': ['temperature-calibration'],
    'nlp/nlp-01-introduction': ['ngram', 'subword-merges'],
    'nlp/nlp-02-linguistic-essentials': ['quantifier-scope', 'subword-merges'],
    'nlp/nlp-03-neural-networks-for-nlp': ['nonlinear-network', 'pooling-order'],
    'nlp/nlp-04-lexical-semantics-word-embeddings': ['pooling-order', 'contrastive-alignment'],
    'nlp/nlp-05-recurrent-neural-networks': ['selective-memory'],
    'nlp/nlp-06-sequence-labeling': ['sequence-decoding'],
    'nlp/nlp-07-attention-transformers': ['attention'],
    'nlp/nlp-08-seq2seq-transformers': ['attention', 'decoding-controls'],
    'nlp/nlp-09-language-modeling-transformers': ['decoding-controls', 'retrieval-ranking'],
    'nlp/nlp-10-pretrained-bert': ['masked-objectives', 'subword-merges'],
    'nlp/nlp-11-classification-bert': ['masked-objectives', 'temperature-calibration'],
    'nlp/nlp-12-syntactic-semantic-tasks': ['quantifier-scope', 'sequence-decoding'],
    'optimization/opt-01-modeling-convex-sets': ['convex-combinations'],
    'optimization/opt-02-convex-functions': ['convex-chords'],
    'optimization/opt-03-convex-programs': ['constrained-quadratic'],
    'optimization/opt-04-duality-kkt': ['duality-certificate'],
    'optimization/opt-05-fitting-estimation': ['robust-location'],
    'optimization/opt-06-gradient-newton': ['gradient-descent'],
    'optimization/opt-07-constrained-interior-point': ['barrier-central-path'],
    'optimization/opt-08-applications-modern-methods': ['proximal-threshold'],
    'information-theory/it-01-entropy-mutual-information': ['binary-channel', 'probability-mismatch'],
    'information-theory/it-02-typicality-entropy-rates': ['typical-sequences'],
    'information-theory/it-03-lossless-universal-coding': ['prefix-coding'],
    'information-theory/it-04-channel-capacity': ['binary-channel'],
    'information-theory/it-05-continuous-gaussian-channels': ['gaussian-water-filling'],
    'information-theory/it-06-rate-distortion': ['binary-rate-distortion'],
    'information-theory/it-07-statistics-decision-growth': ['hypothesis-errors'],
    'information-theory/it-08-networks-information-learning': ['information-synergy'],
};
/** Chapters where the text, figures and exercises already carry the idea. They show no experiment. */
export const readingOnlyChapters = [
    'foundations/getting-started',
    'image-processing/dip-01-introduction',
    'image-processing/dip-07-wavelets-transforms',
    'ml/ml-01-introduction',
];
const defaultReason = L('This chapter\u2019s own figures, code and exercises carry its main idea; a separate interactive demo would not add to them.', '本章的圖、程式與練習已經足以說明主要概念；另外加一個互動實驗不會增加理解。');
const reasons: Record<string, Localized> = {
    'foundations/getting-started': L('This chapter is a reading and setup guide. Choose a learning route and run its first example; a separate numerical playground would not test that objective.', '本章重點是閱讀路線與環境設定。請選擇路線並執行第一個範例；獨立的數值實驗無法檢驗這項學習目標。'),
};
export const readingOnly: Record<string, Localized> = Object.fromEntries(readingOnlyChapters.map(slug => [slug, reasons[slug] ?? defaultReason]));
/** Removed or merged ids keep working as `#id` links on the playground page. */
export const idAliases: Record<string, string> = {
    'blur-compare': 'kernel-filter',
    'prototype-scaling': 'neighbor-geometry',
    'dct-quantization': 'transform-coding',
    'haar-details': 'transform-coding',
    'latent-bottleneck': 'transform-coding',
};
export function chapterExperiments(slug: string): PlaygroundEntry[] {
    return (chapterPlan[slug] ?? []).flatMap(id => { const e = playgroundEntries.find(x => x.id === id); return e ? [e] : []; });
}
/** Chapters that list this experiment, for filters and read-next links. */
export function chaptersForExperiment(id: string): string[] {
    return Object.entries(chapterPlan).filter(([, ids]) => ids.includes(id)).map(([slug]) => slug);
}
