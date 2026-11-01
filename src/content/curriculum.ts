// The course tree: the single source of truth for which chapters exist, their order, their nesting and their numbers.
// Chapter numbers are computed from the position in this tree (src/lib/curriculum.ts): a top-level entry is "<PREFIX> NN",
// its children "<PREFIX> NNA", "NNB", …, the next level "NNA.1", "NNA.2", and so on.
// To add a chapter: add { slug: '<track>/<file name without .mdx>' } where it belongs (or run node scripts/curriculum.mjs,
// which adds new files by the letter rule). To move one: move its entry (with its children). Mark a chapter that has no
// zh-tw file yet with enOnly: true. The build and node scripts/curriculum.mjs --check fail if a tutorial is missing here.
import type { CurriculumTree } from '../lib/curriculum.ts';

export const curriculum: CurriculumTree = {
  'foundations': [
    { slug: 'foundations/getting-started' },
    { slug: 'foundations/images-as-arrays' },
  ],
  'image-processing': [
    { slug: 'image-processing/dip-01-introduction' },
    { slug: 'image-processing/dip-02-fundamentals' },
    { slug: 'image-processing/dip-03-intensity-spatial-filtering' },
    { slug: 'image-processing/dip-04-frequency-domain' },
    { slug: 'image-processing/dip-05-restoration-reconstruction' },
    { slug: 'image-processing/dip-06-color' },
    { slug: 'image-processing/dip-07-wavelets-transforms' },
    { slug: 'image-processing/dip-08-compression-watermarking' },
    { slug: 'image-processing/dip-09-morphology' },
    { slug: 'image-processing/dip-10-segmentation' },
    { slug: 'image-processing/dip-11-active-contours' },
    { slug: 'image-processing/dip-12-feature-extraction' },
    { slug: 'image-processing/dip-13-pattern-classification' },
  ],
  'nlp': [
    { slug: 'nlp/nlp-01-introduction' },
    { slug: 'nlp/nlp-02-linguistic-essentials' },
    { slug: 'nlp/nlp-03-neural-networks-for-nlp' },
    { slug: 'nlp/nlp-04-lexical-semantics-word-embeddings' },
    { slug: 'nlp/nlp-05-recurrent-neural-networks' },
    { slug: 'nlp/nlp-06-sequence-labeling' },
    { slug: 'nlp/nlp-07-attention-transformers' },
    { slug: 'nlp/nlp-08-seq2seq-transformers' },
    { slug: 'nlp/nlp-09-language-modeling-transformers' },
    { slug: 'nlp/nlp-10-pretrained-bert' },
    { slug: 'nlp/nlp-11-classification-bert' },
    { slug: 'nlp/nlp-12-syntactic-semantic-tasks' },
  ],
  'pattern-recognition': [
    { slug: 'pattern-recognition/pr-01-bayes-decisions' },
    { slug: 'pattern-recognition/pr-02-density-estimation' },
    { slug: 'pattern-recognition/pr-03-linear-models' },
    { slug: 'pattern-recognition/pr-04-kernels-nonlinear-classifiers' },
    { slug: 'pattern-recognition/pr-05-features-dimensionality' },
    { slug: 'pattern-recognition/pr-06-templates-structured-models' },
    { slug: 'pattern-recognition/pr-07-clustering-mixtures' },
    { slug: 'pattern-recognition/pr-08-approximate-inference' },
    { slug: 'pattern-recognition/pr-09-ensembles-validation' },
    { slug: 'pattern-recognition/pr-10-recognition-capstone' },
  ],
  'ml': [
    { slug: 'ml/ml-01-introduction', children: [
      { slug: 'ml/ml-01a-linear-algebra' },
      { slug: 'ml/ml-01b-probability-information' },
      { slug: 'ml/ml-01c-numerical-computation' },
    ] },
    { slug: 'ml/ml-02-neural-networks' },
    { slug: 'ml/ml-03-convolutional-networks' },
    { slug: 'ml/ml-03a-recurrent-recursive-networks' },
    { slug: 'ml/ml-04-vision-transformers' },
    { slug: 'ml/ml-05-representation-learning', children: [
      { slug: 'ml/ml-05a-factor-models-autoencoders' },
    ] },
    { slug: 'ml/ml-06-generative-models' },
    { slug: 'ml/ml-07-diffusion-flow-matching' },
    { slug: 'ml/ml-08-autoregressive-vision' },
    { slug: 'ml/ml-09-foundation-models' },
    { slug: 'ml/ml-10-evaluation-deployment' },
  ],
  'deep-learning': [
    { slug: 'deep-learning/dl-01-deep-learning-for-image-processing', children: [
      { slug: 'deep-learning/dl-01a-training-practice' },
      { slug: 'deep-learning/dl-01b-building-blocks' },
      { slug: 'deep-learning/dl-01c-model-compression' },
    ] },
    { slug: 'deep-learning/dl-02-image-classification', children: [
      { slug: 'deep-learning/dl-02a-face-metric-learning' },
    ] },
    { slug: 'deep-learning/dl-03-object-detection', children: [
      { slug: 'deep-learning/dl-03a-real-time-detection' },
      { slug: 'deep-learning/dl-09-3d-object-detection' },
    ] },
    { slug: 'deep-learning/dl-03b-image-segmentation' },
    { slug: 'deep-learning/dl-06a-pose-keypoints' },
    { slug: 'deep-learning/dl-11a-text-in-images' },
    { slug: 'deep-learning/dl-06-object-tracking' },
    { slug: 'deep-learning/dl-06b-video-understanding' },
    { slug: 'deep-learning/dl-04-image-restoration', children: [
      { slug: 'deep-learning/dl-04a-super-resolution', children: [
        { slug: 'deep-learning/dl-04b-video-super-resolution' },
      ] },
      { slug: 'deep-learning/dl-04c-generative-restoration' },
      { slug: 'deep-learning/dl-05-low-light-enhancement', children: [
        { slug: 'deep-learning/dl-05a-lightweight-enhancement' },
      ] },
    ] },
    { slug: 'deep-learning/dl-05b-image-fusion' },
    { slug: 'deep-learning/dl-08-3d-reconstruction', children: [
      { slug: 'deep-learning/dl-08a-depth-optical-flow' },
    ] },
    { slug: 'deep-learning/dl-10-image-generation', children: [
      { slug: 'deep-learning/dl-10a-video-generation' },
    ] },
    { slug: 'deep-learning/dl-07-world-models' },
    { slug: 'deep-learning/dl-11-robustness' },
  ],
  'optimization': [
    { slug: 'optimization/opt-01-modeling-convex-sets' },
    { slug: 'optimization/opt-02-convex-functions' },
    { slug: 'optimization/opt-03-convex-programs' },
    { slug: 'optimization/opt-04-duality-kkt' },
    { slug: 'optimization/opt-05-fitting-estimation' },
    { slug: 'optimization/opt-06-gradient-newton' },
    { slug: 'optimization/opt-07-constrained-interior-point' },
    { slug: 'optimization/opt-08-applications-modern-methods' },
  ],
  'information-theory': [
    { slug: 'information-theory/it-01-entropy-mutual-information' },
    { slug: 'information-theory/it-02-typicality-entropy-rates' },
    { slug: 'information-theory/it-03-lossless-universal-coding' },
    { slug: 'information-theory/it-04-channel-capacity' },
    { slug: 'information-theory/it-05-continuous-gaussian-channels' },
    { slug: 'information-theory/it-06-rate-distortion' },
    { slug: 'information-theory/it-07-statistics-decision-growth' },
    { slug: 'information-theory/it-08-networks-information-learning' },
  ],
};
