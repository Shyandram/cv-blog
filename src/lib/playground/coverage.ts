import { experiments } from './catalog';
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
    { id: 'blur-compare', track: 'image-processing', title: L('What blur does to an edge', '模糊如何改變邊緣'), description: L('Compare edge transitions and fine textures before and after smoothing.', '比較平滑前後的邊緣過渡與細緻紋理。'), related: ['image-processing/dip-03-intensity-spatial-filtering'] },
];
export const playgroundEntries: PlaygroundEntry[] = [...experiments, ...legacy];
/** Reuse only when the experiment answers a key question in the chapter. First entry is the recommended starting point. */
const reuse: Record<string, string[]> = {
    'foundations/images-as-arrays': ['array-dtype', 'pixel-sampling'],
    'image-processing/dip-01-introduction': ['pixel-sampling', 'kernel-filter'],
    'image-processing/dip-02-fundamentals': ['pixel-sampling'],
    'image-processing/dip-03-intensity-spatial-filtering': ['kernel-filter', 'blur-compare'],
    'image-processing/dip-04-frequency-domain': ['frequency-filter'],
    'nlp/nlp-01-introduction': ['ngram'],
    'nlp/nlp-05-recurrent-neural-networks': ['ngram'],
    'nlp/nlp-04-lexical-semantics-word-embeddings': ['word2vec'],
    'nlp/nlp-07-attention-transformers': ['attention'],
    'nlp/nlp-02-linguistic-essentials': ['quantifier-scope', 'subword-merges', 'dialogue-belief'],
    'nlp/nlp-03-neural-networks-for-nlp': ['nonlinear-network', 'pooling-order', 'gradient-check'],
    'nlp/nlp-08-seq2seq-transformers': ['attention', 'decoding-controls', 'dialogue-belief'],
    'nlp/nlp-09-language-modeling-transformers': ['decoding-controls', 'autoregressive-pixels', 'masked-objectives', 'attention', 'ngram'],
    'nlp/nlp-10-pretrained-bert': ['masked-objectives', 'subword-merges'],
    'nlp/nlp-12-syntactic-semantic-tasks': ['quantifier-scope', 'sequence-decoding'],
    'nlp/nlp-11-classification-bert': ['masked-objectives', 'temperature-calibration'],
    'ml/ml-02-neural-networks': ['nonlinear-network', 'gradient-check'],
    'ml/ml-08-autoregressive-vision': ['autoregressive-pixels', 'decoding-controls'],
    'deep-learning/dl-02-image-classification': ['classification-threshold', 'temperature-calibration'],
    'ml/ml-01-introduction': ['ridge-polynomial'],
    'ml/ml-01a-linear-algebra': ['linear-transform'],
    'ml/ml-01b-probability-information': ['probability-mismatch', 'bayes-decision'],
    'ml/ml-03-convolutional-networks': ['kernel-filter', 'receptive-field'],
    'ml/ml-04-vision-transformers': ['attention'],
    'ml/ml-07-diffusion-flow-matching': ['diffusion-targets'],
};
export const readingOnly: Record<string, Localized> = {
    'foundations/getting-started': L('This chapter is a reading and setup guide. Choose a learning route and run its first example; a separate numerical playground would not test that objective.', '本章重點是閱讀路線與環境設定。請選擇路線並執行第一個範例；獨立的數值實驗無法檢驗這項學習目標。'),
};
export function chapterExperiments(slug: string): PlaygroundEntry[] {
    if (readingOnly[slug])
        return [];
    const preferred = reuse[slug] ?? [];
    const ids = [...preferred, ...playgroundEntries.filter(e => e.related.includes(slug)).map(e => e.id)];
    return [...new Set(ids)].flatMap(id => { const e = playgroundEntries.find(x => x.id === id); return e ? [e] : []; });
}
/** Include reused chapter routes in filters and read-next links as well as the chapter index. */
export function chaptersForExperiment(id: string): string[] {
    const e = playgroundEntries.find(e => e.id === id);
    return [...new Set([...(e?.related ?? []), ...Object.entries(reuse).filter(([, ids]) => ids.includes(id)).map(([slug]) => slug)])].filter(slug => !readingOnly[slug]);
}
