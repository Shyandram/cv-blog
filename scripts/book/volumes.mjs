// The book is published as a few volumes. Each volume is a run of consecutive tracks (in site order); the blog essays
// ride with the last volume. Used by export.mjs, publish.mjs and written into src/data/books.json for the download page.
export const volumes = [
  { id: 'vol1', tracks: ['foundations', 'image-processing'], title: { en: 'Foundations and Image Processing', 'zh-tw': '基礎與影像處理' } },
  { id: 'vol2', tracks: ['nlp', 'pattern-recognition'], title: { en: 'Language and Pattern Recognition', 'zh-tw': '自然語言與圖形識別' } },
  { id: 'vol3', tracks: ['ml', 'deep-learning'], title: { en: 'Machine Learning and Deep Learning for Vision', 'zh-tw': '機器學習與電腦視覺深度學習' } },
  { id: 'vol4', tracks: ['optimization', 'information-theory'], posts: true, title: { en: 'Optimization, Information Theory and Essays', 'zh-tw': '最佳化、資訊理論與部落格文章' } },
];
export const stem = (lang, volume) => `pixels-perception-${lang}-${volume.id}`;
