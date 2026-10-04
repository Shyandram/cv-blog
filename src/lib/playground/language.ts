import type { Experiment } from './types.ts';
import { L, pick, fmt, svg, text } from './types.ts';

const bounded = (value: unknown, fallback: number, min: number, max: number) => {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : fallback;
};
const short = (value: string, budget = 16) => {
  let width = 0;
  let result = '';
  for (const character of value) {
    width += /[^\u0000-\u00ff]/u.test(character) ? 2 : 1;
    if (width > budget - 1) return result + '…';
    result += character;
  }
  return result;
};
const tokens = (value: string) => value.trim().toLocaleLowerCase('en').split(/\s+/u).filter(Boolean);
const dot = (a: number[], b: number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0);

/** A completely masked row has no distribution and is represented by zeros. */
export function stableSoftmax(scores: number[], allowed = scores.map(() => true)): number[] {
  const active = scores.map((s, i) => allowed[i] && Number.isFinite(s));
  const valid = scores.filter((_, i) => active[i]);
  if (!valid.length) return scores.map(() => 0);
  const maximum = Math.max(...valid);
  const weights = scores.map((score, i) => active[i] ? Math.exp(score - maximum) : 0);
  const sum = weights.reduce((a, b) => a + b, 0);
  return weights.map(w => w / sum);
}

export interface NgramResult {
  vocabulary: string[];
  context: string[];
  counts: number[];
  probabilities: number[];
  contextCount: number;
  unknown: string[];
  defined: boolean;
  sentences: number;
}

/** Whitespace tokens; every nonempty line is a separate sentence. No query fitting. */
export function ngramDistribution(corpus: string, query: string, order: number, alpha: number): NgramResult {
  const n = Math.round(bounded(order, 2, 1, 3));
  const smoothing = bounded(alpha, 0.1, 0, 2);
  const sentences = corpus.slice(0, 8000).split(/\r?\n/u).map(tokens).filter(row => row.length);
  if (!sentences.length) return { vocabulary: [], context: [], counts: [], probabilities: [], contextCount: 0, unknown: [], defined: false, sentences: 0 };
  const words = new Set(sentences.flat());
  const vocabulary = [...words].sort().concat('[EOS]', '[UNK]');
  const supplied = n > 1 ? tokens(query.slice(0, 8000)).slice(-(n - 1)) : [];
  const unknown = supplied.filter(word => !words.has(word));
  const context = supplied.map(word => words.has(word) ? word : '[UNK]');
  while (context.length < n - 1) context.unshift('[BOS]');
  const counts = vocabulary.map(() => 0);
  const positions = new Map(vocabulary.map((word, i) => [word, i]));
  for (const sentence of sentences) {
    const sequence = [...Array<string>(n - 1).fill('[BOS]'), ...sentence, '[EOS]'];
    for (let i = n - 1; i < sequence.length; i++) {
      if (context.every((word, j) => sequence[i - n + 1 + j] === word)) counts[positions.get(sequence[i])!]++;
    }
  }
  const contextCount = counts.reduce((a, b) => a + b, 0);
  const denominator = contextCount + smoothing * vocabulary.length;
  return {
    vocabulary, context, counts,
    probabilities: counts.map(count => denominator > 0 ? (count + smoothing) / denominator : 0),
    contextCount, unknown, defined: denominator > 0, sentences: sentences.length,
  };
}

export const WORD_TOKENS = ['small', 'red', 'robots', 'move', 'slowly'];
/** Explicit synthetic input and output lookup tables; no training or semantic claims. */
export const WORD_E = [[1, 0], [0, 1], [1, 1], [-1, 0.5], [0.5, -1]];
export const WORD_U = [[0.5, 1], [-0.5, 0.5], [1, -0.5], [-1, -0.5], [0, 1]];

export function word2vecExample(center: number, radius: number, mode: 'cbow' | 'skipgram') {
  const index = Math.round(bounded(center, 2, 0, WORD_TOKENS.length - 1));
  const window = Math.round(bounded(radius, 1, 1, 2));
  const neighbors = WORD_TOKENS.map((_, i) => i).filter(i => i !== index && Math.abs(i - index) <= window);
  const hidden = mode === 'cbow'
    ? [0, 1].map(d => neighbors.reduce((sum, i) => sum + WORD_E[i][d], 0) / neighbors.length)
    : [...WORD_E[index]];
  const probabilities = stableSoftmax(WORD_U.map(output => dot(output, hidden)));
  const targets = mode === 'cbow' ? [index] : neighbors;
  const losses = targets.map(target => -Math.log(probabilities[target]));
  return { index, neighbors, hidden, probabilities, targets, losses, loss: losses.reduce((a, b) => a + b, 0) / losses.length };
}

export const ATTENTION_TOKENS = ['we', 'see', 'small', 'robots', 'move'];
export const ATTENTION_Q = [[1, 0], [0.5, 1], [1, 1], [-0.5, 1], [0, -1]];
export const ATTENTION_K = [[1, 0], [0, 1], [1, 1], [-1, 0.5], [0.5, -1]];
export const ATTENTION_V = [[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1]];

export function attentionExample(query: number, causal: boolean, temperature: number) {
  const index = Math.round(bounded(query, 2, 0, ATTENTION_TOKENS.length - 1));
  const tau = bounded(temperature, 1, 0.2, 2);
  const scores = ATTENTION_K.map(key => dot(ATTENTION_Q[index], key) / Math.sqrt(2) / tau);
  const allowed = scores.map((_, i) => !causal || i <= index);
  const weights = stableSoftmax(scores, allowed);
  const output = [0, 1].map(d => weights.reduce((sum, weight, i) => sum + weight * ATTENTION_V[i][d], 0));
  return { index, scores, allowed, weights, output };
}

const ngram: Experiment = {
  id: 'ngram', track: 'nlp',
  title: L('Count the next word', '數一數下一個詞'),
  description: L('Build an n-gram distribution from a tiny editable corpus. Inspect counts, sentence boundaries and smoothing.', '用可編輯的小型語料建立 n-gram 分布，觀察次數、句子邊界與平滑化。'),
  challenge: L('Enter an unseen context and set α to zero. Why is the estimate undefined? Increase α and inspect where the probability goes.', '輸入未見過的上下文，把 α 設為零。估計為何沒有定義？提高 α，觀察機率如何分配。'),
  note: L('Each nonempty line is one sentence. Tokens are lowercased and split on whitespace; punctuation stays attached. The first 8,000 characters are used. Vocabulary comes only from the corpus, plus [EOS] and [UNK]. [BOS] pads the start but is never predicted. The last n − 1 query tokens form the context; an empty query means sentence start. This is a counting demonstration, not a Chinese word segmenter.', '每個非空白行是一個句子。詞元轉為小寫並以空白切分，標點保留在原詞元中；最多使用前 8,000 個字元。詞彙表只由語料建立，再加入 [EOS] 與 [UNK]。[BOS] 用於補齊句首，不是預測目標。查詢最後 n − 1 個詞元構成上下文，空查詢表示句首。這是計數示範，不是中文斷詞器。'),
  related: ['nlp/nlp-01-introduction', 'nlp/nlp-09-language-modeling-transformers'],
  controls: [
    { key: 'corpus', label: L('Training corpus: one sentence per line', '訓練語料：每行一句'), kind: 'textarea', value: 'small robots move\nsmall robots learn\nred robots move\nsmall birds move' },
    { key: 'context', label: L('Context to inspect', '要觀察的上下文'), kind: 'text', value: 'robots' },
    { key: 'order', label: L('Order n', '階數 n'), kind: 'select', value: '2', options: [1, 2, 3].map(n => ({ value: String(n), label: L(`${n}-gram`, `${n}-gram`) })) },
    { key: 'alpha', label: L('Additive smoothing α', '加法平滑化 α'), kind: 'range', value: 0.1, min: 0, max: 2, step: 0.05 },
  ],
  compute(state, lang) {
    const result = ngramDistribution(String(state.corpus ?? ''), String(state.context ?? ''), Number(state.order), Number(state.alpha));
    if (!result.vocabulary.length) return { svg: svg(text(28, 80, pick(lang, 'Add a sentence to build the vocabulary.', '請先加入句子，建立詞彙表。'))), summary: pick(lang, 'An empty corpus has no estimated distribution.', '空語料無法估計分布。'), metrics: [] };
    const rows = result.vocabulary.map((word, i) => ({ word, count: result.counts[i], p: result.probabilities[i] })).sort((a, b) => b.p - a.p || b.count - a.count || a.word.localeCompare(b.word));
    const shown = rows.slice(0, 12);
    const label = result.context.length ? result.context.map(word => short(word, 20)).join(' · ') : pick(lang, '(no context: unigram)', '（無上下文：unigram）');
    const body = text(20, 28, pick(lang, 'Next token', '下一個詞元')) + text(460, 28, pick(lang, 'Count', '次數')) + text(540, 28, 'P(w | h)') + shown.map((row, i) => {
      const y = 58 + i * 29;
      return text(20, y + 4, short(row.word)) + `<rect x="158" y="${y - 13}" width="${row.p * 270}" height="20" fill="var(--plot-a)" opacity="0.85"/>` + text(470, y + 4, row.count) + text(542, y + 4, result.defined ? fmt(row.p, 4) : '—');
    }).join('');
    const omitted = rows.slice(shown.length).reduce((sum, row) => sum + row.p, 0);
    const status = result.defined
      ? pick(lang, 'P(w | h) = (count(h,w) + α) / (count(h) + α|V|).', 'P(w | h) = (count(h,w) + α) / (count(h) + α|V|)。')
      : pick(lang, 'This context has zero observations and α = 0, so the denominator is zero. Increase α to define a distribution.', '此上下文沒有觀測值，且 α = 0，因此分母為零。提高 α 才能定義分布。');
    return {
      svg: svg(body, Math.max(180, 75 + shown.length * 29)),
      summary: `${pick(lang, 'Context', '上下文')}: ${label}. ${status} ${result.unknown.length ? pick(lang, 'Unseen query tokens map to [UNK]; the vocabulary is unchanged.', '未見過的查詢詞元映射為 [UNK]，詞彙表維持不變。') : ''} ${rows.length > 12 ? pick(lang, `Showing 12 of ${rows.length} tokens; omitted probability mass ${fmt(omitted, 4)}.`, `顯示 ${rows.length} 個詞元中的前 12 個；省略的機率總和為 ${fmt(omitted, 4)}。`) : ''}`,
      metrics: [
        { label: pick(lang, 'Context count', '上下文次數'), value: String(result.contextCount) },
        { label: pick(lang, 'Vocabulary size', '詞彙表大小'), value: String(rows.length) },
        { label: pick(lang, 'Probability sum (all tokens)', '機率總和（全部詞元）'), value: result.defined ? fmt(result.probabilities.reduce((a, b) => a + b, 0), 6) : pick(lang, 'Undefined', '未定義') },
      ],
    };
  },
};

const word2vec: Experiment = {
  id: 'word2vec', track: 'nlp',
  title: L('CBOW and Skip-gram: who predicts whom?', 'CBOW 與 Skip-gram：誰預測誰？'),
  description: L('Choose a center token and window. Build the actual training examples, then calculate a full-softmax loss.', '選擇中心詞與視窗，建立實際訓練樣本，再計算完整 softmax 損失。'),
  challenge: L('Select “robots” with radius 2. Switch objectives: how many targets are scored, and which input vectors are averaged?', '選擇「robots」並把半徑設為 2。切換目標函數：計算了幾個目標？哪些輸入向量會取平均？'),
  note: L('Sentence: small red robots move slowly. E is the input table; U is the output table. All two-dimensional vectors shown are fixed, synthetic and untrained. CBOW averages neighbors to predict the center; Skip-gram predicts each neighbor separately from the center. The displayed Skip-gram loss is the mean over these pairs. Full softmax uses all five vocabulary words, with no negative sampling.', '句子：small red robots move slowly。E 是輸入向量表，U 是輸出向量表。顯示的二維向量皆為固定、人工設定且未經訓練的數值。CBOW 將鄰詞向量取平均後預測中心詞；Skip-gram 由中心詞分別預測每個鄰詞。顯示的 Skip-gram 損失是各詞對損失的平均。完整 softmax 使用全部五個詞，未使用負取樣。'),
  related: ['nlp/nlp-03-neural-networks-for-nlp'],
  controls: [
    { key: 'mode', label: L('Prediction objective', '預測目標'), kind: 'select', value: 'cbow', options: [{ value: 'cbow', label: L('CBOW', 'CBOW') }, { value: 'skipgram', label: L('Skip-gram', 'Skip-gram') }] },
    { key: 'center', label: L('Center token', '中心詞'), kind: 'select', value: '2', options: WORD_TOKENS.map((word, i) => ({ value: String(i), label: L(word, word) })) },
    { key: 'radius', label: L('Window radius', '視窗半徑'), kind: 'range', value: 1, min: 1, max: 2, step: 1 },
  ],
  compute(state, lang) {
    const mode = state.mode === 'skipgram' ? 'skipgram' : 'cbow';
    const result = word2vecExample(Number(state.center), Number(state.radius), mode);
    const pairs = mode === 'cbow'
      ? [`mean(${result.neighbors.map(i => WORD_TOKENS[i]).join(', ')}) → ${WORD_TOKENS[result.index]}`]
      : result.neighbors.map(i => `${WORD_TOKENS[result.index]} → ${WORD_TOKENS[i]}`);
    const offset = 45 + pairs.length * 26;
    const body = text(20, 26, pick(lang, 'Prediction examples', '預測樣本')) + pairs.map((pair, i) => text(20, 53 + i * 26, pair)).join('')
      + text(20, offset + 35, pick(lang, 'Word', '詞')) + text(155, offset + 35, 'P(word | input)') + text(366, offset + 35, 'E[word]') + text(496, offset + 35, 'U[word]')
      + WORD_TOKENS.map((word, i) => {
        const y = offset + 68 + i * 34;
        return text(20, y, word) + `<rect x="145" y="${y - 16}" width="${result.probabilities[i] * 180}" height="22" fill="var(--plot-${result.targets.includes(i) ? 'b' : 'a'})"/>` + text(286, y, fmt(result.probabilities[i], 3)) + text(366, y, `[${WORD_E[i].join(', ')}]`) + text(496, y, `[${WORD_U[i].join(', ')}]`);
      }).join('');
    return {
      svg: svg(body, offset + 235),
      summary: `${pick(lang, 'Hidden vector h', '隱藏向量 h')} = [${result.hidden.map(n => fmt(n)).join(', ')}]. ${pick(lang, 'Scores are U[word] · h; probabilities are their softmax. Target rows use the second accent color. Each target contributes −ln P(target | input).', '分數為 U[word] · h，機率由 softmax 得到。目標列以第二種強調色顯示，每個目標貢獻 −ln P(target | input)。')}`,
      metrics: [
        { label: pick(lang, 'Context tokens', '上下文詞元數'), value: String(result.neighbors.length) },
        { label: pick(lang, 'Prediction targets', '預測目標數'), value: String(result.targets.length) },
        { label: pick(lang, 'Mean loss (natural log)', '平均損失（自然對數）'), value: fmt(result.loss, 4) },
      ],
    };
  },
};

const attention: Experiment = {
  id: 'attention', track: 'ml',
  title: L('Attention: which tokens are visible?', '注意力：哪些詞元可見？'),
  description: L('Compare bidirectional and causal attention using the same query, keys and values.', '使用相同的查詢、鍵與值，比較雙向注意力與因果注意力。'),
  challenge: L('Choose the first query in causal mode: its weight must be 1 on itself. Switch to bidirectional mode, then lower the temperature.', '在因果模式選擇第一個查詢：它對自己的權重必須為 1。切換至雙向模式，再降低溫度。'),
  note: L('All Q, K and V vectors are fixed, synthetic two-dimensional numbers, not learned language representations. Scores are Q · K / (√2 τ); τ is an extra temperature control and τ = 1 gives the usual scaling. Causal attention masks positions after the query before softmax. The output is the weighted sum of V. These weights illustrate routing, not an explanation of linguistic meaning.', 'Q、K、V 皆為固定、人工設定的二維向量，並非學得的語言表徵。分數為 Q · K / (√2 τ)；τ 是額外的溫度控制，τ = 1 時為一般縮放方式。因果注意力在 softmax 前遮蔽查詢之後的位置，輸出為 V 的加權和。這些權重用來展示資訊如何聚合，無法解釋語言意義。'),
  related: ['nlp/nlp-09-language-modeling-transformers', 'ml/ml-04-vision-transformers'],
  controls: [
    { key: 'query', label: L('Query token', '查詢詞元'), kind: 'select', value: '2', options: ATTENTION_TOKENS.map((word, i) => ({ value: String(i), label: L(`${i + 1}: ${word}`, `${i + 1}：${word}`) })) },
    { key: 'visibility', label: L('Attention visibility', '注意力可見範圍'), kind: 'select', value: 'causal', options: [{ value: 'causal', label: L('Causal: past + self', '因果：過去與自身') }, { value: 'bidirectional', label: L('Bidirectional: all tokens', '雙向：全部詞元') }] },
    { key: 'temperature', label: L('Temperature τ', '溫度 τ'), kind: 'range', value: 1, min: 0.2, max: 2, step: 0.1 },
  ],
  compute(state, lang) {
    const result = attentionExample(Number(state.query), state.visibility !== 'bidirectional', Number(state.temperature));
    const body = text(20, 27, `Q[${ATTENTION_TOKENS[result.index]}] = [${ATTENTION_Q[result.index].join(', ')}]`)
      + text(20, 64, pick(lang, 'Key token', '鍵詞元')) + text(165, 64, pick(lang, 'Weight', '權重')) + text(353, 64, 'K') + text(463, 64, 'V') + text(552, 64, pick(lang, 'Score', '分數'))
      + ATTENTION_TOKENS.map((word, i) => {
        const y = 101 + i * 37;
        return text(20, y, `${i + 1}. ${word}`) + `<rect x="155" y="${y - 17}" width="${result.weights[i] * 125}" height="23" fill="var(--plot-a)"/>` + text(286, y, fmt(result.weights[i], 3)) + text(350, y, `[${ATTENTION_K[i].join(',')}]`) + text(463, y, `[${ATTENTION_V[i].join(',')}]`) + text(553, y, result.allowed[i] ? fmt(result.scores[i], 2) : '−∞');
      }).join('');
    return {
      svg: svg(body, 290),
      summary: pick(lang, 'The score column includes temperature scaling. A masked score is −∞ and its weight is exactly zero. Changing the mask changes normalization over the remaining keys.', '分數欄包含溫度縮放。被遮蔽位置的分數為 −∞，權重恰為零。改變遮罩後，剩餘鍵的機率會重新正規化。'),
      metrics: [
        { label: pick(lang, 'Visible tokens', '可見詞元數'), value: String(result.allowed.filter(Boolean).length) },
        { label: pick(lang, 'Weight sum', '權重總和'), value: fmt(result.weights.reduce((a, b) => a + b, 0), 6) },
        { label: pick(lang, 'Attention output', '注意力輸出'), value: `[${result.output.map(n => fmt(n, 4)).join(', ')}]` },
      ],
    };
  },
};

export const languageExperiments: Experiment[] = [ngram, word2vec, attention];
