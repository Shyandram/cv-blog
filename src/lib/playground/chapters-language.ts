import { L, pick, num, fmt, svg, text, path, escape } from './types.ts';
import type { Control, Experiment } from './types.ts';
import type { Lang } from '../../i18n/ui';

const range = (key: string, en: string, zh: string, value: number, min: number, max: number, step = 1): Control => ({ key, label: L(en, zh), kind: 'range', value, min, max, step });
const select = (key: string, en: string, zh: string, value: string, options: [string, string, string][]): Control => ({ key, label: L(en, zh), kind: 'select', value, options: options.map(([value, en, zh]) => ({ value, label: L(en, zh) })) });
const metric = (lang: Lang, en: string, zh: string, value: string | number) => ({ label: pick(lang, en, zh), value: String(value) });
const stroke = (color = 'a') => `stroke="var(--plot-${color})" stroke-width="2.5"`;
const line = (x1: number, y1: number, x2: number, y2: number, color = 'grid') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--plot-${color})"/>`;
const dot = (x: number, y: number, r = 4, color = 'a') => `<circle cx="${x}" cy="${y}" r="${r}" fill="var(--plot-${color})"/>`;
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const softmax = (xs: number[]) => { const m = Math.max(...xs); const e = xs.map(x => Math.exp(x - m)); const z = sum(e); return e.map(x => x / z); };
const normal = (x: number, mu = 0, sigma = 1) => Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
function bars(labels: string[], values: number[], max = Math.max(...values, 1), colors?: string[]) {
  return labels.map((label, i) => { const y = 37 + i * 32; return text(12, y + 15, label) + `<rect x="160" y="${y}" width="${Math.max(0, 370 * values[i] / max)}" height="21" fill="var(--plot-${colors?.[i] ?? 'a'})"/>` + text(540, y + 15, fmt(values[i])); }).join('');
}
function curves(series: { values: number[]; color: string }[], yMin: number, yMax: number, labels: string[]) {
  const height = 270;
  return line(45, 230, 610, 230) + line(45, 25, 45, 230) + series.map(({ values, color }) => path(values.map((v, i) => [45 + i * 560 / Math.max(1, values.length - 1), 230 - 195 * (v - yMin) / (yMax - yMin)] as [number, number]), stroke(color)) + (values.length === 1 ? dot(45, 230 - 195 * (values[0] - yMin) / (yMax - yMin), 5, color) : '')).join('') + labels.map((label, i) => text(50 + i * 270, height, label, `fill="var(--plot-${series[i]?.color ?? 'ink'})"`)).join('');
}

export function scopeTruth(reading: number[]) {
  return { eachSome: reading.every(mask => mask !== 0), shared: [1, 2, 4].some(book => reading.every(mask => (mask & book) !== 0)) };
}

const scope: Experiment = {
  id: 'quantifier-scope', track: 'nlp',
  title: L('Does “everyone read a book” mean the same book?', '「每個人都讀了一本書」是同一本嗎？'),
  description: L('Change a small reading relation and evaluate two quantifier orders exactly.', '改變小型閱讀關係，精確判斷兩種量詞順序的真值。'),
  challenge: L('Make “each person read some book” true while “one book was read by everyone” is false. Can the reverse happen?', '讓「每個人都讀過某本書」成立，但「有一本書每個人都讀過」不成立。反過來可能嗎？'),
  note: L('This is a finite truth-condition model, not an automatic interpretation of an ambiguous sentence. A shared book satisfies both readings.', '這是有限情境中的真值條件模型，不會自動解讀歧義句。所有人讀過同一本書時，兩種解讀都成立。'),
  related: ['nlp/nlp-02-linguistic-essentials', 'nlp/nlp-06-sequence-labeling'],
  controls: ['A', 'B', 'C'].map((person, i) => select(`p${i}`, `Person ${person} read`, `${person} 讀過的書`, String(1 << i), [['0', 'Nothing', '沒有'], ['1', 'Book 1', '書 1'], ['2', 'Book 2', '書 2'], ['4', 'Book 3', '書 3'], ['3', 'Books 1 and 2', '書 1 和 2'], ['7', 'All three books', '三本都讀過']])),
  compute(s, lang) {
    const masks = [0, 1, 2].map(i => num(s, `p${i}`)); const t = scopeTruth(masks);
    let body = text(30, 30, pick(lang, 'People → books: an edge means “read”', '人 → 書：連線代表「讀過」'));
    masks.forEach((mask, i) => { [1, 2, 4].forEach((bit, j) => { if (mask & bit) body += line(140, 75 + i * 65, 460, 75 + j * 65, 'a'); }); });
    masks.forEach((_, i) => { body += dot(140, 75 + i * 65, 8) + text(85, 80 + i * 65, 'ABC'[i]); body += dot(460, 75 + i * 65, 8, 'b') + text(485, 80 + i * 65, String(i + 1)); });
    const truth = (b: boolean) => pick(lang, b ? 'True' : 'False', b ? '真' : '假');
    return { svg: svg(body, 260), summary: pick(lang, '∀ person ∃ book allows a different book per person. ∃ book ∀ person requires one shared book; changing quantifier order changes the claim.', '∀ 人 ∃ 書允許每個人讀不同的書；∃ 書 ∀ 人要求有一本共同讀過的書。改變量詞順序會改變命題。'), metrics: [metric(lang, 'Each person: some book', '每個人：至少一本', truth(t.eachSome)), metric(lang, 'One book: everyone', '某一本：所有人', truth(t.shared))] };
  },
};

export function trainBpe(steps: number, input: string) {
  const corpus = [['low', 5], ['lower', 2], ['lowest', 3], ['slow', 2], ['slower', 1]] as [string, number][];
  const words = corpus.map(([word, count]) => ({ pieces: [...word, '▁'], count }));
  const merges: { a: string; b: string; count: number }[] = [];
  const merge = (pieces: string[], a: string, b: string) => { const out: string[] = []; for (let i = 0; i < pieces.length; i++) { if (pieces[i] === a && pieces[i + 1] === b) { out.push(a + b); i++; } else out.push(pieces[i]); } return out; };
  for (let step = 0; step < steps; step++) {
    const counts = new Map<string, number>();
    for (const word of words) for (let i = 0; i < word.pieces.length - 1; i++) { const key = JSON.stringify(word.pieces.slice(i, i + 2)); counts.set(key, (counts.get(key) ?? 0) + word.count); }
    const best = [...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))[0]; if (!best) break;
    const [a, b] = JSON.parse(best[0]) as [string, string]; merges.push({ a, b, count: best[1] });
    words.forEach(w => { w.pieces = merge(w.pieces, a, b); });
  }
  let pieces = [...Array.from(input.trim().slice(0, 30)), '▁'];
  for (const { a, b } of merges) pieces = merge(pieces, a, b);
  return { merges, pieces, trainingTokens: sum(words.map(w => w.pieces.length * w.count)) };
}
const bpe: Experiment = {
  id: 'subword-merges', track: 'nlp', title: L('Build subwords one merge at a time', '逐步合併字元，建立子詞'),
  description: L('Train byte-pair encoding on a tiny weighted vocabulary and apply the learned merge order to a new word.', '在小型加權詞彙表上訓練 BPE，再將學得的合併順序套用到新字詞。'),
  challenge: L('Compare “lower”, “slowest” and “glow”. Which pieces transfer to unseen words, and which merges only compress the training vocabulary?', '比較 lower、slowest 與 glow。哪些子詞能用於未見字詞？哪些合併只是壓縮訓練詞彙？'),
  note: L('Character-level BPE with an explicit end marker ▁; lexicographic tie breaking. This is not a byte tokenizer or a pretrained vocabulary. New characters remain individual pieces; spaces inside the input are treated as characters.', '這是字元層級 BPE，使用詞尾符號 ▁，同分時依字典順序決定。不使用位元組切詞器或預訓練詞彙表。新字元保留為單獨子詞，輸入中的空格也視為字元。'),
  related: ['nlp/nlp-01-introduction', 'nlp/nlp-02-linguistic-essentials', 'nlp/nlp-10-pretrained-bert'],
  controls: [range('merges', 'Merge steps', '合併步數', 4, 0, 12), { key: 'word', kind: 'text', label: L('Apply to one word (first 30 characters)', '套用字詞（前 30 個字元）'), value: 'slowest' }],
  compute(s, lang) {
    const result = trainBpe(num(s, 'merges'), String(s.word));
    const body = text(20, 27, 'low ×5 · lower ×2 · lowest ×3 · slow ×2 · slower ×1') + result.merges.map((m, i) => text(25 + Math.floor(i / 6) * 310, 65 + (i % 6) * 27, `${i + 1}. ${m.a} + ${m.b} → ${m.a + m.b} (${m.count})`)).join('') + text(20, 252, pick(lang, 'Output pieces:', '輸出子詞：')) + text(20, 278, result.pieces.slice(0, 16).join(' | ')) + text(20, 304, result.pieces.slice(16).join(' | '));
    return { svg: svg(body, 335), summary: pick(lang, 'Merge counts come only from the fixed training vocabulary. Encoding applies the learned rules in order; it does not refit them to your input.', '合併次數只由固定訓練詞彙計算。編碼依序套用已學得的規則，不會用你的輸入重新訓練。'), metrics: [metric(lang, 'Learned merges', '學得合併', result.merges.length), metric(lang, 'Training pieces, weighted', '訓練子詞總數（加權）', result.trainingTokens), metric(lang, 'Input pieces', '輸入子詞數', result.pieces.length)] };
  },
};

export const poolingEmbeddings = { dogs: 1, chase: 0.4, cats: 0.7 };
export function poolingExample(reverse: boolean, mode: string, pads: number) {
  const { dogs, chase, cats } = poolingEmbeddings;
  const values = reverse ? [cats, chase, dogs] : [dogs, chase, cats];
  const total = sum(values);
  const weighted = sum(values.map((v, i) => (i + 1) * v));
  const denominator = mode === 'position' ? 6 : mode === 'unmasked' ? 3 + pads : 3;
  const pooled = (mode === 'position' ? weighted : total) / denominator;
  const reference = total / 3;
  return { values, pooled, probability: sigmoid(5 * pooled), reference, referenceProbability: sigmoid(5 * reference), denominator, numerator: mode === 'position' ? weighted : total };
}
const pooling: Experiment = {
  id: 'pooling-order', track: 'nlp', title: L('Does padding change a mean-pooled sentence?', '填補會改變平均池化後的句子表示嗎？'),
  description: L('Pad a three-word sentence with zero vectors and compare a correct masked mean with a mean that counts the padding in its denominator.', '把三個詞的句子補上零向量，比較正確的遮罩平均，與把填補算進分母的平均。'),
  challenge: L('Keep the padding bug selected and drag the padding count from 0 to 9: predict h and P(class 1) before each step, and explain why the classifier drifts toward 0.5 although the sentence never changed. Then switch to the masked mean and confirm that padding cannot matter. Last, try position weights on the reversed sentence: that is the only mode where word order changes the answer.', '保持「填補計入分母」的錯誤設定，把填補數從 0 拖到 9：每一步先預測 h 與 P(類別 1)，並解釋句子明明沒變，分類器為何趨近 0.5。接著切換為遮罩平均，確認填補不再有影響。最後在反轉句子上試位置權重：只有這個模式下，詞序才會改變答案。'),
  note: L('Fixed scalar embeddings: dogs=1, chase=0.4, cats=0.7; classifier p=σ(5h). The masked mean divides by the 3 real tokens, so zero padding adds nothing. The bug divides by 3 + padding. The reference bar always shows the masked mean of the same sentence. A plain mean ignores word order by construction, so swapping the sentence changes nothing there; position weights 1,2,3 are a diagnostic, not a trained syntax model.', '固定純量嵌入：dogs=1、chase=0.4、cats=0.7；分類器 p=σ(5h)。遮罩平均除以 3 個真實詞元，所以零向量填補不會加進任何東西；錯誤版本則除以 3 + 填補數。參考長條永遠是同一個句子的遮罩平均。一般平均依構造就不看詞序，交換句子不會有任何變化；位置權重 1、2、3 只用來診斷，不是訓練過的句法模型。'),
  predict: { question: L('Padding bug selected, with 9 zero-padding tokens added to the 3 real words. Compared with the correct masked mean, the pooled value h becomes', '選擇「填補計入分母」並加入 9 個零向量填補。和正確的遮罩平均相比，池化後的 h 會變成'), options: [{ value: 'half', label: L('half as large', '一半') }, { value: 'quarter', label: L('a quarter as large', '四分之一') }, { value: 'same', label: L('the same', '不變') }], answer: 'quarter', explain: L('The sum of the embeddings is unchanged (zeros add nothing), but it is divided by 3 + 9 = 12 instead of 3, so h shrinks by a factor of 3/12 = 1/4: from 0.7 to 0.175. The classifier output falls from 0.971 to 0.706, toward the undecided value 0.5, even though the sentence is identical.', '嵌入的總和沒有改變（零向量不貢獻任何值），但被除以 3 + 9 = 12 而不是 3，所以 h 縮小為 3/12 = 1/4 倍：從 0.7 變成 0.175。分類器輸出從 0.971 降到 0.706，趨近未決定的 0.5，儘管句子完全相同。') },
  related: ['nlp/nlp-03-neural-networks-for-nlp', 'nlp/nlp-04-lexical-semantics-word-embeddings'],
  controls: [select('order', 'Sentence', '句子', 'forward', [['forward', 'dogs chase cats', 'dogs chase cats'], ['reverse', 'cats chase dogs', 'cats chase dogs']]), select('mode', 'Pooling', '池化方式', 'unmasked', [['mean', 'Mean over real tokens (correct)', '只對真實詞元取平均（正確）'], ['unmasked', 'Mean counting padding (bug)', '平均時計入填補（錯誤）'], ['position', 'Position weights 1,2,3', '位置權重 1、2、3']]), range('pads', 'Zero padding tokens', '零向量填補數', 6, 0, 9)],
  compute(s, lang) {
    const pads = num(s, 'pads'), r = poolingExample(s.order === 'reverse', String(s.mode), pads);
    const names = s.order === 'reverse' ? ['cats', 'chase', 'dogs'] : ['dogs', 'chase', 'cats'];
    let body = text(10, 22, pick(lang, 'Tokens entering the average (dashed = zero padding)', '進入平均的詞元（虛線＝零向量填補）'));
    for (let i = 0; i < 3 + pads; i++) {
      const x = 32 + i * 48, real = i < 3;
      body += `<rect x="${x}" y="34" width="44" height="52" rx="5" fill="${real ? 'var(--plot-grid)' : 'none'}"${real ? '' : ' stroke="var(--plot-muted)" stroke-dasharray="4 3"'}/>` + (real ? text(x + 22, 55, names[i], 'text-anchor="middle"') + text(x + 22, 76, r.values[i], 'text-anchor="middle"') : text(x + 22, 66, '0', 'text-anchor="middle"'));
    }
    const formula = s.mode === 'position' ? `h = Σ(i·e) / 6 = ${fmt(r.numerator)} / 6 = ${fmt(r.pooled)}` : `h = Σe / D = ${fmt(r.numerator)} / ${r.denominator} = ${fmt(r.pooled)}`;
    const dText = s.mode === 'position' ? pick(lang, 'Weights 1+2+3 = 6 in the denominator', '分母為權重總和 1+2+3 = 6') : s.mode === 'unmasked' ? pick(lang, `D = 3 real + ${pads} padding = ${r.denominator} (bug)`, `D = 3 個真實詞元 + ${pads} 個填補 = ${r.denominator}（錯誤）`) : pick(lang, 'D = 3 real tokens (padding not counted)', 'D = 3 個真實詞元（不計入填補）');
    body += text(10, 114, formula) + text(10, 138, dText);
    body += text(10, 174, pick(lang, 'P(class 1) = σ(5h); 0.5 means undecided', 'P(類別 1) = σ(5h)；0.5 代表無法決定'));
    const barX = 250, scale = 320;
    [[pick(lang, 'Masked mean (reference)', '遮罩平均（參考）'), r.referenceProbability, 'a'], [pick(lang, 'This pooling', '目前的池化'), r.probability, 'b']].forEach(([label, p, color], k) => {
      const y = 186 + k * 32;
      body += text(10, y + 16, label as string) + `<rect x="${barX}" y="${y}" width="${fmt((p as number) * scale, 2)}" height="22" fill="var(--plot-${color})"/>` + text(barX + (p as number) * scale + 8, y + 16, fmt(p as number));
    });
    body += `<line x1="${barX + scale / 2}" y1="180" x2="${barX + scale / 2}" y2="254" stroke="var(--plot-ink)" stroke-dasharray="4 3"/>` + text(barX + scale / 2, 274, '0.5', 'text-anchor="middle"');
    return { svg: svg(body, 290), summary: pick(lang, 'Zero padding must not change a masked mean. Counting the padding in the denominator shrinks h by 3 divided by the counted length and pulls P(class 1) toward 0.5, although the sentence is the same. An unweighted mean also ignores word order; only the position-weighted diagnostic tells the two sentences apart.', '正確遮罩下，零向量填補不應改變平均值。把填補算進分母，會讓 h 縮小為 3 除以計入的長度，並把 P(類別 1) 拉向 0.5，儘管句子沒有改變。未加權平均也不看詞序；只有位置加權的診斷設定能分辨兩個句子。'), metrics: [metric(lang, 'Pooled h', '池化 h', fmt(r.pooled)), metric(lang, 'Masked-mean h (reference)', '遮罩平均 h（參考）', fmt(r.reference)), metric(lang, 'P(class 1)', 'P(類別 1)', fmt(r.probability)), metric(lang, 'Reference P(class 1)', '參考 P(類別 1)', fmt(r.referenceProbability)), metric(lang, 'Real / counted length', '實際／計入長度', `3 / ${r.denominator}`)] };
  },
};

export function sequenceDecode(evidence: number, bonus: number, constrained: boolean) {
  const emissions = [[2, 1, -1], [1.2, 0, evidence], [1.5, 0, 0.3], [2, -0.5, -1]];
  const local = emissions.map(row => row.indexOf(Math.max(...row)));
  const valid = (tags: number[]) => tags.every((tag, i) => tag !== 2 || (i > 0 && tags[i - 1] !== 0));
  const score = (tags: number[]) => sum(tags.map((tag, i) => emissions[i][tag] + (i > 0 && tags[i - 1] === 1 && tag === 2 ? bonus : 0)));
  let best: number[] = [], bestScore = -Infinity;
  for (let n = 0; n < 81; n++) { const tags = [0, 1, 2, 3].map(i => Math.floor(n / 3 ** (3 - i)) % 3); if (constrained && !valid(tags)) continue; const value = score(tags); if (value > bestScore) { best = tags; bestScore = value; } }
  return { emissions, local, best, bestScore, localScore: score(local), localValid: valid(local), bestValid: valid(best) };
}
const sequence: Experiment = {
  id: 'sequence-decoding', track: 'nlp', title: L('Local tag choices versus a coherent sequence', '局部標籤選擇與完整序列'),
  description: L('Combine emission scores, transition scores and BIO span constraints to choose a complete tag path.', '結合發射分數、轉移分數與 BIO 片段限制，選出完整標籤路徑。'),
  challenge: L('Raise the score for I at “York”. Can a transition preference change the best path even when the local winner stays fixed?', '提高 York 的 I 分數。即使局部最佳標籤不變，轉移偏好能否改變最佳路徑？'),
  note: L('O=outside, B=begin, I=inside one entity type. Scores are synthetic, not probabilities. All 81 paths are enumerated exactly; dynamic programming would find the same maximum more efficiently on long sequences.', 'O=外部、B=起始、I=內部，只使用一種實體類型。這些合成分數不是機率。此處精確枚舉 81 條路徑；長序列可用動態規劃更有效率地找到相同最大值。'),
  related: ['nlp/nlp-12-syntactic-semantic-tasks', 'nlp/nlp-06-sequence-labeling'],
  controls: [range('evidence', 'York: I emission', 'York：I 發射分數', 0.8, -2, 4, 0.2), range('bonus', 'B → I transition bonus', 'B → I 轉移加分', 2, -2, 3, 0.2), select('constraints', 'Legal BIO spans', '合法 BIO 片段', 'yes', [['yes', 'Enforce', '強制遵守'], ['no', 'Ignore constraints', '忽略限制']])],
  compute(s, lang) { const r = sequenceDecode(num(s, 'evidence'), num(s, 'bonus'), s.constraints === 'yes'); const names = ['New', 'York', 'office', 'opens']; const tags = ['O', 'B', 'I']; let body = names.map((n, i) => text(85 + i * 145, 25, n)).join(''); r.emissions.forEach((row, i) => row.forEach((v, j) => { body += `<rect x="${65 + i * 145}" y="${48 + j * 58}" width="112" height="42" rx="5" fill="var(--plot-${r.best[i] === j ? 'a' : 'grid'})" fill-opacity="${r.best[i] === j ? 0.3 : 1}"/>` + text(82 + i * 145, 75 + j * 58, `${tags[j]}: ${fmt(v)}`); })); body += path(r.best.map((tag, i) => [120 + i * 145, 69 + tag * 58] as [number, number]), `${stroke('b')} stroke-dasharray="5 4"`); body += text(20, 255, pick(lang, 'Local', '局部') + `: ${r.local.map(i => tags[i]).join(' → ')} | ` + pick(lang, 'Global', '全域') + `: ${r.best.map(i => tags[i]).join(' → ')}`); return { svg: svg(body), summary: pick(lang, 'The highlighted cells maximize the total path score, not each token separately. An I tag can only continue a B or I when constraints are enforced.', '反白格子最大化整條路徑的總分，而不是各詞獨立的分數。啟用限制時，I 只能接在 B 或 I 後面。'), metrics: [metric(lang, 'Best total score', '最佳總分', fmt(r.bestScore)), metric(lang, 'Local path score', '局部路徑總分', fmt(r.localScore)), metric(lang, 'Local path legal?', '局部路徑合法？', pick(lang, r.localValid ? 'Yes' : 'No', r.localValid ? '是' : '否'))] }; },
};

const documents = ['robots see red objects', 'red red red objects', 'robots use cameras to see distant objects in a room', 'language models predict words', 'cameras record light', 'red flowers grow'];
export function lexicalRetrieval(query: string, k1: number, b: number) {
  const tokenize = (v: string): string[] => v.toLowerCase().match(/[a-z]+/g) ?? [];
  const docs = documents.map(tokenize); const terms = [...new Set(tokenize(query.slice(0, 200)))]; const avg = sum(docs.map(d => d.length)) / docs.length;
  return docs.map((doc, index) => ({ index, score: sum(terms.map(term => { const tf = doc.filter(t => t === term).length; const df = docs.filter(d => d.includes(term)).length; const idf = Math.log(1 + (docs.length - df + 0.5) / (df + 0.5)); return tf ? idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc.length / avg)) : 0; })) })).sort((a, b) => b.score - a.score || a.index - b.index);
}
const retrieval: Experiment = {
  id: 'retrieval-ranking', track: 'nlp', title: L('Why does this document rank first?', '為何這份文件排在第一？'),
  description: L('Inspect BM25 term saturation and document-length normalization on a fully visible local corpus.', '在完整可見的本地語料上，觀察 BM25 詞頻飽和與文件長度正規化。'),
  challenge: L('Search “red”, then “robots objects”. Change length normalization and saturation; explain each rank swap from document terms.', '先搜尋 red，再搜尋 robots objects。改變長度正規化與飽和程度，根據文件中的詞解釋排名交換。'),
  note: L('Six synthetic documents, lowercase English word tokenization, distinct query terms, positive smoothed IDF. A lexical relevance score is not evidence that a document answers a question.', '六份合成文件，採小寫英文字詞切分、不重複查詢詞與正值平滑 IDF。詞彙相關分數不代表文件足以回答問題。'),
  related: ['nlp/nlp-09-language-modeling-transformers'],
  controls: [{ key: 'query', label: L('Query (English words)', '查詢（英文字詞）'), kind: 'text', value: 'red objects' }, range('k1', 'Term saturation k₁', '詞頻飽和 k₁', 1.2, 0, 3, 0.1), range('b', 'Length normalization b', '長度正規化 b', 0.75, 0, 1, 0.05)],
  compute(s, lang) { const ranked = lexicalRetrieval(String(s.query), num(s, 'k1'), num(s, 'b')); const body = bars(ranked.map(r => `D${r.index + 1}`), ranked.map(r => r.score)) + documents.map((d, i) => text(20, 265 + i * 25, `D${i + 1}: ${d}`)).join(''); return { svg: svg(body, 430), summary: pick(lang, 'IDF rewards terms found in fewer documents. k₁ controls how repeated matches saturate; b controls the penalty for length relative to this corpus mean.', 'IDF 提高少數文件中才出現的詞之權重。k₁ 控制重複命中的飽和程度；b 控制相對於此語料平均長度的懲罰。'), metrics: [metric(lang, 'Top document', '最高分文件', ranked[0].score > 0 ? `D${ranked[0].index + 1}` : pick(lang, 'No lexical matches', '沒有詞彙命中')), metric(lang, 'Top score', '最高分', fmt(ranked[0].score)), metric(lang, 'Matching documents', '命中文件', ranked.filter(d => d.score > 0).length)] }; },
};

export function decodingDistribution(temperature: number, k: number, p: number) {
  const base = softmax([3, 2, 1, 0, -1].map(x => x / temperature)); const top = base.slice(0, k); const z = sum(top); let cumulative = 0, keep = 0;
  for (const v of top) { cumulative += v / z; keep++; if (cumulative >= p - 1e-12) break; }
  const retained = sum(base.slice(0, keep)); const distribution = base.map((v, i) => i < keep ? v / retained : 0);
  return { base, distribution, keep, entropy: -sum(distribution.filter(v => v > 0).map(v => v * Math.log2(v))) };
}
const decoding: Experiment = {
  id: 'decoding-controls', track: 'nlp', title: L('Temperature, top-k and top-p change the distribution', '溫度、top-k 與 top-p 如何改變分布'),
  description: L('Follow fixed logits through temperature scaling, top-k truncation, nucleus truncation and renormalization.', '追蹤固定 logits 經過溫度縮放、top-k 截斷、核心截斷與重新正規化的結果。'),
  challenge: L('Keep temperature fixed and lower top-p. Why can the output entropy change abruptly even though the slider moves continuously?', '固定溫度並降低 top-p。為何滑桿連續移動，輸出熵卻可能突然改變？'),
  note: L('Five synthetic candidate tokens with logits [3,2,1,0,−1]. This applies top-k first, then top-p to its renormalized distribution. No model is trained and no quality score is inferred from entropy.', '五個合成候選詞的 logits 為 [3,2,1,0,−1]。先套用 top-k，再對重新正規化的分布套用 top-p。不訓練模型，也不從熵推論文字品質。'),
  related: ['nlp/nlp-09-language-modeling-transformers', 'nlp/nlp-08-seq2seq-transformers', 'ml/ml-08-autoregressive-vision'],
  predict: { question: L("Hold the temperature at 1 and lower top-p slowly from 1.0. How does the output entropy change?", "把溫度固定為 1，從 1.0 慢慢降低 top-p。輸出熵會怎麼變？"), options: [{ value: "smooth", label: L("Smoothly", "平滑地變化") }, { value: "jumps", label: L("It stays flat, then jumps", "維持不變，然後跳動") }], answer: "jumps", explain: L("Top-p keeps a whole number of tokens, so the kept set changes only when the cumulative mass crosses p. With the default logits the entropy is flat, then steps from 0 to 0.84 near p = 0.65, and again near 0.9.", "top-p 保留的是整數個詞元，只有在累積機率越過 p 時，保留集合才會改變。以預設 logits 來說，熵維持不變，在 p 約 0.65 時從 0 跳到 0.84，在 0.9 附近再跳一次。") },
  controls: [range('temperature', 'Temperature', '溫度', 1, 0.2, 3, 0.1), range('k', 'Top-k candidates', 'Top-k 候選數', 5, 1, 5), range('p', 'Top-p cumulative mass', 'Top-p 累積機率', 0.9, 0.1, 1, 0.05)],
  compute(s, lang) { const r = decodingDistribution(num(s, 'temperature'), num(s, 'k'), num(s, 'p')); let body = text(20, 25, pick(lang, 'Filled: final probabilities · outline: before truncation', '實心：最終機率・外框：截斷前機率')); body += bars(['A', 'B', 'C', 'D', 'E'], r.distribution, 1); body += r.base.map((v, i) => `<rect x="160" y="${37 + i * 32}" width="${370 * v}" height="21" fill="none" stroke="var(--plot-b)"/>`).join(''); return { svg: svg(body, 230), summary: pick(lang, 'Temperature rescales relative logits. Truncation sets excluded candidates to zero and renormalizes the survivors; it cannot improve the information available to the model.', '溫度縮放 logits 的相對差距。截斷將排除候選的機率設為零，再正規化保留候選；它無法增加模型掌握的訊息。'), metrics: [metric(lang, 'Candidates retained', '保留候選數', r.keep), metric(lang, 'Entropy, bits', '熵（位元）', fmt(r.entropy)), metric(lang, 'Probability sum', '機率總和', fmt(sum(r.distribution)))] }; },
};

export function maskingHeads(causal: boolean, clue: number, head: string) {
  // Four fixed token vectors; the masked position has no content vector.
  const vectors = [[0.1, 0.2], [0.2, 0.1], [0, 0], [clue, 0.5]];
  const visible = vectors.map((_, i) => !causal || i < 2);
  const context = [0, 1].map(j => sum(vectors.filter((_, i) => visible[i] && i !== 2).map(v => v[j])) / visible.filter((v, i) => v && i !== 2).length);
  const logits = head === 'masked' ? [3 * context[0], -3 * context[0]] : [2 * context[0] + context[1], -2 * context[0] - context[1]];
  return { vectors, visible, context, probabilities: softmax(logits) };
}
const masking: Experiment = {
  id: 'masked-objectives', track: 'nlp', title: L('What may a masked predictor see?', '遮罩預測器能看見哪些位置？'),
  description: L('Change a right-context clue, hide it causally, then attach different output heads to the same small contextual vector.', '改變右側語境線索，以因果遮罩隱藏它，再將不同輸出頭接到同一個小型語境向量。'),
  challenge: L('Change the right-hand clue under each mask. When must the masked-position prediction remain identical, and why?', '在兩種遮罩下分別改變右側線索。哪一種情況的遮罩位置預測必須保持相同？為什麼？'),
  note: L('A hand-built uniform context encoder and two linear heads illustrate BERT-style visibility and task interfaces. These are not BERT embeddings or pretrained predictions. The sequence head reads this toy pooled context, not an actual [CLS] representation.', '此處以手工設定的均勻語境編碼器與兩個線性輸出頭，示範 BERT 類型的可見範圍及任務介面。這些不是 BERT 嵌入或預訓練預測。序列輸出頭讀取玩具池化語境，不是真正的 [CLS] 表徵。'),
  related: ['nlp/nlp-09-language-modeling-transformers', 'nlp/nlp-10-pretrained-bert', 'nlp/nlp-11-classification-bert', 'nlp/nlp-07-attention-transformers'],
  controls: [select('mask', 'Visible context', '可見語境', 'bidirectional', [['bidirectional', 'Left and right, target masked', '左右語境，目標已遮罩'], ['causal', 'Strictly left of target', '僅目標左側']]), range('clue', 'Right-context feature', '右側語境特徵', 1, -1, 1, 0.1), select('head', 'Output head', '輸出頭', 'masked', [['masked', 'Masked token: candidates A/B', '遮罩詞：候選 A／B'], ['sequence', 'Sequence label: classes +/−', '序列標籤：類別 +／−']])],
  compute(s, lang) { const r = maskingHeads(s.mask === 'causal', num(s, 'clue'), String(s.head)); let body = (lang === 'zh-tw' ? ['左 1', '左 2', '[MASK]', '右側線索'] : ['left 1', 'left 2', '[MASK]', 'right clue']).map((name, i) => `<rect x="${15 + i * 155}" y="35" width="140" height="65" rx="5" fill="var(--plot-${r.visible[i] && i !== 2 ? 'a' : 'grid'})" fill-opacity="0.3"/>` + text(25 + i * 155, 62, name) + text(25 + i * 155, 87, `[${r.vectors[i].map(v => fmt(v)).join(', ')}]`)).join(''); body += text(25, 140, `h = [${r.context.map(v => fmt(v)).join(', ')}]`); const labels = s.head === 'masked' ? (lang === 'zh-tw' ? ['詞元 A', '詞元 B'] : ['Token A', 'Token B']) : (lang === 'zh-tw' ? ['類別 +', '類別 −'] : ['Class +', 'Class −']); body += `<g transform="translate(0 135)">${bars(labels, r.probabilities, 1)}</g>`; return { svg: svg(body, 270), summary: pick(lang, 'The target content is always absent. A bidirectional objective can use right context; a causal next-token objective cannot. An output head maps representations to the labels required by a task.', '目標內容始終不可見。雙向目標可利用右側語境，因果次詞預測則不行。輸出頭將表徵映射為任務所需標籤。'), metrics: [metric(lang, 'Context positions used', '使用語境位置數', r.visible.filter((v, i) => v && i !== 2).length), metric(lang, 'First output probability', '第一個輸出機率', fmt(r.probabilities[0])), metric(lang, 'Output probabilities sum', '輸出機率總和', fmt(sum(r.probabilities)))] }; },
};

export function finiteDifference(x: number, exponent: number, central: boolean) {
  const h = 10 ** exponent; const exact = Math.exp(x); const estimate = central ? (Math.exp(x + h) - Math.exp(x - h)) / (2 * h) : (Math.exp(x + h) - Math.exp(x)) / h;
  return { h, exact, estimate, relative: Math.abs(estimate - exact) / exact };
}
const gradients: Experiment = {
  id: 'gradient-check', track: 'ml', title: L('Why a smaller finite-difference step can be worse', '為何有限差分步長越小反而越差？'),
  description: L('Check the derivative of exp(x), separating truncation error from floating-point cancellation.', '檢查 exp(x) 的導數，區分截斷誤差與浮點相減造成的消去誤差。'),
  challenge: L('Sweep h from 10⁻¹ to 10⁻¹⁶. Find the useful range for forward and central differences rather than choosing the smallest step.', '將 h 從 10⁻¹ 掃描至 10⁻¹⁶。找出前向差分與中央差分的有效區間，而非直接選最小步長。'),
  note: L('Computed with JavaScript binary64 arithmetic; exact means the analytic derivative exp(x) evaluated in the same arithmetic. A zero measured error is rounded agreement, not proof of exact real arithmetic.', '使用 JavaScript binary64 運算；此處「解析值」是以相同精度計算的解析導數 exp(x)。測得零誤差只代表捨入後相同，不是實數運算完全精確的證明。'),
  related: ['ml/ml-01c-numerical-computation', 'ml/ml-02-neural-networks'],
  controls: [range('x', 'Evaluation point x', '計算位置 x', 1, -2, 2, 0.1), range('exponent', 'log₁₀(step h)', 'log₁₀（步長 h）', -5, -16, -1), select('scheme', 'Difference formula', '差分公式', 'central', [['central', 'Central difference', '中央差分'], ['forward', 'Forward difference', '前向差分']])],
  compute(s, lang) { const r = finiteDifference(num(s, 'x'), num(s, 'exponent'), s.scheme === 'central'); const errors = Array.from({ length: 61 }, (_, i) => Math.log10(Math.max(1e-17, finiteDifference(num(s, 'x'), -16 + i / 4, s.scheme === 'central').relative))); const body = curves([{ values: errors, color: 'a' }], -17, 1, ['log₁₀(h): −16 → −1']) + dot(45 + (num(s, 'exponent') + 16) / 15 * 560, 230 - 195 * (Math.log10(Math.max(1e-17, r.relative)) + 17) / 18, 6, 'b') + text(50, 20, pick(lang, 'y: log₁₀(relative error), floor −17', 'y：log₁₀（相對誤差），下限 −17')); return { svg: svg(body), summary: pick(lang, 'Large h introduces approximation error. Very small h subtracts nearly identical floating-point values, losing useful digits. Central differences cancel the leading truncation term but still suffer roundoff.', 'h 太大會產生近似誤差；h 太小則會相減兩個極接近的浮點值，流失有效位數。中央差分消去領先的截斷項，但仍受捨入誤差影響。'), metrics: [metric(lang, 'Analytic derivative', '解析導數', fmt(r.exact, 8)), metric(lang, 'Numerical derivative', '數值導數', fmt(r.estimate, 8)), metric(lang, 'Relative error', '相對誤差', r.relative.toExponential(3))] }; },
};

export function mlpProbability(x: number, y: number, activation: string, gain: number, bias: number) {
  const act = (v: number) => activation === 'relu' ? Math.max(0, v) : activation === 'tanh' ? Math.tanh(v) : v;
  const h1 = act(gain * (x + y)), h2 = act(gain * (x + y - 1));
  return sigmoid(4 / gain * (h1 - 2 * h2) - 2 + bias);
}
const network: Experiment = {
  id: 'nonlinear-network', track: 'ml', title: L('Two hidden units can bend a decision region', '兩個隱藏單元如何彎折決策區域'),
  description: L('Compare a fixed two-unit network with and without nonlinear activations on the XOR corners.', '在 XOR 四個角點上，比較固定雙單元網路使用與不使用非線性激活的差異。'),
  challenge: L('Start with the linear activation and try every bias: can all four corners be right at once? Then switch to ReLU and tune the gain and bias. Why does the nonlinearity change the answer?', '先用線性激活，試試各種偏差：四個角點能同時分類正確嗎？再切換為 ReLU，調整倍率與偏差。為何非線性會改變答案？'),
  note: L('Weights are hand constructed, not trained: h₁=a(g(x+y)), h₂=a(g(x+y−1)), p=σ(4(h₁−2h₂)/g−2+b). Color is P(class 1); circles show known corner labels.', '權重由手工設定而非訓練：h₁=a(g(x+y))、h₂=a(g(x+y−1))、p=σ(4(h₁−2h₂)/g−2+b)。顏色為 P(類別 1)，圓圈標示已知角點標籤。'),
  related: ['ml/ml-02-neural-networks', 'nlp/nlp-03-neural-networks-for-nlp'],
  predict: { question: L("With the linear activation selected, can some output bias classify all four XOR corners correctly?", "選擇線性激活時，有沒有任何輸出偏差能讓四個 XOR 角點都分類正確？"), options: [{ value: "yes", label: L("Yes, with the right bias", "可以，只要偏差合適") }, { value: "no", label: L("No, no bias works", "不行，沒有任何偏差可行") }], answer: "no", explain: L("A linear activation collapses the network to one linear score that falls steadily as x + y grows. XOR needs the middle value of x + y to be class 1 and both ends to be class 0, which a single threshold on that score cannot do. At best three of the four corners are right.", "線性激活會讓整個網路縮成單一線性分數，它隨 x + y 增加而單調下降。XOR 需要 x + y 的中間值是類別 1、兩端都是類別 0，單一門檻做不到。最好的情況也只能答對四個角點中的三個。") },
  controls: [select('activation', 'Activation a', '激活函數 a', 'linear', [['linear', 'Linear', '線性'], ['relu', 'ReLU', 'ReLU'], ['tanh', 'tanh', 'tanh']]), range('gain', 'Hidden gain g', '隱藏層倍率 g', 1, 0.5, 4, 0.25), range('bias', 'Output bias b', '輸出偏差 b', 0, -3, 3, 0.25)],
  compute(s, lang) { const probability = (x: number, y: number) => mlpProbability(x, y, String(s.activation), num(s, 'gain'), num(s, 'bias')); let body = ''; for (let iy = 0; iy < 24; iy++) for (let ix = 0; ix < 24; ix++) { const p = probability(ix / 23, iy / 23); body += `<rect x="${185 + ix * 10}" y="${260 - iy * 10}" width="10.2" height="10.2" fill="var(--plot-a)" fill-opacity="${0.08 + 0.92 * p}"/>`; } const corners = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]; corners.forEach(([x, y, label]) => { body += dot(190 + x * 230, 265 - y * 230, 8, label ? 'b' : 'c') + text(450, 55 + corners.findIndex(c => c[0] === x && c[1] === y) * 52, `${x},${y}: ${fmt(probability(x, y))}`); }); body += text(185, 300, pick(lang, 'x: 0 → 1; y: 0 → 1 upward', 'x：0 → 1；y：0 → 1（向上）')); return { svg: svg(body, 330), summary: pick(lang, 'A linear hidden layer collapses to one affine decision boundary. Nonlinear hidden units create pieces that can assign the two opposite XOR corners to the same class.', '線性隱藏層會合併成單一仿射決策邊界。非線性單元組合出分段區域，能把 XOR 的兩個對角點分到同一類。'), metrics: [metric(lang, 'Correct corners', '正確角點', `${corners.filter(([x, y, label]) => Number(probability(x, y) >= 0.5) === label).length} / 4`), metric(lang, 'P(1 | 0,1)', 'P(1 | 0,1)', fmt(probability(0, 1))), metric(lang, 'P(1 | 1,1)', 'P(1 | 1,1)', fmt(probability(1, 1)))] }; },
};

export function selectiveMemory(retain: number, distractor: number, selective: boolean) {
  const inputs = [1, 0.6, -0.5, 0.3, -0.8, 0.5, 0.2, -0.4, -1, 0.7, -0.5, 0.4].map((v, i) => i === 0 || i === 8 ? v : v * distractor);
  let h = 0, derivative = 1; const states: number[] = []; const gates: number[] = [];
  inputs.forEach((x, i) => { const a = selective && i !== 0 && i !== 8 ? 1 : retain; h = a * h + (1 - a) * x; if (i > 0) derivative *= a; states.push(h); gates.push(a); });
  return { inputs, states, gates, derivative };
}
const memory: Experiment = {
  id: 'selective-memory', track: 'ml', title: L('Selective state updates: remember the useful inputs', '選擇性狀態更新：記住重要輸入'),
  description: L('Compare a fixed recurrent decay with input-dependent write decisions while distractors pass through a sequence.', '在序列加入干擾值，比較固定遞迴衰減與依輸入決定的寫入行為。'),
  challenge: L('Compare the two policies: with selective updates the state written at position 1 survives the distractors, with the same update at every step the distractors overwrite it. Then raise the distractor amplitude and see which state changes. Finally set retention to one: a state that never changes keeps its old value perfectly, so nothing new can be written either. Why is that not memory?', '比較兩種策略：選擇性更新下，位置 1 寫入的狀態能撐過干擾；每一步使用相同更新則會被干擾覆寫。接著提高干擾振幅，看哪些狀態會改變。最後把保留係數設為一：永不改變的狀態會完美保留舊值，所以也無法寫入任何新資訊。為什麼這不算記憶？'),
  note: L('Scalar recurrence hₜ=aₜhₜ₋₁+(1−aₜ)xₜ with h₀=0. Retention aₜ=1 keeps the old state, aₜ=0 replaces it with the input. Relevant positions 1 and 9 are provided by an oracle; in the selective policy every other position has aₜ=1. With retention 1 at the relevant positions the state stays at its starting value 0: perfect preservation, nothing written. This illustrates gating and Mamba’s input-selectivity motivation; it is not an implementation of Mamba, LSTM or GRU.', '純量遞迴 hₜ=aₜhₜ₋₁+(1−aₜ)xₜ，h₀=0。保留係數 aₜ=1 表示保留舊狀態，aₜ=0 表示以輸入取代。重要位置 1 與 9 由事先標記提供；選擇性策略在其他位置的 aₜ 皆為 1。若在重要位置把保留係數設為 1，狀態會停在起始值 0：完美保留，卻沒有寫入任何東西。此處示範閘控與 Mamba 輸入選擇性的動機，不是 Mamba、LSTM 或 GRU 的實作。'),
  predict: { question: L('Policy = skip marked distractors, and you set retention a to 1 at the write positions (1 and 9). What is the state after the last position?', '策略為「略過已標記干擾」，並把寫入位置（1 與 9）的保留係數 a 設為 1。最後一個位置之後，狀態是多少？'), options: [{ value: 'zero', label: L('0: nothing was ever written', '0：從未寫入任何東西') }, { value: 'first', label: L('1: the first important input is kept', '1：保留了第一個重要輸入') }, { value: 'last', label: L('−1: the second important input wins', '−1：第二個重要輸入勝出') }], answer: 'zero', explain: L('With a = 1 the update is hₜ = 1·hₜ₋₁ + 0·xₜ, so no input ever enters the state and it stays at its starting value 0 at every position. The derivative ∂h₁₂/∂h₁ is exactly 1, the best possible, but there is nothing in h₁ to remember. Try a = 0.25: the state is 0.75 after position 1, stays 0.75 through the distractors, and ends at −0.5625 after position 9.', '當 a = 1，更新式為 hₜ = 1·hₜ₋₁ + 0·xₜ，沒有任何輸入進入狀態，狀態在每個位置都停在起始值 0。導數 ∂h₁₂/∂h₁ 恰為 1，已是最佳，但 h₁ 裡根本沒有東西可記。改試 a = 0.25：位置 1 之後狀態為 0.75，撐過干擾後仍是 0.75，位置 9 之後變成 −0.5625。') },
  related: ['ml/ml-03a-recurrent-recursive-networks', 'ml/ml-04-vision-transformers', 'nlp/nlp-05-recurrent-neural-networks'],
  controls: [range('retain', 'Retention a when writing', '寫入時的保留係數 a', 0.25, 0, 1, 0.05), range('noise', 'Distractor amplitude', '干擾振幅', 1, 0, 2, 0.1), select('mode', 'State update policy', '狀態更新策略', 'selective', [['selective', 'Skip marked distractors', '略過已標記干擾'], ['fixed', 'Same update at every step', '每一步使用相同更新']])],
  compute(s, lang) {
    const r = selectiveMemory(num(s, 'retain'), num(s, 'noise'), s.mode === 'selective');
    const X = (i: number) => 90 + i * 500 / 11, Y = (v: number) => 148 - 84 * v / 1.7;
    let body = text(10, 22, 'hₜ = aₜ·hₜ₋₁ + (1−aₜ)·xₜ');
    body += `<line x1="22" y1="46" x2="52" y2="46" stroke="var(--plot-b)" stroke-width="2.5" stroke-dasharray="5 3"/>` + text(58, 51, pick(lang, 'Input x', '輸入 x'));
    body += `<line x1="170" y1="46" x2="200" y2="46" stroke="var(--plot-a)" stroke-width="2.5"/>` + text(206, 51, pick(lang, 'State h (the memory)', '狀態 h（記憶）'));
    body += `<rect x="390" y="38" width="26" height="16" fill="var(--plot-c)" opacity=".25"/>` + text(422, 51, pick(lang, 'Important input', '重要輸入'));
    [0, 8].forEach(i => { body += `<rect x="${X(i) - 18}" y="64" width="36" height="168" fill="var(--plot-c)" opacity=".18"/>`; });
    [1, 0, -1].forEach(v => { body += line(70, Y(v), 610, Y(v), v === 0 ? 'muted' : 'grid') + text(60, Y(v) + 5, v === -1 ? '−1' : String(v), 'text-anchor="end"'); });
    body += path(r.inputs.map((v, i) => [X(i), Y(v)] as [number, number]), `${stroke('b')} stroke-dasharray="5 3"`);
    body += path(r.states.map((v, i) => [X(i), Y(v)] as [number, number]), stroke('a'));
    r.states.forEach((v, i) => { body += dot(X(i), Y(v), 3.5, 'a'); });
    r.inputs.forEach((_, i) => { body += text(X(i), 252, String(i + 1), 'text-anchor="middle"'); });
    body += text(340, 274, pick(lang, 'Position t in the sequence', '序列中的位置 t'), 'text-anchor="middle"');
    body += text(10, 302, pick(lang, 'Retention aₜ (dark = keep old state, light = overwrite)', '保留係數 aₜ（深＝保留舊狀態，淺＝覆寫）'));
    r.gates.forEach((a, i) => { body += `<rect x="${X(i) - 19}" y="312" width="38" height="26" fill="var(--plot-ink)" fill-opacity="${fmt(a, 3)}" stroke="var(--plot-muted)"/>`; });
    body += `<text transform="rotate(-90 22 148)" x="22" y="148" text-anchor="middle">${escape(pick(lang, 'Value of x and h', 'x 與 h 的數值'))}</text>`;
    return { svg: svg(body, 352), summary: pick(lang, 'Fixed decay trades persistence against interference at every step. Selection can preserve state through irrelevant inputs, but the gate must identify relevance correctly; the oracle here removes that learning problem. Retention 1 preserves perfectly and writes nothing.', '固定衰減在每一步都取捨持續記憶與干擾。選擇機制能跨越無關輸入保留狀態，但必須正確辨認重要性；此處事先標記省略了這項學習問題。保留係數為 1 時完美保留，卻什麼也寫不進去。'), metrics: [metric(lang, 'State h₁ → h₈ (before the second write)', '狀態 h₁ → h₈（第二次寫入前）', `${fmt(r.states[0])} → ${fmt(r.states[7])}`), metric(lang, 'Final state', '最終狀態', fmt(r.states.at(-1)!)), metric(lang, '∂h₁₂ / ∂h₁', '∂h₁₂ / ∂h₁', fmt(r.derivative, 6)), metric(lang, 'Largest |state|', '狀態絕對值最大值', fmt(Math.max(...r.states.map(Math.abs)))), metric(lang, 'Write opportunities', '可寫入位置數', r.gates.filter(a => a < 1).length)] };
  },
};

export function contrastiveExample(positive: number, negative: number, temperature: number, collapsed: boolean) {
  const angles = collapsed ? [0, 0, 0, 0] : [positive, negative, 120, -100]; const cosine = angles.map(a => Math.cos(a * Math.PI / 180)); const probabilities = softmax(cosine.map(c => c / temperature)); return { angles, cosine, probabilities, loss: -Math.log(probabilities[0]) };
}
const contrastive: Experiment = {
  id: 'contrastive-alignment', track: 'ml', title: L('Alignment, hard negatives and collapsed features', '對齊、困難負例與特徵坍縮'),
  description: L('Move paired and unpaired unit vectors, then inspect the normalized matching loss used by contrastive learning.', '移動配對與未配對單位向量，觀察對比學習使用的正規化匹配損失。'),
  challenge: L('Bring a negative closer to the anchor than the positive. Does a lower temperature help the correct pair? Then collapse all vectors.', '讓負例比正例更靠近錨點。降低溫度會幫助正確配對嗎？接著讓所有向量坍縮。'),
  note: L('One anchor, one positive and three negative candidates; vectors are controlled directly rather than trained. The same matching geometry explains image–text retrieval, but these are not CLIP features or semantic predictions.', '一個錨點、一個正例與三個負例；直接控制向量，不進行訓練。相同匹配幾何可解釋影像與文字檢索，但這些不是 CLIP 特徵或語意預測。'),
  related: ['ml/ml-05-representation-learning', 'ml/ml-09-foundation-models', 'nlp/nlp-04-lexical-semantics-word-embeddings'],
  controls: [range('positive', 'Positive angle, degrees', '正例角度', 20, 0, 180, 5), range('negative', 'Hard negative angle', '困難負例角度', 60, -180, 180, 5), range('temperature', 'Temperature', '溫度', 0.2, 0.05, 1, 0.05), select('collapse', 'Feature geometry', '特徵幾何', 'no', [['no', 'Use selected angles', '使用所選角度'], ['yes', 'Collapse all vectors', '全部向量坍縮']])],
  compute(s, lang) { const r = contrastiveExample(num(s, 'positive'), num(s, 'negative'), num(s, 'temperature'), s.collapse === 'yes'); let body = `<circle cx="160" cy="140" r="100" fill="none" stroke="var(--plot-grid)"/>` + line(160, 140, 260, 140, 'c') + text(268, 145, pick(lang, 'anchor', '錨點')); r.angles.forEach((a, i) => { const x = 160 + 100 * Math.cos(a * Math.PI / 180), y = 140 - 100 * Math.sin(a * Math.PI / 180); body += line(160, 140, x, y, i === 0 ? 'a' : 'b') + dot(x, y, 5, i === 0 ? 'a' : 'b') + text(x - 10, y - 12 - (s.collapse === 'yes' ? i * 18 : 0), i === 0 ? '+' : `−${i}`); }); body += r.probabilities.map((p, i) => text(335, 50 + i * 51, `${i === 0 ? '+' : '−' + i}: ${fmt(p)}`) + `<rect x="335" y="${59 + i * 51}" width="${260 * p}" height="16" fill="var(--plot-${i === 0 ? 'a' : 'b'})"/>`).join(''); return { svg: svg(body), summary: pick(lang, 'Loss = −log(exp(sim⁺/τ) / Σ exp(sim/τ)). Temperature sharpens whichever candidate already has the highest similarity. Collapsed unit vectors give equal probabilities and loss log(4).', '損失 = −log(exp(sim⁺/τ) / Σ exp(sim/τ))。溫度會強化原本相似度最高的候選。坍縮的單位向量產生相等機率與 log(4) 損失。'), metrics: [metric(lang, 'Positive probability', '正例機率', fmt(r.probabilities[0])), metric(lang, 'Matching loss, nats', '匹配損失（nat）', fmt(r.loss)), metric(lang, 'Positive cosine', '正例餘弦值', fmt(r.cosine[0]))] }; },
};

export function vaeObjective(mu: number, sigma: number, beta: number) {
  const kl = 0.5 * (mu * mu + sigma * sigma - 1 - Math.log(sigma * sigma)); const nll = 0.5 * ((1 - mu) ** 2 + sigma * sigma) / 0.25 + Math.log(0.5 * Math.sqrt(2 * Math.PI)); return { kl, nll, total: nll + beta * kl };
}
const vae: Experiment = {
  id: 'vae-tradeoff', track: 'ml', title: L('A VAE trades reconstruction against prior agreement', 'VAE 如何取捨重建與先驗一致性'),
  description: L('Move and widen a Gaussian posterior, then compute expected decoder loss and the exact KL penalty.', '移動並調整高斯後驗寬度，計算期望解碼損失與精確 KL 懲罰。'),
  challenge: L('Set q(z|x) equal to the prior. Why is KL zero but reconstruction still poor for x=1? Can shrinking σ improve one term while worsening the other?', '將 q(z|x) 設為先驗。為何 KL 為零，但 x=1 的重建仍差？縮小 σ 能否改善一項卻使另一項惡化？'),
  note: L('One observation x=1, q=N(μ,σ²), prior=N(0,1), fixed decoder p(x|z)=N(z,0.5²). Expected negative log likelihood is analytic; no samples or network training. β=1 gives the negative evidence lower bound for this model; other β values are weighted objectives.', '單一觀察 x=1、q=N(μ,σ²)、先驗=N(0,1)，固定解碼器 p(x|z)=N(z,0.5²)。期望負對數概似以解析式計算，不抽樣也不訓練網路。β=1 時是此模型的負證據下界；其他 β 是加權目標。'),
  related: ['ml/ml-06-generative-models', 'ml/ml-05a-factor-models-autoencoders'],
  controls: [range('mu', 'Posterior mean μ', '後驗平均 μ', 0.8, -2, 2, 0.1), range('sigma', 'Posterior standard deviation σ', '後驗標準差 σ', 0.5, 0.1, 2, 0.1), range('beta', 'KL weight β', 'KL 權重 β', 1, 0, 4, 0.1)],
  compute(s, lang) { const r = vaeObjective(num(s, 'mu'), num(s, 'sigma'), num(s, 'beta')); const xs = Array.from({ length: 121 }, (_, i) => -6 + i / 10); return { svg: svg(curves([{ values: xs.map(x => normal(x)), color: 'b' }, { values: xs.map(x => normal(x, num(s, 'mu'), num(s, 'sigma'))), color: 'a' }], 0, Math.max(0.45, normal(num(s, 'mu'), num(s, 'mu'), num(s, 'sigma')) * 1.05), [pick(lang, 'Prior N(0,1)', '先驗 N(0,1)'), pick(lang, 'Posterior q(z|x)', '後驗 q(z|x)')]) + text(50, 20, pick(lang, 'z: −6 → 6; y: probability density', 'z：−6 → 6；y：機率密度'))), summary: pick(lang, 'Reconstruction favors latent values near x=1 and a narrow posterior. KL penalizes both moving away from zero and changing variance away from one. Prior agreement alone does not guarantee useful reconstruction.', '重建偏好接近 x=1 的潛在值與較窄後驗。KL 同時懲罰平均偏離零及變異數偏離一。只符合先驗不保證有用的重建。'), metrics: [metric(lang, 'Expected decoder NLL', '期望解碼負對數概似', fmt(r.nll)), metric(lang, 'KL(q || prior), nats', 'KL(q || 先驗)（nat）', fmt(r.kl)), metric(lang, 'NLL + β KL', '負對數概似 + β KL', fmt(r.total))] }; },
};

export function flowTransform(z: number, scale: number, bend: number) {
  const x = scale * z + bend * Math.tanh(z); const jacobian = scale + bend * (1 - Math.tanh(z) ** 2); return { x, jacobian, density: normal(z) / jacobian };
}
const flow: Experiment = {
  id: 'flow-density', track: 'ml', title: L('Stretch space, change density: the Jacobian matters', '拉伸空間會改變密度：Jacobian 的作用'),
  description: L('Push a Gaussian through an invertible nonlinear map and preserve probability mass with its derivative.', '將高斯分布通過可逆非線性映射，利用導數保持機率質量。'),
  challenge: L('Increase the nonlinear stretch near zero. Why does the output density at zero fall even though zero still maps to zero?', '增加零附近的非線性拉伸。為何零仍映射到零，該處輸出密度卻下降？'),
  note: L('One-dimensional normalizing flow x=az+b tanh(z), with a>0 and b≥0. The derivative is always positive, so the map is invertible. Curves are computed from the exact change-of-variables formula, without fitting a model.', '一維正規化流 x=az+b tanh(z)，其中 a>0、b≥0。導數始終為正，因此映射可逆。曲線由精確變數變換公式計算，不擬合模型。'),
  related: ['ml/ml-06-generative-models', 'ml/ml-01b-probability-information'],
  controls: [range('scale', 'Linear scale a', '線性縮放 a', 1, 0.3, 2, 0.1), range('bend', 'Nonlinear stretch b', '非線性拉伸 b', 1, 0, 2, 0.1)],
  compute(s, lang) { const scale = num(s, 'scale'), bend = num(s, 'bend'); const xs = Array.from({ length: 161 }, (_, i) => -4 + i * 0.05); const transformed = xs.map(z => flowTransform(z, scale, bend)); const ymax = Math.max(0.42, ...transformed.map(r => r.density)); const map = (x: number, y: number): [number, number] => [320 + x * 27, 240 - y / ymax * 190]; const area = sum(transformed.slice(1).map((r, i) => (r.x - transformed[i].x) * (r.density + transformed[i].density) / 2)); const body = line(45, 240, 610, 240) + path(xs.map(z => map(z, normal(z))), stroke('b')) + path(transformed.map(r => map(r.x, r.density)), stroke('a')) + text(40, 280, pick(lang, 'Output p(x)', '輸出 p(x)'), 'fill="var(--plot-a)"') + text(320, 280, pick(lang, 'Base p(z)', '基底 p(z)'), 'fill="var(--plot-b)"') + text(40, 20, pick(lang, 'Common x-axis: −10 → 10; y: density', '共同 x 軸：−10 → 10；y：密度')); return { svg: svg(body, 310), summary: pick(lang, 'pₓ(f(z)) = p_z(z) / |f′(z)|. A local stretch spreads the same probability over more output space. Omitting the Jacobian would change the total probability mass.', 'pₓ(f(z)) = p_z(z) / |f′(z)|。局部拉伸將相同機率分散到更大的輸出空間；省略 Jacobian 會改變總機率質量。'), metrics: [metric(lang, 'Derivative at zero', '零點導數', fmt(scale + bend)), metric(lang, 'Output density at zero', '零點輸出密度', fmt(flowTransform(0, scale, bend).density)), metric(lang, 'Numerical area (±4 base σ)', '數值面積（基底 ±4σ）', fmt(area, 5))] }; },
};

export function transportSolve(initial: number, rate: number, steps: number, midpoint: boolean) {
  let x = initial; const trajectory = [x]; const dt = 1 / steps; const field = (v: number) => rate * v + 1;
  for (let i = 0; i < steps; i++) { x += dt * field(midpoint ? x + dt * field(x) / 2 : x); trajectory.push(x); }
  const exact = Math.exp(rate) * initial + (Math.abs(rate) < 1e-12 ? 1 : Math.expm1(rate) / rate);
  return { trajectory, exact, error: Math.abs(x - exact) };
}
const transport: Experiment = {
  id: 'transport-steps', track: 'ml', title: L('A correct velocity field still needs a numerical solver', '正確速度場仍需要數值求解器'),
  description: L('Transport particles through a known velocity field and compare integration error at equal step counts.', '讓粒子通過已知速度場，比較相同步數下的積分誤差。'),
  challenge: L('Set the rate to zero: why is one Euler step exact? Then use a nonzero rate and compare steps with actual field evaluations.', '將變化率設為零：為何一步 Euler 就精確？再使用非零變化率，比較步數與實際速度場評估次數。'),
  note: L('Known field v(x,t)=λx+1, integrated from t=0 to 1. This isolates solver error relevant to flow-matching sampling; no velocity network is trained and no generation quality is measured. The dashed paths use the analytic solution.', '已知速度場 v(x,t)=λx+1，從 t=0 積分到 1。此處隔離流匹配取樣中的求解器誤差，不訓練速度網路，也不衡量生成品質。虛線使用解析解。'),
  related: ['ml/ml-07-diffusion-flow-matching', 'ml/ml-01c-numerical-computation'],
  controls: [range('rate', 'Field slope λ', '速度場斜率 λ', 0.8, -1, 1, 0.1), range('steps', 'Integration steps', '積分步數', 4, 1, 40), select('solver', 'Integrator', '積分器', 'euler', [['euler', 'Euler: 1 evaluation per step', 'Euler：每步評估 1 次'], ['midpoint', 'Midpoint: 2 evaluations per step', '中點法：每步評估 2 次']])],
  compute(s, lang) { const rate = num(s, 'rate'), steps = num(s, 'steps'), midpoint = s.solver === 'midpoint'; const starts = [-1, -0.5, 0, 0.5, 1]; const solved = starts.map(x => transportSolve(x, rate, steps, midpoint)); const project = (t: number, x: number): [number, number] => [45 + 550 * t, 245 - (x + 1.2) * 37]; let body = line(45, 245, 600, 245) + text(50, 280, pick(lang, 't: 0 → 1 · vertical position: state x', 't：0 → 1 · 縱軸：狀態 x')); solved.forEach((r, i) => { body += path(r.trajectory.map((x, j) => project(j / steps, x)), stroke('a')); body += path(Array.from({ length: 41 }, (_, j) => { const t = j / 40; return project(t, Math.exp(rate * t) * starts[i] + (Math.abs(rate) < 1e-12 ? t : Math.expm1(rate * t) / rate)); }), `${stroke('b')} stroke-dasharray="4 4"`); }); return { svg: svg(body), summary: pick(lang, 'Step count and function evaluations are different costs. Reducing integration error cannot fix an incorrect learned field; this oracle example keeps field error at zero so the two issues stay separate.', '步數與函數評估次數是不同成本。減少積分誤差無法修正錯誤的學得速度場；此處已知精確速度場，因此速度場誤差為零。'), metrics: [metric(lang, 'Mean endpoint error', '平均終點誤差', fmt(sum(solved.map(r => r.error)) / solved.length, 6)), metric(lang, 'Field evaluations / particle', '每粒子速度場評估次數', steps * (midpoint ? 2 : 1)), metric(lang, 'Exact endpoint from x₀=0', 'x₀=0 的精確終點', fmt(solved[2].exact))] }; },
};

export function binaryAutoregression(copy: number, bits: number[], column: boolean) {
  const order = column ? [0, 2, 1, 3] : [0, 1, 2, 3]; const sequence = order.map(i => bits[i]); const factors = sequence.map((v, i) => i === 0 ? 0.5 : v === sequence[i - 1] ? copy : 1 - copy); const probability = factors.reduce((a, b) => a * b, 1);
  const distribution = Array.from({ length: 16 }, (_, n) => { const seq = order.map(i => (n >> (3 - i)) & 1); return seq.slice(1).reduce((p, value, i) => p * (value === seq[i] ? copy : 1 - copy), 0.5); });
  return { sequence, factors, probability, distribution, bitsPerPixel: -Math.log2(probability) / 4 };
}
const autoregression: Experiment = {
  id: 'autoregressive-pixels', track: 'ml', title: L('Factor a tiny image into conditional predictions', '把微型影像分解為條件預測'),
  description: L('Enumerate all sixteen binary 2×2 images under a first-order pixel model, then change the generation order.', '在一階像素模型下枚舉十六種二元 2×2 影像，再改變生成順序。'),
  challenge: L('Compare row stripes and a checkerboard when copying is likely. Switch raster to column order: which model assumptions changed?', '在偏好複製的設定下比較橫條紋與棋盤格。把逐列順序換成逐行順序：哪些模型假設改變了？'),
  note: L('First pixel has probability 1/2; each later pixel copies the previous one with probability c. The chain rule is exact, but this model deliberately retains only one previous pixel. Order dependence here comes from that restricted conditional model.', '第一個像素的機率為 1/2；後續像素以機率 c 複製前一像素。鏈式法則精確成立，但此模型刻意只保留前一像素。這裡的順序依賴來自受限的條件模型。'),
  related: ['ml/ml-08-autoregressive-vision', 'nlp/nlp-09-language-modeling-transformers'],
  controls: [range('copy', 'Copy-previous probability c', '複製前一像素的機率 c', 0.8, 0.05, 0.95, 0.05), select('pattern', 'Observed image', '觀察影像', 'stripe', [['stripe', 'Row stripes 00/11', '橫條紋 00／11'], ['checker', 'Checkerboard 01/10', '棋盤格 01／10'], ['solid', 'Solid 11/11', '全白 11／11']]), select('order', 'Generation order', '生成順序', 'raster', [['raster', 'Row-major raster', '逐列掃描'], ['column', 'Column-major', '逐行掃描']])],
  compute(s, lang) { const bits = s.pattern === 'stripe' ? [0, 0, 1, 1] : s.pattern === 'checker' ? [0, 1, 1, 0] : [1, 1, 1, 1]; const r = binaryAutoregression(num(s, 'copy'), bits, s.order === 'column'); let body = bits.map((bit, i) => `<rect x="${35 + i % 2 * 55}" y="${40 + Math.floor(i / 2) * 55}" width="52" height="52" fill="var(--plot-${bit ? 'a' : 'grid'})"/>` + text(55 + i % 2 * 55, 72 + Math.floor(i / 2) * 55, bit)).join(''); body += text(180, 65, `${pick(lang, 'Sequence', '序列')}: ${r.sequence.join(' → ')}`) + text(180, 105, `P = ${r.factors.map(v => fmt(v)).join(' × ')}`); r.distribution.forEach((p, i) => { body += `<rect x="${25 + i * 37}" y="${255 - p * 280}" width="27" height="${p * 280}" fill="var(--plot-b)"/>` + text(25 + i * 37, 280, i); }); body += text(25, 310, pick(lang, 'Image index 0…15 (four bits, row-major)', '影像索引 0…15（四個位元，列優先）')); return { svg: svg(body, 340), summary: pick(lang, 'Joint probability multiplies normalized conditional factors. All sixteen image probabilities sum to one. A high-copy model prefers sequences with fewer transitions, which need not correspond to the same spatial pattern in each order.', '聯合機率是各個正規化條件因子的乘積。十六種影像機率總和為一。高複製機率模型偏好較少轉換的序列，但不同順序對應的空間圖樣不一定相同。'), metrics: [metric(lang, 'Image probability', '影像機率', fmt(r.probability, 6)), metric(lang, 'Bits per pixel', '每像素位元', fmt(r.bitsPerPixel)), metric(lang, 'All-image probability sum', '全影像機率總和', fmt(sum(r.distribution)))] }; },
};

export function temperatureCalibration(temperature: number) {
  const logits = [-3, -2, -1, -0.4, 0.2, 0.8, 1.7, 3]; const labels = [0, 0, 1, 0, 1, 0, 1, 1]; const probabilities = logits.map(z => sigmoid(z / temperature));
  const accuracy = sum(probabilities.map((p, i) => Number(Number(p >= 0.5) === labels[i]))) / labels.length;
  const nll = -sum(probabilities.map((p, i) => Math.log(labels[i] ? p : 1 - p))) / labels.length;
  const brier = sum(probabilities.map((p, i) => (p - labels[i]) ** 2)) / labels.length;
  return { logits, labels, probabilities, accuracy, nll, brier };
}
const calibration: Experiment = {
  id: 'temperature-calibration', track: 'ml', title: L('Same predictions, different probability quality', '預測相同，機率品質卻不同'),
  description: L('Apply one positive temperature to fixed validation logits and compare accuracy, log loss and squared probability error.', '對固定驗證 logits 套用正溫度，比較準確率、對數損失與機率平方誤差。'),
  challenge: L('Find a temperature with lower log loss than T=1. Why does accuracy stay fixed even while confidence changes?', '找出對數損失低於 T=1 的溫度。為何信心改變時，準確率保持不變？'),
  note: L('Eight synthetic validation observations, not a benchmark. Binary probabilities use σ(logit/T), with a fixed 0.5 decision threshold. Choosing T here illustrates validation tuning; it does not establish performance on an unseen test set.', '八筆合成驗證觀察，不是基準測試。二元機率使用 σ(logit/T)，決策閾值固定為 0.5。在此挑選 T 是示範驗證集調整，不代表未見測試集上的表現。'),
  related: ['ml/ml-10-evaluation-deployment', 'nlp/nlp-11-classification-bert', 'nlp/nlp-03-neural-networks-for-nlp'],
  predict: { question: L("Raise the temperature T from 1 to 4. Does the classifier's accuracy change?", "把溫度 T 從 1 提高到 4。分類器的準確率會改變嗎？"), options: [{ value: "drops", label: L("It drops", "下降") }, { value: "same", label: L("It stays the same", "不變") }, { value: "rises", label: L("It rises", "上升") }], answer: "same", explain: L("Dividing every logit by T never changes its sign, so every predicted class is unchanged (accuracy stays 0.75 here). Only the confidence moves, and with it the log loss.", "把每個 logit 除以 T 不會改變正負號，所以每個預測類別都不變（此處準確率維持 0.75）。改變的只有信心，對數損失也隨之改變。") },
  controls: [range('temperature', 'Calibration temperature T', '校準溫度 T', 1, 0.2, 4, 0.1)],
  compute(s, lang) { const r = temperatureCalibration(num(s, 'temperature')); let body = text(30, 25, pick(lang, 'Bars: P(class 1) · dots: observed labels 0 or 1', '長條：P(類別 1)・圓點：觀察標籤 0 或 1')); r.probabilities.forEach((p, i) => { const x = 50 + i * 72; body += `<rect x="${x}" y="${235 - 170 * p}" width="35" height="${170 * p}" fill="var(--plot-a)"/>` + dot(x + 17, 235 - 170 * r.labels[i], 6, 'b') + text(x, 265, r.logits[i]); }); body += text(50, 295, pick(lang, 'x: original logit · y: 0 (bottom) → 1 (top)', 'x：原始 logit · y：0（下）→ 1（上）')); return { svg: svg(body, 320), summary: pick(lang, 'Positive temperature preserves logit ordering and binary decisions. It can adjust overconfidence or underconfidence, but cannot correct which examples the classifier gets wrong.', '正溫度保留 logits 排序與二元決策。它能調整過度自信或信心不足，卻無法修正分類器判錯的樣本。'), metrics: [metric(lang, 'Accuracy', '準確率', `${fmt(100 * r.accuracy)}%`), metric(lang, 'Mean log loss, nats', '平均對數損失（nat）', fmt(r.nll)), metric(lang, 'Brier score', 'Brier 分數', fmt(r.brier))] }; },
};

export const chapterLanguageExperiments: Experiment[] = [scope, bpe, pooling, sequence, retrieval, decoding, masking, gradients, network, memory, contrastive, vae, flow, transport, autoregression, calibration];
