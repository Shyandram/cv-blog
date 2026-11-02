// Facts for the `ctc-decoding` lab (DLCV 06): a toy per-frame probability matrix for a word with a double letter, decoded greedily
// (best path) and by prefix beam search, with the exact text probabilities and the CTC forward algorithm. Pure computation is
// exported so that scripts/tests/lab-ctc.test.ts can check it against brute-force enumeration of every alignment.
import type { LabDef } from './types.ts';
import { L, pick } from '../types.ts';
import { plotColors, startPlot, label } from './plot2d.ts';
import type { Lang } from '../../../i18n/ui';

export const WORDS = ['BOOK', 'SEE'] as const;
/** Index of the blank symbol in every matrix. */
export const BLANK = 0;
/** A "sure" frame: the letter has this probability and the blank the rest. */
export const SURE = 0.85;
export const BEAM = 3;
/** Number of frames between two equal letters in the toy. */
export const GAP_FRAMES = 2;
export const DEFAULTS = { q: 0.45, decoder: 'greedy', word: 'BOOK' };

export interface Toy {
  word: string;
  /** Distinct letters in order of first appearance; symbol k >= 1 is letters[k − 1], symbol 0 is the blank. */
  letters: string[];
  /** probs[t][s]: probability of symbol s at frame t (each row sums to 1). */
  probs: number[][];
  /** True for the frames between two equal letters (their blank probability is the control q). */
  gap: boolean[];
  /** The correct label as symbol indices. */
  target: number[];
}
/**
 * One sure frame per letter (letter 0.85, blank 0.15), and between two equal letters GAP_FRAMES frames in which the network
 * hesitates: blank with probability q, the letter with 1 − q. Every row has at most two non-zero entries, so the only question
 * the matrix asks is "blank or letter?".
 */
export function buildMatrix(word: string, q: number): Toy {
  const letters = [...new Set(word)], V = letters.length + 1, target = [...word].map(c => letters.indexOf(c) + 1);
  const probs: number[][] = [], gap: boolean[] = [];
  const row = (s: number, pBlank: number) => { const r = new Array<number>(V).fill(0); r[BLANK] = pBlank; r[s] = 1 - pBlank; return r; };
  target.forEach((s, j) => {
    if (j > 0 && target[j - 1] === s) for (let g = 0; g < GAP_FRAMES; g++) { probs.push(row(s, q)); gap.push(true); }
    probs.push(row(s, 1 - SURE)); gap.push(false);
  });
  return { word, letters, probs, gap, target };
}

// ---------------------------------------------------------------------------------------------- CTC
/** The map B: merge consecutive repeats, then delete blanks. */
export function collapse(path: readonly number[]): number[] {
  const out: number[] = [];
  path.forEach((s, i) => { if (s !== BLANK && (i === 0 || path[i - 1] !== s)) out.push(s); });
  return out;
}
export const textOf = (letters: readonly string[], seq: readonly number[]) => seq.map(s => letters[s - 1]).join('');
export const pathProbability = (probs: number[][], path: readonly number[]) => path.reduce((p, s, t) => p * probs[t][s], 1);
/** Best-path (greedy) decoding: the largest entry of every frame (ties go to the lowest index, so to the blank), then B. */
export function greedyDecode(probs: number[][]) {
  const path = probs.map(r => r.indexOf(Math.max(...r)));
  return { path, label: collapse(path), prob: pathProbability(probs, path) };
}
/** p(label | x) by the forward recursion over the blank-extended label sequence (O(T·L)). */
export function forwardProbability(probs: number[][], label: readonly number[]): number {
  const T = probs.length, ext: number[] = [BLANK];
  for (const s of label) ext.push(s, BLANK);
  const S = ext.length;
  let alpha = new Array<number>(S).fill(0);
  alpha[0] = probs[0][BLANK];
  if (S > 1) alpha[1] = probs[0][ext[1]];
  for (let t = 1; t < T; t++) {
    const next = new Array<number>(S).fill(0);
    for (let s = 0; s < S; s++) {
      let a = alpha[s] + (s >= 1 ? alpha[s - 1] : 0);
      if (s >= 2 && ext[s] !== BLANK && ext[s] !== ext[s - 2]) a += alpha[s - 2];
      next[s] = a * probs[t][ext[s]];
    }
    alpha = next;
  }
  return alpha[S - 1] + (S >= 2 ? alpha[S - 2] : 0);
}
/** Exact probability of every text: add up all V^T alignments that collapse to it (feasible only for tiny T). Keys are symbol lists joined by ','. */
export function exactTexts(probs: number[][]): Map<string, number> {
  const T = probs.length, V = probs[0].length, out = new Map<string, number>(), path = new Array<number>(T).fill(0);
  const rec = (t: number, p: number) => {
    if (p === 0) return;
    if (t === T) { const k = collapse(path).join(','); out.set(k, (out.get(k) ?? 0) + p); return; }
    for (let s = 0; s < V; s++) { path[t] = s; rec(t + 1, p * probs[t][s]); }
  };
  rec(0, 1);
  return out;
}
export interface Prefix { label: number[]; pb: number; pnb: number; p: number }
/**
 * CTC prefix beam search: every prefix keeps p_b (alignments ending in a blank) and p_nb (ending in its last label); a repeat of
 * the last label extends the prefix only through a blank. After each frame only the `beam` prefixes with the largest p_b + p_nb
 * survive. Returns the surviving prefixes after the last frame, best first.
 */
export function prefixBeamSearch(probs: number[][], beam: number): Prefix[] {
  let beams = new Map<string, Prefix>([['', { label: [], pb: 1, pnb: 0, p: 1 }]]);
  for (const f of probs) {
    const next = new Map<string, Prefix>();
    const add = (label: number[], pb: number, pnb: number) => {
      const k = label.join(','), o = next.get(k) ?? { label, pb: 0, pnb: 0, p: 0 };
      o.pb += pb; o.pnb += pnb; next.set(k, o);
    };
    for (const { label, pb, pnb } of beams.values()) {
      add(label, (pb + pnb) * f[BLANK], 0);
      const last = label.length ? label[label.length - 1] : -1;
      for (let s = 1; s < f.length; s++) {
        if (f[s] === 0) continue;
        if (s === last) { add(label, 0, pnb * f[s]); add([...label, s], 0, pb * f[s]); }
        else add([...label, s], 0, (pb + pnb) * f[s]);
      }
    }
    beams = new Map([...next.values()].map(v => { v.p = v.pb + v.pnb; return v; }).sort((a, b) => b.p - a.p).slice(0, beam).map(v => [v.label.join(','), v] as const));
  }
  return [...beams.values()];
}

export interface Analysis {
  toy: Toy; q: number;
  greedy: ReturnType<typeof greedyDecode>;
  beam: Prefix[];
  /** Texts by exact probability, best first. */
  exact: { label: number[]; text: string; p: number }[];
  pTarget: number;
  loss: number;
}
export function analyse(word: string, q: number): Analysis {
  const toy = buildMatrix(word, q), greedy = greedyDecode(toy.probs), beam = prefixBeamSearch(toy.probs, BEAM);
  const exact = [...exactTexts(toy.probs)].map(([k, p]) => { const label = k === '' ? [] : k.split(',').map(Number); return { label, text: textOf(toy.letters, label), p }; }).sort((a, b) => b.p - a.p);
  const pTarget = forwardProbability(toy.probs, toy.target);
  return { toy, q, greedy, beam, exact, pTarget, loss: -Math.log(pTarget) };
}
const exactOf = (a: Analysis, label: readonly number[]) => a.exact.find(e => e.label.join(',') === label.join(','))?.p ?? 0;

// ---------------------------------------------------------------------------------------------- drawing
const MAT_W = 420, MAT_H = 392, BAR_W = 420, BAR_H = 300;
const f2 = (v: number) => v.toFixed(2);
const f3 = (v: number) => v.toFixed(3);
const sym = (toy: Toy, s: number) => s === BLANK ? '−' : toy.letters[s - 1];

function drawMatrix(canvas: HTMLCanvasElement, lang: Lang, a: Analysis, decoder: string) {
  const col = plotColors(canvas), ctx = startPlot(canvas, MAT_W, MAT_H, col), { toy } = a, V = toy.letters.length + 1, T = toy.probs.length;
  const x0 = 76, cw = (MAT_W - x0 - 12) / V, y0 = 66, rh = 34;
  label(ctx, pick(lang, `Network output for “${toy.word}”: p(symbol) per frame`, `網路對「${toy.word}」的輸出：各畫格的符號機率`), 12, 22, { bold: true, color: col.ink, size: 15 });
  label(ctx, pick(lang, 'frame', '畫格'), 12, y0 - 10, { color: col.muted, size: 14 });
  for (let s = 0; s < V; s++) label(ctx, s === BLANK ? pick(lang, 'blank −', '空白 −') : toy.letters[s - 1], x0 + s * cw + cw / 2, y0 - 10, { align: 'center', bold: true, color: col.ink, size: 16 });
  for (let t = 0; t < T; t++) {
    const y = y0 + t * rh;
    label(ctx, String(t + 1), 50, y + rh / 2 + 5, { align: 'right', color: col.muted, size: 15 });
    for (let s = 0; s < V; s++) {
      const p = toy.probs[t][s], x = x0 + s * cw;
      ctx.globalAlpha = 0.08 + 0.55 * p; ctx.fillStyle = col.a; ctx.fillRect(x + 1, y + 1, cw - 2, rh - 2); ctx.globalAlpha = 1;
      ctx.lineWidth = 1; ctx.strokeStyle = col.grid; ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, rh - 1);
      label(ctx, p === 0 ? '0' : f2(p), x + cw / 2, y + rh / 2 + 5, { align: 'center', color: p === 0 ? col.muted : col.ink, size: 15, bold: p >= 0.5 });
      if (decoder === 'greedy' && a.greedy.path[t] === s) { ctx.lineWidth = 3; ctx.strokeStyle = col.ink; ctx.strokeRect(x + 2, y + 2, cw - 4, rh - 4); }
    }
  }
  // The frames between the repeated letters: an orange bar on the left.
  const first = toy.gap.indexOf(true), last = toy.gap.lastIndexOf(true);
  if (first >= 0) {
    ctx.fillStyle = col.b; ctx.fillRect(58, y0 + first * rh + 2, 6, (last - first + 1) * rh - 4);
    label(ctx, pick(lang, 'gap', '間隔'), 8, y0 + first * rh + rh + 5, { color: col.b, size: 13, bold: true });
  }
  const yb = y0 + T * rh + 28;
  if (decoder === 'greedy') {
    const path = a.greedy.path.map(s => sym(toy, s)), merged = path.filter((s, i) => i === 0 || s !== path[i - 1]), kept = merged.filter(s => s !== '−');
    label(ctx, pick(lang, 'Greedy: largest entry of every frame (outlined)', '貪婪解碼：每個畫格的最大值（框起來）'), 12, yb, { bold: true, color: col.ink, size: 14 });
    const rows: [string, string, boolean][] = [
      [pick(lang, 'path', '路徑'), path.join(' '), false],
      [pick(lang, 'merge repeats', '合併重複'), merged.join(' '), false],
      [pick(lang, 'drop blanks', '刪除空白'), `${kept.join(' ') || pick(lang, '(empty)', '（空）')}  →  ${kept.join('') || pick(lang, '(empty)', '（空）')}`, true],
    ];
    rows.forEach(([k, v, bold], i) => { label(ctx, k, 12, yb + 22 + 22 * i, { color: col.muted, size: 14 }); label(ctx, v, 128, yb + 22 + 22 * i, { color: col.ink, bold, size: 15 }); });
  } else {
    label(ctx, pick(lang, `Prefix beam search (beam ${BEAM}): final prefixes`, `前綴束搜尋（束寬 ${BEAM}）：最後的前綴`), 12, yb, { bold: true, color: col.ink, size: 14 });
    a.beam.forEach((b, i) => label(ctx, `${i + 1}.  ${textOf(toy.letters, b.label) || pick(lang, '(empty)', '（空）')}   p = ${f3(b.p)}`, 12, yb + 22 + 22 * i, { color: col.ink, bold: i === 0, size: 15 }));
  }
}

/** The five most probable texts; if the correct word is not among them it replaces the fifth, so it is always visible. */
export function barRows(a: Analysis) {
  const top = a.exact.slice(0, 5), key = a.toy.target.join(',');
  if (top.some(r => r.label.join(',') === key)) return top;
  return [...top.slice(0, 4), { label: a.toy.target, text: a.toy.word, p: a.pTarget }];
}
function drawBars(canvas: HTMLCanvasElement, lang: Lang, a: Analysis, chosen: number[]) {
  const col = plotColors(canvas), ctx = startPlot(canvas, BAR_W, BAR_H, col), rows = barRows(a), x0 = 92, x1 = BAR_W - 62, top = 44, rh = 40;
  label(ctx, pick(lang, 'Exact probability of each text (all alignments added)', '每段文字的精確機率（所有對齊相加）'), 12, 22, { bold: true, color: col.ink, size: 15 });
  const key = (l: number[]) => l.join(',');
  rows.forEach((r, i) => {
    const y = top + i * rh, w = Math.max(2, r.p * (x1 - x0)), isChosen = key(r.label) === key(chosen), isTarget = key(r.label) === key(a.toy.target);
    ctx.fillStyle = isChosen ? col.b : col.grid; ctx.globalAlpha = isChosen ? 0.9 : 0.8; ctx.fillRect(x0, y + 4, w, rh - 12); ctx.globalAlpha = 1;
    if (isTarget) { ctx.lineWidth = 3; ctx.strokeStyle = '#1a8f3c'; ctx.strokeRect(x0 - 1.5, y + 2.5, w + 3, rh - 9); }
    label(ctx, r.text || pick(lang, '(empty)', '（空）'), x0 - 8, y + rh / 2 + 4, { align: 'right', bold: isTarget || isChosen, color: col.ink, size: 16 });
    label(ctx, f3(r.p), x0 + w + 6, y + rh / 2 + 4, { color: col.ink, size: 14 });
  });
  const ly = top + 5 * rh + 14;
  ctx.fillStyle = col.b; ctx.globalAlpha = 0.9; ctx.fillRect(12, ly - 11, 18, 14); ctx.globalAlpha = 1; label(ctx, pick(lang, 'text chosen by the decoder', '解碼器選出的文字'), 38, ly, { color: col.ink, size: 14 });
  ctx.lineWidth = 3; ctx.strokeStyle = '#1a8f3c'; ctx.strokeRect(13.5, ly + 11.5, 15, 11); label(ctx, pick(lang, `correct word ${a.toy.word}`, `正確的單字 ${a.toy.word}`), 38, ly + 22, { color: col.ink, size: 14 });
}

// ---------------------------------------------------------------------------------------------- text
export function describe(lang: Lang, a: Analysis, decoder: string): string {
  const { toy } = a, word = toy.word, rep = toy.letters[toy.target.find((s, j) => j > 0 && toy.target[j - 1] === s)! - 1];
  const q = a.q, chosen = decoder === 'greedy' ? a.greedy.label : a.beam[0].label, text = textOf(toy.letters, chosen) || pick(lang, '(empty)', '（空）'), ok = chosen.join(',') === toy.target.join(',');
  const blankFrames = toy.gap.filter((g, t) => g && a.greedy.path[t] === BLANK).length;
  const pc = a.pTarget, px = exactOf(a, chosen);
  if (decoder === 'greedy') {
    return ok
      ? pick(lang,
        `Greedy decoding keeps the blank in ${blankFrames} of the ${GAP_FRAMES} frames between the two ${rep}s (blank ${f2(q)} against ${rep} ${f2(1 - q)}), so the merge cannot join them and it writes ${text}. The probability of the correct word, summed over all alignments, is ${f3(pc)}.`,
        `貪婪解碼在兩個 ${rep} 之間的 ${GAP_FRAMES} 個畫格中，有 ${blankFrames} 個畫格取到空白（空白 ${f2(q)}，${rep} ${f2(1 - q)}），合併時兩個 ${rep} 不會被接在一起，所以寫出 ${text}。把所有對齊加起來，正確單字的機率是 ${f3(pc)}。`)
      : pick(lang,
        `Greedy decoding writes ${text}, not ${word}: in none of the ${GAP_FRAMES} frames between the two ${rep}s is the blank the largest entry (${f2(q)} against ${f2(1 - q)}), so the repeats merge into one ${rep}. The correct word has probability ${f3(pc)} summed over its alignments, ${pc > px ? 'more' : 'less'} than the ${f3(px)} of ${text}.`,
        `貪婪解碼寫出 ${text}，而不是 ${word}：在兩個 ${rep} 之間的 ${GAP_FRAMES} 個畫格裡，沒有任何一個畫格的最大值是空白（${f2(q)} 對 ${f2(1 - q)}），所以重複的 ${rep} 被合併成一個。把正確單字的所有對齊相加，機率是 ${f3(pc)}，${pc > px ? '高於' : '低於'} ${text} 的 ${f3(px)}。`);
  }
  return ok
    ? pick(lang,
      `Prefix beam search (beam ${BEAM}) writes ${text}. It adds up every alignment, so a blank in any of the ${GAP_FRAMES} frames between the two ${rep}s counts: ${word} has probability ${f3(pc)}, ${pc > exactOf(a, a.greedy.label) ? 'more than' : 'not more than'} ${f3(exactOf(a, a.greedy.label))} for the greedy text ${textOf(toy.letters, a.greedy.label) || pick(lang, '(empty)', '（空）')}.`,
      `前綴束搜尋（束寬 ${BEAM}）寫出 ${text}。它把所有對齊加起來，所以兩個 ${rep} 之間的 ${GAP_FRAMES} 個畫格只要任何一個有空白就算數：${word} 的機率是 ${f3(pc)}，${pc > exactOf(a, a.greedy.label) ? '高於' : '不高於'}貪婪解碼那段文字 ${textOf(toy.letters, a.greedy.label) || '（空）'} 的 ${f3(exactOf(a, a.greedy.label))}。`)
    : pick(lang,
      `Prefix beam search (beam ${BEAM}) also writes ${text}: with the blank at only ${f2(q)} in each of the ${GAP_FRAMES} frames between the two ${rep}s, the alignments that read ${word} add up to ${f3(pc)}, less than the ${f3(px)} of ${text}. The network is more likely to have seen one long ${rep}.`,
      `前綴束搜尋（束寬 ${BEAM}）同樣寫出 ${text}：兩個 ${rep} 之間的 ${GAP_FRAMES} 個畫格裡，空白的機率各只有 ${f2(q)}，讀成 ${word} 的對齊加總為 ${f3(pc)}，低於 ${text} 的 ${f3(px)}。網路比較可能是看到一個拉長的 ${rep}。`);
}

// ---------------------------------------------------------------------------------------------- the lab
export const lab: LabDef = {
  kind: 'image', id: 'ctc-decoding', track: 'deep-learning',
  title: L('DLCV 06 · How does CTC write a double letter?', 'DLCV 06・CTC 要怎麼寫出連續兩個相同的字母？'),
  description: L('Edit the blank probability in the frames between two equal letters of a toy recognizer output, and watch greedy best-path decoding and prefix beam search read it.', '在玩具辨識器的輸出中，改變兩個相同字母之間畫格的空白機率，觀察貪婪（最佳路徑）解碼與前綴束搜尋怎麼讀它。'),
  challenge: L(
    'At the default setting (blank 0.45 and O 0.55 in each of the two frames between the O\u2019s, greedy decoding) the largest entry of every frame gives the path B O O O O K, which merges to BOK: one O is lost. Yet the correct word has the largest total probability, 0.410, against 0.310 for BOK, because every alignment with a blank in either gap frame reads BOOK. Switch to prefix beam search (beam 3): it writes BOOK. Now move the blank probability. At 0.50 greedy decoding also writes BOOK (a tie goes to the blank). With beam search, 0.40 still gives BOOK, but at 0.35 it writes BOK (0.377 against 0.343 for BOOK). At 0.05 the network almost surely saw one long O: both decoders write BOK and the loss \u2212ln p(BOOK) is 2.819. Finally choose SEE: greedy decoding flips at 0.50 and beam search between 0.35 and 0.40, just as for BOOK. Why can the sum over alignments recover a double letter that no single frame votes for?',
    '預設設定（兩個 O 之間的兩個畫格，空白各 0.45、O 各 0.55，貪婪解碼）下，每個畫格取最大值得到路徑 B O O O O K，合併後是 BOK：少了一個 O。然而正確單字的總機率最大，是 0.410，BOK 則是 0.310，因為只要間隔的兩個畫格中任何一個有空白，這個對齊就讀成 BOOK。切換成前綴束搜尋（束寬 3）：它寫出 BOOK。接著移動空白機率。0.50 時貪婪解碼也寫出 BOOK（平手時取空白）。束搜尋在 0.40 仍得到 BOOK，但 0.35 時寫出 BOK（0.377 對 BOOK 的 0.343）。0.05 時網路幾乎確定看到的是一個拉長的 O：兩種解碼器都寫出 BOK，損失 \u2212ln p(BOOK) 是 2.819。最後選 SEE：貪婪解碼在 0.50 翻轉，束搜尋在 0.35 與 0.40 之間翻轉，和 BOOK 一樣。為什麼把所有對齊加起來，就能找回沒有任何單一畫格投票支持的連續字母？'),
  note: L(
    'A hand-made toy, not the output of a trained recognizer. Each letter of the word has one sure frame (the letter 0.85, the blank 0.15); between two equal letters there are two gap frames in which the network hesitates between the blank (probability q, the slider) and the letter (1 \u2212 q). Every other entry is 0, so the matrix asks one question only: blank or letter? BOOK has 6 frames (B, O, two gap frames, O, K) and SEE has 5. The collapsing map merges repeats first and then deletes blanks, so a double letter can be written only with a blank between its copies. Greedy decoding takes the largest entry of every frame; a tie goes to the blank, as NumPy\u2019s argmax goes to the first index. The right-hand bars add up every alignment of each text by enumeration (4\u2076 = 4,096 alignments for BOOK, 3\u2075 = 243 for SEE); the probability of the correct word is also computed with the forward algorithm, and the tests check the two against each other. Prefix beam search keeps, for each prefix, the probability of alignments that end in a blank and of those that end in the last letter, and only the 3 best prefixes after every frame; pruning can lose some alignments, so its own probability can be a little below the exact one (0.366 against 0.377 for BOK at q = 0.35). The same code reproduces the 5-frame \u2018cat\u2019 example of the chapter (greedy ca, beam 3 cat 0.344, exact 0.363).',
    '手工做的玩具，不是訓練好的辨識器的輸出。單字的每個字母各有一個確定的畫格（字母 0.85、空白 0.15）；兩個相同字母之間有兩個間隔畫格，網路在空白（機率 q，也就是滑桿）與字母（1 \u2212 q）之間猶豫。其餘的項都是 0，所以這個矩陣只問一個問題：是空白還是字母？BOOK 有 6 個畫格（B、O、兩個間隔畫格、O、K），SEE 有 5 個。合併映射先合併重複、再刪除空白，所以連續字母只有在兩個複本之間有空白時才寫得出來。貪婪解碼取每個畫格的最大值；平手時取空白，如同 NumPy 的 argmax 取第一個索引。右邊的長條用枚舉的方式把每段文字的所有對齊相加（BOOK 有 4\u2076 = 4,096 種對齊，SEE 有 3\u2075 = 243 種）；正確單字的機率也用前向演算法計算，測試會把兩者互相比對。前綴束搜尋對每個前綴保留以空白結尾與以最後一個字母結尾的對齊機率，且每個畫格之後只留最好的 3 個前綴；剪枝可能丟掉一些對齊，所以它自己算出的機率可能略低於精確值（q = 0.35 時 BOK 是 0.366，精確值是 0.377）。同一份程式也重現了章節中五個畫格的「cat」範例（貪婪 ca，束寬 3 得 cat 0.344，精確值 0.363）。'),
  related: ['deep-learning/dl-11a-text-in-images'],
  controls: [
    { key: 'q', label: L('Blank probability in the two frames between the repeated letters', '重複字母之間兩個畫格的空白機率'), kind: 'range', value: DEFAULTS.q, min: 0.05, max: 0.95, step: 0.05 },
    { key: 'decoder', label: L('Decoder', '解碼器'), kind: 'select', value: DEFAULTS.decoder, options: [
      { value: 'greedy', label: L('Greedy best path', '貪婪最佳路徑') },
      { value: 'beam', label: L(`Prefix beam search (beam ${BEAM})`, `前綴束搜尋（束寬 ${BEAM}）`) },
    ] },
    { key: 'word', label: L('Word', '單字'), kind: 'select', value: DEFAULTS.word, options: WORDS.map(w => ({ value: w, label: L(w, w) })) },
  ],
  predict: {
    question: L(
      'In the default matrix each of the two frames between the O\u2019s says blank 0.45 and O 0.55, and every other frame is sure of its letter. What does greedy best-path decoding write for BOOK?',
      '預設矩陣中，兩個 O 之間的兩個畫格各是空白 0.45、O 0.55，其餘畫格都很確定自己的字母。貪婪最佳路徑解碼會為 BOOK 寫出什麼？'),
    options: [
      { value: 'BOOK', label: L('BOOK', 'BOOK') },
      { value: 'BOK', label: L('BOK', 'BOK') },
      { value: 'BOOOK', label: L('BOOOK', 'BOOOK') },
    ],
    answer: 'BOK',
    explain: L(
      'BOK. In both gap frames the largest entry is the O (0.55 against 0.45), so the best path is B O O O O K; merging repeats gives B O K, and no blank is left to separate two O\u2019s. The collapsing map never writes three O\u2019s from this path either. The path has probability 0.158, yet the correct word BOOK has probability 0.410 summed over its alignments, more than the 0.310 of BOK: greedy decoding looks at one path, not at the text.',
      'BOK。兩個間隔畫格的最大值都是 O（0.55 對 0.45），所以最佳路徑是 B O O O O K；合併重複得到 B O K，沒有留下能隔開兩個 O 的空白。合併映射也不會從這條路徑寫出三個 O。這條路徑的機率是 0.158，但正確單字 BOOK 把所有對齊加起來是 0.410，高於 BOK 的 0.310：貪婪解碼看的是一條路徑，而不是整段文字。'),
  },
  assets: {},
  paneMin: 340,
  panes: [
    { key: 'matrix', kind: 'canvas', label: L('Toy network output: one row per frame, one column per symbol (orange bar: the frames between the repeated letters)', '玩具網路輸出：每個畫格一列、每個符號一欄（橘色長條：重複字母之間的畫格）'), width: MAT_W, height: MAT_H },
    { key: 'texts', kind: 'canvas', label: L('The five most probable texts, with all alignments of each added up', '機率最高的五段文字，各自把所有對齊加總'), width: BAR_W, height: BAR_H },
  ],
  sources: [
    { title: 'A. Graves, S. Fernández, F. Gomez and J. Schmidhuber, Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks, ICML, 2006 (the blank, the collapsing map, the forward algorithm, best path and prefix search decoding)', url: 'https://doi.org/10.1145/1143844.1143891' },
    { title: 'A. Y. Hannun, A. L. Maas, D. Jurafsky and A. Y. Ng, First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional Recurrent DNNs, arXiv:1408.2873, 2014 (prefix beam search with a blank and a non-blank probability per prefix)', url: 'https://arxiv.org/abs/1408.2873' },
  ],
  create({ lang }) {
    return {
      render(state, canvases) {
        const word = (WORDS as readonly string[]).includes(String(state.word)) ? String(state.word) : DEFAULTS.word, decoder = state.decoder === 'beam' ? 'beam' : 'greedy';
        const q = Math.round(Math.max(0.05, Math.min(0.95, Number(state.q))) * 100) / 100, a = analyse(word, q);
        const chosen = decoder === 'greedy' ? a.greedy.label : a.beam[0].label, text = textOf(a.toy.letters, chosen) || pick(lang, '(empty)', '（空）');
        drawMatrix(canvases.matrix, lang, a, decoder);
        drawBars(canvases.texts, lang, a, chosen);
        const ok = chosen.join(',') === a.toy.target.join(',');
        return {
          summary: describe(lang, a, decoder),
          metrics: [
            { label: pick(lang, 'Decoded text', '解碼出的文字'), value: text },
            { label: pick(lang, 'Double letter kept', '連續字母是否保留'), value: ok ? pick(lang, 'yes', '是') : pick(lang, `no (${chosen.length} of ${a.toy.target.length} letters)`, `否（${a.toy.target.length} 個字母只剩 ${chosen.length} 個）`) },
            { label: pick(lang, 'Greedy path probability (one alignment)', '貪婪路徑的機率（單一對齊）'), value: f3(a.greedy.prob) },
            { label: pick(lang, 'Probability of the decoded text (all alignments)', '解碼文字的機率（所有對齊）'), value: f3(exactOf(a, chosen)) },
            { label: pick(lang, `Probability of the correct word ${a.toy.word} (forward algorithm)`, `正確單字 ${a.toy.word} 的機率（前向演算法）`), value: f3(a.pTarget) },
            { label: pick(lang, 'CTC loss −ln p(correct word)', 'CTC 損失 −ln p(正確單字)'), value: f3(a.loss) },
          ],
        };
      },
    };
  },
};
