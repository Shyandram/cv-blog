// Facts for the `ctc-decoding` lab (DLCV 11A): the collapsing map, the forward algorithm checked against brute-force enumeration of
// every alignment, greedy and prefix-beam decoding (including the chapter's 5-frame 'cat' example), the toy matrix, and every
// number quoted in the lab's text.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../../src/lib/imagekit/rng.ts';
import {
  lab, BEAM, BLANK, SURE, GAP_FRAMES, DEFAULTS, WORDS, buildMatrix, collapse, textOf, pathProbability, greedyDecode, forwardProbability,
  exactTexts, prefixBeamSearch, analyse, barRows, describe,
} from '../../src/lib/playground/labs/ctc-decoding.ts';

const near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);
const softmaxRows = (T: number, V: number, seed: number) => {
  const r = mulberry32(seed);
  return Array.from({ length: T }, () => { const e = Array.from({ length: V }, () => Math.exp(2 * (r() - 0.5) * 2)), z = e.reduce((s, v) => s + v, 0); return e.map(v => v / z); });
};
const allPaths = (T: number, V: number): number[][] => (T === 0 ? [[]] : allPaths(T - 1, V).flatMap(p => Array.from({ length: V }, (_, s) => [...p, s])));

// ------------------------------------------------------------------------------------------------ CTC

test('collapse: merge repeats first, then delete blanks (the blank is what allows a double letter)', () => {
  assert.deepEqual(collapse([1, 1, 0, 1, 2, 2]), [1, 1, 2]);
  assert.deepEqual(collapse([1, 1]), [1]);
  assert.deepEqual(collapse([1, 0, 1]), [1, 1]);
  assert.deepEqual(collapse([0, 0, 0]), []);
  assert.deepEqual(collapse([1, 2, 1, 2]), [1, 2, 1, 2]);
  assert.deepEqual(collapse([]), []);
});

test('forward algorithm equals brute-force enumeration of all alignments on random matrices, for single, repeated and long labels', () => {
  const labels = [[1, 2], [1, 1], [2, 1, 2], [1], [1, 1, 2], [2, 2], [1, 2, 1]];
  for (let seed = 1; seed <= 12; seed++) {
    const T = 5 + (seed % 2), V = 3, probs = softmaxRows(T, V, seed), paths = allPaths(T, V);
    for (const y of labels) {
      const brute = paths.filter(p => collapse(p).join() === y.join()).reduce((s, p) => s + pathProbability(probs, p), 0);
      near(forwardProbability(probs, y), brute, 1e-12, `seed ${seed} y=${y}`);
    }
    const ex = exactTexts(probs);
    near([...ex.values()].reduce((s, v) => s + v, 0), 1, 1e-12, 'probabilities of all texts add up to 1');
    for (const y of labels) near(forwardProbability(probs, y), ex.get(y.join(',')) ?? 0, 1e-12, 'forward equals exact enumeration');
  }
});

test('forward algorithm: a label needing more frames than there are has probability 0 (a double letter needs a blank frame)', () => {
  const probs = softmaxRows(2, 3, 5);
  assert.equal(forwardProbability(probs, [1, 1]), 0, '"aa" needs T ≥ 3');
  assert.ok(forwardProbability(probs, [1, 2]) > 0);
  assert.equal(forwardProbability(softmaxRows(3, 3, 5), [1, 1]) > 0, true);
  near(forwardProbability([[0.3, 0.7]], [1]), 0.7, 1e-15, 'one frame');
});

test('prefix beam search: with an unlimited beam it returns the exact probabilities; beam 1 can only miss', () => {
  for (let seed = 1; seed <= 8; seed++) {
    const probs = softmaxRows(5, 3, seed), ex = exactTexts(probs), top = prefixBeamSearch(probs, 1000);
    for (const b of top) near(b.p, ex.get(b.label.join(',')) ?? 0, 1e-12, `seed ${seed} prefix ${b.label}`);
    const best = [...ex].sort((x, y) => y[1] - x[1])[0];
    assert.equal(top[0].label.join(','), best[0], 'best text found');
    for (const B of [1, 3]) assert.ok(prefixBeamSearch(probs, B)[0].p <= best[1] + 1e-12);
  }
});

test("the chapter's 'cat' toy: greedy gives ca, exact cat 0.363 against ca 0.179, beam 1 ca 0.155, beam 3 cat 0.344, beam 20 cat 0.363", () => {
  const probs = [[0.10, 0.80, 0.05, 0.05], [0.20, 0.05, 0.70, 0.05], [0.60, 0.05, 0.30, 0.05], [0.55, 0.02, 0.03, 0.40], [0.55, 0.02, 0.03, 0.40]], L = ['c', 'a', 't'];
  assert.equal(textOf(L, greedyDecode(probs).label), 'ca');
  const ex = [...exactTexts(probs)].sort((x, y) => y[1] - x[1]);
  assert.deepEqual(ex.slice(0, 2).map(([k, p]) => [textOf(L, k.split(',').map(Number)), p.toFixed(3)]), [['cat', '0.363'], ['ca', '0.179']]);
  assert.equal(path(prefixBeamSearch(probs, 1)), 'ca 0.155');
  assert.equal(path(prefixBeamSearch(probs, 3)), 'cat 0.344');
  assert.equal(path(prefixBeamSearch(probs, 20)), 'cat 0.363');
  function path(r: ReturnType<typeof prefixBeamSearch>) { return `${textOf(L, r[0].label)} ${r[0].p.toFixed(3)}`; }
});

// ------------------------------------------------------------------------------------------------ the toy

test('toy matrix: BOOK has 6 frames, SEE has 5; rows sum to 1; the gap frames carry q; the target is the word', () => {
  const b = buildMatrix('BOOK', 0.45), s = buildMatrix('SEE', 0.45);
  assert.deepEqual([b.probs.length, s.probs.length, b.probs[0].length, s.probs[0].length], [6, 5, 4, 3]);
  assert.deepEqual(b.gap, [false, false, true, true, false, false]);
  assert.deepEqual(s.gap, [false, false, true, true, false]);
  assert.equal(GAP_FRAMES, 2);
  for (const t of [b, s]) for (const r of t.probs) near(r.reduce((x, y) => x + y, 0), 1, 1e-12, 'row sum');
  assert.deepEqual(b.probs[2], [0.45, 0, 0.55, 0]);
  assert.deepEqual(b.probs[0].map(v => +v.toFixed(2)), [0.15, 0.85, 0, 0]);
  assert.equal(SURE, 0.85);
  assert.equal(textOf(b.letters, b.target), 'BOOK');
  assert.equal(textOf(s.letters, s.target), 'SEE');
  assert.deepEqual(b.letters, ['B', 'O', 'K']);
});

test('greedy decoding drops the double letter unless a gap frame has the blank as its largest entry; a tie goes to the blank', () => {
  for (const word of WORDS) for (let k = 1; k <= 19; k++) {
    const q = k / 20, a = analyse(word, q), written = textOf(a.toy.letters, a.greedy.label);
    if (q >= 0.5) assert.equal(written, word, `${word} at q ${q}`);
    else assert.equal(written, word === 'BOOK' ? 'BOK' : 'SE', `${word} at q ${q}`);
  }
  // 0.5 is a tie: blank wins
  assert.equal(greedyDecode([[0.5, 0, 0.5, 0]]).path[0], BLANK);
});

test('beam search recovers the double letter from q = 0.40 on (BOOK and SEE), earlier than greedy decoding (0.50), and not at 0.35', () => {
  for (const word of WORDS) for (let k = 1; k <= 19; k++) {
    const q = k / 20, a = analyse(word, q), written = textOf(a.toy.letters, a.beam[0].label);
    assert.equal(written === word, q >= 0.4 - 1e-9, `${word} beam at q ${q}: ${written}`);
    // the exact best text agrees with the beam
    assert.equal(a.exact[0].text === word, q >= 0.4 - 1e-9, `${word} exact at q ${q}`);
  }
});

test('defaults: BOOK, q = 0.45, greedy writes BOK (path probability 0.158); BOOK has probability 0.410 against 0.310; beam 3 writes BOOK', () => {
  assert.deepEqual(Object.fromEntries(lab.controls.map(c => [c.key, c.value])), { q: DEFAULTS.q, decoder: 'greedy', word: 'BOOK' });
  const a = analyse(DEFAULTS.word, DEFAULTS.q), L = a.toy.letters, ex = (t: string) => a.exact.find(e => e.text === t)!.p;
  assert.deepEqual(a.greedy.path, [1, 2, 2, 2, 2, 3]);
  assert.equal(textOf(L, a.greedy.path), 'BOOOOK');
  assert.equal(textOf(L, a.greedy.label), 'BOK');
  assert.equal(a.greedy.prob.toFixed(3), '0.158');
  assert.equal(ex('BOOK').toFixed(3), '0.410');
  assert.equal(ex('BOK').toFixed(3), '0.310');
  assert.equal(a.pTarget.toFixed(3), '0.410');
  assert.equal(a.loss.toFixed(3), '0.892');
  assert.equal(textOf(L, a.beam[0].label), 'BOOK');
  assert.equal(a.beam[0].p.toFixed(3), '0.410');
  assert.equal(a.beam.length, BEAM);
  assert.equal(a.exact[0].text, 'BOOK');
  assert.equal(a.exact[1].text, 'BOK');
  near(a.pTarget, ex('BOOK'), 1e-12, 'forward equals enumeration at the default');
  // SEE at the default
  const s = analyse('SEE', 0.45);
  assert.equal(textOf(s.toy.letters, s.greedy.label), 'SE');
  assert.equal(textOf(s.toy.letters, s.beam[0].label), 'SEE');
  assert.equal(s.pTarget.toFixed(3), '0.482');
});

test('the correct word is always among the five bars, even if it were not one of the five most probable texts', () => {
  const a = analyse('BOOK', 0.45), worst = { ...a, exact: [...a.exact].reverse() };
  assert.ok(worst.exact.findIndex(e => e.text === 'BOOK') >= 5, 'in the reversed list BOOK is last');
  assert.ok(barRows(worst).some(r => r.text === 'BOOK'));
  assert.equal(barRows(worst).length, 5);
  near(barRows(worst).find(r => r.text === 'BOOK')!.p, a.pTarget, 1e-15, 'probability shown for the correct word');
  for (const q of [0.05, 0.45, 0.95]) for (const w of WORDS) assert.ok(barRows(analyse(w, q)).some(r => r.text === w));
});

// ------------------------------------------------------------------------------------------------ text

const both = (field: 'challenge' | 'note' | 'explain') => (s: string) => {
  const text = field === 'explain' ? lab.predict.explain : lab[field];
  assert.ok(text.en.includes(s) && text['zh-tw'].includes(s), `${field} mentions ${s}`);
};

test('text: the numbers in the challenge are what the code produces', () => {
  const at = (w: string, q: number) => analyse(w, q), ex = (a: ReturnType<typeof analyse>, t: string) => a.exact.find(e => e.text === t)!.p.toFixed(3);
  const d = at('BOOK', 0.45);
  assert.equal(ex(d, 'BOOK'), '0.410'); assert.equal(ex(d, 'BOK'), '0.310');
  const q35 = at('BOOK', 0.35);
  assert.equal(ex(q35, 'BOK'), '0.377'); assert.equal(ex(q35, 'BOOK'), '0.343');
  assert.equal(textOf(q35.toy.letters, q35.beam[0].label), 'BOK');
  assert.equal(textOf(at('BOOK', 0.4).toy.letters, at('BOOK', 0.4).beam[0].label), 'BOOK');
  const q05 = at('BOOK', 0.05);
  assert.equal(textOf(q05.toy.letters, q05.greedy.label), 'BOK'); assert.equal(textOf(q05.toy.letters, q05.beam[0].label), 'BOK');
  assert.equal(q05.loss.toFixed(3), '2.819');
  ['0.45', '0.55', 'B O O O O K', 'BOK', '0.410', '0.310', '0.50', '0.40', '0.35', '0.377', '0.343', '0.05', '2.819', 'SEE', 'BOOK'].forEach(both('challenge'));
});

test('text: the numbers in the note and the explanation are what the code produces', () => {
  assert.equal(4 ** 6, 4096); assert.equal(3 ** 5, 243);
  assert.equal(exactTexts(buildMatrix('BOOK', 0.45).probs).size > 0, true);
  const q35 = analyse('BOOK', 0.35);
  assert.equal(q35.beam.find(b => textOf(q35.toy.letters, b.label) === 'BOK')!.p.toFixed(3), '0.366');
  const probs = [[0.10, 0.80, 0.05, 0.05], [0.20, 0.05, 0.70, 0.05], [0.60, 0.05, 0.30, 0.05], [0.55, 0.02, 0.03, 0.40], [0.55, 0.02, 0.03, 0.40]];
  assert.equal(prefixBeamSearch(probs, 3)[0].p.toFixed(3), '0.344');
  ['0.85', '0.15', '4,096', '243', '0.366', '0.377', '0.344', '0.363', 'ca'].forEach(both('note'));
  ['BOK', '0.55', '0.45', 'B O O O O K', 'B O K', '0.158', '0.410', '0.310'].forEach(both('explain'));
  assert.equal(lab.predict.answer, 'BOK');
  assert.ok(lab.predict.options.some(o => o.value === lab.predict.answer) && lab.predict.options.length <= 4);
});

test('text: describe reads in both languages for every decoder and outcome, with no placeholders or Simplified wording', () => {
  for (const w of WORDS) for (const q of [0.05, 0.45, 0.95]) for (const dec of ['greedy', 'beam']) {
    const a = analyse(w, q), en = describe('en', a, dec), zh = describe('zh-tw', a, dec);
    assert.ok(en.length > 40 && zh.length > 20 && !/NaN|undefined|TBD/.test(en + zh), `${w} ${q} ${dec}`);
  }
  assert.match(describe('en', analyse('BOOK', 0.45), 'greedy'), /writes BOK, not BOOK/);
  assert.match(describe('en', analyse('BOOK', 0.45), 'beam'), /writes BOOK/);
  assert.match(describe('en', analyse('BOOK', 0.05), 'beam'), /one long O/);
  assert.match(describe('zh-tw', analyse('BOOK', 0.45), 'greedy'), /貪婪解碼寫出 BOK/);
  assert.ok(!/TBD/.test(JSON.stringify(lab)));
  assert.ok(!/[数据图网络视频质样优误]/.test(JSON.stringify(lab)), 'Simplified characters');
});
