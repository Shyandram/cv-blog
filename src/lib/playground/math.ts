import { L, pick, num, fmt, svg, text, path } from './types.ts';
import type { Experiment } from './types.ts';

export function bayesDecision(prior: number, falsePositiveCost: number, observation: number) {
  const logOdds = 2 * observation + Math.log(prior / (1 - prior));
  const posterior = 1 / (1 + Math.exp(-logOdds));
  const threshold = (Math.log(falsePositiveCost) - Math.log(prior / (1 - prior))) / 2;
  const risk0 = posterior;
  const risk1 = falsePositiveCost * (1 - posterior);
  return { posterior, threshold, risk0, risk1, decision: risk1 < risk0 ? 1 : 0 };
}

export function quadraticDescent(condition: number, step: number, iterations: number) {
  const points: [number, number][] = [[3, 2]];
  let [x, y] = points[0];
  for (let i = 0; i < iterations; i++) {
    x *= 1 - step;
    y *= 1 - step * condition;
    points.push([x, y]);
  }
  return { points, objective: (x * x + condition * y * y) / 2, limit: 2 / condition };
}

export function binaryEntropy(p: number) {
  if (p === 0 || p === 1) return 0;
  return -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
}
export function binaryChannel(inputPrior: number, crossover: number) {
  const outputPrior = crossover + inputPrior * (1 - 2 * crossover);
  return {
    outputPrior,
    inputEntropy: binaryEntropy(inputPrior),
    outputEntropy: binaryEntropy(outputPrior),
    mutualInformation: binaryEntropy(outputPrior) - binaryEntropy(crossover),
    capacity: 1 - binaryEntropy(crossover),
  };
}

type CodingNode = { probability: number; key: string; left?: CodingNode; right?: CodingNode };
export function huffmanCode(probabilities: number[]) {
  const nodes: CodingNode[] = probabilities.map((probability, index) => ({ probability, key: String.fromCharCode(65 + index) }));
  while (nodes.length > 1) {
    nodes.sort((a, b) => a.probability - b.probability || a.key.localeCompare(b.key, 'en'));
    const left = nodes.shift()!;
    const right = nodes.shift()!;
    nodes.push({ probability: left.probability + right.probability, key: [left.key, right.key].sort().join(''), left, right });
  }
  const codes: Record<string, string> = {};
  function visit(node: CodingNode, prefix: string) {
    if (!node.left || !node.right) codes[node.key] = prefix || '0';
    else { visit(node.left, prefix + '0'); visit(node.right, prefix + '1'); }
  }
  visit(nodes[0], '');
  const entropy = probabilities.reduce((sum, probability) => sum - (probability ? probability * Math.log2(probability) : 0), 0);
  const averageLength = probabilities.reduce((sum, probability, index) => sum + probability * codes[String.fromCharCode(65 + index)].length, 0);
  return { codes, entropy, averageLength };
}

const axes = (left = 55, top = 35, right = 600, bottom = 240) => `<path d="M${left},${top} V${bottom} H${right}" fill="none" stroke="var(--plot-grid)"/>`;
const metricNumber = (n: number) => n !== 0 && (Math.abs(n) >= 1e5 || Math.abs(n) < 0.001) ? n.toExponential(2) : fmt(n);

export const mathExperiments: Experiment[] = [
  {
    id: 'bayes-decision', track: 'pattern-recognition',
    title: L('When should a classifier change its mind?', '分類器何時應該改變判斷？'),
    description: L('Move a decision boundary by changing class frequency and the cost of a false alarm.', '改變類別出現機率與誤報成本，觀察決策邊界如何移動。'),
    challenge: L('Keep the observation fixed. Can increasing the false-positive cost switch the decision without changing the posterior?', '固定觀測值。提高偽陽性成本，能否在後驗機率不變時改變決策？'),
    note: L('A synthetic population: X|class 0 ~ N(−1,1), X|class 1 ~ N(1,1). The slider sets the true class prior. A false negative costs 1; correct decisions cost 0. Curves show class-conditional densities, not posterior probabilities.', '合成母體：X｜類別 0 ~ N(−1,1)，X｜類別 1 ~ N(1,1)。滑桿設定真實的類別先驗機率。偽陰性成本為 1，正確決策成本為 0。曲線是類別條件機率密度，不是後驗機率。'),
    related: ['pattern-recognition/pr-01-bayes-decisions'],
    controls: [
      { key: 'prior', label: L('Prior P(class 1)', '先驗機率 P(類別 1)'), kind: 'range', min: 0.05, max: 0.95, step: 0.05, value: 0.5 },
      { key: 'cost', label: L('False-positive cost', '偽陽性成本'), kind: 'range', min: 0.25, max: 5, step: 0.25, value: 1 },
      { key: 'observation', label: L('Observed x', '觀測值 x'), kind: 'range', min: -3, max: 3, step: 0.1, value: 0.5 },
    ],
    compute(state, lang) {
      const prior = num(state, 'prior'), cost = num(state, 'cost'), observation = num(state, 'observation');
      const result = bayesDecision(prior, cost, observation);
      const px = (x: number) => 55 + (x + 4) / 8 * 545;
      const py = (y: number) => 240 - y * 440;
      let body = `<rect x="${px(result.threshold)}" y="35" width="${600 - px(result.threshold)}" height="205" fill="var(--plot-b)" opacity="0.09"/>${axes()}`;
      for (const [mean, color] of [[-1, 'a'], [1, 'b']] as const) {
        body += path(Array.from({ length: 161 }, (_, i) => {
          const x = -4 + i / 20;
          return [px(x), py(Math.exp(-0.5 * (x - mean) ** 2) / Math.sqrt(2 * Math.PI))];
        }), `stroke="var(--plot-${color})" stroke-width="3"`);
      }
      body += `<path d="M${px(result.threshold)},35 V240" stroke="var(--plot-c)" stroke-width="2" stroke-dasharray="6 4"/>`;
      body += `<path d="M${px(observation)},220 V246" stroke="currentColor" stroke-width="3"/>`;
      body += text(65, 24, pick(lang, 'Class 0 density', '類別 0 密度'), 'fill="var(--plot-a)"');
      body += text(335, 24, pick(lang, 'Class 1 density', '類別 1 密度'), 'fill="var(--plot-b)"');
      for (const x of [-4, -2, 0, 2, 4]) body += text(px(x), 263, x, 'text-anchor="middle"');
      for (const density of [0, 0.2, 0.4]) body += text(44, py(density) + 5, density, 'text-anchor="end"');
      body += text(325, 291, pick(lang, 'Shaded region: choose class 1 · horizontal axis: x', '陰影區域：選擇類別 1 · 橫軸：x'), 'text-anchor="middle"');
      return {
        svg: svg(body, 310),
        summary: pick(lang, `At x = ${fmt(observation)}, choose class ${result.decision}. Compare conditional risks: choosing 0 costs ${fmt(result.risk0)}; choosing 1 costs ${fmt(result.risk1)}. The boundary obeys 2x + log(prior odds) = log(false-positive cost).`, `在 x = ${fmt(observation)} 時，選擇類別 ${result.decision}。比較條件風險：選擇 0 的成本為 ${fmt(result.risk0)}，選擇 1 的成本為 ${fmt(result.risk1)}。邊界滿足 2x + log(先驗勝算) = log(偽陽性成本)。`),
        metrics: [
          { label: pick(lang, 'Decision boundary x', '決策邊界 x'), value: fmt(result.threshold) },
          { label: pick(lang, 'Posterior P(1 | x)', '後驗機率 P(1｜x)'), value: fmt(result.posterior) },
          { label: pick(lang, 'Required posterior for class 1', '選擇類別 1 所需的後驗機率門檻'), value: fmt(cost / (1 + cost)) },
        ],
      };
    },
  },
  {
    id: 'gradient-descent', track: 'optimization',
    title: L('Follow a gradient down a narrow valley', '沿著梯度走下狹窄谷地'),
    description: L('See why a step size that works on a round bowl can diverge on a narrow one.', '觀察適用於圓形碗面的步長，為何可能在狹窄谷地發散。'),
    challenge: L('Set k = 10. Compare step sizes 0.1, 0.2 and 0.3. What happens to the y coordinate at the stability boundary?', '設定 k = 10，比較步長 0.1、0.2 與 0.3。在穩定邊界上，y 座標有什麼變化？'),
    note: L('An exact quadratic: f(x,y) = (x² + ky²)/2, starting at (3,2). Updates are x ← (1−η)x and y ← (1−ηk)y. The contour plot has a fixed viewing window; displayed objective values are never clipped.', '精確的二次函數：f(x,y) = (x² + ky²)/2，起點為 (3,2)。更新式為 x ← (1−η)x、y ← (1−ηk)y。等高線圖使用固定視窗，顯示的目標函數值不會截斷。'),
    related: ['optimization/opt-06-gradient-newton'],
    predict: { question: L("The curvature ratio is k = 5. Which step size makes the iterate blow up instead of settling?", "曲率比是 k = 5。哪一個步長會讓迭代發散而不是收斂？"), options: [{ value: "0.15", label: L("η = 0.15", "η = 0.15") }, { value: "0.30", label: L("η = 0.30", "η = 0.30") }, { value: "0.45", label: L("η = 0.45", "η = 0.45") }], answer: "0.45", explain: L("The steep direction is multiplied by 1 − ηk each step, which has magnitude below 1 only when η is below 2/k = 0.4. So 0.15 and 0.30 converge (0.30 overshoots back and forth) and 0.45 diverges.", "陡峭方向每步乘上 1 − ηk，只有在 η 小於 2/k = 0.4 時，其絕對值才小於 1。因此 0.15 與 0.30 會收斂（0.30 會來回越過最低點），0.45 則發散。") },
    controls: [
      { key: 'condition', label: L('Curvature ratio k', '曲率比 k'), kind: 'range', min: 1, max: 20, step: 1, value: 5 },
      { key: 'step', label: L('Step size η', '步長 η'), kind: 'range', min: 0.01, max: 1, step: 0.01, value: 0.15 },
      { key: 'iterations', label: L('Update count', '更新次數'), kind: 'range', min: 1, max: 40, step: 1, value: 15 },
    ],
    compute(state, lang) {
      const condition = num(state, 'condition'), step = num(state, 'step');
      const result = quadraticDescent(condition, step, num(state, 'iterations'));
      const px = (x: number) => 320 + x * 66;
      const py = (y: number) => 157 - y * 40;
      let body = `<svg x="55" y="25" width="530" height="264" viewBox="55 25 530 264" overflow="hidden"><path d="M55,157 H585 M320,25 V289" stroke="var(--plot-grid)" fill="none"/>`;
      for (const level of [0.25, 1, 4, 9, 16]) body += `<ellipse cx="320" cy="157" rx="${Math.sqrt(2 * level) * 66}" ry="${Math.sqrt(2 * level / condition) * 40}" fill="none" stroke="var(--plot-grid)"/>`;
      const visible: [number, number][] = [];
      for (const point of result.points) {
        if (Math.abs(point[0]) > 4 || Math.abs(point[1]) > 3) break;
        visible.push([px(point[0]), py(point[1])]);
      }
      body += path(visible, 'stroke="var(--plot-a)" stroke-width="2.5"');
      visible.forEach(([x, y], i) => { body += `<circle cx="${x}" cy="${y}" r="${i === 0 ? 5 : 3}" fill="var(--plot-${i === 0 ? 'b' : 'a'})"/>`; });
      body += '</svg>';
      body += text(px(3) - 8, py(2) - 13, pick(lang, 'Start (3,2)', '起點 (3,2)'), 'text-anchor="end"');
      body += text(581, 178, 'x') + text(330, 34, 'y') + text(330, 177, '(0,0)');
      if (visible.length < result.points.length) body += text(320, 314, pick(lang, 'Later iterates leave the viewing window.', '後續迭代點超出繪圖視窗。'), 'text-anchor="middle" fill="var(--plot-c)"');
      const stable = step < result.limit - 1e-12;
      const boundary = Math.abs(step - result.limit) < 1e-12;
      const summary = stable
        ? pick(lang, 'Both coordinate multipliers have magnitude below 1, so the iterates converge. A negative multiplier produces an alternating path; a multiplier close to 1 or −1 produces slow progress.', '兩個座標乘數的絕對值皆小於 1，因此迭代會收斂。負乘數造成交替移動；接近 1 或 −1 的乘數則使進展緩慢。')
        : boundary ? pick(lang, 'At η = 2/k, the y multiplier is −1. Its magnitude never decreases: the path oscillates and does not converge to the minimum.', '在 η = 2/k 時，y 的乘數為 −1，振幅不會縮小；路徑持續振盪，無法收斂至最小值。')
          : pick(lang, 'The step exceeds 2/k. The y multiplier has magnitude above 1, so this trajectory diverges. Large objective values below are the actual computed values.', '步長超過 2/k。y 乘數的絕對值大於 1，因此路徑發散。下方的大目標值是實際計算結果。');
      return { svg: svg(body, 335), summary, metrics: [
        { label: pick(lang, 'Final objective', '最終目標函數值'), value: metricNumber(result.objective) },
        { label: pick(lang, 'Converges when 0 < η <', '收斂條件：0 < η <'), value: fmt(result.limit) },
        { label: pick(lang, 'Coordinate multipliers (x, y)', '座標乘數 (x, y)'), value: `${fmt(1 - step)}, ${fmt(1 - step * condition)}` },
      ] };
    },
  },
  {
    id: 'binary-channel', track: 'information-theory',
    title: L('How much information survives noise?', '雜訊之後還留下多少資訊？'),
    description: L('Separate source uncertainty, transmitted information and channel capacity.', '區分來源的不確定性、實際傳遞的資訊與通道容量。'),
    challenge: L('With crossover p = 0.1, move the input prior away from 0.5. Does the channel capacity change? Does the transmitted information change?', '固定翻轉機率 p = 0.1，將輸入先驗機率移離 0.5。通道容量會改變嗎？實際傳遞的資訊會改變嗎？'),
    note: L('Independent bit flips define a binary symmetric channel. q = P(X=1), p = P(Y≠X), and P(Y=1) = p + q(1−2p). All values are bits per channel use. Capacity maximizes mutual information over q; here q = 0.5 achieves it.', '獨立的位元翻轉構成二元對稱通道。q = P(X=1)、p = P(Y≠X)，且 P(Y=1) = p + q(1−2p)。數值單位皆為每次通道使用的位元數。容量是對 q 最大化後的互資訊；此處 q = 0.5 可達到容量。'),
    related: ['information-theory/it-01-entropy-mutual-information', 'information-theory/it-04-channel-capacity'],
    predict: { question: L("Move the input probability q = P(X=1) away from 0.5 with the flip probability fixed. What happens to the channel capacity C?", "固定位元翻轉機率，把輸入機率 q = P(X=1) 從 0.5 移開。通道容量 C 會怎麼變？"), options: [{ value: "down", label: L("C shrinks", "C 變小") }, { value: "same", label: L("C stays the same", "C 不變") }, { value: "up", label: L("C grows", "C 變大") }], answer: "same", explain: L("Capacity is the largest I(X;Y) over all input distributions, so it is a property of the channel (here 1 − h₂(p)). Moving q lowers I(X;Y) below C; it does not change C.", "容量是所有輸入分布中最大的 I(X;Y)，所以它是通道本身的性質（此處為 1 − h₂(p)）。移動 q 會讓 I(X;Y) 低於 C，但不會改變 C。") },
    controls: [
      { key: 'prior', label: L('Input prior q = P(X=1)', '輸入先驗機率 q = P(X=1)'), kind: 'range', min: 0, max: 1, step: 0.01, value: 0.5 },
      { key: 'crossover', label: L('Bit-flip probability p', '位元翻轉機率 p'), kind: 'range', min: 0, max: 0.5, step: 0.01, value: 0.1 },
    ],
    compute(state, lang) {
      const prior = num(state, 'prior'), crossover = num(state, 'crossover');
      const result = binaryChannel(prior, crossover);
      const px = (q: number) => 55 + q * 545, py = (v: number) => 240 - v * 190;
      let body = axes();
      body += path(Array.from({ length: 101 }, (_, i) => [px(i / 100), py(binaryEntropy(i / 100))]), 'stroke="var(--plot-b)" stroke-width="2" stroke-dasharray="5 4"');
      body += path(Array.from({ length: 101 }, (_, i) => [px(i / 100), py(binaryChannel(i / 100, crossover).mutualInformation)]), 'stroke="var(--plot-a)" stroke-width="3"');
      body += `<path d="M55,${py(result.capacity)} H600" stroke="var(--plot-c)" stroke-dasharray="3 5"/><circle cx="${px(prior)}" cy="${py(result.mutualInformation)}" r="6" fill="var(--plot-a)"/>`;
      body += text(60, 22, 'H(X)', 'fill="var(--plot-b)"') + text(215, 22, 'I(X;Y)', 'fill="var(--plot-a)"') + text(400, 22, pick(lang, 'Capacity', '容量'), 'fill="var(--plot-c)"');
      for (const q of [0, 0.25, 0.5, 0.75, 1]) body += text(px(q), 264, q, 'text-anchor="middle"');
      body += text(42, 244, '0', 'text-anchor="end"') + text(42, 54, '1', 'text-anchor="end"');
      body += text(325, 292, pick(lang, 'Horizontal axis: input prior q · vertical axis: bits', '橫軸：輸入先驗機率 q · 縱軸：位元'), 'text-anchor="middle"');
      return { svg: svg(body, 310), summary: pick(lang,
        `The output has P(Y=1) = ${fmt(result.outputPrior)}. I(X;Y) = H(Y) − h₂(p) = ${fmt(result.outputEntropy)} − ${fmt(binaryEntropy(crossover))} = ${fmt(result.mutualInformation)}. Random output alone does not imply useful information: at p = 0.5 the output is independent of the input.`,
        `輸出機率 P(Y=1) = ${fmt(result.outputPrior)}。I(X;Y) = H(Y) − h₂(p) = ${fmt(result.outputEntropy)} − ${fmt(binaryEntropy(crossover))} = ${fmt(result.mutualInformation)}。輸出具有隨機性不代表攜帶有用資訊：p = 0.5 時，輸出與輸入互相獨立。`), metrics: [
          { label: 'H(X)', value: fmt(result.inputEntropy) }, { label: 'H(Y)', value: fmt(result.outputEntropy) },
          { label: 'I(X;Y)', value: fmt(result.mutualInformation) }, { label: pick(lang, 'Capacity 1 − h₂(p)', '容量 1 − h₂(p)'), value: fmt(result.capacity) },
        ] };
    },
  },
  {
    id: 'prefix-coding', track: 'information-theory',
    title: L('Give common symbols shorter codes', '讓常見符號使用較短的編碼'),
    description: L('Build a binary prefix code and compare its average length with source entropy.', '建立二元前綴碼，比較平均碼長與來源熵。'),
    challenge: L('Compare P(A) = 0.25 with P(A) = 0.7. Why does the optimal code change in discrete jumps while entropy changes smoothly?', '比較 P(A) = 0.25 與 P(A) = 0.7。為何最佳編碼會跳躍改變，熵卻平滑變化？'),
    note: L('Huffman coding repeatedly merges the two least probable nodes. Ties use alphabetical order. B, C and D each have probability (1−P(A))/3. The expected length assumes this source distribution; the example message has its own symbol counts. Codebook transmission cost is excluded.', 'Huffman 編碼反覆合併機率最小的兩個節點，同機率時依字母排序。B、C、D 的機率各為 (1−P(A))/3。平均碼長依此來源分布計算；範例訊息則有自己的符號次數。此處未計入傳送碼表的成本。'),
    related: ['information-theory/it-03-lossless-universal-coding'],
    controls: [
      { key: 'dominant', label: L('Source probability P(A)', '來源機率 P(A)'), kind: 'range', min: 0.1, max: 0.9, step: 0.01, value: 0.7 },
      { key: 'message', label: L('Example message', '範例訊息'), kind: 'select', value: 'ABACABAD', options: [
        { value: 'ABACABAD', label: L('ABACABAD', 'ABACABAD') },
        { value: 'AAAABAAA', label: L('AAAABAAA', 'AAAABAAA') },
        { value: 'ABCDABCD', label: L('ABCDABCD', 'ABCDABCD') },
      ] },
    ],
    compute(state, lang) {
      const dominant = num(state, 'dominant');
      const probabilities = [dominant, ...Array(3).fill((1 - dominant) / 3)] as number[];
      const result = huffmanCode(probabilities);
      const message = String(state.message);
      const words = [...message].map(symbol => result.codes[symbol]);
      let body = text(45, 28, pick(lang, 'Symbol', '符號')) + text(150, 28, pick(lang, 'Source probability', '來源機率')) + text(430, 28, pick(lang, 'Code', '編碼')) + text(550, 28, pick(lang, 'Length', '碼長'));
      probabilities.forEach((probability, i) => {
        const y = 66 + i * 45, symbol = String.fromCharCode(65 + i);
        body += text(60, y, symbol) + `<rect x="150" y="${y - 18}" width="${probability * 200}" height="23" rx="3" fill="var(--plot-a)" opacity="0.28"/>`;
        body += text(155, y, fmt(probability)) + text(440, y, result.codes[symbol], 'font-family="monospace"') + text(565, y, result.codes[symbol].length);
      });
      body += text(45, 264, `${message} → ${words.join(' ')}`, 'font-family="monospace"');
      body += text(45, 295, pick(lang, `Message: ${words.join('').length} bits · fixed 2-bit code: ${message.length * 2} bits`, `訊息：${words.join('').length} 位元 · 固定 2 位元編碼：${message.length * 2} 位元`));
      return { svg: svg(body, 320), summary: pick(lang,
        `No codeword is the prefix of another, so concatenated bits can be decoded without separators. Expected length is Σ p(symbol) × code length = ${fmt(result.averageLength)} bits/symbol. Entropy is ${fmt(result.entropy)}; their gap is ${fmt(result.averageLength - result.entropy)}. Spaces in the example only show codeword boundaries.`,
        `沒有任何碼字是其他碼字的前綴，因此串接的位元不需要分隔符號就能解碼。平均碼長為 Σ p(符號) × 碼長 = ${fmt(result.averageLength)} 位元／符號。熵為 ${fmt(result.entropy)}，兩者差距為 ${fmt(result.averageLength - result.entropy)}。範例中的空格只是標示碼字邊界。`), metrics: [
          { label: pick(lang, 'Source entropy', '來源熵'), value: fmt(result.entropy) },
          { label: pick(lang, 'Expected code length', '平均碼長'), value: fmt(result.averageLength) },
          { label: pick(lang, 'Redundancy (bits/symbol)', '冗餘（位元／符號）'), value: fmt(result.averageLength - result.entropy) },
        ] };
    },
  },
];
