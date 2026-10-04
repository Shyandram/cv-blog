import { L, pick, num, fmt, svg, text, path, type Experiment, type Control } from './types.ts';
import { binaryEntropy } from './math.ts';
const range = (key: string, en: string, zh: string, min: number, max: number, step: number, value: number): Control => ({ key, label: L(en, zh), kind: 'range', min, max, step, value });
export function bernoulliMismatch(p: number, q: number) { const entropy = binaryEntropy(p), cross = -(p === 0 ? 0 : p * Math.log2(q)) - (p === 1 ? 0 : (1 - p) * Math.log2(1 - q)); return { entropy, cross, kl: cross - entropy }; }
export function binomial(n: number, p: number) { return Array.from({ length: n + 1 }, (_, k) => { let count = 1; for (let j = 1; j <= k; j++)
    count *= (n - j + 1) / j; return { k, count: Math.round(count), probability: count * p ** k * (1 - p) ** (n - k), surprise: -(k * Math.log2(p) + (n - k) * Math.log2(1 - p)) / n }; }); }
export function typicalSet(n: number, p: number, epsilon: number) { const h = binaryEntropy(p), rows = binomial(n, p).map(r => ({ ...r, typical: Math.abs(r.surprise - h) <= epsilon + 1e-12 })); return { rows, h, mass: rows.filter(r => r.typical).reduce((s, r) => s + r.probability, 0), fraction: rows.filter(r => r.typical).reduce((s, r) => s + r.count, 0) / 2 ** n }; }
export function waterFill(power: number, noise2: number) { const allocation = [Math.min(power, Math.max(0, (power + noise2 - 1) / 2)), 0]; allocation[1] = power - allocation[0]; const capacity = .5 * Math.log2(1 + allocation[0]) + .5 * Math.log2(1 + allocation[1] / noise2), equal = .5 * Math.log2(1 + power / 2) + .5 * Math.log2(1 + power / (2 * noise2)); return { allocation, capacity, equal, level: 1 + allocation[0] }; }
export function binaryRateDistortion(p: number, d: number) { const q = (p - d) / (1 - 2 * d || 1); const joint = [(1 - q) * (1 - d), q * d, (1 - q) * d, q * (1 - d)]; return { rate: Math.max(0, binaryEntropy(p) - binaryEntropy(d)), q, joint, distortion: joint[1] + joint[2] }; }
export function testingErrors(n: number, k: number) { const nullRows = binomial(n, .3), alternative = binomial(n, .7); return { nullRows, alternative, falseAlarm: nullRows.filter(r => r.k >= k).reduce((s, r) => s + r.probability, 0), miss: alternative.filter(r => r.k < k).reduce((s, r) => s + r.probability, 0) }; }
export const chapterInformationExperiments: Experiment[] = [
    {
        id: 'probability-mismatch', track: 'information-theory', title: L('Separate entropy from model mismatch', '分開看熵與模型失配'), description: L('Change the true probability and the model probability; measure the extra coding cost caused by a wrong model.', '改變真實機率與模型機率，計算模型錯誤帶來的額外編碼成本。'), related: ['information-theory/it-01-entropy-mutual-information', 'ml/ml-01b-probability-information'],
        challenge: L('Fix p=0.8. Find the q that minimizes cross-entropy, then compare q=0.2.', '固定 p=0.8，找出使交叉熵最小的 q，再比較 q=0.2。'),
        note: L('For Bernoulli sources, H(p,q)=−p log₂q−(1−p)log₂(1−q)=H(p)+D(p∥q). q stays strictly between 0 and 1. These are expected ideal bit costs, not measured file sizes.', 'Bernoulli 來源的 H(p,q)=−p log₂q−(1−p)log₂(1−q)=H(p)+D(p∥q)。q 嚴格位於 0 與 1 之間。這是期望的理想位元成本，不是實測檔案大小。'),
        controls: [range('p', 'True P(1)', '真實 P(1)', 0, 1, .01, .8), range('q', 'Model Q(1)', '模型 Q(1)', .01, .99, .01, .5)],
        compute(s, lang) {
            const p = num(s, 'p'), q = num(s, 'q'), r = bernoulliMismatch(p, q);
            let b = text(50, 25, pick(lang, 'Expected bits per symbol (shared scale 0–7)', '每符號期望位元數（共用尺度 0–7）'));
            [[r.entropy, 'H(P)', 'a'], [r.cross, 'H(P,Q)', 'b'], [r.kl, 'D(P∥Q)', 'c']].forEach(([v, label, col], i) => { b += text(30, 75 + i * 70, String(label)) + `<rect x="145" y="${52 + i * 70}" width="${Number(v) * 58}" height="30" fill="var(--plot-${col})"/>` + text(155 + Number(v) * 58, 74 + i * 70, fmt(Number(v))); });
            return { svg: svg(b), summary: pick(lang, `H(P,Q) = ${fmt(r.entropy)} + ${fmt(r.kl)} bits. Matching probabilities removes the KL penalty, but cannot remove source uncertainty.`, `H(P,Q) = ${fmt(r.entropy)} + ${fmt(r.kl)} 位元。符合真實機率可消除 KL 懲罰，卻無法消除來源的不確定性。`), metrics: [{ label: pick(lang, 'Entropy', '熵'), value: fmt(r.entropy) }, { label: pick(lang, 'Cross-entropy', '交叉熵'), value: fmt(r.cross) }, { label: pick(lang, 'KL divergence', 'KL 散度'), value: fmt(r.kl) }] };
        },
    },
    {
        id: 'typical-sequences', track: 'information-theory', title: L('A likely block is not a typical set', '高機率區塊與典型集合不同'), description: L('Compare the probability mass of a typical set with its fraction of all possible binary blocks.', '比較典型集合的機率質量與它占所有二元區塊的比例。'), related: ['information-theory/it-02-typicality-entropy-rates'],
        challenge: L('Fix p=0.2 and increase block length. Does the typical set contain most probability while using most of all binary strings?', '固定 p=0.2 並增加區塊長度。典型集合占大部分機率時，也占大部分二元字串嗎？'),
        note: L('Independent Bernoulli(p) blocks of length n. A block is typical when |−log₂ P(block)/n−H(p)|≤ε. Each bar aggregates all C(n,k) strings with k ones. Small n can give an empty typical set; AEP is asymptotic.', '長度 n 的獨立 Bernoulli(p) 區塊。若 |−log₂ P(區塊)/n−H(p)|≤ε 即為典型。每根柱合併所有含 k 個 1 的 C(n,k) 種字串。小 n 的典型集合可能為空；AEP 是漸近結果。'),
        controls: [range('n', 'Block length n', '區塊長度 n', 4, 32, 1, 16), range('p', 'P(1)', 'P(1)', .05, .95, .05, .2), range('eps', 'Typicality tolerance ε (bits)', '典型容許差 ε（位元）', .02, .5, .02, .2)],
        compute(s, lang) {
            const n = num(s, 'n'), r = typicalSet(n, num(s, 'p'), num(s, 'eps')), max = Math.max(...r.rows.map(x => x.probability)), w = 550 / (n + 1);
            const b = r.rows.map(x => `<rect x="${45 + x.k * w}" y="${250 - 200 * x.probability / max}" width="${w * .8}" height="${200 * x.probability / max}" fill="var(--plot-${x.typical ? 'c' : 'a'})"/>`).join('');
            return { svg: svg(text(45, 25, pick(lang, 'Green: typical · Blue: other · Height: probability mass', '綠：典型 · 藍：其餘 · 高度：機率質量')) + b + text(45, 280, 'k = 0') + text(595, 280, `k = ${n}`, 'text-anchor="end"')), summary: pick(lang, 'Typicality concerns information per symbol. A single likely string and a set carrying most probability are different objects.', '典型性關注每符號的資訊量。單一高機率字串與承載大部分機率的集合是不同概念。'), metrics: [{ label: pick(lang, 'Typical probability mass', '典型機率質量'), value: fmt(r.mass, 4) }, { label: pick(lang, 'Fraction of all strings', '占全部字串比例'), value: r.fraction.toPrecision(4) }, { label: 'H(p) (bits)', value: fmt(r.h) }] };
        },
    },
    {
        id: 'gaussian-water-filling', track: 'information-theory', title: L('Allocate power where it carries more information', '把功率分配到資訊效益較高的通道'), description: L('Compare equal allocation with water-filling across two parallel Gaussian channels.', '在兩條平行高斯通道上比較平均分配與注水法。'), related: ['information-theory/it-05-continuous-gaussian-channels'],
        challenge: L('Give the second channel high noise. Increase the total power until it first receives a positive allocation.', '讓第二條通道有較大的雜訊，增加總功率，找出它開始獲得正功率的時刻。'),
        note: L('Independent real Gaussian channels, noise variances N₁=1 and N₂, total power P. Capacity per joint use is ½Σ log₂(1+Pᵢ/Nᵢ); water-filling uses Pᵢ=max(μ−Nᵢ,0). This assumes optimal Gaussian signaling, not a finite code.', '獨立實數高斯通道，雜訊變異數 N₁=1、N₂，總功率 P。每次聯合使用的容量為 ½Σ log₂(1+Pᵢ/Nᵢ)，注水法採 Pᵢ=max(μ−Nᵢ,0)。假設最佳高斯輸入，不代表有限長度編碼。'),
        controls: [range('power', 'Total power P', '總功率 P', 0, 12, .2, 4), range('noise', 'Second noise variance N₂', '第二條雜訊變異數 N₂', 1, 10, .2, 4)],
        compute(s, lang) {
            const n = num(s, 'noise'), r = waterFill(num(s, 'power'), n), scale = 13;
            let b = text(45, 25, pick(lang, 'Gray: noise floor · Blue: allocated power', '灰：雜訊底線 · 藍：分配功率'));
            [1, n].forEach((noise, i) => { const x = 130 + 240 * i; b += `<rect x="${x}" y="${260 - scale * noise}" width="100" height="${scale * noise}" fill="var(--plot-muted)" fill-opacity=".3"/><rect x="${x}" y="${260 - scale * (noise + r.allocation[i])}" width="100" height="${scale * r.allocation[i]}" fill="var(--plot-a)"/>` + text(x + 50, 285, `N${i + 1}=${fmt(noise)}`, 'text-anchor="middle"'); });
            b += path([[80, 260 - scale * r.level], [550, 260 - scale * r.level]], 'stroke="var(--plot-b)" stroke-dasharray="5 4"');
            return { svg: svg(b), summary: pick(lang, `P₁=${fmt(r.allocation[0])}, P₂=${fmt(r.allocation[1])}. Active channels share a water level; a noise floor above it receives no power.`, `P₁=${fmt(r.allocation[0])}，P₂=${fmt(r.allocation[1])}。啟用通道共用水位，高於水位的雜訊底線不分配功率。`), metrics: [{ label: pick(lang, 'Optimal bits / joint use', '最佳位元／聯合使用'), value: fmt(r.capacity) }, { label: pick(lang, 'Equal-allocation bits', '平均分配位元'), value: fmt(r.equal) }, { label: pick(lang, 'Capacity gain', '容量增益'), value: fmt(r.capacity - r.equal) }] };
        },
    },
    {
        id: 'binary-rate-distortion', track: 'information-theory', title: L('How much rate buys a given error?', '多少碼率能換得指定誤差？'), description: L('Inspect the Bernoulli rate–distortion curve and an attaining joint distribution under Hamming error.', '觀察 Bernoulli 碼率失真曲線，以及在 Hamming 誤差下達到它的聯合分布。'), related: ['information-theory/it-06-rate-distortion'],
        challenge: L('Move distortion to its maximum. Why is sending no bits sufficient even though some symbols are wrong?', '把失真移到最大值。為何即使部分符號錯誤，也不必傳送任何位元？'),
        note: L('X~Bernoulli(p), p≤0.5, Hamming distortion D≤p. R(D)=h₂(p)−h₂(D). Construct X=Y XOR Z with independent Z~Bernoulli(D) and P(Y=1)=(p−D)/(1−2D). At p=D=0.5 choose constant Y=0. The curve is an asymptotic block-coding limit, not the rate of this single-symbol channel.', 'X~Bernoulli(p)，p≤0.5，Hamming 失真 D≤p。R(D)=h₂(p)−h₂(D)。令 X=Y XOR Z，獨立的 Z~Bernoulli(D)，且 P(Y=1)=(p−D)/(1−2D)。p=D=0.5 時選常數 Y=0。曲線是漸近區塊編碼極限，不是此單符號通道的實際碼率。'),
        controls: [range('p', 'Source P(1)', '來源 P(1)', .05, .5, .05, .3), range('fraction', 'Allowed distortion / p', '容許失真／p', 0, 1, .02, .4)],
        compute(s, lang) {
            const p = num(s, 'p'), d = p * num(s, 'fraction'), r = binaryRateDistortion(p, d), pts = Array.from({ length: 101 }, (_, i) => [50 + 270 * i / 100, 245 - 200 * (binaryEntropy(p) - binaryEntropy(p * i / 100))] as [
                number,
                number
            ]);
            let b = path([[50, 35], [50, 245], [320, 245]], 'stroke="var(--plot-grid)"') + path(pts, 'stroke="var(--plot-a)" stroke-width="3"') + `<circle cx="${50 + 270 * d / p}" cy="${245 - 200 * r.rate}" r="6" fill="var(--plot-b)"/>` + text(50, 280, 'D=0') + text(320, 280, `D=${fmt(p)}`, 'text-anchor="end"') + text(50, 25, pick(lang, 'R(D): bits / symbol', 'R(D)：位元／符號')) + text(390, 45, 'P(X,Y)');
            r.joint.forEach((v, i) => { const x = 385 + (i % 2) * 105, y = 80 + Math.floor(i / 2) * 70; b += `<rect x="${x}" y="${y}" width="94" height="58" fill="var(--plot-${i === 1 || i === 2 ? 'b' : 'a'})" fill-opacity=".12"/>` + text(x + 47, y + 22, `X${Math.floor(i / 2)} Y${i % 2}`, 'text-anchor="middle"') + text(x + 47, y + 44, fmt(v), 'text-anchor="middle"'); });
            return { svg: svg(b), summary: pick(lang, 'Orange off-diagonal cells are reconstruction errors. Their total equals D; the joint distribution retains the selected source marginal.', '橙色非對角格是重建錯誤，總和等於 D；聯合分布保留所選的來源邊際分布。'), metrics: [{ label: 'D', value: fmt(d) }, { label: 'R(D)', value: fmt(r.rate) }, { label: 'P(Y=1)', value: fmt(r.q) }] };
        },
    },
    {
        id: 'hypothesis-errors', track: 'information-theory', title: L('Trade false alarms for missed alternatives', '在誤警與漏判之間取捨'), description: L('Slide a decision threshold across two exact binomial distributions and compute both error probabilities.', '在兩個精確二項分布上移動決策閾值，計算兩種錯誤機率。'), related: ['information-theory/it-07-statistics-decision-growth'],
        challenge: L('Try thresholds that always accept or always reject H₀. Then increase n while keeping the threshold near n/2.', '試試永遠接受或永遠拒絕 H₀ 的閾值，再增加 n 並維持閾值接近 n/2。'),
        note: L('n independent binary observations. H₀:p=0.3, H₁:p=0.7; reject H₀ if number of ones k≥threshold. Curves show exact finite-sample probability masses. False alarm is conditional on H₀, miss on H₁; neither is a posterior probability.', 'n 筆獨立二元觀測。H₀:p=0.3、H₁:p=0.7；1 的數量 k≥閾值時拒絕 H₀。曲線是精確有限樣本機率質量。誤警以 H₀ 為條件、漏判以 H₁ 為條件，皆不是後驗機率。'),
        controls: [range('n', 'Observation count n', '觀測數 n', 2, 30, 1, 10), range('threshold', 'Threshold position (0: always reject, 1: never)', '閾值位置（0：總是拒絕，1：永不拒絕）', 0, 1, .02, .5)],
        compute(s, lang) {
            const n = num(s, 'n'), k = Math.round((n + 1) * num(s, 'threshold')), r = testingErrors(n, k), max = Math.max(...r.nullRows.map(x => x.probability));
            let b = text(45, 25, pick(lang, 'Blue: H₀ · Orange: H₁ · Shade: reject H₀', '藍：H₀ · 橙：H₁ · 底色：拒絕 H₀'));
            const boundary = 45 + 550 * Math.min(1, Math.max(0, (k - .5) / n));
            b += `<rect x="${boundary}" y="40" width="${595 - boundary}" height="215" fill="var(--plot-c)" fill-opacity=".08"/>`;
            [r.nullRows, r.alternative].forEach((rows, i) => b += path(rows.map(x => [45 + 550 * x.k / n, 250 - 190 * x.probability / max]), `stroke="var(--plot-${i ? 'b' : 'a'})" stroke-width="3"`));
            return { svg: svg(b + text(45, 285, 'k=0') + text(595, 285, `k=${n}`, 'text-anchor="end"')), summary: pick(lang, `Reject H₀ for k ≥ ${k}. Changing a threshold changes both errors; increasing evidence can improve their tradeoff.`, `k ≥ ${k} 時拒絕 H₀。調整閾值會同時改變兩種錯誤，增加證據量則可能改善取捨。`), metrics: [{ label: pick(lang, 'False alarm', '誤警'), value: fmt(r.falseAlarm, 5) }, { label: pick(lang, 'Miss', '漏判'), value: fmt(r.miss, 5) }, { label: pick(lang, 'Detection probability', '偵測機率'), value: fmt(1 - r.miss, 5) }] };
        },
    },
    {
        id: 'information-synergy', track: 'information-theory', title: L('Two useless clues can be useful together', '兩個無用線索合起來可以有用'), description: L('Compute information about a noisy XOR label from each independent bit and from their pair.', '計算各自獨立的兩個位元及其組合，對帶雜訊 XOR 標籤提供多少資訊。'), related: ['information-theory/it-08-networks-information-learning'],
        challenge: L('Set label noise to zero. Why does each feature have zero information while their pair has one bit?', '把標籤雜訊設為零。為何單一特徵的資訊為零，組合卻有一位元？'),
        note: L('A and B are independent fair bits, Y=A XOR B XOR E, E~Bernoulli(ε) independent. Thus I(A;Y)=I(B;Y)=0 and I(A,B;Y)=1−h₂(ε). This population identity does not establish a finite-sample generalization bound.', 'A、B 為獨立公平位元，Y=A XOR B XOR E，獨立的 E~Bernoulli(ε)。因此 I(A;Y)=I(B;Y)=0，而 I(A,B;Y)=1−h₂(ε)。此母體恆等式不等於有限樣本的泛化界限。'),
        controls: [range('noise', 'Label flip probability ε', '標籤翻轉機率 ε', 0, .5, .01, .1)],
        compute(s, lang) {
            const e = num(s, 'noise'), info = 1 - binaryEntropy(e);
            let b = text(50, 30, pick(lang, 'Conditional table: P(Y=1 | A,B)', '條件機率表：P(Y=1 | A,B)'));
            for (let a = 0; a < 2; a++)
                for (let c = 0; c < 2; c++) {
                    const v = a === c ? e : 1 - e, x = 70 + c * 245, y = 60 + a * 105;
                    b += `<rect x="${x}" y="${y}" width="220" height="90" fill="var(--plot-a)" fill-opacity="${.08 + .65 * v}"/>` + text(x + 110, y + 30, `A=${a}, B=${c}`, 'text-anchor="middle"') + text(x + 110, y + 64, fmt(v), 'text-anchor="middle"');
                }
            return { svg: svg(b), summary: pick(lang, 'Averaging either row or column gives 0.5: a single feature does not change the label distribution. Observing both can.', '任一列或欄的平均都是 0.5：單看一個特徵不改變標籤分布，兩個一起看則可能改變。'), metrics: [{ label: 'I(A;Y)', value: '0 bits' }, { label: 'I(B;Y)', value: '0 bits' }, { label: 'I(A,B;Y)', value: `${fmt(info)} bits` }, { label: 'I(A;Y|B)', value: `${fmt(info)} bits` }] };
        },
    }
];
