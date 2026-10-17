import type { Experiment, Control, State } from "./types.ts";
import { L, pick, fmt, svg, text, path, escape } from "./types.ts";
const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
const n = (s: State, k: string) => Number(s[k]);
const range = (
  key: string,
  en: string,
  zh: string,
  value: number,
  min: number,
  max: number,
  step: number,
): Control => ({ key, label: L(en, zh), kind: "range", value, min, max, step });
const metric = (
  lang: "en" | "zh-tw",
  en: string,
  zh: string,
  value: number | string,
) => ({
  label: pick(lang, en, zh),
  value: typeof value === "number" ? fmt(value, 5) : value,
});
const mse = (a: number[], b: number[]) =>
  a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0) / a.length;
const lines = (
  sets: number[][],
  labels: string[],
  lo?: number,
  hi?: number,
) => {
  const low = lo ?? Math.min(...sets.flat()),
    high = hi ?? Math.max(...sets.flat()),
    span = Math.max(high - low, 1e-9);
  return (
    text(34, 42, fmt(high)) +
    text(34, 244, fmt(low)) +
    sets
      .map(
        (a, k) =>
          path(
            a.map((v, i) => [
              70 + (530 * i) / Math.max(a.length - 1, 1),
              240 - (190 * (v - low)) / span,
            ]),
            `stroke="var(--plot-${"abcd"[k]})" stroke-width="2.5"${k === 0 ? ' stroke-dasharray="5 3"' : ""}`,
          ) +
          text(70 + k * 180, 280, labels[k], `fill="var(--plot-${"abcd"[k]})"`),
      )
      .join("")
  );
};
const IP = (slug: string) => `image-processing/${slug}`;
const DL = (slug: string) => `deep-learning/${slug}`;

/** A real diagonal inverse problem in a cosine basis. DC is left unchanged. */
/** A deliberately restricted circular snake: radius is the only degree of freedom. */
export function radialSnake(initial: number, tension: number, steps: number) {
  const energy = (r: number) =>
    tension * r * r - Math.exp(-(((r - 1) / 0.16) ** 2));
  const derivative = (r: number) =>
    2 * tension * r +
    ((2 * (r - 1)) / 0.16 ** 2) * Math.exp(-(((r - 1) / 0.16) ** 2));
  let r = initial;
  const history = [r];
  for (let i = 0; i < steps; i++) {
    const g = derivative(r);
    let rate = 0.025;
    for (
      let j = 0;
      j < 20 && energy(clamp(r - rate * g, 0, 1.8)) > energy(r);
      j++
    )
      rate *= 0.5;
    r = clamp(r - rate * g, 0, 1.8);
    history.push(r);
  }
  return {
    radius: r,
    history,
    energy,
    initialEnergy: energy(initial),
    finalEnergy: energy(r),
  };
}
export function classificationThreshold(
  threshold: number,
  falsePositiveCost: number,
) {
  const scores = [0.95, 0.9, 0.82, 0.73, 0.66, 0.58, 0.47, 0.35, 0.21, 0.08];
  const truth = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0];
  const prediction = scores.map((score) => Number(score >= threshold));
  let tp = 0,
    fp = 0,
    fn = 0,
    tn = 0;
  prediction.forEach((p, i) => {
    if (p) {
      if (truth[i]) tp++;
      else fp++;
    } else {
      if (truth[i]) fn++;
      else tn++;
    }
  });
  return {
    scores,
    truth,
    prediction,
    tp,
    fp,
    fn,
    tn,
    precision: tp + fp ? tp / (tp + fp) : null,
    recall: tp / (tp + fn),
    f1: (2 * tp) / (2 * tp + fp + fn),
    accuracy: (tp + tn) / 10,
    cost: (falsePositiveCost * fp + fn) / 10,
  };
}
/** ROC and precision-recall points from the same fixed scores, one vertex per distinct score (threshold = that score). */
export function rocCurve() {
  const { scores, truth } = classificationThreshold(0.5, 1);
  const positives = truth.reduce((a, v) => a + v, 0),
    negatives = truth.length - positives;
  const roc: { threshold: number; fpr: number; tpr: number; precision: number }[] = [
    { threshold: Infinity, fpr: 0, tpr: 0, precision: NaN },
  ];
  [...new Set(scores)]
    .sort((x, y) => y - x)
    .forEach((threshold) => {
      const r = classificationThreshold(threshold, 1);
      roc.push({
        threshold,
        fpr: r.fp / negatives,
        tpr: r.tp / positives,
        precision: r.tp / (r.tp + r.fp),
      });
    });
  let auc = 0;
  roc.slice(1).forEach((p, i) => {
    auc += ((p.fpr - roc[i].fpr) * (p.tpr + roc[i].tpr)) / 2;
  });
  return { roc, auc, positives, negatives };
}
export const totalVariation = (values: number[]) =>
  values.slice(1).reduce((a, v, i) => a + Math.abs(v - values[i]), 0);
export function ambiguousRestoration(probability: number, mix: number) {
  const a = [0, 0, 1, 1, 0, 0, 1, 1],
    b = [1, 1, 0, 0, 1, 1, 0, 0];
  const mean = a.map((v, i) => probability * v + (1 - probability) * b[i]);
  const output = mean.map((v, i) => (1 - mix) * v + mix * a[i]);
  return {
    a,
    b,
    mean,
    output,
    risk: probability * mse(a, output) + (1 - probability) * mse(b, output),
    riskMean: probability * mse(a, mean) + (1 - probability) * mse(b, mean),
  };
}
export function trackingAssignment(
  motionWeight: number,
  appearanceError: number,
  crossing: number,
) {
  const predictions = [0.3, 0.7],
    detections = [0.3 + 0.4 * crossing, 0.7 - 0.4 * crossing];
  const cost = predictions.map((x, i) =>
    detections.map(
      (y, j) =>
        motionWeight * Math.abs(x - y) +
        (1 - motionWeight) * (i === j ? appearanceError : 1 - appearanceError),
    ),
  );
  const direct = cost[0][0] + cost[1][1],
    swapped = cost[0][1] + cost[1][0];
  return {
    predictions,
    detections,
    cost,
    direct,
    swapped,
    assignment: direct <= swapped ? [0, 1] : [1, 0],
    ties: Math.abs(direct - swapped) < 1e-10,
  };
}
export function worldRollout(
  coefficient: number,
  steps: number,
  action: number,
  refresh: number,
) {
  const truth = [1],
    open = [1],
    corrected = [1],
    one = [];
  for (let t = 0; t < steps; t++) {
    const u = action * Math.sin(t * 0.45);
    truth.push(0.96 * truth[t] + u);
    open.push(coefficient * open[t] + u);
    one.push(coefficient * truth[t] + u);
    corrected.push(
      coefficient *
        (refresh > 0 && t % refresh === 0 ? truth[t] : corrected[t]) +
        u,
    );
  }
  return {
    truth,
    open,
    corrected,
    oneError: mse(one, truth.slice(1)),
    openError: mse(open.slice(1), truth.slice(1)),
    correctedError: mse(corrected.slice(1), truth.slice(1)),
  };
}

export const chapterVisionExperiments: Experiment[] = [
  {
    id: "contour-energy",
    track: "image-processing",
    title: L(
      "IP 11 · Why a contour can miss the boundary",
      "IP 11・輪廓為何可能錯過邊界",
    ),
    description: L(
      "Move a circular contour through a narrow edge-attraction energy and observe initialization and tension.",
      "讓圓形輪廓沿狹窄的邊緣吸引能量移動，觀察初始化與張力的影響。",
    ),
    challenge: L(
      "Use the same tension but start at radius 0.5 and 1.2. Why can decreasing energy lead to different final contours?",
      "使用相同張力，分別從半徑 0.5 與 1.2 開始。為何能量都下降，最後輪廓卻不同？",
    ),
    note: L(
      "A one-parameter circular restriction, not a full snake or level-set solver: E(r)=αr²−exp(−((r−1)/0.16)²). Radius lies in [0,1.8]. Projected gradient steps use backtracking to avoid increasing energy. The dashed circle is the target boundary.",
      "這是只有一個參數的圓形限制模型，並非完整 snake 或水平集求解器：E(r)=αr²−exp(−((r−1)/0.16)²)，半徑限制在 [0,1.8]。投影梯度步驟使用回溯，避免能量增加。虛線圓為目標邊界。",
    ),
    related: [IP("dip-11-active-contours")],
    controls: [
      range("radius", "Initial radius", "初始半徑", 1.3, 0.1, 1.8, 0.05),
      range("tension", "Tension α", "張力 α", 0.15, 0, 1, 0.05),
      range("steps", "Descent steps", "下降步數", 30, 0, 80, 1),
    ],
    compute(s, l) {
      const r = radialSnake(n(s, "radius"), n(s, "tension"), n(s, "steps"));
      const energies = Array.from({ length: 81 }, (_, i) =>
          r.energy((i * 1.8) / 80),
        ),
        lo = Math.min(...energies),
        hi = Math.max(...energies),
        py = (v: number) => 240 - (180 * (v - lo)) / (hi - lo);
      const body =
        `<circle cx="165" cy="155" r="65" fill="none" stroke="var(--plot-a)" stroke-dasharray="5 3"/><circle cx="165" cy="155" r="${65 * n(s, "radius")}" fill="none" stroke="var(--plot-muted)"/><circle cx="165" cy="155" r="${65 * r.radius}" fill="none" stroke="var(--plot-b)" stroke-width="3"/>` +
        path(
          energies.map((e, i) => [350 + i * 3.1, py(e)]),
          `stroke="var(--plot-a)" stroke-width="2"`,
        ) +
        `<circle cx="${350 + (r.radius / 1.8) * 248}" cy="${py(r.finalEnergy)}" r="5" fill="var(--plot-b)"/>` +
        text(
          60,
          285,
          pick(l, "Gray: initial; color: final", "灰色：起點；彩色：終點"),
        ) +
        text(365, 285, pick(l, "Radius 0 → 1.8", "半徑 0 → 1.8"));
      return {
        svg: svg(body),
        summary: pick(
          l,
          "Local descent balances shrinking tension against edge attraction. Outside the edge’s capture range, the contour can collapse even though every step lowers energy.",
          "局部下降平衡收縮張力與邊緣吸引力。在邊緣的捕捉範圍之外，即使每一步都降低能量，輪廓仍可能塌縮。",
        ),
        metrics: [
          metric(l, "Final radius", "最終半徑", r.radius),
          metric(l, "Initial energy", "初始能量", r.initialEnergy),
          metric(l, "Final energy", "最終能量", r.finalEnergy),
        ],
      };
    },
  },
  {
    id: "classification-threshold",
    track: "deep-learning",
    title: L(
      "DLCV 02 · A decision threshold changes which mistakes you make",
      "DLCV 02・決策門檻改變錯誤的種類",
    ),
    description: L(
      "Move the decision threshold on fixed classifier scores. The confusion matrix changes, and the operating point slides along the ROC and precision-recall curves built from the same scores.",
      "移動固定分類分數的決策門檻。混淆矩陣隨之改變，操作點則沿著由同一組分數畫出的 ROC 與精確率–召回率曲線滑動。",
    ),
    challenge: L(
      "Find a threshold with no false positives and note where the dot sits on the ROC curve. How many positives are missed? Then raise the false-positive cost c: the curves do not move, but which point has the lowest mean error cost does.",
      "找出沒有偽陽性的門檻，並看操作點落在 ROC 曲線的哪裡。漏掉了多少正例？接著提高偽陽性成本 c：曲線不會動，但平均錯誤成本最低的點會改變。",
    ),
    note: L(
      "Ten fixed synthetic scores with five positive labels. Predict class 1 when score ≥ threshold; no model is trained. Each vertex of the curves is one threshold (one of the ten scores); lines between vertices are only drawn to guide the eye. ROC: x is FP/(FP+TN), y is recall TP/(TP+FN); AUC is the trapezoid area. Precision-recall: precision is undefined if nothing is predicted positive, so no dot is drawn. Mean error cost = (c×FP+FN)/10, with false-negative cost fixed at 1. Adjusting c changes evaluation, not scores, decisions or curves.",
      "十筆固定合成分數包含五個正例。分數 ≥ 門檻時預測類別 1；未進行模型訓練。曲線上每個頂點對應一個門檻（十個分數之一），頂點之間的連線只是輔助視線。ROC：橫軸為 FP/(FP+TN)，縱軸為召回率 TP/(TP+FN)；AUC 以梯形法計算面積。精確率–召回率：若未預測任何正例，精確率無定義，因此不畫點。平均錯誤成本 = (c×FP+FN)/10，偽陰性成本固定為 1。調整 c 改變評估成本，不會改變分數、決策或曲線。",
    ),
    predict: {
      question: L(
        "The threshold is 0.5. You lower it to 0.3. Where does the operating point move on the ROC curve?",
        "門檻目前是 0.5，你把它降到 0.3。操作點在 ROC 曲線上會往哪裡移動？",
      ),
      options: [
        { value: "up-right", label: L("Up and to the right: more recall, more false alarms", "往右上：召回率變高，誤報也變多") },
        { value: "down-left", label: L("Down and to the left: fewer false alarms and lower recall", "往左下：誤報變少，召回率也變低") },
        { value: "up-left", label: L("Up and to the left: better on both axes", "往左上：兩個軸都變好") },
      ],
      answer: "up-right",
      explain: L(
        "A lower threshold admits more samples as positive. At 0.5 the point is (FPR, TPR) = (0.4, 0.8); at 0.3 it is (0.6, 1.0). Recall rose from 4 of 5 positives to 5 of 5, but false positives rose from 2 to 3. No threshold can move up-left on this curve: it only slides along it.",
        "較低的門檻會讓更多樣本被判為正例。門檻 0.5 時點在 (FPR, TPR) = (0.4, 0.8)；降到 0.3 時移到 (0.6, 1.0)。5 個正例中找到的數量從 4 個增加到 5 個，但偽陽性從 2 個增加為 3 個。在這條曲線上沒有任何門檻能往左上移動，只能沿著曲線滑動。",
      ),
    },
    related: [DL("dl-02-image-classification")],
    controls: [
      range("threshold", "Decision threshold", "決策門檻", 0.5, 0, 1, 0.01),
      range("cost", "False-positive cost c", "偽陽性成本 c", 1, 0.25, 5, 0.25),
    ],
    compute(s, l) {
      const r = classificationThreshold(n(s, "threshold"), n(s, "cost"));
      const curve = rocCurve();
      const t = n(s, "threshold");
      const fpr = r.fp / curve.negatives,
        tpr = r.tp / curve.positives;
      // Row 1: confusion matrix (left) and the scores on a line (right).
      let body =
        text(126, 30, pick(l, "Predict 1", "預測 1"), 'text-anchor="middle"') +
        text(244, 30, pick(l, "Predict 0", "預測 0"), 'text-anchor="middle"') +
        text(10, 78, pick(l, "True 1", "真實 1")) +
        text(10, 148, pick(l, "True 0", "真實 0"));
      [r.tp, r.fn, r.fp, r.tn].forEach((v, i) => {
        const x = 70 + (i % 2) * 118,
          y = 40 + Math.floor(i / 2) * 70;
        body +=
          `<rect x="${x}" y="${y}" width="112" height="64" rx="5" fill="var(--plot-${i === 0 || i === 3 ? "a" : "b"})" opacity=".2"/>` +
          text(x + 56, y + 38, `${["TP", "FN", "FP", "TN"][i]} = ${v}`, 'text-anchor="middle"');
      });
      body +=
        text(
          10,
          202,
          pick(
            l,
            "Scores, high to low (● = true 1):",
            "分數由高至低（●＝真實 1）：",
          ),
        ) +
        text(
          10,
          226,
          r.scores
            .slice(0, 5)
            .map((v, i) => `${r.truth[i] ? "●" : "○"}${fmt(v, 2)}`)
            .join("  "),
        ) +
        text(
          10,
          250,
          r.scores
            .slice(5)
            .map((v, i) => `${r.truth[i + 5] ? "●" : "○"}${fmt(v, 2)}`)
            .join("  "),
        );
      const sx = (v: number) => Math.round((350 + 260 * v) * 10) / 10,
        tx = sx(clamp(t));
      body +=
        text(350, 24, pick(l, "Scores on a line", "分數排成一條線")) +
        `<rect x="${tx}" y="52" width="${610 - tx}" height="94" fill="var(--plot-a)" opacity=".12"/>` +
        `<line x1="350" y1="146" x2="610" y2="146" stroke="var(--plot-muted)"/>`;
      [0, 0.5, 1].forEach((v) => {
        body +=
          `<line x1="${sx(v)}" y1="146" x2="${sx(v)}" y2="152" stroke="var(--plot-muted)"/>` +
          text(sx(v), 168, fmt(v, 1), 'text-anchor="middle"');
      });
      r.scores.forEach((v, i) => {
        body += r.truth[i]
          ? `<circle cx="${sx(v)}" cy="82" r="6.5" fill="var(--plot-a)"/>`
          : `<circle cx="${sx(v)}" cy="118" r="6" fill="none" stroke="var(--plot-b)" stroke-width="2.5"/>`;
      });
      body +=
        `<line x1="${tx}" y1="52" x2="${tx}" y2="146" stroke="var(--plot-ink)" stroke-width="2" stroke-dasharray="5 3"/>` +
        text(clamp(tx, 380, 580), 46, `t = ${fmt(t, 2)}`, 'text-anchor="middle"') +
        text(480, 190, pick(l, "classifier score", "分類器分數"), 'text-anchor="middle"') +
        text(350, 214, pick(l, "● true 1", "● 真實 1"), 'fill="var(--plot-a)"') +
        text(450, 214, pick(l, "○ true 0", "○ 真實 0"), 'fill="var(--plot-b)"') +
        text(350, 238, pick(l, "Shaded: predicted 1", "底色：預測為 1"));
      // Row 2: ROC (left) and precision-recall (right), both built from the same ten scores.
      const box = (x0: number, title: string, xl: string, yl: string) => {
        const x1 = x0 + 210;
        return (
          text(x0, 300, title) +
          `<rect x="${x0}" y="316" width="210" height="210" fill="none" stroke="var(--plot-muted)"/>` +
          text(x0 - 8, 530, "0", 'text-anchor="end"') +
          text(x0 - 8, 326, "1", 'text-anchor="end"') +
          text(x0, 548, "0", 'text-anchor="middle"') +
          text(x1, 548, "1", 'text-anchor="middle"') +
          text(x0 + 105, 572, xl, 'text-anchor="middle"') +
          `<text transform="rotate(-90 ${x0 - 40} 421)" x="${x0 - 40}" y="421" text-anchor="middle">${escape(yl)}</text>`
        );
      };
      const px = (x0: number, v: number) => x0 + 210 * v,
        py = (v: number) => 526 - 210 * v;
      const dotAt = (x0: number, vx: number, vy: number) =>
        `<line x1="${px(x0, vx)}" y1="${py(vy)}" x2="${px(x0, vx)}" y2="526" stroke="var(--plot-b)" stroke-dasharray="3 3"/>` +
        `<line x1="${x0}" y1="${py(vy)}" x2="${px(x0, vx)}" y2="${py(vy)}" stroke="var(--plot-b)" stroke-dasharray="3 3"/>` +
        `<circle cx="${px(x0, vx)}" cy="${py(vy)}" r="7.5" fill="var(--plot-b)" stroke="var(--plot-bg)" stroke-width="2"/>`;
      body +=
        box(
          80,
          pick(l, "ROC curve", "ROC 曲線"),
          pick(l, "False-positive rate", "偽陽性率"),
          pick(l, "Recall (TPR)", "召回率（TPR）"),
        ) +
        `<line x1="80" y1="526" x2="290" y2="316" stroke="var(--plot-grid)" stroke-dasharray="5 4"/>` +
        path(
          curve.roc.map((p) => [px(80, p.fpr), py(p.tpr)] as [number, number]),
          'stroke="var(--plot-a)" stroke-width="2.5"',
        ) +
        curve.roc
          .map(
            (p) =>
              `<circle cx="${px(80, p.fpr)}" cy="${py(p.tpr)}" r="3" fill="var(--plot-a)"/>`,
          )
          .join("") +
        dotAt(80, fpr, tpr) +
        box(
          400,
          pick(l, "Precision–recall curve", "精確率–召回率曲線"),
          pick(l, "Recall", "召回率"),
          pick(l, "Precision", "精確率"),
        ) +
        `<line x1="400" y1="${py(0.5)}" x2="610" y2="${py(0.5)}" stroke="var(--plot-grid)" stroke-dasharray="5 4"/>` +
        path(
          curve.roc
            .slice(1)
            .map((p) => [px(400, p.tpr), py(p.precision)] as [number, number]),
          'stroke="var(--plot-a)" stroke-width="2.5"',
        ) +
        curve.roc
          .slice(1)
          .map(
            (p) =>
              `<circle cx="${px(400, p.tpr)}" cy="${py(p.precision)}" r="3" fill="var(--plot-a)"/>`,
          )
          .join("") +
        (r.precision === null
          ? text(505, 430, pick(l, "no positive predictions", "未預測任何正例"), 'text-anchor="middle"')
          : dotAt(400, r.recall, r.precision));
      return {
        svg: svg(body, 590),
        summary: pick(
          l,
          `True positives (TP): ${r.tp}; false positives (FP): ${r.fp}; false negatives (FN): ${r.fn}; true negatives (TN): ${r.tn}. Lowering the threshold can recover missed positives while admitting false positives. Precision and recall use different denominators; accuracy alone cannot express unequal error costs.`,
          `真陽性（TP）：${r.tp}；偽陽性（FP）：${r.fp}；偽陰性（FN）：${r.fn}；真陰性（TN）：${r.tn}。降低門檻可能找回漏掉的正例，也可能增加偽陽性。精確率與召回率使用不同分母；單靠正確率無法表達不對稱的錯誤成本。`,
        ),
        metrics: [
          metric(
            l,
            "Precision",
            "精確率",
            r.precision === null ? pick(l, "Undefined", "無定義") : r.precision,
          ),
          metric(l, "Recall", "召回率", r.recall),
          metric(l, "ROC AUC (all thresholds)", "ROC AUC（所有門檻）", curve.auc),
          metric(l, "F1", "F1", r.f1),
          metric(l, "Accuracy", "正確率", r.accuracy),
          metric(l, "Mean error cost", "平均錯誤成本", r.cost),
        ],
      };
    },
  },
  {
    id: "restoration-ambiguity",
    track: "deep-learning",
    title: L(
      "DLCV 04 · A sharp guess can be less faithful",
      "DLCV 04・清晰的猜測可能較不忠實",
    ),
    description: L(
      "A blurry observation fits two sharp clean signals. The blurry average has the lowest expected squared error; a sharp guess looks better but is usually farther from the truth.",
      "一個模糊的觀測同時符合兩種清晰的乾淨訊號。模糊的平均值期望平方誤差最低；清晰的猜測看起來較好，但通常離真相較遠。",
    ),
    challenge: L(
      "Start from the sharp copy of A (commitment 1): its risk is higher than the blurry mean's. Slide the commitment to 0 and watch the risk fall to the minimum. Then raise the probability of A to 0.9: does the sharp copy of A catch up, and can it ever win?",
      "從清晰的 A（偏向程度 1）開始：它的風險比模糊平均值高。把偏向程度滑到 0，看風險降到最小。再把 A 的機率提高到 0.9：清晰的 A 追上來了嗎？它有可能勝過平均值嗎？",
    ),
    note: L(
      "Two fixed 8-sample patterns, A = 00110011 and B = 11001100, are the only clean signals that could have produced the same blurry observation. P(A|y) = p is supplied, not learned. The blurry mean is p·A + (1−p)·B. Your reconstruction blends the mean toward A: (1−m)·mean + m·A, so m = 1 is exactly A (a sharp sample) and m = 0 is the mean. Risk = the expected mean squared error, p·MSE(A, output) + (1−p)·MSE(B, output), averaged over which pattern is the truth. Sharpness is the total variation (sum of absolute differences between neighbouring samples), not a perceptual quality score.",
      "兩個固定的 8 取樣值模式 A = 00110011 與 B = 11001100，是唯二可能產生同一個模糊觀測的乾淨訊號。P(A|y) = p 由設定提供，並非學得。模糊平均值為 p·A + (1−p)·B。你的重建把平均值混合向 A：(1−m)·平均值 + m·A，所以 m = 1 恰好是 A（清晰的單一樣本），m = 0 則是平均值。風險 = 期望均方誤差，即 p·MSE(A, 輸出) + (1−p)·MSE(B, 輸出)，對「哪個模式才是真相」取平均。清晰程度為總變差（相鄰取樣值差的絕對值總和），並非感知品質分數。",
    ),
    predict: {
      question: L(
        "Set P(A) = 0.9. The sharp copy of A is now very likely correct. Does the blurry mean still have the lower expected squared error?",
        "把 P(A) 設為 0.9。清晰的 A 現在很可能是對的。模糊平均值的期望平方誤差還是比較低嗎？",
      ),
      options: [
        { value: "mean", label: L("Yes, the blurry mean is still lower", "是，模糊平均值仍然較低") },
        { value: "sharp", label: L("No, the sharp copy of A is lower now", "不是，清晰的 A 現在較低") },
        { value: "tie", label: L("They tie", "兩者相同") },
      ],
      answer: "mean",
      explain: L(
        "The mean has risk p(1−p) = 0.09 and the sharp copy of A has risk 1−p = 0.10. The gap shrinks as p approaches 1 but never closes, because the mean is the exact minimizer of expected squared error.",
        "平均值的風險是 p(1−p) = 0.09，清晰的 A 的風險是 1−p = 0.10。p 越接近 1 差距越小，但永遠不會消失，因為平均值正是期望平方誤差的精確最小化者。",
      ),
    },
    related: [DL("dl-04-image-restoration")],
    controls: [
      range(
        "probability",
        "Probability of clean pattern A",
        "乾淨模式 A 的機率",
        0.5,
        0.1,
        0.9,
        0.05,
      ),
      range(
        "mix",
        "Commitment m (0 = blurry mean, 1 = sharp A)",
        "偏向程度 m（0＝模糊平均值，1＝清晰的 A）",
        1,
        0,
        1,
        0.05,
      ),
    ],
    compute(s, l) {
      const r = ambiguousRestoration(n(s, "probability"), n(s, "mix"));
      const p = n(s, "probability");
      const variation = totalVariation(r.output);
      // Rows of 8 samples: dark = 1, light = 0.
      const rows: [string, number[]][] = [
        [pick(l, `Clean A (P = ${fmt(p, 2)})`, `乾淨訊號 A（P = ${fmt(p, 2)}）`), r.a],
        [pick(l, `Clean B (P = ${fmt(1 - p, 2)})`, `乾淨訊號 B（P = ${fmt(1 - p, 2)}）`), r.b],
        [pick(l, "Blurry mean of A and B", "A、B 的模糊平均值"), r.mean],
        [
          pick(
            l,
            `Your reconstruction (m = ${fmt(n(s, "mix"), 2)})`,
            `你的重建（m = ${fmt(n(s, "mix"), 2)}）`,
          ),
          r.output,
        ],
      ];
      let body = text(
        10,
        22,
        pick(
          l,
          "Eight samples along a signal; darker = larger value",
          "沿訊號排列的 8 個取樣值；越深代表數值越大",
        ),
      );
      rows.forEach(([label, values], k) => {
        const y = 36 + k * 44;
        body += text(10, y + 25, label);
        values.forEach((v, i) => {
          body +=
            `<rect x="${262 + i * 46}" y="${y}" width="44" height="36" fill="var(--plot-ink)" fill-opacity="${fmt(v, 3)}" stroke="var(--plot-muted)" stroke-width="1"/>`;
        });
        if (k === 3)
          body += `<rect x="260" y="${y - 2}" width="372" height="40" fill="none" stroke="var(--plot-b)" stroke-width="2.5" rx="3"/>`;
      });
      // Legend for the shades.
      [0, 0.5, 1].forEach((v, i) => {
        body +=
          `<rect x="${262 + i * 110}" y="218" width="26" height="20" fill="var(--plot-ink)" fill-opacity="${v}" stroke="var(--plot-muted)"/>` +
          text(294 + i * 110, 234, ["0", "0.5", "1"][i]);
      });
      body += text(10, 234, pick(l, "Shade legend:", "色階圖例："));
      // Risk bars.
      const barX = 262,
        scale = 340;
      body +=
        text(
          10,
          272,
          pick(
            l,
            "Risk = expected squared error (smaller is better)",
            "風險 = 期望平方誤差（越小越好）",
          ),
        ) +
        text(10, 302, pick(l, "Blurry mean", "模糊平均值")) +
        `<rect x="${barX}" y="286" width="${fmt(scale * r.riskMean, 2)}" height="22" fill="var(--plot-a)"/>` +
        text(barX + scale * r.riskMean + 8, 302, fmt(r.riskMean, 4)) +
        text(10, 334, pick(l, "Your reconstruction", "你的重建")) +
        `<rect x="${barX}" y="318" width="${fmt(scale * r.risk, 2)}" height="22" fill="var(--plot-b)"/>` +
        text(barX + scale * r.risk + 8, 334, fmt(r.risk, 4));
      return {
        svg: svg(body, 356),
        summary: pick(
          l,
          "Averaging incompatible possibilities minimizes expected squared error, but can remove structures present in every valid clean outcome. Choosing one restores structure while risking the wrong structure.",
          "平均互不相容的可能性可最小化期望平方誤差，卻可能移除所有合法乾淨結果都具有的結構。選定一種結果能恢復結構，也可能恢復錯誤的結構。",
        ),
        metrics: [
          metric(l, "Expected reconstruction MSE", "重建期望均方誤差", r.risk),
          metric(
            l,
            "Conditional-mean minimum MSE",
            "條件平均最小均方誤差",
            r.riskMean,
          ),
          metric(l, "Sharpness (adjacent total variation)", "清晰程度（相鄰取樣總變差）", variation),
        ],
      };
    },
  },
  {
    id: "tracking-association",
    track: "deep-learning",
    title: L(
      "DLCV 06 · Crossing objects challenge identity",
      "DLCV 06・交錯物體考驗身分關聯",
    ),
    description: L(
      "Balance positional and appearance costs and solve the complete two-object one-to-one assignment.",
      "權衡位置與外觀成本，求解兩物體的一對一完整指派。",
    ),
    challenge: L(
      "Make the objects cross, then increase motion weight. Under what appearance error does adding appearance stop helping?",
      "讓兩個物體交錯，再提高運動權重。外觀誤差多大時，加入外觀不再有幫助？",
    ),
    note: L(
      "Two predicted positions are fixed at 0.3 and 0.7. Detection identities are known only for evaluation. Cost = w×absolute position error+(1−w)×appearance cost. Both permutations are enumerated exactly; no detector, learned embedding, birth/death logic or Hungarian implementation. Ties choose identity order.",
      "兩個預測位置固定為 0.3 與 0.7。偵測身分僅供評估。成本 = w×位置絕對誤差+(1−w)×外觀成本。精確列舉兩個排列；未實作偵測器、學得的嵌入、軌跡新增刪除或 Hungarian 演算法。並列時選身分順序。",
    ),
    related: [DL("dl-06-object-tracking")],
    controls: [
      range("crossing", "Crossing progress", "交錯進度", 1, 0, 1, 0.05),
      range("motion", "Motion weight w", "運動權重 w", 0.7, 0, 1, 0.05),
      range(
        "appearance",
        "Same-identity appearance cost",
        "同身分外觀成本",
        0.1,
        0,
        1,
        0.05,
      ),
    ],
    compute(s, l) {
      const r = trackingAssignment(
        n(s, "motion"),
        n(s, "appearance"),
        n(s, "crossing"),
      );
      let body = text(
        40,
        30,
        pick(l, "Predicted tracks → current detections", "預測軌跡 → 當前偵測"),
      );
      r.predictions.forEach((x, i) => {
        body +=
          `<circle cx="${70 + x * 500}" cy="75" r="10" fill="var(--plot-${"ab"[i]})"/><circle cx="${70 + r.detections[i] * 500}" cy="195" r="10" fill="var(--plot-${"ab"[i]})"/>` +
          path(
            [
              [70 + x * 500, 90],
              [70 + r.detections[r.assignment[i]] * 500, 180],
            ],
            `stroke="var(--plot-${"ab"[i]})" stroke-width="2.5"`,
          );
      });
      body +=
        text(
          40,
          250,
          `C = [${r.cost[0].map((v) => fmt(v)).join(", ")}; ${r.cost[1].map((v) => fmt(v)).join(", ")}]`,
        ) +
        text(
          40,
          280,
          pick(
            l,
            "Colors show true identity; lines show assignment.",
            "顏色表示真實身分；連線表示指派。",
          ),
        );
      return {
        svg: svg(body),
        summary: pick(
          l,
          r.ties
            ? "Both assignments have equal cost. The deterministic tie break is not evidence of identity."
            : "Matching nearest positions can switch identities after a crossing. Appearance helps only when its cost is informative.",
          r.ties
            ? "兩種指派成本相同。固定的並列處理規則並非身分證據。"
            : "交錯後，匹配最近位置可能交換身分。只有在外觀成本提供可靠訊息時，外觀才有幫助。",
        ),
        metrics: [
          metric(l, "Identity-preserving cost", "保持身分的成本", r.direct),
          metric(l, "Swapped-identity cost", "交換身分的成本", r.swapped),
          metric(
            l,
            "Identity switches",
            "身分交換數",
            r.assignment[0] === 0 ? 0 : 2,
          ),
        ],
      };
    },
  },
  {
    id: "world-model-rollout",
    track: "deep-learning",
    title: L(
      "DLCV 07 · Small dynamics errors compound in rollouts",
      "DLCV 07・小型動態誤差會在展開中累積",
    ),
    description: L(
      "Compare one-step predictions, open-loop rollout and periodic observation refresh under the same actions.",
      "在相同行動下，比較單步預測、開迴路展開與定期更新觀測。",
    ),
    challenge: L(
      "Set the model coefficient to the true value 0.96. Then change it to 1.02 and increase the horizon. How much can observation refresh correct?",
      "先把模型係數設為真值 0.96，再改成 1.02 並增加展開長度。更新觀測能修正多少誤差？",
    ),
    note: L(
      "True scalar state: x(t+1)=0.96x(t)+u(t), x(0)=1, u(t)=a sin(0.45t). The model replaces 0.96 with a chosen coefficient. Refresh feeds the true state every k steps; k=0 disables it. Fully observed deterministic dynamics, no learned video model or replanning.",
      "真實純量狀態為 x(t+1)=0.96x(t)+u(t)，x(0)=1，u(t)=a sin(0.45t)。模型以選定係數取代 0.96。更新每 k 步輸入真實狀態；k=0 表示停用。此處為完全可觀測的確定性動態，未使用學得的影片模型或重新規劃。",
    ),
    related: [DL("dl-07-world-models")],
    controls: [
      range(
        "coefficient",
        "Model transition coefficient",
        "模型轉移係數",
        1.02,
        0.85,
        1.08,
        0.01,
      ),
      range("steps", "Rollout horizon", "展開長度", 30, 5, 60, 1),
      range("action", "Action amplitude", "行動幅度", 0.08, 0, 0.2, 0.01),
      range(
        "refresh",
        "Observation refresh interval (0 = off)",
        "觀測更新間隔（0＝關閉）",
        5,
        0,
        15,
        1,
      ),
    ],
    compute(s, l) {
      const r = worldRollout(
        n(s, "coefficient"),
        n(s, "steps"),
        n(s, "action"),
        n(s, "refresh"),
      );
      return {
        svg: svg(
          lines(
            [r.truth, r.open, r.corrected],
            [
              pick(l, "True state", "真實狀態"),
              pick(l, "Open loop", "開迴路"),
              pick(l, "With observations", "加入觀測"),
            ],
          ),
        ),
        summary: pick(
          l,
          "One-step evaluation repeatedly supplies the true state. An open-loop rollout must consume its own imperfect predictions. Periodic observations limit drift without improving the transition model itself.",
          "單步評估反覆提供真實狀態；開迴路展開必須使用自身不完美的預測。定期觀測可限制漂移，但不會改善轉移模型本身。",
        ),
        metrics: [
          metric(l, "One-step MSE", "單步均方誤差", r.oneError),
          metric(l, "Open-loop MSE", "開迴路均方誤差", r.openError),
          metric(
            l,
            "Refreshed-rollout MSE",
            "更新觀測後均方誤差",
            r.correctedError,
          ),
        ],
      };
    },
  },
];
