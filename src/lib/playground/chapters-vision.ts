import type { Experiment, Control, State } from "./types.ts";
import { L, pick, fmt, svg, text, path } from "./types.ts";
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
      "Move the decision threshold on fixed classifier scores and inspect the confusion matrix, precision, recall and error cost.",
      "移動固定分類分數的決策門檻，觀察混淆矩陣、精確率、召回率與錯誤成本。",
    ),
    challenge: L(
      "Find a threshold with no false positives. How many positives are missed? Increase the false-positive cost and compare the preferred operating point.",
      "找出沒有偽陽性的門檻。漏掉了多少正例？提高偽陽性成本，比較較合適的操作點。",
    ),
    note: L(
      "Ten fixed synthetic scores with five positive labels. Predict class 1 when score ≥ threshold; no model is trained. Precision is undefined if nothing is predicted positive. Mean error cost = (c×FP+FN)/10, with false-negative cost fixed at 1. Adjusting c changes evaluation, not scores or decisions.",
      "十筆固定合成分數包含五個正例。分數 ≥ 門檻時預測類別 1；未進行模型訓練。若未預測任何正例，精確率無定義。平均錯誤成本 = (c×FP+FN)/10，偽陰性成本固定為 1。調整 c 改變評估成本，不會改變分數或決策。",
    ),
    related: [DL("dl-02-image-classification")],
    controls: [
      range("threshold", "Decision threshold", "決策門檻", 0.5, 0, 1, 0.01),
      range("cost", "False-positive cost c", "偽陽性成本 c", 1, 0.25, 5, 0.25),
    ],
    compute(s, l) {
      const r = classificationThreshold(n(s, "threshold"), n(s, "cost"));
      let body =
        text(160, 30, pick(l, "Predict 1", "預測 1")) +
        text(380, 30, pick(l, "Predict 0", "預測 0")) +
        text(20, 100, pick(l, "True 1", "真實 1")) +
        text(20, 185, pick(l, "True 0", "真實 0"));
      [r.tp, r.fn, r.fp, r.tn].forEach((v, i) => {
        const x = 130 + (i % 2) * 220,
          y = 50 + Math.floor(i / 2) * 85;
        body +=
          `<rect x="${x}" y="${y}" width="195" height="70" rx="5" fill="var(--plot-${i === 0 || i === 3 ? "a" : "b"})" opacity=".2"/>` +
          text(x + 65, y + 42, `${["TP", "FN", "FP", "TN"][i]} = ${v}`);
      });
      body +=
        text(
          40,
          245,
          pick(
            l,
            "Scores, descending (● = true positive class):",
            "分數由高至低（●＝真實正類）：",
          ),
        ) +
        text(
          40,
          275,
          r.scores
            .map((v, i) => `${r.truth[i] ? "●" : "○"}${fmt(v, 2)}`)
            .join("  "),
        );
      return {
        svg: svg(body),
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
      "Two sharp signals produce the same observation. Compare their conditional mean with a sharper, committed reconstruction.",
      "兩個清晰訊號產生相同觀測。比較條件平均值與較清晰、但選定單一假設的重建。",
    ),
    challenge: L(
      "At equal probabilities, compare the mean with a complete guess of A. Which minimizes expected squared error? Does that make it a possible clean signal?",
      "兩種情況等機率時，比較平均值與完全猜測 A。哪個最小化期望平方誤差？它就一定是可能的乾淨訊號嗎？",
    ),
    note: L(
      "Two fixed binary patterns A and B share the same global mean observation. P(A|y)=p is supplied, not learned. The slider blends the conditional mean toward A. Expected MSE averages over both possible truths; sharpness here is adjacent-sample total variation, not a perceptual quality score.",
      "兩個固定二值模式 A、B 具有相同的全域平均觀測。P(A|y)=p 由設定提供，並非學得。滑桿將條件平均值逐漸混合至 A。期望均方誤差涵蓋兩種可能真相；此處以相鄰取樣值的總變差表示清晰程度，並非感知品質分數。",
    ),
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
        "Commitment toward pattern A",
        "偏向模式 A 的程度",
        0.5,
        0,
        1,
        0.05,
      ),
    ],
    compute(s, l) {
      const r = ambiguousRestoration(n(s, "probability"), n(s, "mix"));
      const variation = r.output
        .slice(1)
        .reduce((a, v, i) => a + Math.abs(v - r.output[i]), 0);
      return {
        svg: svg(
          lines(
            [r.a, r.b, r.output],
            ["A", "B", pick(l, "Reconstruction", "重建")],
            0,
            1,
          ),
        ),
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
          metric(l, "Adjacent total variation", "相鄰取樣總變差", variation),
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
