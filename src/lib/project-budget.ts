// Planning assumptions, not observed usage or a statement of paid invoices.
// Standard short-context text API prices checked on 2026-11-05 (USD / 1M tokens).
export const BUDGET_DATE = '2026-11-05';
export const BUDGET_MODELS = [
  { name: 'GPT-6.1 Sol', share: 0.6, input: 2, output: 10 },
  { name: 'Claude Opus 5.5', share: 0.3, input: 4, output: 20 },
  { name: 'GPT-6 Astra', share: 0.1, input: 10, output: 50 },
] as const;
export const BUDGET_INPUT_RATE = BUDGET_MODELS.reduce((sum, m) => sum + m.share * m.input, 0);
export const BUDGET_OUTPUT_RATE = BUDGET_MODELS.reduce((sum, m) => sum + m.share * m.output, 0);
export const RESERVE = 0.25;
export const DEFAULT_USD_TWD = 32; // Explicit planning exchange rate, not a live quote.
type Range = readonly [number, number];
export function projectBudget(scale = 1, usdTwd = DEFAULT_USD_TWD) {
  if (!Number.isFinite(scale) || scale < 0 || scale > 20 || !Number.isFinite(usdTwd) || usdTwd <= 0 || usdTwd > 1000) {
    throw new RangeError('Invalid planning inputs');
  }
  const api = (inputM: number, outputM: number) => scale * (inputM * BUDGET_INPUT_RATE + outputM * BUDGET_OUTPUT_RATE);
  const develop: Range = [api(50, 10) * (1 + RESERVE), api(150, 30) * (1 + RESERVE)];
  // Annual maintenance: 12 months of updates plus a separate USD 20–100 allowance.
  const annual: Range = [(12 * api(1, 0.2) + 20) * (1 + RESERVE), (12 * api(5, 1) + 100) * (1 + RESERVE)];
  const total: Range = [develop[0] + annual[0], develop[1] + annual[1]];
  const amounts = (usd: Range) => ({ usd, twd: [usd[0] * usdTwd, usd[1] * usdTwd] as Range });
  return { develop: amounts(develop), annual: amounts(annual), total: amounts(total) };
}
