// Practice status, kept only in this browser (localStorage through progress.ts, every access in try/catch).
//   pp:practice  { "<problem id>": "attempted" | "solved" }
// "attempted": the problem page was opened. "solved": its prediction question was answered correctly there.
import { load, save } from './progress';

export type PracticeStatus = 'attempted' | 'solved';
const KEY = 'pp:practice';

export function practiceStatus(): Record<string, PracticeStatus> {
  try {
    const raw = JSON.parse(load(KEY) ?? '{}');
    if (!raw || typeof raw !== 'object') return {};
    const out: Record<string, PracticeStatus> = {};
    for (const [id, v] of Object.entries(raw)) if (v === 'attempted' || v === 'solved') out[id] = v;
    return out;
  } catch { return {}; }
}

/** Records a status; "solved" is never lowered back to "attempted". */
export function markPractice(id: string, status: PracticeStatus) {
  const all = practiceStatus();
  if (all[id] === 'solved' || all[id] === status) return;
  all[id] = status;
  save(KEY, JSON.stringify(all));
}
