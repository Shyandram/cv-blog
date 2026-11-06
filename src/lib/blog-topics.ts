export const BLOG_TOPICS = ['agents', 'models-evaluation', 'image-generation', 'video-generation', 'site'] as const;
export type BlogTopic = typeof BLOG_TOPICS[number];

export const BLOG_TOPIC_LABELS = {
  agents: { en: 'Agents & tools', 'zh-tw': 'Agent 與工具' },
  'models-evaluation': { en: 'Models & benchmarks', 'zh-tw': '模型與評測' },
  'image-generation': { en: 'Image generation', 'zh-tw': '影像生成' },
  'video-generation': { en: 'Video generation', 'zh-tw': '影片生成' },
  site: { en: 'About this site', 'zh-tw': '網站記事' },
} satisfies Record<BlogTopic, Record<'en' | 'zh-tw', string>>;

export const BLOG_TOPIC_DESCRIPTIONS = {
  agents: { en: 'How agents work, how to build and operate them, and how to choose tools and workflows.', 'zh-tw': '說明 agent 如何運作、如何建立與維運，以及如何選擇工具與工作流程。' },
  'models-evaluation': { en: 'Read benchmarks, licenses and evidence, and run your own model evaluation.', 'zh-tw': '閱讀基準測試、授權與證據，並自行評估模型。' },
  'image-generation': { en: 'How image generators work and how to test them for a real deliverable.', 'zh-tw': '說明影像生成模型如何運作，以及如何針對實際成品加以測試。' },
  'video-generation': { en: 'Motion, control, evaluation and the cost of usable footage.', 'zh-tw': '動態、控制、評估，以及取得可用影片的成本。' },
  site: { en: 'Introductions and notes about Pixels → Perception.', 'zh-tw': 'Pixels → Perception 的介紹與相關筆記。' },
} satisfies Record<BlogTopic, Record<'en' | 'zh-tw', string>>;

/** Multi-part series: posts listed in reading order, labelled "Part n of N" on the blog index and post pages. */
export const BLOG_SERIES = [
  {
    id: 'agent-fundamentals',
    label: { en: 'Agent fundamentals', 'zh-tw': 'Agent 基礎' },
    slugs: ['agents-from-prompts-to-systems', 'agent-skills-and-harnesses', 'openclaw-practical-agent-operations'],
  },
] as const;

/** Entry points shown above the topic lists: one starting post per reader goal. */
export const BLOG_START_HERE = [
  { slug: 'agents-from-prompts-to-systems', goal: { en: 'New to agents', 'zh-tw': '剛接觸 agent' } },
  { slug: 'models-benchmarks-and-people', goal: { en: 'Choosing or evaluating a model', 'zh-tw': '選擇或評估模型' } },
  { slug: 'choosing-image-generation-models', goal: { en: 'Making images or video', 'zh-tw': '製作影像或影片' } },
] as const;

export function seriesOf(slug: string) {
  for (const s of BLOG_SERIES) {
    const i = (s.slugs as readonly string[]).indexOf(slug);
    if (i >= 0) return { series: s, part: i + 1, total: s.slugs.length };
  }
  return undefined;
}
