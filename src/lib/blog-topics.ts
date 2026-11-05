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
  agents: { en: 'From the first task to skills, harnesses, and choosing tools.', 'zh-tw': '從第一個任務，到 skills、harness 與工具選擇。' },
  'models-evaluation': { en: 'Understand model families, evidence, metrics, and the people behind them.', 'zh-tw': '認識模型家族、評測證據、指標與代表人物。' },
  'image-generation': { en: 'Choose a model and workflow for creating and editing images.', 'zh-tw': '選擇適合影像生成與編輯的模型及工作流程。' },
  'video-generation': { en: 'Compare motion, control, cost, and deployment options.', 'zh-tw': '比較動態品質、控制方式、成本與部署選項。' },
  site: { en: 'Introductions and notes about Pixels → Perception.', 'zh-tw': 'Pixels → Perception 的介紹與相關筆記。' },
} satisfies Record<BlogTopic, Record<'en' | 'zh-tw', string>>;
