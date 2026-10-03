export const LANGS = ['en', 'zh-tw'] as const;
export type Lang = (typeof LANGS)[number];
export const DEFAULT_LANG: Lang = 'en';

export const LANG_LABEL: Record<Lang, string> = { en: 'English', 'zh-tw': '繁體中文' };
export const HTML_LANG: Record<Lang, string> = { en: 'en', 'zh-tw': 'zh-Hant-TW' };

const en = {
  'site.name': 'Pixels → Perception',
  'site.tagline': 'Computer vision, from pixels up',
  'nav.learn': 'Learn',
  'nav.blog': 'Blog',
  'nav.playground': 'Playground',
  'nav.about': 'About',
  'nav.search': 'Search',
  'theme.toggle': 'Toggle dark mode',
  'home.title': 'Learn how machines see, one kernel at a time.',
  'home.lede':
    'Tutorials that go from image arrays and filters to feature matching, classic machine learning and modern deep networks. Each one has the intuition, the math, code you can run, and a demo you can play with.',
  'home.start': 'Start with the basics',
  'home.browse': 'Browse the curriculum',
  'home.curriculum': 'The curriculum',
  'home.latest': 'Latest',
  'home.allPosts': 'All posts →',
  'track.foundations': 'Foundations',
  'track.image-processing': 'Image Processing',
  'track.classical-cv': 'Classical CV',
  'track.ml': 'ML for Vision',
  'track.deep-learning': 'Deep Learning',
  'track.label': 'Track',
  'track.published': 'published',
  'track.empty': 'Coming soon',
  'level.beginner': 'Beginner',
  'level.intermediate': 'Intermediate',
  'level.advanced': 'Advanced',
  'tut.minutes': 'min',
  'tut.needs': 'Needs',
  'tut.colab': 'Open in Colab',
  'tut.notebook': 'View notebook',
  'tut.onThisPage': 'On this page',
  'tut.prev': '← Previous',
  'tut.next': 'Next →',
  'tut.edit': 'Edit this page on GitHub',
  'tut.notTranslated': 'This page has not been translated yet. You are reading the English version.',
  'blog.title': 'Blog',
  'blog.lede': 'Paper notes, news, and longer essays outside the curriculum.',
  'blog.original': 'Originally published on',
  'learn.title': 'Learn',
  'learn.lede': 'Five tracks, ordered from pixels to modern deep models. Follow one top to bottom, or jump in where you need.',
  'playground.title': 'Playground',
  'playground.lede': 'Every interactive demo from the tutorials, in one place.',
  'search.title': 'Search',
  'about.title': 'About',
  'footer.rss': 'RSS',
  'footer.source': 'Source on GitHub',
  'notfound.title': 'Page not found',
} as const;

type Key = keyof typeof en;

const zhTw: Record<Key, string> = {
  'site.name': 'Pixels → Perception',
  'site.tagline': '從像素開始的電腦視覺',
  'nav.learn': '學習',
  'nav.blog': '部落格',
  'nav.playground': '互動實驗',
  'nav.about': '關於',
  'nav.search': '搜尋',
  'theme.toggle': '切換深色模式',
  'home.title': '一次一個卷積核，學會機器如何「看」。',
  'home.lede':
    '從影像陣列與濾波器開始，一路到特徵匹配、傳統機器學習與現代深度網路。每篇教學都包含直覺、數學、可執行的程式碼，以及可以動手玩的互動示範。',
  'home.start': '從基礎開始',
  'home.browse': '瀏覽課程',
  'home.curriculum': '課程地圖',
  'home.latest': '最新文章',
  'home.allPosts': '所有文章 →',
  'track.foundations': '基礎',
  'track.image-processing': '影像處理',
  'track.classical-cv': '傳統電腦視覺',
  'track.ml': '視覺機器學習',
  'track.deep-learning': '深度學習',
  'track.label': '單元',
  'track.published': '已發布',
  'track.empty': '即將推出',
  'level.beginner': '入門',
  'level.intermediate': '中階',
  'level.advanced': '進階',
  'tut.minutes': '分鐘',
  'tut.needs': '先備知識',
  'tut.colab': '在 Colab 開啟',
  'tut.notebook': '查看筆記本',
  'tut.onThisPage': '本頁內容',
  'tut.prev': '← 上一篇',
  'tut.next': '下一篇 →',
  'tut.edit': '在 GitHub 上編輯此頁',
  'tut.notTranslated': '本頁尚未翻譯，以下為英文版本。',
  'blog.title': '部落格',
  'blog.lede': '論文筆記、新知與課程以外的長文。',
  'blog.original': '原文發表於',
  'learn.title': '學習',
  'learn.lede': '五個單元，從像素一路到現代深度模型。可以依序閱讀，也可以直接跳到需要的主題。',
  'playground.title': '互動實驗',
  'playground.lede': '所有教學中的互動示範，集中在這裡。',
  'search.title': '搜尋',
  'about.title': '關於',
  'footer.rss': 'RSS',
  'footer.source': 'GitHub 原始碼',
  'notfound.title': '找不到頁面',
};

const dict: Record<Lang, Record<Key, string>> = { en, 'zh-tw': zhTw };

export function useT(lang: Lang) {
  // Accepts dynamic keys such as `track.${track}`; unknown keys fall back to English, then the key itself.
  return (key: Key | (string & {})): string => dict[lang][key as Key] ?? en[key as Key] ?? key;
}
