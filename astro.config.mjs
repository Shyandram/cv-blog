// @ts-check
import legacyChapterRedirects from './src/content/legacy-chapter-redirects.json' with { type: 'json' };
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { unified, rehypeHeadingIds } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import remarkCjkFriendly from 'remark-cjk-friendly';
import rehypeKatex from 'rehype-katex';
import rehypeCitations from './src/lib/rehypeCitations.mjs';
import rehypeSupplementZones from './src/lib/rehypeSupplementZones.mjs';
import rehypePaperHeadingIds from './src/lib/rehypePaperHeadingIds.mjs';
import rehypeCodeFolding from './src/lib/rehypeCodeFolding.mjs';
import { fileURLToPath } from 'node:url';
import { preserveRedirectFragments } from './scripts/preserve-redirect-fragments.mjs';
import { indexSearch } from './scripts/index-search.mjs';

// Build the Pagefind search index as part of `astro build`, so search works no matter
// which build command the host runs (`astro build`, `npm run build`, …).
/** @returns {import('astro').AstroIntegration} */
const pagefindIndex = () => {
  /** @type {Record<string, string> | undefined} */
  let redirectTargets;
  return {
    name: 'pagefind-index',
    hooks: {
      'astro:config:done': ({ config }) => {
        // Use final destinations, including overrides of historical chapter paths.
        redirectTargets = Object.fromEntries(Object.entries(config.redirects).map(([from, target]) => [from, typeof target === 'string' ? target : target.destination]));
      },
      'astro:build:done': async ({ dir, logger }) => {
        const out = fileURLToPath(dir);
        await preserveRedirectFragments(out, redirectTargets);
        const page_count = await indexSearch(out);
        logger.info(`Pagefind indexed ${page_count} pages`);
      },
    },
  };
};

// Hosted on Cloudflare at the domain root. SITE can override the production URL
// (e.g. for a preview deployment); it defaults to the live domain.
const site = process.env.SITE || 'https://blog.shyandram.dev';
const base = process.env.BASE || '/';

export default defineConfig({
  output: 'static',
  site,
  base,
  trailingSlash: 'always',
  security: {
    csp: {
      directives: [
        "default-src 'self'",
        "base-uri 'none'",
        "object-src 'none'",
        "form-action 'none'",
        "frame-src 'none'",
        "connect-src 'self'",
        "img-src 'self' data:",
        // Vite embeds small local font subsets as data URLs.
        "font-src 'self' data:",
      ],
      // Pagefind runs WebAssembly; arbitrary JavaScript eval remains blocked.
      scriptDirective: { resources: ["'self'", "'wasm-unsafe-eval'"] },
      // KaTeX, syntax highlighting and interactive plots use inline styles.
      styleDirective: { resources: ["'self'", "'unsafe-inline'"] },
    },
  },
  redirects: {
    ...legacyChapterRedirects,
    // Former NLP tutorials, reorganized into the 12-topic NLP syllabus.
    '/en/learn/nlp/nlp-00-language-structure-meaning/': '/en/learn/nlp/nlp-02-linguistic-essentials/',
    '/zh-tw/learn/nlp/nlp-00-language-structure-meaning/': '/zh-tw/learn/nlp/nlp-02-linguistic-essentials/',
    '/en/learn/nlp/nlp-01a-ngram-language-models/': '/en/learn/nlp/nlp-05-recurrent-neural-networks/',
    '/zh-tw/learn/nlp/nlp-01a-ngram-language-models/': '/zh-tw/learn/nlp/nlp-05-recurrent-neural-networks/',
    '/en/learn/nlp/nlp-01b-word-embeddings/': '/en/learn/nlp/nlp-04-lexical-semantics-word-embeddings/',
    '/zh-tw/learn/nlp/nlp-01b-word-embeddings/': '/zh-tw/learn/nlp/nlp-04-lexical-semantics-word-embeddings/',
    '/en/learn/nlp/nlp-02-classification-embeddings/': '/en/learn/nlp/nlp-03-neural-networks-for-nlp/',
    '/zh-tw/learn/nlp/nlp-02-classification-embeddings/': '/zh-tw/learn/nlp/nlp-03-neural-networks-for-nlp/',
    '/en/learn/nlp/nlp-03-syntax-parsing/': '/en/learn/nlp/nlp-12-syntactic-semantic-tasks/',
    '/zh-tw/learn/nlp/nlp-03-syntax-parsing/': '/zh-tw/learn/nlp/nlp-12-syntactic-semantic-tasks/',
    '/en/learn/nlp/nlp-04-semantics-extraction/': '/en/learn/nlp/nlp-06-sequence-labeling/',
    '/zh-tw/learn/nlp/nlp-04-semantics-extraction/': '/zh-tw/learn/nlp/nlp-06-sequence-labeling/',
    '/en/learn/nlp/nlp-05-language-models/': '/en/learn/nlp/nlp-09-language-modeling-transformers/',
    '/zh-tw/learn/nlp/nlp-05-language-models/': '/zh-tw/learn/nlp/nlp-09-language-modeling-transformers/',
    '/en/learn/nlp/nlp-06-retrieval-question-answering/': '/en/learn/nlp/nlp-09-language-modeling-transformers/',
    '/zh-tw/learn/nlp/nlp-06-retrieval-question-answering/': '/zh-tw/learn/nlp/nlp-09-language-modeling-transformers/',
    '/en/learn/nlp/nlp-07-discourse-dialogue/': '/en/learn/nlp/nlp-08-seq2seq-transformers/',
    '/zh-tw/learn/nlp/nlp-07-discourse-dialogue/': '/zh-tw/learn/nlp/nlp-08-seq2seq-transformers/',
    '/en/learn/nlp/nlp-08-generation-deployment/': '/en/learn/nlp/nlp-09-language-modeling-transformers/',
    '/zh-tw/learn/nlp/nlp-08-generation-deployment/': '/zh-tw/learn/nlp/nlp-09-language-modeling-transformers/',
    '/en/learn/image-processing/convolution/': '/en/learn/ml/ml-03-convolutional-networks/',
    '/zh-tw/learn/image-processing/convolution/': '/zh-tw/learn/ml/ml-03-convolutional-networks/',
    // Former imported blog posts, now covered by the tutorials.
    '/en/blog/introduction-to-machine-learning/': '/en/learn/ml/ml-01-introduction/',
    '/zh-tw/blog/ml-neural-network-notes/': '/zh-tw/learn/ml/ml-02-neural-networks/',
    '/zh-tw/blog/dl-image-processing-intro/': '/zh-tw/learn/deep-learning/dl-01-deep-learning-for-image-processing/',
    '/zh-tw/blog/image-restoration-intro/': '/zh-tw/learn/deep-learning/dl-04-image-restoration/',
    '/zh-tw/blog/llie-intro/': '/zh-tw/learn/deep-learning/dl-05-low-light-enhancement/',
    '/zh-tw/blog/object-tracking-intro/': '/zh-tw/learn/deep-learning/dl-06-object-tracking/',
  },
  // Serialize static KaTeX/Shiki HTML instead of loading millions of MDX VNodes during prerendering.
  integrations: [mdx({ optimize: true }), pagefindIndex(), sitemap({ i18n: { defaultLocale: 'en', locales: { en: 'en', 'zh-tw': 'zh-TW' } } })],
  i18n: {
    locales: ['en', 'zh-tw'],
    defaultLocale: 'en',
    routing: { prefixDefaultLocale: true, redirectToDefaultLocale: false },
  },
  markdown: {
    // Astro 7 defaults to Sätteri; we use unified for remark-math + rehype-katex.
    processor: unified({ remarkPlugins: [remarkMath, remarkCjkFriendly], rehypePlugins: [rehypeKatex, rehypeCitations, rehypePaperHeadingIds, rehypeHeadingIds, rehypeCodeFolding, rehypeSupplementZones] }),
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' }, wrap: false },
  },
});
