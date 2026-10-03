// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import remarkCjkFriendly from 'remark-cjk-friendly';
import rehypeKatex from 'rehype-katex';
import { fileURLToPath } from 'node:url';
import * as pagefind from 'pagefind';

// Build the Pagefind search index as part of `astro build`, so search works no matter
// which build command the host runs (`astro build`, `npm run build`, …).
/** @returns {import('astro').AstroIntegration} */
const pagefindIndex = () => ({
  name: 'pagefind-index',
  hooks: {
    'astro:build:done': async ({ dir, logger }) => {
      const out = fileURLToPath(dir);
      const { index, errors } = await pagefind.createIndex();
      if (!index) throw new Error(`Pagefind failed to start: ${errors.join('; ')}`);
      const { page_count } = await index.addDirectory({ path: out });
      await index.writeFiles({ outputPath: `${out}pagefind` });
      await pagefind.close();
      logger.info(`Pagefind indexed ${page_count} pages`);
    },
  },
});

// Hosted on Cloudflare at the domain root. SITE can override the production URL
// (e.g. for a preview deployment); it defaults to the live domain.
const site = process.env.SITE || 'https://blog.shyandram.dev';
const base = process.env.BASE || '/';

export default defineConfig({
  site,
  base,
  trailingSlash: 'always',
  redirects: {
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
  integrations: [mdx(), pagefindIndex(), sitemap({ i18n: { defaultLocale: 'en', locales: { en: 'en', 'zh-tw': 'zh-TW' } } })],
  i18n: {
    locales: ['en', 'zh-tw'],
    defaultLocale: 'en',
    routing: { prefixDefaultLocale: true, redirectToDefaultLocale: false },
  },
  markdown: {
    // Astro 7 defaults to Sätteri; we use unified for remark-math + rehype-katex.
    processor: unified({ remarkPlugins: [remarkMath, remarkCjkFriendly], rehypePlugins: [rehypeKatex] }),
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' }, wrap: false },
  },
});
