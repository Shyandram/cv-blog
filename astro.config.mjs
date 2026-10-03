// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { fileURLToPath } from 'node:url';
import * as pagefind from 'pagefind';

// Build the Pagefind search index as part of `astro build`, so search works no matter
// which build command the host runs (`astro build`, `npm run build`, …).
const pagefindIndex = () => ({
  name: 'pagefind-index',
  hooks: {
    'astro:build:done': async ({ dir, logger }) => {
      const out = fileURLToPath(dir);
      const { index } = await pagefind.createIndex();
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
  integrations: [mdx(), pagefindIndex(), sitemap({ i18n: { defaultLocale: 'en', locales: { en: 'en', 'zh-tw': 'zh-TW' } } })],
  i18n: {
    locales: ['en', 'zh-tw'],
    defaultLocale: 'en',
    routing: { prefixDefaultLocale: true, redirectToDefaultLocale: false },
  },
  markdown: {
    // Astro 7 defaults to Sätteri; we use unified for remark-math + rehype-katex.
    processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }),
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' }, wrap: false },
  },
});
