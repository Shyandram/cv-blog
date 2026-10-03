// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// Hosted on Cloudflare Pages at the domain root. Set SITE in the Cloudflare project's
// environment variables to the production URL (e.g. https://cv-blog.pages.dev or a custom domain);
// CF_PAGES_URL (the per-deployment URL) is the fallback for preview builds.
const site = process.env.SITE || process.env.CF_PAGES_URL || 'http://localhost:4321';
const base = process.env.BASE || '/';

export default defineConfig({
  site,
  base,
  trailingSlash: 'always',
  integrations: [mdx(), sitemap({ i18n: { defaultLocale: 'en', locales: { en: 'en', 'zh-tw': 'zh-TW' } } })],
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
