import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { LANGS, type Lang, useT } from '../../i18n/ui';
import { postsFor, tutorialsFor, url } from '../../lib/site';

export function getStaticPaths() {
  return LANGS.map((lang) => ({ params: { lang } }));
}

export async function GET(context: APIContext) {
  const lang = context.params.lang as Lang;
  const t = useT(lang);
  const posts = await postsFor(lang);
  const tutorials = (await tutorialsFor(lang)).filter((x) => !x.fallback);
  const items = [
    ...posts.map((p) => ({ title: p.entry.data.title, description: p.entry.data.description, pubDate: p.entry.data.published, link: url(`/${lang}/blog/${p.slug}/`) })),
    ...tutorials.map((x) => ({ title: x.entry.data.title, description: x.entry.data.description, pubDate: x.entry.data.published, link: url(`/${lang}/learn/${x.slug}/`) })),
  ].sort((a, b) => b.pubDate.valueOf() - a.pubDate.valueOf());
  return rss({ title: t('site.name'), description: t('site.tagline'), site: context.site!, items, customData: `<language>${lang}</language>` });
}
