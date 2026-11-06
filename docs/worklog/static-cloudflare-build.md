# Static Cloudflare build — 2026-11-06

The reported build completed Vite bundling, printed the static-route heading
and parsed header rules, then produced no first-page result before timeout.
The repository had no Wrangler configuration. Cloudflare's documented automatic
Astro configuration installs `@astrojs/cloudflare`, changing the runtime used
for prerendering. That differs from this site's successful local Node static
build and is unnecessary for its pre-rendered content.

Changes:

- Add `wrangler.json` with static assets from `dist`, a custom `npm run build`
  command and custom 404 handling. There is no Worker entry point or server
  adapter. An explicit Wrangler configuration disables framework autoconfiguration.
- Make Astro's static output explicit and enable the supported MDX static HTML
  optimization. This reduces the JavaScript work needed to load math/code-heavy
  chapters during static-path enumeration. Custom uppercase zone/figure/lab
  components remain dynamic.
- Document Workers Builds with an empty dashboard build command and
  `npx wrangler deploy` as the deploy command, so the custom build runs once.
  Keep the existing Pages instructions. Workers Builds overrides the local
  default Worker name with its connected Worker; no account/domain is embedded.

## Validation

Wrangler **4.158.0** was installed only in a scratch tooling directory. Running
`wrangler deploy --dry-run` against the checked-in configuration ran the complete
custom build and validated static asset packaging without publishing:

- Exit 0; wall time **165.74 seconds** (Astro reports 2m 40s).
- Maximum child-process RSS: **3,828,656 KiB** (Linux `RUSAGE_CHILDREN`, not a sum
  of all processes' simultaneous memory).
- 1,268 generated content pages; 1,267 Pagefind-indexed pages; 8,890 asset files
  recognized by Wrangler. No adapter installation or server entry-point setup.
- All 1,267 non-redirect pages match the previous output's normalized text,
  exact code text, ordered bookmark IDs and link destinations. No content pages
  added or removed.
- Static link scan: 2,233 HTML pages, 149,951 links, zero broken links.
- All 334 unit tests pass. Astro check: zero errors/warnings, three existing hints.
- Playwright: all 396 bilingual desktop/mobile course/note layouts pass default
  folding, keyboard controls, language routes, viewport overflow and JS-error
  checks. Eight homepage Track cases pass with/without JavaScript.
- CSP policy and inline-script hashes verified for all 1,267 content pages;
  existing browser security tests pass search/WASM, local fonts, theme/language
  controls, lab injection checks and blocking injected scripts/string eval.
  Legacy redirect utility pages are outside this content-page CSP audit.

The actual hosted Cloudflare build has not been rerun here. Validation establishes
that Wrangler uses the static path and packages the site locally; hosted timing
still depends on the Cloudflare build environment. No production deployment or
merge was performed.

## Primary documentation opened

- https://developers.cloudflare.com/workers/framework-guides/automatic-configuration/
- https://developers.cloudflare.com/workers/framework-guides/web-apps/astro/#if-you-have-a-static-site
- https://developers.cloudflare.com/workers/ci-cd/builds/troubleshoot/
- https://docs.astro.build/en/guides/integrations-guide/mdx/#optimize
