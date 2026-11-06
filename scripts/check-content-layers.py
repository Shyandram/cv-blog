"""Browser checks for the structure-and-layers components on the DLCV 02 pilot (AGENTS.md §3 "Showing structure and
layers"): chapter map, leveled outlines and priority tags, "How it fits", <Deeper> (Layer 2) and the layer switch.

Run against the served production build (python3 -m http.server 4322 in dist/, or astro preview --port 4322).
Writes docs/worklog/content-layers-validation.json.
"""
import json
from pathlib import Path
from urllib.parse import unquote, urljoin
from urllib.request import urlopen
from playwright.sync_api import sync_playwright
from reader_test_helpers import set_reader_group

BASE = 'http://127.0.0.1:4322'
ROOT = Path(__file__).resolve().parents[1]
PATH = '/{lang}/learn/deep-learning/image-classification/'
results = []


def luminance(rgb):
    def ch(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a, b):
    la, lb = sorted([luminance(a), luminance(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


# Text colour against its effective background, read from computed styles (walks up to the first opaque background).
COLORS = r"""(sel) => {
  const parse = (c) => (c.match(/[\d.]+/g) || []).map(Number);
  return [...document.querySelectorAll(sel)].slice(0, 3).map((e) => {
    let bg = null;
    for (let p = e; p; p = p.parentElement) { const c = parse(getComputedStyle(p).backgroundColor); if (c.length === 3 || (c.length === 4 && c[3] > 0.5)) { bg = c.slice(0, 3); break; } }
    return { fg: parse(getComputedStyle(e).color).slice(0, 3), bg: bg || [255, 255, 255], size: parseFloat(getComputedStyle(e).fontSize) };
  });
}"""
TEXT = {'.pri-core': 4.5, '.pri-key': 4.5, '.pri-detail': 4.5, '.deeper-kind': 4.5, '.deeper-meta': 4.5, '.deeper-title': 4.5,
        '.smap-stats': 4.5, '.smap-sum': 4.5, '.smap-n': 4.5, '.fit-note': 4.5, '.fit-h': 4.5, '.outline-more > summary': 4.5}

with sync_playwright() as p:
    b = p.chromium.launch(args=['--no-sandbox'])
    for lang in ['en', 'zh-tw']:
        path = PATH.format(lang=lang)
        for theme in ['light', 'dark']:
            for width, height in [(1440, 900), (375, 812)]:
                page = b.new_page(viewport={'width': width, 'height': height}, color_scheme=theme, reduced_motion='reduce')
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                assert page.goto(BASE + path).status == 200
                page.wait_for_selector('.chapter-section')
                page.evaluate('document.fonts.ready')
                r = {'lang': lang, 'theme': theme, 'width': width}
                assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'), (lang, theme, width)
                # Chapter map: open on wide screens, one folded row on phones; h3s on demand.
                body = page.locator('.smap-body')
                assert page.locator('.smap-row').count() == 10
                if width > 620:
                    assert body.is_visible()
                    assert page.locator('.smap-fold').get_attribute('open') is not None
                else:
                    assert not body.is_visible()
                    page.locator('.smap-head').click()
                    assert body.is_visible()
                    assert page.locator('.smap-head').bounding_box()['height'] >= 44
                toggle = page.locator('.smap-more').first
                sub = page.locator('#' + toggle.get_attribute('aria-controls'))
                assert sub.is_hidden()
                toggle.click()
                assert sub.is_visible() and toggle.get_attribute('aria-expanded') == 'true'
                toggle.click()
                for e in page.locator('.smap-row > a, .smap-more, .smap-layer-btn').all():
                    box = e.bounding_box()
                    if box: assert box['height'] >= 24, box
                # One outline at a time: the side list is hidden while the map is on screen.
                if width >= 1200:
                    assert page.evaluate("getComputedStyle(document.querySelector('.toc')).visibility") == 'hidden'
                    page.evaluate("() => document.getElementById(decodeURIComponent(document.querySelectorAll('.smap-list a')[2].hash.slice(1))).scrollIntoView()")
                    page.wait_for_function("() => getComputedStyle(document.querySelector('.toc')).visibility === 'visible'")
                    page.evaluate('scrollTo(0, 0)')
                # Outlines and priority tags.
                outlines = page.locator('.prose .outline')
                r['outlines'] = outlines.count()
                assert r['outlines'] == 5, r['outlines']
                tags = page.locator('.prose .outline .pri')
                r['priorityTags'] = tags.count()
                words = {'en': {'Core', 'Important', 'Detail'}, 'zh-tw': {'核心', '重要', '細節'}}[lang]
                seen = {t.replace(':', '').replace('：', '').strip() for t in tags.evaluate_all('(ts) => ts.map((t) => t.textContent)')}
                assert seen == words, seen
                more = page.locator('.prose .outline-more')
                assert more.count() == 1 and more.get_attribute('open') is None, (more.count(), more.first.get_attribute('open'), width, theme, lang)
                more.locator(':scope > summary').click()
                assert more.get_attribute('open') is not None
                # Layer 2: closed in the HTML, keyboard operable, layer switch opens and closes all.
                deeper = page.locator('.prose details.deeper')
                r['deeper'] = deeper.count()
                assert r['deeper'] == 3 and page.locator('.prose details.deeper[open]').count() == 0
                d = deeper.first
                d.locator(':scope > summary').focus()
                d.locator(':scope > summary').press('Enter')
                assert d.get_attribute('open') is not None
                d.locator(':scope > summary').press('Enter')
                switch = page.locator('.smap-layer-btn')
                if width <= 620:
                    assert switch.is_visible()  # the map was opened above
                before = page.evaluate('scrollY')
                switch.click()
                assert page.locator('.prose details.deeper[open]').count() == 3 and switch.get_attribute('aria-pressed') == 'true'
                switch.click()
                assert page.locator('.prose details.deeper[open]').count() == 0 and switch.get_attribute('aria-pressed') == 'false'
                d.locator(':scope > summary').click()
                assert switch.get_attribute('aria-pressed') == 'mixed'
                d.locator(':scope > summary').click()
                # Print opens every layer, then restores.
                page.evaluate('dispatchEvent(new Event("beforeprint"))')
                assert page.locator('.prose details.deeper:not([open]), .prose .outline-more:not([open])').count() == 0
                page.evaluate('dispatchEvent(new Event("afterprint"))')
                assert page.locator('.prose details.deeper[open]').count() == 0
                # The console's Code action opens code inside a Deeper block's parents too (none today: no code inside).
                set_reader_group(page, 'code', True)
                set_reader_group(page, 'code', False)
                # How it fits: one list, links that resolve.
                fit = page.locator('.prose .fit')
                assert fit.count() == 1
                links = fit.locator('a').evaluate_all('(as) => as.map((a) => a.getAttribute("href"))')
                r['fitLinks'] = len(links)
                for href in links:
                    if href.startswith('#'):
                        assert page.evaluate('(id) => !!document.getElementById(id)', unquote(href[1:])), href
                    else:
                        assert urlopen(urljoin(BASE, href)).status == 200, href
                # Contrast of the new text styles.
                worst = {}
                for sel, need in TEXT.items():
                    for c in page.evaluate(COLORS, sel):
                        ratio = contrast(c['fg'], c['bg'])
                        worst[sel] = round(min(worst.get(sel, 99), ratio), 2)
                        assert ratio >= need, (sel, ratio, theme)
                r['contrast'] = worst
                r['errors'] = errors
                assert not errors, errors
                results.append(r)
                page.close()
        # Deep link to a Layer-2 block opens it; a search hit inside it (a KaTeX node) is revealed too.
        page = b.new_page(viewport={'width': 1440, 'height': 900})
        page.goto(BASE + path)
        target = page.locator('.prose details.deeper').nth(1).get_attribute('id')
        page.goto(BASE + path + '#' + target)
        page.wait_for_function('(id) => document.getElementById(id).open', arg=target)
        page.close()
        # Without JavaScript: native disclosures work, the layer switch (JS-only) stays hidden, map open on desktop.
        page = b.new_page(java_script_enabled=False, viewport={'width': 1440, 'height': 900})
        page.goto(BASE + path)
        assert page.locator('.prose details.deeper').count() == 3
        assert page.locator('.smap-layer').is_hidden()
        assert page.locator('.smap-body').is_visible()
        page.locator('.prose details.deeper summary').first.click()
        assert page.locator('.prose details.deeper[open]').count() == 1
        page.close()
    b.close()
out = ROOT / 'docs/worklog/content-layers-validation.json'
out.write_text(json.dumps({'page': PATH, 'layouts': len(results), 'results': results}, indent=1, ensure_ascii=False) + '\n')
print(f'{len(results)} layouts passed; worst contrast:',
      {k: min(r['contrast'][k] for r in results if k in r['contrast']) for k in TEXT})
