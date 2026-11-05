"""Verify every bilingual paper guide, source image, fold and legacy bookmark in dist.
Run after building and serving dist at CVBLOG_BASE (default port 4322).
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from reader_test_helpers import set_reader_group
from playwright.sync_api import sync_playwright

BASE = os.environ.get('CVBLOG_BASE', 'http://127.0.0.1:4322')
coverage = json.loads(Path('docs/paper-guide-coverage.json').read_text())
guides = coverage['guides']
normalize = lambda s: re.sub(r'\s+', ' ', s).strip()

class Headings(HTMLParser):
    def __init__(self, source):
        super().__init__(); self.headings = {}; self.active = None; self.text = []
        self.feed(source)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if re.fullmatch(r'h[1-6]', tag) and a.get('id'):
            self.active = (tag, a['id']); self.text = []
    def handle_data(self, data):
        if self.active: self.text.append(data)
    def handle_endtag(self, tag):
        if self.active and self.active[0] == tag:
            self.headings[self.active[1]] = normalize(''.join(self.text))
            self.active = None

legacy = json.loads(Path('src/content/paper-heading-ids.json').read_text())['headings']
by_title = {(lang, g['chapter'], g['title' if lang == 'en' else 'titleZh']):
            g['url' if lang == 'en' else 'urlZh'] for g in guides for lang in ['en', 'zh-tw']}
parsed = {}
legacy_cases = []
for h in legacy:
    lang, chapter = h['file'].split('/', 1); chapter = chapter.removesuffix('.mdx')
    url = by_title[(lang, chapter, h['guide'])]
    if url not in parsed:
        parsed[url] = Headings((Path('dist') / url.lstrip('/') / 'index.html').read_text()).headings
    assert parsed[url].get(h['id']) == h['text'], (url, h)
    if 'Retinex-Net' in h['guide'] and h['text'] in ['Challenge', '挑戰']:
        legacy_cases.append((url, h['id']))
assert len(legacy_cases) == 2, legacy_cases
print(f'PASS {len(legacy)} retained subsection IDs and texts', flush=True)

def check_layout(setting):
    lang, width = setting
    count = 0; full_images = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': width, 'height': 900})
        errors = []; page.on('pageerror', lambda e: errors.append(str(e)))
        for g in guides:
            url = g['url' if lang == 'en' else 'urlZh']
            other = g['urlZh' if lang == 'en' else 'url']
            assert page.goto(BASE + url).status == 200, url
            page.evaluate('document.fonts.ready')
            folds = page.locator('.chapter-section > summary > h4')
            assert folds.count() == g['sections'], (url, folds.count(), g['sections'])
            figures = page.locator('.prose figure.fig')
            assert figures.count() == len(g['figures']), url
            for image in figures.locator('img').all():
                image.scroll_into_view_if_needed()
                image.evaluate('(img)=>img.loading="eager"')
                page.wait_for_function('(img)=>img.complete&&img.naturalWidth>0', arg=image.element_handle())
            for anchor in figures.locator('a.fig-zoom').all():
                href = anchor.get_attribute('href')
                if href not in full_images:
                    response = page.request.get(BASE + href)
                    assert response.status == 200 and len(response.body()) > 1000, (url, href)
                    full_images.add(href)
            links = page.locator('a[href]').evaluate_all('(a)=>a.map(e=>decodeURI(new URL(e.href).pathname))')
            assert unquote(other) in links, ('language counterpart', url)
            parent = url.split('/papers/')[0] + '/'
            assert unquote(parent) in links, ('parent chapter', url)
            assert not page.evaluate('document.documentElement.scrollWidth>innerWidth'), url
            set_reader_group(page, 'sections', False)
            assert page.locator('.chapter-section[open]').count() == 0, url
            heading = folds.first
            target = heading.get_attribute('id')
            page.evaluate('(id)=>{location.hash=id}', target)
            page.wait_for_function('(id)=>document.getElementById(id).closest("details").open', arg=target)
            summary = heading.locator('..'); summary.focus(); summary.press('Enter')
            assert summary.locator('..').get_attribute('open') is None, url
            set_reader_group(page, 'sections', True)
            assert page.locator('.chapter-section:not([open])').count() == 0, url
            assert not page.evaluate('document.documentElement.scrollWidth>innerWidth'), ('after reopening', url)
            if any(s in g['title'].lower() for s in ['restormer:', 'iat:', 'cpga-net+:', 'nafnet:', 'kind:', 'mathematical morphology', 'agenticir:', 'yolox:', 'yolov10:', 'yolo26:']):
                page.locator('.prose figure').first.scroll_into_view_if_needed()
                slug = url.rstrip('/').split('/')[-1][:75]
                page.screenshot(path=f'artifacts/paper-guides/{lang}-{width}-{slug}.png')
            count += 1
            if count % 25 == 0: print('PASS layouts', lang, width, count, flush=True)
        for url, fragment in legacy_cases:
            if not url.startswith('/' + lang + '/'): continue
            for old in [url, url.split('/papers/')[0] + '/', url.split('/papers/')[0] + '/papers/']:
                page.goto(BASE + old + '#' + fragment)
                page.wait_for_url(lambda u: unquote(urlsplit(u).path) == url and unquote(urlsplit(u).fragment) == fragment)
                page.wait_for_function('(id)=>!!document.getElementById(id)&&!document.getElementById(id).closest("details:not([open])")', arg=fragment)
        assert not errors, errors
        browser.close()
    return {'lang': lang, 'width': width, 'guides': count, 'fullSizeImages': len(full_images)}

Path('artifacts/paper-guides').mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(check_layout, [(lang, width) for lang in ['en', 'zh-tw'] for width in [1440, 375]]))
report = {'guideUnits': len(guides), 'layouts': sum(r['guides'] for r in results),
          'retainedSubsectionBookmarks': len(legacy), 'legacyBrowserRoutes': len(legacy_cases) * 3 * 2, 'results': results,
          'checks': ['images load and open at full size', 'language counterpart', 'parent chapter',
                     'horizontal bounds', 'fold all/open all', 'keyboard', 'fragment reveal',
                     'legacy main/index/item forwarding', 'JavaScript errors']}
Path('docs/worklog/paper-guide-browser-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
