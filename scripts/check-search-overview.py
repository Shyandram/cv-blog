"""Production browser checks for the search palette and generated About overview.

Run after npm run build: .venv/bin/python scripts/check-search-overview.py
"""
import json
import runpy
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
security = runpy.run_path(str(ROOT / 'scripts/check-security.py'))
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(security['Handler'], directory=str(ROOT / 'dist')))
threading.Thread(target=server.serve_forever, daemon=True).start()
origin = f'http://127.0.0.1:{server.server_port}'

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)
        for lang, query in [('en', 'convolution'), ('zh-tw', '卷積')]:
            for width in [1440, 375]:
                page.set_viewport_size({'width': width, 'height': 1000})
                page.goto(f'{origin}/{lang}/about/')
                page.evaluate('document.fonts.ready')
                expect(page.locator('.overview-cards dd')).to_have_count(6)
                expect(page.locator('.track-index tbody tr')).to_have_count(8)
                rows = page.locator('.track-index tbody tr').all()
                for col in [1, 2]:
                    total = sum(int(row.locator('td').nth(col - 1).inner_text().replace(',', '')) for row in rows)
                    assert total == int(page.locator('.track-index tfoot td').nth(col - 1).inner_text().replace(',', ''))
                for a in page.locator('.site-overview a').all():
                    target = urlsplit(a.get_attribute('href'))
                    built = ROOT / 'dist' / target.path.lstrip('/') / 'index.html'
                    assert built.is_file(), target
                    if target.fragment:
                        assert f'id="{target.fragment}"' in built.read_text(), target
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.screenshot(path=f'/tmp/cv-blog-overview-{lang}-{width}.png', full_page=True)
                trigger = page.locator('pagefind-modal-trigger button')
                expect(trigger).to_be_visible()
                trigger.click()
                dialog = page.locator('pagefind-modal dialog')
                field = dialog.locator('input[type=search]')
                expect(dialog).to_be_visible()
                expect(field).to_be_focused()
                field.fill(query)
                results = dialog.locator('pagefind-results a')
                expect(results.first).to_be_visible(timeout=20000)
                assert all(f'/{lang}/' in a.get_attribute('href') for a in results.all())
                field.press('ArrowDown')
                expect(results.first).to_be_focused()
                page.keyboard.press('ArrowDown')
                expect(results.nth(1)).to_be_focused()
                page.keyboard.press('ArrowUp')
                expect(results.first).to_be_focused()
                for _ in range(12):
                    page.keyboard.press('Tab')
                    assert dialog.evaluate('el => el.contains(document.activeElement)')
                assert dialog.evaluate('el => el.scrollWidth <= el.clientWidth')
                field.focus()
                dialog.locator('pagefind-modal-body').evaluate('el => el.scrollTop = 0')
                page.screenshot(path=f'/tmp/cv-blog-palette-{lang}-{width}.png', animations='disabled')
                page.keyboard.press('Escape')
                expect(dialog).not_to_be_visible()
                expect(trigger).to_be_focused()
                expect(trigger).to_have_attribute('aria-expanded', 'false')
                page.keyboard.press('Control+k')
                expect(dialog).to_be_visible()
                expect(field).to_be_focused()
                field.press('ArrowDown')
                target = results.first.get_attribute('href')
                page.keyboard.press('Enter')
                page.wait_for_url(origin + target)
                expect(page.locator('pagefind-modal-trigger button')).to_be_visible()
                page.keyboard.press('Control+k')
                expect(dialog).to_be_visible()
                field.fill('"zzzxqvunlikelyresult"')
                expect(dialog.locator('pagefind-summary')).to_contain_text('zzzxqvunlikelyresult')
                expect(dialog.locator('pagefind-results a')).to_have_count(0)
                field.fill('<svg onload="window.__searchInjected=1">')
                expect(dialog.locator('pagefind-summary')).to_contain_text('onload')
                assert page.evaluate('window.__searchInjected === undefined')
                page.keyboard.press('Escape')
                page.locator('#theme-toggle').click()
                page.keyboard.press('Control+k')
                field.fill(query)
                expect(dialog.locator('pagefind-summary')).to_contain_text(query)
                expect(dialog.locator('.pf-results')).to_have_attribute('aria-busy', 'false')
                expect(results.first).to_be_visible()
                page.screenshot(path=f'/tmp/cv-blog-palette-themed-{lang}-{width}.png', animations='disabled')
                # Mobile has an explicit close button; desktop also closes on backdrop click.
                if width == 375:
                    dialog.locator('.pf-modal-close').click()
                else:
                    page.mouse.click(5, 5)
                expect(dialog).not_to_be_visible()
        assert not errors, errors
        mac = browser.new_context().new_page()
        mac.add_init_script("Object.defineProperty(navigator, 'userAgentData', {value: {platform: 'macOS'}})")
        mac.goto(origin + '/en/')
        expect(mac.locator('pagefind-modal-trigger button')).to_have_attribute('aria-keyshortcuts', 'Meta+k')
        mac.keyboard.press('Meta+k')
        expect(mac.locator('pagefind-modal dialog')).to_be_visible()
        mac.keyboard.press('Escape')
        expect(mac.locator('pagefind-modal dialog')).not_to_be_visible()
        fallback = browser.new_context(java_script_enabled=False).new_page()
        fallback.goto(origin + '/en/')
        link = fallback.locator('pagefind-modal-trigger a')
        expect(link).to_be_visible()
        assert link.get_attribute('href') == '/en/search/'
        expect(fallback.locator('pagefind-modal')).not_to_be_visible()
        browser.close()
finally:
    server.shutdown()
    server.server_close()
print(json.dumps({'search_palette': 'passed', 'about_overview': 'passed', 'languages': 2, 'widths': [1440, 375]}))
