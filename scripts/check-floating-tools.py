"""Browser regression for shared inline/floating reading controls."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:4322'
report = []
with sync_playwright() as p:
    browser = p.chromium.launch(args=['--no-sandbox'])
    for lang in ['en', 'zh-tw']:
        for theme in ['light', 'dark']:
            for width, height in [(320, 800), (375, 900), (768, 1024), (1440, 1000), (844, 390)]:
                page = browser.new_page(viewport={'width': width, 'height': height}, color_scheme=theme, reduced_motion='reduce')
                errors = []
                floating_requests = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.on('request', lambda r: floating_requests.append(r.url) if 'floating-ui.dom.' in r.url else None)
                page.goto(BASE + f'/{lang}/learn/deep-learning/image-classification/')
                page.evaluate('document.fonts.ready')
                toolbar = page.locator('.reader-toolbar')
                trigger = page.locator('.reader-float-trigger')
                popup = page.locator('.reader-popover')
                expect(toolbar).to_have_count(1)
                expect(trigger).to_be_hidden()
                assert not floating_requests, 'Floating UI must load on demand'
                sections = page.locator('.chapter-section')
                codes = page.locator('.code-fold')
                ns, nc = sections.count(), codes.count()
                expect(page.locator('.section-controls .reader-count')).to_contain_text(f'{ns} / {ns}')
                expect(page.locator('.code-controls .reader-count')).to_contain_text(f'0 / {nc}')
                assert page.locator('.reader-toolbar > .section-controls').count() == 1
                assert page.locator('.reader-toolbar > .code-controls').count() == 1
                assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
                if width in [375, 1440]:
                    page.screenshot(path=str(ROOT / f'artifacts/polish-{lang}-{theme}-{width}-entry.png'))
                # The original panel remains in the layout while its controls are moved into the popover.
                page.locator('.code-fold').first.scroll_into_view_if_needed()
                expect(trigger).to_be_visible()
                scroll_before = page.evaluate('scrollY')
                trigger.click()
                expect(popup).to_be_visible()
                expect(trigger).to_have_attribute('aria-expanded', 'true')
                expect(popup.locator('.reader-toolbar')).to_have_count(1)
                expect(popup.locator('.reader-popover-header button')).to_be_focused()
                page.wait_for_function('()=>document.querySelector(".reader-popover").style.left!==""')
                assert floating_requests, 'Floating UI chunk was not loaded'
                assert abs(page.evaluate('scrollY') - scroll_before) < 3
                box = popup.bounding_box()
                assert box['x'] >= 0 and box['y'] >= 0 and box['x'] + box['width'] <= width + 1 and box['y'] + box['height'] <= height + 1, box
                for button in popup.locator('button').all():
                    assert button.bounding_box()['height'] >= 44
                if width in [375, 1440]:
                    page.screenshot(path=str(ROOT / f'artifacts/polish-{lang}-{theme}-{width}-floating.png'))
                # Counts update from individual disclosures, even when controls are floating.
                page.locator('.code-fold').first.evaluate('(e)=>e.open=true')
                expect(popup.locator('.code-controls .reader-count')).to_contain_text(f'1 / {nc}')
                popup.locator('.code-controls button').nth(1).click()
                expect(popup.locator('.code-controls .reader-count')).to_contain_text(f'0 / {nc}')
                # Escape returns focus; normal outside clicks keep their own focus.
                page.keyboard.press('Escape')
                expect(popup).to_be_hidden()
                expect(trigger).to_be_focused()
                expect(page.locator('.reader-toolbar-host > .reader-toolbar')).to_have_count(1)
                trigger.click()
                page.evaluate('''()=>{const b=document.createElement('button');b.id='outside-test';b.textContent='Outside';b.style.cssText='position:fixed;top:0;left:0;z-index:100';document.body.append(b)}''')
                page.locator('#outside-test').click()
                expect(popup).to_be_hidden()
                expect(page.locator('#outside-test')).to_be_focused()
                page.locator('#outside-test').evaluate('(e)=>e.remove()')
                # A resize while open must keep the popover inside the new viewport.
                trigger.click()
                page.set_viewport_size({'width': 320, 'height': 568})
                page.wait_for_function('()=>{const b=document.querySelector(".reader-popover").getBoundingClientRect();return b.x>=0&&b.y>=0&&b.right<=innerWidth+1&&b.bottom<=innerHeight+1}')
                popup.locator('.reader-popover-header button').click()
                expect(popup).to_be_hidden()
                # Printing restores the panel to the document before printing its article.
                trigger.click()
                page.evaluate('dispatchEvent(new Event("beforeprint"))')
                expect(popup).to_be_hidden()
                expect(page.locator('.reader-toolbar-host > .reader-toolbar')).to_have_count(1)
                page.evaluate('dispatchEvent(new Event("afterprint"))')
                assert not errors, errors
                report.append({'lang': lang, 'theme': theme, 'width': width, 'height': height})
                page.close()
    # Optional paper pages without snippets still get one functioning section-only toolbar.
    guides = json.loads((ROOT / 'docs/paper-guide-coverage.json').read_text())['guides']
    paper = next(g for g in guides if g['title'].startswith('Restormer:'))
    for key in ['url', 'urlZh']:
        page = browser.new_page(viewport={'width': 375, 'height': 900}, reduced_motion='reduce')
        page.goto(BASE + paper[key])
        expect(page.locator('.reader-toolbar')).to_have_count(1)
        expect(page.locator('.code-controls')).to_have_count(0)
        page.locator('.chapter-section').nth(2).scroll_into_view_if_needed()
        page.locator('.reader-float-trigger').click()
        expect(page.locator('.reader-popover .section-controls')).to_be_visible()
        page.keyboard.press('Escape')
        expect(page.locator('.reader-popover')).to_be_hidden()
        page.close()
    # Prevent repeated clipboard writes while one is pending; show and announce the final state.
    page = browser.new_page()
    page.goto(BASE + '/en/learn/deep-learning/image-classification/')
    page.evaluate('''()=>{window.copyCalls=0;navigator.clipboard.writeText=()=>{window.copyCalls++;return new Promise(r=>window.finishCopy=r)}}''')
    copy = page.locator('.rt-copy').first
    copy.click()
    expect(copy).to_have_attribute('aria-busy', 'true')
    expect(copy).to_have_attribute('data-state', 'copying')
    copy.click()
    assert page.evaluate('window.copyCalls') == 1
    page.evaluate('window.finishCopy()')
    expect(copy).to_have_attribute('data-state', 'success')
    expect(copy).to_have_text('Copied')
    expect(copy).not_to_have_attribute('aria-busy', 'true')
    expect(copy).to_have_text('Copy', timeout=4000)
    page.close()
    browser.close()
(ROOT / 'docs/worklog/floating-tools-validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'{len(report)} layouts passed: panel state, counts, positioning, resize, focus, dismissal and print; clipboard feedback passed')
