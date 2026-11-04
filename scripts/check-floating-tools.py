"""Persistent four-button console: production-browser behavior and geometry."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from reader_test_helpers import set_reader_group

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:4322'
report = []

def bounded(page):
    page.wait_for_function('''()=>{const r=document.querySelector('.reader-console').getBoundingClientRect();
      return r.left>=11 && r.top>=11 && r.right<=innerWidth-11 && r.bottom<=innerHeight-11}''')
    assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')

with sync_playwright() as p:
    browser = p.chromium.launch(args=['--no-sandbox'])
    for lang in ['en', 'zh-tw']:
        for theme in ['light', 'dark']:
            for width, height in [(320, 800), (375, 900), (768, 1024), (1440, 1000), (844, 390)]:
                page = browser.new_page(viewport={'width': width, 'height': height}, color_scheme=theme, reduced_motion='reduce')
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                assert page.goto(BASE + f'/{lang}/learn/deep-learning/image-classification/').status == 200
                page.evaluate('document.fonts.ready')
                panel = page.locator('.reader-console')
                body = panel.locator('.reader-console-body')
                contents = panel.locator('.reader-console-toggle')
                handle = panel.locator('.reader-console-handle')
                expect(panel).to_have_count(1)
                expect(panel).to_be_visible()
                expect(body).to_be_hidden()
                expect(panel.locator('.reader-console-header > button:not(.reader-console-handle)')).to_have_count(4)
                assert panel.bounding_box()['height'] <= 70
                assert page.locator('.code-fold[open]').count() == 0
                for button in panel.locator('.reader-console-header button').all():
                    box = button.bounding_box()
                    assert box['width'] >= 44 and box['height'] >= 44, box
                bounded(page)
                if width in [375, 1440]:
                    page.screenshot(path=str(ROOT / f'artifacts/console-{lang}-{theme}-{width}-compact.png'))
                # One primary action toggles all; partial individual state stays truthful.
                set_reader_group(page, 'sections', False)
                expect(panel.locator('.reader-console-fold-sections')).to_have_attribute('aria-pressed', 'false')
                set_reader_group(page, 'code', True)
                expect(panel.locator('.reader-console-fold-code')).to_have_attribute('aria-pressed', 'true')
                page.locator('.code-fold').first.evaluate('(e)=>e.open=false')
                expect(panel.locator('.reader-console-fold-code')).to_have_attribute('aria-pressed', 'mixed')
                set_reader_group(page, 'code', False)
                set_reader_group(page, 'sections', False)
                contents.click()
                expect(body).to_be_visible()
                bounded(page)
                options = panel.locator('select option')
                assert options.count() > 10
                panel.locator('select').select_option('2')
                page.wait_for_function('()=>location.hash.length>1 && document.querySelectorAll(".chapter-section[open]").length>0')
                assert panel.locator('progress').get_attribute('value') is not None
                bounded(page)
                # It remains usable after scrolling/clicking outside; only its tray closes on Escape.
                page.evaluate('''()=>{const b=document.createElement('button');b.id='outside-test';b.textContent='Outside';b.style.cssText='position:fixed;top:0;left:0;z-index:100';document.body.append(b)}''')
                page.locator('#outside-test').click()
                expect(body).to_be_visible()
                page.locator('#outside-test').evaluate('(e)=>e.remove()')
                if width in [375, 1440]:
                    page.screenshot(path=str(ROOT / f'artifacts/console-{lang}-{theme}-{width}-contents.png'))
                contents.focus()
                page.keyboard.press('Escape')
                expect(body).to_be_hidden()
                expect(contents).to_be_focused()
                expect(panel).to_be_visible()
                # Pointer drag and keyboard movement are clamped, saved, and restored.
                before = panel.bounding_box()
                grip = handle.bounding_box()
                page.mouse.move(grip['x'] + 20, grip['y'] + 20)
                page.mouse.down()
                page.mouse.move(grip['x'] + 20, grip['y'] - 90, steps=5)
                page.mouse.up()
                assert panel.bounding_box()['y'] < before['y'] - 80
                handle.focus()
                before = panel.bounding_box()
                page.keyboard.press('ArrowUp')
                assert abs(panel.bounding_box()['y'] - before['y'] + 16) < 2
                saved = panel.bounding_box()
                page.reload()
                expect(panel).to_be_visible()
                assert abs(panel.bounding_box()['y'] - saved['y']) < 2
                expect(body).to_be_hidden()
                handle.focus()
                page.keyboard.press('Home')
                assert abs(panel.bounding_box()['y'] + panel.bounding_box()['height'] - height + 12) < 2
                contents.click()
                page.set_viewport_size({'width': 320, 'height': 300})
                bounded(page)
                page.set_viewport_size({'width': width, 'height': height})
                bounded(page)
                panel.locator('.reader-console-top').click()
                page.wait_for_function('()=>scrollY===0')
                page.emulate_media(media='print')
                expect(panel).to_be_hidden()
                page.emulate_media(media='screen')
                expect(body).to_be_visible()
                assert not errors, errors
                report.append({'lang': lang, 'theme': theme, 'width': width, 'height': height})
                print('PASS console', lang, theme, width, flush=True)
                page.close()
    # Paper pages without code still expose working contents and chapter controls.
    guides = json.loads((ROOT / 'docs/paper-guide-coverage.json').read_text())['guides']
    paper = next(g for g in guides if g['title'].startswith('Restormer:'))
    for key in ['url', 'urlZh']:
        page = browser.new_page(viewport={'width': 375, 'height': 900}, reduced_motion='reduce')
        page.goto(BASE + paper[key])
        expect(page.locator('.reader-console')).to_have_count(1)
        expect(page.locator('.reader-console-fold-code')).to_be_disabled()
        set_reader_group(page, 'sections', False)
        page.locator('.reader-console-toggle').click()
        page.locator('.reader-console-nav button').last.click()
        page.wait_for_function('()=>location.hash.length>1')
        bounded(page)
        page.close()
    # Position persistence is optional; restrictive storage must not break reading.
    page = browser.new_page(viewport={'width': 375, 'height': 900})
    page.add_init_script('''Storage.prototype.getItem=Storage.prototype.setItem=()=>{throw new Error('Storage blocked')}''')
    page.goto(BASE + '/en/learn/deep-learning/image-classification/')
    expect(page.locator('.reader-console')).to_be_visible()
    page.locator('.reader-console-toggle').click()
    page.locator('.reader-console-dock').click()
    bounded(page)
    page.close()
    # Real touch events on the grip move the console without scrolling the article.
    context = browser.new_context(viewport={'width': 375, 'height': 812}, is_mobile=True, has_touch=True)
    page = context.new_page()
    page.goto(BASE + '/zh-tw/learn/deep-learning/image-classification/')
    grip = page.locator('.reader-console-handle')
    expect(grip).to_be_visible()
    box = grip.bounding_box()
    client = context.new_cdp_session(page)
    point = {'x': box['x'] + 22, 'y': box['y'] + 22}
    start_scroll = page.evaluate('scrollY')
    start_y = page.locator('.reader-console').bounding_box()['y']
    client.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [point]})
    point['y'] -= 100
    client.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [point]})
    client.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
    assert page.locator('.reader-console').bounding_box()['y'] < start_y - 80
    assert page.evaluate('scrollY') == start_scroll
    context.close()
    browser.close()
(ROOT / 'docs/worklog/floating-console-validation.json').write_text(json.dumps({'layouts': report, 'additional': ['code-free papers in both languages', 'blocked storage', 'mobile touch drag']}, indent=2) + '\n')
print(f'{len(report)} layouts passed: four actions, tray, pointer/keyboard movement, persistence, resizing and print; paper/storage/touch checks passed')
