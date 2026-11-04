"""Check folded code against a production build served on port 4322."""
import json
import re
from pathlib import Path
from html.parser import HTMLParser
from reader_test_helpers import set_reader_group
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:4322'
results = []

class CodeMarkup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.details = []
        self.pre = None
        self.blocks = 0
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'details':
            folded_code = 'code-fold' in a.get('class', '').split()
            if folded_code: assert 'open' not in a
            self.details.append(folded_code)
        if tag == 'pre': self.pre = a
        if tag == 'code' and self.pre is not None:
            assert any(self.details) and self.pre.get('id'), 'code must start in a native disclosure with a bookmark'
            self.blocks += 1
    def handle_endtag(self, tag):
        if tag == 'details': self.details.pop()
        if tag == 'pre': self.pre = None

static_pages = 0
static_blocks = 0
for html in (ROOT / 'dist').rglob('*.html'):
    scan = CodeMarkup()
    scan.feed(html.read_text())
    if scan.blocks:
        static_pages += 1
        static_blocks += scan.blocks

with sync_playwright() as p:
    browser = p.chromium.launch(args=['--no-sandbox'])
    for lang in ['en', 'zh-tw']:
        path = f'/{lang}/learn/deep-learning/image-classification/'
        source = (ROOT / f'src/content/tutorials/{lang}/deep-learning/dl-02-image-classification.mdx').read_text()
        expected = re.search(r'```python\n(.*?)\n```', source, re.S).group(1)
        for theme in ['light', 'dark']:
            for width in [320, 375, 1440]:
                page = browser.new_page(viewport={'width': width, 'height': 900}, color_scheme=theme,
                                        permissions=['clipboard-read', 'clipboard-write'], reduced_motion='reduce')
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                assert page.goto(BASE + path).status == 200
                page.evaluate('document.fonts.ready')
                page.wait_for_selector('.rt-copy')
                folds = page.locator('.code-fold')
                count = folds.count()
                assert count > 1 and page.locator('.code-fold[open]').count() == 0
                first = folds.first
                summary = first.locator('summary')
                pre = first.locator('pre')
                assert not pre.is_visible()
                assert 'Python' in summary.inner_text()
                assert str(len(expected.splitlines())) in summary.inner_text()
                copy = first.locator('..').locator('.rt-copy')
                copy.click()
                assert page.evaluate('navigator.clipboard.readText()') == expected
                assert first.get_attribute('open') is None
                assert copy.bounding_box()['height'] >= 44 and summary.bounding_box()['height'] >= 44
                label = summary.locator('.code-label').bounding_box()
                assert label['x'] + label['width'] <= copy.bounding_box()['x']
                if width in [375, 1440]:
                    first.locator('..').screenshot(path=str(ROOT / f'artifacts/code-{lang}-{theme}-{width}-closed.png'))
                summary.focus(); summary.press('Enter')
                assert pre.is_visible()
                assert pre.locator('code').text_content().removesuffix('\n') == expected
                assert pre.locator('span[style]').count() > 0
                assert copy.bounding_box()['y'] + copy.bounding_box()['height'] <= pre.bounding_box()['y']
                assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
                if width in [375, 1440]:
                    first.locator('..').screenshot(path=str(ROOT / f'artifacts/code-{lang}-{theme}-{width}-open.png'))
                summary.press('Space'); assert first.get_attribute('open') is None
                # Code expansion makes its ancestors visible; chapter-only controls do not change code state.
                set_reader_group(page, 'sections', False)
                set_reader_group(page, 'code', True)
                assert page.locator('.code-fold[open]').count() == count and pre.is_visible()
                set_reader_group(page, 'code', False)
                assert page.locator('.code-fold[open]').count() == 0
                set_reader_group(page, 'sections', True)
                assert page.locator('.code-fold[open]').count() == 0
                # Restore a mix of open and closed code, as well as closed parent sections, after printing.
                summary.click()
                set_reader_group(page, 'sections', False)
                page.evaluate('dispatchEvent(new Event("beforeprint"))')
                assert page.locator('.code-fold[open]').count() == count
                assert pre.is_visible()
                page.evaluate('dispatchEvent(new Event("afterprint"))')
                assert page.locator('.code-fold[open]').count() == 1
                assert page.locator('.chapter-section[open]').count() == 0
                # A direct code bookmark reveals both levels on a fresh navigation.
                code_id = pre.get_attribute('id')
                assert code_id
                page.goto(BASE + path + '#' + code_id)
                page.wait_for_function('(id)=>document.getElementById(id).closest("details").open', arg=code_id)
                assert page.locator('#' + code_id).is_visible()
                page.goto(BASE + path)
                assert page.locator('.code-fold[open]').count() == 0
                # Permission/API failure exposes code for manual copying and announces the failure.
                page.evaluate('()=>{navigator.clipboard.writeText=async()=>{throw new Error("denied")};}')
                page.locator('.rt-copy').first.click()
                assert page.locator('.code-fold').first.get_attribute('open') is not None
                page.wait_for_function('()=>[...document.querySelectorAll("[role=status]")].some(e=>/Copy failed|無法複製/.test(e.textContent))')
                assert not errors, errors
                results.append({'lang': lang, 'theme': theme, 'width': width, 'codeBlocks': count})
                page.close()
        page = browser.new_page(java_script_enabled=False)
        page.goto(BASE + path)
        first = page.locator('.code-fold').first
        assert first.get_attribute('open') is None and not first.locator('pre').is_visible()
        first.locator('summary').click()
        assert first.locator('pre').is_visible()
        first.locator('summary').click()
        page.emulate_media(media='print')
        assert first.locator('pre').is_visible()
        page.close()
    # Retain pending/success feedback coverage from the previous floating-tools suite.
    page = browser.new_page()
    page.goto(BASE + '/en/learn/deep-learning/image-classification/')
    page.evaluate('()=>{window.copyCalls=0;navigator.clipboard.writeText=()=>{window.copyCalls++;return new Promise(r=>window.finishCopy=r)}}')
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
    # Search must still index text inside native closed disclosures.
    page = browser.new_page()
    page.goto(BASE + '/en/learn/deep-learning/image-classification/')
    matches = page.evaluate('''async()=>{
      const pf=await import('/pagefind/pagefind.js');
      const r=await pf.search('torch.manual_seed');
      return (await Promise.all(r.results.map(x=>x.data()))).map(x=>({url:x.url, hasCode:x.content.includes('F.cross_entropy(logits')}));
    }''')
    assert any('/image-classification/' in x['url'] and x['hasCode'] for x in matches), matches
    page.close()
    browser.close()
(ROOT / 'docs/worklog/code-folding-validation.json').write_text(json.dumps({
    'staticPagesWithCode': static_pages, 'staticCodeBlocks': static_blocks, 'layouts': results,
    'checks': ['default closed', 'copy while closed', 'source text unchanged', 'keyboard', 'touch targets',
               'bulk controls', 'print restoration', 'direct bookmarks', 'clipboard failure fallback',
               'native no-JS disclosure and print', 'Pagefind code search'],
}, indent=2) + '\n')
print(f'{static_blocks} blocks on {static_pages} pages start folded; {len(results)} layouts passed; native no-JS disclosure/print and search indexing passed')
