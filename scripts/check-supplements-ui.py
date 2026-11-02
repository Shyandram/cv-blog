"""Verify CNN live controls and the FusionGAN/SeAFusion examples at both viewport sizes."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
index=json.loads((root/'docs/content-index.json').read_text())
fusion=next(t for t in index['tutorials'] if t['slug']=='deep-learning/dl-05b-image-fusion')
items=[s for s in fusion['sections'] if s['depth']==3 and s['text'].startswith(('FusionGAN:', 'SeAFusion:'))]
assert len(items)==2, 'Expected FusionGAN and SeAFusion guides'
results=[]
with sync_playwright() as p:
 browser=p.chromium.launch(args=['--no-sandbox'])
 for lang in ['en','zh-tw']:
  for width in [1440,375]:
   page=browser.new_page(viewport={'width':width,'height':900});errors=[]
   page.on('pageerror',lambda e:errors.append(str(e)))
   route=f'/{lang}/learn/ml/convolutional-networks/'
   assert page.goto('http://127.0.0.1:4322'+route).status==200
   lab=page.locator('image-lab[data-lab="receptive-field"]')
   assert lab.count()==1
   lab.scroll_into_view_if_needed()
   control=lab.locator('input[type=range]').first
   page.wait_for_function('()=>document.querySelector(\'image-lab[data-lab="receptive-field"] input[data-key]\').disabled === false')
   before=lab.locator('[data-summary]').inner_text()
   control.evaluate('(el)=>{el.value=el.max;el.dispatchEvent(new Event("input",{bubbles:true}))}')
   page.wait_for_function('(before)=>document.querySelector(\'image-lab[data-lab="receptive-field"] [data-summary]\').textContent!==before',arg=before)
   after=lab.locator('[data-summary]').inner_text()
   assert before!=after,'Receptive-field control did not update'
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth'), (route,width)
   results.append({'route':route,'width':width,'lab':'receptive-field'})
   for item in items:
    route=item['url'] if lang=='en' else item['urlZh']
    assert page.goto('http://127.0.0.1:4322'+route).status==200
    if item['text'].startswith('FusionGAN:'):
     assert page.locator('.prose table').count()==1
     table=page.locator('.prose table').inner_text()
     assert all(value in table for value in ['0.254','0.0716','0.0350'])
    else:
     assert page.locator('.prose pre').count()==2
     assert 'intensity=0.0000 gradient=0.2625 total=2.6250' in page.locator('.prose').inner_text()
    page.locator('.section-controls button').nth(1).click()
    assert page.locator('.chapter-section[open]').count()==0
    page.locator('.section-controls button').first.click()
    assert page.locator('.chapter-section:not([open])').count()==0
    assert not page.evaluate('document.documentElement.scrollWidth>innerWidth'), (route,width)
    assert not errors,errors
    page.screenshot(path=str(root/'artifacts'/f'{lang}-{width}-{item["text"].split(":")[0].lower()}-supplement.png'))
    results.append({'route':route,'width':width,'paper':item['text']})
   page.close()
 browser.close()
(root/'artifacts'/'supplements-ui.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
print(len(results),'supplement/CNN layouts, live controls and folding checks passed')
