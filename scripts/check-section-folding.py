"""Check folding, keyboard controls, anchor navigation and printing on both tutorial languages."""
from pathlib import Path
from reader_test_helpers import set_reader_group
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:4322'
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox'])
 for lang in ['en','zh-tw']:
  for width in [1440,375]:
   page=b.new_page(viewport={'width':width,'height':900})
   errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   path=f'/{lang}/learn/deep-learning/image-classification/'
   assert page.goto(BASE+path).status==200
   page.wait_for_selector('.chapter-section')
   sections=page.locator('.chapter-section');count=sections.count()
   assert count>10,count
   assert page.locator('.chapter-section:not([open])').count()==0
   first=sections.first;summary=first.locator(':scope > summary')
   summary.click();assert first.get_attribute('open') is None
   summary.focus();summary.press('Enter');assert first.get_attribute('open') is not None
   set_reader_group(page, 'sections', False)
   assert page.locator('.chapter-section[open]').count()==0
   target=page.locator('.section-body h3[id]').first.get_attribute('id')
   page.evaluate('(id)=>{location.hash=id}',target)
   page.wait_for_function('(id)=>document.getElementById(id).closest("details").open',arg=target)
   assert page.locator('.notes-teaser').first.is_visible()
   set_reader_group(page, 'sections', False)
   page.evaluate('dispatchEvent(new Event("beforeprint"))')
   assert page.locator('.chapter-section[open]').count()==count
   page.evaluate('dispatchEvent(new Event("afterprint"))')
   assert page.locator('.chapter-section[open]').count()==0
   set_reader_group(page, 'sections', True)
   assert page.locator('.chapter-section[open]').count()==count
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
   assert not errors,errors
   set_reader_group(page, 'sections', False)
   page.screenshot(path=f'artifacts/{lang}-{width}-folded.png')
   page.close()
  page=b.new_page(java_script_enabled=False)
  page.goto(BASE+f'/{lang}/learn/deep-learning/image-classification/')
  assert page.locator('.prose h2').count()>10
  assert page.locator('.reader-console').count()==0
  page.close()
 # Paper items retain h4 headings in the source but have the same folding controls.
 paper_paths={
  'en':'/en/learn/deep-learning/building-blocks/papers/repvgg-training-with-branches-deploying-a-plain-network/',
  'zh-tw':'/zh-tw/learn/deep-learning/building-blocks/papers/repvgg訓練用分支部署用直線網路/',
 }
 for lang,path in paper_paths.items():
  for width in [1440,375]:
   page=b.new_page(viewport={'width':width,'height':900})
   errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   assert page.goto(BASE+path).status==200
   page.wait_for_selector('.chapter-section > summary > h4')
   sections=page.locator('.chapter-section');count=sections.count()
   assert page.locator('.chapter-section > summary > h4').count()==6
   set_reader_group(page, 'sections', False)
   assert page.locator('.chapter-section[open]').count()==0
   heading=page.locator('.chapter-section > summary > h4').nth(1)
   target=heading.get_attribute('id')
   page.evaluate('(id)=>{location.hash=id}',target)
   page.wait_for_function('(id)=>document.getElementById(id).closest("details").open',arg=target)
   summary=heading.locator('..');summary.focus();summary.press('Enter')
   assert summary.locator('..').get_attribute('open') is None
   page.evaluate('dispatchEvent(new Event("beforeprint"))')
   assert page.locator('.chapter-section[open]').count()==count
   page.evaluate('dispatchEvent(new Event("afterprint"))')
   assert page.locator('.chapter-section[open]').count()==0
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
   assert not errors,errors
   page.screenshot(path=f'artifacts/{lang}-{width}-paper-folded.png')
   print('PASS paper folding',lang,width,flush=True)
   page.close()
  page=b.new_page(java_script_enabled=False)
  assert page.goto(BASE+path).status==200
  assert page.locator('.prose h4').count()==6
  assert page.locator('.reader-console').count()==0
  page.close()
 b.close()
print('8 folding layouts, keyboard, deep links, print restore and no-JS reading passed')
