"""Reading layout regression checks against the locally served production build."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:4322'
root=Path(__file__).resolve().parents[1]
guides=json.loads((root/'docs/paper-guide-coverage.json').read_text())['guides']
paper=next(g for g in guides if g['title'].startswith('Restormer:'))
long_reference=next(g for g in guides if 'Large Minibatch SGD' in g['title'])
results=[]
with sync_playwright() as p:
 b=p.chromium.launch(args=['--no-sandbox'])
 for lang in ['en','zh-tw']:
  for theme in ['light','dark']:
   for width,height in [(320,800),(375,900),(768,1024),(1024,900),(1440,1000),(844,390)]:
    page=b.new_page(viewport={'width':width,'height':height},color_scheme=theme,reduced_motion='reduce')
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    url_key='url' if lang=='en' else 'urlZh'
    for kind,path in [('chapter',f'/{lang}/learn/deep-learning/image-classification/'),('paper',paper[url_key]),('long-reference',long_reference[url_key])]:
     assert page.goto(BASE+path).status==200
     page.evaluate('document.fonts.ready')
     assert not page.evaluate('document.documentElement.scrollWidth>innerWidth'),(path,width,theme)
     size=page.locator('.prose').evaluate('(e)=>parseFloat(getComputedStyle(e).fontSize)')
     assert size >= (18 if width<=620 else 19),size
     assert page.evaluate('getComputedStyle(document.documentElement).scrollBehavior')!='smooth'
     for e in page.locator('.chapter-tabs a, .section-controls button').all():
      assert e.bounding_box()['height']>=44
     if width>=1200:
      for toggle in page.locator('.toc .toc-more').all():
       box=toggle.bounding_box();link=toggle.locator('..').locator(':scope > a').bounding_box()
       assert box['width']>=44 and box['height']>=44
       assert box['x']>=link['x']+link['width'] and abs(box['y']-link['y'])<1
     copy=page.locator('.rt-copy').first
     if copy.count():
      copy.scroll_into_view_if_needed();assert copy.bounding_box()['height']>=44
      pre=copy.locator('..').locator('pre');assert copy.bounding_box()['y']+copy.bounding_box()['height'] <= pre.bounding_box()['y']+1
     page.locator('.section-controls button').nth(1).click()
     assert page.locator('.chapter-section[open]').count()==0
     page.locator('.section-controls button').first.click()
     assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
     if width<1200:
      summary=page.locator('.rt-mnav > summary');summary.click()
      assert page.locator('.rt-mnav').get_attribute('open') is not None
      toggle=page.locator('.rt-mnav .ct-toggle:visible').first
      if toggle.count():
       box=toggle.bounding_box();assert box['width']>=44 and box['height']>=44,box
      summary.focus();summary.press('Escape')
      assert page.locator('.rt-mnav').get_attribute('open') is None
     # Validate actual resolved semantic text colors in both themes.
     colors=page.evaluate('()=>{const s=getComputedStyle(document.documentElement);return ["--ink","--muted","--bg","--surface"].map(k=>s.getPropertyValue(k).trim())}')
     def luminance(h):
      if len(h)==4: h='#'+''.join(c*2 for c in h[1:])
      rgb=[int(h[i:i+2],16)/255 for i in [1,3,5]]
      rgb=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
      return sum(v*w for v,w in zip(rgb,[.2126,.7152,.0722]))
     for foreground in colors[:2]:
      for background in colors[2:]:
       a,c=sorted([luminance(foreground),luminance(background)])
       assert (c+.05)/(a+.05)>=4.5,(foreground,background)
     page.emulate_media(media='print')
     print_ratio=page.locator('.prose').evaluate('(e)=>{const s=getComputedStyle(e);return parseFloat(s.lineHeight)/parseFloat(s.fontSize)}')
     assert abs(print_ratio-1.5)<.01,print_ratio
     page.emulate_media(media='screen')
     if width in [375,1440]:
      page.locator('.chapter-section').first.scroll_into_view_if_needed()
      page.screenshot(path=str(root/'artifacts'/f'readability-{lang}-{theme}-{width}-{kind}.png'))
     assert not errors,errors
     results.append({'lang':lang,'theme':theme,'width':width,'height':height,'kind':kind,'bodyPixels':size})
    page.close()
 b.close()
(root/'docs/worklog/readability-validation.json').write_text(json.dumps(results,indent=2)+'\n')
print(len(results),'layouts passed: typography, contrast, touch controls, code toolbar, folding, reduced motion and mobile navigation')
