"""Verify renamed/moved section bookmarks and explicit redirect fragments in a real browser."""
from pathlib import Path
import json
from urllib.parse import urlsplit,unquote
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:4322'
aliases=json.loads(Path('src/content/legacy-chapter-fragments.json').read_text())
cases=[(path+'#'+old,target) for path,items in aliases.items() for old,target in items.items()]
cases.extend([
 ('/en/learn/deep-learning/dl-04a-super-resolution/#post-upsampling-and-sub-pixel-convolution','/en/learn/deep-learning/super-resolution/#decision-2-how-to-upsample-transposed-or-sub-pixel-convolution'),
 ('/en/learn/deep-learning/super-resolution/papers/chen-et-al-real-world-single-image-super-resolution-a-brief-review-information-fusion-2022/#challenge','/en/learn/deep-learning/blind-real-world-super-resolution/#survey-papers'),
 ('/en/learn/deep-learning/dl-04a-super-resolution/notes/three-kernel-aware-methods/','/en/learn/deep-learning/blind-real-world-super-resolution/#explicit-degradation-modelling'),
])
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':375,'height':900})
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 for old,target in cases:
  page.goto(BASE+old)
  expected=urlsplit(BASE+target)
  page.wait_for_url(lambda u: unquote(urlsplit(u).path)==unquote(expected.path) and (not expected.fragment or unquote(urlsplit(u).fragment)==unquote(expected.fragment)),timeout=10000)
  fragment=unquote(urlsplit(page.url).fragment)
  if fragment:
   page.wait_for_function('(id)=>{const e=document.getElementById(id);return !!e&&!e.closest("details:not([open])")}',arg=fragment)
  print('PASS',old,flush=True)
 assert not errors,errors
 browser.close()
print(len(cases),'moved topic bookmarks passed')
