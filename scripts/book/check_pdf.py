"""Verify generated book structure, embedded fonts, links, and representative pages.

Requires PyMuPDF and Pillow. Run after both PDFs have compiled.
"""
import json
import re
import sys
from pathlib import Path

import fitz
from PIL import Image, ImageOps, ImageDraw

root = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'book-output'
manifest = json.loads((out / 'manifest.json').read_text())
previews = out / 'previews'
previews.mkdir(exist_ok=True)
report = {}
normalize = lambda s: re.sub(r'\s+', '', s)

for lang, edition in manifest['editions'].items():
    stem = f'pixels-perception-{lang}'
    source = (out / f'{stem}.tex').read_text()
    assert source.count('\\bookchapter{') == len(edition['chapters'])
    assert source.count('\\begin{Verbatim}') == edition['codeBlocks']
    assert source.count('\\bookdisplay{') == edition['displayEquations']
    assert source.count('\\bookimage{') == edition['figures'] + edition['comparisons'] * 2
    log = (out / f'{stem}.log').read_text(errors='replace')
    assert 'Missing character:' not in log, f'{lang}: missing font glyphs'
    assert not re.search(r'^! ', log, re.M), f'{lang}: LaTeX error'
    assert 'There were undefined references' not in log, f'{lang}: unresolved references'
    assert 'Label(s) may have changed' not in log, f'{lang}: rerun required'
    doc = fitz.open(out / f'{stem}.pdf')
    toc = doc.get_toc()
    titles = {normalize(item[1]): item[2] for item in toc}
    for chapter in edition['chapters']:
        assert normalize(chapter['title']) in titles, f'Missing chapter: {chapter["title"]}'
    fonts, links, images = set(), 0, 0
    clipped = []
    for number, page in enumerate(doc):
        fonts.update(font[0] for font in page.get_fonts())
        images += len(page.get_images())
        for link in page.get_links():
            links += 1
            if link.get('kind') == fitz.LINK_GOTO:
                assert 0 <= link['page'] < len(doc), (number, link)
        # Reject text physically outside the page, excluding insignificant rounding.
        for block in page.get_text('blocks'):
            if block[6] == 0 and (block[0] < -1 or block[2] > page.rect.width + 1):
                clipped.append(number + 1)
    assert not clipped, f'{lang}: clipped text on pages {sorted(set(clipped))}'
    assert images >= edition['figures'] + edition['comparisons'] * 2
    for xref in fonts:
        if xref:
            assert doc.extract_font(xref)[3], f'{lang}: font {xref} is not embedded'
    chosen = {0, 1, 2, len(doc) - 1}
    part_pages = [row[2] - 1 for row in toc if row[0] == 1]
    for page in part_pages:
        chosen.add(page)
        chosen.add(min(page + 1, len(doc) - 1))
        chosen.add(min(page + 8, len(doc) - 1))
    chosen = sorted(chosen)
    thumbs = []
    for number in chosen:
        page = doc[number]
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
        name = f'{lang}-page-{number + 1:04d}.png'
        pix.save(previews / name)
        im = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        im.thumbnail((288, 370))
        tile = Image.new('RGB', (308, 404), '#e8edf2')
        tile.paste(im, ((308 - im.width) // 2, 8))
        ImageDraw.Draw(tile).text((12, 385), f'{lang} / PDF page {number + 1}', fill='#162638')
        thumbs.append(tile)
    for index in range(0, len(thumbs), 12):
        group = thumbs[index:index + 12]
        sheet = Image.new('RGB', (308 * 4, 404 * ((len(group) + 3) // 4)), 'white')
        for j, thumb in enumerate(group):
            sheet.paste(thumb, ((j % 4) * 308, (j // 4) * 404))
        sheet.save(previews / f'{lang}-contact-{index // 12 + 1}.png')
    overflows = [float(n) for n in re.findall(r'Overfull \\hbox \(([0-9.]+)pt', log)]
    report[lang] = {
        'pages': len(doc), 'tutorials': edition['tutorials'], 'posts': edition['posts'],
        'figure_images': images, 'code_blocks': edition['codeBlocks'],
        'display_equations': edition['displayEquations'], 'bookmarks': len(toc),
        'links': links, 'embedded_fonts': len(fonts), 'bytes': (out / f'{stem}.pdf').stat().st_size,
        'max_overfull_points': max(overflows, default=0), 'preview_pages': [i + 1 for i in chosen],
    }
    print(lang, json.dumps(report[lang]), flush=True)
(out / 'validation.json').write_text(json.dumps(report, indent=2))
