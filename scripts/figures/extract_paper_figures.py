"""Extract the attributed paper figures explicitly requested by the user.

The manifest pins the PDF bytes, page, figure number and crop. Full papers stay
in a local research cache; only the selected architecture/flow excerpt is saved.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import fitz
from PIL import Image

fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/cvblog-paper-cache"))
parser.add_argument("--only", nargs="*")
args = parser.parse_args()
args.cache_dir.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ROOT / "docs/paper-figure-sources.json").read_text())
for item in manifest:
    if args.only and item["key"] not in args.only:
        continue
    if item["kind"] == "original-author-figure":
        original = args.cache_dir / (item["key"] + ".source")
        if not original.exists():
            with urllib.request.urlopen(item["imageUrl"], timeout=120) as response:
                original.write_bytes(response.read())
        if hashlib.sha256(original.read_bytes()).hexdigest() != item["sourceSha256"]:
            raise ValueError(f"Source image changed for {item['key']}: verify its version")
        source = Image.open(original)
        source.load()
        scale = min(1, 1600 / source.width)
        dest = ROOT / item["output"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        while True:
            size = (round(source.width * scale), round(source.height * scale))
            result = source.resize(size, Image.Resampling.LANCZOS) if size != source.size else source
            result.save(dest, format="PNG", optimize=True)
            if dest.stat().st_size < 1_000_000:
                break
            scale *= .9
            if size[0] < 800:
                raise ValueError(f"Figure size limit exceeded: {dest}")
        print(f"{item['key']}: official source image, {result.width}x{result.height}")
        continue
    pdf = args.cache_dir / (item["key"] + ".pdf")
    if not pdf.exists():
        with urllib.request.urlopen(item["sourceUrl"], timeout=120) as response:
            pdf.write_bytes(response.read())
    actual = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if actual != item["pdfSha256"]:
        raise ValueError(f"Source PDF changed for {item['key']}: verify its version and crop")
    doc = fitz.open(pdf)
    page = doc[item["page"] - 1]
    clip = fitz.Rect(item["crop"])
    if not page.rect.contains(clip):
        raise ValueError(f"Crop exceeds the page: {item['key']}")
    scale = min(4, 1590 / clip.width)
    dest = ROOT / item["output"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    while True:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        pix.pil_save(dest, optimize=True)
        if pix.width <= 1600 and dest.stat().st_size < 1_000_000:
            break
        scale *= .9
        if scale < 1:
            raise ValueError(f"Figure size limit exceeded: {dest}")
    print(f"{item['key']}: figure {item['figure']}, PDF page {item['page']}, {pix.width}x{pix.height}")
