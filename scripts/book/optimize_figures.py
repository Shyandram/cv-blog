"""Shrink the figures of an exported book so the PDF stays small enough to host (Cloudflare Pages: 25 MiB per file).

Run by scripts/book/publish.mjs on <book-output>/assets/figures, in place, after the export has copied the originals
there (the originals in src/assets are never touched). Needs Pillow.

  * Width is capped at MAX_WIDTH pixels (6.5 inch text width at about 123 dpi).
  * Figures with few colours (plots, diagrams) stay PNG as 256-colour palette images, with no dithering.
  * Figures with many colours (photographs, gradients, image grids) become JPEG (quality JPEG_QUALITY).
  * Anything with real transparency stays a palette PNG.
Converted figures change extension, so a JSON map {old name: new name} is written for the LaTeX sources to be rewritten.

Usage: python optimize_figures.py <figures-dir> <map.json>
"""
import json
import sys
from pathlib import Path

from PIL import Image

MAX_WIDTH = 800
JPEG_QUALITY = 70
PHOTO_COLOURS = 1500


def optimize(path: Path) -> tuple[int, int, Path]:
    before = path.stat().st_size
    im = Image.open(path)
    has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    im = im.convert("RGBA" if has_alpha else "RGB")
    if has_alpha and im.getchannel("A").getextrema() == (255, 255):  # alpha present but fully opaque
        im, has_alpha = im.convert("RGB"), False
    if im.width > MAX_WIDTH:
        im = im.resize((MAX_WIDTH, round(im.height * MAX_WIDTH / im.width)), Image.LANCZOS)
    photographic = im.getcolors(maxcolors=PHOTO_COLOURS + 1) is None
    if photographic and not has_alpha:
        target = path.with_suffix(".jpg")
        im.save(target, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
        if target != path:
            path.unlink()
        return before, target.stat().st_size, target
    q = im.quantize(colors=256, method=Image.Quantize.FASTOCTREE if has_alpha else Image.Quantize.MEDIANCUT)
    q.save(path, "PNG", optimize=True)
    return before, path.stat().st_size, path


def main(folder: str, map_file: str) -> None:
    root = Path(folder)
    files = sorted(root.rglob("*.png"))
    if not files:
        raise SystemExit(f"No PNG files under {folder}")
    before = after = 0
    renamed: dict[str, str] = {}
    for f in files:
        b, a, target = optimize(f)
        before += b
        after += a
        if target != f:
            renamed[f.relative_to(root.parent).as_posix()] = target.relative_to(root.parent).as_posix()
    Path(map_file).write_text(json.dumps(renamed, indent=1))
    print(f"Optimized {len(files)} figures ({len(renamed)} to JPEG): {before / 2**20:.1f} MiB -> {after / 2**20:.1f} MiB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
