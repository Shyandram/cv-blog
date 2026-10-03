"""Import Medium articles into src/content/posts as Markdown.

Usage:
    python scripts/import_medium.py FEED.xml SLUG=MEDIUM_POST_ID [SLUG=ID ...]

FEED.xml is https://medium.com/feed/@<user> (it holds the full HTML of the 10 most
recent posts). Each SLUG=ID pair picks one post by the hex id at the end of its URL,
e.g. llie-intro=9d4d948f3c2e. Images are downloaded next to the post so the site
does not depend on Medium's CDN. Requires: beautifulsoup4, curl.
"""
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parent.parent
POSTS = ROOT / "src" / "content" / "posts"
NS = {"content": "http://purl.org/rss/1.0/modules/content/", "dc": "http://purl.org/dc/elements/1.1/"}
# WSL's DNS proxy can be flaky; DoH avoids it. Set DOH="" to use the system resolver.
DOH = ["--doh-url", "https://1.1.1.1/dns-query"]


def download(url: str, dest_stem: Path) -> str:
    """Download url to dest_stem.<ext>; return the file name."""
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_stem.with_suffix(".part")
    ctype = subprocess.run(
        ["curl", "-sSfL", "-m", "60", *DOH, "-A", "Mozilla/5.0", "-o", str(tmp), "-w", "%{content_type}", url],
        check=True, capture_output=True, text=True,
    ).stdout
    ext = {"image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}.get(ctype.split(";")[0], ".jpg")
    final = dest_stem.with_suffix(ext)
    tmp.replace(final)
    return final.name


def esc(text: str) -> str:
    """Escape characters that Markdown, HTML or remark-math would otherwise interpret."""
    text = text.replace("\\", "\\\\").replace("$", "\\$").replace("<", "&lt;")
    return re.sub(r"([*_`\[\]])", r"\\\1", text)


def inline(node) -> str:
    if isinstance(node, NavigableString):
        return esc(str(node))
    if not isinstance(node, Tag):
        return ""
    inner = "".join(inline(c) for c in node.children)
    name = node.name
    if name == "br":
        return "  \n"
    if name in ("strong", "b"):
        return f"**{inner.strip()}**" if inner.strip() else inner
    if name in ("em", "i"):
        return f"*{inner.strip()}*" if inner.strip() else inner
    if name == "code":
        return f"`{node.get_text()}`"
    if name == "a":
        href = node.get("href", "")
        return f"[{inner}]({href})" if href and inner.strip() else inner
    return inner


class Converter:
    def __init__(self, img_dir: Path, rel_prefix: str):
        self.img_dir, self.rel_prefix, self.n = img_dir, rel_prefix, 0

    def image(self, img: Tag, caption: str) -> str:
        src = img.get("src", "")
        if not src or "/_/stat" in src:
            return ""
        self.n += 1
        name = download(src, self.img_dir / f"{self.n:02d}")
        alt = caption or img.get("alt", "") or f"Figure {self.n}"
        out = f"![{esc(alt)}]({self.rel_prefix}/{name})"
        if caption:
            out += f"\n\n*{esc(caption)}*"
        return out

    def block(self, node) -> str:
        if isinstance(node, NavigableString):
            return esc(str(node)).strip()
        name = node.name
        if name in ("h1", "h2", "h3"):
            return "## " + inline(node).strip()
        if name in ("h4", "h5", "h6"):
            return "### " + inline(node).strip()
        if name == "p":
            return inline(node).strip()
        if name == "blockquote":
            return "\n".join("> " + line for line in inline(node).strip().splitlines())
        if name == "pre":
            code = node.get_text("\n") if node.find("br") else node.get_text()
            return "```\n" + code.rstrip() + "\n```"
        if name in ("ul", "ol"):
            items = []
            for i, li in enumerate(node.find_all("li", recursive=False), 1):
                marker = f"{i}." if name == "ol" else "-"
                items.append(f"{marker} {inline(li).strip()}")
            return "\n".join(items)
        if name == "figure":
            img = node.find("img")
            cap = node.find("figcaption")
            return self.image(img, cap.get_text(" ", strip=True) if cap else "") if img else ""
        if name == "img":
            return self.image(node, "")
        if name == "hr":
            return "---"
        return "\n\n".join(filter(None, (self.block(c) for c in node.children)))


def yaml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main():
    feed, pairs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    items = {}
    for it in ET.parse(feed).iter("item"):
        link = it.findtext("link").split("?")[0]
        items[link.rsplit("-", 1)[-1]] = it
    for slug, pid in pairs:
        it = items.get(pid)
        if it is None:
            print(f"skip {slug}: post {pid} is not in the feed")
            continue
        title = it.findtext("title").strip()
        html = it.find("content:encoded", NS).text
        lang = "zh-tw" if re.search(r"[一-鿿]", title + html[:2000]) else "en"
        soup = BeautifulSoup(html, "html.parser")
        first = soup.find(["h3", "h4"])
        if first and first.get_text(strip=True) == title:
            first.decompose()  # Medium repeats the title as the first heading
        conv = Converter(POSTS / lang / "images" / slug, f"./images/{slug}")
        body = "\n\n".join(filter(None, (conv.block(c) for c in soup.children)))
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        first_p = next((p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 20), title)
        desc = first_p[:110] + ("…" if len(first_p) > 110 else "")
        date = parsedate_to_datetime(it.findtext("pubDate")).date().isoformat()
        tags = [c.text for c in it.findall("category")]
        link = it.findtext("link").split("?")[0]
        front = "\n".join([
            "---",
            f"title: {yaml_str(title)}",
            f"description: {yaml_str(desc)}",
            f"tags: [{', '.join(tags)}]",
            f"published: {date}",
            f"original: {link}",
            "---",
        ])
        out = POSTS / lang / f"{slug}.md"
        out.write_text(front + "\n\n" + body + "\n", encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)} ({conv.n} images)")


if __name__ == "__main__":
    main()
