"""Build the static blog from the Markdown posts in ../blog into ../_site.

    python3 site/build.py          -> build into _site/
    python3 site/build.py --serve  -> build, then preview at http://localhost:8000

Adding a post: write blog/NN-some-slug.md with front matter (number, title, part,
date, chapters, summary), make sure blog/series.json lists the number, and rebuild.
On GitHub, pushing to main rebuilds and deploys automatically (.github/workflows/deploy.yml).
"""
import hashlib
import html
import http.server
import json
import re
import shutil
import sys
from datetime import date
from email.utils import format_datetime
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

import markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename

ROOT = Path(__file__).resolve().parent.parent
BLOG = ROOT / "blog"
SITE = ROOT / "site"
OUT = ROOT / "_site"

FENCE_RE = re.compile(r"^```.*?^```[ \t]*$", re.M | re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\$)([^\n$]+?)(?<!\\)\$")
MERMAID_RE = re.compile(r"^```mermaid[ \t]*\n(.*?)^```[ \t]*$", re.M | re.S)
SECTION_RE = re.compile(r'<h2 id="([^"]+)">(\d+)\.\s+')


# ---------------------------------------------------------------- markdown --

def parse_front_matter(text):
    meta = {}
    if text.startswith("---\n"):
        end = text.index("\n---", 4)
        for line in text[4:end].splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip().strip('"')
        text = text[end + 4:]
    return meta, text


def strip_title_block(text):
    """Drop the '# Title' line and the italic byline under it; the page template renders its own header."""
    text = text.lstrip()
    text = re.sub(r"\A# .*\n+", "", text)
    text = re.sub(r"\A\*[^\n]*\*[^\n]*\n+", "", text)
    text = re.sub(r"\A---\n+", "", text)
    return text


def protect_math(text):
    """Pull math out before Markdown can mangle `_` and `*`, skipping code spans and fences."""
    store = []

    def keep(kind, body):
        store.append((kind, body.strip()))
        return f"MATHTOKEN{len(store) - 1}X"

    def in_prose(segment):
        pieces = INLINE_CODE_RE.split(segment)
        codes = INLINE_CODE_RE.findall(segment)
        out = []
        for i, piece in enumerate(pieces):
            piece = DISPLAY_MATH_RE.sub(lambda m: "\n\n" + keep("display", m.group(1)) + "\n\n", piece)
            piece = INLINE_MATH_RE.sub(lambda m: keep("inline", m.group(1)), piece)
            piece = piece.replace(r"\$", "$")
            out.append(piece)
            if i < len(codes):
                out.append(codes[i])
        return "".join(out)

    result, last = [], 0
    for fence in FENCE_RE.finditer(text):
        result.append(in_prose(text[last:fence.start()]))
        result.append(fence.group(0))
        last = fence.end()
    result.append(in_prose(text[last:]))
    return "".join(result), store


def restore_math(body, store):
    def render(i):
        kind, tex = store[i]
        tex = html.escape(tex, quote=False)
        if kind == "display":
            return f'<div class="math-display">\\[{tex}\\]</div>'
        return f'<span class="math-inline">\\({tex}\\)</span>'

    body = re.sub(r"<p>MATHTOKEN(\d+)X</p>", lambda m: render(int(m.group(1))), body)
    return re.sub(r"MATHTOKEN(\d+)X", lambda m: render(int(m.group(1))), body)


def render_post(path):
    meta, text = parse_front_matter(path.read_text(encoding="utf-8"))
    text = strip_title_block(text)

    mermaid_blocks = []

    def keep_mermaid(m):
        mermaid_blocks.append(m.group(1))
        return f"\n\nMERMAIDTOKEN{len(mermaid_blocks) - 1}X\n\n"

    text = MERMAID_RE.sub(keep_mermaid, text)
    text, math = protect_math(text)

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "codehilite", "sane_lists"],
        extension_configs={
            "codehilite": {"css_class": "hl", "guess_lang": False},
            "toc": {"toc_depth": "2-3"},
        },
    )
    body = md.convert(text)
    body = restore_math(body, math)
    body = re.sub(
        r"<p>MERMAIDTOKEN(\d+)X</p>",
        lambda m: f'<figure class="diagram"><pre class="mermaid">{html.escape(mermaid_blocks[int(m.group(1))])}</pre></figure>',
        body,
    )
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    body = body.replace("<hr />", "")  # sections are separated by the heading styles instead
    body = SECTION_RE.sub(lambda m: f'<h2 id="{m.group(1)}"><span class="sec">Section {int(m.group(2)):02d}</span>', body)

    toc = [
        {"id": t["id"], "title": re.sub(r"^\d+\.\s*", "", html.unescape(re.sub(r"MATHTOKEN\d+X", "", t["name"])))}
        for t in md.toc_tokens
    ]
    words = len(re.sub(r"<[^>]+>", " ", body).split())
    post_date = date.fromisoformat(meta["date"]) if meta.get("date") else None
    return {
        "slug": path.stem,
        "number": meta.get("number", path.stem[:2]),
        "title": meta.get("title", path.stem),
        "part": int(meta.get("part", 0) or 0),
        "date": post_date.isoformat() if post_date else "",
        "date_human": f"{post_date:%B} {post_date.day}, {post_date.year}" if post_date else "",
        "date_obj": post_date,
        "chapters": meta.get("chapters", ""),
        "summary": meta.get("summary", ""),
        "minutes": max(1, round(words / 220)),
        "toc": toc,
        "html": body,
        "has_mermaid": bool(mermaid_blocks),
    }


# ------------------------------------------------------------------- build --

def write(rel_path, content):
    target = OUT / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def build():
    series = json.loads((BLOG / "series.json").read_text(encoding="utf-8"))
    site = series["site"]
    site["url"] = site["url"].rstrip("/")
    base_path = urlparse(site["url"]).path.rstrip("/") + "/"

    posts = [render_post(p) for p in sorted(BLOG.glob("[0-9][0-9]-*.md"))]
    by_number = {p["number"]: p for p in posts}
    parts = [
        {**part, "posts": [{**item, "post": by_number.get(item["number"])} for item in part["posts"]]}
        for part in series["parts"]
    ]
    planned = [item for part in series["parts"] for item in part["posts"]]

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SITE / "assets", OUT / "assets")
    (OUT / ".nojekyll").write_text("")

    version = hashlib.sha1(
        b"".join(p.read_bytes() for p in sorted((SITE / "assets").iterdir()))
    ).hexdigest()[:8]
    env = Environment(loader=FileSystemLoader(SITE / "templates"), autoescape=select_autoescape(["html"]))
    common = {"site": site, "version": version, "description": site["description"]}

    def page(template, rel_path, root, **ctx):
        path = rel_path[:-len("index.html")] if rel_path.endswith("index.html") else rel_path
        write(rel_path, env.get_template(template).render(**{**common, **ctx}, root=root, path=path))

    page("home.html", "index.html", "", posts=posts, parts=parts, planned_count=len(planned))
    page("about.html", "about/index.html", "../", page_title="About")
    page("404.html", "404.html", base_path, page_title="Not found")

    code_owner = {}
    for i, post in enumerate(posts):
        root = "../../"
        body = re.sub(r'href="code/([^"/]+)\.(\w+)"', lambda m: f'href="{root}code/{m.group(1)}/"', post["html"])
        for m in re.finditer(r'href="code/([^"/]+)"', post["html"]):
            code_owner.setdefault(m.group(1), post)
        next_planned = next((p for p in planned if p["number"] > post["number"]), None)
        page(
            "post.html", f"posts/{post['slug']}/index.html", root,
            page_title=post["title"], description=post["summary"], og_type="article",
            post=post, body=body, has_mermaid=post["has_mermaid"],
            part=next((p for p in series["parts"] if p["part"] == post["part"]), None),
            prev=posts[i - 1] if i > 0 else None,
            next=posts[i + 1] if i + 1 < len(posts) else None,
            next_planned=next_planned,
        )

    formatter = HtmlFormatter(cssclass="hl")
    code_files = sorted(p for p in (BLOG / "code").glob("*") if p.is_file())
    for src in code_files:
        (OUT / "code").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, OUT / "code" / src.name)  # raw file for the download link
        code_html = highlight(src.read_text(encoding="utf-8"), get_lexer_for_filename(src.name), formatter)
        owner = code_owner.get(src.name)
        page(
            "code.html", f"code/{src.stem}/index.html", "../../",
            page_title=src.name, name=src.name, code=code_html, owner=owner,
            description=f"Companion code for post {owner['number']}: {owner['title']}" if owner else site["description"],
        )

    # RSS feed and sitemap
    items = []
    for post in sorted(posts, key=lambda p: p["date"], reverse=True):
        link = f"{site['url']}/posts/{post['slug']}/"
        pub = ""
        if post["date_obj"]:
            d = post["date_obj"]
            pub = f"<pubDate>{format_datetime(datetime(d.year, d.month, d.day, tzinfo=timezone.utc))}</pubDate>"
        items.append(
            f"<item><title>{html.escape(post['number'] + ' · ' + post['title'])}</title><link>{link}</link>"
            f"<guid>{link}</guid>{pub}<description>{html.escape(post['summary'])}</description></item>"
        )
    write("feed.xml", (
        '<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0"><channel>'
        f"<title>{html.escape(site['name'])}</title><link>{site['url']}/</link>"
        f"<description>{html.escape(site['description'])}</description>{''.join(items)}</channel></rss>\n"
    ))
    urls = [f"{site['url']}/", f"{site['url']}/about/"] + [f"{site['url']}/posts/{p['slug']}/" for p in posts]
    write("sitemap.xml", (
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + "".join(f"<url><loc>{u}</loc></url>" for u in urls) + "</urlset>\n"
    ))
    write("robots.txt", f"User-agent: *\nAllow: /\nSitemap: {site['url']}/sitemap.xml\n")

    print(f"Built _site/: {len(posts)} post(s), {len(code_files)} code file(s)")


def serve(port=8000):
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
    print(f"Preview at http://localhost:{port}/  (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    build()
    if "--serve" in sys.argv:
        serve()
