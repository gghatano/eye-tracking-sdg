"""Render docs/report.md -> docs/index.html as a self-contained, styled page.

Usage: uv run --with markdown python scripts/build_html.py
GitHub Pages (Deploy from branch: main, /docs) serves docs/index.html.
"""

from __future__ import annotations

from pathlib import Path

import markdown

DOCS = Path("docs")
SRC = DOCS / "report.md"
OUT = DOCS / "index.html"

CSS = """
:root { --fg:#1f2328; --muted:#656d76; --accent:#2563eb; --border:#d0d7de; --bg:#ffffff; --code:#f6f8fa; }
* { box-sizing: border-box; }
body { margin:0; color:var(--fg); background:var(--bg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,"Hiragino Kaku Gothic ProN","Noto Sans JP",sans-serif;
  line-height:1.65; }
.wrap { max-width: 940px; margin: 0 auto; padding: 48px 24px 96px; }
h1 { font-size: 2.1rem; line-height:1.25; border-bottom:2px solid var(--border); padding-bottom:.4em; }
h2 { font-size: 1.5rem; margin-top: 2.2em; border-bottom:1px solid var(--border); padding-bottom:.3em; }
h3 { font-size: 1.18rem; margin-top: 1.6em; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
img { max-width:100%; height:auto; display:block; margin: 1.2em auto; border:1px solid var(--border); border-radius:8px; }
code { background:var(--code); padding:.15em .4em; border-radius:6px; font-size:.88em;
  font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace; }
pre { background:var(--code); padding:16px; border-radius:8px; overflow:auto; border:1px solid var(--border); }
pre code { background:none; padding:0; }
table { border-collapse: collapse; width:100%; margin:1.2em 0; font-size:.93rem; display:block; overflow-x:auto; }
th, td { border:1px solid var(--border); padding:7px 12px; text-align:left; }
th { background:var(--code); }
tr:nth-child(even) td { background:#fbfcfd; }
blockquote { border-left:4px solid var(--accent); margin:1.2em 0; padding:.4em 1.2em; color:var(--muted); background:#f8fafc; }
.muted { color:var(--muted); }
hr { border:none; border-top:1px solid var(--border); margin:2.4em 0; }
.toc { background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:8px 20px; }
figure { margin: 1.2em 0; }
figcaption { text-align:center; color:var(--muted); font-size:.88rem; margin-top:-.6em; }
"""

HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic Eye-Tracking Data — Report</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
{body}
</div>
</body>
</html>
"""


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
    )
    OUT.write_text(HTML.format(css=CSS, body=body), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
