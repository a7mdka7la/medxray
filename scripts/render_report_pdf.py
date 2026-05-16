"""Render docs/REPORT.md to docs/REPORT.pdf, embedding the screenshots.

Pipeline: markdown -> HTML -> headless Chromium -> PDF (via Playwright).
This handles images and tables much better than markdown-pdf.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown as md_lib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "REPORT.md"
OUT = ROOT / "docs" / "REPORT.pdf"
FIG_DIR = ROOT / "docs"


CSS = """
@page { size: A4; margin: 18mm 18mm 22mm 18mm; }
* { box-sizing: border-box; }
html, body {
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #1f2328;
    margin: 0;
}
h1 { font-size: 20pt; margin: 0 0 6pt 0; }
h2 { font-size: 14pt; margin: 16pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 10pt 0 4pt; page-break-after: avoid; }
p  { margin: 4pt 0; }
hr { border: none; border-top: 2px solid #888; margin: 8pt 0 12pt; }
ul, ol { margin: 4pt 0 4pt 18pt; }
li { margin-bottom: 2pt; }
table { border-collapse: collapse; font-size: 9.5pt; margin: 6pt 0 10pt; width: 100%; }
th, td { border: 1px solid #aaa; padding: 4pt 7pt; text-align: left; vertical-align: top; }
th { background: #eef0f3; }
code, pre { font-family: "Menlo", "SF Mono", monospace; }
code { background: #f3f4f6; padding: 0 3px; border-radius: 2px; font-size: 9.2pt; }
pre  { background: #f6f8fa; padding: 7pt 10pt; border-radius: 4pt; font-size: 8.8pt;
       white-space: pre-wrap; line-height: 1.35; page-break-inside: avoid; }
pre code { background: transparent; padding: 0; }
blockquote { color: #444; border-left: 3px solid #999; padding: 2pt 10pt; margin: 6pt 0;
             background: #f7f7f8; }
img { max-width: 100%; border: 1px solid #d0d7de; margin: 6pt 0; display: block;
      page-break-inside: avoid; }
figure { margin: 8pt 0; page-break-inside: avoid; }
em { color: #444; }
a { color: #0366d6; text-decoration: none; }
header { margin-bottom: 12pt; }
.meta { color: #555; font-size: 10pt; }
"""


def _img_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif"}.get(
        suffix, "application/octet-stream"
    )
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def inline_images(html: str) -> str:
    """Replace relative image src with data URIs so the PDF is self-contained."""

    def _sub(m: re.Match[str]) -> str:
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        p = (FIG_DIR / src).resolve()
        if not p.exists():
            return m.group(0)
        return m.group(0).replace(src, _img_to_data_uri(p))

    return re.sub(r'<img[^>]+src="([^"]+)"', _sub, html)


def build_html() -> str:
    md = SRC.read_text()
    html_body = md_lib.markdown(
        md,
        extensions=["tables", "fenced_code", "codehilite", "attr_list", "sane_lists"],
        extension_configs={"codehilite": {"noclasses": True, "pygments_style": "tango"}},
    )
    html_body = inline_images(html_body)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>MedXray Report</title>"
        f"<style>{CSS}</style></head><body>"
        f"{html_body}"
        "</body></html>"
    )


def main() -> None:
    from playwright.sync_api import sync_playwright

    html = build_html()
    tmp = ROOT / "docs" / "_report.html"
    tmp.write_text(html)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(viewport={"width": 1200, "height": 1600}).new_page()
        page.goto(tmp.as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(OUT),
            format="A4",
            margin={"top": "18mm", "bottom": "22mm", "left": "18mm", "right": "18mm"},
            print_background=True,
        )
        browser.close()
    tmp.unlink(missing_ok=True)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
