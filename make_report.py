"""Make REPORT.pdf from REPORT.md and count the words.

The brief gives a limit for the number of the words. The script compares the
report with that limit.

The script makes the PDF file with Chrome in the headless mode. Chrome is
already on this machine. It puts the figures, the tables and the page breaks
in the correct positions.

To start the script, use the command:  python make_report.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "REPORT.md")
HTML = os.path.join(HERE, "REPORT.html")
PDF = os.path.join(HERE, "REPORT.pdf")
WORD_LIMIT = 3000

CHROME = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
          "/Applications/Chromium.app/Contents/MacOS/Chromium",
          shutil.which("google-chrome") or "",
          shutil.which("chromium") or "")

CSS = """
/* Exact files exposed by the three supplied OnlineWebFonts stylesheets. */
@font-face {
    font-family: "Plus Jakarta Sans";
    src: url("report_assets/fonts/plus-jakarta-sans-regular.woff2") format("woff2");
    font-weight: normal;
    font-style: normal;
    font-display: block;
}
@font-face {
    font-family: "Plus Jakarta Sans SemiBold";
    src: url("report_assets/fonts/plus-jakarta-sans-semibold.woff2") format("woff2");
    font-weight: normal;
    font-style: normal;
    font-display: block;
}
@font-face {
    font-family: "Plus Jakarta Sans ExtraBold";
    src: url("report_assets/fonts/plus-jakarta-sans-extrabold.woff2") format("woff2");
    font-weight: normal;
    font-style: normal;
    font-display: block;
}

:root {
    --ink: #243247;
    --navy: #14243d;
    --muted: #5c687a;
    --accent: #087f8c;
    --accent-soft: #eaf5f5;
    --line: #cad5df;
    --paper-soft: #f5f8fa;
}

@page {
    size: A4;
    margin: 17mm 15mm 18mm;

    @bottom-right {
        content: counter(page);
        font-family: "Plus Jakarta Sans SemiBold", sans-serif;
        font-size: 7.5pt;
        color: #718096;
    }
}

* { box-sizing: border-box; }
html { font-synthesis: style; }
body {
    max-width: 100%;
    margin: 0;
    font-family: "Plus Jakarta Sans", sans-serif;
    font-size: 9.8pt;
    line-height: 1.48;
    color: var(--ink);
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

h1, h2 {
    font-family: "Plus Jakarta Sans ExtraBold", sans-serif;
    font-weight: normal;
    color: var(--navy);
    page-break-after: avoid;
    break-after: avoid-page;
}
h1 {
    margin: 0 0 10pt;
    font-size: 23pt;
    line-height: 1.12;
    letter-spacing: -0.45pt;
}
h1::after {
    content: "";
    display: block;
    width: 34mm;
    height: 2.5pt;
    margin-top: 8pt;
    border-radius: 999px;
    background: var(--accent);
}
h2 {
    margin: 17pt 0 6pt;
    padding-bottom: 4pt;
    border-bottom: 1.25pt solid var(--line);
    font-size: 14.5pt;
    line-height: 1.2;
    letter-spacing: -0.15pt;
}
h3 {
    margin: 12pt 0 4pt;
    font-family: "Plus Jakarta Sans SemiBold", sans-serif;
    font-size: 11.2pt;
    font-weight: normal;
    line-height: 1.25;
    color: #176b75;
    page-break-after: avoid;
    break-after: avoid-page;
}

h1 + p,
h1 + p + p {
    color: var(--muted);
    font-size: 9.7pt;
    line-height: 1.45;
    text-align: left;
}
h1 + p + p { margin-bottom: 14pt; }

p, li {
    margin: 0 0 5.5pt;
    text-align: justify;
    orphans: 3;
    widows: 3;
}
ol, ul {
    margin: 3pt 0 8pt;
    padding-left: 19pt;
}
li { padding-left: 2pt; }
li::marker { color: var(--accent); }

strong, b {
    font-family: "Plus Jakarta Sans SemiBold", sans-serif;
    font-weight: normal;
    color: var(--navy);
}
em, i {
    font-family: "Plus Jakarta Sans ExtraBold", sans-serif;
    font-weight: normal;
    font-style: italic;
}

code {
    padding: 0.5pt 2.5pt;
    border: 0.5pt solid #d8e1e8;
    border-radius: 2.5pt;
    background: #eef3f6;
    font-family: 'SF Mono', Menlo, monospace;
    font-size: 8.4pt;
    color: #294158;
}

table {
    width: 100%;
    margin: 7pt 0 10pt;
    border: 0.75pt solid var(--line);
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 4pt;
    overflow: hidden;
    font-size: 8.2pt;
    line-height: 1.35;
    page-break-inside: avoid;
    break-inside: avoid-page;
}
thead { display: table-header-group; }
th, td {
    padding: 4pt 5pt;
    border-right: 0.5pt solid #dbe3e9;
    border-bottom: 0.5pt solid #dbe3e9;
    text-align: right;
}
th {
    background: var(--navy);
    font-family: "Plus Jakarta Sans SemiBold", sans-serif;
    font-weight: normal;
    color: #fff;
    text-align: center;
}
tr:nth-child(even) td { background: var(--paper-soft); }
tr:last-child td { border-bottom: 0; }
th:last-child, td:last-child { border-right: 0; }
td:first-child, th:first-child { text-align: left; }

p:has(> img) {
    margin: 7pt 0 0;
    page-break-after: avoid;
    break-after: avoid-page;
}
img {
    display: block;
    max-width: 100%;
    max-height: 76mm;
    width: auto;
    height: auto;
    margin: 0 auto;
    padding: 2pt;
    border: 0.5pt solid #d8e1e8;
    border-radius: 4pt;
    background: #fff;
    box-shadow: 0 1pt 3pt rgba(20, 36, 61, 0.10);
    page-break-inside: avoid;
    break-inside: avoid-page;
}
.caption {
    display: block;
    margin: 2.5pt 8pt 10pt;
    color: var(--muted);
    font-family: "Plus Jakarta Sans ExtraBold", sans-serif;
    font-size: 7.7pt;
    font-weight: normal;
    font-style: italic;
    line-height: 1.35;
    text-align: center;
    page-break-before: avoid;
    break-before: avoid-page;
}

blockquote {
    margin: 7pt 0 9pt;
    padding: 6pt 10pt 6pt 11pt;
    border: 0.5pt solid #d5e5e6;
    border-left: 3pt solid var(--accent);
    border-radius: 0 4pt 4pt 0;
    background: var(--accent-soft);
    color: #33485a;
    font-size: 9.1pt;
    page-break-inside: avoid;
    break-inside: avoid-page;
}
blockquote p { margin: 0; text-align: left; }

a { color: #08717d; text-decoration-color: #8abec3; }
hr {
    height: 1.5pt;
    margin: 15pt 0;
    border: 0;
    background: linear-gradient(to right, var(--accent) 0 34mm, var(--line) 34mm 100%);
}
.subtitle {
    margin-bottom: 12pt;
    font-family: "Plus Jakarta Sans SemiBold", sans-serif;
    font-size: 10pt;
    color: var(--muted);
}
"""


def body_word_count(md_text: str) -> int:
    """Count the words before the heading References.

    The function first removes the markup and the image tags.
    """
    body = re.split(r"^##\s+References\b", md_text, flags=re.M)[0]
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)        # figures
    body = re.sub(r"\{:[^}]*\}", " ", body)                  # attr_list annotations
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)      # html comments
    body = re.sub(r"[|`*_#>-]", " ", body)                   # markup punctuation
    return len(body.split())


def main() -> int:
    if not os.path.exists(SRC):
        print(f"missing {SRC}")
        return 1
    md_text = open(SRC, encoding="utf-8").read()

    n = body_word_count(md_text)
    status = "OK" if n <= WORD_LIMIT else "OVER LIMIT"
    print(f"[words] {n} excluding references (limit {WORD_LIMIT}) -- {status}")

    html = markdown.markdown(md_text, extensions=["tables", "attr_list"])
    open(HTML, "w", encoding="utf-8").write(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{html}</body></html>")

    chrome = next((c for c in CHROME if c and os.path.exists(c)), None)
    if chrome is None:
        print("[pdf] no Chrome/Chromium found; REPORT.html written, "
              "print it to PDF manually")
        return 0 if n <= WORD_LIMIT else 1
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", "file://" + HTML],
                   check=True, capture_output=True)
    print(f"[pdf] wrote {PDF} ({os.path.getsize(PDF)//1024} kB)")
    return 0 if n <= WORD_LIMIT else 1


if __name__ == "__main__":
    sys.exit(main())
