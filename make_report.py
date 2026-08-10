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
@page { size: A4; margin: 18mm 16mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt;
       line-height: 1.42; color: #111; max-width: 100%; }
h1 { font-size: 19pt; margin: 0 0 2pt; }
h2 { font-size: 13.5pt; margin: 16pt 0 4pt; border-bottom: 1px solid #bbb;
     padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 11pt 0 3pt; page-break-after: avoid; }
p, li { margin: 0 0 6pt; text-align: justify; }
code { font-family: 'SF Mono', Menlo, monospace; font-size: 9pt;
       background: #f2f2f2; padding: 0 2px; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt;
        font-size: 8.8pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; text-align: right; }
th { background: #eef2f7; font-weight: bold; text-align: center; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; max-height: 78mm; width: auto; height: auto;
      display: block; margin: 4pt auto 2pt; page-break-inside: avoid; }
.caption { display: block; text-align: center; font-size: 8.8pt; color: #444;
           margin: 0 0 10pt; page-break-before: avoid; }
blockquote { margin: 6pt 0 8pt; padding: 4pt 10pt; border-left: 3px solid #9bb;
             background: #f6f9fa; font-size: 9.6pt; }
hr { border: 0; border-top: 1px solid #ccc; margin: 12pt 0; }
.subtitle { color: #555; font-size: 10pt; margin-bottom: 12pt; }
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
