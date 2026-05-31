#!/usr/bin/env python3
"""Convert markdown to publication-quality PDF via weasyprint.

Handles:
- Inline image paths (relative to markdown location)
- Tables, headers, code blocks
- Page numbers, margins
"""
import argparse
import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML, CSS


CSS_STYLE = """
@page {
    size: letter;
    margin: 1in;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: Helvetica, Arial, sans-serif;
        font-size: 9pt;
        color: #666;
    }
    @top-right {
        content: "v18 CV Validation Report";
        font-family: Helvetica, Arial, sans-serif;
        font-size: 9pt;
        color: #999;
    }
}

@page :first {
    @top-right { content: ""; }
}

body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.5;
    color: #222;
    max-width: 100%;
}

h1 {
    font-size: 22pt;
    font-weight: bold;
    margin-top: 24pt;
    margin-bottom: 12pt;
    color: #111;
    page-break-after: avoid;
    border-bottom: 2px solid #333;
    padding-bottom: 6pt;
}

h1:first-of-type {
    margin-top: 0;
    font-size: 26pt;
    text-align: center;
    border-bottom: 3px double #333;
}

h2 {
    font-size: 16pt;
    font-weight: bold;
    margin-top: 18pt;
    margin-bottom: 8pt;
    color: #222;
    page-break-after: avoid;
    border-bottom: 1px solid #ccc;
    padding-bottom: 3pt;
}

h3 {
    font-size: 13pt;
    font-weight: bold;
    margin-top: 12pt;
    margin-bottom: 6pt;
    color: #333;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    font-weight: bold;
    margin-top: 8pt;
    margin-bottom: 4pt;
}

p {
    margin-top: 0;
    margin-bottom: 8pt;
    text-align: left;
}

strong {
    font-weight: bold;
    color: #000;
}

em {
    font-style: italic;
}

code {
    font-family: "Courier New", monospace;
    font-size: 9.5pt;
    background-color: #f4f4f4;
    padding: 1pt 3pt;
    border-radius: 2pt;
}

pre {
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 8pt;
    border-left: 3px solid #888;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background: none;
    padding: 0;
}

table {
    border-collapse: collapse;
    margin: 8pt 0;
    width: 100%;
    page-break-inside: avoid;
    font-size: 9.5pt;
}

th, td {
    border: 1px solid #aaa;
    padding: 4pt 8pt;
    text-align: left;
    vertical-align: top;
}

th {
    background-color: #e8e8e8;
    font-weight: bold;
}

tr:nth-child(even) td {
    background-color: #f7f7f7;
}

img {
    max-width: 100%;
    height: auto;
    margin: 8pt auto;
    display: block;
    page-break-inside: avoid;
}

blockquote {
    margin: 8pt 18pt;
    padding-left: 12pt;
    border-left: 3px solid #888;
    color: #444;
    font-style: italic;
}

ul, ol {
    margin: 4pt 0 8pt 0;
    padding-left: 24pt;
}

li {
    margin-bottom: 3pt;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 18pt 0;
}

/* Page break helpers */
h1 {
    page-break-before: auto;
}
h2 {
    page-break-after: avoid;
}

/* Figures + captions */
img + em {
    display: block;
    text-align: center;
    font-size: 9pt;
    color: #666;
    margin-top: -4pt;
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="markdown file")
    ap.add_argument("output", help="output PDF path")
    args = ap.parse_args()

    md_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    base_dir = md_path.parent

    md_text = md_path.read_text(encoding="utf-8")

    # Convert markdown to HTML with extensions
    md = markdown.Markdown(
        extensions=['extra', 'tables', 'fenced_code', 'codehilite', 'toc'],
        extension_configs={
            'codehilite': {'noclasses': True, 'pygments_style': 'tango'},
        }
    )
    html_body = md.convert(md_text)

    # Wrap in full HTML
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>v18 Validation Report</title>
</head>
<body>
{html_body}
</body>
</html>
"""

    # Generate PDF with weasyprint
    print(f"Generating PDF from {md_path}...", file=sys.stderr)
    HTML(string=full_html, base_url=str(base_dir)).write_pdf(
        str(out_path), stylesheets=[CSS(string=CSS_STYLE)]
    )
    print(f"Wrote: {out_path}", file=sys.stderr)
    print(out_path)


if __name__ == "__main__":
    main()
