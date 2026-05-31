#!/usr/bin/env python3
"""Stamp a discreet version footer onto every page of the report PDF.

Usage:
  python3 stamp_version.py INPUT.pdf OUTPUT.pdf "iter4 · 2026-05-26 · commit dd961f3"

The stamp is rendered in 7 pt Helvetica grey, bottom-right corner, inside the
page margin so it does not collide with IEEEtran's own footer.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas


def make_stamp(width_pt: float, height_pt: float, text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(width_pt, height_pt))
    grey = Color(0.45, 0.45, 0.45)
    c.setFillColor(grey)
    c.setFont("Helvetica", 7)
    # Bottom-right, with a comfortable margin
    margin_pt = 22
    text_width = c.stringWidth(text, "Helvetica", 7)
    c.drawString(width_pt - margin_pt - text_width, margin_pt - 8, text)
    c.showPage()
    c.save()
    return buf.getvalue()


def stamp_pdf(in_path: Path, out_path: Path, text: str) -> None:
    reader = PdfReader(str(in_path))
    writer = PdfWriter()
    for page in reader.pages:
        box: RectangleObject = page.mediabox
        w, h = float(box.width), float(box.height)
        stamp_bytes = make_stamp(w, h, text)
        stamp_reader = PdfReader(io.BytesIO(stamp_bytes))
        page.merge_page(stamp_reader.pages[0])
        writer.add_page(page)
    with out_path.open("wb") as f:
        writer.write(f)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: stamp_version.py INPUT.pdf OUTPUT.pdf 'STAMP TEXT'", file=sys.stderr)
        return 1
    in_p = Path(sys.argv[1]).resolve()
    out_p = Path(sys.argv[2]).resolve()
    text = sys.argv[3]
    stamp_pdf(in_p, out_p, text)
    print(f"stamped: {in_p.name} -> {out_p}  ({text!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
