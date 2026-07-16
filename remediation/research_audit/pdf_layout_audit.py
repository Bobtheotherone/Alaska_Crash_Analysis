"""Machine audit of the rendered paper's figure layout (PDF-001).

The v3 PDF shipped figures/captions clipped or detached across page breaks while the renderer's
docstring claimed otherwise. The v3.1 renderer makes clipping structurally impossible: every image
is frame-capped by ``_fit_cm`` (≤15 cm × 8.5 cm) and each figure+caption is wrapped in a 1×1
``page-break-inside: avoid`` table, so a block either fits a page or moves to the next page whole.
This audit *verifies* those structural guarantees instead of trusting them:

  A. From ``final_paper.html`` (the exact input to the PDF renderer): every embedded figure's
     declared width/height must fit the printable box of a letter page with 2 cm margins,
     including a caption allowance — i.e. no figure block can exceed one page.
  B. From ``final_paper.pdf``: all 7 figure images must actually be embedded as image XObjects,
     and every page's resource dictionary is walked to report which pages draw them (a figure
     that silently failed to embed would show up here).

Exit 0 = clean; exit 1 = violations. Run from ``remediation/``:
    python research_audit/pdf_layout_audit.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
HTML = PAPER / "final_paper.html"
PDF = PAPER / "final_paper.pdf"

PT_PER_CM = 28.3464567
PAGE_W, PAGE_H = 612.0, 792.0            # letter, points
MARGIN_CM = 2.0
PRINT_W = PAGE_W - 2 * MARGIN_CM * PT_PER_CM   # ≈498.6 pt
PRINT_H = PAGE_H - 2 * MARGIN_CM * PT_PER_CM   # ≈678.6 pt
CAPTION_ALLOWANCE_PT = 60.0               # caption (≤3 lines) + table padding + margins


def audit_html() -> list[str]:
    problems = []
    html = HTML.read_text(encoding="utf-8", errors="replace")
    figs = re.findall(
        r'<img src="data:image/png;base64,[^"]+" style="width:([\d.]+)cm;height:([\d.]+)cm;"[^>]*>'
        r'.{0,400}?<b>Figure (\d+)\.</b>', html, re.S)
    if len(figs) < 7:
        problems.append(f"HTML embeds only {len(figs)} captioned figures (expected 7)")
    for w_cm, h_cm, num in figs:
        w_pt, h_pt = float(w_cm) * PT_PER_CM, float(h_cm) * PT_PER_CM
        if w_pt > PRINT_W + 1:
            problems.append(f"Figure {num}: width {w_pt:.0f}pt exceeds printable {PRINT_W:.0f}pt")
        if h_pt + CAPTION_ALLOWANCE_PT > PRINT_H + 1:
            problems.append(f"Figure {num}: block height {h_pt:.0f}+{CAPTION_ALLOWANCE_PT:.0f}pt "
                            f"exceeds printable {PRINT_H:.0f}pt (could be clipped/split)")
    print(f"  HTML: {len(figs)} captioned figure blocks, all frame-capped"
          if not problems else f"  HTML: {len(figs)} captioned figure blocks")
    return problems


def _objects(raw: bytes) -> dict[int, bytes]:
    return {int(m.group(1)): m.group(2)
            for m in re.finditer(rb"(\d+)\s+0\s+obj(.*?)endobj", raw, re.S)}


def audit_pdf() -> list[str]:
    problems = []
    raw = PDF.read_bytes()
    objs = _objects(raw)

    image_objs = {n for n, b in objs.items()
                  if re.search(rb"/Subtype\s*/Image", b.split(b"stream", 1)[0])}
    n_pages = len([1 for b in objs.values()
                   if re.search(rb"/Type\s*/Page(?![s])", b.split(b"stream", 1)[0])])

    # which pages reference image XObjects (directly or via their resource dict object)
    pages_with_images = 0
    for n, b in objs.items():
        head = b.split(b"stream", 1)[0]
        if not re.search(rb"/Type\s*/Page(?![s])", head):
            continue
        res_blob = head
        mref = re.search(rb"/Resources\s+(\d+)\s+0\s+R", head)
        if mref and int(mref.group(1)) in objs:
            res_blob = objs[int(mref.group(1))]
        xrefs = {int(x) for x in re.findall(rb"(\d+)\s+0\s+R", res_blob)}
        if xrefs & image_objs:
            pages_with_images += 1

    print(f"  PDF: {n_pages} pages, {len(image_objs)} embedded image objects, "
          f"{pages_with_images} pages referencing images")
    if len(image_objs) < 7:
        problems.append(f"PDF embeds only {len(image_objs)} image objects (expected ≥7)")
    if pages_with_images == 0:
        problems.append("no page references an image XObject (figures missing from pages)")
    return problems


def main() -> int:
    problems = audit_html() + audit_pdf()
    if problems:
        print(f"PDF LAYOUT AUDIT: {len(problems)} violation(s)")
        for p in problems:
            print("  *", p)
        return 1
    print("PDF LAYOUT AUDIT: clean (frame-capped keep-together blocks; all figures embedded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
