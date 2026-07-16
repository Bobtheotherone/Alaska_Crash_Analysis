"""Render final_paper.md -> final_paper.html and final_paper.pdf with all figures embedded.

AA2-023 / PDF-001: every figure is embedded at a frame-capped size; each figure and its caption
are wrapped in a 1x1 keep-together table (page-break-inside: avoid) so they cannot be split or
clipped across pages; captions come from FIG_FILES below and must match final_paper.md's Figures
list terminology (the v3 release shipped stale captions from this list); title/author/date PDF
metadata is set and the byline date is pinned (deterministic builds). Layout is machine-audited
by research_audit/pdf_layout_audit.py and terminology by research_audit/claim_scan.py.
Figures are read from paper/generated/ and inlined as base64 so the HTML/PDF is self-contained.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
MD = HERE / "final_paper.md"
FIGS = HERE / "generated"

# (number, caption, filename) — order matches the paper's figure list. PDF-001: these captions
# are RENDERED into the HTML/PDF and MUST match the terminology of final_paper.md's "Figures"
# list — the v3 release shipped stale "sealed … grouped-bootstrap" captions from this very list
# while the .md was clean; the claim scan now checks the rendered HTML too.
FIG_FILES = [
    (1, "Out-of-time ordinal MAE with 95% crash-level bootstrap CIs (held-out, exposed retrospective 2012 evaluation).", "fig_final_ordinal_mae_ci.png"),
    (2, "The core tension: severe-class recall vs ordinal MAE (2012 evaluation).", "fig_final_tradeoff.png"),
    (3, "Severe-class precision–recall on the held-out out-of-time 2012 evaluation.", "fig_final_severe_pr.png"),
    (4, "Probabilistic quality (log loss, all 14 models, log scale) and calibration ECE, raw vs development-fit.", "fig_final_probabilistic.png"),
    (5, "Original re-analysis: accuracy and ordinal MAE vs the majority baseline.", "fig_reanalysis_accuracy_and_mae.png"),
    (6, "Original re-analysis: severe-class precision–recall tradeoff.", "fig_reanalysis_severe_pr.png"),
    (7, "Original re-analysis: predicted vs true class distribution.", "fig_reanalysis_predicted_dist.png"),
    (8, "Severe-class one-vs-rest precision–recall RANKING curves vs the no-skill prevalence (retrospective, exploratory; no operating threshold selected).", "fig_severe_ranking_pr.png"),
]

TITLE = "Leakage-Controlled, Governed Ordinal Classification of Alaska Crash Severity: A Reproducible, Retrospective Out-of-Time Study"
AUTHOR = "Naythan Mercado"
DATE = "2026-07-11"  # pinned to the paper's canonical version date (deterministic; matches title block: v3.1)

CSS = """
@page {
  size: letter; margin: 2cm; margin-bottom: 2.4cm;
  @frame footer_frame { -pdf-frame-content: footerContent; bottom: 0.9cm; margin-left: 2cm; margin-right: 2cm; height: 0.7cm; }
}
body { font-family: 'Helvetica','Arial',sans-serif; font-size: 10.5pt; line-height: 1.42; color: #111; }
h1 { font-size: 16pt; line-height: 1.25; margin-bottom: 2px; }
h2 { font-size: 12.5pt; border-bottom: 1px solid #ccc; padding-bottom: 2px; margin-top: 15px; }
h3 { font-size: 10.8pt; margin-top: 10px; }
p, li { text-align: left; }
table { border-collapse: collapse; width: 100%; font-size: 7.4pt; margin: 8px 0; }
th, td { border: 1px solid #999; padding: 2px 4px; text-align: left; }
th { background: #eee; }
code { font-family: 'Courier New', monospace; font-size: 8.2pt; background: #f3f3f3; }
img { max-width: 100%; }
figure { margin: 10px 0 16px 0; }
figcaption { font-size: 8.8pt; color: #444; font-style: italic; margin-top: 4px; }
/* keep each figure and its caption on ONE page (the v3 PDF detached/clipped some of them):
   xhtml2pdf honours page-break-inside:avoid on tables, so figures render inside a 1x1 table. */
table.figtable { page-break-inside: avoid; border: none; width: 100%; margin: 10px 0 16px 0; }
table.figtable td { border: none; padding: 0; }
span.figcap { font-size: 8.8pt; color: #444; font-style: italic; }
.byline { color: #333; font-size: 10pt; margin: 2px 0 10px 0; }
"""


def _patch_xhtml2pdf_getplaintext() -> None:
    """Fix an xhtml2pdf 0.2.17 bug: reportlab_paragraph.Paragraph.getPlainText joins a
    generator of *lists* (``[frag.text]``) instead of strings, raising
    ``TypeError: sequence item 0: expected str instance, list found`` whenever reportlab tries
    to identify an overflowing flowable (a wide table or an unbreakable DOI token). We replace
    it with the corrected one-line implementation."""
    try:
        from xhtml2pdf import reportlab_paragraph as rp

        def getPlainText(self, identify=None):
            frags = getattr(self, "frags", None)
            if not frags:
                return ""
            return "".join(f.text for f in frags if hasattr(f, "text") and isinstance(f.text, str))

        rp.Paragraph.getPlainText = getPlainText
    except Exception:  # pragma: no cover
        pass


def _png_size(data: bytes):
    """(width, height) in px from a PNG's IHDR — no PIL dependency."""
    import struct
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def _fit_cm(w_px: int, h_px: int, max_w_cm=15.0, max_h_cm=8.5):
    """Scale a pixel image to fit within (max_w_cm x max_h_cm), preserving aspect."""
    aspect = w_px / h_px
    w_cm, h_cm = max_w_cm, max_w_cm / aspect
    if h_cm > max_h_cm:
        h_cm, w_cm = max_h_cm, max_h_cm * aspect
    return round(w_cm, 2), round(h_cm, 2)


def embed_figs_html() -> str:
    parts = ['<h2>Figures</h2>']
    for num, caption, fname in FIG_FILES:
        p = FIGS / fname
        if not p.exists():
            parts.append(f'<p><em>[Figure {num} missing: {fname}]</em></p>')
            continue
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode()
        size = _png_size(data)
        # Set BOTH width and height (aspect-preserved, frame-capped): xhtml2pdf otherwise
        # renders at native pixel size or drops aspect, overflowing the page (AA2-023).
        style = f"width:{_fit_cm(*size)[0]}cm;height:{_fit_cm(*size)[1]}cm;" if size else "width:15cm;"
        parts.append(
            f'<table class="figtable"><tr><td>'
            f'<img src="data:image/png;base64,{b64}" style="{style}"/><br/>'
            f'<span class="figcap"><b>Figure {num}.</b> {caption}</span>'
            f'</td></tr></table>'
        )
    return "\n".join(parts)


def main():
    md_text = MD.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    header = (
        f"<div class='byline'>{AUTHOR} &middot; {DATE} &middot; "
        f"Graduate research portfolio (predictive, non-causal study)</div>"
    )
    footer = ("<div id='footerContent' style='text-align:center; font-size:8pt; color:#666;'>"
              "Page <pdf:pagenumber /> of <pdf:pagecount /></div>")
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='author' content='{AUTHOR}'><title>{TITLE}</title>"
        f"<style>{CSS}</style></head><body>"
        f"{footer}{header}{body}{embed_figs_html()}"
        f"</body></html>"
    )
    (HERE / "final_paper.html").write_text(html, encoding="utf-8")

    try:
        from xhtml2pdf import pisa
        _patch_xhtml2pdf_getplaintext()
        with open(HERE / "final_paper.pdf", "wb") as f:
            status = pisa.CreatePDF(html, dest=f, encoding="utf-8")  # metadata via <title>/<meta>
        n_figs = sum((FIGS / fn).exists() for _, _, fn in FIG_FILES)
        print(f"wrote final_paper.pdf (err={status.err}); {n_figs}/{len(FIG_FILES)} figures embedded")
    except Exception as exc:  # pragma: no cover
        print("PDF render failed (HTML still produced):", repr(exc))


if __name__ == "__main__":
    main()
