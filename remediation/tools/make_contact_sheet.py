"""Render every page of the final PDF into Final_Paper/page_review/ and
build a grid contact sheet (contact_sheet.png).

Usage: python make_contact_sheet.py <pdf> <outdir>
"""
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main():
    pdf = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("page-*.png"):
        old.unlink()
    subprocess.run(["pdftoppm", "-png", "-r", "110", str(pdf),
                    str(out / "page")], check=True)
    pages = sorted(out.glob("page-*.png"))
    n = len(pages)
    cols = 7
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.65))
    for ax in axes.flat:
        ax.axis("off")
    for i, p in enumerate(pages):
        ax = axes.flat[i]
        ax.imshow(mpimg.imread(str(p)))
        ax.set_title(p.stem.replace("page-", "p. "), fontsize=7, pad=2)
        for s in ("top", "bottom", "left", "right"):
            ax.spines[s].set_visible(True)
            ax.spines[s].set_linewidth(0.4)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{pdf.name} — {n} pages", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "contact_sheet.png", dpi=140)
    print(f"wrote {n} page images + contact_sheet.png to {out}")


if __name__ == "__main__":
    main()
