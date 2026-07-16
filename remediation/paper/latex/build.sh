#!/usr/bin/env bash
set -euo pipefail

pdflatex -interaction=nonstopmode -halt-on-error main.tex
if command -v bibtex >/dev/null 2>&1 && bibtex --version >/dev/null 2>&1; then
  bibtex main
elif [[ -x /usr/bin/bibtex.original ]]; then
  /usr/bin/bibtex.original main
else
  echo "BibTeX was not found. Install a TeX distribution that includes bibtex." >&2
  exit 1
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
