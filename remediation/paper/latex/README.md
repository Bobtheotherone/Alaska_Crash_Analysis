# UAA-Informed Visual Revision

This package is a visual reconstruction of the research manuscript. It uses a restrained UAA student-paper layout: US Letter, one-inch margins, Times-style serif type, 1.5 line spacing, centered footer page numbers, black headings, conventional captions, front matter, and portrait appendices. It is not represented as an officially approved UAA thesis template.

## Build

Run from this directory:

```bash
./build.sh
```

The script runs LaTeX, BibTeX, and the required cross-reference passes.

## Figures

Place replacement images in `figures/user/` using the exact filenames in `FIGURE_REPLACEMENT_GUIDE.md`. The manuscript will detect them automatically. Until then, clean dimensionally accurate placeholders are shown.
