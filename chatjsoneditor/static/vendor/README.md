# Vendored libraries (offline)

These are pinned, unmodified copies of well-known MIT/Apache-licensed libraries, served locally so
the editor needs no internet at runtime.

| File            | Library   | Version | Source (jsDelivr / npm)                                          |
|-----------------|-----------|---------|------------------------------------------------------------------|
| `marked.min.js` | marked    | 12.0.2  | `https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js`       |
| `purify.min.js` | DOMPurify | 3.1.6   | `https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js`|
| `tex-svg.js`    | MathJax   | 3.2.2   | `https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js`      |

MathJax `tex-svg` is used (SVG output) because it is a single self-contained file with no external
font files — ideal for offline vendoring. To refresh, re-download the same URLs.
