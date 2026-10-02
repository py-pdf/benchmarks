# Benchmark History

Tracks what changed between benchmark runs: library version bumps, and any
resulting shifts in extraction speed, text-extraction quality, or output
characteristics worth calling out.

Each run gets a dated section. Each library that changed gets a subsection:

```
### <library>
- Version: <old> -> <new>
- Speed: <change, or "no change">
- Text extraction quality: <change, or "no change">
- Notes: <anything qualitative - new garbling, fixed hyphenation, etc.>
```

## 2026-10-02

### tika
- Version: 3.1.0 -> 3.3.2
- Speed: no change
- Text extraction quality: no change

### pypdf
- Version: 5.7.0 -> 6.19.0
- Speed: watermarking ~2.5x slower; text and image extraction no change
- Text extraction quality: no change
- Notes: prose is unchanged; nearly all changes are in inline math. The
  space around a math symbol now tends to follow it instead of preceding it,
  so words get glued to the next variable ("where n" -> "wheren",
  "2.042 ±0.08" -> "2.042± 0.08"). Some math glyphs map differently
  ("≳" -> "&", "≠" -> "⁄=", "ℓ" -> "𝓁").

### pdfminer.six
- Version: 20250506 -> 20260107
- Speed: no change
- Text extraction quality: no change

### pdfplumber
- Version: 0.11.7 -> 0.11.10
- Speed: no change
- Text extraction quality: no change

### PyMuPDF
- Version: 1.26.1 -> 1.28.2
- Speed: no change
- Text extraction quality: no change
- Notes: the combining slash of "≠" now attaches to the preceding character
  ("x ̸= y" -> "x̸ = y").

### pypdfium2
- Version: 4.30.1 -> 5.13.0
- Speed: no change
- Text extraction quality: slightly better
- Notes: words hyphenated across a line break are now re-joined; before,
  they contained a stray control character (`experi\x02ence`). Each re-joined
  word is now followed by a line break. Math symbols that used to be dropped
  are now extracted ("3 above the mean" -> "3𝝈 above the mean").

### pdftotext
- Version: Poppler 24.02.0 (previous version not recorded)
- Speed: no change
- Text extraction quality: slightly better
- Notes: text that belongs on one line now stays together. Before, footnote
  and affiliation markers, bullet points, `#include <...>` lines and
  table-of-contents entries were split into separate blocks. Side-by-side
  subfigure captions are now interleaved line by line.
