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

## 2026-11-06

No library changed its version; the benchmark itself changed. pdftotext is
now listed with its Poppler version (24.02.0) instead of 0.86.1.

- Text extraction quality: every library's text is post-processed in the
  same way (ligatures, hyphenation, page labels); before, only the text of
  pypdf and pypdfium2 was. Their scores did not change; PyMuPDF 94% -> 95%,
  Tika 93% -> 94%, pdfminer.six 88% -> 89%. Tika's text no longer includes
  the outline, as it is now taken per page from its XHTML output.
- Speed: every time is the shortest of three runs instead of a single run,
  so most averages became 5-15% shorter, without a change of the libraries.
- Image extraction: pdfminer.six now decodes the images (before, every image
  was read as a 1-bit image) and finds the images of all containers; it is
  faster than before (7.5s -> 6.6s on average) although it extracts more
  images. The report shows how many images each library extracted.
- Watermarking: PyMuPDF shows the watermark page directly instead of
  copying it into a new document for every page, so the watermark is shared:
  0.2s -> 0.1s and 3.7MB -> 3.5MB on average. The report marks outputs that
  lack the watermark text on a page; none does.
- Documents: the URLs point to fixed versions (arXiv version, sample-files
  commit), and the files are checked by SHA-256. The files did not change.
