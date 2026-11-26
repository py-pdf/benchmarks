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

The first run since the benchmark itself changed: every library's text is
post-processed in the same way, a library is measured again when its version
or one of its optional dependencies changes, and the documents are pinned to
fixed versions and checked by SHA-256 (they did not change). pdftotext is now
listed with its Poppler version (24.02.0) instead of 0.86.1.

- Speed: every time is now the shortest of three runs, calibrated to the
  reference machine (see "Benchmarking machine" in the README), with Python
  pinned to 3.14. The pure-Python libraries became about 40% faster without a
  change of their code, e.g. pdfminer.six 5.6s -> 3.3s and pdfplumber
  8.9s -> 5.6s for text extraction: their raw times dropped by about 30%
  (shortest of three runs, and possibly another Python; the last run did not
  record its version), and the calibration takes off another 15%, as the
  machine was busy during the run (the calibration took 137ms on average
  instead of 115ms). The averages of the other libraries changed by at most
  0.2s, mostly from the calibration.
- Text extraction quality: before, only the text of pypdf and pypdfium2 was
  post-processed (ligatures, hyphenation, page labels). PyMuPDF
  94.2% -> 94.9%, Tika 93.1% -> 93.7%, pdfminer.six 87.9% -> 88.6%. Tika's
  text no longer includes the outline, as it is now taken per page from its
  XHTML output.

### pypdf
- Version: 6.21.3 -> 7.0.0, now as pypdf[full] (fontTools, brotli,
  arabic-reshaper and python-bidi were missing)
- Speed: text extraction 3.3s -> 1.8s, image extraction 4.2s -> 2.7s,
  watermarking 0.4s -> 0.3s; mostly due to the measurement, see above
- Text extraction quality: no change (94.9%)
- Notes: subscripts and superscripts in formulas are now separated by a
  space ("∑ki=1" -> "∑k i=1", "1Z(x)" -> "1 Z(x)"), which lowers the score of
  six documents by up to 0.1 points. The warnings that fontTools is needed to
  read CFF fonts are gone. The watermarked files differ by up to 22KB.

### pdfminer.six
- Version: 20260107 (no change), now as pdfminer.six[image]
- Speed: text extraction 5.6s -> 3.3s, image extraction 7.5s -> 4.3s; due to
  the measurement, see above
- Text extraction quality: 87.9% -> 88.6%, due to the post-processing

### PyMuPDF
- Version: 2.0 -> 2.1
- Speed: no change
- Text extraction quality: 94.2% -> 94.9%, due to the post-processing

### pypdfium2
- Version: 5.14.0 -> 5.15.0
- Speed: no change; image extraction 1.1s -> 0.9s from the calibration
- Text extraction quality: no change; the extracted text is the same

## 2026-11-06, image counts

pdfminer.six's image extraction is fixed, and the report now shows how many
images each library extracted. As the cache had no image counts, pypdf,
pdfminer.six, PyMuPDF and pypdfium2 were measured again. No library changed
its version, and only pdfminer.six's results changed.

### pdfminer.six
- Version: 20260107 (no change)
- Speed: image extraction 4.3s -> 4.5s. Its text extraction, which did not
  change, went from 3.3s to 3.5s in the same run, so the difference is
  within the noise of the measurement.
- Text extraction quality: no change
- Notes: the images are now decoded by pdfminer.six's ImageWriter. Before,
  every image was read as a 1-bit image, and the first failure ended the
  extraction of a document (1601.03642: 0 -> 5 images, 2201.00029: 0 -> 10,
  1602.06541: 1 -> 8). It now extracts 187 images in total, pypdf 184,
  pypdfium2 135 and PyMuPDF 108.

## 2026-11-06, watermark check

The watermarked documents are now checked: the report marks results with
pages that lack the watermark text, or with missing pages. The outputs of
all three libraries have the watermark on every page. As the cache had no
results of the check, pdfrw, PyMuPDF and pypdf were measured again. No
library changed its version.

### PyMuPDF
- Version: 2.1 (no change)
- Speed: watermarking 0.14s -> 0.10s on average
- Text extraction quality: no change
- Notes: the watermarking function now shows the page of the watermark
  document on every page, so all pages share the watermark. Before, it
  copied the watermark into a new document for every page. The watermarked
  files became 3.7MB -> 3.5MB on average (52.3MB -> 48.3MB for the 14
  documents), the same size as those of pypdf and pdfrw.

## 2026-11-06, feature matrices

The report now has feature matrices for image and text extraction: what
each library extracts from small PDFs with one feature each. No library was
measured again, and no other result changed.

## 2026-11-13

pypdf, pypdfium2 and Tika were measured again after a version change, and
the ground truth of 2201.00022 was replaced: it was the text of arXiv v1,
but the benchmark pins v2 (the abstract, the section numbers and the
references differ). Every library scored 58%-78% on that document, as
the text of v2 was compared with the text of v1; now 88%-99%, except
pdfplumber (58%, unchanged). The text of the libraries that did not change
is identical to the last run, so their average quality only rose through
the new ground truth: PyMuPDF 94.9% -> 96.4%, pdftotext 90.2% -> 91.6%,
pdfminer.six 88.6% -> 89.7%, pdfplumber 74.0% -> 74.0%. The libraries rank
the same way, except that pypdf and PyMuPDF swap places.

### pypdf
- Version: 7.0.0 -> 7.1.0
- Speed: text extraction 1.8s -> 1.7s, image extraction and watermarking
  no change; within the noise of the measurement
- Text extraction quality: 94.9% -> 96.6%; 94.9% -> 95.2% (+0.3 points)
  without the new ground truth of 2201.00022
- Notes: GeoTopo-book 94.6% -> 96.3% and 2201.00214 93.9% -> 94.4%. Spaces
  around symbols are right again in headings and formulas: "I NTRODUCTION"
  -> "INTRODUCTION", "n∈ N" -> "n ∈ N", "[SPB +14]" -> "[SPB+14]", and the
  space before a comma after an italic word is gone ("Computer models of
  musical creativity ." -> "creativity."). Words that were split by a
  stray space ("Astr ophysical", "KI C") are joined.

### pypdfium2
- Version: 5.15.0 -> 5.15.2
- Speed: no change
- Text extraction quality: 95.2% -> 96.7%, only due to the new ground truth

### Tika
- Version: 3.3.2 -> 3.4.0
- Speed: text extraction 0.24s -> 0.20s on average (uncalibrated); the
  report shows 0.2s in both runs
- Text extraction quality: 93.7% -> 95.1%, only due to the new ground truth

## 2026-11-26

pypdf, PyMuPDF and pdfplumber were measured again after a version change,
still on CPython 3.14.10. Python 3.15 is not possible yet: PyMuPDF depends on
onnxruntime, which has no wheels for it. The ground truth of 2201.00022 now
ends with a newline, which changes the score of the three re-measured
libraries on it by less than 0.001 points.

### pypdf
- Version: 7.1.1 -> 7.2.0; fonttools 4.68.0 -> 4.69.1
- Speed: text extraction 1.8s -> 1.7s, image extraction 2.8s -> 2.6s,
  watermarking 0.27s -> 0.25s on average. The machine was busier than in the
  last run (the calibration took 133ms instead of 111ms), so the raw times
  went up and only the calibrated ones went down; this is within the noise
  of the measurement.
- Text extraction quality: 96.6% -> 96.7%
- Notes: only the text of 2201.00069 changed (96.7% -> 97.2%). The symbols of
  its math font are now extracted: "∼", "−", "×", "±", "≤", "≲", "†", "◦" and
  "′′" were missing or wrong before ("139  10 16" -> "1𝑌39 × 10−16"). The comma
  and the decimal point of that font are now extracted as "𝑊" and "𝑌" instead
  of being dropped ("1,2" -> "1𝑊2", "0.2365" -> "0𝑌2365"), and "<" and ">"
  become "𝑚" and "𝑜".

### PyMuPDF
- Version: 2.1 -> 2.2
- Speed: text extraction no change (0.10s), watermarking 0.10s -> 0.12s,
  image extraction 0.47s -> 0.51s on average; the raw times rose by about as
  much
- Text extraction quality: no change; the extracted text is the same
- Notes: the image counts and the watermark check did not change.

### pdfplumber
- Version: 0.11.10 -> 0.11.11
- Speed: text extraction 5.6s -> 4.8s (raw 6.4s -> 5.9s at the same
  calibration time)
- Text extraction quality: no change; the extracted text is the same
