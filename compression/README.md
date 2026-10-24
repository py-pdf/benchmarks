# PDF Compression Benchmark

How much smaller does a PDF get when it is rewritten with each library's
best lossless settings? All tools write unencrypted output. Ghostscript
re-renders the document and is included as a lossy reference only.

Run it with [uv](https://docs.astral.sh/uv/) from the repository root:

```bash
uv run compression/benchmark.py
# Single documents / compressors, e.g. after a pypdf change:
uv run compression/benchmark.py --documents GeoTopo-book --compressors pypdf
# Benchmark a local pypdf checkout instead of the locked release:
uv run --with-editable ../pypdf compression/benchmark.py --compressors pypdf
```

The script declares its dependencies inline; `benchmark.py.lock` pins them.
Documents are downloaded to `pdfs/` (shared with the main benchmark) and
verified by SHA-256; outputs are written to `compression/output/`.

## Environment

* Date: 2026-10-24
* Processor: AMD Ryzen 7 PRO 4750U with Radeon Graphics (16 cores)
* System: Linux 7.0.0-38-generic, Python 3.15.0

## Compressors

| Name          | Version | Lossless | Settings                                                                                      |
| :------------ | :------ | :------- | :-------------------------------------------------------------------------------------------- |
| pypdf         | 6.21.2  | yes      | `PdfWriter(clone_from=reader)` + `write()`                                                    |
| pypdf-streams | 6.21.2  | yes      | + `page.compress_content_streams(level=9)` for every page                                     |
| pypdf-dedupe  | 6.21.2  | yes      | + `compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)`              |
| pypdf-all     | 6.21.2  | yes      | both of the above                                                                             |
| pikepdf       | 10.17.0 | yes      | `save(compress_streams=True, recompress_flate=True, object_stream_mode=generate)`             |
| pymupdf       | 1.28.2  | yes      | `save(garbage=4, deflate=True, deflate_images=True, deflate_fonts=True, use_objstms=1)`       |
| ghostscript   | 10.02.1 | no       | `gs -sDEVICE=pdfwrite -dPDFSETTINGS=/default` (re-renders the document, may re-encode images) |

## Summary

Total size of all documents: 115.3 MiB. *Median* is the median of the per-document
output size relative to the input. Failed runs and runs whose text
differs from the input are counted with the input size.

| Compressor    |     Total | Total vs. input |  Median | Smallest |    Time | Problems |
| :------------ | --------: | --------------: | ------: | -------: | ------: | -------: |
| pypdf         | 119.1 MiB |         103.3 % | 100.2 % |        0 |  58.0 s |        0 |
| pypdf-streams | 118.8 MiB |         103.1 % | 100.0 % |        0 |  82.0 s |        0 |
| pypdf-dedupe  | 114.2 MiB |          99.1 % | 100.0 % |        0 | 101.9 s |        1 |
| pypdf-all     | 113.9 MiB |          98.8 % | 100.0 % |        0 |  83.9 s |        1 |
| pikepdf       |  98.3 MiB |          85.3 % |  95.6 % |        5 |  13.1 s |        0 |
| pymupdf       | 108.8 MiB |          94.4 % |  95.9 % |        9 |  27.3 s |        1 |
| ghostscript   | 112.9 MiB |          97.9 % | 100.0 % |        5 |  36.1 s |       10 |

## Documents

| Document                                                                                                                                  | Producer                                  | Pages |      Size | Object streams |
| :---------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------- | ----: | --------: | :------------- |
| [2201.00214](https://arxiv.org/pdf/2201.00214.pdf)                                                                                        | pdfTeX                                    |    22 |   2.4 MiB | no             |
| [1707.09725](https://arxiv.org/pdf/1707.09725.pdf)                                                                                        | pdfTeX                                    |   134 |   7.0 MiB | yes            |
| [2201.00069](https://arxiv.org/pdf/2201.00069.pdf)                                                                                        | pdfTeX (image-heavy)                      |    15 |  14.7 MiB | yes            |
| [GeoTopo-book](https://github.com/py-pdf/sample-files/raw/main/009-pdflatex-geotopo/GeoTopo.pdf)                                          | pdfTeX                                    |   117 |   5.1 MiB | yes            |
| [PDF32000_2008](https://github.com/py-pdf/pypdf/files/11714214/PDF32000_2008.pdf)                                                         | Acrobat Distiller 8.1 (tagged, encrypted) |   756 |  21.4 MiB | no             |
| [tika-972174](https://github.com/user-attachments/files/18381744/tika-972174.pdf)                                                         | Acrobat Distiller 8.0                     |   163 |   3.7 MiB | yes            |
| [Ballinasloe_WS](https://github.com/py-pdf/pypdf/files/10983477/Ballinasloe_WS.pdf)                                                       | Adobe Acrobat Pro DC                      |    35 |  11.1 MiB | yes            |
| [2021_book_security](https://github.com/user-attachments/files/15605648/2021_book_security.pdf)                                           | Adobe PDF Library 15                      |   268 |   3.7 MiB | no             |
| [USC-EMBA](https://github.com/py-pdf/pypdf/files/11578953/USC.EMBA.-.Pre-Season.and.Theme.I.pdf)                                          | PDFium                                    |   225 |  15.4 MiB | no             |
| [crystallography](https://github.com/py-pdf/pypdf/files/8884470/fdocuments.in_sweet-fundamentals-of-crystallography.pdf)                  | OpenOffice.org 2.3                        |   154 |   9.1 MiB | no             |
| [irbookonlinereading](https://github.com/py-pdf/pypdf/files/15234776/irbookonlinereading.pdf)                                             | dvips + Ghostscript 8.51                  |   581 |   6.6 MiB | no             |
| [FTX-claim](https://github.com/py-pdf/pypdf/files/13127197/FTX.Claim.SC30.01072023101624File595287144.pdf)                                | iTextSharp 5.5                            |    10 |   4.8 MiB | no             |
| [WBT-whitepaper](https://github.com/py-pdf/pypdf/files/12543290/whitepaper.WBT.token.blockchain.whitepaper.pdf)                           | iLovePDF                                  |    39 |   9.7 MiB | yes            |
| [libreoffice-writer](https://github.com/py-pdf/sample-files/raw/main/002-trivial-libre-office-writer/002-trivial-libre-office-writer.pdf) | LibreOffice                               |     1 |  12.3 KiB | no             |
| [google-doc](https://github.com/py-pdf/sample-files/raw/main/011-google-doc-document/google-doc-document.pdf)                             | Google Docs (Skia)                        |     1 |  78.2 KiB | no             |
| [reportlab-overlay](https://github.com/py-pdf/sample-files/raw/main/013-reportlab-overlay/reportlab-overlay.pdf)                          | ReportLab                                 |     1 |  13.5 KiB | no             |
| [pdfkit](https://github.com/py-pdf/sample-files/raw/main/022-pdfkit/pdfkit.pdf)                                                           | pdfkit (JavaScript)                       |     1 |  14.1 KiB | no             |
| [imagemagick-images](https://github.com/py-pdf/sample-files/raw/main/007-imagemagick-images/imagemagick-images.pdf)                       | ImageMagick                               |     6 |  15.6 KiB | no             |
| [cmyk-image](https://github.com/py-pdf/sample-files/raw/main/023-cmyk-image/cmyk-image.pdf)                                               | unknown (CMYK image)                      |     1 | 433.5 KiB | no             |

## Output size relative to the input

| Document            |   pypdf | pypdf-streams | pypdf-dedupe | pypdf-all | pikepdf | pymupdf |  ghostscript |
| :------------------ | ------: | ------------: | -----------: | --------: | ------: | ------: | -----------: |
| 2201.00214          | 100.2 % |        99.9 % |      100.1 % |    99.9 % |  97.7 % |  97.7 % |       99.3 % |
| 1707.09725          |  99.0 % |        99.0 % |       99.0 % |    99.0 % | 115.4 % |  93.6 % | text differs |
| 2201.00069          |  99.9 % |        99.9 % |       99.9 % |    99.9 % | 101.9 % |  99.1 % | text differs |
| GeoTopo-book        | 107.6 % |       107.6 % |      107.5 % |   107.5 % | 101.1 % |  99.3 % | text differs |
| PDF32000_2008       | 104.6 % |       104.0 % |       96.5 % |    95.9 % |  34.5 % |   error | text differs |
| tika-972174         | 146.4 % |       145.8 % |      124.6 % |   124.0 % |  85.9 % |  85.5 % | text differs |
| Ballinasloe_WS      | 103.8 % |       103.4 % |      103.4 % |   103.1 % |  98.6 % |  99.6 % | text differs |
| 2021_book_security  | 102.7 % |       102.6 % |      101.2 % |   101.0 % |  79.2 % |  79.5 % |       95.8 % |
| USC-EMBA            | 100.3 % |       100.0 % |       86.0 % |    85.7 % |  94.3 % |  83.0 % | text differs |
| crystallography     |  99.9 % |        99.9 % |       99.8 % |    99.8 % |  99.5 % |  99.4 % |       93.1 % |
| irbookonlinereading | 101.8 % |       101.5 % |      101.4 % |   101.0 % |  72.7 % |  74.0 % | text differs |
| FTX-claim           | 100.0 % |       100.0 % |      100.0 % |   100.0 % |  98.4 % | 100.0 % |       75.9 % |
| WBT-whitepaper      | 101.6 % |       101.0 % |      101.5 % |   100.9 % |  99.7 % |  99.6 % | text differs |
| libreoffice-writer  |  99.7 % |        99.6 % |       99.7 % |    99.6 % |  95.6 % |  95.9 % |      111.7 % |
| google-doc          | 100.2 % |       100.1 % |      100.2 % |   100.0 % |  92.6 % |  92.4 % | text differs |
| reportlab-overlay   | 100.6 % |       100.1 % |        error |     error |  91.8 % |  92.4 % |       58.2 % |
| pdfkit              |  99.8 % |        99.8 % |       99.8 % |    99.8 % |  88.3 % |  88.7 % |       47.6 % |
| imagemagick-images  |  87.8 % |        88.5 % |       60.2 % |    60.9 % |  57.6 % |  42.5 % |       75.8 % |
| cmyk-image          | 100.0 % |       100.0 % |      100.0 % |   100.0 % | 100.0 % | 100.0 % |        0.5 % |

## Time

| Document            |   pypdf | pypdf-streams | pypdf-dedupe | pypdf-all | pikepdf | pymupdf |  ghostscript |
| :------------------ | ------: | ------------: | -----------: | --------: | ------: | ------: | -----------: |
| 2201.00214          |  0.15 s |        1.01 s |       0.39 s |    1.23 s |  0.54 s |  0.11 s |       2.49 s |
| 1707.09725          |  0.64 s |        0.68 s |       8.26 s |    7.69 s |  1.45 s |  0.29 s | text differs |
| 2201.00069          |  0.25 s |        0.26 s |       1.09 s |    0.78 s |  2.14 s |  0.13 s | text differs |
| GeoTopo-book        |  0.80 s |        0.98 s |       1.41 s |    1.28 s |  0.69 s |  1.16 s | text differs |
| PDF32000_2008       | 46.68 s |       56.17 s |      61.32 s |   52.46 s |  1.71 s |   error | text differs |
| tika-972174         |  2.76 s |        7.45 s |       6.25 s |    2.81 s |  0.34 s |  9.28 s | text differs |
| Ballinasloe_WS      |  0.59 s |        1.82 s |       1.80 s |    0.74 s |  0.16 s |  0.92 s | text differs |
| 2021_book_security  |  1.31 s |        2.42 s |       4.01 s |    1.71 s |  0.35 s |  1.15 s |       2.30 s |
| USC-EMBA            |  0.88 s |        1.49 s |       3.12 s |    1.72 s |  2.05 s |  0.36 s | text differs |
| crystallography     |  0.19 s |        0.34 s |       0.44 s |    0.35 s |  0.37 s |  0.12 s |       1.54 s |
| irbookonlinereading |  2.82 s |        4.15 s |      12.55 s |    9.49 s |  0.99 s | 12.89 s | text differs |
| FTX-claim           |  0.08 s |        0.08 s |       0.16 s |    0.14 s |  0.23 s |  0.10 s |       0.28 s |
| WBT-whitepaper      |  0.42 s |        4.33 s |       0.68 s |    3.09 s |  1.75 s |  0.21 s | text differs |
| libreoffice-writer  |  0.07 s |        0.15 s |       0.08 s |    0.07 s |  0.04 s |  0.09 s |       0.07 s |
| google-doc          |  0.08 s |        0.18 s |       0.09 s |    0.09 s |  0.05 s |  0.10 s | text differs |
| reportlab-overlay   |  0.07 s |        0.12 s |        error |     error |  0.04 s |  0.10 s |       0.07 s |
| pdfkit              |  0.07 s |        0.10 s |       0.08 s |    0.07 s |  0.04 s |  0.10 s |       0.06 s |
| imagemagick-images  |  0.07 s |        0.13 s |       0.08 s |    0.08 s |  0.04 s |  0.10 s |       0.12 s |
| cmyk-image          |  0.07 s |        0.13 s |       0.08 s |    0.08 s |  0.08 s |  0.09 s |       0.06 s |

## Peak memory (whole process)

| Document            |     pypdf | pypdf-streams | pypdf-dedupe | pypdf-all |   pikepdf |   pymupdf |  ghostscript |
| :------------------ | --------: | ------------: | -----------: | --------: | --------: | --------: | -----------: |
| 2201.00214          | 117.4 MiB |     136.3 MiB |    144.8 MiB | 144.8 MiB | 147.2 MiB | 156.6 MiB |    159.7 MiB |
| 1707.09725          | 117.4 MiB |     136.3 MiB |    233.4 MiB | 234.5 MiB | 147.2 MiB | 156.6 MiB | text differs |
| 2201.00069          | 117.4 MiB |     136.3 MiB |    225.9 MiB | 226.2 MiB | 147.2 MiB | 156.6 MiB | text differs |
| GeoTopo-book        | 118.3 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 147.3 MiB | 159.7 MiB | text differs |
| PDF32000_2008       | 572.9 MiB |     585.9 MiB |    617.7 MiB | 631.6 MiB | 234.9 MiB |     error | text differs |
| tika-972174         | 133.1 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 154.7 MiB | 159.7 MiB | text differs |
| Ballinasloe_WS      | 133.6 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 154.7 MiB | 159.7 MiB | text differs |
| 2021_book_security  | 136.3 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 156.6 MiB | 159.7 MiB |    159.7 MiB |
| USC-EMBA            | 136.3 MiB |     144.8 MiB |    302.7 MiB | 305.9 MiB | 156.6 MiB | 159.7 MiB | text differs |
| crystallography     | 136.3 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 156.6 MiB | 159.7 MiB |    159.7 MiB |
| irbookonlinereading | 136.3 MiB |     144.8 MiB |    199.6 MiB | 211.5 MiB | 156.6 MiB | 159.7 MiB | text differs |
| FTX-claim           | 136.3 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 156.6 MiB | 159.7 MiB |    159.7 MiB |
| WBT-whitepaper      | 136.3 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 156.6 MiB | 159.7 MiB | text differs |
| libreoffice-writer  | 136.3 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 156.6 MiB | 159.7 MiB |    159.7 MiB |
| google-doc          | 136.3 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 156.6 MiB | 159.7 MiB | text differs |
| reportlab-overlay   | 136.3 MiB |     144.8 MiB |        error |     error | 156.6 MiB | 159.7 MiB |    159.7 MiB |
| pdfkit              | 136.3 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 156.6 MiB | 159.7 MiB |    159.7 MiB |
| imagemagick-images  | 136.3 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 156.6 MiB | 159.7 MiB |    159.7 MiB |
| cmyk-image          | 136.3 MiB |     144.8 MiB |    144.8 MiB | 147.2 MiB | 156.6 MiB | 159.7 MiB |    159.7 MiB |

## Problems

Ghostscript re-renders the document, so changed text extraction is
expected for it; for the lossless compressors it indicates a bug.

* pypdf-dedupe / reportlab-overlay: pypdf.errors.PdfReadError: Object IndirectObject(12, 0, 130040471039888) not found!
* pypdf-all / reportlab-overlay: pypdf.errors.PdfReadError: Object IndirectObject(12, 0, 128230370324368) not found!
* pymupdf / PDF32000_2008: timeout after 900 s
* ghostscript / 1707.09725: text differs
* ghostscript / 2201.00069: text differs
* ghostscript / GeoTopo-book: text differs
* ghostscript / PDF32000_2008: text differs
* ghostscript / tika-972174: text differs
* ghostscript / Ballinasloe_WS: text differs
* ghostscript / USC-EMBA: text differs
* ghostscript / irbookonlinereading: text differs
* ghostscript / WBT-whitepaper: text differs
* ghostscript / google-doc: text differs
