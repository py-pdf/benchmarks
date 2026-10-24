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
* Processor: Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz (8 cores)
* System: Linux 6.8.0-147-generic, Python 3.13.3

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
| pypdf         | 119.1 MiB |         103.3 % | 100.2 % |        0 |  74.2 s |        0 |
| pypdf-streams | 118.8 MiB |         103.1 % | 100.0 % |        0 |  80.4 s |        0 |
| pypdf-dedupe  | 114.2 MiB |          99.1 % | 100.0 % |        0 |  90.1 s |        1 |
| pypdf-all     | 113.9 MiB |          98.8 % | 100.0 % |        0 | 100.4 s |        1 |
| pikepdf       |  98.3 MiB |          85.3 % |  95.6 % |        5 |  16.6 s |        0 |
| pymupdf       |  95.4 MiB |          82.7 % |  93.6 % |        9 | 865.7 s |        0 |
| ghostscript   | 112.9 MiB |          97.9 % | 100.0 % |        5 |  45.3 s |       10 |

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
| PDF32000_2008       | 104.6 % |       104.0 % |       96.5 % |    95.9 % |  34.5 % |  37.2 % | text differs |
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

| Document            |   pypdf | pypdf-streams | pypdf-dedupe | pypdf-all | pikepdf |  pymupdf |  ghostscript |
| :------------------ | ------: | ------------: | -----------: | --------: | ------: | -------: | -----------: |
| 2201.00214          |  0.55 s |        1.31 s |       0.27 s |    1.40 s |  0.76 s |   0.13 s |       2.93 s |
| 1707.09725          |  0.67 s |        0.78 s |       8.20 s |    8.41 s |  1.79 s |   0.44 s | text differs |
| 2201.00069          |  0.27 s |        0.32 s |       0.93 s |    0.92 s |  2.77 s |   0.18 s | text differs |
| GeoTopo-book        |  0.82 s |        1.07 s |       1.13 s |    1.34 s |  0.81 s |   1.98 s | text differs |
| PDF32000_2008       | 61.51 s |       62.59 s |      61.89 s |   65.13 s |  2.08 s | 836.06 s | text differs |
| tika-972174         |  3.25 s |        2.91 s |       2.69 s |    2.91 s |  0.37 s |  10.50 s | text differs |
| Ballinasloe_WS      |  0.72 s |        0.69 s |       0.71 s |    0.81 s |  0.18 s |   1.07 s | text differs |
| 2021_book_security  |  1.48 s |        1.67 s |       1.56 s |    1.91 s |  0.43 s |   1.43 s |       2.67 s |
| USC-EMBA            |  0.90 s |        1.05 s |       1.74 s |    1.97 s |  2.92 s |   0.41 s | text differs |
| crystallography     |  0.20 s |        0.29 s |       0.29 s |    0.41 s |  0.59 s |   0.13 s |       1.84 s |
| irbookonlinereading |  2.80 s |        3.67 s |       9.48 s |   10.89 s |  1.25 s |  12.41 s | text differs |
| FTX-claim           |  0.09 s |        0.09 s |       0.17 s |    0.17 s |  0.29 s |   0.10 s |       0.34 s |
| WBT-whitepaper      |  0.43 s |        3.51 s |       0.64 s |    3.66 s |  2.10 s |   0.30 s | text differs |
| libreoffice-writer  |  0.08 s |        0.08 s |       0.08 s |    0.08 s |  0.04 s |   0.09 s |       0.07 s |
| google-doc          |  0.09 s |        0.09 s |       0.10 s |    0.10 s |  0.05 s |   0.09 s | text differs |
| reportlab-overlay   |  0.08 s |        0.08 s |        error |     error |  0.04 s |   0.09 s |       0.07 s |
| pdfkit              |  0.08 s |        0.08 s |       0.08 s |    0.08 s |  0.04 s |   0.09 s |       0.07 s |
| imagemagick-images  |  0.08 s |        0.09 s |       0.09 s |    0.09 s |  0.04 s |   0.09 s |       0.14 s |
| cmyk-image          |  0.08 s |        0.08 s |       0.09 s |    0.09 s |  0.09 s |   0.09 s |       0.06 s |

## Peak memory (whole process)

| Document            |     pypdf | pypdf-streams | pypdf-dedupe | pypdf-all |   pikepdf |   pymupdf |  ghostscript |
| :------------------ | --------: | ------------: | -----------: | --------: | --------: | --------: | -----------: |
| 2201.00214          | 170.3 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB |    172.8 MiB |
| 1707.09725          | 172.3 MiB |     172.8 MiB |    238.5 MiB | 238.8 MiB | 172.8 MiB | 172.8 MiB | text differs |
| 2201.00069          | 172.3 MiB |     172.8 MiB |    242.1 MiB | 242.2 MiB | 172.8 MiB | 172.8 MiB | text differs |
| GeoTopo-book        | 172.3 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB | text differs |
| PDF32000_2008       | 638.4 MiB |     640.8 MiB |    679.4 MiB | 682.6 MiB | 229.9 MiB | 172.8 MiB | text differs |
| tika-972174         | 172.4 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB | text differs |
| Ballinasloe_WS      | 172.6 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB | text differs |
| 2021_book_security  | 172.6 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB |    172.8 MiB |
| USC-EMBA            | 172.6 MiB |     172.8 MiB |    303.6 MiB | 304.2 MiB | 172.8 MiB | 172.8 MiB | text differs |
| crystallography     | 172.6 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB |    172.8 MiB |
| irbookonlinereading | 172.6 MiB |     172.8 MiB |    197.4 MiB | 201.1 MiB | 172.8 MiB | 172.8 MiB | text differs |
| FTX-claim           | 172.8 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB |    172.8 MiB |
| WBT-whitepaper      | 172.8 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB | text differs |
| libreoffice-writer  | 172.8 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB |    172.8 MiB |
| google-doc          | 172.8 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB | text differs |
| reportlab-overlay   | 172.8 MiB |     172.8 MiB |        error |     error | 172.8 MiB | 172.8 MiB |    172.8 MiB |
| pdfkit              | 172.8 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB |    172.8 MiB |
| imagemagick-images  | 172.8 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB |    172.8 MiB |
| cmyk-image          | 172.8 MiB |     172.8 MiB |    172.8 MiB | 172.8 MiB | 172.8 MiB | 172.8 MiB |    172.8 MiB |

## Problems

Ghostscript re-renders the document, so changed text extraction is
expected for it; for the lossless compressors it indicates a bug.

* pypdf-dedupe / reportlab-overlay: pypdf.errors.PdfReadError: Object IndirectObject(12, 0, 137934123795856) not found!
* pypdf-all / reportlab-overlay: pypdf.errors.PdfReadError: Object IndirectObject(12, 0, 139880192673168) not found!
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
