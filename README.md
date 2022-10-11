# PDF Text Extraction Benchmark
This benchmark is about reading pure PDF files - notscanned documents and not documents that applied OCR.

## Benchmarking machine
 Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz

## Input Documents
| #  |                                               Name                                               | File Size | Pages |
| -: | :----------------------------------------------------------------------------------------------- | --------: | ----: |
|  1 | [2201.00214](https://arxiv.org/pdf/2201.00214.pdf)                                               |    2.4MiB |    22 |
|  2 | [GeoTopo-book](https://github.com/py-pdf/sample-files/raw/main/009-pdflatex-geotopo/GeoTopo.pdf) |    5.1MiB |   117 |
|  3 | [2201.00151](https://arxiv.org/pdf/2201.00151.pdf)                                               |    1.5MiB |    12 |
|  4 | [1707.09725](https://arxiv.org/pdf/1707.09725.pdf)                                               |    7.0MiB |   134 |
|  5 | [2201.00021](https://arxiv.org/pdf/2201.00021.pdf)                                               |    2.6MiB |    10 |
|  6 | [2201.00037](https://arxiv.org/pdf/2201.00037.pdf)                                               |    2.9MiB |    33 |
|  7 | [2201.00069](https://arxiv.org/pdf/2201.00069.pdf)                                               |   14.7MiB |    15 |
|  8 | [2201.00178](https://arxiv.org/pdf/2201.00178.pdf)                                               |    2.3MiB |    16 |
|  9 | [2201.00201](https://arxiv.org/pdf/2201.00201.pdf)                                               |    1.3MiB |     9 |
| 10 | [1602.06541](https://arxiv.org/pdf/1602.06541.pdf)                                               |    2.9MiB |    16 |
| 11 | [2201.00200](https://arxiv.org/pdf/2201.00200.pdf)                                               |  284.8KiB |     7 |
| 12 | [2201.00022](https://arxiv.org/pdf/2201.00022.pdf)                                               |    1.1MiB |    11 |
| 13 | [2201.00029](https://arxiv.org/pdf/2201.00029.pdf)                                               |  797.6KiB |    12 |
| 14 | [1601.03642](https://arxiv.org/pdf/1601.03642.pdf)                                               | 1004.9KiB |     8 |

## Libraries
|     Name     | Last PyPI Release |             License             | Version  |                       Dependencies                        |
| -----------: | :---------------- | ------------------------------: | -------: | :-------------------------------------------------------- |
|         Borb | 2022-09-15        |                 AGPL/Commercial |    2.1.3 |                                                           |
|    pypdfium2 | 2022-10-11        |      Apache-2.0 or BSD-3-Clause |    3.3.0 | PDFium (Foxit/Google)                                     |
| pdfminer.six | 2022-05-24        |                           MIT/X | 20220524 |                                                           |
|   pdfplumber | 2022-07-21        |                             MIT |    0.7.4 |                                                           |
|    pdftotext | -                 |                             GPL |   0.86.1 | build-essential libpoppler-cpp-dev pkg-config python3-dev |
|      PyMuPDF | 2022-08-31        | GNU AFFERO GPL 3.0 / Commerical |   1.20.2 | MuPDF                                                     |
|       PyPDF2 | 2022-09-25        |                    BSD 3-Clause |   2.11.1 |                                                           |
|         Tika | 2020-03-21        |                       Apache v2 |     1.24 | Apache Tika                                               |


## Text Extraction Speed

| #  |                          Library                          | Average | [   1   ](https://arxiv.org/pdf/2201.00214.pdf) | [   2   ](https://github.com/py-pdf/sample-files/raw/main/009-pdflatex-geotopo/GeoTopo.pdf) | [   3   ](https://arxiv.org/pdf/2201.00151.pdf) | [   4   ](https://arxiv.org/pdf/1707.09725.pdf) | [   5   ](https://arxiv.org/pdf/2201.00021.pdf) | [   6   ](https://arxiv.org/pdf/2201.00037.pdf) | [   7   ](https://arxiv.org/pdf/2201.00069.pdf) | [   8   ](https://arxiv.org/pdf/2201.00178.pdf) | [   9   ](https://arxiv.org/pdf/2201.00201.pdf) | [  10   ](https://arxiv.org/pdf/1602.06541.pdf) | [  11   ](https://arxiv.org/pdf/2201.00200.pdf) | [  12   ](https://arxiv.org/pdf/2201.00022.pdf) | [  13   ](https://arxiv.org/pdf/2201.00029.pdf) | [  14   ](https://arxiv.org/pdf/1601.03642.pdf) |
| :- | :-------------------------------------------------------- | :------ | :---------------------------------------------- | :------------------------------------------------------------------------------------------ | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- |
| 1  | [PyMuPDF        ](https://pypi.org/project/PyMuPDF/)      |    0.1s | 0.4s                                            | 0.2s                                                                                        | 0.2s                                            | 0.2s                                            | 0.0s                                            | 0.1s                                            | 0.0s                                            | 0.0s                                            | 0.0s                                            | 0.1s                                            | 0.0s                                            | 0.0s                                            | 0.0s                                            | 0.0s                                            |
| 2  | [Tika           ](https://pypi.org/project/tika/)         |    0.2s | 1.0s                                            | 0.5s                                                                                        | 0.4s                                            | 0.4s                                            | 0.1s                                            | 0.2s                                            | 0.1s                                            | 0.1s                                            | 0.1s                                            | 0.1s                                            | 0.1s                                            | 0.1s                                            | 0.1s                                            | 0.0s                                            |
| 3  | [pypdfium2      ](https://pypi.org/project/pypdfium2/)    |    0.2s | 2.1s                                            | 0.3s                                                                                        | 0.2s                                            | 0.3s                                            | 0.0s                                            | 0.1s                                            | 0.1s                                            | 0.1s                                            | 0.0s                                            | 0.1s                                            | 0.0s                                            | 0.0s                                            | 0.0s                                            | 0.0s                                            |
| 4  | [pdftotext      ](https://poppler.freedesktop.org/)       |    0.2s | 0.7s                                            | 0.9s                                                                                        | 0.2s                                            | 0.7s                                            | 0.1s                                            | 0.2s                                            | 0.2s                                            | 0.1s                                            | 0.1s                                            | 0.1s                                            | 0.1s                                            | 0.1s                                            | 0.0s                                            | 0.0s                                            |
| 5  | [PyPDF2         ](https://pypi.org/project/PyPDF2/)       |    2.6s | 19.8s                                           | 4.5s                                                                                        | 5.4s                                            | 2.1s                                            | 0.6s                                            | 1.0s                                            | 0.4s                                            | 0.4s                                            | 0.3s                                            | 0.6s                                            | 0.5s                                            | 0.3s                                            | 0.4s                                            | 0.2s                                            |
| 6  | [pdfminer.six   ](https://pypi.org/project/pdfminer.six/) |    6.8s | 41.2s                                           | 20.1s                                                                                       | 11.2s                                           | 6.5s                                            | 2.1s                                            | 3.0s                                            | 1.8s                                            | 1.5s                                            | 1.1s                                            | 2.5s                                            | 1.3s                                            | 1.1s                                            | 0.8s                                            | 0.8s                                            |
| 7  | [pdfplumber     ](https://pypi.org/project/pdfplumber/)   |    7.9s | 54.8s                                           | 13.4s                                                                                       | 14.5s                                           | 8.0s                                            | 2.4s                                            | 4.0s                                            | 1.9s                                            | 1.8s                                            | 1.6s                                            | 2.4s                                            | 1.9s                                            | 1.5s                                            | 1.5s                                            | 0.9s                                            |
| 8  | [Borb           ](https://pypi.org/project/borb/)         |   62.9s | 208.6s                                          | 303.0s                                                                                      | 2.9s                                            | 111.3s                                          | 25.9s                                           | 29.5s                                           | 92.9s                                           | 26.5s                                           | 22.8s                                           | 10.7s                                           | 8.0s                                            | 28.6s                                           | 6.2s                                            | 3.5s                                            |


## Image Extraction Speed

| #  |                          Library                          | Average | [   1   ](https://arxiv.org/pdf/2201.00214.pdf) | [   2   ](https://github.com/py-pdf/sample-files/raw/main/009-pdflatex-geotopo/GeoTopo.pdf) | [   3   ](https://arxiv.org/pdf/2201.00151.pdf) | [   4   ](https://arxiv.org/pdf/1707.09725.pdf) | [   5   ](https://arxiv.org/pdf/2201.00021.pdf) | [   6   ](https://arxiv.org/pdf/2201.00037.pdf) | [   7   ](https://arxiv.org/pdf/2201.00069.pdf) | [   8   ](https://arxiv.org/pdf/2201.00178.pdf) | [   9   ](https://arxiv.org/pdf/2201.00201.pdf) | [  10   ](https://arxiv.org/pdf/1602.06541.pdf) | [  11   ](https://arxiv.org/pdf/2201.00200.pdf) | [  12   ](https://arxiv.org/pdf/2201.00022.pdf) | [  13   ](https://arxiv.org/pdf/2201.00029.pdf) | [  14   ](https://arxiv.org/pdf/1601.03642.pdf) |
| :- | :-------------------------------------------------------- | :------ | :---------------------------------------------- | :------------------------------------------------------------------------------------------ | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- |
| 1  | [PyMuPDF        ](https://pypi.org/project/PyMuPDF/)      |    0.6s | 0.4s                                            | 0.9s                                                                                        | 0.0s                                            | 2.0s                                            | 0.6s                                            | 0.0s                                            | 3.2s                                            | 0.4s                                            | 0.4s                                            | 0.3s                                            | 0.0s                                            | 0.3s                                            | 0.2s                                            | 0.0s                                            |
| 2  | [PyPDF2         ](https://pypi.org/project/PyPDF2/)       |    1.0s | 0.4s                                            | 1.5s                                                                                        | 0.0s                                            | 3.6s                                            | 0.9s                                            | 0.0s                                            | 5.7s                                            | 0.7s                                            | 0.7s                                            | 0.2s                                            | 0.0s                                            | 0.5s                                            | 0.0s                                            | 0.0s                                            |
| 3  | [pdfminer.six   ](https://pypi.org/project/pdfminer.six/) |    9.1s | 48.6s                                           | 18.3s                                                                                       | 12.7s                                           | 30.8s                                           | 1.9s                                            | 3.5s                                            | 1.9s                                            | 2.1s                                            | 1.4s                                            | 2.0s                                            | 1.6s                                            | 1.6s                                            | 0.8s                                            | 0.7s                                            |


## Watermarking Speed

| #  |                       Library                       | Average | [   1   ](https://arxiv.org/pdf/2201.00214.pdf) | [   2   ](https://github.com/py-pdf/sample-files/raw/main/009-pdflatex-geotopo/GeoTopo.pdf) | [   3   ](https://arxiv.org/pdf/2201.00151.pdf) | [   4   ](https://arxiv.org/pdf/1707.09725.pdf) | [   5   ](https://arxiv.org/pdf/2201.00021.pdf) | [   6   ](https://arxiv.org/pdf/2201.00037.pdf) | [   7   ](https://arxiv.org/pdf/2201.00069.pdf) | [   8   ](https://arxiv.org/pdf/2201.00178.pdf) | [   9   ](https://arxiv.org/pdf/2201.00201.pdf) | [  10   ](https://arxiv.org/pdf/1602.06541.pdf) | [  11   ](https://arxiv.org/pdf/2201.00200.pdf) | [  12   ](https://arxiv.org/pdf/2201.00022.pdf) | [  13   ](https://arxiv.org/pdf/2201.00029.pdf) | [  14   ](https://arxiv.org/pdf/1601.03642.pdf) |
| :- | :-------------------------------------------------- | :------ | :---------------------------------------------- | :------------------------------------------------------------------------------------------ | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- |
| 1  | [PyPDF2         ](https://pypi.org/project/PyPDF2/) |    6.1s | 19.8s                                           | 4.5s                                                                                        | 5.4s                                            | 2.1s                                            | 0.6s                                            | 1.0s                                            | 0.4s                                            | 0.4s                                            | 0.3s                                            | 0.6s                                            | 0.5s                                            | 0.3s                                            | 0.4s                                            | 0.2s                                            |

## Text Extraction Quality

| #  |                          Library                          | Average | [   1   ](https://arxiv.org/pdf/2201.00214.pdf) | [   2   ](https://github.com/py-pdf/sample-files/raw/main/009-pdflatex-geotopo/GeoTopo.pdf) | [   3   ](https://arxiv.org/pdf/2201.00151.pdf) | [   4   ](https://arxiv.org/pdf/1707.09725.pdf) | [   5   ](https://arxiv.org/pdf/2201.00021.pdf) | [   6   ](https://arxiv.org/pdf/2201.00037.pdf) | [   7   ](https://arxiv.org/pdf/2201.00069.pdf) | [   8   ](https://arxiv.org/pdf/2201.00178.pdf) | [   9   ](https://arxiv.org/pdf/2201.00201.pdf) | [  10   ](https://arxiv.org/pdf/1602.06541.pdf) | [  11   ](https://arxiv.org/pdf/2201.00200.pdf) | [  12   ](https://arxiv.org/pdf/2201.00022.pdf) | [  13   ](https://arxiv.org/pdf/2201.00029.pdf) | [  14   ](https://arxiv.org/pdf/1601.03642.pdf) |
| :- | :-------------------------------------------------------- | :------ | :---------------------------------------------- | :------------------------------------------------------------------------------------------ | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------- |
| 1  | [pypdfium2      ](https://pypi.org/project/pypdfium2/)    |  98%    |  99%                                            |  97%                                                                                        |  95%                                            |  97%                                            |  98%                                            |  96%                                            |  99%                                            |  96%                                            |  99%                                            |  99%                                            |  98%                                            |  98%                                            |  99%                                            |  99%                                            |
| 2  | [PyMuPDF        ](https://pypi.org/project/PyMuPDF/)      |  97%    |  98%                                            |  97%                                                                                        |  94%                                            |  95%                                            |  98%                                            |  96%                                            |  99%                                            |  95%                                            |  99%                                            |  98%                                            |  98%                                            |  98%                                            |  98%                                            |  99%                                            |
| 3  | [Tika           ](https://pypi.org/project/tika/)         |  97%    |  99%                                            |  99%                                                                                        |  94%                                            |  99%                                            |  98%                                            |  97%                                            |  94%                                            |  99%                                            |  99%                                            |  93%                                            |  98%                                            |  94%                                            |  98%                                            |  96%                                            |
| 4  | [PyPDF2         ](https://pypi.org/project/PyPDF2/)       |  96%    |  98%                                            |  87%                                                                                        |  94%                                            |  94%                                            |  97%                                            |  94%                                            |  96%                                            |  93%                                            |  98%                                            |  98%                                            |  97%                                            |  97%                                            |  98%                                            |  99%                                            |
| 5  | [pdftotext      ](https://poppler.freedesktop.org/)       |  93%    |  96%                                            |  93%                                                                                        |  91%                                            |  92%                                            |  92%                                            |  96%                                            |  96%                                            |  94%                                            |  97%                                            |  83%                                            |  94%                                            |  97%                                            |  97%                                            |  79%                                            |
| 6  | [pdfminer.six   ](https://pypi.org/project/pdfminer.six/) |  90%    |  95%                                            |  79%                                                                                        |  87%                                            |  90%                                            |  86%                                            |  94%                                            |  96%                                            |  91%                                            |  92%                                            |  92%                                            |  94%                                            |  86%                                            |  98%                                            |  86%                                            |
| 7  | [pdfplumber     ](https://pypi.org/project/pdfplumber/)   |  74%    |  93%                                            |  84%                                                                                        |  61%                                            |  94%                                            |  61%                                            |  93%                                            |  61%                                            |  86%                                            |  57%                                            |  59%                                            |  67%                                            |  59%                                            |  97%                                            |  67%                                            |
| 8  | [Borb           ](https://pypi.org/project/borb/)         |  53%    |  72%                                            |  86%                                                                                        |   0%                                            |  40%                                            |  67%                                            |  94%                                            |   0%                                            |  62%                                            |  69%                                            |  56%                                            |  75%                                            |  52%                                            |   0%                                            |  64%                                            |
