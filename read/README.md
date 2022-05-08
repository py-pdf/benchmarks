# PDF Read Benchmark

This benachmark is about reading pure PDF files - not
scanned documents and not documents that applied OCR.

## Benchmarking machine
Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz

## Input Documents
1. [2201.00021](https://arxiv.org/pdf/2201.00021.pdf)
2. [2201.00022](https://arxiv.org/pdf/2201.00022.pdf)
3. [2201.00029](https://arxiv.org/pdf/2201.00029.pdf)
4. [2201.00037](https://arxiv.org/pdf/2201.00037.pdf)
5. [2201.00069](https://arxiv.org/pdf/2201.00069.pdf)
6. [2201.00151](https://arxiv.org/pdf/2201.00151.pdf)
7. [2201.00178](https://arxiv.org/pdf/2201.00178.pdf)
8. [2201.00200](https://arxiv.org/pdf/2201.00200.pdf)
9. [2201.00201](https://arxiv.org/pdf/2201.00201.pdf)
10. [2201.00214](https://arxiv.org/pdf/2201.00214.pdf)
11. [1707.09725](https://arxiv.org/pdf/1707.09725.pdf)
12. [1601.03642](https://arxiv.org/pdf/1601.03642.pdf)
13. [1602.06541](https://arxiv.org/pdf/1602.06541.pdf)

## Libraries
* pdfminer.six: 20220319
* pdfplumber: 0.6.2
* PyMuPDF: 1.19.6 (depends on MuPDF)
* PyPDF2: 1.27.12
* Tika: 1.24

## Text Extraction Speed

| # | Library                                                  | Avgerage |  [   1   ](https://arxiv.org/pdf/2201.00021.pdf)|[   2   ](https://arxiv.org/pdf/2201.00022.pdf)|[   3   ](https://arxiv.org/pdf/2201.00029.pdf)|[   4   ](https://arxiv.org/pdf/2201.00037.pdf)|[   5   ](https://arxiv.org/pdf/2201.00069.pdf)|[   6   ](https://arxiv.org/pdf/2201.00151.pdf)|[   7   ](https://arxiv.org/pdf/2201.00178.pdf)|[   8   ](https://arxiv.org/pdf/2201.00200.pdf)|[   9   ](https://arxiv.org/pdf/2201.00201.pdf)|[  10   ](https://arxiv.org/pdf/2201.00214.pdf)|[  11   ](https://arxiv.org/pdf/1707.09725.pdf)|[  12   ](https://arxiv.org/pdf/1601.03642.pdf)|[  13   ](https://arxiv.org/pdf/1602.06541.pdf)
|---|----------------------------------------------------------|----------|---------|------|------|------|------|------|------|------|------|------|------|------|------
| 1 | [PyMuPDF        ](https://pypi.org/project/PyMuPDF/)     |   0.11s  |   0.05s|  0.07s|  0.02s|  0.09s|  0.04s|  0.17s|  0.06s|  0.03s|  0.03s|  0.56s|  0.22s|  0.03s|  0.07s
| 2 | [Tika           ](https://pypi.org/project/tika/)        |   0.24s  |   0.09s|  0.07s|  0.08s|  0.16s|  0.13s|  0.39s|  0.12s|  0.06s|  0.09s|  1.28s|  0.44s|  0.08s|  0.12s
| 3 | [PyPDF2         ](https://pypi.org/project/PyPDF2/)      |   3.32s  |   0.24s|  0.28s|  0.00s|  0.61s|  0.35s|  7.02s|  0.38s|  0.16s|  0.26s| 30.66s|  2.34s|  0.21s|  0.59s
| 4 | [pdfminer.six   ](https://pypi.org/project/pdfminer.six/)|   6.88s  |   1.61s|  1.10s|  0.74s|  2.93s|  1.56s| 13.12s|  1.83s|  1.44s|  1.20s| 50.95s|  9.30s|  1.05s|  2.59s
| 5 | [pdfplumber     ](https://pypi.org/project/pdfplumber/)  |   8.80s  |   2.04s|  1.22s|  0.99s|  3.73s|  1.75s| 14.95s|  1.94s|  1.84s|  1.49s| 71.04s|  9.63s|  1.15s|  2.62s

## Text Extraction Quality

| # | Library        |  Avgerage |  [   1   ](https://arxiv.org/pdf/2201.00021.pdf)|[   2   ](https://arxiv.org/pdf/2201.00022.pdf)|[   3   ](https://arxiv.org/pdf/2201.00029.pdf)|[   4   ](https://arxiv.org/pdf/2201.00037.pdf)|[   5   ](https://arxiv.org/pdf/2201.00069.pdf)|[   6   ](https://arxiv.org/pdf/2201.00151.pdf)|[   7   ](https://arxiv.org/pdf/2201.00178.pdf)|[   8   ](https://arxiv.org/pdf/2201.00200.pdf)|[   9   ](https://arxiv.org/pdf/2201.00201.pdf)|[  10   ](https://arxiv.org/pdf/2201.00214.pdf)|[  11   ](https://arxiv.org/pdf/1707.09725.pdf)|[  12   ](https://arxiv.org/pdf/1601.03642.pdf)|[  13   ](https://arxiv.org/pdf/1602.06541.pdf)
|---|----------------|---|---------|------|------|------|------|------|------|------|------|------|------|------|------
| 1 | [Tika           ](https://pypi.org/project/tika/) |   1% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100%
| 2 | [PyMuPDF        ](https://pypi.org/project/PyMuPDF/) |   1% |  98% |  94% |  99% |  97% |  93% |  97% |  95% |  98% |  99% |  97% |  95% |  95% |  92%
| 3 | [pdfminer.six   ](https://pypi.org/project/pdfminer.six/) |   1% |  86% |  83% |  99% |  94% |  90% |  89% |  90% |  94% |  92% |  95% |  90% |  83% |  87%
| 4 | [PyPDF2         ](https://pypi.org/project/PyPDF2/) |   1% |  95% |  91% |   0% |  92% |  90% |  89% |  89% |  97% |  97% |  89% |  91% |  94% |  90%
| 5 | [pdfplumber     ](https://pypi.org/project/pdfplumber/) |   1% |  61% |  57% |  98% |  94% |  58% |  61% |  86% |  67% |  57% |  92% |  94% |  65% |  55%
