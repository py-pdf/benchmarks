# Table Extraction

1. Table format: Pandas dataframe
2. Assume we have perfect table detection?

https://github.com/py-pdf/pypdf/issues/1395

## Tabula

```python
from tabula import read_pdf

tables = read_pdf("missing_newlines.pdf", pages=7)
```
