"""
Make the measured times less dependent on the machine.

Every measurement is divided by the time of a fixed workload measured right
before it, and multiplied by the time of that workload on the reference
machine. The report then shows the seconds the reference machine would need.
This also makes results of different runs comparable, as the cache keeps the
results of a library until its version changes.

The workload is pure Python, like pypdf, pdfminer.six and pdfplumber.
PyMuPDF, pypdfium2, pdftotext and Tika are written in C, C++ and Java, so
their times follow the speed of the machine less closely, and a faster
Python makes them look slower.

The workload is the same as in pypdf's benchmarks (tests/bench.py).
"""

import re
import zlib
from io import BytesIO

# .python-version pins the Python version to the one of the reference machine
REFERENCE_MACHINE = "Intel Core i7-6700HQ CPU @ 2.60GHz, CPython 3.14.7"
# The shortest time of three runs of calibration_workload() on the reference
# machine. Never change it without measuring all libraries again.
REFERENCE_SECONDS = 0.115

CALIBRATION_DATA = b"".join(
    b"/Key%d %d 0 R [%d %d.5 (text %d)] << /Type /Page >>\n" % (i, i, i, i, i)
    for i in range(5000)
)


def calibration_workload() -> int:
    """
    Do pypdf-like work without pypdf: tokenizing bytes, dictionaries, zlib,
    regular expressions.

    Never change this function, as all published results are relative to it.
    """
    raw = zlib.decompress(zlib.compress(CALIBRATION_DATA))
    stream = BytesIO(raw)
    tokens = []
    current = []
    while True:
        char = stream.read(1)
        if not char:
            break
        if char in b" \n[]()<>/":
            if current:
                tokens.append(b"".join(current))
                current = []
        else:
            current.append(char)
    counts: dict[bytes, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    numbers = [int(token) for token in tokens if token.isdigit()]
    names = re.findall(rb"/(\w+)", raw)
    text = "".join(token.decode("latin-1") for token in tokens)
    return len(counts) + sum(numbers) + len(names) + len(text)


def to_reference_seconds(seconds: float, calibration_seconds: float) -> float:
    """Return the seconds the reference machine would need for a measurement."""
    return seconds / calibration_seconds * REFERENCE_SECONDS
