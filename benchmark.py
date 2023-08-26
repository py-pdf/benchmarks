"""
Compare text extraction performance of different PDF parsers.
"""

import json
import os
import time
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import Literal

import fitz as PyMuPDF
import pdfminer
import pdfplumber
import pdfrw
import pypdf
import pypdfium2
import tika
from pdfminer.high_level import extract_text as pdfminder_extract_text
from rich.progress import track
from tika import parser

from pdf_benchmark.data_structures import Cache, Document, Library
from pdf_benchmark.library_code import (
    borb_get_text,
    pdfium_get_text,
    pdfminer_image_extraction,
    pdfplubmer_get_text,
    pdfrw_watermarking,
    pdftotext_get_text,
    pymupdf_get_text,
    pymupdf_image_extraction,
    pymupdf_watermarking,
    pypdf_get_text,
    pypdf_image_extraction,
    pypdf_watermarking,
)
from pdf_benchmark.output import write_benchmark_report
from pdf_benchmark.score import get_text_extraction_score

tika.initVM()


def main(
    docs: list[Document],
    libraries: dict[str, Library],
) -> None:
    cache_path = Path("cache.json")
    if cache_path.exists():
        with open(cache_path) as f:
            cache = Cache.parse_obj(json.load(f))
    else:
        cache = Cache()
    names = sorted(list(libraries.keys()))

    watermark_file = os.path.join(
        os.path.dirname(__file__), "watermark", "pdfs", "python-quote.pdf"
    )
    with open(watermark_file, "rb") as f:
        watermark_data = f.read()

    # Run the benchmarks
    for doc, name in track(list(product(docs, names))):
        data = doc.data
        lib = libraries[name]
        if cache.has_doc(lib, doc):
            print(f"Skip {doc.name} for {lib.name}")
            continue
        if lib.text_extraction_function:
            print(f"{name} now parses {doc.name}...")
            t0 = time.time()
            text = lib.text_extraction_function(data)
            t1 = time.time()
            write_single_result("read", name, doc.name, text, "txt")
            cache.benchmark_times[lib.pathname][doc.name]["read"] = t1 - t0
            cache.read_quality[lib.pathname][doc.name] = get_text_extraction_score(
                doc, lib.pathname
            )
        if lib.watermarking_function:
            t0 = time.time()
            watermarked = lib.watermarking_function(watermark_data, data)
            t1 = time.time()
            write_single_result("watermark", name, doc.name, watermarked, "pdf")
            cache.benchmark_times[lib.pathname][doc.name]["watermark"] = t1 - t0
            cache.watermarking_result_file_size[lib.pathname][doc.name] = len(
                watermarked
            )
        if lib.image_extraction_function:
            t0 = time.time()
            extracted_images = lib.image_extraction_function(data)
            t1 = time.time()
            write_single_result(
                "image_extraction", name, doc.name, extracted_images, "image-list"
            )
            cache.benchmark_times[lib.pathname][doc.name]["image_extraction"] = t1 - t0
        cache.write(cache_path)
    write_benchmark_report(
        names,
        libraries,
        docs,
        cache,
    )


def write_single_result(
    benchmark: Literal["read", "watermark", "image_extraction"],
    pdf_library_name: str,
    pdf_file_name: str,
    data: str | bytes | list[tuple[str, bytes]],
    extension: Literal["txt", "pdf", "image-list"],
) -> None:
    folder = f"{benchmark}/results/{pdf_library_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    if isinstance(data, list):
        folder = f"{folder}/{pdf_file_name}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        for image_name, image_data in data:
            with open(f"{folder}/{image_name}", "wb") as fp:
                fp.write(image_data)
    else:
        mode = "wb" if extension == "pdf" or isinstance(data, bytes) else "w"
        with open(f"{folder}/{pdf_file_name}.{extension}", mode) as f:
            try:
                f.write(data)
            except Exception as exc:
                print(exc)


if __name__ == "__main__":
    docs = [
        Document(name="2201.00214", url="https://arxiv.org/pdf/2201.00214.pdf"),
        Document(
            name="GeoTopo-book",
            url="https://github.com/py-pdf/sample-files/raw/main/009-pdflatex-geotopo/GeoTopo.pdf",
        ),
        Document(name="2201.00151", url="https://arxiv.org/pdf/2201.00151.pdf"),
        Document(name="1707.09725", url="https://arxiv.org/pdf/1707.09725.pdf"),
        Document(name="2201.00021", url="https://arxiv.org/pdf/2201.00021.pdf"),
        Document(name="2201.00037", url="https://arxiv.org/pdf/2201.00037.pdf"),
        Document(name="2201.00069", url="https://arxiv.org/pdf/2201.00069.pdf"),
        Document(name="2201.00178", url="https://arxiv.org/pdf/2201.00178.pdf"),
        Document(name="2201.00201", url="https://arxiv.org/pdf/2201.00201.pdf"),
        Document(name="1602.06541", url="https://arxiv.org/pdf/1602.06541.pdf"),
        Document(name="2201.00200", url="https://arxiv.org/pdf/2201.00200.pdf"),
        Document(name="2201.00022", url="https://arxiv.org/pdf/2201.00022.pdf"),
        Document(name="2201.00029", url="https://arxiv.org/pdf/2201.00029.pdf"),
        Document(name="1601.03642", url="https://arxiv.org/pdf/1601.03642.pdf"),
    ]
    libraries = {
        "tika": Library(
            "Tika",
            "tika",
            "https://pypi.org/project/tika/",
            text_extraction_function=lambda n: parser.from_buffer(BytesIO(n))[
                "content"
            ],
            version=tika.__version__,
            dependencies="Apache Tika",
            license="Apache v2",
            last_release_date="2023-01-01",
        ),
        "pypdf": Library(
            "pypdf",
            "pypdf",
            "https://pypi.org/project/pypdf/",
            text_extraction_function=pypdf_get_text,
            version=pypdf.__version__,
            watermarking_function=pypdf_watermarking,
            license="BSD 3-Clause",
            last_release_date="2023-08-26",
            image_extraction_function=pypdf_image_extraction,
        ),
        "pdfminer": Library(
            "pdfminer.six",
            "pdfminer",
            "https://pypi.org/project/pdfminer.six/",
            text_extraction_function=lambda n: pdfminder_extract_text(BytesIO(n)),
            version=pdfminer.__version__,
            license="MIT/X",
            last_release_date="2022-11-05",
            image_extraction_function=pdfminer_image_extraction,
        ),
        "pdfplumber": Library(
            "pdfplumber",
            "pdfplumber",
            "https://pypi.org/project/pdfplumber/",
            text_extraction_function=pdfplubmer_get_text,
            version=pdfplumber.__version__,
            license="MIT",
            last_release_date="2023-07-29",
            dependencies="pdfminer.six",
        ),
        "pymupdf": Library(
            "PyMuPDF",
            "pymupdf",
            "https://pypi.org/project/PyMuPDF/",
            text_extraction_function=lambda n: pymupdf_get_text(n),
            version=PyMuPDF.version[0],
            watermarking_function=pymupdf_watermarking,
            image_extraction_function=pymupdf_image_extraction,
            dependencies="MuPDF",
            license="GNU AFFERO GPL 3.0 / Commerical",
            last_release_date="2023-08-24",
        ),
        "pdftotext": Library(
            "pdftotext",
            "pdftotext",
            "https://poppler.freedesktop.org/",
            text_extraction_function=pdftotext_get_text,
            version="0.86.1",
            watermarking_function=None,
            dependencies="build-essential libpoppler-cpp-dev pkg-config python3-dev",
            last_release_date="-",
            license="GPL",
        ),
        "borb": Library(
            "Borb",
            "borb",
            "https://pypi.org/project/borb/",
            text_extraction_function=borb_get_text,
            version="2.1.16",
            watermarking_function=None,
            license="AGPL/Commercial",
            last_release_date="2023-06-23",
        ),
        "pdfium": Library(
            "pypdfium2",
            "pdfium",
            "https://pypi.org/project/pypdfium2/",
            text_extraction_function=pdfium_get_text,
            version=pypdfium2.V_PYPDFIUM2,
            watermarking_function=None,
            license="Apache-2.0 or BSD-3-Clause",
            last_release_date="2023-07-04",
            dependencies="PDFium (Foxit/Google)",
        ),
        "pdfrw": Library(
            "pdfrw",
            "pdfrw",
            "https://pypi.org/project/pdfrw/",
            text_extraction_function=None,
            version=pdfrw.__version__,
            watermarking_function=pdfrw_watermarking,
            license="MIT",
            last_release_date="2017-09-18",
            dependencies="",
        ),
    }
    main(docs, libraries)
