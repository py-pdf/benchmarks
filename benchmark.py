"""
Compare text extraction performance of different PDF parsers.
"""

import json
import time
from importlib.metadata import version as pkg_version
from io import BytesIO
from itertools import product
from json import JSONDecodeError
from pathlib import Path
from typing import Literal

import pdfminer
import pdfplumber
import pdfrw
import pymupdf as PyMuPDF
import pypdf
import tika
from pdfminer.high_level import extract_text as pdfminder_extract_text
from rich.progress import track

from pdf_benchmark.data_structures import Cache, Document, Library
from pdf_benchmark.library_code import (
    pdfium_get_text,
    pdfium_image_extraction,
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
    tika_get_text,
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
        try:
            with open(cache_path) as f:
                cache = Cache.model_validate(json.load(f))
        except JSONDecodeError:
            cache = Cache()
    else:
        cache = Cache()
    names = sorted(libraries.keys())

    watermark_file = Path(__file__).parent / "watermark" / "pdfs" / "python-quote.pdf"
    watermark_data = watermark_file.read_bytes()

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

    libraries = {
        name: lib._replace(last_release_date=cache.resolve_release_date(lib))
        for name, lib in libraries.items()
    }
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
    folder = Path(benchmark) / "results" / pdf_library_name
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(data, list):
        folder = folder / pdf_file_name
        folder.mkdir(parents=True, exist_ok=True)
        for image_name, image_data in data:
            (folder / image_name).write_bytes(image_data)
    else:
        mode = "wb" if extension == "pdf" or isinstance(data, bytes) else "w"
        with open(folder / f"{pdf_file_name}.{extension}", mode) as f:
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
            text_extraction_function=tika_get_text,
            version=tika.__version__,
            dependencies="Apache Tika",
            license="Apache v2",
            last_release_date="2026-08-01",
            pypi_name="tika",
        ),
        "pypdf": Library(
            "pypdf",
            "pypdf",
            "https://pypi.org/project/pypdf/",
            text_extraction_function=pypdf_get_text,
            version=pypdf.__version__,
            watermarking_function=pypdf_watermarking,
            license="BSD 3-Clause",
            last_release_date="2025-06-29",
            image_extraction_function=pypdf_image_extraction,
            pypi_name="pypdf",
        ),
        "pdfminer": Library(
            "pdfminer.six",
            "pdfminer",
            "https://pypi.org/project/pdfminer.six/",
            text_extraction_function=lambda n: pdfminder_extract_text(BytesIO(n)),
            version=pdfminer.__version__,
            license="MIT/X",
            last_release_date="2025-05-06",
            image_extraction_function=pdfminer_image_extraction,
            pypi_name="pdfminer.six",
        ),
        "pdfplumber": Library(
            "pdfplumber",
            "pdfplumber",
            "https://pypi.org/project/pdfplumber/",
            text_extraction_function=pdfplubmer_get_text,
            version=pdfplumber.__version__,
            license="MIT",
            last_release_date="2025-06-12",
            dependencies="pdfminer.six",
            pypi_name="pdfplumber",
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
            last_release_date="2025-06-12",
            pypi_name="pymupdf",
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
        "pdfium": Library(
            "pypdfium2",
            "pdfium",
            "https://pypi.org/project/pypdfium2/",
            text_extraction_function=pdfium_get_text,
            version=pkg_version("pypdfium2"),
            watermarking_function=None,
            image_extraction_function=pdfium_image_extraction,
            license="Apache-2.0 or BSD-3-Clause",
            last_release_date="2024-12-19",
            dependencies="PDFium (Foxit/Google)",
            pypi_name="pypdfium2",
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
            pypi_name="pdfrw",
        ),
    }
    main(docs, libraries)
