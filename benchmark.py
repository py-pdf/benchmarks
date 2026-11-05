"""
Compare text extraction performance of different PDF parsers.
"""

import json
import math
import re
import subprocess
import time
from collections.abc import Callable
from importlib.metadata import version as pkg_version
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
from rich.progress import track

from pdf_benchmark.data_structures import Cache, Document, Library
from pdf_benchmark.library_code import (
    pdfium_get_text,
    pdfium_image_extraction,
    pdfminer_get_text,
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
from pdf_benchmark.text_extraction_post_processing import postprocess

# Every measurement is repeated, and the shortest time is reported: the
# slower runs only add the noise of other processes.
REPETITIONS = 3

# Pinned to a commit, so that the files cannot change with the main branch
SAMPLE_FILES = "https://github.com/py-pdf/sample-files/raw/80d67b91f3e28909756ca7807bbe8f43b599d12f"


def measure[T](function: Callable[..., T], *args: bytes) -> tuple[T, float]:
    """
    Return the result of the function and the shortest time of REPETITIONS runs.

    This is the time on the wall clock, as pdftotext and Tika work in another
    process.
    """
    best = math.inf
    for _ in range(REPETITIONS):
        t0 = time.perf_counter()
        result = function(*args)
        best = min(best, time.perf_counter() - t0)
    return result, best


def warm_up_tika(data: bytes) -> None:
    """
    Start the Tika server and let the JVM warm up before timing anything.

    tika-python starts the Java server lazily on the first request and keeps
    it running after this process exits. Without this, the first run of the
    benchmark would include server startup and JIT warm-up in Tika's timings.
    """
    print("Warming up Tika server...")
    tika_get_text(data)


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
    for lib in libraries.values():
        cache.prepare_results(lib)
    if "tika" in names and docs:
        warm_up_tika(docs[0].data)

    watermark_file = Path(__file__).parent / "watermark" / "pdfs" / "python-quote.pdf"
    watermark_data = watermark_file.read_bytes()

    # Run the benchmarks
    for doc, name in track(list(product(docs, names))):
        data = doc.data
        lib = libraries[name]
        if cache.has_doc(lib, doc):
            print(f"Skip {doc.name} for {lib.name}")
            continue
        cache.benchmark_times[lib.pathname][doc.name] = {}
        if lib.text_extraction_function:
            print(f"{name} now parses {doc.name}...")
            pages, seconds = measure(lib.text_extraction_function, data)
            text = postprocess_pages(pages, doc, name)
            write_single_result("read", name, doc.name, text, "txt")
            cache.benchmark_times[lib.pathname][doc.name]["read"] = seconds
            cache.read_quality[lib.pathname][doc.name] = get_text_extraction_score(
                doc, lib.pathname
            )
        if lib.watermarking_function:
            watermarked, seconds = measure(
                lib.watermarking_function, watermark_data, data
            )
            write_single_result("watermark", name, doc.name, watermarked, "pdf")
            cache.benchmark_times[lib.pathname][doc.name]["watermark"] = seconds
            cache.watermarking_result_file_size[lib.pathname][doc.name] = len(
                watermarked
            )
        if lib.image_extraction_function:
            extracted_images, seconds = measure(lib.image_extraction_function, data)
            write_single_result(
                "image_extraction", name, doc.name, extracted_images, "image-list"
            )
            cache.benchmark_times[lib.pathname][doc.name]["image_extraction"] = seconds
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


def get_pdftotext_version() -> str:
    """Return the Poppler version of the pdftotext command."""
    output = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True)
    match = re.search(r"version (\S+)", output.stderr)
    return match.group(1) if match else ""


def postprocess_pages(pages: list[str], doc: Document, library_name: str) -> str:
    """Post-process the pages of every library in the same way, see postprocess()."""
    page_labels = doc.page_labels
    if len(pages) != len(page_labels):
        print(
            f"{library_name} returned {len(pages)} pages for {doc.name},"
            f" which has {len(page_labels)}"
        )
        return "\n".join(pages)
    return postprocess(pages, page_labels)


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
        Document(
            name="2201.00214",
            url="https://arxiv.org/pdf/2201.00214v1",
            sha256="9ea7ed2fd1e4eb364ee96c089a61c71f29613c3768cdb7c41daf7314ab62ae91",
        ),
        Document(
            name="GeoTopo-book",
            url=f"{SAMPLE_FILES}/009-pdflatex-geotopo/GeoTopo.pdf",
            sha256="a0b9242aef9b5a2c4850160cc9aaf64f5c7b8be705d6872ae28aebf5ab512540",
        ),
        Document(
            name="2201.00151",
            url="https://arxiv.org/pdf/2201.00151v1",
            sha256="886051363eeb8c219bff72cf46437b2f5787bbe7f22ecbce058515549e15aed9",
        ),
        Document(
            name="1707.09725",
            url="https://arxiv.org/pdf/1707.09725v1",
            sha256="15d73fe4d8abe5122d547118bdb0accb930737a7c656c9228ad4bee5a5753532",
        ),
        Document(
            name="2201.00021",
            url="https://arxiv.org/pdf/2201.00021v3",
            sha256="984ace89eb23ff4cb9cd6705a191a3eeea267a6ef42d05c782a10ca97d0f579a",
        ),
        Document(
            name="2201.00037",
            url="https://arxiv.org/pdf/2201.00037v1",
            sha256="a1a62e45b4e3f3c89e2a0336de29631afa7e9e712d3a268f14f3ad5e35284a20",
        ),
        Document(
            name="2201.00069",
            url="https://arxiv.org/pdf/2201.00069v1",
            sha256="51ff3326533a385e57506bbcb142cda45e2cdcfa170f77c9c6f512c9e1dddbf3",
        ),
        Document(
            name="2201.00178",
            url="https://arxiv.org/pdf/2201.00178v1",
            sha256="f30563de6c6973cfc6aa4670113d52855aadba7ddb006011a9978ca791f411f0",
        ),
        Document(
            name="2201.00201",
            url="https://arxiv.org/pdf/2201.00201v2",
            sha256="095227891d4fcb6190f74a133c4895af73d2cedcf3676c8292fdf5915e46ed47",
        ),
        Document(
            name="1602.06541",
            url="https://arxiv.org/pdf/1602.06541v2",
            sha256="8618a96fc965a31e843e6b96709e68bcbe2a825f32dc6b2ca67a4e60d38614c3",
        ),
        Document(
            name="2201.00200",
            url="https://arxiv.org/pdf/2201.00200v1",
            sha256="edce7ae9870ad1608384ae40f86691566b8ce0013685496e3e4d1fda992dcac8",
        ),
        Document(
            name="2201.00022",
            url="https://arxiv.org/pdf/2201.00022v2",
            sha256="c64924d5856c83bbd712aeec09cb7bbe6e4e71ef484eb6f8013aceccec015fa2",
        ),
        Document(
            name="2201.00029",
            url="https://arxiv.org/pdf/2201.00029v1",
            sha256="ce6b2e656e9981ee0f4497f6638407370e603d0f7395c65e4431d3b7ddfca024",
        ),
        Document(
            name="1601.03642",
            url="https://arxiv.org/pdf/1601.03642v1",
            sha256="c18ea7dc2e8d075f22ca0030e83000adc05cd4f75218c1d774564aa91b5edd94",
        ),
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
            text_extraction_function=pdfminer_get_text,
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
            text_extraction_function=pymupdf_get_text,
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
            version=get_pdftotext_version(),
            watermarking_function=None,
            dependencies="Poppler (poppler-utils)",
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
