"""
Compare text extraction performance of different PDF parsers.
"""

import os
import platform
import re
import subprocess
import time
from io import BytesIO
from itertools import product
from typing import Callable, Dict, List, NamedTuple

import fitz as PyMuPDF
import numpy as np
import pdfminer
import pdfplumber
import PyPDF2
import requests
import tika
from Levenshtein import ratio  # python-Levenshtein
from pdfminer.high_level import extract_text
from rich.progress import track
from tika import parser


def get_processor_name():
    """Credits: https://stackoverflow.com/a/13078519/562769"""
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
    return ""


class Library(NamedTuple):
    name: str
    url: str
    extraction_function: Callable[[bytes], str]
    version: str
    dependencies: str = ""


def pymupdf_get_text(data: bytes) -> str:
    with PyMuPDF.open(stream=data, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


def pypdf2_get_text(data: bytes) -> str:
    text = ""
    reader = PyPDF2.PdfFileReader(BytesIO(data))
    for i in range(reader.getNumPages()):
        page = reader.getPage(i)
        text += page.extractText()
    return text


def pdfplubmer_get_text(data: bytes) -> str:
    text = ""
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
            text += "\n"
    return text


def get_pdf_bytes(doi: str) -> bytes:
    destination = f"pdfs/{doi}.pdf"
    if not os.path.exists(destination):
        if not os.path.exists("pdfs"):
            os.makedirs("pdfs")
        url = f"https://arxiv.org/pdf/{doi}.pdf"
        print(f"Now {url} ...")
        response = requests.get(url)
        with open(destination, "wb") as f:
            f.write(response.content)
    with open(destination, "rb") as f:
        return f.read()


def main(extract_functions: Dict[str, Library]) -> None:
    names = sorted(list(extract_functions.keys()))
    times_all: Dict[str, List[float]] = {name: [] for name in names}

    DOIs = [
        "2201.00021",  #  2.7 MB: 10 pages
        "2201.00022",  #  1.1 MB: 11 pages
        "2201.00029",  #  0.8 MB: 12 pages
        "2201.00037",  #  3.1 MB: 33 pages
        "2201.00069",  # 15.4 MB: 15 pages
        "2201.00151",  #  1.6 MB: 12 pages
        "2201.00178",  #  2.4 MB: 16 pages
        "2201.00200",  #  0.3 MB:  7 pages,
        "2201.00201",  #  1.3 MB:  9 pages,
        "2201.00214",  #  2.5 MB: 22 pages
        "1707.09725",  #  7.3 MB: 73 pages, 39 figures
        "1601.03642",  #  1.0 MB:  5 pages, 4 figures
        "1602.06541",  #  3.1 MB: 16 pages
    ]

    for doi, name in track(list(product(DOIs, names))):
        data = get_pdf_bytes(doi)
        print(f"{name} now parses {doi}...")
        t0 = time.time()
        text = extract_functions[name].extraction_function(data)
        t1 = time.time()
        times_all[name].append(t1 - t0)
        write_single_result(name, doi, text)
    write_benchmark_report(names, extract_functions, times_all, DOIs)


def write_single_result(pdf_library_name: str, doi: str, extracted_text: str) -> None:
    folder = f"results/{pdf_library_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(f"{folder}/{doi}.txt", "w") as f:
        f.write(extracted_text)


def write_benchmark_report(
    names: List[str],
    extract_functions: Dict[str, Library],
    times_all: Dict[str, List[float]],
    dois: List[str],
) -> None:
    """Create a benchmark report from all timing results."""
    # avg_times = {name: np.mean(times_all[name]) for name in names}
    with open("README.md", "w") as f:
        f.write(f"# PDF Read Benchmark\n")

        f.write("This benachmark is about reading pure PDF files - not")
        f.write("scanned documents and not documents that applied OCR.\n\n")

        f.write(f"## Benchmarking machine\n")
        f.write(f"{get_processor_name()}\n\n")

        f.write("## Input Documents\n")
        for i, doi in enumerate(dois, start=1):
            f.write(f"{i}. [{doi}](https://arxiv.org/pdf/{doi}.pdf)\n")
        f.write("\n")

        f.write("## Libraries\n")
        for name in names:
            lib = extract_functions[name]
            if lib.dependencies:
                f.write(
                    f"* {lib.name}: {lib.version} (depends on {lib.dependencies})\n"
                )
            else:
                f.write(f"* {lib.name}: {lib.version}\n")

        # ---------------------------------------------------------------------

        f.write("\n")
        f.write("## Text Extraction Speed\n\n")
        f.write(
            f"| # | {'Library':<15}|  "
            + f"{'Avgerage':<6} |  "
            + "|".join(
                [
                    f"[{i:^7}](https://arxiv.org/pdf/{doi}.pdf)"
                    for i, doi in enumerate(dois, start=1)
                ]
            )
            + "\n"
        )
        f.write(
            "|---|"
            + "-" * 15
            + "|---------|---"
            + "|".join(["-" * 6 for _ in dois])
            + "\n"
        )
        averages = [np.mean(times_all[name]) for name in names]
        sort_order = np.argsort([avg for avg in averages])
        for place, index in enumerate(sort_order, start=1):
            library_name = names[index]
            lib = extract_functions[library_name]
            avg = averages[index]
            details = "|".join([f"{t:6.2f}s" for t in times_all[library_name]])
            f.write(
                f"|{place:>2} | [{lib.name:<15}]({lib.url})| {avg:6.2f}s | {details}\n"
            )

        # ---------------------------------------------------------------------

        f.write("\n")
        f.write("## Text Extraction Quality\n\n")
        f.write(
            f"| # | {'Library':<15}|  "
            + f"{'Avgerage':<6} |  "
            + "|".join(
                [
                    f"[{i:^7}](https://arxiv.org/pdf/{doi}.pdf)"
                    for i, doi in enumerate(dois, start=1)
                ]
            )
            + "\n"
        )
        # Get data
        all_scores: Dict[str, List[float]] = {}
        for library_name in names:
            lib = extract_functions[library_name]
            all_scores[library_name] = []
            for doi in track(dois):
                all_scores[library_name].append(get_score(doi, library_name))

        # Print table
        f.write(
            "|---|" + "-" * 15 + "|---|---" + "|".join(["-" * 6 for _ in dois]) + "\n"
        )
        averages = [np.mean(all_scores[name]) for name in names]
        sort_order = np.argsort([-avg for avg in averages])
        for place, index in enumerate(sort_order, start=1):
            library_name = names[index]
            lib = extract_functions[library_name]
            avg = averages[index]
            f.write(
                f"|{place:>2} | [{lib.name:<15}]({lib.url}) | {avg*100:3.0f}% | {' | '.join([f'{score*100:3.0f}%' for score in all_scores[library_name]])}\n"
            )


def load_extracted_data(path: str) -> str:
    with open(path) as fp:
        return fp.read()


def get_score(doi: str, library_name: str):
    gt_data = load_extracted_data(f"extraction-ground-truth/{doi}.txt")
    extracted_data = load_extracted_data(f"results/{library_name}/{doi}.txt")
    return ratio(gt_data, extracted_data)


if __name__ == "__main__":
    libraries = {
        "tika": Library(
            "Tika",
            "https://pypi.org/project/tika/",
            lambda n: parser.from_buffer(BytesIO(n))["content"],
            tika.__version__,
        ),
        "pypdf2": Library(
            "PyPDF2",
            "https://pypi.org/project/PyPDF2/",
            pypdf2_get_text,
            PyPDF2.__version__,
        ),
        "pdfminer": Library(
            "pdfminer.six",
            "https://pypi.org/project/pdfminer.six/",
            lambda n: extract_text(BytesIO(n)),
            pdfminer.__version__,
        ),
        "pdfplumber": Library(
            "pdfplumber",
            "https://pypi.org/project/pdfplumber/",
            pdfplubmer_get_text,
            pdfplumber.__version__,
        ),
        "pymupdf": Library(
            "PyMuPDF",
            "https://pypi.org/project/PyMuPDF/",
            lambda n: pymupdf_get_text(n),
            PyMuPDF.version[0],
            "MuPDF",
        ),
    }
    main(libraries)
