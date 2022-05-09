"""
Compare text extraction performance of different PDF parsers.
"""

import os
import platform
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
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
from utils import sizeof_fmt, table_to_markdown


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


@dataclass(frozen=True)
class Document:
    name: str
    url: str
    layout: str = ""

    def __post_init__(self):
        if not os.path.exists(self.path):
            self.download()

    def download(self):
        response = requests.get(self.url)
        with open(self.path, "wb") as f:
            f.write(response.content)

    @property
    def data(self):
        with open(self.path, "rb") as f:
            return f.read()

    @property
    def path(self):
        return os.path.join(os.path.dirname(__file__), "pdfs", f"{self.name}.pdf")

    @property
    def filesize(self):
        return os.path.getsize(self.path)

    @property
    def nb_pages(self):
        doc = PyMuPDF.open(self.path)
        return doc.page_count


class Library(NamedTuple):
    name: str
    url: str
    extraction_function: Callable[[bytes], str]
    version: str
    dependencies: str = ""
    license: str = ""
    last_release_date: str = ""


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


def pdftotext_get_text(data: bytes) -> str:
    new_file, filename = tempfile.mkstemp()
    with open(filename, "wb") as fp:
        fp.write(data)
    args = ["/usr/bin/pdftotext", "-enc", "UTF-8", filename, "-"]
    res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = res.stdout.decode("utf-8")
    os.close(new_file)
    os.remove(filename)
    return output


def main(
    docs: List[Document], extract_functions: Dict[str, Library], add_quality=True
) -> None:
    names = sorted(list(extract_functions.keys()))
    times_all: Dict[str, List[float]] = {name: [] for name in names}

    for doc, name in track(list(product(docs, names))):
        data = doc.data
        print(f"{name} now parses {doc.name}...")
        t0 = time.time()
        text = extract_functions[name].extraction_function(data)
        t1 = time.time()
        times_all[name].append(t1 - t0)
        write_single_result(name, doc.name, text)
    write_benchmark_report(names, extract_functions, times_all, docs, add_quality)


def write_single_result(pdf_library_name: str, name: str, extracted_text: str) -> None:
    folder = f"results/{pdf_library_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(f"{folder}/{name}.txt", "w") as f:
        f.write(extracted_text)


def write_benchmark_report(
    names: List[str],
    extract_functions: Dict[str, Library],
    times_all: Dict[str, List[float]],
    docs: List[Document],
    add_quality: bool = True,
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
        table = []
        header = ["#", "Name", "File Size", "Pages"]
        alignment = ["^>", "^<", "^>", "^>"]
        for i, doc in enumerate(docs, start=1):
            row = [
                i,
                f"[{doc.name}]({doc.url})",
                sizeof_fmt(doc.filesize),
                doc.nb_pages,
            ]
            table.append(row)
        f.write(table_to_markdown(table, header, alignment=alignment))
        f.write("\n")

        f.write("## Libraries\n")
        for name in names:
            lib = extract_functions[name]
            if lib.dependencies:
                f.write(f"* {lib.name} {lib.version} (depends on {lib.dependencies})\n")
            else:
                f.write(f"* {lib.name}: {lib.version}\n")

        doc_headers = [f"[{i:^7}]({doc.url})" for i, doc in enumerate(docs, start=1)]
        # ---------------------------------------------------------------------

        f.write("\n")
        f.write("## Text Extraction Speed\n\n")
        table = []
        headings = ["#", "Library", "Average"] + doc_headers
        averages = [np.mean(times_all[name]) for name in names]
        sort_order = np.argsort([avg for avg in averages])
        for place, index in enumerate(sort_order, start=1):
            library_name = names[index]
            lib = extract_functions[library_name]
            avg = averages[index]
            row = [place, f"[{lib.name:<15}]({lib.url})", f"{avg:6.2f}s"]
            row += [f"{t:0.2f}s" for t in times_all[library_name]]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n")

        # ---------------------------------------------------------------------
        if add_quality:
            f.write("## Text Extraction Quality\n\n")
            # Get data
            all_scores: Dict[str, List[float]] = {}
            for library_name in names:
                lib = extract_functions[library_name]
                all_scores[library_name] = []
                for doc in track(docs):
                    all_scores[library_name].append(get_score(doc, library_name))

            # Print table
            table = []
            averages = [np.mean(all_scores[name]) for name in names]
            sort_order = np.argsort([-avg for avg in averages])
            for place, index in enumerate(sort_order, start=1):
                library_name = names[index]
                lib = extract_functions[library_name]
                avg = averages[index]
                row = [place, f"[{lib.name:<15}]({lib.url})", f"{avg*100:3.0f}%"]
                row += [f"{score*100:3.0f}%" for score in all_scores[library_name]]
                table.append(row)
            f.write(table_to_markdown(table, headings=headings))


def load_extracted_data(path: str) -> str:
    with open(path) as fp:
        return fp.read()


def get_score(doc: Document, library_name: str):
    gt_data = load_extracted_data(f"extraction-ground-truth/{doc.name}.txt")
    extracted_data = load_extracted_data(f"results/{library_name}/{doc.name}.txt")
    return ratio(gt_data, extracted_data)


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
        "pdftotext": Library(
            "pdftotext",
            "https://pypi.org/project/pdftotext/",
            lambda n: pdftotext_get_text(n),
            "0.86.1",
            "build-essential libpoppler-cpp-dev pkg-config python3-dev",
        ),
    }
    main(docs, libraries, add_quality=True)
