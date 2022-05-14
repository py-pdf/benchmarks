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
from typing import Callable, Dict, List, Literal, NamedTuple, Optional, Union

import fitz as PyMuPDF
import numpy as np
import pdfminer
import pdfplumber
import PyPDF2
import requests
import tika
from borb.pdf.pdf import PDF
from borb.toolkit.text.simple_text_extraction import SimpleTextExtraction
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
    text_extraction_function: Callable[[bytes], str]
    version: str
    watermarking_function: Optional[Callable[[bytes, bytes], bytes]] = None
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


def pypdf2_watermarking(watermark_data: bytes, data: bytes) -> bytes:
    watermark_pdf = PyPDF2.PdfFileReader(BytesIO(watermark_data))
    watermark_page = watermark_pdf.getPage(0)
    reader = PyPDF2.PdfFileReader(BytesIO(data))
    writer = PyPDF2.PdfFileWriter()
    for page in reader.pages:
        page.mergePage(watermark_page)
        writer.addPage(page)
    with BytesIO() as bytes_stream:
        writer.write(bytes_stream)
        bytes_stream.seek(0)
        return bytes_stream.read()


def borb_get_text(data: bytes) -> str:
    text = ""
    try:
        l = SimpleTextExtraction()
        d = PDF.loads(BytesIO(data), [l])
        page_nb = 0
        extr = l.get_text_for_page(page_nb)
        while extr != "":
            text += extr
            extr = l.get_text_for_page(page_nb)
            page_nb += 1
    except Exception as exc:
        print(exc)
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
    docs: List[Document],
    libraries: Dict[str, Library],
    add_text_extraction_quality=True,
) -> None:
    names = sorted(list(libraries.keys()))
    text_extraction_times: Dict[str, List[float]] = {name: [] for name in names}
    watermarking_times: Dict[str, List[float]] = {
        name: [] for name in names if libraries[name].watermarking_function
    }

    watermark_file = os.path.join(
        os.path.dirname(__file__), "watermark", "pdfs", "python-quote.pdf"
    )
    with open(watermark_file, "rb") as f:
        watermark_data = f.read()

    for doc, name in track(list(product(docs, names))):
        data = doc.data
        lib = libraries[name]
        print(f"{name} now parses {doc.name}...")
        t0 = time.time()
        text = lib.text_extraction_function(data)
        t1 = time.time()
        text_extraction_times[name].append(t1 - t0)
        write_single_result("read", name, doc.name, text, "txt")

        if lib.watermarking_function:
            t0 = time.time()
            watermarked = lib.watermarking_function(watermark_data, data)
            t1 = time.time()
            watermarking_times[name].append(t1 - t0)
            write_single_result("watermark", name, doc.name, watermarked, "pdf")
    write_benchmark_report(
        names,
        libraries,
        text_extraction_times,
        watermarking_times,
        docs,
        add_text_extraction_quality,
    )


def write_single_result(
    benchmark: Literal["read", "watermark"],
    pdf_library_name: str,
    pdf_file_name: str,
    data: Union[str, bytes],
    extension: Literal["txt", "pdf"],
) -> None:
    folder = f"{benchmark}/results/{pdf_library_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    mode = "wb" if extension == "pdf" or isinstance(data, bytes) else "w"
    with open(f"{folder}/{pdf_file_name}.{extension}", mode) as f:
        try:
            f.write(data)
        except Exception as exc:
            print(exc)


def write_benchmark_report(
    names: List[str],
    extract_functions: Dict[str, Library],
    text_extraction_times: Dict[str, List[float]],
    watermarking_times: Dict[str, List[float]],
    docs: List[Document],
    add_text_extraction_quality: bool = True,
) -> None:
    """Create a benchmark report from all timing results."""
    # avg_times = {name: np.mean(times_all[name]) for name in names}
    with open("README.md", "w") as f:
        f.write(f"# PDF Text Extraction Benchmark\n")

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
        table = []
        header = ["Name", "Version", "Dependencies", "License"]
        for name in names:
            lib = extract_functions[name]
            row = [lib.name, lib.version, lib.dependencies, lib.license]

        f.write(table_to_markdown(table, header, alignment=alignment))
        f.write("\n")

        doc_headers = [f"[{i:^7}]({doc.url})" for i, doc in enumerate(docs, start=1)]
        # ---------------------------------------------------------------------

        f.write("\n")
        f.write("## Text Extraction Speed\n\n")
        table = []
        headings = ["#", "Library", "Average"] + doc_headers
        averages = [np.mean(text_extraction_times[name]) for name in names]
        sort_order = np.argsort([avg for avg in averages])
        for place, index in enumerate(sort_order, start=1):
            library_name = names[index]
            lib = extract_functions[library_name]
            avg = averages[index]
            row = [place, f"[{lib.name:<15}]({lib.url})", f"{avg:6.2f}s"]
            row += [f"{t:0.2f}s" for t in text_extraction_times[library_name]]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n")

        f.write("\n")
        f.write("## Watermarking Speed\n\n")
        table = []
        headings = ["#", "Library", "Average"] + doc_headers
        names = list(watermarking_times.keys())
        averages = [
            np.mean(watermarking_times[name])
            for name in names
            if name in watermarking_times
        ]
        sort_order = np.argsort([avg for avg in averages])
        for place, index in enumerate(sort_order, start=1):
            library_name = names[index]
            lib = extract_functions[library_name]
            avg = averages[index]
            row = [place, f"[{lib.name:<15}]({lib.url})", f"{avg:6.2f}s"]
            row += [f"{t:0.2f}s" for t in text_extraction_times[library_name]]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n")

        # ---------------------------------------------------------------------
        if add_text_extraction_quality:
            names = list(text_extraction_times.keys())
            f.write("## Text Extraction Quality\n\n")
            # Get data
            all_scores: Dict[str, List[float]] = {}
            for library_name in names:
                lib = extract_functions[library_name]
                all_scores[library_name] = []
                for doc in track(docs):
                    all_scores[library_name].append(
                        get_text_extraction_score(doc, library_name)
                    )

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


def get_text_extraction_score(doc: Document, library_name: str):
    gt_data = load_extracted_data(f"read/extraction-ground-truth/{doc.name}.txt")
    extracted_data = load_extracted_data(f"read/results/{library_name}/{doc.name}.txt")
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
            version=PyPDF2.__version__,
            watermarking_function=pypdf2_watermarking,
            license="BSD 3-Clause",
        ),
        "pdfminer": Library(
            "pdfminer.six",
            "https://pypi.org/project/pdfminer.six/",
            lambda n: extract_text(BytesIO(n)),
            version=pdfminer.__version__,
        ),
        "pdfplumber": Library(
            "pdfplumber",
            "https://pypi.org/project/pdfplumber/",
            pdfplubmer_get_text,
            version=pdfplumber.__version__,
        ),
        "pymupdf": Library(
            "PyMuPDF",
            "https://pypi.org/project/PyMuPDF/",
            lambda n: pymupdf_get_text(n),
            version=PyMuPDF.version[0],
            watermarking_function=None,
            dependencies="MuPDF",
        ),
        "pdftotext": Library(
            "pdftotext",
            "https://pypi.org/project/pdftotext/",
            pdftotext_get_text,
            "0.86.1",
            None,
            "build-essential libpoppler-cpp-dev pkg-config python3-dev",
        ),
        "borb": Library("Borb", "https://pypi.org/project/borb/", borb_get_text, "?"),
    }
    main(docs, libraries, add_text_extraction_quality=True)
