"""
Compare text extraction performance of different PDF parsers.
"""

import json
import os
import platform
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Literal, NamedTuple, Optional, Tuple, Union

import fitz as PyMuPDF
import numpy as np
import pdfminer
import pdfplumber
import pypdf
import pypdfium2 as pdfium
import requests
import tika
from borb.pdf.pdf import PDF
from borb.toolkit.text.simple_text_extraction import SimpleTextExtraction
from Levenshtein import ratio  # python-Levenshtein
from pdfminer.high_level import extract_pages, extract_text
from pydantic import BaseModel, Field
from rich.progress import track
from tika import parser

from utils import sizeof_fmt, table_to_markdown

tika.initVM()


def get_processor_name():
    """Credits: https://stackoverflow.com/a/13078519/562769"""
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command, shell=True).strip().decode("utf-8")
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
    pathname: str
    url: str
    text_extraction_function: Callable[[bytes], str]
    version: str
    watermarking_function: Optional[Callable[[bytes, bytes], bytes]] = None
    dependencies: str = ""
    license: str = ""
    last_release_date: str = ""
    image_extraction_function: Optional[
        Callable[[bytes], List[Tuple[str, bytes]]]
    ] = None


class Cache(BaseModel):
    # First str: lib
    # Second str: doc
    benchmark_times: Dict[str, Dict[str, Dict[str, float]]] = Field(
        default_factory=dict
    )
    read_quality: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    watermarking_result_file_size: Dict[str, Dict[str, float]] = Field(
        default_factory=dict
    )

    def has_doc(self, library: Library, document: Document) -> bool:
        lib = library.pathname
        doc = document.name

        if lib not in self.benchmark_times:
            self.benchmark_times[lib] = {}
        if doc not in self.benchmark_times[lib]:
            self.benchmark_times[lib][doc] = {}

        if lib not in self.read_quality:
            self.read_quality[lib] = {}

        if lib not in self.watermarking_result_file_size:
            self.watermarking_result_file_size[lib] = {}

        return doc in self.benchmark_times[lib] and doc in self.read_quality[lib]

    def write(self, path: Path):
        with open(path, "w") as f:
            f.write(self.json(indent=4, sort_keys=True))


def pymupdf_get_text(data: bytes) -> str:
    with PyMuPDF.open(stream=data, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
    return text


def pypdf_get_text(data: bytes) -> str:
    text = ""
    reader = pypdf.PdfReader(BytesIO(data))
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def pdfium_get_text(data: bytes) -> str:
    text = ""
    pdf = pdfium.PdfDocument(data)
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        textpage = page.get_textpage()
        text += textpage.get_text_range() + "\n"
    return text


def pypdf_watermarking(watermark_data: bytes, data: bytes) -> bytes:
    watermark_pdf = pypdf.PdfReader(BytesIO(watermark_data))
    watermark_page = watermark_pdf.pages[0]
    reader = pypdf.PdfReader(BytesIO(data))
    writer = pypdf.PdfWriter()
    for page in reader.pages:
        page.merge_page(watermark_page)
        writer.add_page(page)
    with BytesIO() as bytes_stream:
        writer.write(bytes_stream)
        bytes_stream.seek(0)
        return bytes_stream.read()


def pypdf_image_extraction(data: bytes) -> List[Tuple[str, bytes]]:
    images = []
    try:
        reader = pypdf.PdfReader(BytesIO(data))
        for page in reader.pages:
            for image in page.images:
                images.append((image.name, image.data))
    except Exception as exc:
        print(f"pypdf Image extraction failure: {exc}")
    return images


def pymupdf_image_extraction(data: bytes) -> List[Tuple[str, bytes]]:
    images = []
    with PyMuPDF.open(stream=data, filetype="pdf") as pdf_file:
        for page_index in range(len(pdf_file)):
            page = pdf_file[page_index]
            for image_index, img in enumerate(page.get_images(), start=1):
                xref = img[0]
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                images.append(
                    (f"image{page_index+1}_{image_index}.{image_ext}", image_bytes)
                )
    return images


def pymupdf_watermarking(watermark_data: bytes, data: bytes) -> bytes:
    pdf_file = PyMuPDF.open(stream=data, filetype="pdf")
    overlay = PyMuPDF.open(stream=watermark_data, filetype="pdf")
    for i in range(pdf_file.page_count):
        page = pdf_file.load_page(i)
        page_front = PyMuPDF.open()
        page_front.insert_pdf(overlay, from_page=i, to_page=i)
        page.show_pdf_page(
            page.rect,
            page_front,
            pno=0,
            keep_proportion=True,
            overlay=True,
            oc=0,
            rotate=0,
            clip=None,
        )
    return pdf_file.write()


def pdfminer_image_extraction(data: bytes) -> List[Tuple[str, bytes]]:
    from PIL import Image

    def get_image(layout_object):
        if isinstance(layout_object, pdfminer.layout.LTImage):
            return layout_object
        if isinstance(layout_object, pdfminer.layout.LTContainer):
            for child in layout_object:
                return get_image(child)
        else:
            return None

    images = []
    try:
        pages = list(extract_pages(BytesIO(data)))
        for page in pages:
            ex_images = list(filter(bool, map(get_image, page)))
            for image in ex_images:
                image_pil = Image.frombytes(
                    "1", image.srcsize, image.stream.get_data(), "raw"
                )

                img_byte_arr = BytesIO()
                image_pil.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()

                images.append((f"{image.name}.png", img_byte_arr))
    except Exception as exc:
        print(f"pdfminer Image extraction failure: {exc}")
    return images


def borb_get_text(data: bytes) -> str:
    text = ""
    try:
        l = SimpleTextExtraction()
        PDF.loads(BytesIO(data), [l])
        obj = l.get_text()
        for page_index in range(len(obj)):
            text += obj[page_index]
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
    add_text_extraction_quality: bool = True,
) -> None:
    cache_path = Path("cache.json")
    if cache_path.exists():
        with open(cache_path, "r") as f:
            cache = Cache.parse_obj(json.load(f))
    else:
        cache = Cache()
    names = sorted(list(libraries.keys()))

    watermark_file = os.path.join(
        os.path.dirname(__file__), "watermark", "pdfs", "python-quote.pdf"
    )
    with open(watermark_file, "rb") as f:
        watermark_data = f.read()

    for doc, name in track(list(product(docs, names))):
        data = doc.data
        lib = libraries[name]
        if cache.has_doc(lib, doc):
            print(f"Skip {doc.name} for {lib.name}")
            continue
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
    data: Union[str, bytes, List[Tuple[str, bytes]]],
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


def get_times(
    cache: Cache, docs: List[Document], benchmark_type: str
) -> Dict[str, List[Optional[float]]]:
    """Create a dict pointing library names to the list of execution times."""
    times = {}  # library : [doc1, doc2, ...]
    for lib_name in cache.benchmark_times:
        times[lib_name] = []
        tmp = cache.benchmark_times[lib_name]  # doc:
        doc2read_time = {doc: tmp[doc].get(benchmark_type) for doc in tmp}
        times[lib_name] = [doc2read_time[doc.name] for doc in docs]
    return times


def write_benchmark_report(
    names: List[str],
    libname2details: Dict[str, Library],
    docs: List[Document],
    cache: Cache,
) -> None:
    """Create a benchmark report from all timing results."""
    # avg_times = {name: np.mean(times_all[name]) for name in names}
    with open("README.md", "w") as f:
        f.write(f"# PDF Text Extraction Benchmark\n")

        f.write("This benchmark is about reading pure PDF files - not")
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
        header = ["Name", "Last PyPI Release", "License", "Version", "Dependencies"]
        for name in names:
            lib = libname2details[name]
            row = [
                lib.name,
                lib.last_release_date,
                lib.license,
                lib.version,
                lib.dependencies,
            ]
            table.append(row)

        f.write(table_to_markdown(table, header, alignment=alignment))
        f.write("\n")

        doc_headers = [f"[{i:^7}]({doc.url})" for i, doc in enumerate(docs, start=1)]
        # ---------------------------------------------------------------------

        f.write("\n")
        f.write("## Text Extraction Speed\n\n")
        table = []
        headings = ["#", "Library", "Average"] + doc_headers
        text_extraction_times = get_times(cache, docs, "read")
        names = [
            name
            for name in text_extraction_times.keys()
            if len(text_extraction_times[name]) > 0
        ]
        averages = [np.mean(text_extraction_times[name]) for name in names]
        sort_order = np.argsort([avg for avg in averages])
        for place, index in enumerate(sort_order, start=1):
            library_name = names[index]
            lib = libname2details[library_name]
            avg = averages[index]
            row = [place, f"[{lib.name:<15}]({lib.url})", f"{avg:6.1f}s"]
            row += [f"{t:0.1f}s" for t in text_extraction_times[library_name]]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n")

        f.write("\n")
        f.write("## Image Extraction Speed\n\n")
        table = []
        headings = ["#", "Library", "Average"] + doc_headers
        image_extraction_times = get_times(cache, docs, "image_extraction")
        names = [
            name
            for name in image_extraction_times.keys()
            if len([el for el in image_extraction_times[name] if el is not None]) > 0
        ]
        print(names)
        print(image_extraction_times["pypdf"])
        averages = [np.mean(image_extraction_times[name]) for name in names]
        sort_order = np.argsort([avg for avg in averages])
        for place, index in enumerate(sort_order, start=1):
            library_name = names[index]
            lib = libname2details[library_name]
            avg = averages[index]
            row = [place, f"[{lib.name:<15}]({lib.url})", f"{avg:6.1f}s"]
            row += [f"{t:0.1f}s" for t in image_extraction_times[library_name]]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n")

        f.write("\n")
        f.write("## Watermarking Speed\n\n")
        table = []
        headings = ["#", "Library", "Average"] + doc_headers
        watermarking_times = get_times(cache, docs, "watermark")
        names = list(watermarking_times.keys())
        names = [
            name
            for name in names
            if len([el for el in watermarking_times[name] if el is not None]) > 0
        ]
        averages = [
            np.mean([el for el in watermarking_times[name] if el is not None])
            for name in names
            if name in watermarking_times
        ]
        sort_order = np.argsort([avg for avg in averages])
        for place, index in enumerate(sort_order, start=1):
            library_name = names[index]
            lib = libname2details[library_name]
            avg = averages[index]
            row = [place, f"[{lib.name:<15}]({lib.url})", f"{avg:6.1f}s"]
            row += [f"{t:0.1f}s" for t in text_extraction_times[library_name]]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n")

        # ---------------------------------------------------------------------
        f.write("\n")
        f.write("## Watermarking File Size\n\n")
        # Get data
        all_scores: Dict[str, List[float]] = {}
        for library_name in names:
            lib = libname2details[library_name]
            all_scores[library_name] = []
            for doc in track(docs):
                all_scores[library_name].append(
                    cache.watermarking_result_file_size[library_name][doc.name]
                )
        # Print table
        table = []
        averages = [np.mean(all_scores[name]) for name in names]
        sort_order = np.argsort([-avg for avg in averages])
        for place, index in enumerate(sort_order, start=1):
            library_name = names[index]
            lib = libname2details[library_name]
            avg = averages[index]
            row = [place, f"[{lib.name:<15}]({lib.url})", f"{avg/10**6:1.1f}MB"]
            row += [f"{score/10**6:1.1f}MB" for score in all_scores[library_name]]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n")

        # ---------------------------------------------------------------------
        names = list(text_extraction_times.keys())
        f.write("## Text Extraction Quality\n\n")
        # Get data
        all_scores: Dict[str, List[float]] = {}
        for library_name in names:
            lib = libname2details[library_name]
            all_scores[library_name] = []
            for doc in track(docs):
                all_scores[library_name].append(
                    cache.read_quality[library_name][doc.name]
                )

        # Print table
        table = []
        averages = [np.mean(all_scores[name]) for name in names]
        sort_order = np.argsort([-avg for avg in averages])
        for place, index in enumerate(sort_order, start=1):
            library_name = names[index]
            lib = libname2details[library_name]
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
            "tika",
            "https://pypi.org/project/tika/",
            lambda n: parser.from_buffer(BytesIO(n))["content"],
            tika.__version__,
            dependencies="Apache Tika",
            license="Apache v2",
            last_release_date="2023-01-01",
        ),
        "pypdf": Library(
            "pypdf",
            "pypdf",
            "https://pypi.org/project/pypdf/",
            pypdf_get_text,
            version=pypdf.__version__,
            watermarking_function=pypdf_watermarking,
            license="BSD 3-Clause",
            last_release_date="2023-06-27",
            image_extraction_function=pypdf_image_extraction,
        ),
        "pdfminer": Library(
            "pdfminer.six",
            "pdfminer",
            "https://pypi.org/project/pdfminer.six/",
            lambda n: extract_text(BytesIO(n)),
            version=pdfminer.__version__,
            license="MIT/X",
            last_release_date="2022-11-05",
            image_extraction_function=pdfminer_image_extraction,
        ),
        "pdfplumber": Library(
            "pdfplumber",
            "pdfplumber",
            "https://pypi.org/project/pdfplumber/",
            pdfplubmer_get_text,
            version=pdfplumber.__version__,
            license="MIT",
            last_release_date="2023-04-13",
            dependencies="pdfminer.six",
        ),
        "pymupdf": Library(
            "PyMuPDF",
            "pymupdf",
            "https://pypi.org/project/PyMuPDF/",
            lambda n: pymupdf_get_text(n),
            version=PyMuPDF.version[0],
            watermarking_function=pymupdf_watermarking,
            image_extraction_function=pymupdf_image_extraction,
            dependencies="MuPDF",
            license="GNU AFFERO GPL 3.0 / Commerical",
            last_release_date="2023-06-21",
        ),
        "pdftotext": Library(
            "pdftotext",
            "pdftotext",
            "https://poppler.freedesktop.org/",
            pdftotext_get_text,
            "0.86.1",
            None,
            "build-essential libpoppler-cpp-dev pkg-config python3-dev",
            last_release_date="-",
            license="GPL",
        ),
        "borb": Library(
            "Borb",
            "borb",
            "https://pypi.org/project/borb/",
            borb_get_text,
            "2.1.15",
            None,
            license="AGPL/Commercial",
            last_release_date="2023-06-23",
        ),
        "pdfium": Library(
            "pypdfium2",
            "pdfium",
            "https://pypi.org/project/pypdfium2/",
            pdfium_get_text,
            "4.17.0",
            None,
            license="Apache-2.0 or BSD-3-Clause",
            last_release_date="2023-06-27",
            dependencies="PDFium (Foxit/Google)",
        ),
    }
    main(docs, libraries, add_text_extraction_quality=True)
