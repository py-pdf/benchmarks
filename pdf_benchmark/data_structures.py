import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import fitz as PyMuPDF
import requests
from pydantic import BaseModel, Field


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
        return os.path.join(os.path.dirname(__file__), "../pdfs", f"{self.name}.pdf")

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
    version: str
    text_extraction_function: Callable[[bytes], str] | None = None
    watermarking_function: Callable[[bytes, bytes], bytes] | None = None
    dependencies: str = ""
    license: str = ""
    last_release_date: str = ""
    image_extraction_function: None | (
        Callable[[bytes], list[tuple[str, bytes]]]
    ) = None


class Cache(BaseModel):
    # First str: lib
    # Second str: doc
    benchmark_times: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict
    )
    read_quality: dict[str, dict[str, float]] = Field(default_factory=dict)
    watermarking_result_file_size: dict[str, dict[str, float]] = Field(
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
            f.write(self.model_dump_json(indent=4))
