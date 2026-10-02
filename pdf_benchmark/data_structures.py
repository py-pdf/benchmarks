from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pymupdf as PyMuPDF
import requests
from pydantic import BaseModel, Field

from .pypi import get_release_date


@dataclass(frozen=True)
class Document:
    name: str
    url: str
    layout: str = ""

    def __post_init__(self):
        if not self.path.exists():
            self.download()

    def download(self):
        response = requests.get(self.url)
        self.path.write_bytes(response.content)

    @property
    def data(self) -> bytes:
        return self.path.read_bytes()

    @property
    def path(self) -> Path:
        return Path(__file__).parent / "../pdfs" / f"{self.name}.pdf"

    @property
    def filesize(self) -> int:
        return self.path.stat().st_size

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
    image_extraction_function: Callable[[bytes], list[tuple[str, bytes]]] | None = None
    pypi_name: str | None = None


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
    # Keyed by "{pypi_name}=={version}".
    pypi_release_dates: dict[str, str] = Field(default_factory=dict)

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

    def resolve_release_date(self, library: Library) -> str:
        """Return `library`'s last PyPI release date, fetching and caching it
        if it isn't already known."""
        if not library.pypi_name:
            return library.last_release_date
        key = f"{library.pypi_name}=={library.version}"
        if not self.pypi_release_dates.get(key):
            self.pypi_release_dates[key] = get_release_date(
                library.pypi_name, library.version
            )
        return self.pypi_release_dates[key] or library.last_release_date

    def write(self, path: Path):
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))
