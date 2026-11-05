import hashlib
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
    # A URL that always gives the same file, e.g. an arXiv URL with version
    url: str
    # The SHA-256 of the file the ground truth was made for
    sha256: str
    layout: str = ""

    def __post_init__(self):
        if not self.path.exists():
            self.download()
        sha256 = hashlib.sha256(self.data).hexdigest()
        if sha256 != self.sha256:
            raise ValueError(
                f"{self.path} has the SHA-256 {sha256} instead of {self.sha256}."
                " Its results would not be comparable with earlier runs, and"
                " the ground truth may not fit it."
            )

    def download(self):
        response = requests.get(
            self.url, headers={"User-Agent": "py-pdf-benchmarks"}, timeout=120
        )
        response.raise_for_status()
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

    @property
    def page_labels(self) -> list[str]:
        """The label of every page, or its number if it has none."""
        with PyMuPDF.open(self.path) as doc:
            return [page.get_label() or str(page.number + 1) for page in doc]


class Library(NamedTuple):
    name: str
    pathname: str
    url: str
    version: str
    text_extraction_function: Callable[[bytes], list[str]] | None = None
    watermarking_function: Callable[[bytes, bytes], bytes] | None = None
    dependencies: str = ""
    license: str = ""
    last_release_date: str = ""
    image_extraction_function: Callable[[bytes], list[tuple[str, bytes]]] | None = None
    pypi_name: str | None = None

    @property
    def benchmarks(self) -> list[str]:
        """The benchmarks the library takes part in, named as in the cache."""
        functions = {
            "read": self.text_extraction_function,
            "watermark": self.watermarking_function,
            "image_extraction": self.image_extraction_function,
        }
        return [name for name, function in functions.items() if function]


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
    image_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    pages_without_watermark: dict[str, dict[str, int]] = Field(default_factory=dict)
    # Keyed by "{pypi_name}=={version}".
    pypi_release_dates: dict[str, str] = Field(default_factory=dict)
    # The library version the results were measured with.
    library_versions: dict[str, str] = Field(default_factory=dict)

    def prepare_results(self, library: Library) -> None:
        """Make room for the results of `library`.

        Results measured with another version of the library are dropped, so
        that they are measured again.
        """
        lib = library.pathname
        if self.library_versions.get(lib) != library.version:
            self.benchmark_times.pop(lib, None)
            self.read_quality.pop(lib, None)
            self.watermarking_result_file_size.pop(lib, None)
            self.image_counts.pop(lib, None)
            self.pages_without_watermark.pop(lib, None)
            self.library_versions[lib] = library.version
        self.benchmark_times.setdefault(lib, {})
        self.read_quality.setdefault(lib, {})
        self.watermarking_result_file_size.setdefault(lib, {})
        self.image_counts.setdefault(lib, {})
        self.pages_without_watermark.setdefault(lib, {})

    def has_doc(self, library: Library, document: Document) -> bool:
        """Return whether all benchmarks of `library` ran for `document`."""
        times = self.benchmark_times[library.pathname].get(document.name, {})
        return all(benchmark in times for benchmark in library.benchmarks)

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
