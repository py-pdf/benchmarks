import hashlib
import importlib.metadata
import re
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
    # The extra of the PyPI package that installs all optional dependencies
    extra: str = ""

    @property
    def optional_dependencies(self) -> list[tuple[str, str]]:
        """Return the name and version of every package of the extra.

        Raises an error if one is not installed, so that the library never runs
        without its optional dependencies.
        """
        if not self.extra:
            return []
        assert self.pypi_name, "An extra needs the name of the package"
        marker = re.compile(rf"""extra\s*==\s*['"]{re.escape(self.extra)}['"]""")
        dependencies = []
        for requirement in importlib.metadata.requires(self.pypi_name) or []:
            if marker.search(requirement):
                name = re.match(r"[A-Za-z0-9_.-]+", requirement).group(0)
                try:
                    dependencies.append((name, importlib.metadata.version(name)))
                except importlib.metadata.PackageNotFoundError as exc:
                    raise RuntimeError(
                        f"{name} of {self.pypi_name}[{self.extra}] is not installed"
                    ) from exc
        return sorted(dependencies, key=lambda item: item[0].lower())

    @property
    def cache_version(self) -> str:
        """The version of the library and of its optional dependencies."""
        return " ".join(
            [self.version]
            + [f"{name}=={version}" for name, version in self.optional_dependencies]
        )

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
    # The time of the calibration workload, measured right before the
    # benchmarks of the library and the document, see calibration.py
    calibration_seconds: dict[str, dict[str, float]] = Field(default_factory=dict)
    read_quality: dict[str, dict[str, float]] = Field(default_factory=dict)
    watermarking_result_file_size: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )
    image_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    pages_without_watermark: dict[str, dict[str, int]] = Field(default_factory=dict)
    # Keyed by "{pypi_name}=={version}".
    pypi_release_dates: dict[str, str] = Field(default_factory=dict)
    # The version of the library and of its optional dependencies the results
    # were measured with
    library_versions: dict[str, str] = Field(default_factory=dict)

    def prepare_results(self, library: Library) -> None:
        """Make room for the results of `library`.

        Results measured with another version of the library are dropped, so
        that they are measured again.
        """
        lib = library.pathname
        if self.library_versions.get(lib) != library.cache_version:
            self.benchmark_times.pop(lib, None)
            self.calibration_seconds.pop(lib, None)
            self.read_quality.pop(lib, None)
            self.watermarking_result_file_size.pop(lib, None)
            self.image_counts.pop(lib, None)
            self.pages_without_watermark.pop(lib, None)
            self.library_versions[lib] = library.cache_version
        self.benchmark_times.setdefault(lib, {})
        self.calibration_seconds.setdefault(lib, {})
        self.read_quality.setdefault(lib, {})
        self.watermarking_result_file_size.setdefault(lib, {})
        self.image_counts.setdefault(lib, {})
        self.pages_without_watermark.setdefault(lib, {})

    def has_doc(self, library: Library, document: Document) -> bool:
        """Return whether all benchmarks of `library` ran for `document`."""
        if document.name not in self.calibration_seconds[library.pathname]:
            return False
        if (
            library.image_extraction_function
            and document.name not in self.image_counts[library.pathname]
        ):
            return False
        if (
            library.watermarking_function
            and document.name not in self.pages_without_watermark[library.pathname]
        ):
            return False
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
