from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from rich.progress import track

from pdf_benchmark.data_structures import Cache, Document, Library
from pdf_benchmark.environment_info import get_processor_name
from pdf_benchmark.features import (
    IMAGE_FEATURES,
    TEXT_FEATURES,
    ImageFeature,
    TextFeature,
)
from pdf_benchmark.utils import sizeof_fmt, table_to_markdown


def speed_section(
    title: str,
    benchmark_type: Literal["read", "image_extraction"],
    f,
    doc_headers,
    cache,
    docs,
    libname2details,
) -> dict[str, list[float | None]]:
    f.write(f"## {title}\n\n")
    table = []
    headings = ["#", "Library", "Average"] + doc_headers
    text_extraction_times = get_times(cache, docs, benchmark_type)
    names = [
        name
        for name in text_extraction_times
        if len([el for el in text_extraction_times[name] if el is not None]) > 0
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
    return text_extraction_times


def image_count_section(
    f,
    doc_headers,
    cache: Cache,
    docs: list[Document],
    libname2details,
    image_extraction_times: dict[str, list[float | None]],
) -> None:
    """Write how many images every library extracted, next to its speed."""
    f.write("Number of extracted images:\n\n")
    table = []
    for name, times in image_extraction_times.items():
        if all(t is None for t in times):
            continue
        lib = libname2details[name]
        counts = [cache.image_counts[name].get(doc.name) for doc in docs]
        total = sum(count or 0 for count in counts)
        row = [f"[{lib.name:<15}]({lib.url})", total]
        row += ["-" if count is None else count for count in counts]
        table.append(row)
    f.write(table_to_markdown(table, headings=["Library", "Total"] + doc_headers))
    f.write("\n")
    f.write(
        "pypdf and PyMuPDF extract every image of a page once, pdfminer.six and"
        " pypdfium2 every time it is drawn. PyMuPDF does not extract inline"
        " images, and pypdfium2 fails on some of them.\n\n"
    )


def mark_missing_watermark(cache: Cache, library_name: str, doc: Document) -> str:
    """Return a mark if pages of the watermarked document lack the watermark."""
    missing = cache.pages_without_watermark[library_name].get(doc.name)
    return " ✗" if missing else ""


def write_missing_watermark_note(f, cache: Cache, names: list[str]) -> None:
    """Explain the mark of mark_missing_watermark(), if a result has it."""
    if any(any(cache.pages_without_watermark[name].values()) for name in names):
        f.write(
            "✗: the watermark text is missing on pages of the document, or pages"
            " are missing.\n\n"
        )


def get_times(
    cache: Cache, docs: list[Document], benchmark_type: str
) -> dict[str, list[float | None]]:
    """Create a dict pointing library names to the list of execution times."""
    times = {}  # library : [doc1, doc2, ...]
    for lib_name in cache.benchmark_times:
        times[lib_name] = []
        tmp = cache.benchmark_times[lib_name]  # doc:
        doc2read_time = {doc: tmp[doc].get(benchmark_type) for doc in tmp}
        times[lib_name] = [doc2read_time[doc.name] for doc in docs]
    return times


@dataclass
class FeatureResults:
    """The mark and the details of every feature, per library."""

    text: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)
    images: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)


def feature_section(
    f,
    features: list[TextFeature] | list[ImageFeature],
    results: dict[str, dict[str, tuple[str, str]]],
    libname2details: dict[str, Library],
    legend: str,
) -> None:
    """Write a matrix of the features and the libraries, and what went wrong."""
    names = list(results)
    headings = ["Feature"] + [libname2details[name].name for name in names]
    table = [
        [feature.name] + [results[name][feature.name][0] for name in names]
        for feature in features
    ]
    alignment = [("^", "<")] + [("^", "^")] * len(names)
    f.write(table_to_markdown(table, headings=headings, alignment=alignment))
    f.write(f"\n{legend}\n\n")
    f.write("<details>\n<summary>The features, and what went wrong</summary>\n\n")
    for feature in features:
        f.write(f"* **{feature.name}**: {feature.description}")
        if isinstance(feature, TextFeature):
            expected = feature.expected.replace("\n", " / ")
            f.write(f" Expected: `{expected}`.")
        problems = [
            f"{libname2details[name].name}: `{results[name][feature.name][1]}`"
            for name in names
            if results[name][feature.name][1]
        ]
        if problems:
            f.write("\n  " + "; ".join(problems).replace("\n", " "))
        f.write("\n")
    f.write("\n</details>\n\n")


def write_benchmark_report(
    names: list[str],
    libname2details: dict[str, Library],
    docs: list[Document],
    cache: Cache,
    features: FeatureResults,
) -> None:
    """Create a benchmark report from all timing results."""
    # avg_times = {name: np.mean(times_all[name]) for name in names}
    with open("README.md", "w") as f:
        f.write("# PDF Library Benchmarks\n")

        f.write("This benchmark is about reading pure PDF files - not ")
        f.write("scanned documents and not documents that applied OCR.\n\n")

        f.write("## Benchmarking machine\n")
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
        header = [
            "Name",
            "Last PyPI Release",
            "License",
            "Version",
            "Dependencies",
            "Optional dependencies",
        ]
        for name in names:
            lib = libname2details[name]
            optional = ", ".join(
                f"{package} {version}" for package, version in lib.optional_dependencies
            )
            row = [
                lib.name,
                lib.last_release_date,
                lib.license,
                lib.version,
                lib.dependencies,
                f"[{lib.extra}]: {optional}" if lib.extra else "",
            ]
            table.append(row)

        f.write(table_to_markdown(table, header, alignment=alignment))
        f.write("\n")

        doc_headers = [f"[{i:^7}]({doc.url})" for i, doc in enumerate(docs, start=1)]
        # ---------------------------------------------------------------------

        f.write("\n")
        text_extraction_times = speed_section(
            "Text Extraction Speed",
            "read",
            f,
            doc_headers,
            cache,
            docs,
            libname2details,
        )
        image_extraction_times = speed_section(
            "Image Extraction Speed",
            "image_extraction",
            f,
            doc_headers,
            cache,
            docs,
            libname2details,
        )
        image_count_section(
            f, doc_headers, cache, docs, libname2details, image_extraction_times
        )
        f.write("## Image Extraction Features\n\n")
        f.write(
            "Which kinds of images the libraries extract, each tested with a"
            " small PDF.\n\n"
        )
        feature_section(
            f,
            IMAGE_FEATURES,
            features.images,
            libname2details,
            "✓: the image is extracted. ◐: an image is extracted, but it is"
            " not the picture or cannot be opened. ✗: no image. ⚠: an error."
            " For *Drawn twice*, the number of extracted images.",
        )

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
            row += [
                f"{t:0.1f}s" + mark_missing_watermark(cache, library_name, doc)
                for t, doc in zip(watermarking_times[library_name], docs, strict=True)
            ]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n")
        write_missing_watermark_note(f, cache, names)

        # ---------------------------------------------------------------------
        f.write("\n")
        f.write("## Watermarking File Size\n\n")
        # Get data
        all_scores: dict[str, list[float]] = {}
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
        sort_order = np.argsort([avg for avg in averages])
        for place, index in enumerate(sort_order, start=1):
            library_name = names[index]
            lib = libname2details[library_name]
            avg = averages[index]
            row = [place, f"[{lib.name:<15}]({lib.url})", f"{avg / 10**6:1.1f}MB"]
            row += [
                f"{score / 10**6:1.1f}MB"
                + mark_missing_watermark(cache, library_name, doc)
                for score, doc in zip(all_scores[library_name], docs, strict=True)
            ]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n")
        write_missing_watermark_note(f, cache, names)

        # ---------------------------------------------------------------------
        names = list(text_extraction_times.keys())
        f.write("## Text Extraction Quality\n\n")
        # Get data
        all_scores: dict[str, list[float]] = {}
        for library_name in names:
            lib = libname2details[library_name]
            all_scores[library_name] = []
            for doc in track(docs):
                if doc.name not in cache.read_quality[library_name]:
                    all_scores[library_name].append(0)
                else:
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
            if avg == 0:
                continue
            row = [place, f"[{lib.name:<15}]({lib.url})", f"{avg * 100:3.0f}%"]
            row += [f"{score * 100:3.0f}%" for score in all_scores[library_name]]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n## Text Extraction Features\n\n")
        f.write(
            "Which properties of a PDF the libraries handle, each tested with a"
            " small PDF. They explain some differences of the text extraction"
            " quality.\n\n"
        )
        feature_section(
            f,
            TEXT_FEATURES,
            features.text,
            libname2details,
            "✓: the expected text, ignoring whitespace except where the feature is"
            " about it. ✗: another text. ⚠: an error.",
        )
