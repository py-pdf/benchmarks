import platform
from typing import Literal

import numpy as np
from rich.progress import track

from pdf_benchmark.calibration import REFERENCE_MACHINE, to_reference_seconds
from pdf_benchmark.data_structures import Cache, Document, Library
from pdf_benchmark.environment_info import get_processor_name
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


def get_times(
    cache: Cache, docs: list[Document], benchmark_type: str
) -> dict[str, list[float | None]]:
    """
    Create a dict pointing library names to the list of execution times.

    The times are the seconds on the reference machine, see calibration.py.
    """
    times = {}  # library : [doc1, doc2, ...]
    for lib_name in cache.benchmark_times:
        tmp = cache.benchmark_times[lib_name]  # doc:
        calibration = cache.calibration_seconds[lib_name]
        doc2read_time = {
            doc: (
                None
                if tmp[doc].get(benchmark_type) is None
                else to_reference_seconds(tmp[doc][benchmark_type], calibration[doc])
            )
            for doc in tmp
        }
        times[lib_name] = [doc2read_time[doc.name] for doc in docs]
    return times


def write_benchmark_report(
    names: list[str],
    libname2details: dict[str, Library],
    docs: list[Document],
    cache: Cache,
) -> None:
    """Create a benchmark report from all timing results."""
    # avg_times = {name: np.mean(times_all[name]) for name in names}
    with open("README.md", "w") as f:
        f.write("# PDF Library Benchmarks\n")

        f.write("This benchmark is about reading pure PDF files - not ")
        f.write("scanned documents and not documents that applied OCR.\n\n")

        f.write("## Benchmarking machine\n")
        f.write(
            f"{get_processor_name().strip()},"
            f" {platform.python_implementation()} {platform.python_version()}\n\n"
        )
        f.write(
            "Every time is the shortest of three runs. To make the times less"
            " dependent on the machine, each one is divided by the time of a"
            " fixed pure-Python workload measured right before it, and"
            " multiplied by the time of that workload on the reference machine"
            f" ({REFERENCE_MACHINE}). The times are the seconds the reference"
            " machine would need. This works best for the pure-Python"
            " libraries (pypdf, pdfminer.six, pdfplumber), and roughly for the"
            " others.\n\n"
        )

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
            row += [f"{t:0.1f}s" for t in watermarking_times[library_name]]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n")

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
            row += [f"{score / 10**6:1.1f}MB" for score in all_scores[library_name]]
            table.append(row)
        f.write(table_to_markdown(table, headings=headings))
        f.write("\n")

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
