import numpy as np
from rich.progress import track

from pdf_benchmark.data_structures import Cache, Document, Library
from pdf_benchmark.environment_info import get_processor_name
from pdf_benchmark.utils import sizeof_fmt, table_to_markdown


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

        f.write("This benchmark is about reading pure PDF files - not")
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
        all_scores: dict[str, list[float]] = {}
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
