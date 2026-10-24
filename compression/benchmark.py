# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pikepdf",
#     # PyMuPDF 2 adds heavy layout-analysis dependencies (onnxruntime, numpy, ...)
#     # without wheels for every Python version; saving does not need them.
#     "pymupdf>=1.26,<2",
#     "pypdf",
# ]
# ///
"""
Compare how much PDF libraries can reduce the size of existing PDF documents.

Every compressor reads a document and writes it again with its best *lossless*
settings (Ghostscript, if installed, is included as a lossy reference). Each
run happens in its own process, so that time and peak memory can be measured
and a crash does not stop the benchmark. Every output is checked with PyMuPDF:
same number of pages and same extracted text as the input.

Usage (from the repository root):

    uv run compression/benchmark.py                 # run everything
    uv run compression/benchmark.py --documents GeoTopo-book --compressors pypdf
    uv run compression/benchmark.py --report-only   # rebuild README.md

The results are merged into ``compression/results/results.json`` and rendered
into ``compression/README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distribution as distribution_of
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
# Shared with the main benchmark (benchmark.py), which stores its documents in
# the same folder under the same names.
PDF_DIRECTORY = REPOSITORY_ROOT / "pdfs"
OUTPUT_DIRECTORY = HERE / "output"
RESULTS_FILE = HERE / "results" / "results.json"
README_FILE = HERE / "README.md"
RESULTS_FORMAT = 1


@dataclass(frozen=True)
class Document:
    name: str
    url: str
    sha256: str
    producer: str

    @property
    def path(self) -> Path:
        return PDF_DIRECTORY / f"{self.name}.pdf"


ARXIV = "https://arxiv.org/pdf"
PYPDF_FILES = "https://github.com/py-pdf/pypdf/files"
ATTACHMENTS = "https://github.com/user-attachments/files"
SAMPLE_FILES = "https://github.com/py-pdf/sample-files/raw/main"

DOCUMENTS = [
    # pdfTeX, as in the main benchmark: already compressed, uses object streams.
    Document(
        "2201.00214",
        f"{ARXIV}/2201.00214.pdf",
        "9ea7ed2fd1e4eb364ee96c089a61c71f29613c3768cdb7c41daf7314ab62ae91",
        "pdfTeX",
    ),
    Document(
        "1707.09725",
        f"{ARXIV}/1707.09725.pdf",
        "15d73fe4d8abe5122d547118bdb0accb930737a7c656c9228ad4bee5a5753532",
        "pdfTeX",
    ),
    Document(
        "2201.00069",
        f"{ARXIV}/2201.00069.pdf",
        "51ff3326533a385e57506bbcb142cda45e2cdcfa170f77c9c6f512c9e1dddbf3",
        "pdfTeX (image-heavy)",
    ),
    Document(
        "GeoTopo-book",
        f"{SAMPLE_FILES}/009-pdflatex-geotopo/GeoTopo.pdf",
        "a0b9242aef9b5a2c4850160cc9aaf64f5c7b8be705d6872ae28aebf5ab512540",
        "pdfTeX",
    ),
    # Real-world documents from pypdf's test suite, one or two per producer.
    Document(
        "PDF32000_2008",
        f"{PYPDF_FILES}/11714214/PDF32000_2008.pdf",
        "9de0ca9e8570d6209e8bd48a355be8eb6ec376acfc3fc3ae97cd8730351417ff",
        "Acrobat Distiller 8.1 (tagged, encrypted)",
    ),
    Document(
        "tika-972174",
        f"{ATTACHMENTS}/18381744/tika-972174.pdf",
        "41e1db3c67c24a9f24d1d5509d3b81b75ef77684b427037cad1bc790e414b7ba",
        "Acrobat Distiller 8.0",
    ),
    Document(
        "Ballinasloe_WS",
        f"{PYPDF_FILES}/10983477/Ballinasloe_WS.pdf",
        "6f3cc502e413e5f0319ddebfe9e0ac6591b84b893cf12d65ae7603f10adc5e7b",
        "Adobe Acrobat Pro DC",
    ),
    Document(
        "2021_book_security",
        f"{ATTACHMENTS}/15605648/2021_book_security.pdf",
        "29a83d67bdddf5fb6ac16d072f2ec151a87ed7c5e0aed8e67f9515428e368e45",
        "Adobe PDF Library 15",
    ),
    Document(
        "USC-EMBA",
        f"{PYPDF_FILES}/11578953/USC.EMBA.-.Pre-Season.and.Theme.I.pdf",
        "8680205a769c47e6617b8c29c6db70521c92f92aac87c98b3c09361a6dcb57e0",
        "PDFium",
    ),
    Document(
        "crystallography",
        f"{PYPDF_FILES}/8884470/fdocuments.in_sweet-fundamentals-of-crystallography.pdf",
        "1b8d88974226840ecc9580f32dcbe07bc1c25f510dc5aeb1b0f89b7418f562d1",
        "OpenOffice.org 2.3",
    ),
    Document(
        "irbookonlinereading",
        f"{PYPDF_FILES}/15234776/irbookonlinereading.pdf",
        "ab67ad67d89309a3d03de219173cfdea471e856a5083c83e49b11be8c01581ea",
        "dvips + Ghostscript 8.51",
    ),
    Document(
        "FTX-claim",
        f"{PYPDF_FILES}/13127197/FTX.Claim.SC30.01072023101624File595287144.pdf",
        "b1a1e3b693efa47faea737bb5b70bb3cd7429cecef2bb6afcb24bd7ff4542907",
        "iTextSharp 5.5",
    ),
    Document(
        "WBT-whitepaper",
        f"{PYPDF_FILES}/12543290/whitepaper.WBT.token.blockchain.whitepaper.pdf",
        "a296ad22b963fbfaf16afdd941acd2754617c389cfa88837ddcd17ccebe74437",
        "iLovePDF",
    ),
    # Small documents from py-pdf/sample-files.
    Document(
        "libreoffice-writer",
        f"{SAMPLE_FILES}/002-trivial-libre-office-writer/002-trivial-libre-office-writer.pdf",
        "fc67ce4f76ffb44e818ebe4f673dbeb6002ad93a59f3856ff14fb1d3625f10a5",
        "LibreOffice",
    ),
    Document(
        "google-doc",
        f"{SAMPLE_FILES}/011-google-doc-document/google-doc-document.pdf",
        "69f6b7f493b1bc55d518942976cbeadc4ec0a36f6d8a6dc24feffc516d35b2c9",
        "Google Docs (Skia)",
    ),
    Document(
        "reportlab-overlay",
        f"{SAMPLE_FILES}/013-reportlab-overlay/reportlab-overlay.pdf",
        "fdbdd49a118053577240850826a7eff6ac4ce7288527bf4c483714f0113860ed",
        "ReportLab",
    ),
    Document(
        "pdfkit",
        f"{SAMPLE_FILES}/022-pdfkit/pdfkit.pdf",
        "8820ba44cd62264fd561e921aacc214cee7ba76723f525d591cdb2104a87f0dd",
        "pdfkit (JavaScript)",
    ),
    Document(
        "imagemagick-images",
        f"{SAMPLE_FILES}/007-imagemagick-images/imagemagick-images.pdf",
        "0f2076573bfed1107300a2383b88bbbbc2b85a57f06b3ff478a0faa7ded57b4e",
        "ImageMagick",
    ),
    Document(
        "cmyk-image",
        f"{SAMPLE_FILES}/023-cmyk-image/cmyk-image.pdf",
        "5a5f76a951e403a5b357992789afc5164fd6c2914583741de7a1dd08ec029ab2",
        "unknown (CMYK image)",
    ),
]


# --- Compressors -----------------------------------------------------------
# Each compressor reads ``source`` and writes ``target``. They run inside the
# worker process (see ``run_worker``), so imports happen there.


def _pypdf(source: Path, target: Path, *, streams: bool, deduplicate: bool) -> None:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(source))
    if streams:
        for page in writer.pages:
            page.compress_content_streams(level=9)
    if deduplicate:
        try:
            writer.compress_identical_objects(
                remove_duplicates=True, remove_unreferenced=True
            )
        except TypeError:  # pypdf < 6.5 uses the old parameter names
            writer.compress_identical_objects(
                remove_identicals=True, remove_orphans=True
            )
    with open(target, "wb") as stream:
        writer.write(stream)


def pypdf_rewrite(source: Path, target: Path) -> None:
    _pypdf(source, target, streams=False, deduplicate=False)


def pypdf_streams(source: Path, target: Path) -> None:
    _pypdf(source, target, streams=True, deduplicate=False)


def pypdf_deduplicate(source: Path, target: Path) -> None:
    _pypdf(source, target, streams=False, deduplicate=True)


def pypdf_all(source: Path, target: Path) -> None:
    _pypdf(source, target, streams=True, deduplicate=True)


def pikepdf_compress(source: Path, target: Path) -> None:
    import pikepdf

    with pikepdf.open(source) as pdf:
        pdf.save(
            target,
            compress_streams=True,
            recompress_flate=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
        )


def pymupdf_compress(source: Path, target: Path) -> None:
    import pymupdf

    with pymupdf.open(source) as document:
        document.save(
            target,
            garbage=4,
            deflate=True,
            deflate_images=True,
            deflate_fonts=True,
            use_objstms=1,
        )


def ghostscript_command(source: Path, target: Path) -> list[str]:
    return [
        shutil.which("gs") or "gs",
        "-q",
        "-dNOPAUSE",
        "-dBATCH",
        "-dSAFER",
        "-sDEVICE=pdfwrite",
        "-dPDFSETTINGS=/default",
        f"-sOutputFile={target}",
        str(source),
    ]


@dataclass(frozen=True)
class Compressor:
    name: str
    library: str
    description: str
    lossless: bool
    function: Callable[[Path, Path], None] | None = None
    command: Callable[[Path, Path], list[str]] | None = None

    def is_available(self) -> bool:
        if self.command is not None:
            return shutil.which("gs") is not None
        return True

    def version(self) -> str:
        if self.command is not None:
            result = subprocess.run(
                ["gs", "--version"], capture_output=True, text=True, check=False
            )
            return result.stdout.strip()
        try:
            distribution = distribution_of(self.library)
        except PackageNotFoundError:
            return "unknown"
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct_url.get("dir_info", {}).get("editable"):
            return f"{distribution.version} (local checkout)"
        return distribution.version


COMPRESSORS = {
    compressor.name: compressor
    for compressor in [
        Compressor(
            "pypdf",
            "pypdf",
            "`PdfWriter(clone_from=reader)` + `write()`",
            True,
            pypdf_rewrite,
        ),
        Compressor(
            "pypdf-streams",
            "pypdf",
            "+ `page.compress_content_streams(level=9)` for every page",
            True,
            pypdf_streams,
        ),
        Compressor(
            "pypdf-dedupe",
            "pypdf",
            "+ `compress_identical_objects(remove_duplicates=True,"
            " remove_unreferenced=True)`",
            True,
            pypdf_deduplicate,
        ),
        Compressor(
            "pypdf-all",
            "pypdf",
            "both of the above",
            True,
            pypdf_all,
        ),
        Compressor(
            "pikepdf",
            "pikepdf",
            "`save(compress_streams=True, recompress_flate=True,"
            " object_stream_mode=generate)`",
            True,
            pikepdf_compress,
        ),
        Compressor(
            "pymupdf",
            "pymupdf",
            "`save(garbage=4, deflate=True, deflate_images=True,"
            " deflate_fonts=True, use_objstms=1)`",
            True,
            pymupdf_compress,
        ),
        Compressor(
            "ghostscript",
            "ghostscript",
            "`gs -sDEVICE=pdfwrite -dPDFSETTINGS=/default` (re-renders the"
            " document, may re-encode images)",
            False,
            command=ghostscript_command,
        ),
    ]
}


# --- Running ---------------------------------------------------------------


def run_worker(name: str, source: Path, target: Path) -> None:
    """Entry point of the worker process: compress and report the time."""
    function = COMPRESSORS[name].function
    assert function is not None
    start = time.perf_counter()
    function(source, target)
    print(json.dumps({"seconds": time.perf_counter() - start}))


def _max_rss_bytes(rusage) -> int:
    # ru_maxrss is in KiB on Linux and in bytes on macOS.
    factor = 1 if sys.platform == "darwin" else 1024
    return int(rusage.ru_maxrss * factor)


def run_process(command: list[str], timeout: float) -> tuple[int, str, str, dict]:
    """Run ``command``, measuring wall time and the peak memory of the child."""
    measurements: dict = {}
    # Files instead of pipes: a child writing many warnings cannot block on a
    # full pipe, and we can reap it with os.wait4 to get its resource usage.
    with tempfile.TemporaryFile("w+") as stdout, tempfile.TemporaryFile("w+") as stderr:
        start = time.perf_counter()
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
        timer = threading.Timer(timeout, process.kill)
        timer.start()
        try:
            if hasattr(os, "wait4"):
                _, status, rusage = os.wait4(process.pid, 0)
                timer.cancel()
                process.returncode = os.waitstatus_to_exitcode(status)
                measurements["max_rss_bytes"] = _max_rss_bytes(rusage)
            else:  # Windows
                process.wait()
        finally:
            timer.cancel()
        measurements["wall_seconds"] = time.perf_counter() - start
        stdout.seek(0)
        stderr.seek(0)
        return process.returncode, stdout.read(), stderr.read(), measurements


def verify(source: Path, target: Path) -> dict:
    """Compare page count and extracted text of input and output."""
    import pymupdf

    def pages_and_text(path: Path) -> tuple[int, list[str]]:
        with pymupdf.open(path) as document:
            return document.page_count, [
                " ".join(page.get_text().split()) for page in document
            ]

    try:
        pages_in, text_in = pages_and_text(source)
        pages_out, text_out = pages_and_text(target)
    except Exception as exception:  # broken output
        return {"pages_ok": False, "text_ok": False, "verify_error": repr(exception)}
    return {"pages_ok": pages_in == pages_out, "text_ok": text_in == text_out}


def run_one(compressor: Compressor, document: Document, timeout: float) -> dict:
    target = OUTPUT_DIRECTORY / compressor.name / f"{document.name}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    if compressor.command is not None:
        command = compressor.command(document.path, target)
    else:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            compressor.name,
            str(document.path),
            str(target),
        ]
    returncode, stdout, stderr, result = run_process(command, timeout)
    if result["wall_seconds"] >= timeout:
        return {**result, "error": f"timeout after {timeout:.0f} s"}
    if returncode != 0 or not target.exists():
        last_line = (stderr.strip().splitlines() or [f"exit code {returncode}"])[-1]
        return {**result, "error": last_line[:300]}
    if compressor.command is None:
        result["seconds"] = json.loads(stdout.strip().splitlines()[-1])["seconds"]
    else:
        result["seconds"] = result["wall_seconds"]
    result["size"] = target.stat().st_size
    result.update(verify(document.path, target))
    return result


def download(document: Document) -> dict:
    """Download the document if needed and describe it."""
    PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if not document.path.exists():
        print(f"Downloading {document.url}")
        request = urllib.request.Request(
            document.url, headers={"User-Agent": "py-pdf-benchmarks"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            document.path.write_bytes(response.read())
    data = document.path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != document.sha256:
        print(
            f"WARNING: {document.name} has SHA-256 {sha256}, expected"
            f" {document.sha256}. Results are not comparable to other runs."
        )
    import pymupdf

    with pymupdf.open(document.path) as pdf:
        pages = pdf.page_count
    return {
        "url": document.url,
        "producer": document.producer,
        "size": len(data),
        "pages": pages,
        "sha256": sha256,
        "uses_object_streams": b"/ObjStm" in data,
    }


def environment() -> dict:
    processor = platform.processor()
    if sys.platform.startswith("linux"):
        match = re.search(
            r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text(), re.MULTILINE
        )
        processor = match.group(1) if match else processor
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "processor": processor,
        "cpu_count": os.cpu_count(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


# --- Results and report ----------------------------------------------------


def load_results() -> dict:
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
        if results.get("format") == RESULTS_FORMAT:
            return results
    return {"format": RESULTS_FORMAT, "documents": {}, "compressors": {}, "runs": {}}


def save_results(results: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def size_format(size: float) -> str:
    for unit in ("B", "KiB", "MiB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
        size /= 1024
    return f"{size:.1f} GiB"


def markdown_table(headings: list[str], rows: list[list[str]], align: str) -> str:
    """``align`` has one character per column: ``l`` or ``r``."""
    widths = [
        max(len(str(cell)) for cell in column)
        for column in zip(headings, *rows, strict=True)
    ]
    rule = {"l": lambda w: ":" + "-" * (w - 1), "r": lambda w: "-" * (w - 1) + ":"}

    def line(cells: list[str]) -> str:
        padded = [
            str(cell).ljust(width) if a == "l" else str(cell).rjust(width)
            for cell, width, a in zip(cells, widths, align, strict=True)
        ]
        return "| " + " | ".join(padded) + " |"

    separator = (
        "| " + " | ".join(rule[a](w) for a, w in zip(align, widths, strict=True)) + " |"
    )
    return "\n".join([line(headings), separator, *(line(row) for row in rows)]) + "\n"


def run_status(run: dict) -> str:
    if "error" in run:
        return "error"
    if not run.get("pages_ok"):
        return "pages differ"
    if not run.get("text_ok"):
        return "text differs"
    return "ok"


def write_report(results: dict) -> None:
    documents = [d for d in DOCUMENTS if d.name in results["documents"]]
    compressors = [c for c in COMPRESSORS.values() if c.name in results["compressors"]]
    runs = results["runs"]

    def run(compressor: Compressor, document: Document) -> dict | None:
        return runs.get(compressor.name, {}).get(document.name)

    lines = [
        "# PDF Compression Benchmark",
        "",
        "How much smaller does a PDF get when it is rewritten with each library's",
        "best lossless settings? All tools write unencrypted output. Ghostscript",
        "re-renders the document and is included as a lossy reference only.",
        "",
        "Run it with [uv](https://docs.astral.sh/uv/) from the repository root:",
        "",
        "```bash",
        "uv run compression/benchmark.py",
        "# Single documents / compressors, e.g. after a pypdf change:",
        "uv run compression/benchmark.py --documents GeoTopo-book --compressors pypdf",
        "# Benchmark a local pypdf checkout instead of the locked release:",
        "uv run --with-editable ../pypdf compression/benchmark.py --compressors pypdf",
        "```",
        "",
        "The script declares its dependencies inline; `benchmark.py.lock` pins them.",
        "Documents are downloaded to `pdfs/` (shared with the main benchmark) and",
        "verified by SHA-256; outputs are written to `compression/output/`.",
        "",
        "## Environment",
        "",
    ]
    env = results.get("environment", {})
    lines += [
        f"* Date: {env.get('date', '-')}",
        f"* Processor: {env.get('processor', '-')} ({env.get('cpu_count', '-')} cores)",
        f"* System: {env.get('system', '-')}, Python {env.get('python', '-')}",
        "",
        "## Compressors",
        "",
    ]
    lines.append(
        markdown_table(
            ["Name", "Version", "Lossless", "Settings"],
            [
                [
                    c.name,
                    results["compressors"][c.name]["version"],
                    "yes" if c.lossless else "no",
                    c.description,
                ]
                for c in compressors
            ],
            "llll",
        )
    )

    total_input = sum(results["documents"][d.name]["size"] for d in documents)
    lines += ["## Summary", ""]
    lines += [
        "Total size of all documents: "
        f"{size_format(total_input)}. *Median* is the median of the per-document",
        "output size relative to the input. Failed runs and runs whose text",
        "differs from the input are counted with the input size.",
        "",
    ]
    summary_rows = []
    for compressor in compressors:
        total, ratios, seconds, problems, best = 0, [], 0.0, 0, 0
        for document in documents:
            size_in = results["documents"][document.name]["size"]
            result = run(compressor, document)
            ok = result is not None and run_status(result) == "ok"
            size_out = result["size"] if ok else size_in
            total += size_out
            ratios.append(size_out / size_in)
            seconds += result.get("seconds", 0) if result else 0
            problems += 0 if ok else 1
            sizes = [
                r["size"]
                for c in compressors
                if (r := run(c, document)) and run_status(r) == "ok"
            ]
            best += 1 if ok and sizes and size_out == min(sizes) else 0
        summary_rows.append(
            [
                compressor.name,
                size_format(total),
                f"{100 * total / total_input:.1f} %",
                f"{100 * statistics.median(ratios):.1f} %",
                str(best),
                f"{seconds:.1f} s",
                str(problems),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Compressor",
                "Total",
                "Total vs. input",
                "Median",
                "Smallest",
                "Time",
                "Problems",
            ],
            summary_rows,
            "lrrrrrr",
        )
    )

    lines += ["## Documents", ""]
    lines.append(
        markdown_table(
            ["Document", "Producer", "Pages", "Size", "Object streams"],
            [
                [
                    f"[{d.name}]({d.url})",
                    d.producer,
                    str(results["documents"][d.name]["pages"]),
                    size_format(results["documents"][d.name]["size"]),
                    "yes"
                    if results["documents"][d.name]["uses_object_streams"]
                    else "no",
                ]
                for d in documents
            ],
            "llrrl",
        )
    )

    def per_document_table(title: str, cell: Callable[[dict, Document], str]) -> None:
        lines.extend([f"## {title}", ""])
        rows = []
        for document in documents:
            row = [document.name]
            for compressor in compressors:
                result = run(compressor, document)
                if result is None:
                    row.append("-")
                elif run_status(result) != "ok":
                    row.append(run_status(result))
                else:
                    row.append(cell(result, document))
            rows.append(row)
        lines.append(
            markdown_table(
                ["Document", *(c.name for c in compressors)],
                rows,
                "l" + "r" * len(compressors),
            )
        )

    def relative_size(result: dict, document: Document) -> str:
        size_in = results["documents"][document.name]["size"]
        return f"{100 * result['size'] / size_in:.1f} %"

    per_document_table("Output size relative to the input", relative_size)
    per_document_table("Time", lambda result, _: f"{result['seconds']:.2f} s")
    per_document_table(
        "Peak memory (whole process)",
        lambda result, _: (
            size_format(result["max_rss_bytes"]) if "max_rss_bytes" in result else "-"
        ),
    )

    problems = [
        f"* {c.name} / {d.name}: {r.get('error') or run_status(r)}"
        for c in compressors
        for d in documents
        if (r := run(c, d)) is not None and run_status(r) != "ok"
    ]
    if problems:
        lines += [
            "## Problems",
            "",
            "Ghostscript re-renders the document, so changed text extraction is",
            "expected for it; for the lossless compressors it indicates a bug.",
            "",
            *problems,
            "",
        ]
    README_FILE.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--documents", nargs="+", choices=[d.name for d in DOCUMENTS])
    parser.add_argument("--compressors", nargs="+", choices=list(COMPRESSORS))
    parser.add_argument("--timeout", type=float, default=900, help="seconds per run")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--worker",
        nargs=3,
        metavar=("COMPRESSOR", "INPUT", "OUTPUT"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.worker:
        name, source, target = args.worker
        run_worker(name, Path(source), Path(target))
        return

    results = load_results()
    if not args.report_only:
        documents = [
            d for d in DOCUMENTS if not args.documents or d.name in args.documents
        ]
        compressors = [
            c
            for c in COMPRESSORS.values()
            if (not args.compressors or c.name in args.compressors) and c.is_available()
        ]
        results["environment"] = environment()
        for document in documents:
            results["documents"][document.name] = download(document)
        for compressor in compressors:
            results["compressors"][compressor.name] = {"version": compressor.version()}
            for document in documents:
                print(f"{compressor.name:<14} {document.name:<22}", end=" ", flush=True)
                result = run_one(compressor, document, args.timeout)
                results["runs"].setdefault(compressor.name, {})[document.name] = result
                save_results(results)
                size = size_format(result["size"]) if "size" in result else "-"
                seconds = result.get("seconds", 0)
                print(f"{run_status(result):<12} {size:>10} {seconds:7.2f} s")
    write_report(results)
    print(f"Wrote {README_FILE.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
