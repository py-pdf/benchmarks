import json
import os
import ssl
import urllib.request
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader


class Cell(BaseModel):
    row_start: int
    row_end: int
    column_start: int
    column_end: int
    content: str


class Table(BaseModel):
    title: str = ""
    description: str = ""
    columns: int  # just for convenience, could be derived from "cells"
    rows: int  # just for convenience, could be derived from "cells"
    cells: list[Cell]


def structural_similarity(table1: Table, table2: Table) -> float:
    def table_to_tuples(table: Table):
        return {
            (
                cell.row_start,
                cell.row_end,
                cell.column_start,
                cell.column_end,
                cell.content,
            )
            for cell in table.cells
        }

    def severity_penalty(cell_content: str):
        return 0.5 if cell_content == "" else 1.0

    gt_set = table_to_tuples(table1)
    ext_set = table_to_tuples(table2)

    tp = len(gt_set.intersection(ext_set))
    fp = sum(severity_penalty(content) for (_, _, _, _, content) in ext_set - gt_set)
    fn = sum(severity_penalty(content) for (_, _, _, _, content) in gt_set - ext_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return f1_score


files: list[tuple[str, str]] = [
    (
        "https://www.msvracing.com/media/9982/51-british-gt-warm-up-classification.pdf",
        "51-british-gt-warm-up-classification.pdf",
    )
]


def get_pdf_from_url(url: str, name: str) -> bytes:
    """
    Download a PDF from a URL and return its contents.

    This function makes sure the PDF is not downloaded too often.
    This function is a last resort for PDF files where we are uncertain if
    we may add it for testing purposes to https://github.com/py-pdf/sample-files

    :param str url: location of the PDF file
    :param str name: unique name across all files
    """
    if url.startswith("file://"):
        with open(url[7:].replace("\\", "/"), "rb") as fp:
            return fp.read()
    cache_dir = os.path.join(os.path.dirname(__file__), "pdf_cache")
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    cache_path = os.path.join(cache_dir, name)
    if not os.path.exists(cache_path):
        ssl._create_default_https_context = ssl._create_unverified_context
        with urllib.request.urlopen(url) as response, open(
            cache_path, "wb"
        ) as out_file:
            out_file.write(response.read())
    with open(cache_path, "rb") as fp:
        data = fp.read()
    return data


def extract_tables_with_tabula(data: bytes) -> dict[int, list[Table]]:
    from tabula import read_pdf  # pip install tabula-py

    stream = BytesIO(data)

    # Convert raw tables to Table objects, grouped by page index
    tables_by_page: dict[int, list[Table]] = {}
    reader = PdfReader(stream)
    for page_index in range(len(reader.pages)):
        raw_tables = read_pdf(
            stream,
            pages=page_index + 1,
            multiple_tables=True,
            stream=True,
            output_format="json",
            silent=False,
        )
        for table_info in raw_tables:
            raw_table = table_info["data"]
            rows = len(raw_table)
            columns = len(raw_table[0]) if rows > 0 else 0

            cells = []
            for row in range(rows):
                for col in range(columns):
                    content = str(raw_table[row][col]["text"])
                    cell = Cell(
                        row_start=row,
                        row_end=row + 1,
                        column_start=col,
                        column_end=col + 1,
                        content=content,
                    )
                    cells.append(cell)

            table = Table(columns=columns, rows=rows, cells=cells)

            if page_index not in tables_by_page:
                tables_by_page[page_index] = []
            tables_by_page[page_index].append(table)
    return tables_by_page


def derive_table_from_layouted_text(layouted_text: str):
    whitespaces = []
    rows = layouted_text.split("\n")
    max_row_len = max([len(row) for row in rows])
    for row in rows:
        whitespace_row = []
        for i, char in enumerate(row):
            if char == " ":
                whitespaces.append(i)
        # If the row is empty, add whitespaces
        for i in range(len(row), max_row_len):
            whitespace_row.append(i)
        whitespaces.append(whitespace_row)

    # Get columns that have whitespace in all rows
    whitespace_rows = set(range(max_row_len))
    for row in whitespaces:
        whitespace_rows = whitespace_rows & set(row)


def get_truth(sink: str) -> dict[int, list[Table]]:
    with open(Path("ground_truth") / sink) as fp:
        data = json.load(fp)
    gt = {}
    for page_index, tables in data.items():
        gt[int(page_index)] = []
        for table in tables:
            t = Table.parse_obj(table)
            gt[int(page_index)].append(t)
    return gt


def write_extracted_tables(data: dict[int, list[Table]], target: Path):
    with open(target, "w") as fp:
        fp.write(
            json.dumps(
                {
                    page: [t.dict() for t in table_list]
                    for page, table_list in data.items()
                },
                indent=4,
            )
        )


algorithms: list[tuple[str, Callable[[bytes], dict[int, list[Table]]]]] = [
    ("tabula", extract_tables_with_tabula)
]


def main():
    for url, sink in files:
        data = get_pdf_from_url(url, sink)
        name = sink.replace(".pdf", ".json")
        truth = get_truth(name)

        nb_total_tables = sum(len(page_tables) for _, page_tables in truth.items())

        for alg_name, extraction_alg in algorithms:
            tables = extraction_alg(data)
            write_extracted_tables(tables, Path(alg_name) / name)
            nb_alg = sum(len(page_tables) for _, page_tables in tables.items())
            print(f"{alg_name} found : {nb_alg}")
            print(f"Actual tables: {nb_total_tables}")
            for page_index in truth.keys():
                for actual, found in zip(truth[page_index], tables[int(page_index)]):
                    print(f"Score: {structural_similarity(actual, found)}")


if __name__ == "__main__":
    main()
