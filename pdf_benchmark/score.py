import re

import pypdfium2 as pdfium
from Levenshtein import ratio  # python-Levenshtein

from pdf_benchmark.data_structures import Document


def load_extracted_data(path: str) -> str:
    with open(path) as fp:
        return fp.read()


def get_text_extraction_score(doc: Document, library_name: str):
    gt_data = load_extracted_data(f"read/extraction-ground-truth/{doc.name}.txt")
    extracted_data = load_extracted_data(f"read/results/{library_name}/{doc.name}.txt")
    return ratio(gt_data, extracted_data)


def get_page_texts(data: bytes) -> list[str]:
    """Return the text of every page, extracted with pypdfium2."""
    pdf = pdfium.PdfDocument(data)
    return [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]


def count_pages_without_watermark(
    watermark_data: bytes, data: bytes, watermarked: bytes
) -> int:
    """
    Return how many pages of the watermarked document lack the watermark text.

    Pages missing from the watermarked document count as well. pypdfium2,
    which does not take part in the watermarking benchmark, extracts the text.
    Every line of the watermark is searched for on its own and whitespace is
    ignored, as the text of the page can end up between two lines.
    """
    [watermark] = get_page_texts(watermark_data)
    lines = [re.sub(r"\s+", "", line) for line in watermark.splitlines()]
    lines = [line for line in lines if line]
    texts = [re.sub(r"\s+", "", text) for text in get_page_texts(watermarked)]
    missing_pages = max(0, len(pdfium.PdfDocument(data)) - len(texts))
    return missing_pages + sum(
        not all(line in text for line in lines) for text in texts
    )
