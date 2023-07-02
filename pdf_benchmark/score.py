from Levenshtein import ratio  # python-Levenshtein

from pdf_benchmark.data_structures import Document


def load_extracted_data(path: str) -> str:
    with open(path) as fp:
        return fp.read()


def get_text_extraction_score(doc: Document, library_name: str):
    gt_data = load_extracted_data(f"read/extraction-ground-truth/{doc.name}.txt")
    extracted_data = load_extracted_data(f"read/results/{library_name}/{doc.name}.txt")
    return ratio(gt_data, extracted_data)
