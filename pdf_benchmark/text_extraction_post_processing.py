PDFIUM_ZERO_WIDTH_NO_BREAK_SPACE = "\ufffe"

def postprocess(extracted_texts: list[str], page_labels: list[str]) -> str:
    """Pass a list of all extracted texts from all pages."""
    extracted_texts = [replace_ligatures(t) for t in extracted_texts]
    extracted_texts = [remove_hyphens(t) for t in extracted_texts]
    extracted_texts = remove_footer(extracted_texts, page_labels)
    return "\n".join(extracted_texts)


def replace_ligatures(text: str) -> str:
    ligatures_dict = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
        # "Ꜳ": "AA",
        # "Æ": "AE",
        "ꜳ": "aa",
    }
    for search, replace in ligatures_dict.items():
        text = text.replace(search, replace)
    return text


def remove_hyphens(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]

    # Find dashes
    line_numbers = []
    for line_no, line in enumerate(lines[:-1]):
        if line.endswith("-") or line.endswith(PDFIUM_ZERO_WIDTH_NO_BREAK_SPACE):
            line_numbers.append(line_no)

    # Replace
    for line_no in line_numbers:
        lines = dehyphenate(lines, line_no)

    return "\n".join(lines)


def dehyphenate(lines: list[str], line_no: int) -> list[str]:
    next_line = lines[line_no + 1]
    word_suffix = next_line.split(" ")[0]

    lines[line_no] = lines[line_no][:-1] + word_suffix
    lines[line_no + 1] = lines[line_no + 1][len(word_suffix) :]
    return lines


def remove_footer(extracted_texts: list[str], page_labels: list[str]):
    def remove_page_labels(extracted_texts, page_labels):
        processed = []
        for text, label in zip(extracted_texts, page_labels):
            text_left = text.lstrip()
            if text_left.startswith(label):
                text = text_left[len(label) :]

            text_right = text.rstrip()
            if text_right.endswith(label):
                text = text_right[: -len(label)]

            processed.append(text)
        return processed

    extracted_texts = remove_page_labels(extracted_texts, page_labels)
    return extracted_texts
