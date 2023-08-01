def postprocess(extracted_texts: list[str]) -> str:
    """Pass a list of all extracted texts from all pages."""
    extracted_texts = [replace_ligatures(t) for t in extracted_texts]
    extracted_texts = [remove_hyphens(t) for t in extracted_texts]
    # footer_remover = FooterRemover()
    # footer_remover.fit(extracted_texts)
    # extracted_texts = [footer_remover.extract(t) for t in extracted_texts]

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
        if line.endswith("-"):
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


class FooterRemover:
    def __init__(self):
        self.footer = None

    def fit(self, extracted_texts: list[str]) -> None:
        """
        Find the common footer by comparing all extracted texts
        and finding the common suffix.
        We assume that the footer appears at the end of each text.
        """
        common_suffix = None
        for text in extracted_texts:
            if common_suffix is None:
                common_suffix = text
            else:
                i = 1
                while i <= min(len(common_suffix), len(text)):
                    if common_suffix[-i:] != text[-i:]:
                        break
                    i += 1
                common_suffix = common_suffix[-(i - 1) :]

        self.footer = common_suffix

    def extract(self, extracted_text: str) -> str:
        """Remove the detected footer from the extracted text."""
        if self.footer is not None and extracted_text.endswith(self.footer):
            return extracted_text[: -len(self.footer)]
        return extracted_text
