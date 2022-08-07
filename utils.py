"""
Credits to https://stackoverflow.com/a/15445930/562769 who generated the core
of this.
"""

# Generates tables for Doxygen flavored Markdown.  See the Doxygen
# documentation for details:
#   http://www.doxygen.nl/manual/markdown.html#md_tables

# Translation dictionaries for table alignment
left_rule = {"<": ":", "^": ":", ">": "-"}
right_rule = {"<": "-", "^": ":", ">": ":"}


def evaluate_field(record, field_spec):
    """
    Evaluate a field of a record using the type of the field_spec as a guide.
    """
    if type(field_spec) is int:
        return str(record[field_spec])
    elif type(field_spec) is str:
        return str(getattr(record, field_spec))
    else:
        return str(field_spec(record))


def table_to_markdown(records, headings=None, fields=None, alignment=None) -> str:
    """
    Generate a Doxygen-flavor Markdown table from records.

    file -- Any object with a 'write' method that takes a single string
        parameter.
    records -- Iterable.  Rows will be generated from this.
    fields -- List of fields for each row.  Each entry may be an integer,
        string or a function.  If the entry is an integer, it is assumed to be
        an index of each record.  If the entry is a string, it is assumed to be
        a field of each record.  If the entry is a function, it is called with
        the record and its return value is taken as the value of the field.
    headings -- List of column headings.
    alignment - List of pairs alignment characters.  The first of the pair
        specifies the alignment of the header (Doxygen won't respect this, but
        it might look good), the second specifies the alignment of the cells in
        the column.

        Possible alignment characters are:
            '<' = Left align (default for cells)
            '>' = Right align
            '^' = Center (default for column headings)
    """
    if headings is None:
        headings = ["" for _ in range(len(headings))]
    if fields is None:
        fields = list(range(len(headings)))

    num_columns = len(fields)
    assert len(headings) == num_columns

    # Compute the table cell data
    columns = [[] for _ in range(num_columns)]
    for record in records:
        for i, field in enumerate(fields):
            columns[i].append(evaluate_field(record, field))

    # Fill out any missing alignment characters.
    extended_align = alignment if alignment is not None else []
    if len(extended_align) > num_columns:
        extended_align = extended_align[0:num_columns]
    elif len(extended_align) < num_columns:
        extended_align += [("^", "<") for i in range(num_columns - len(extended_align))]

    heading_align, cell_align = [x for x in zip(*extended_align)]

    field_widths = [
        len(max(column, key=len)) if len(column) > 0 else 0 for column in columns
    ]
    heading_widths = [max(len(head), 2) for head in headings]
    column_widths = [max(x) for x in zip(field_widths, heading_widths)]

    _ = " | ".join(
        ["{:" + a + str(w) + "}" for a, w in zip(heading_align, column_widths)]
    )
    heading_template = "| " + _ + " |"
    _ = " | ".join(["{:" + a + str(w) + "}" for a, w in zip(cell_align, column_widths)])
    row_template = "| " + _ + " |"

    _ = " | ".join(
        [
            left_rule[a] + "-" * (w - 2) + right_rule[a]
            for a, w in zip(cell_align, column_widths)
        ]
    )
    ruling = "| " + _ + " |"

    ret_val = ""

    ret_val += heading_template.format(*headings).rstrip() + "\n"
    ret_val += ruling.rstrip() + "\n"
    for row in zip(*columns):
        ret_val += row_template.format(*row).rstrip() + "\n"
    return ret_val


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
