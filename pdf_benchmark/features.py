"""
Small PDFs that each contain one feature of text or images.

Every PDF is written here, so the expected text or picture is known. The
feature matrices of the report show which library extracts what; they explain
differences of the image counts and of the text extraction quality.
"""

import base64
import io
import re
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

# --- A minimal PDF writer --------------------------------------------------


@dataclass(frozen=True)
class Name:
    value: str


@dataclass(frozen=True)
class Ref:
    number: int


@dataclass(frozen=True)
class Stream:
    dictionary: dict[str, Any]
    data: bytes


def serialize(value: Any) -> bytes:
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if isinstance(value, (int, float)):
        return str(round(value, 4)).encode()
    if isinstance(value, Name):
        return b"/" + value.value.encode()
    if isinstance(value, Ref):
        return f"{value.number} 0 R".encode()
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return b"(" + escaped.encode("latin-1") + b")"
    if isinstance(value, bytes):
        return b"<" + value.hex().upper().encode() + b">"
    if isinstance(value, list):
        return b"[" + b" ".join(serialize(item) for item in value) + b"]"
    if isinstance(value, dict):
        entries = b" ".join(
            b"/" + key.encode() + b" " + serialize(item) for key, item in value.items()
        )
        return b"<< " + entries + b" >>"
    raise TypeError(f"Cannot serialize {value!r}")


class PdfFile:
    """Collect the objects of a PDF with one page."""

    def __init__(self) -> None:
        self.objects: list[Any] = []

    def add(self, value: Any) -> Ref:
        self.objects.append(value)
        return Ref(len(self.objects))

    def write(self, content: bytes, resources: dict, size: tuple[int, int]) -> bytes:
        contents = self.add(Stream({}, content))
        pages = Ref(len(self.objects) + 2)
        page = self.add(
            {
                "Type": Name("Page"),
                "Parent": pages,
                "MediaBox": [0, 0, *size],
                "Resources": resources,
                "Contents": contents,
            }
        )
        self.add({"Type": Name("Pages"), "Kids": [page], "Count": 1})
        root = self.add({"Type": Name("Catalog"), "Pages": pages})
        body = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for number, value in enumerate(self.objects, start=1):
            offsets.append(len(body))
            body += f"{number} 0 obj\n".encode()
            if isinstance(value, Stream):
                dictionary = {**value.dictionary, "Length": len(value.data)}
                body += serialize(dictionary) + b"\nstream\n" + value.data
                body += b"\nendstream"
            else:
                body += serialize(value)
            body += b"\nendobj\n"
        xref = len(body)
        body += f"xref\n0 {len(self.objects) + 1}\n0000000000 65535 f\r\n".encode()
        for offset in offsets:
            body += f"{offset:010d} 00000 n\r\n".encode()
        trailer = {"Size": len(self.objects) + 1, "Root": root}
        body += b"trailer\n" + serialize(trailer)
        body += f"\nstartxref\n{xref}\n%%EOF\n".encode()
        return bytes(body)


Build = Callable[[PdfFile], tuple[bytes, dict]]

# --- Text features ---------------------------------------------------------

HELVETICA = {
    "Type": Name("Font"),
    "Subtype": Name("Type1"),
    "BaseFont": Name("Helvetica"),
    "Encoding": Name("WinAnsiEncoding"),
}


# Invisible characters that mark the direction of right-to-left text
BIDI_CONTROLS = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def normalize(text: str) -> str:
    """Join all whitespace to a single space and drop direction marks."""
    return " ".join(BIDI_CONTROLS.sub("", text).split())


@dataclass(frozen=True)
class TextFeature:
    name: str
    description: str
    build: Build
    expected: str
    # Whether the extracted text is right; by default the text with
    # normalized whitespace must be the expected text.
    check: Callable[[str], bool] | None = None

    def pdf(self) -> bytes:
        doc = PdfFile()
        content, resources = self.build(doc)
        return doc.write(content, resources, (400, 200))

    def is_correct(self, text: str) -> bool:
        if self.check:
            return self.check(text)
        return normalize(text) == self.expected


def helvetica_text(operators: str) -> Build:
    """Show text with Helvetica as /F1."""
    return lambda doc: (operators.encode("latin-1"), {"Font": {"F1": HELVETICA}})


def cid_font(doc: PdfFile, characters: str, base_font: str) -> Ref:
    """Return a Type0 font with two-byte codes 1, 2, ... for the characters.

    The font program is not embedded, and only the ToUnicode CMap tells which
    character a code is (ISO 32000-2:2020, 9.10.3).
    """
    mappings = "".join(
        f"<{code:04X}> <{ord(char):04X}>\n"
        for code, char in enumerate(characters, start=1)
    )
    cmap = (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        f"{len(characters)} beginbfchar\n{mappings}endbfchar\n"
        "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    )
    descriptor = doc.add(
        {
            "Type": Name("FontDescriptor"),
            "FontName": Name(base_font),
            "Flags": 4,
            "FontBBox": [0, -200, 1000, 900],
            "ItalicAngle": 0,
            "Ascent": 900,
            "Descent": -200,
            "CapHeight": 700,
            "StemV": 80,
        }
    )
    descendant = doc.add(
        {
            "Type": Name("Font"),
            "Subtype": Name("CIDFontType2"),
            "BaseFont": Name(base_font),
            "CIDSystemInfo": {
                "Registry": "Adobe",
                "Ordering": "Identity",
                "Supplement": 0,
            },
            "FontDescriptor": descriptor,
            "DW": 1000,
            "CIDToGIDMap": Name("Identity"),
        }
    )
    return doc.add(
        {
            "Type": Name("Font"),
            "Subtype": Name("Type0"),
            "BaseFont": Name(base_font),
            "Encoding": Name("Identity-H"),
            "DescendantFonts": [descendant],
            "ToUnicode": doc.add(Stream({}, cmap.encode())),
        }
    )


def build_cjk(doc: PdfFile) -> tuple[bytes, dict]:
    font = cid_font(doc, "日本語", "MS-Mincho")
    return b"BT /C1 24 Tf 20 100 Td <000100020003> Tj ET", {"Font": {"C1": font}}


def build_hebrew(doc: PdfFile) -> tuple[bytes, dict]:
    # The glyphs are shown from left to right, as most programs write them: the
    # last letter first.
    font = cid_font(doc, "םולש", "Arial")
    return b"BT /C1 24 Tf 20 100 Td <0001000200030004> Tj ET", {"Font": {"C1": font}}


def build_glyph_names(doc: PdfFile) -> tuple[bytes, dict]:
    font = {
        "Type": Name("Font"),
        "Subtype": Name("Type1"),
        "BaseFont": Name("Symbol"),
        "Encoding": {
            "Type": Name("Encoding"),
            "Differences": [
                65,
                Name("alpha"),
                Name("beta"),
                Name("summation"),
                Name("minus"),
            ],
        },
    }
    return b"BT /S1 24 Tf 20 100 Td (ABCD) Tj ET", {"Font": {"S1": font}}


def build_type3(doc: PdfFile) -> tuple[bytes, dict]:
    h = doc.add(
        Stream(
            {},
            b"750 0 0 0 650 700 d1 0 0 120 700 re f 530 0 120 700 re f"
            b" 120 300 410 100 re f",
        )
    )
    i = doc.add(
        Stream({}, b"300 0 0 0 170 700 d1 50 0 120 500 re f 50 580 120 120 re f")
    )
    font = {
        "Type": Name("Font"),
        "Subtype": Name("Type3"),
        "FontBBox": [0, 0, 750, 700],
        "FontMatrix": [0.001, 0, 0, 0.001, 0, 0],
        "CharProcs": {"H": h, "i": i},
        "Encoding": {
            "Type": Name("Encoding"),
            "Differences": [72, Name("H"), 105, Name("i")],
        },
        "FirstChar": 72,
        "LastChar": 105,
        "Widths": [750] + [0] * 32 + [300],
        "Resources": {},
    }
    return b"BT /T3 24 Tf 20 100 Td (Hi) Tj ET", {"Font": {"T3": font}}


def build_form_xobject(doc: PdfFile) -> tuple[bytes, dict]:
    form = doc.add(
        Stream(
            {
                "Type": Name("XObject"),
                "Subtype": Name("Form"),
                "BBox": [0, 0, 400, 200],
                "Resources": {"Font": {"F1": HELVETICA}},
            },
            b"BT /F1 12 Tf 20 100 Td (Text in a form) Tj ET",
        )
    )
    return b"/Fm1 Do", {"XObject": {"Fm1": form}}


COLUMNS = (
    [
        "The left column starts",
        "here and continues in",
        "two more short lines",
        "until it ends here.",
    ],
    [
        "The right column comes",
        "after the left one and",
        "has to be read after",
        "the whole left column.",
    ],
)


def build_columns(doc: PdfFile) -> tuple[bytes, dict]:
    # The lines are written row by row: left, right, left, right, ...
    operators = ""
    for row, lines in enumerate(zip(*COLUMNS, strict=True)):
        for x, line in zip((20, 220), lines, strict=True):
            operators += f"BT /F1 10 Tf {x} {160 - 14 * row} Td ({line}) Tj ET\n"
    return operators.encode(), {"Font": {"F1": HELVETICA}}


def is_column_order(text: str) -> bool:
    normalized = normalize(text)
    positions = [normalized.find(line) for line in COLUMNS[0] + COLUMNS[1]]
    return -1 not in positions and positions == sorted(positions)


def has_two_lines(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [normalize(line) for line in lines] == ["First line", "Second line"]


TEXT_FEATURES = [
    TextFeature(
        "Simple text",
        "A line in a standard font, as a reference.",
        helvetica_text("BT /F1 12 Tf 20 100 Td (Hello World) Tj ET"),
        "Hello World",
    ),
    TextFeature(
        "Line break",
        "Two lines; the text must have a line break between them.",
        helvetica_text(
            "BT /F1 12 Tf 20 150 Td (First line) Tj 0 -16 Td (Second line) Tj ET"
        ),
        "First line\nSecond line",
        check=has_two_lines,
    ),
    TextFeature(
        "Space from a TJ offset",
        "No space character; the words are apart by an offset in a TJ array,"
        " as pdfTeX writes most spaces.",
        helvetica_text("BT /F1 12 Tf 20 100 Td [(Hello) -500 (World)] TJ ET"),
        "Hello World",
    ),
    TextFeature(
        "Space from the position",
        "No space character; every word is placed on its own.",
        helvetica_text(
            "BT /F1 12 Tf 20 100 Td (Hello) Tj ET BT /F1 12 Tf 55 100 Td (World) Tj ET"
        ),
        "Hello World",
    ),
    TextFeature(
        "WinAnsiEncoding",
        "Characters of the codes 128 to 255 in WinAnsiEncoding.",
        helvetica_text("BT /F1 12 Tf 20 100 Td (\x80 \x93\xe4\x94) Tj ET"),
        "€ “ä”",
    ),
    TextFeature(
        "Glyph names",
        "No ToUnicode CMap; the /Differences of the encoding name the glyphs"
        " (alpha, beta, summation, minus), as in many fonts for math.",
        build_glyph_names,
        "αβ∑−",
    ),
    TextFeature(
        "CID font",
        "Two-byte codes of a Type0 font, mapped to Unicode by a ToUnicode CMap.",
        build_cjk,
        "日本語",
    ),
    TextFeature(
        "Right to left",
        "Hebrew letters, shown from left to right in the order they appear,"
        " so the last letter first.",
        build_hebrew,
        "שלום",
    ),
    TextFeature(
        "Type 3 font",
        "A font whose glyphs are drawn by content streams, with glyph names.",
        build_type3,
        "Hi",
    ),
    TextFeature(
        "Form XObject",
        "Text inside a form XObject.",
        build_form_xobject,
        "Text in a form",
    ),
    TextFeature(
        "Invisible text",
        "Text with rendering mode 3 (invisible), like the text layer of a"
        " scanned page.",
        helvetica_text("BT 3 Tr /F1 12 Tf 20 100 Td (Invisible text) Tj ET"),
        "Invisible text",
    ),
    TextFeature(
        "Rotated text",
        "A line rotated by 90 degrees.",
        helvetica_text("BT /F1 12 Tf 0 1 -1 0 150 40 Tm (Rotated text) Tj ET"),
        "Rotated text",
    ),
    TextFeature(
        "Superscript",
        "A smaller digit raised with Ts.",
        helvetica_text("BT /F1 12 Tf 20 100 Td (E = mc) Tj /F1 8 Tf 5 Ts (2) Tj ET"),
        "E = mc2",
    ),
    TextFeature(
        "Fake bold",
        "The same word drawn twice, 0.4 points apart, to look bold.",
        helvetica_text(
            "BT /F1 12 Tf 20 100 Td (Bold) Tj ET BT /F1 12 Tf 20.4 100 Td (Bold) Tj ET"
        ),
        "Bold",
    ),
    TextFeature(
        "Two columns",
        "Two columns whose lines are written row by row; the left column must"
        " come first.",
        build_columns,
        " / ".join(COLUMNS[0] + COLUMNS[1]),
        check=is_column_order,
    ),
]

# --- Image features --------------------------------------------------------

WIDTH, HEIGHT = 48, 32


def make_picture() -> Image.Image:
    """Return an RGB picture with gradients and a dark disk."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    disk = (x - 30) ** 2 + (y - 14) ** 2 < 64
    red = np.where(disk, 30, x * 5)
    green = np.where(disk, 20, y * 7)
    blue = np.where(disk, 60, 220 - x * 3)
    return Image.fromarray(np.dstack([red, green, blue]).astype(np.uint8), "RGB")


def make_bitmap() -> Image.Image:
    """Return a black and white picture: a black disk and a bar on white."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    black = ((x - 16) ** 2 + (y - 16) ** 2 < 100) | ((x > 32) & (x < 38))
    return Image.fromarray(np.where(black, 0, 255).astype(np.uint8), "L").convert("1")


PICTURE = make_picture()
BITMAP = make_bitmap()
PALETTE_PICTURE = PICTURE.quantize(16)


def image_xobject(data: bytes, **entries: Any) -> Stream:
    return Stream(
        {
            "Type": Name("XObject"),
            "Subtype": Name("Image"),
            "Width": WIDTH,
            "Height": HEIGHT,
            "BitsPerComponent": 8,
            **entries,
        },
        data,
    )


def draw(*names: str) -> bytes:
    """Draw the image XObjects next to each other."""
    return b"".join(
        f"q 144 0 0 96 {20 + 160 * index} 20 cm /{name} Do Q\n".encode()
        for index, name in enumerate(names)
    )


def page_with(image: Stream) -> Build:
    def build(doc: PdfFile) -> tuple[bytes, dict]:
        return draw("Im1"), {"XObject": {"Im1": doc.add(image)}}

    return build


def rgb_flate() -> Stream:
    return image_xobject(
        zlib.compress(PICTURE.tobytes()),
        ColorSpace=Name("DeviceRGB"),
        Filter=Name("FlateDecode"),
    )


def jpeg(picture: Image.Image, color_space: str) -> Stream:
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=95)
    return image_xobject(
        buffer.getvalue(), ColorSpace=Name(color_space), Filter=Name("DCTDecode")
    )


def cmyk_flate() -> Stream:
    rgb = np.asarray(PICTURE).astype(np.int16)
    cmyk = np.dstack([255 - rgb, np.zeros((HEIGHT, WIDTH), np.int16)]).astype(np.uint8)
    return image_xobject(
        zlib.compress(cmyk.tobytes()),
        ColorSpace=Name("DeviceCMYK"),
        Filter=Name("FlateDecode"),
    )


def gray_flate() -> Stream:
    return image_xobject(
        zlib.compress(PICTURE.convert("L").tobytes()),
        ColorSpace=Name("DeviceGray"),
        Filter=Name("FlateDecode"),
    )


def bitmap_flate() -> Stream:
    # In mode "1", Pillow packs the rows with 1 for white, like DeviceGray.
    return image_xobject(
        zlib.compress(BITMAP.tobytes()),
        ColorSpace=Name("DeviceGray"),
        BitsPerComponent=1,
        Filter=Name("FlateDecode"),
    )


def indexed() -> Stream:
    palette = bytes(PALETTE_PICTURE.getpalette()[: 16 * 3])
    return image_xobject(
        zlib.compress(PALETTE_PICTURE.tobytes()),
        ColorSpace=[Name("Indexed"), Name("DeviceRGB"), 15, palette],
        Filter=Name("FlateDecode"),
    )


def rgb_16_bit() -> Stream:
    samples = np.asarray(PICTURE).astype(">u2") * 257
    return image_xobject(
        zlib.compress(samples.tobytes()),
        ColorSpace=Name("DeviceRGB"),
        BitsPerComponent=16,
        Filter=Name("FlateDecode"),
    )


def soft_mask(doc: PdfFile) -> tuple[bytes, dict]:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    alpha = np.where((x < 4) | (y < 4), 0, 255).astype(np.uint8)
    mask = doc.add(
        image_xobject(
            zlib.compress(alpha.tobytes()),
            ColorSpace=Name("DeviceGray"),
            Filter=Name("FlateDecode"),
        )
    )
    image = rgb_flate()
    image = Stream({**image.dictionary, "SMask": mask}, image.data)
    return draw("Im1"), {"XObject": {"Im1": doc.add(image)}}


def stencil_mask(doc: PdfFile) -> tuple[bytes, dict]:
    # Sample value 0 marks the painted pixels (ISO 32000-2:2020, 8.9.6.2).
    image = Stream(
        {
            "Type": Name("XObject"),
            "Subtype": Name("Image"),
            "Width": WIDTH,
            "Height": HEIGHT,
            "ImageMask": True,
            "BitsPerComponent": 1,
            "Filter": Name("FlateDecode"),
        },
        zlib.compress(BITMAP.tobytes()),
    )
    return b"0 0 0.8 rg\n" + draw("Im1"), {"XObject": {"Im1": doc.add(image)}}


def ccitt_group_4() -> Stream:
    buffer = io.BytesIO()
    BITMAP.save(buffer, "TIFF", compression="group4")
    tiff = Image.open(buffer)
    [offset], [length] = tiff.tag_v2[273], tiff.tag_v2[279]
    # Group 4 codes white runs as 0 bits. Pillow stores the bitmap with
    # 0 for black (photometric interpretation 1), so the runs are swapped.
    assert tiff.tag_v2[262] == 1
    return image_xobject(
        buffer.getvalue()[offset : offset + length],
        ColorSpace=Name("DeviceGray"),
        BitsPerComponent=1,
        Filter=Name("CCITTFaxDecode"),
        DecodeParms={"K": -1, "Columns": WIDTH, "Rows": HEIGHT, "BlackIs1": True},
    )


def jpeg_2000() -> Stream:
    buffer = io.BytesIO()
    PICTURE.save(buffer, "JPEG2000", irreversible=False)
    return image_xobject(buffer.getvalue(), Filter=Name("JPXDecode"))


def lzw_encode(data: bytes) -> bytes:
    """Encode the data with LZW, with EarlyChange 1 (ISO 32000-2:2020, 7.4.4)."""
    clear, end = 256, 257
    bits = 0
    length = 0
    output = bytearray()

    def emit(code: int, width: int) -> None:
        nonlocal bits, length
        bits = (bits << width) | code
        length += width
        while length >= 8:
            length -= 8
            output.append((bits >> length) & 0xFF)

    table = {bytes([i]): i for i in range(256)}
    width = 9
    emit(clear, width)
    word = b""
    for byte in data:
        extended = word + bytes([byte])
        if extended in table:
            word = extended
            continue
        emit(table[word], width)
        table[extended] = len(table) + 2
        next_code = len(table) + 2
        if next_code == 4094:
            emit(clear, width)
            table = {bytes([i]): i for i in range(256)}
            width = 9
        elif next_code + 1 > (1 << width) and width < 12:
            width += 1
        word = bytes([byte])
    emit(table[word], width)
    emit(end, width)
    if length:
        output.append((bits << (8 - length)) & 0xFF)
    return bytes(output)


def rgb_lzw() -> Stream:
    return image_xobject(
        lzw_encode(PICTURE.tobytes()),
        ColorSpace=Name("DeviceRGB"),
        Filter=Name("LZWDecode"),
    )


def filter_chain() -> Stream:
    data = base64.a85encode(zlib.compress(PICTURE.tobytes())) + b"~>"
    return image_xobject(
        data,
        ColorSpace=Name("DeviceRGB"),
        Filter=[Name("ASCII85Decode"), Name("FlateDecode")],
    )


def inline_image(doc: PdfFile) -> tuple[bytes, dict]:
    data = PICTURE.tobytes().hex().upper().encode() + b">"
    return (
        b"q 144 0 0 96 20 20 cm\nBI /W 48 /H 32 /CS /RGB /BPC 8 /F /AHx ID\n"
        + data
        + b"\nEI\nQ\n",
        {},
    )


def image_in_form(doc: PdfFile) -> tuple[bytes, dict]:
    image = doc.add(rgb_flate())
    form = doc.add(
        Stream(
            {
                "Type": Name("XObject"),
                "Subtype": Name("Form"),
                "BBox": [0, 0, 1, 1],
                "Resources": {"XObject": {"Im1": image}},
            },
            b"/Im1 Do",
        )
    )
    return b"q 144 0 0 96 20 20 cm /Fm1 Do Q\n", {"XObject": {"Fm1": form}}


def drawn_twice(doc: PdfFile) -> tuple[bytes, dict]:
    return draw("Im1", "Im1"), {"XObject": {"Im1": doc.add(rgb_flate())}}


@dataclass(frozen=True)
class ImageFeature:
    name: str
    description: str
    build: Build
    # The picture the image shows; PICTURE if None
    expected: Image.Image | None = None
    # A stencil mask has no colors: the extracted image may show the painted
    # pixels dark or bright.
    either_polarity: bool = False
    # Whether the number of extracted images is reported instead
    count_only: bool = False

    def pdf(self) -> bytes:
        doc = PdfFile()
        content, resources = self.build(doc)
        return doc.write(content, resources, (360, 140))


IMAGE_FEATURES = [
    ImageFeature(
        "RGB, Flate", "8 bits per component, FlateDecode.", page_with(rgb_flate())
    ),
    ImageFeature("JPEG", "DCTDecode.", page_with(jpeg(PICTURE, "DeviceRGB"))),
    ImageFeature("JPEG 2000", "JPXDecode.", page_with(jpeg_2000())),
    ImageFeature(
        "Grayscale", "DeviceGray.", page_with(gray_flate()), PICTURE.convert("L")
    ),
    ImageFeature("CMYK", "DeviceCMYK.", page_with(cmyk_flate())),
    ImageFeature(
        "Indexed", "A palette of 16 colors.", page_with(indexed()), PALETTE_PICTURE
    ),
    ImageFeature("16 bits", "16 bits per component.", page_with(rgb_16_bit())),
    ImageFeature(
        "1 bit", "Black and white, 1 bit per pixel.", page_with(bitmap_flate()), BITMAP
    ),
    ImageFeature(
        "CCITT Group 4",
        "CCITTFaxDecode, as in scanned documents.",
        page_with(ccitt_group_4()),
        BITMAP,
    ),
    ImageFeature("LZW", "LZWDecode.", page_with(rgb_lzw())),
    ImageFeature(
        "Filter chain", "ASCII85Decode and FlateDecode.", page_with(filter_chain())
    ),
    ImageFeature(
        "Inline image",
        "An image inside the content stream (BI ... ID ... EI).",
        inline_image,
    ),
    ImageFeature(
        "In a form XObject", "An image drawn by a form XObject.", image_in_form
    ),
    ImageFeature("Soft mask", "An image with an /SMask (transparency).", soft_mask),
    ImageFeature(
        "Stencil mask",
        "An /ImageMask, painted with the fill color.",
        stencil_mask,
        BITMAP,
        either_polarity=True,
    ),
    ImageFeature(
        "Drawn twice",
        "The same image XObject drawn twice: the number of extracted images.",
        drawn_twice,
        count_only=True,
    ),
]


def luminance(picture: Image.Image) -> np.ndarray:
    if picture.mode.startswith("I"):
        samples = np.asarray(picture, dtype=float)
        return samples * 255 / max(samples.max(), 1)
    return np.asarray(picture.convert("L"), dtype=float)


def compare_images(
    feature: ImageFeature, images: list[tuple[str, bytes]]
) -> tuple[str, str]:
    """Return the mark of the feature matrix and what went wrong."""
    if feature.count_only:
        return str(len(images)), ""
    if not images:
        return "✗", "no image"
    expected_picture = feature.expected or PICTURE
    expected = luminance(expected_picture)
    problems = []
    for name, data in images:
        try:
            picture = Image.open(io.BytesIO(data))
            picture.load()
        except Exception:
            problems.append(f"{name}: Pillow cannot open it")
            continue
        if picture.size != expected_picture.size:
            problems.append(f"{name}: {picture.width}x{picture.height} pixels")
            continue
        extracted = luminance(picture)
        correlation = np.corrcoef(extracted.ravel(), expected.ravel())[0, 1]
        if feature.either_polarity:
            correlation = abs(correlation)
        if correlation > 0.9:
            return "✓", ""
        problems.append(f"{name}: another picture (correlation {correlation:.2f})")
    return "◐", "; ".join(problems)


def run_text_feature(
    feature: TextFeature, extract: Callable[[bytes], list[str]]
) -> tuple[str, str]:
    """Return the mark of the feature matrix and the extracted text if wrong."""
    try:
        text = "\n".join(extract(feature.pdf()))
    except Exception as exc:
        return "⚠", f"{type(exc).__name__}: {exc}"
    if feature.is_correct(text):
        return "✓", ""
    shown = re.sub(r"\s*\n\s*", " / ", text.strip())
    return "✗", shown[:80]


def run_image_feature(
    feature: ImageFeature, extract: Callable[[bytes], list[tuple[str, bytes]]]
) -> tuple[str, str]:
    try:
        images = extract(feature.pdf())
    except Exception as exc:
        return "⚠", f"{type(exc).__name__}: {exc}"
    return compare_images(feature, images)
