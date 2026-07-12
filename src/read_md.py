import re


md_path = "/mnt/user-data/uploads/AgenticAI.md"

# Matches the markdown-ified citation links produced by the PDF->MD conversion,
# e.g. "[[1, 2]]([1, 2])" in body text or "[[8]](https://example.com/8)" in tables.
CITATION_PATTERN = re.compile(r"\[\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]\]\([^)]*\)")

CAPTION_PATTERN = re.compile(r"^(\*{0,2})(Figure|Table)\s+\d+[:.]", re.IGNORECASE)
PAGE_TAG_PATTERN = re.compile(
    r"<page_number>\s*\d+\s*</page_number>", re.IGNORECASE | re.DOTALL
)
PAGE_NUM_PATTERN = re.compile(r"^\d{1,4}$")
ARXIV_LINE_PATTERN = re.compile(r"^arXiv:\S+")
SECTION_NUM_ONLY = re.compile(
    r"^#{0,6}\s*\d+(\.\d+)*\s*$"
)  # bare heading number, e.g. "1.2"
HEADING_PATTERN = re.compile(r"^#{1,6}\s+")


def extract_paragraphs(md_path):
    """
    Read a markdown export of the paper and return body-paragraph text only,
    dropping:
      - <page_number>N</page_number> marker lines
      - standalone page-number lines left behind by the PDF->MD conversion
      - the arXiv identifier line
      - bare section-number-only lines
    Headings are kept (stripped of leading '#'s) since they carry useful
    context for chunking/retrieval. HTML tables are kept as-is.
    Returns a list of paragraph/block strings, split on blank lines.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Strip page-number tags up front: the PDF->MD conversion sometimes
    # spreads them across blank lines (e.g. "<page_number>\n\n29\n</page_number>"),
    # which would otherwise get split into separate junk blocks below.
    text = PAGE_TAG_PATTERN.sub("", text)

    # Blocks are separated by one or more blank lines. This keeps multi-line
    # blocks (e.g. <table>...</table>) intact as a single unit, since there
    # are no blank lines inside them.
    raw_blocks = re.split(r"\n\s*\n", text)

    paragraphs = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        if len(block.splitlines()) == 1:
            only = block.strip()
            if (
                PAGE_NUM_PATTERN.match(only)
                or ARXIV_LINE_PATTERN.match(only)
                or SECTION_NUM_ONLY.match(only)
            ):
                continue

        # Figure/Table captions: keep the explanatory text, drop the boilerplate label.
        first_line, *rest = block.splitlines()
        if CAPTION_PATTERN.match(first_line):
            remainder = CAPTION_PATTERN.sub("", first_line, count=1).strip()
            remainder = re.sub(r"^[.:]\s*", "", remainder)
            remainder = remainder.rstrip("*").strip()
            block = "\n".join(([remainder] if remainder else []) + rest).strip()
            if not block:
                continue

        # Keep headings but drop the leading '#' markers.
        if HEADING_PATTERN.match(block):
            paragraphs.append(block)
            continue

        paragraphs.append(block)

    return paragraphs


def remove_citations(text):
    cleaned = CITATION_PATTERN.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip()


def clean_md_for_rag(md_path, drop_references=True):
    paragraphs = extract_paragraphs(md_path)

    cleaned_paragraphs = []
    for p in paragraphs:
        if drop_references and p.strip() == "References":
            break  # everything after this is the bibliography -> stop
        cleaned_paragraphs.append(remove_citations(p))

    # drop empty / very short junk paragraphs (e.g. leftover single characters)
    cleaned_paragraphs = [p for p in cleaned_paragraphs if len(p) > 3]

    return "\n\n".join(cleaned_paragraphs)


if __name__ == "__main__":
    text = clean_md_for_rag(md_path)
    print(text[:3000])
    print("\n...\n")
    print(f"Total length: {len(text)} chars")
    with open("/home/claude/cleaned_preview.txt", "w") as f:
        f.write(text)
