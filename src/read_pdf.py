import fitz
import re


pdf_path = "/mnt/user-data/uploads/AgenticAI.pdf"

CITATION_PATTERN = re.compile(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]")
CAPTION_PATTERN = re.compile(r"^(Figure|Table)\s+\d+[:.]", re.IGNORECASE)
PAGE_NUM_PATTERN = re.compile(r"^\d{1,4}$")
ARXIV_LINE_PATTERN = re.compile(r"^arXiv:\S+")
SECTION_NUM_ONLY = re.compile(r"^\d+(\.\d+)*$")  # e.g. "1.2" heading number alone


def is_horizontal(line_dir, tol=0.05):
    dx, dy = line_dir
    return abs(dy) < tol  # roughly pointing along x-axis


def extract_paragraphs(pdf_path):
    """
    Extract body-paragraph text only, dropping:
      - rotated text (sideways figure captions)
      - normal Figure/Table caption lines
      - standalone page-number lines
      - the arXiv identifier line
      - bare section-number lines
    Returns a list of paragraph strings (one block per PDF text block).
    """
    doc = fitz.open(pdf_path)
    paragraphs = []

    for page in doc:
        d = page.get_text("dict")
        for block in d["blocks"]:
            if block["type"] != 0:  # skip images
                continue

            lines_text = []
            skip_block = False

            for line in block["lines"]:
                if not is_horizontal(line["dir"]):
                    skip_block = True  # rotated caption -> drop whole block
                    break
                text = "".join(span["text"] for span in line["spans"]).strip()
                if text:
                    lines_text.append(text)

            if skip_block or not lines_text:
                continue

            # Drop blocks that are pure noise.
            # NOTE: figure/table captions in this paper are often full
            # explanatory paragraphs, not just a "Figure N:" label - so we
            # only strip the boilerplate label prefix from the first line
            # instead of discarding the whole block (which used to delete
            # real content, e.g. the PRISMA counts embedded in the Figure 7
            # caption).
            first_line = lines_text[0]
            if CAPTION_PATTERN.match(first_line):
                remainder = CAPTION_PATTERN.sub("", first_line, count=1).strip()
                # also strip a leading ". " or ": " left behind after the label removal
                remainder = re.sub(r"^[.:]\s*", "", remainder)
                lines_text = ([remainder] if remainder else []) + lines_text[1:]
                if not lines_text:
                    continue

            if len(lines_text) == 1:
                only = lines_text[0]
                if (
                    PAGE_NUM_PATTERN.match(only)
                    or ARXIV_LINE_PATTERN.match(only)
                    or SECTION_NUM_ONLY.match(only)
                ):
                    continue

            # Join wrapped lines into one paragraph, fixing hyphenation
            joined = ""
            for line in lines_text:
                if joined.endswith("-"):
                    if line and line[0].islower():
                        # true line-wrap hyphenation (e.g. "informa-" + "tion")
                        joined = joined[:-1] + line
                    else:
                        # next word starts uppercase -> this is very likely a
                        # genuine compound-word hyphen (e.g. "Retrieval-" +
                        # "Augmented") that just happened to fall at a line
                        # break. Keep the hyphen, don't insert a space, so we
                        # don't produce a broken "Retrieval- Augmented" token.
                        joined = joined + line
                else:
                    joined = (joined + " " + line).strip() if joined else line

            paragraphs.append(joined)

    doc.close()
    return paragraphs


def remove_citations(text):
    cleaned = CITATION_PATTERN.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip()


def clean_pdf_for_rag(pdf_path, drop_references=True):
    paragraphs = extract_paragraphs(pdf_path)

    cleaned_paragraphs = []
    for p in paragraphs:
        if drop_references and p.strip() == "References":
            break  # everything after this is the bibliography -> stop
        cleaned_paragraphs.append(remove_citations(p))

    # drop empty / very short junk paragraphs (e.g. leftover single characters)
    cleaned_paragraphs = [p for p in cleaned_paragraphs if len(p) > 3]

    return "\n\n".join(cleaned_paragraphs)


if __name__ == "__main__":
    text = clean_pdf_for_rag(pdf_path)
    print(text[:3000])
    print("\n...\n")
    print(f"Total length: {len(text)} chars")
    with open("/home/claude/cleaned_preview.txt", "w") as f:
        f.write(text)
