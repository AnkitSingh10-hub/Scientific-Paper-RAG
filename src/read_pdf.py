import fitz


pdf_path = "data/papers/AgenticAI.pdf"


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)

    all_text = ""

    for page in doc:
        all_text += page.get_text()

    doc.close()

    return all_text
