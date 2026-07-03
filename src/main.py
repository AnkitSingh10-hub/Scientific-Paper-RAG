from read_pdf import extract_text
from chunking import chunk_text

pdf_path = "data/papers/AgenticAI.pdf"

text = extract_text(pdf_path)

chunks = chunk_text(text, chunk_size=500)

print(f"Total chunks: {len(chunks)}")

print("\nFirst chunk:\n")

print(chunks[0])
