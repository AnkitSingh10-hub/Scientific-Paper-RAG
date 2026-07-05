import os

from read_pdf import extract_text
from chunking import fixed_chunks
from embeddings import embed_chunks
from vector_store import store


def ingest(pdf_path):
    print(f"Reading {pdf_path} ...")
    text = extract_text(pdf_path)

    print("Chunking text ...")
    chunks = fixed_chunks(text)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks ...")
    embeddings = embed_chunks(chunks)

    print("Storing in Chroma ...")
    paper_name = os.path.basename(pdf_path)
    store(chunks, embeddings, paper_name=paper_name)

    print("Done. Chunks stored:", len(chunks))


if __name__ == "__main__":
    ingest("data/papers/AgenticAI.pdf")
