import os

from read_pdf import clean_pdf_for_rag
from chunking import (
    FixedChunker,
    SentenceChunker,
    CharacterChunker,
    RecursiveCharacterChunker,
    TokenChunker,
)
from embeddings import embed_chunks
from vector_store import store


def ingest(pdf_path, chunker=None):
    if chunker is None:
        chunker = FixedChunker(chunk_size=1500, overlap=100)

    print(f"Reading {pdf_path} ...")
    text = clean_pdf_for_rag(pdf_path)

    print(f"Chunking text with {type(chunker).__name__} ...")
    chunks = chunker.chunk(text)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks ...")
    embeddings = embed_chunks(chunks)

    print("Storing in Chroma ...")
    paper_name = os.path.basename(pdf_path)
    store(chunks, embeddings, paper_name=paper_name)

    print("Done. Chunks stored:", len(chunks))


if __name__ == "__main__":
    ingest(
        "data/papers/AgenticAI.pdf",
        chunker=TokenChunker(chunk_size=1500, chunk_overlap=100),
    )
