import os

from read_pdf import clean_pdf_for_rag
from chunking import FixedChunker, TokenChunker, RecursiveCharacterChunker
from embeddings import E5Embedding
from vector_store import store


def ingest(pdf_path, chunker=None, embedder=None):
    if chunker is None:
        chunker = FixedChunker(chunk_size=1000, overlap=100)

    if embedder is None:
        embedder = E5Embedding()

    print(f"Reading {pdf_path} ...")
    text = clean_pdf_for_rag(pdf_path)

    print(f"Chunking text with {type(chunker).__name__} ...")
    chunks = chunker.chunk(text)
    print(f"Created {len(chunks)} chunks.")

    print(f"Embedding with {embedder.model_name} ...")
    embeddings = embedder.embed(chunks)

    print("Storing in Chroma ...")
    paper_name = os.path.basename(pdf_path)

    store(
        chunks,
        embeddings,
        paper_name=paper_name,
        embedding_model=embedder.model_name,
    )

    print("Done. Chunks stored:", len(chunks))


if __name__ == "__main__":
    ingest(
        "data/papers/AgenticAI.pdf",
        chunker=RecursiveCharacterChunker(
            chunk_size=700,
            chunk_overlap=100,
        ),
        embedder=E5Embedding(),
    )
