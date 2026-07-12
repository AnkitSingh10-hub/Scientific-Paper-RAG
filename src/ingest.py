import os

from read_md import clean_md_for_rag
from chunking import get_chunker
from embeddings import get_embedder
from vector_store import store


def ingest(md_path, chunker=None, embedder=None):
    if embedder is None:
        embedder = get_embedder()

    if chunker is None:
        chunker = get_chunker(
            "markdown_semantic",
            embedder=embedder,
            chunk_size=800,
            chunk_overlap=100,
        )

    print(f"Reading {md_path} ...")
    text = clean_md_for_rag(md_path)

    print(f"Chunking text with {chunker.name} ...")
    chunks = chunker.chunk(text)
    print(f"Created {len(chunks)} chunks.")

    print(f"Embedding with {embedder.model_name} ...")
    embeddings = embedder.embed(chunks)

    print("Storing in Chroma ...")
    paper_name = os.path.basename(md_path)

    store(
        chunks,
        embeddings,
        paper_name=paper_name,
        embedding_model=embedder.model_name,
    )

    print("Done. Chunks stored:", len(chunks))


if __name__ == "__main__":
    ingest("data/papers/AgenticAI.md")
