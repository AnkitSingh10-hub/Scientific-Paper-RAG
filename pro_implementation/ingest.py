from pathlib import Path

from chromadb import PersistentClient
from tqdm import tqdm

from .chunking import create_chunks
from .embeddings import embed_texts

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs"
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge_base")
BATCH_SIZE = 32


def fetch_documents() -> list[dict]:
    """A homemade version of the LangChain DirectoryLoader."""
    documents = []

    for folder in Path(KNOWLEDGE_BASE).iterdir():
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append(
                    {"type": doc_type, "source": file.as_posix(), "text": f.read()}
                )

    print(f"Loaded {len(documents)} documents")
    return documents


def create_embeddings(chunks, batch_size: int = BATCH_SIZE):
    chroma = PersistentClient(path=DB_NAME)
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)

    texts = [chunk.page_content for chunk in chunks]

    vectors = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        vectors.extend(embed_texts(batch))

    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")
    return collection


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
