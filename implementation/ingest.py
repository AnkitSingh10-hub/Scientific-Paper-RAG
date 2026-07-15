import os
import glob
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from chunking import get_chunker
from embeddings import get_embedder

DB_NAME = str(Path(__file__).parent.parent / "vector_database")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge_base")

# Swap these to try different embedding models / chunking strategies.
# See EMBEDDERS in embeddings.py for embedding options ("e5", ...)
# and CHUNKERS in chunking.py for chunking options ("doc_type",
# "fixed_size", "heading").
# NOTE: whatever EMBEDDING_STRATEGY you use here must match the one in
# answer.py, or queries will be embedded differently than the documents.
EMBEDDING_STRATEGY = "e5"
CHUNKING_STRATEGY = "doc_type"

load_dotenv(override=True)

embeddings = get_embedder(EMBEDDING_STRATEGY)(verbose=True)
create_chunks = get_chunker(CHUNKING_STRATEGY)


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME, embedding_function=embeddings
        ).delete_collection()

    vector_database = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vector_database._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(
        f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
    )
    return vector_database


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
