import chromadb

client = chromadb.PersistentClient(path="db/chroma_db")


collection = client.get_or_create_collection(name="research_papers")


def store(chunks, embeddings, paper_name="AgenticAI.pdf", embedding_model=None):
    metadata = [{"paper": paper_name, "chunk": i} for i in range(len(chunks))]

    ids = [f"{paper_name}::{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata,
    )
