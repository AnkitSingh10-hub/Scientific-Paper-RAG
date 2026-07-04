import chromadb

client = chromadb.PersistentClient(path="db/chroma_db")


collection = client.get_or_create_collection(name="research_papers")


def store(chunks, embeddings):

    metadata = [{"paper": "AgenticAI.pdf", "chunk": i} for i in range(len(chunks))]

    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata,
    )
