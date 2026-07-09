from embeddings import E5Embedding
from vector_store import collection

embedder = E5Embedding()


def retrieve(query, k=5):
    query_embedding = embedder.embed([query])[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=k)
    return results
