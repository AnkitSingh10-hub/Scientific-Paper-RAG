from embeddings import get_embedder
from vector_store import collection

embedder = get_embedder()


def retrieve(query, k=15):
    query_embedding = embedder.embed([query], is_query=True)[0]
    query_embedding = (
        query_embedding.tolist()
        if hasattr(query_embedding, "tolist")
        else list(query_embedding)
    )
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results
