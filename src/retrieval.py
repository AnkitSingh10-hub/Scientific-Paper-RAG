from embeddings import E5Embedding
from vector_store import collection

embedder = E5Embedding()


def retrieve(query, k=10):

    query_embedding = embedder.embed([query], is_query=True)[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=k)
    return results
