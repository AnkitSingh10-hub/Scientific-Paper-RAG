from embeddings import model
from vector_store import collection


def retrieve(query, k=5):
    query_embedding = model.encode(query)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=k)
    return results
