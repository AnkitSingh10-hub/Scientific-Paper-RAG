from embeddings import (
    E5Embedding,
    BGEEmbedding,
    MiniLMEmbedding,
    MPNetEmbedding,
    AzureAICohereEmbedding,
    AzureOpenAIEmbedding,
    AzureE5Embedding,
)
from vector_store import collection

embedder = AzureE5Embedding()


def retrieve(query, k=10):
    query_embedding = embedder.embed([query], is_query=True)[0]
    query_embedding = (
        query_embedding.tolist()
        if hasattr(query_embedding, "tolist")
        else list(query_embedding)
    )
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results
