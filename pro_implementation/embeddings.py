import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

AZURE_ENDPOINT = (
    "https://ankitsinghtheweeknd691-9348-reso.services.ai.azure.com/openai/v1"
)

# Same embedding model used in both notebooks — must stay identical between
# ingest (embed_texts) and query time (embed_query) or retrieval silently
# degrades, same caveat as the LangChain version.
EMBEDDING_MODEL = "qwen--qwen3-embedding-8b"

# Plain OpenAI-compatible client pointed at the Azure AI Foundry endpoint
# (this is the same client shape used for both notebook 1's embedding calls
# and notebook 2's `embedding_client`).
embedding_client = OpenAI(
    base_url=AZURE_ENDPOINT,
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a list of documents/passages. No passage:/query: prefixing —
    Qwen3 is used as-is here, matching the notebooks (unlike the e5 wrapper
    in implementation/embeddings.py).
    """
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL, input=batch
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
