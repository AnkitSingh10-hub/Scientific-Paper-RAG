import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

AZURE_ENDPOINT = (
    "https://ankitsinghtheweeknd691-9348-reso.services.ai.azure.com/openai/v1"
)

embedding_client = OpenAI(
    base_url=AZURE_ENDPOINT,
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
)

EMBEDDING_MODELS = {
    "qwen": {
        "model": "qwen--qwen3-embedding-8b",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5": {
        "model": "intfloat--e5-large-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}


def embed_texts(
    texts: list[str],
    embedding_model: str = "qwen",
    batch_size: int = 32,
) -> list[list[float]]:

    config = EMBEDDING_MODELS[embedding_model]

    texts = [config["passage_prefix"] + text for text in texts]

    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        response = embedding_client.embeddings.create(
            model=config["model"],
            input=batch,
        )

        vectors.extend(item.embedding for item in response.data)

    return vectors


def embed_query(
    text: str,
    embedding_model: str = "qwen",
) -> list[float]:

    config = EMBEDDING_MODELS[embedding_model]

    response = embedding_client.embeddings.create(
        model=config["model"],
        input=config["query_prefix"] + text,
    )

    return response.data[0].embedding
