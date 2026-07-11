from collections import namedtuple

from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

from openai import OpenAI

from azure.ai.inference import EmbeddingsClient
from azure.ai.inference.models import EmbeddingInputType
from azure.core.credentials import AzureKeyCredential

load_dotenv()

_token = os.getenv("HF_TOKEN")
if _token:
    login(token=_token)


# Every embedder factory below returns one of these. `embed` is a plain
# function you call directly, `model_name` is just a string for logging.
#
#   embedder = make_bge_embedder()
#   embedder.embed(["hello world"])
#   embedder.model_name  -> "bge-small-en-v1.5"
Embedder = namedtuple("Embedder", ["embed", "model_name"])


def _as_list(chunks):
    """Convert a single string to a list so we never iterate over characters."""
    if isinstance(chunks, str):
        return [chunks]
    return chunks


# ---------------------------------------------------------------------------
# Local sentence-transformers models
# ---------------------------------------------------------------------------


def make_bge_embedder():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def embed(chunks, is_query=False):
        chunks = _as_list(chunks)

        # BGE v1.5 requires this specific instruction prefix for search queries
        if is_query:
            instruction = "Represent this sentence for searching relevant passages: "
            chunks = [f"{instruction}{c}" for c in chunks]

        return model.encode(
            chunks,
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    return Embedder(embed=embed, model_name="bge-small-en-v1.5")


def make_minilm_embedder():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(chunks, is_query=False):
        # MiniLM does not use prefixes, but we accept is_query to match the interface
        chunks = _as_list(chunks)
        return model.encode(
            chunks,
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    return Embedder(embed=embed, model_name="MiniLM-L6-v2")


def make_mpnet_embedder():
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def embed(chunks, is_query=False):
        # MPNet does not use prefixes, but we accept is_query to match the interface
        chunks = _as_list(chunks)
        return model.encode(
            chunks,
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    return Embedder(embed=embed, model_name="all-mpnet-base-v2")


def make_e5_embedder():
    model = SentenceTransformer("intfloat/e5-base-v2")

    def embed(chunks, is_query=False):
        chunks = _as_list(chunks)
        prefix = "query: " if is_query else "passage: "
        chunks = [f"{prefix}{c}" for c in chunks]

        return model.encode(
            chunks, batch_size=1, normalize_embeddings=True, show_progress_bar=True
        )

    return Embedder(embed=embed, model_name="e5-base-v2")


# ---------------------------------------------------------------------------
# Azure-hosted models
# ---------------------------------------------------------------------------


def make_azure_cohere_embedder(chunk_size=96):
    api_key = os.getenv("AZURE_COHERE_KEY")
    endpoint = os.getenv("AZURE_COHERE_ENDPOINT")

    if not api_key or not endpoint:
        raise ValueError(
            "AZURE_COHERE_KEY and AZURE_COHERE_ENDPOINT must be set in your environment."
        )

    # Cohere's embed-v-4-0 caps requests at 96 texts per call
    chunk_size = min(chunk_size, 96) if chunk_size else 96
    client = EmbeddingsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

    def embed(chunks, is_query=False):
        chunks = _as_list(chunks)
        input_type = (
            EmbeddingInputType.QUERY if is_query else EmbeddingInputType.DOCUMENT
        )

        all_embeddings = []
        for i in range(0, len(chunks), chunk_size):
            batch = chunks[i : i + chunk_size]
            response = client.embed(
                input=batch, model="embed-v-4-0", input_type=input_type
            )
            all_embeddings.extend(item.embedding for item in response.data)

        return all_embeddings

    return Embedder(embed=embed, model_name="embed-v-4-0")


def make_azure_openai_embedder(chunk_size=96):
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not endpoint:
        raise ValueError(
            "AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set in your environment."
        )

    chunk_size = chunk_size or 96
    client = EmbeddingsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

    def embed(chunks, is_query=False):
        chunks = _as_list(chunks)

        all_embeddings = []
        for i in range(0, len(chunks), chunk_size):
            batch = chunks[i : i + chunk_size]
            response = client.embed(input=batch, model="text-embedding-3-small")
            all_embeddings.extend(item.embedding for item in response.data)

        return all_embeddings

    return Embedder(embed=embed, model_name="text-embedding-3-small")


def make_azure_e5_embedder(chunk_size=32):
    api_key = os.getenv("AZURE_E5_KEY")
    endpoint = os.getenv("AZURE_E5_ENDPOINT")

    if not api_key or not endpoint:
        raise ValueError(
            "AZURE_E5_KEY and AZURE_E5_ENDPOINT must be set in your environment."
        )

    deployment_name = "intfloat--e5-large-v2"
    chunk_size = chunk_size or 32
    client = OpenAI(base_url=endpoint, api_key=api_key)

    def embed(chunks, is_query=False):
        chunks = _as_list(chunks)
        prefix = "query: " if is_query else "passage: "
        chunks = [f"{prefix}{c}" for c in chunks]

        all_embeddings = []
        for i in range(0, len(chunks), chunk_size):
            batch = chunks[i : i + chunk_size]
            response = client.embeddings.create(input=batch, model=deployment_name)
            all_embeddings.extend(item.embedding for item in response.data)

        return all_embeddings

    return Embedder(embed=embed, model_name="e5-large-v2")


# ---------------------------------------------------------------------------
# Registry — lets you pick an embedder by string name (e.g. from .env)
# instead of importing a specific make_*_embedder function.
# ---------------------------------------------------------------------------

EMBEDDER_REGISTRY = {
    "bge": make_bge_embedder,
    "minilm": make_minilm_embedder,
    "mpnet": make_mpnet_embedder,
    "e5": make_e5_embedder,
    "azure-cohere": make_azure_cohere_embedder,
    "azure-openai": make_azure_openai_embedder,
    "azure-e5": make_azure_e5_embedder,
}


def get_embedder(name=None, **kwargs):
    """Look up and build an embedder by name.

    name defaults to the EMBEDDER env var, falling back to "azure-e5".
    Any kwargs (e.g. chunk_size=64) are passed through to the factory.

        embedder = get_embedder()              # uses .env / default
        embedder = get_embedder("bge")          # explicit override
    """
    name = name or os.getenv("EMBEDDER", "azure-e5")

    try:
        factory = EMBEDDER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown embedder '{name}'. Available options: "
            f"{', '.join(EMBEDDER_REGISTRY)}"
        )

    return factory(**kwargs)
