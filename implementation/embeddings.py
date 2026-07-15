import os

from langchain_core.embeddings import Embeddings
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

AZURE_EMBEDDING_ENDPOINT = (
    "https://ankitsinghtheweeknd691-9348-reso.services.ai.azure.com/models"
)


class AzureE5Embeddings(Embeddings):
    """LangChain-compatible wrapper around Azure AI Foundry's e5-large-v2.

    Must be used identically at ingest time and query time (same model,
    same prefixing scheme) or retrieval quality silently degrades.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str = "intfloat--e5-large-v2",
        batch_size: int = 32,
        verbose: bool = False,
    ):
        self.client = EmbeddingsClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )
        self.model = model
        self.batch_size = batch_size
        self.verbose = verbose

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        all_embeddings = []
        for i in range(0, len(prefixed), self.batch_size):
            batch = prefixed[i : i + self.batch_size]
            response = self.client.embed(model=self.model, input=batch)
            all_embeddings.extend(item.embedding for item in response.data)
            if self.verbose:
                print(
                    f"  embedded {min(i + self.batch_size, len(prefixed))}/{len(prefixed)}"
                )
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embed(model=self.model, input=[f"query: {text}"])
        return response.data[0].embedding


class AzureQwen3Embeddings(Embeddings):
    """LangChain-compatible wrapper around Azure AI Foundry's Qwen3-Embedding-8B.

    IMPORTANT — different convention from e5:
      * Documents/passages are embedded AS-IS, no "passage: " prefix.
      * Queries get an instruction-style prefix:
            "Instruct: {task description}\\nQuery: {text}"
        This is how Qwen3-Embedding was trained/recommended to be used;
        applying the e5 scheme here would silently hurt retrieval quality.

    Must be used identically at ingest time and query time (same model,
    same prefixing scheme) or retrieval quality silently degrades.
    """

    DEFAULT_INSTRUCTION = (
        "Given a search query, retrieve relevant passages that answer the query"
    )

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str = "qwen--qwen3-embedding-8b",
        batch_size: int = 32,
        instruction: str = DEFAULT_INSTRUCTION,
        verbose: bool = False,
    ):
        self.client = EmbeddingsClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )
        self.model = model
        self.batch_size = batch_size
        self.instruction = instruction
        self.verbose = verbose

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embed(model=self.model, input=batch)
            all_embeddings.extend(item.embedding for item in response.data)
            if self.verbose:
                print(f"  embedded {min(i + self.batch_size, len(texts))}/{len(texts)}")
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"Instruct: {self.instruction}\nQuery: {text}"
        response = self.client.embed(model=self.model, input=[prefixed])
        return response.data[0].embedding


# ----------------------------------------------------------------------
# Strategy 1: Azure AI Foundry e5-large-v2
# ----------------------------------------------------------------------


def get_e5_embeddings(verbose: bool = False) -> AzureE5Embeddings:
    """Azure AI Foundry e5-large-v2, batch size 32."""
    return AzureE5Embeddings(
        endpoint=AZURE_EMBEDDING_ENDPOINT,
        api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
        model="intfloat--e5-large-v2",
        batch_size=32,
        verbose=verbose,
    )


# ----------------------------------------------------------------------
# Strategy 2: Azure AI Foundry Qwen3-Embedding-8B
# ----------------------------------------------------------------------


def get_qwen3_embeddings(verbose: bool = False) -> AzureQwen3Embeddings:
    """Azure AI Foundry Qwen3-Embedding-8B, batch size 32.

    Note: batch_size 32 is inherited from the e5 default. Qwen3-Embedding-8B
    is a much larger model (8B params, 4096-dim output) than e5-large-v2, so
    if you hit request-size/timeout limits on the Azure endpoint, try
    lowering batch_size (e.g. 8-16).
    """
    return AzureQwen3Embeddings(
        endpoint=AZURE_EMBEDDING_ENDPOINT,
        api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
        model="qwen--qwen3-embedding-8b",
        batch_size=32,
        verbose=verbose,
    )


# ----------------------------------------------------------------------
# Add more strategies here as you try them, e.g.:
#
# def get_openai_embeddings(verbose: bool = False):
#     from langchain_openai import OpenAIEmbeddings
#     return OpenAIEmbeddings(model="text-embedding-3-large")
#
# def get_bge_embeddings(verbose: bool = False) -> AzureE5Embeddings:
#     return AzureE5Embeddings(
#         endpoint=AZURE_EMBEDDING_ENDPOINT,
#         api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
#         model="BAAI--bge-large-en-v1.5",
#         batch_size=32,
#         verbose=verbose,
#     )
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Registry / factory — register each strategy above with a short name,
# then just type that name in ingest.py / answer.py.
# ----------------------------------------------------------------------

EMBEDDERS = {
    "e5": get_e5_embeddings,
    "qwen3": get_qwen3_embeddings,
}


def get_embedder(strategy: str = "e5"):
    """Return the embedding factory function for the given strategy name.

    Usage:
        embeddings = get_embedder("qwen3")(verbose=True)

    IMPORTANT: ingest.py and answer.py must use the SAME strategy — the
    vectors in Chroma were produced with whichever embedder ran at
    ingest time, and querying with a different one silently returns
    garbage results (no error, just bad retrieval).
    """
    try:
        return EMBEDDERS[strategy]
    except KeyError:
        raise ValueError(
            f"Unknown embedding strategy '{strategy}'. "
            f"Available: {', '.join(EMBEDDERS)}"
        )


# Backwards-compatible default so existing call sites (`get_embeddings()`)
# keep working without changes.
get_embeddings = get_e5_embeddings
