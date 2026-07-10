from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod

load_dotenv()

_token = os.getenv("HF_TOKEN")
if _token:
    login(token=_token)


class Embedder(ABC):
    @abstractmethod
    def embed(self, chunks, is_query: bool = False):
        """
        Embeds the input chunks.

        Args:
            chunks (str | list[str]): A single text string or list of text strings.
            is_query (bool): Set to True if embedding search queries rather than document chunks.
        """
        pass

    @property
    @abstractmethod
    def model_name(self):
        pass


class BGEEmbedding(Embedder):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    @property
    def model_name(self):
        return "bge-small-en-v1.5"

    def embed(self, chunks, is_query=False):
        # Convert a single string to a list to avoid iterating over characters
        if isinstance(chunks, str):
            chunks = [chunks]

        # BGE v1.5 requires this specific instruction prefix for search queries
        if is_query:
            instruction = "Represent this sentence for searching relevant passages: "
            chunks = [f"{instruction}{c}" for c in chunks]

        return self.model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


class MiniLMEmbedding(Embedder):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    @property
    def model_name(self):
        return "MiniLM-L6-v2"

    def embed(self, chunks, is_query=False):
        # MiniLM does not use prefixes, but we accept is_query to match the interface
        return self.model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


class MPNetEmbedding(Embedder):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    @property
    def model_name(self):
        return "all-mpnet-base-v2"

    def embed(self, chunks, is_query=False):
        # MPNet does not use prefixes, but we accept is_query to match the interface
        return self.model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


class E5Embedding(Embedder):
    def __init__(self):
        self.model = SentenceTransformer("intfloat/e5-base-v2")

    @property
    def model_name(self):
        return "e5-base-v2"

    def embed(self, chunks, is_query=False):
        # Convert a single string to a list to avoid iterating over characters
        if isinstance(chunks, str):
            chunks = [chunks]

        prefix = "query: " if is_query else "passage: "
        chunks = [f"{prefix}{c}" for c in chunks]

        return self.model.encode(
            chunks, normalize_embeddings=True, show_progress_bar=True
        )
