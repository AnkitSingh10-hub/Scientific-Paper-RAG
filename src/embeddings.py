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
    def embed(self, chunks):
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

    def embed(self, chunks):
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

    def embed(self, chunks):
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

    def embed(self, chunks):
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
        prefix = "query: " if is_query else "passage: "
        chunks = [f"{prefix}{c}" for c in chunks]
        return self.model.encode(
            chunks, normalize_embeddings=True, show_progress_bar=True
        )
