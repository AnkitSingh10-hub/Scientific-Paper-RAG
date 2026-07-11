from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
import requests
import torch
from openai import OpenAI
import os

from azure.ai.inference import EmbeddingsClient
from azure.ai.inference.models import EmbeddingInputType
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

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
            batch_size=1,
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
            batch_size=1,
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
            batch_size=1,
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
            chunks, batch_size=1, normalize_embeddings=True, show_progress_bar=True
        )


class AzureAICohereEmbedding(Embedder):
    def __init__(self, chunk_size=96, **kwargs):
        api_key = os.getenv("AZURE_COHERE_KEY")
        endpoint = os.getenv("AZURE_COHERE_ENDPOINT")

        if not api_key or not endpoint:
            raise ValueError(
                "AZURE_COHERE_KEY and AZURE_COHERE_ENDPOINT must be set in your environment."
            )

        # Cohere's embed-v-4-0 caps requests at 96 texts per call
        self.chunk_size = min(chunk_size, 96) if chunk_size else 96

        self.client = EmbeddingsClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )

    @property
    def model_name(self):
        return "embed-v-4-0"

    def embed(self, chunks, is_query=False):
        # Convert a single string to a list to avoid character iteration
        if isinstance(chunks, str):
            chunks = [chunks]

        input_type = (
            EmbeddingInputType.QUERY if is_query else EmbeddingInputType.DOCUMENT
        )

        all_embeddings = []
        for i in range(0, len(chunks), self.chunk_size):
            batch = chunks[i : i + self.chunk_size]
            response = self.client.embed(
                input=batch, model="embed-v-4-0", input_type=input_type
            )
            all_embeddings.extend(item.embedding for item in response.data)

        return all_embeddings


class AzureOpenAIEmbedding(Embedder):
    def __init__(self, chunk_size=None, **kwargs):
        api_key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not api_key or not endpoint:
            raise ValueError(
                "AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set in your environment."
            )

        self.chunk_size = chunk_size or 96

        self.client = EmbeddingsClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )

    @property
    def model_name(self):
        return "text-embedding-3-small"

    def embed(self, chunks, is_query=False):
        if isinstance(chunks, str):
            chunks = [chunks]

        all_embeddings = []
        for i in range(0, len(chunks), self.chunk_size):
            batch = chunks[i : i + self.chunk_size]
            response = self.client.embed(input=batch, model="text-embedding-3-small")
            all_embeddings.extend(item.embedding for item in response.data)

        return all_embeddings


class AzureE5Embedding(Embedder):
    def __init__(self, chunk_size=32, **kwargs):
        api_key = os.getenv("AZURE_E5_KEY")
        endpoint = os.getenv("AZURE_E5_ENDPOINT")

        if not api_key or not endpoint:
            raise ValueError(
                "AZURE_E5_KEY and AZURE_E5_ENDPOINT must be set in your environment."
            )

        self.deployment_name = "intfloat--e5-large-v2"
        self.chunk_size = chunk_size or 32

        self.client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )

    @property
    def model_name(self):
        return "e5-large-v2"

    def embed(self, chunks, is_query=False):
        if isinstance(chunks, str):
            chunks = [chunks]

        prefix = "query: " if is_query else "passage: "
        chunks = [f"{prefix}{c}" for c in chunks]

        all_embeddings = []
        for i in range(0, len(chunks), self.chunk_size):
            batch = chunks[i : i + self.chunk_size]
            response = self.client.embeddings.create(
                input=batch, model=self.deployment_name
            )
            all_embeddings.extend(item.embedding for item in response.data)

        return all_embeddings
