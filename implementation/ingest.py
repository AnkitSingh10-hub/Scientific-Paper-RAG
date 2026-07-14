# import os
# import glob
# from pathlib import Path
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
# from azure.ai.inference import EmbeddingsClient
# from azure.core.credentials import AzureKeyCredential


# from dotenv import load_dotenv

# MODEL = "intfloat--e5-large-v2"

# DB_NAME = str(Path(__file__).parent.parent / "vector_database")
# KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge_base")


# load_dotenv(override=True)


# # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# client = EmbeddingsClient(
#     endpoint="https://ankitsinghtheweeknd691-9348-reso.services.ai.azure.com/models",
#     credential=AzureKeyCredential(os.getenv("AZURE_FOUNDRY_API_KEY")),
# )

# response = client.embed(
#     model=MODEL,
#     input=["passage: Azure AI Foundry is Microsoft's AI platform."],
# )

# embeddings = response.data[0].embedding
# print(len(embeddings))


# def fetch_documents():
#     folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
#     documents = []
#     for folder in folders:
#         doc_type = os.path.basename(folder)
#         loader = DirectoryLoader(
#             folder,
#             glob="**/*.md",
#             loader_cls=TextLoader,
#             loader_kwargs={"encoding": "utf-8"},
#         )
#         folder_docs = loader.load()
#         for doc in folder_docs:
#             doc.metadata["doc_type"] = doc_type
#             documents.append(doc)
#     return documents


# def create_chunks(documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
#     chunks = text_splitter.split_documents(documents)
#     return chunks


# def create_embeddings(chunks):
#     if os.path.exists(DB_NAME):
#         Chroma(
#             persist_directory=DB_NAME, embedding_function=embeddings
#         ).delete_collection()

#     vector_database = Chroma.from_documents(
#         documents=chunks, embedding=embeddings, persist_directory=DB_NAME
#     )

#     collection = vector_database._collection
#     count = collection.count()

#     sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
#     dimensions = len(sample_embedding)
#     print(
#         f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
#     )
#     return vector_database


# if __name__ == "__main__":
#     documents = fetch_documents()
#     chunks = create_chunks(documents)
#     create_embeddings(chunks)
#     print("Ingestion complete")


import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from langchain_core.documents import Document
import re
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv

MODEL = "intfloat--e5-large-v2"

DB_NAME = str(Path(__file__).parent.parent / "vector_database")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge_base")

load_dotenv(override=True)


class AzureE5Embeddings(Embeddings):
    """LangChain-compatible wrapper around Azure AI Foundry's e5-large-v2."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str = "intfloat--e5-large-v2",
        batch_size: int = 32,
    ):
        self.client = EmbeddingsClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        all_embeddings = []
        for i in range(0, len(prefixed), self.batch_size):
            batch = prefixed[i : i + self.batch_size]
            response = self.client.embed(model=self.model, input=batch)
            all_embeddings.extend(item.embedding for item in response.data)
            print(
                f"  embedded {min(i + self.batch_size, len(prefixed))}/{len(prefixed)}"
            )
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embed(model=self.model, input=[f"query: {text}"])
        return response.data[0].embedding


embeddings = AzureE5Embeddings(
    endpoint="https://ankitsinghtheweeknd691-9348-reso.services.ai.azure.com/models",
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
    batch_size=32,
)


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)


def create_chunks(documents):
    """
    Chunk strategy

    Employees:
        One employee file = one chunk

    Contracts:
        Markdown heading chunks

    Products:
        Markdown heading chunks

    Company:
        Markdown heading chunks

    Large sections:
        Recursive split
    """

    chunks = []

    for doc in documents:
        doc_type = doc.metadata["doc_type"].lower()

        # ==================================================
        # EMPLOYEES
        # ==================================================

        if doc_type == "employees":
            titles = re.findall(
                r"^#\s+(.+)$",
                doc.page_content,
                re.MULTILINE,
            )

            employee = (
                titles[1] if len(titles) > 1 else titles[0] if titles else "Unknown"
            )

            metadata = dict(doc.metadata)
            metadata["employee"] = employee

            chunks.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

            continue

        # ==================================================
        # CONTRACTS / PRODUCTS / COMPANY
        # ==================================================

        sections = markdown_splitter.split_text(doc.page_content)

        for section in sections:
            metadata = dict(doc.metadata)
            metadata.update(section.metadata)

            # --------------------------------------------
            # Product metadata
            # --------------------------------------------

            if doc_type == "products":
                titles = re.findall(
                    r"^#\s+(.+)$",
                    doc.page_content,
                    re.MULTILINE,
                )

                if len(titles) > 1:
                    metadata["product"] = titles[1]

            # --------------------------------------------
            # Contract metadata
            # --------------------------------------------

            elif doc_type == "contracts":
                match = re.search(
                    r"Contract with (.+?) for",
                    doc.page_content,
                    re.IGNORECASE,
                )

                if match:
                    metadata["customer"] = match.group(1)

            text = section.page_content.strip()

            # --------------------------------------------
            # Small section
            # --------------------------------------------

            if len(text.split()) <= 250:
                chunks.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )

            # --------------------------------------------
            # Large section
            # --------------------------------------------

            else:
                heading = (
                    metadata.get("h3") or metadata.get("h2") or metadata.get("h1") or ""
                )

                sub_docs = recursive_splitter.create_documents(
                    [text],
                    metadatas=[metadata],
                )

                for sub_doc in sub_docs:
                    chunks.append(
                        Document(
                            page_content=f"{heading}\n\n{sub_doc.page_content}",
                            metadata=sub_doc.metadata,
                        )
                    )

    print(f"Created {len(chunks)} chunks")

    return chunks


def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME, embedding_function=embeddings
        ).delete_collection()

    vector_database = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vector_database._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(
        f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
    )
    return vector_database


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
