import os
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from .embeddings import embed_query
from .models import Result

load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs"  # must match ingest.py exactly

# Generation/rerank/query-rewrite model — Mistral direct API (not Azure).
DEFAULT_MODEL = "mistral-large-2512"

RETRIEVAL_K = 20

# Client for chat completions (Mistral direct API)
client = OpenAI(
    api_key=os.getenv("MISTRIAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
)

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_collection(COLLECTION_NAME)


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


def fetch_context_unranked(question: str) -> list[Result]:
    query_embedding = embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding], n_results=RETRIEVAL_K
    )
    chunks = []
    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=document, metadata=metadata))
    return chunks


def rerank(question: str, chunks: list[Result]) -> list[Result]:
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "RankOrder",
                "schema": RankOrder.model_json_schema(),
                "strict": True,
            },
        },
    )
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


def fetch_context(question: str) -> list[Result]:
    chunks = fetch_context_unranked(question)
    return rerank(question, chunks)


def make_rag_message(
    question: str, history: list[dict], chunks: list[Result]
) -> list[dict]:
    """Builds the full messages list (system + history + new user question)
    given pre-retrieved chunks for a RAG-augmented chat completion call.
    """
    context = "\n\n".join(
        f"# Source: {chunk.metadata.get('source', 'unknown')}\n{chunk.page_content}"
        for chunk in chunks
    )

    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(context=context),
    }

    messages = [system_message] + history + [{"role": "user", "content": question}]
    return messages


def rewrite_query(question: str, history: list[dict] = []) -> str:
    """Rewrite the user's question to be a more specific question that is
    more likely to surface relevant content in the Knowledge Base.
    """
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "system", "content": message}],
    )
    return response.choices[0].message.content


def answer_question(
    question: str,
    history: list[dict] = [],
) -> tuple[str, list[Result]]:
    """Answer a question using RAG and return the answer and the retrieved context."""
    query = rewrite_query(question, history)
    chunks = fetch_context(query)
    messages = make_rag_message(question, history, chunks)
    response = client.chat.completions.create(model=DEFAULT_MODEL, messages=messages)
    return response.choices[0].message.content, chunks
