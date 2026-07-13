from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv(override=True)
AZURE_ENDPOINT = (
    "https://ankitsinghtheweeknd691-9348-reso.services.ai.azure.com/openai/v1"
)

DEFAULT_MODEL = "Mistral-Large-3"

DB_NAME = str(Path(__file__).parent.parent / "vector_database")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.

Context:
{context}
"""

vector_database = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings,
)

retriever = vector_database.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(
    base_url=AZURE_ENDPOINT,
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
    model=DEFAULT_MODEL,
    default_query={"api-version": "preview"},
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def fetch_context(question: str) -> list[Document]:

    return retriever.invoke(question)


def combined_question(question: str, history: list[dict] = []) -> str:
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(
    question: str,
    history: list[dict] = [],
) -> tuple[str, list[Document]]:

    combined = combined_question(question, history)

    docs = fetch_context(combined)

    context = format_docs(docs)

    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
            "history": history,
        }
    )

    response = llm.invoke(messages)

    answer = StrOutputParser().invoke(response)

    return answer, docs
