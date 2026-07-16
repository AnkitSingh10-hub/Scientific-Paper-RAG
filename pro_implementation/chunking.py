import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError, RateLimitError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm import tqdm

from .models import Result

load_dotenv(override=True)

# Chunking here is done BY an LLM (Mistral), not by character/heading
# splitting — the model reads a whole document and decides how to carve it
# into overlapping, self-describing chunks.
DEFAULT_MODEL = "mistral-medium-latest"

AVERAGE_CHUNK_SIZE = 500
CACHE_DIR = Path(__file__).parent.parent / "chunk_cache"
CACHE_DIR.mkdir(exist_ok=True)

client = OpenAI(
    api_key=os.getenv("MISTRIAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
)


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document: dict) -> Result:
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline
            + "\n\n"
            + self.summary
            + "\n\n"
            + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def make_prompt(document: dict) -> str:
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document: dict) -> list[dict]:
    return [
        {"role": "user", "content": make_prompt(document)},
    ]


def get_cache_path(document: dict) -> Path:
    return CACHE_DIR / (Path(document["source"]).stem + ".json")


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential_jitter(initial=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,  # after exhausting retries, raise the original RateLimitError
)
def call_chat_completion(messages: list[dict]):
    return client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Chunks",
                "schema": Chunks.model_json_schema(),
                "strict": True,
            },
        },
    )


def process_document(document: dict) -> list[Result]:
    cache_path = get_cache_path(document)

    # Load from cache if available — avoids re-paying for an LLM chunking
    # call every time ingest.py is re-run on the same knowledge base.
    if cache_path.exists():
        parsed = Chunks.model_validate_json(cache_path.read_text(encoding="utf-8"))
        return [chunk.as_result(document) for chunk in parsed.chunks]

    messages = make_messages(document)

    try:
        response = call_chat_completion(messages)
    except BadRequestError as e:
        print(f"\nFAILED: {document['source']}")
        print(e)
        return []
    except RateLimitError as e:
        # tenacity exhausted all 5 attempts
        raise RuntimeError(f"Failed after retries: {document['source']}") from e

    reply = response.choices[0].message.content
    parsed = Chunks.model_validate_json(reply)

    # Save to cache
    cache_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")

    return [chunk.as_result(document) for chunk in parsed.chunks]


def create_chunks(documents: list[dict], max_workers: int = 2) -> list[Result]:
    chunks: list[Result] = []
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_document, doc): doc for doc in documents}
        for future in tqdm(as_completed(futures), total=len(documents)):
            doc = futures[future]
            try:
                chunks.extend(future.result())
            except RuntimeError as e:
                print(f"Skipping {doc['source']}: {e}")
                failed.append(doc["source"])

    if failed:
        print(f"\n{len(failed)} documents failed after retries: {failed}")

    print(f"Created {len(chunks)} chunks")
    return chunks
