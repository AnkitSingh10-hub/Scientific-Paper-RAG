import os
import re
from collections import namedtuple

from langchain_text_splitters import (
    CharacterTextSplitter as _LCCharacterTextSplitter,
    RecursiveCharacterTextSplitter as _LCRecursiveCharacterTextSplitter,
    TokenTextSplitter as _LCTokenTextSplitter,
    MarkdownHeaderTextSplitter,
)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Every chunker factory below returns one of these. `chunk` is a plain
# function you call directly, `name` is just a string for logging.
#
#   chunker = make_fixed_chunker(chunk_size=500, overlap=100)
#   chunker.chunk(text)   -> list[str]
#   chunker.name          -> "FixedChunker"
Chunker = namedtuple("Chunker", ["chunk", "name"])


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _split_sentences(text):
    text = text.strip()
    if not text:
        return []

    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def make_fixed_chunker(chunk_size=500, overlap=100):
    """Splits text into fixed-size character windows with overlap.

    Fast and simple, but can cut sentences/words in half.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    def chunk(text):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks

    return Chunker(chunk=chunk, name="FixedChunker")


def make_sentence_chunker(chunk_size=500, overlap_sentences=1):
    """Splits text into sentences, then groups sentences into chunks up to
    ~chunk_size characters, carrying the last `overlap_sentences` sentences
    of a chunk over into the next one for context continuity.

    Avoids cutting sentences in half, which FixedChunker can do.
    """
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    def chunk(text):
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence) + 1  # +1 for the joining space

            if current and current_len + sentence_len > chunk_size:
                chunks.append(" ".join(current))

                # carry over the last N sentences for overlap
                current = current[-overlap_sentences:] if overlap_sentences else []
                current_len = sum(len(s) + 1 for s in current)

            current.append(sentence)
            current_len += sentence_len

        if current:
            chunks.append(" ".join(current))

        return chunks

    return Chunker(chunk=chunk, name="SentenceChunker")


def make_character_chunker(chunk_size=500, chunk_overlap=100, separator="\n\n"):
    """Wraps LangChain's CharacterTextSplitter.

    Splits on a single separator (default: double newline, i.e. paragraphs).
    If a chunk is still too big after splitting on the separator, it will
    NOT be split further — this is the key difference from Recursive below.
    """
    splitter = _LCCharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    def chunk(text):
        return splitter.split_text(text)

    return Chunker(chunk=chunk, name="CharacterChunker")


def make_recursive_character_chunker(
    chunk_size=500, chunk_overlap=100, separators=None
):
    """Wraps LangChain's RecursiveCharacterTextSplitter.

    Tries a list of separators in order (paragraph -> sentence -> word ->
    character), recursively splitting oversized pieces with the next
    separator down the list until each chunk fits chunk_size. Generally
    the best default general-purpose splitter.
    """
    splitter = _LCRecursiveCharacterTextSplitter(
        separators=separators or ["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    def chunk(text):
        return splitter.split_text(text)

    return Chunker(chunk=chunk, name="RecursiveCharacterChunker")


def make_token_chunker(chunk_size=300, chunk_overlap=50, encoding_name="cl100k_base"):
    """Wraps LangChain's TokenTextSplitter.

    Splits by tiktoken token count rather than characters, so chunk_size
    aligns with the actual token ids that embedding/LLM models consume.
    """
    splitter = _LCTokenTextSplitter(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    def chunk(text):
        return splitter.split_text(text)

    return Chunker(chunk=chunk, name="TokenChunker")


def make_markdown_semantic_chunker(
    embedder,
    chunk_size=800,
    chunk_overlap=100,
    similarity_threshold=0.85,
):
    """
    1. Split by markdown headers.
    2. Recursively split oversized sections.
    3. Merge neighboring chunks if embeddings are highly similar.
    """

    headers = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

    recursive_splitter = _LCRecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    def chunk(text):

        docs = md_splitter.split_text(text)

        chunks = []

        for doc in docs:
            pieces = recursive_splitter.split_text(doc.page_content)

            for piece in pieces:
                header = ""

                if doc.metadata.get("h1"):
                    header += f"# {doc.metadata['h1']}\n"

                if doc.metadata.get("h2"):
                    header += f"## {doc.metadata['h2']}\n"

                if doc.metadata.get("h3"):
                    header += f"### {doc.metadata['h3']}\n"

                chunks.append(header + "\n" + piece)

        if len(chunks) <= 1:
            return chunks

        embeddings = np.array(embedder.embed(chunks))

        merged = []
        current_text = chunks[0]
        current_emb = embeddings[0]

        for i in range(1, len(chunks)):
            sim = cosine_similarity(
                current_emb.reshape(1, -1),
                embeddings[i].reshape(1, -1),
            )[0][0]

            if sim >= similarity_threshold:
                current_text += "\n\n" + chunks[i]

                current_emb = (current_emb + embeddings[i]) / 2

            else:
                merged.append(current_text)
                current_text = chunks[i]
                current_emb = embeddings[i]

        merged.append(current_text)

        return merged

    return Chunker(
        chunk=chunk,
        name="MarkdownSemanticChunker",
    )


# ---------------------------------------------------------------------------
# Registry — lets you pick a chunker by string name (e.g. from .env)
# instead of importing a specific make_*_chunker function.
# ---------------------------------------------------------------------------

CHUNKER_REGISTRY = {
    "fixed": make_fixed_chunker,
    "sentence": make_sentence_chunker,
    "character": make_character_chunker,
    "recursive": make_recursive_character_chunker,
    "token": make_token_chunker,
    "markdown_semantic": make_markdown_semantic_chunker,
}


def get_chunker(name=None, **kwargs):
    """Look up and build a chunker by name.

    name defaults to the CHUNKER env var, falling back to "recursive".
    Any kwargs (e.g. chunk_size=500, chunk_overlap=100) are passed through
    to the factory.

        chunker = get_chunker()                 # uses .env / default
        chunker = get_chunker("fixed", chunk_size=500, overlap=100)
    """
    name = name or os.getenv("CHUNKER", "markdown_semantic")
    print(name)
    print("This is the chunker used")
    try:
        factory = CHUNKER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown chunker '{name}'. Available options: "
            f"{', '.join(CHUNKER_REGISTRY)}"
        )

    return factory(**kwargs)
