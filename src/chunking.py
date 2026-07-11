import re
from abc import ABC, abstractmethod

from langchain_text_splitters import (
    CharacterTextSplitter as _LCCharacterTextSplitter,
    RecursiveCharacterTextSplitter as _LCRecursiveCharacterTextSplitter,
    TokenTextSplitter as _LCTokenTextSplitter,
    
)


class Chunker(ABC):
    """Base interface all chunkers implement."""

    @abstractmethod
    def chunk(self, text):
        """Split text into a list of chunk strings."""
        raise NotImplementedError


class FixedChunker(Chunker):
    """Splits text into fixed-size character windows with overlap.

    Fast and simple, but can cut sentences/words in half.
    """

    def __init__(self, chunk_size=500, overlap=100):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text):
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap

        return chunks


class SentenceChunker(Chunker):
    """Splits text into sentences, then groups sentences into chunks up to
    ~chunk_size characters, carrying the last `overlap_sentences` sentences
    of a chunk over into the next one for context continuity.

    Avoids cutting sentences in half, which FixedChunker can do.
    """

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

    def __init__(self, chunk_size=500, overlap_sentences=1):
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")

        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text):
        text = text.strip()
        if not text:
            return []

        sentences = self._SENTENCE_SPLIT_RE.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text):
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence) + 1  # +1 for the joining space

            if current and current_len + sentence_len > self.chunk_size:
                chunks.append(" ".join(current))

                # carry over the last N sentences for overlap
                current = (
                    current[-self.overlap_sentences :] if self.overlap_sentences else []
                )
                current_len = sum(len(s) + 1 for s in current)

            current.append(sentence)
            current_len += sentence_len

        if current:
            chunks.append(" ".join(current))

        return chunks


class CharacterChunker(Chunker):
    """Wraps LangChain's CharacterTextSplitter.

    Splits on a single separator (default: double newline, i.e. paragraphs).
    If a chunk is still too big after splitting on the separator, it will
    NOT be split further — this is the key difference from Recursive below.
    """

    def __init__(self, chunk_size=500, chunk_overlap=100, separator="\n\n"):
        self._splitter = _LCCharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk(self, text):
        return self._splitter.split_text(text)


class RecursiveCharacterChunker(Chunker):
    """Wraps LangChain's RecursiveCharacterTextSplitter.

    Tries a list of separators in order (paragraph -> sentence -> word ->
    character), recursively splitting oversized pieces with the next
    separator down the list until each chunk fits chunk_size. Generally
    the best default general-purpose splitter.
    """

    def __init__(self, chunk_size=500, chunk_overlap=100, separators=None):
        self._splitter = _LCRecursiveCharacterTextSplitter(
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk(self, text):
        return self._splitter.split_text(text)


class TokenChunker(Chunker):
    """Wraps LangChain's TokenTextSplitter.

    Splits by tiktoken token count rather than characters, so chunk_size
    aligns with the actual token ids that embedding/LLM models consume.
    """

    def __init__(self, chunk_size=300, chunk_overlap=50, encoding_name="cl100k_base"):
        self._splitter = _LCTokenTextSplitter(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, text):
        return self._splitter.split_text(text)


