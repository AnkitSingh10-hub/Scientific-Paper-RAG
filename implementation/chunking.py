import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)


# ----------------------------------------------------------------------
# Strategy 1: doc-type-aware heading chunking (the original approach)
# ----------------------------------------------------------------------


def chunk_by_doc_type(documents: list[Document]) -> list[Document]:
    """
    Chunking strategy

    Employees
        • Entire employee profile = one chunk

    Contracts
        • Split by markdown headings
        • Recursively split oversized sections

    Products
        • Split by markdown headings
        • Recursively split oversized sections

    Company
        • about.md -> one chunk
        • culture.md -> heading chunks
        • overview.md -> heading chunks
        • careers.md -> heading chunks
    """

    chunks = []

    for doc in documents:
        doc_type = doc.metadata["doc_type"].lower()
        filename = Path(doc.metadata["source"]).stem.lower()

        # ======================================================
        # EMPLOYEES
        # ======================================================

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

        # ======================================================
        # COMPANY - about.md
        # ======================================================

        if doc_type == "company" and filename == "about":
            metadata = dict(doc.metadata)
            metadata["document"] = filename

            chunks.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

            continue

        # ======================================================
        # CONTRACTS / PRODUCTS / COMPANY
        # ======================================================

        sections = markdown_splitter.split_text(doc.page_content)

        for section in sections:
            metadata = dict(doc.metadata)
            metadata.update(section.metadata)

            # Track company document name

            if doc_type == "company":
                metadata["document"] = filename

            # --------------------------------------------------
            # Product metadata
            # --------------------------------------------------

            if doc_type == "products":
                titles = re.findall(
                    r"^#\s+(.+)$",
                    doc.page_content,
                    re.MULTILINE,
                )

                if len(titles) > 1:
                    metadata["product"] = titles[1]

            # --------------------------------------------------
            # Contract metadata
            # --------------------------------------------------

            elif doc_type == "contracts":
                match = re.search(
                    r"Contract with (.+?) for",
                    doc.page_content,
                    re.IGNORECASE,
                )

                if match:
                    metadata["customer"] = match.group(1)

            text = section.page_content.strip()

            # Preserve heading for retrieval

            heading = (
                metadata.get("h3") or metadata.get("h2") or metadata.get("h1") or ""
            )

            # --------------------------------------------------
            # Small section
            # --------------------------------------------------

            if len(text.split()) <= 250:
                chunks.append(
                    Document(
                        page_content=f"{heading}\n\n{text}",
                        metadata=metadata,
                    )
                )

            # --------------------------------------------------
            # Large section
            # --------------------------------------------------

            else:
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


# ----------------------------------------------------------------------
# Strategy 2: markdown headings only, no doc-type special-casing and no
# recursive sub-splitting of oversized sections
# ----------------------------------------------------------------------


def chunk_by_heading(documents: list[Document]) -> list[Document]:
    """Split every document on markdown headings, one chunk per section,
    regardless of section length or doc_type.
    """
    chunks = []
    for doc in documents:
        sections = markdown_splitter.split_text(doc.page_content)
        for section in sections:
            metadata = dict(doc.metadata)
            metadata.update(section.metadata)
            heading = (
                metadata.get("h3") or metadata.get("h2") or metadata.get("h1") or ""
            )
            text = section.page_content.strip()
            chunks.append(
                Document(
                    page_content=f"{heading}\n\n{text}",
                    metadata=metadata,
                )
            )
    print(f"Created {len(chunks)} chunks")
    return chunks


# ----------------------------------------------------------------------
# Strategy 3: LangChain's CharacterTextSplitter — splits on a single
# separator only (default "\n\n"), falls back to a hard cut if a chunk
# is still too big. Simplest possible splitter, no structural awareness.
# ----------------------------------------------------------------------


def chunk_character(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separator: str = "\n\n",
) -> list[Document]:
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


# ----------------------------------------------------------------------
# Strategy 4: LangChain's RecursiveCharacterTextSplitter — tries a list
# of separators in order (paragraph, then line, then word, then char)
# until chunks fit. Generally a better default than CharacterTextSplitter
# since it degrades gracefully instead of cutting mid-word.
# ----------------------------------------------------------------------


def chunk_recursive_character(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


# ----------------------------------------------------------------------
# Registry / factory — add new strategies above, then register them here
# ----------------------------------------------------------------------

CHUNKERS = {
    "doc_type": chunk_by_doc_type,
    "heading": chunk_by_heading,
    "character": chunk_character,
    "recursive_character": chunk_recursive_character,
}


def get_chunker(strategy: str = "doc_type"):
    """Return the chunking function for the given strategy name.

    Usage:
        chunker = get_chunker("recursive_character")
        chunks = chunker(documents)
    """
    try:
        return CHUNKERS[strategy]
    except KeyError:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Available: {', '.join(CHUNKERS)}"
        )


# Backwards-compatible default so existing call sites (`create_chunks(docs)`)
# keep working without changes.
create_chunks = chunk_by_doc_type
