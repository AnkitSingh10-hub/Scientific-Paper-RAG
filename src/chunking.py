def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    sample = """
    Artificial Intelligence is transforming healthcare.
    Machine learning helps doctors diagnose diseases.
    Deep learning improves medical imaging.
    """

    chunks = chunk_text(sample, chunk_size=50)

    for i, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {i}")
        print(chunk)
