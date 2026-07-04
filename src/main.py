from read_pdf import extract_text
from chunking import fixed_chunks
from embeddings import embed_chunks
from vector_store import store, collection
from retrieval import retrieve

text = extract_text("data/papers/AgenticAI.pdf")

chunks = fixed_chunks(text)

embeddings = embed_chunks(chunks)


store(chunks, embeddings)


results = retrieve("What is Agentic AI?")

docs = results["documents"][0]
distances = results["distances"][0]
ids = results["ids"][0]

for rank, (doc_id, doc, distance) in enumerate(zip(ids, docs, distances), start=1):
    print("=" * 80)
    print(f"Rank: {rank}")
    print(f"Chunk ID: {doc_id}")
    print(f"Distance: {distance:.4f}")
    print()
    print(doc[:400])  # first 400 characters
    print()

print("Done indexing")
print("Number of chunks: ", collection.count())
