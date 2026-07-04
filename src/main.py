from read_pdf import extract_text
from chunking import fixed_chunks
from embedding import embed_chunks
from vector_store import store

text = extract_text("data/papers/AgenticAI.pdf")

chunks = fixed_chunks(text)

embeddings = embed_chunks(chunks)

store(chunks, embeddings)

print("Finished indexing.")
