from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

_token = os.getenv("RAG_TOKEN")
if _token:
    login(token=_token)

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed_chunks(chunks):
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings
