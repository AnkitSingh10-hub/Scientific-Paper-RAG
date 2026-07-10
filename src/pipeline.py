from retrieval import retrieve
from prompt import create_prompt
from llm import generate


def ask_with_context(question, k=10):
    """Run retrieval + generation, returning both the answer and the
    retrieved context chunks (needed for evaluation)."""
    results = retrieve(question, k=k)

    context = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    print("\nRetrieved Chunks")
    print("=" * 80)

    for doc_id, distance, doc in zip(ids, distances, context):
        print(f"\nChunk: {doc_id}")
        print(f"Distance: {distance:.4f}")
        print(doc[:250])
        print("-" * 80)

    system_prompt, user_prompt = create_prompt(context, question)

    answer = generate(system_prompt, user_prompt)

    return answer, context


def ask(question):
    answer, _ = ask_with_context(question)
    return answer
