from retrieval import retrieve
from prompt import create_prompt
from llm import generate


def ask(question):
    results = retrieve(question)

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

    prompt = create_prompt(context, question)

    answer = generate(prompt)

    return answer
