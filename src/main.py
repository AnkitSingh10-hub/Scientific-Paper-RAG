from pipeline import ask


while True:
    question = input("\nWhat is Agentic AI ")

    if question.lower() == "exit":
        break

    answer = ask(question)

    print("\nThis is the Answer:\n")
    print(answer)
