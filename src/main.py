from pipeline import ask


while True:
    question = input("\nWrite the question here?")

    if question.lower() == "exit":
        break

    answer = ask(question)

    print("\nThis is the Answer:\n")
    print(answer)
