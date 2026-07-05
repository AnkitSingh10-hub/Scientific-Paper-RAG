def create_prompt(context, question):

    context_text = "\n\n".join(context)

    prompt = f"""
    You are a research assistant.

    Use ONLY the information provided in the context to answer the user's question.

    If the answer cannot be found in the context, say:
    "I don't have enough information from the retrieved documents."

    Context:
    {context_text}

    Question:
    {question}

    Answer:
    """

    return prompt
