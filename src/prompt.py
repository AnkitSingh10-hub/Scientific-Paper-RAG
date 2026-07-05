SYSTEM_PROMPT = """You are a research assistant.

Use ONLY the information provided in the context to answer the user's question.

If the answer cannot be found in the context, say:
"I don't have enough information from the retrieved documents."
"""


def create_prompt(context, question):
    context_text = "\n\n".join(context)

    user_prompt = f"""Context:
{context_text}

Question:
{question}

Answer:
"""

    return SYSTEM_PROMPT, user_prompt
