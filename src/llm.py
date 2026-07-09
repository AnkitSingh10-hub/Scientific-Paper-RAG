from openai import OpenAI
import os
from dotenv import load_dotenv


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

load_dotenv(override=True)

google_api_key = os.getenv("GOOGLE_API_KEY")


gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)

DEFAULT_MODEL = "gemini-2.5-flash-lite"


# OLLAMA_BASE_URL = "http://localhost:11434/v1"

# ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def generate(system_prompt, user_prompt, model=DEFAULT_MODEL):
    response = gemini.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.choices[0].message.content


def generate_json(system_prompt, user_prompt, model=DEFAULT_MODEL):
    """Same as generate(), but asks the model to return a raw JSON object.

    Used for the LLM-as-judge evaluation, where we need structured
    (feedback, accuracy, completeness, relevance) output we can parse
    straight into a pydantic model.
    """
    response = gemini.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content
