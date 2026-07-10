from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv(override=True)

AZURE_ENDPOINT = (
    "https://ankitsinghtheweeknd691-9348-reso.services.ai.azure.com/openai/v1"
)

DEFAULT_MODEL = "Mistral-Large-3"

client = OpenAI(
    base_url=AZURE_ENDPOINT,
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
    default_query={"api-version": "preview"},
)


def generate(system_prompt, user_prompt, model=DEFAULT_MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    if not response.choices:
        raise RuntimeError(f"Foundry returned no choices: {response.model_dump_json()}")

    return response.choices[0].message.content


def generate_json(system_prompt, user_prompt, model=DEFAULT_MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    if not response.choices:
        raise RuntimeError(f"Foundry returned no choices: {response.model_dump_json()}")

    return response.choices[0].message.content
