from openai import OpenAI
import os
from dotenv import load_dotenv


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

load_dotenv(override=True)

google_api_key = os.getenv("GOOGLE_API_KEY")


gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)


def generate(system_prompt, user_prompt):
    response = gemini.chat.completions.create(
        model="gemini-2.5-flash-lite",
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
