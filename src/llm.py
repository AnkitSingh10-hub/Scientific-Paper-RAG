from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"

ollama = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)


def generate(prompt):
    print("\n")
    print(prompt)
    response = ollama.chat.completions.create(
        model="mistral:latest",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content
