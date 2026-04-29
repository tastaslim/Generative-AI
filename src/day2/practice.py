import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
openAiClient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = openAiClient.responses.create(
    model="gpt-5.4-mini-2026-03-17",
    input="Who is the PM of India?"
)
print(response.output_text)