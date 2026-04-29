import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
openAiClient = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
response = openAiClient.responses.create(
    model="gpt-5.5",
    input="Who is the PM of India?"
)
print(response.output_text)