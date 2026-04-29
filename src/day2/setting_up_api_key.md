# API Keys
Before we start working on GEN AI projects using LLM APIs, we need to create our account on various LLM platforms like OpenAPI, Anthropic, Google Gemini,Hugging Face, Groq etc. You can use below links to do so.
- https://platform.claude.com/dashboard
- https://platform.openai.com/api-keys
- https://aistudio.google.com/api-keys?project=gen-lang-client-0880569018
- https://huggingface.co/settings/tokens
- https://console.groq.com/keys

Once Done, Please create a .env file in your project and start using it
```dotenv
OPEN_API_KEY="PUT_OPENAI_API_KEY"
GOOGLE_API_KEY="PUT_GEMINI_API_KEY"
ANTHROPIC_API_KEY="PUT_ANTHROPIC_API_KEY"
GROQ_API_KEY="PUT_GROQ_API_KEY"
HUGGINGFACE_HUB_API_TOKEN="PUT_HUGGINGFACE_API_KEY"
```

```python
import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()
openAiClient = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
response = openAiClient.responses.create(
    model="gpt-5.5",
    input="Who is the PM of India?"
)
print(response.output_text)

anthropicClient = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
message = anthropicClient.messages.create(
    model="claude-opus-4-7",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "What should I search for to find the latest developments in renewable energy?",
        }
    ],
)
print(message.content)
```

You see, when it comes to using single LLM models it is fine but if we have to build production grade application where we need to use combination of multiple LLM models,
it will be very difficult for us to manage because LLM model has different library, coding structure, message parsing , input, output etc.

**For same use case, we use LangChain which is a framework powered by LLLMs to build Gen AI solutions** 
