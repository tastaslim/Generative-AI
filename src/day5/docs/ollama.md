# Ollama

- **Ollama** is a tool that lets you run AI models locally on your own machine (no internet, no API key), privately
  and for free.
- You can download **Ollama** from [here](https://ollama.com/) and check supported open source models
  from [here](https://ollama.com/search)
- In this tutorial, we will learn how to integrate ollama with langchain. First need to install **langchain-ollama**
  package

```text
#requirements.txt
langchain-ollama
```

```python
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

outputParser = StrOutputParser()
llmProvider: BaseChatModel = ChatOllama(model="gemma4:e2b")
chain: Runnable = llmProvider | outputParser
output = llmProvider.invoke("Who is the Prime Minister of India?")
print(output)
```

If you see in above code, we are not providing any API_KEY or loading any API_KEY from env. It is because we have
installed ollama in our machine and downloaded the Google's open source **gemma4:e2b** model inside it locally.

---

# Ollama Cloud Models

- Ollama’s cloud models are a new kind of model in Ollama that can run without a powerful GPU. Instead, cloud models are
  automatically offloaded to Ollama’s cloud service while offering the same capabilities as local models, making it
  possible to keep using your local tools while running larger models that wouldn’t fit on a personal computer.

- Find more details [here](https://docs.ollama.com/cloud)