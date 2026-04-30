# LLM Memory

Every **chain.invoke()** is stateless — LLM has no memory of previous messages.

```text
User: "My name is Taslim"
AI:   "Hi Taslim!"

User: "What is my name?"
AI:   "I don't know your name."  ❌
```

## How Memory Works

You manually maintain and pass conversation history with every call.

```text
[system]           → role/behavior
[human] → [ai]    → turn 1
[human] → [ai]    → turn 2
[human]            → current question
```

### In LangChain

```python
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
llmProvider = ChatGroq(model="llama-3.3-70b-versatile")
chatHistory = []

while True:
    query = input("User: ")
    if query.lower() in ["quit", "exit", "bye"]:
        break
    chatHistory.append({"role": "user", "content": query})
    chain = llmProvider | StrOutputParser()
    response = chain.invoke(chatHistory)
    chatHistory.append({"role": "ai", "content": response})
    print(f"AI: {response}")
```

## How Memory Flows Per Turn

```text
Turn 1:
  prompt → [system] + [] (empty history) + "My name is Taslim"
  LLM    → "Hi Taslim!"
  store  → {human: "My name is Taslim", ai: "Hi Taslim!"}

Turn 2:
  prompt → [system] + [human: "My name is Taslim", ai: "Hi Taslim!"] + "What is my name?"
  LLM    → "Your name is Taslim" ✅
  store  → history grows with each turn
  
```

- That is the reason, the LLM providers have a limited memor/context size because if they keep on storing all history,
  they will need machine sof infinite size which is not possible.
- Also, one more thing to notice here is that, if we run a new window/process, the history would not be utilized there,
  this is why in ChatGPT or Claude Code, if we open a new chat window, previous history is not re-used by LLMs.
