# Problem

- If you've been following along, you've noticed something annoying.Every time you ask the LLM a question — you wait.
  Nothing happens for 5-10 seconds, then the entire response dumps on screen at once.
- It's a bad user experience. Real AI products like ChatGPT, Claude, and Gemini all type out responses word by word.
  That's streaming.

```python
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

outputParser = StrOutputParser()
llmProvider: BaseChatModel = ChatOllama(model="gemma4:e2b")
chain: Runnable = llmProvider | outputParser
output = llmProvider.invoke("Who is the Prime Minister of India?")
print(output)  # ⏳ wait... wait... wait... then BOOM everything at once
```

# What is Streaming?

Instead of waiting for the full response, you get tokens as they are generated — just like watching someone type in real
time.

```text
Without streaming:  ⏳⏳⏳⏳⏳ → "RAG stands for Retrieval Augmented Generation..."
With streaming:     R → A → G →  s → t → a → n → d → s ...  (instant, live)
```

---

# How to Implement Streaming?

In our earlier code, we just need to swap **.invoke()** → **.stream()** and loop over chunks:

```python
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "You are a Senior Python Developer.",
    },
    {
        "role": "user",
        "content": "Explain {topic} with an example.",
    },
])

llm = ChatGroq(model="llama-3.3-70b-versatile")
chain = prompt | llm | StrOutputParser()

# ✅ Streaming — token by token
for chunk in chain.stream({"topic": "RAG"}):
    # end="" prevents newline after each token. flush=True forces immediate print without buffering.
    print(chunk, end="", flush=True)
```

---

If you are using **Streamlit streaming**, it natively provides **st.write_stream()**

```python
import streamlit as st
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


def chatBotWithStreaming(llmProvider: BaseChatModel) -> None:
    """
    Streamlit chatbot with real-time token streaming.

    Uses ``st.write_stream()`` to render tokens live as the LLM generates them,
    instead of waiting for the full response.

    :param llmProvider: The LLM backend to invoke.
    :type llmProvider: BaseChatModel
    """
    chain = llmProvider | StrOutputParser()

    # Render history
    for message in st.session_state.messages:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message["content"])

    query = st.chat_input("Ask anything...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # ✅ st.write_stream renders tokens live
            response = st.write_stream(chain.stream(st.session_state.messages))
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    load_dotenv()
    groqLLM: ChatGroq = ChatGroq(model="llama-3.3-70b-versatile")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant.",
            }
        ]

    st.title("💕 AI ChatBot — Streaming")
    chatBotWithStreaming(groqLLM)
```

---

## Note

1. **Structured output** and **streaming** are incompatible — you can't do both together because Pydantic validation
   runs on the complete JSON object. Streaming gives partial chunks like: