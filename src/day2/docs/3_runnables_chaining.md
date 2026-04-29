# Runnables In LangChain

Any object or type that can be run are called runnables in LangChain. Prompt, LLM, parser, retriever etc. are runnables
in LangChain. A Runnable is anything that has below methods:

- **.invoke()** → single input
- **.batch()** → multiple inputs
- **.stream()** → streaming output

---

# Chaining

Runnables can be chained together to achieve a particular task. Meaning output of one runnable can be input of another
runnable. In this manner we can chain them together. You chain Runnables using the **| (pipe operator)**. Once chained,
you can run them using **.invoke()** for a single input, **.batch()** for multiple inputs, or **.stream()** for
token-by-token output.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "You are a Senior Python Developer."},
    {"role": "user", "content": "Explain {topic} with an example."}
])

llm = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

# Chain
chain = prompt | llm | parser

# Invoke
output = chain.invoke({"topic": "RAG"})
print(output)

"""
{"topic": "RAG"}
      ↓
   prompt        → formats messages
      ↓
     llm          → sends to LLM, gets AIMessage
      ↓
   parser         → extracts plain string from AIMessage
      ↓
   "RAG is..."
"""
```

---

You might ask me, Isn't below code chaining

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "You are a Senior Python Developer who answers in {language}.",
    },
    {
        "role": "user",
        "content": "Explain {topic} with an example.",
    },
])
parser = StrOutputParser()
llm = ChatGroq(model="llama-3.1-8b-instant")
output = parser.invoke(llm.invoke(prompt.invoke({"topic": "RAG", "language": "English"})))
print(output)
```

Technically it works, but it's not chaining — it's nesting. And it will become very hard if we have to provide output of
1 runnable as input to another runnable in case of several runnables.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "You are a Senior Python Developer who answers in {language}.",
    },
    {
        "role": "user",
        "content": "Explain {topic} with an example.",
    },
])

llm = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()
chain = prompt | llm | parser
output = chain.invoke({"topic": "RAG", "language": "English"})
print(output)
```

