# Text Splitters

Text splitters break large docs into smaller chunks that will be retrievable individually and fit within model context
window limit. There are several strategies for splitting documents, each with its own advantages.

For most use cases, start with the **RecursiveCharacterTextSplitter**. It provides a solid balance between keeping
context intact and managing chunk size. This default strategy works well out of the box, and you should only consider
adjusting it if you need to fine-tune performance for your specific application.

**Read [this](https://docs.langchain.com/oss/python/integrations/splitters) doc for more details**

---

# Types of Text Splitters

## 1. Text Structure-Based

**`RecursiveCharacterTextSplitter`**
The default and recommended splitter. Tries to split on natural language boundaries in order: `\n\n` → `\n` → `" "` →
`""`. Keeps paragraphs, then sentences, then words together as long as possible.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
LangChain is a framework for building LLM-powered applications.
It provides abstractions for document loading, text splitting, and retrieval.

Text splitters break large documents into smaller chunks.
Each chunk is embedded and stored in a vector store for retrieval.

RAG (Retrieval-Augmented Generation) is the most common use case.
"""

splitter = RecursiveCharacterTextSplitter(
	chunk_size=100,  # max characters per chunk
	chunk_overlap=20,  # shared characters between consecutive chunks
	length_function=len,  # how chunk size is measured
)

# Returns plain strings
chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
	print(f"[Chunk {i + 1}] {chunk}")

# Returns LangChain Document objects (use this in RAG pipelines)
documents = splitter.create_documents(
	[text],
	metadatas=[{"source": "langchain_intro.txt"}]
)
for document in documents:
	print(document.page_content)
	print(document.metadata)
```

## 2. Length-Based

**`CharacterTextSplitter`**
Splits on a single separator (default `\n\n`) and enforces a hard character count limit. Simple and predictable.

```python
from langchain_text_splitters import CharacterTextSplitter

text = """LangChain simplifies building RAG systems.

It has components for loading, splitting, embedding, and retrieving documents.

Vector stores index embeddings for fast similarity search.
"""

splitter = CharacterTextSplitter(
	separator="\n\n",  # split only on double newlines
	chunk_size=100,
	chunk_overlap=20,
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
	print(f"[Chunk {i + 1}] {chunk}")
```

**`CharacterTextSplitter.from_tiktoken_encoder`**
Same as above but measures size in **tokens** (not characters) using a **tokenizer**. Essential when working with
token-limited APIs.

> **What is a tokenizer?**
> A tokenizer converts raw text into tokens — the units an LLM actually reads and charges by.
> `tiktoken` is OpenAI's tokenizer library. Other tokenizers exist (e.g. `sentencepiece` by Google, `tokenizers` by
> HuggingFace), but `tiktoken` is the most common in LangChain splits.
> Rule of thumb: **1 token ≈ 4 characters** in English.

```python
# pip install tiktoken
from langchain_text_splitters import CharacterTextSplitter

text = """LangChain simplifies building RAG systems.

It has components for loading, splitting, embedding, and retrieving documents.

Vector stores index embeddings for fast similarity search.
"""

splitter = CharacterTextSplitter.from_tiktoken_encoder(
	encoding_name="cl100k_base",  # tokenizer encoding used by GPT-4 / GPT-3.5
	chunk_size=50,  # now in TOKENS, not characters
	chunk_overlap=10,
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
	print(f"[Chunk {i + 1}] {chunk}")
```

---

**`TokenTextSplitter`**
Splits purely by token count using tokenizer, with no separator awareness at all.

```python
# pip install tiktoken
from langchain_text_splitters import TokenTextSplitter

text = """LangChain simplifies building RAG systems.
It has components for loading, splitting, embedding, and retrieving documents.
Vector stores index embeddings for fast similarity search.
"""

splitter = TokenTextSplitter(
	encoding_name="cl100k_base",
	chunk_size=20,  # tokens per chunk
	chunk_overlap=5,
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
	print(f"[Chunk {i + 1}] {chunk}")
```

---

## 3. Document Structure-Based

**`MarkdownHeaderTextSplitter`**
Splits on `#`, `##`, `###` headers. The header text is preserved as **metadata** on each chunk — useful for filtering in
vector stores.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# Introduction
LangChain is a framework for LLM apps.

## Installation
Run pip install langchain to get started.

## Core Components
It includes loaders, splitters, embeddings, and retrievers.

# Advanced Topics
Agents and chains extend LangChain for complex workflows.

## LangGraph
LangGraph adds stateful, multi-step agent orchestration.
"""

headers_to_split_on = [
	("#", "Header 1"),
	("##", "Header 2"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(markdown_text)

for doc in docs:
	print(doc.page_content)
	print(doc.metadata)  # e.g. {'Header 1': 'Introduction', 'Header 2': 'Installation'}
	print()
```

---

**`HTMLHeaderTextSplitter`**
Splits on HTML heading tags (`<h1>`, `<h2>`, etc.), similarly attaching them as metadata.

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_text = """
<html>
  <body>
    <h1>LangChain Overview</h1>
    <p>LangChain is a framework for building LLM applications.</p>
    <h2>Text Splitters</h2>
    <p>Splitters break documents into chunks for retrieval.</p>
    <h2>Vector Stores</h2>
    <p>Vector stores index embeddings for similarity search.</p>
  </body>
</html>
"""

headers_to_split_on = [
	("h1", "Header 1"),
	("h2", "Header 2"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(html_text)

for doc in docs:
	print(doc.page_content)
	print(doc.metadata)  # e.g. {'Header 1': 'LangChain Overview', 'Header 2': 'Text Splitters'}
	print()
```

---

**`RecursiveJsonSplitter`**
Walks a JSON tree and splits by object/array boundaries, keeping nested keys together as much as possible.

```python
from langchain_text_splitters import RecursiveJsonSplitter

json_data = {
	"framework": "LangChain",
	"version": "0.3",
	"components": {
		"splitters": ["RecursiveCharacter", "Markdown", "HTML", "Code", "JSON"],
		"loaders": ["PDF", "Web", "CSV", "Notion"],
		"stores": ["FAISS", "Chroma", "Pinecone", "pgvector"]
	},
	"use_cases": ["RAG", "Agents", "Summarization", "Extraction"]
}

splitter = RecursiveJsonSplitter(max_chunk_size=100)

# Returns list of dicts (JSON subtrees)
chunks = splitter.split_json(json_data=json_data)
for i, chunk in enumerate(chunks):
	print(f"[Chunk {i + 1}] {chunk}")

# Returns LangChain Document objects
docs = splitter.create_documents(texts=[json_data])
for doc in docs:
	print(doc.page_content)
```

---

**`RecursiveCharacterTextSplitter.from_language`**
A preconfigured recursive splitter aware of code syntax boundaries — splits on functions, classes, and logical blocks
rather than arbitrary characters. Supports Python, JS, Java, Go, Rust, C++, Markdown, HTML, LaTeX, and more.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_code = """
import os

def load_document(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def split_text(text: str, chunk_size: int = 500) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    return splitter.split_text(text)

class RAGPipeline:
    def __init__(self, doc_path: str):
        self.text = load_document(doc_path)
        self.chunks = split_text(self.text)

    def get_chunks(self) -> list[str]:
        return self.chunks
"""

splitter = RecursiveCharacterTextSplitter.from_language(
	language=Language.PYTHON,
	chunk_size=150,
	chunk_overlap=20,
)

chunks = splitter.split_text(python_code)
for i, chunk in enumerate(chunks):
	print(f"[Chunk {i + 1}]\n{chunk}\n")
```

---

## Quick Reference

| Splitter                         | Splits By                                          |
|----------------------------------|----------------------------------------------------|
| `RecursiveCharacterTextSplitter` | Natural language hierarchy (`\n\n` → `\n` → space) |
| `CharacterTextSplitter`          | Single separator + character count                 |
| `from_tiktoken_encoder`          | Single separator + token count                     |
| `TokenTextSplitter`              | Token count only, no separator                     |
| `MarkdownHeaderTextSplitter`     | Markdown `#` headers                               |
| `HTMLHeaderTextSplitter`         | HTML `<h>` tags                                    |
| `RecursiveJsonSplitter`          | JSON object/array boundaries                       |
| `from_language(Language.X)`      | Code syntax boundaries (functions, classes)        |
