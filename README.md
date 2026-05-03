# Generative AI — Complete Tutorial Series

> A hands-on, beginner-to-advanced tutorial series covering the full Generative AI landscape using Python.
> From understanding LLMs to building production RAG pipelines and Agentic AI systems.

---

## What You Will Learn

This series takes you from **zero to production** across the core pillars of modern Generative AI.

```
LLMs → Prompt Engineering → LangChain → RAG → Agents → Agentic AI → Streamlit Apps
```

---

## Curriculum Overview

### Module 1 — Large Language Models (LLMs)

Understanding the foundation of Generative AI.

- What is an LLM and how does it work
- Tokens, context windows, and temperature
- Popular models — GPT, Claude, Gemini, LLaMA
- Calling LLM APIs in Python
- Prompt design basics

---

### Module 2 — Prompt Engineering

Getting the best output from LLMs.

- Zero-shot, few-shot, and chain-of-thought prompting
- System prompts vs user prompts
- Prompt templates with LangChain
- Structured output (JSON mode)
- Common pitfalls and how to avoid them

---

### Module 3 — LangChain in Python

The most popular framework for building LLM applications.

- LangChain architecture and core abstractions
- Document loaders — PDF, web, CSV, Notion
- Text splitters — RecursiveCharacter, Markdown, Code, JSON
- Prompt templates and output parsers
- Chains — LLMChain, SequentialChain, RouterChain
- Memory — conversation history in chains

---

### Module 4 — RAG (Retrieval-Augmented Generation)

Grounding LLM responses in your own data.

- Why RAG — solving hallucination and knowledge gaps
- RAG pipeline architecture
- Embedding models — what they are and how to choose
- Vector stores — FAISS, Chroma, Pinecone, pgvector
- Chunking strategy — chunk_size, chunk_overlap, splitter selection
- Retrieval strategies — similarity search, MMR, self-query
- Evaluating RAG quality

---

### Module 5 — RAG Pipeline (End-to-End)

Building a complete, production-grade RAG system.

- Document ingestion pipeline
- Embedding and indexing
- Query pipeline — retrieve → rerank → generate
- Metadata filtering
- Conversational RAG with memory
- Streaming responses
- Evaluation and benchmarking

---

### Module 6 — Agents

LLMs that can take actions, not just answer questions.

- What is an Agent — ReAct pattern
- Tools and toolkits in LangChain
- Building custom tools
- Agent types — ReAct, OpenAI Functions, Structured Chat
- Agent memory and state
- Debugging agent traces with LangSmith

---

### Module 7 — Agentic AI

Multi-step, autonomous AI workflows.

- Agentic AI vs single-turn AI
- Planning, acting, and reflecting loops
- Multi-agent systems — roles, communication, orchestration
- LangGraph — stateful agent workflows
- Human-in-the-loop patterns
- Real-world agentic use cases

---

### Module 8 — Streamlit for GenAI Apps

Building interactive frontends for your AI systems.

- Streamlit basics — layout, widgets, state
- Chat UI with `st.chat_message` and `st.chat_input`
- Integrating LangChain chains with Streamlit
- File upload → RAG pipeline in a web app
- Streaming LLM responses to UI
- Deploying Streamlit apps

---

## Tech Stack

| Layer              | Technology                     |
|--------------------|--------------------------------|
| Language           | Python 3.10+                   |
| LLM Framework      | LangChain, LangGraph           |
| LLM Providers      | OpenAI, Anthropic, HuggingFace |
| Embeddings         | OpenAI, sentence-transformers  |
| Vector Stores      | FAISS, Chroma                  |
| Frontend           | Streamlit                      |
| Tokenizer          | tiktoken                       |
| Package Management | pip / requirements.txt         |

---

## Repository Structure

```
Generative-AI/
├── src/
│   ├── 01_llms/
│   ├── 02_prompt_engineering/
│   ├── 03_langchain/
│   ├── 04_rag/
│   ├── 05_rag_pipeline/
│   ├── 06_agents/
│   ├── 07_agentic_ai/
│   └── 08_streamlit/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/tastaslim/Generative-AI.git
cd Generative-AI

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API keys
export OPENAI_API_KEY=your_key_here
```

---

## Who Is This For

- Developers new to Generative AI
- Backend engineers (Python) looking to build LLM-powered systems
- Anyone wanting a structured path from LLM basics to production RAG and Agents

No prior AI/ML knowledge required. Python basics assumed.

---

## License

MIT © [Taslim](https://github.com/tastaslim)