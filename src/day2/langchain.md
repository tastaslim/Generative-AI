# LangChain
LangChain is a framework for building applications powered by large language models (LLMs) like OpenAI’s GPT models, Anthropic’s Claude, Google Gemini or open-source models.
Instead of just calling an LLM once, LangChain helps you orchestrate complex workflows where the model can:
- Remember context
- Call external tools (APIs, databases)
- Chain multiple steps together
- Act more like an intelligent agent

## Core Idea (Why LangChain exists)

Raw LLM usage looks like:
```text
Input → LLM → Output
```

LangChain turns it into something more powerful:
```text
Input → Prompt → LLM → Tool/API → Memory → Next Step → Final Output
```
It helps you build real-world AI systems, not just chat responses.