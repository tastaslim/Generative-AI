# LLM vs Agent Memory — The Core Difference

This is the most important thing to understand before anything else.

1. **How an LLM handles memory**

   An LLM has **no memory at all**. Every API call is completely stateless — it receives tokens in, produces tokens out,
   and immediately forgets everything.

    ```
    Call 1:  "My name is Taslim"  →  LLM processes → "Nice to meet you Taslim!"
                                                              ↓
                                                        [forgotten]
    Call 2:  "What is my name?"   →  LLM processes → "I don't know your name"
    ```

   The LLM is a **pure function** — same inputs always produce similar outputs, with zero awareness of prior calls.

2. **How an Agent handles memory**

   An agent wraps the LLM with a **memory layer** that persists and injects context:

    ```
    Call 1:  "My name is Taslim"  →  Agent saves to memory store
                                      LLM processes → "Nice to meet you Taslim!"
    
    Call 2:  "What is my name?"   →  Agent retrieves memory
                                      Injects: "Earlier the user said: My name is Taslim"
                                      LLM processes → "Your name is Taslim"
    ```

   The LLM still has no memory. The **agent gives it the illusion of memory** by injecting past context into each new
   prompt.

### Side by side

|                         | LLM (bare)                     | AI Agent                       |
|-------------------------|--------------------------------|--------------------------------|
| Memory across calls     | None — stateless               | Yes — via checkpointer         |
| Memory across sessions  | None                           | Yes — via persistent store     |
| Knows previous messages | Only if you manually pass them | Automatically managed          |
| Stores user preferences | No                             | Yes — via custom state         |
| Mechanism               | Pure function                  | State machine with persistence |
| Analogy                 | Genius with amnesia            | Genius with a notebook         |

> **Key insight:** The LLM never gains memory. The agent manages what context to feed into each LLM call. Memory is an
> agent concern, not an LLM concern.

---

# What is Memory in an Agent?

Memory is the system that **persists, retrieves, and manages context** across one or many conversations. It determines
what the agent knows about past interactions when responding to the current message.

Without memory:

```
User: "I prefer formal responses"
Agent: "Noted!"

[next session]

User: "Write me an email"
Agent: [writes casual email — forgot the preference entirely]
```

With memory:

```
User: "I prefer formal responses"
Agent: "Noted!"  → saved to memory

[next session]

User: "Write me an email"
Agent: [retrieves preference] → writes formal email automatically
```

---

## Types of Memory

There are 4 types of memory.

### 1. Sensory Memory

The raw input of the **current request only** — the message just sent, any uploaded files, screenshots, or tool results
in this single call. It lives in the context window and is gone the moment the call ends.

**Scope:** Current request only
**Storage:** Context window (RAM)
**Survives restart:** No

```python
# Sensory memory — just what is in this call right now
response = agent.invoke({
	"messages": [
		{"role": "user", "content": "Summarise this PDF"},
		# the PDF content injected here is sensory memory
	]
})
# After this call — everything gone
```

**Use cases:**

- Single-turn Q&A (maths, factual lookups)
- File analysis — user uploads a PDF for this session only
- Image understanding — screenshot passed as base64
- Tool observations — API results fed back in the same call

---

### 2. Short-term Memory

The **rolling conversation history** of the current session. The agent remembers what was said earlier in the same chat
thread. Cleared when the session ends (unless persisted with a checkpointer).

**Scope:** Current session / thread
**Storage:** In-memory buffer or checkpointer
**Survives restart:** Only with a persistent checkpointer

```python
from langchain_core.messages import HumanMessage, AIMessage

# Short-term memory is the growing message list
messages = [
	HumanMessage(content="My name is Taslim"),
	AIMessage(content="Nice to meet you Taslim!"),
	HumanMessage(content="What is my name?"),
	# LLM sees the full history above — can answer "Taslim"
]
```

**Use cases:**

- Multi-turn chat — agent recalls what was said earlier in the chat
- Clarification loops — "as I mentioned above, the budget is 50k"
- Debugging sessions — agent tracks all steps tried so far
- Form filling — collects name, email, preferences across multiple turns

---

### 3. Long-term Memory

Information **persisted across sessions** in an external store — a vector database, SQL database, or file system. The
agent retrieves relevant facts at query time via semantic search. Survives restarts, deployments, and weeks of
inactivity.

**Scope:** Across all sessions permanently
**Storage:** Vector DB (Chroma, FAISS, Pinecone) or SQL
**Survives restart:** Yes — explicitly persisted

```python
from langchain_community.vectorstores import Chroma
from langchain.memory import VectorStoreRetrieverMemory

vectorstore = Chroma(
	collection_name="agent_memory",
	embedding_function=embeddings,
	persist_directory="./memory_store"  # survives restarts
)

memory = VectorStoreRetrieverMemory(
	retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Save a fact permanently
memory.save_context(
	{"input": "I prefer dark mode and formal responses"},
	{"output": "Preference saved"}
)

# Weeks later, in a new session — retrieves relevant facts automatically
```

**Use cases:**

- Personal assistant — remembers user preferences and habits across weeks
- Customer support — recalls all past tickets and resolutions for a user
- Knowledge base — company docs and policies retrieved when relevant
- RAG pipelines — semantic retrieval over thousands of documents

---

### 4. Entity Memory

**Structured memory** that tracks specific named entities — people, organisations, products, places — and facts about
them. Unlike vector memory which retrieves by similarity, entity memory is explicitly keyed by entity name.

**Scope:** Session or persistent
**Storage:** Key-value store or graph DB
**Survives restart:** Configurable

```python
from langchain.memory import ConversationEntityMemory
from langchain_groq import ChatGroq

entity_memory = ConversationEntityMemory(
	llm=ChatGroq(model="llama-3.3-70b-versatile"),
	return_messages=True
)

# After: "Taslim is a principal engineer based in Hyderabad"
# Entity memory extracts and stores:
# {
#   "Taslim": "principal engineer, based in Hyderabad"
# }

# Later: "What does Taslim do?"
# Retrieves: "principal engineer, based in Hyderabad"
```

**Use cases:**

- CRM agent — tracks each customer's plan, location, open issues
- HR assistant — remembers each employee's role, team, leave history
- Sales pipeline — tracks each lead's stage, contact, last interaction
- Project tracker — knows each project's status, owner, and deadline

---

### Memory Types at a Glance

| Type       | Scope           | Storage               | Lost when               | Best for                         |
|------------|-----------------|-----------------------|-------------------------|----------------------------------|
| Sensory    | Current call    | Context window        | Call ends               | Single-turn tasks, file analysis |
| Short-term | Current session | Buffer / checkpointer | Session ends            | Multi-turn chat                  |
| Long-term  | All sessions    | Vector DB / SQL       | Never (explicit delete) | Personalisation, RAG             |
| Entity     | Session or all  | Key-value / graph     | Configurable            | Tracking facts about things      |

---

## Implementation — Short-term Memory

### The one thing you add — `checkpointer`

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_react_agent(
	model=llm,
	tools=[],
	checkpointer=InMemorySaver()  # ← this single line adds memory
)

# thread_id groups messages into one conversation
agent.invoke(
	{"messages": [{"role": "user", "content": "My name is Taslim"}]},
	{"configurable": {"thread_id": "session_1"}}
)

agent.invoke(
	{"messages": [{"role": "user", "content": "What is my name?"}]},
	{"configurable": {"thread_id": "session_1"}}  # same thread_id = same conversation
)
# → "Your name is Taslim"
```

> `thread_id` works like an email thread — it groups all messages from one conversation together. Different users should
> have different thread IDs.

### Dev vs Production

```python
# Development — stores in RAM, lost on restart
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# Production — stores in Postgres, survives restarts
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/agentdb"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
	checkpointer.setup()  # auto-creates required tables
	agent = create_react_agent(
		model=llm,
		tools=[],
		checkpointer=checkpointer
	)
```

Other supported checkpointers: `SQLiteSaver`, `PostgresSaver`, `AzureCosmosDBSaver`.

---

## The Long Conversation Problem

As conversations grow, the message list grows too. Two things go wrong:

```
Short conversation  →  fits in context window  →  works fine
Long conversation   →  exceeds context window  →  errors / degraded responses
```

Even if the context fits, LLMs perform worse over very long contexts — they get distracted by stale content, slow down,
and cost more.

LangChain gives you 3 solutions.

---

### Solution 1 — Trim Messages

Keep only the most recent N messages. Fast, zero token cost, but **loses old context**.

```python
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver


@before_model
def trim_messages(state, runtime):
	messages = state["messages"]

	if len(messages) <= 3:
		return None  # nothing to trim yet

	first_msg = messages[0]  # always keep the very first message
	recent = messages[-3:]  # keep last 3 messages
	return {
		"messages": [
			RemoveMessage(id=REMOVE_ALL_MESSAGES),
			first_msg,
			*recent
		]
	}


agent = create_react_agent(
	model=llm,
	tools=[],
	middleware=[trim_messages],  # plug in as middleware
	checkpointer=InMemorySaver()
)
```

**When to use:** Simple chatbots, short-lived sessions, when old context is irrelevant.

---

### Solution 2 — Delete Messages

Remove specific messages or wipe the entire history. Useful for explicit control or compliance requirements.

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES


# Delete the oldest 2 messages
def delete_old(state):
	messages = state["messages"]
	if len(messages) > 2:
		return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}


# Delete everything
def delete_all(state):
	return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

> **Warning:** After deleting, ensure the history still starts with a `user` message — most LLM providers require this.
> Also ensure every `assistant` tool call message is followed by its corresponding `tool` result message.

**When to use:** Compliance workflows, removing sensitive content, hard resets.

---

### Solution 3 — Summarise Messages (Recommended)

Compress old messages into a summary instead of deleting them. **Preserves context, saves tokens.** The smartest
approach for production.

```python
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_react_agent(
	model=llm,
	tools=[],
	middleware=[
		SummarizationMiddleware(
			model=llm,
			trigger=("tokens", 4000),  # start summarising at 4000 tokens
			keep=("messages", 20)  # always keep last 20 messages raw
		)
	],
	checkpointer=InMemorySaver()
)

agent.invoke({"messages": "hi, my name is Taslim"}, config)
agent.invoke({"messages": "I work as a principal engineer"}, config)
agent.invoke({"messages": "I am based in Hyderabad"}, config)
final = agent.invoke({"messages": "What do you know about me?"}, config)

# → "Your name is Taslim, you are a principal engineer based in Hyderabad"
```

**When to use:** Long-running assistants, customer support agents, any session where context must be preserved.

---

### Which Solution to Use

| Situation                                     | Solution                               |
|-----------------------------------------------|----------------------------------------|
| Short conversations                           | No trimming needed                     |
| Long conversations, old context is irrelevant | `trim_messages`                        |
| Must remove specific messages                 | `delete_messages`                      |
| Long conversations, context must be preserved | `SummarizationMiddleware`              |
| Production, multi-user                        | `PostgresSaver` + `thread_id` per user |

---

## Custom State — Store More Than Messages

By default the agent only remembers messages. Extend `AgentState` to store anything you need:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState


class MyState(MessagesState):
	user_id: str  # custom field
	preferences: dict  # custom field
	language: str  # custom field


agent = create_react_agent(
	model=llm,
	tools=[],
	state_schema=MyState,
	checkpointer=InMemorySaver()
)

agent.invoke(
	{
		"messages": [{"role": "user", "content": "Hello"}],
		"user_id": "user_123",
		"preferences": {"theme": "dark", "tone": "formal"},
		"language": "en"
	},
	{"configurable": {"thread_id": "1"}}
)
```

**Use cases:** Storing user tier, language preference, onboarding step, subscription plan — anything the agent needs to
personalise its behaviour.

---

## Reading and Writing State from Tools

Tools can read and write agent memory during execution — enabling agents that update their own state mid-task.

### Read state in a tool

```python
from langchain_core.tools import tool


# Access agent state inside a tool
@tool
def get_user_info(config: dict) -> str:
	"""Look up current user information from agent state."""
	user_id = config.get("configurable", {}).get("user_id")
	return f"User ID: {user_id}"
```

### Write state from a tool

```python
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage


@tool
def save_user_preference(preference: str, tool_call_id: str) -> Command:
	"""Save a user preference to agent state."""
	return Command(update={
		"preferences": {"saved": preference},
		"messages": [
			ToolMessage(
				content=f"Preference '{preference}' saved.",
				tool_call_id=tool_call_id
			)
		]
	})
```

---

## Complete Working Example

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool
def get_weather(city: str) -> str:
	"""Get the weather in a city."""
	return f"The weather in {city} is 28°C and sunny."


llm = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_react_agent(
	model=llm,
	tools=[get_weather],
	checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "user_taslim_001"}}

# Turn 1
agent.invoke(
	{"messages": [{"role": "user", "content": "My name is Taslim and I live in Hyderabad"}]},
	config
)

# Turn 2 — agent remembers name and location
agent.invoke(
	{"messages": [{"role": "user", "content": "What is the weather where I live?"}]},
	config
	# agent recalls "Hyderabad" from turn 1 → calls get_weather("Hyderabad")
)

# Turn 3 — agent still remembers
result = agent.invoke(
	{"messages": [{"role": "user", "content": "What is my name?"}]},
	config
)

print(result["messages"][-1].content)
# → "Your name is Taslim."
```

---

## Summary

```
LLM alone        →  stateless pure function, no memory
Agent + memory   →  LLM + checkpointer + context injection

Short-term       →  InMemorySaver + thread_id
Long-term        →  PostgresSaver + vector store
Custom state     →  extend MessagesState
Long context     →  trim / delete / summarise
```

> Memory is not a feature of the LLM. It is a feature of the agent runtime built around it.
