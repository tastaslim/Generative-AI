## What is a Tool / Tools in AI Agents?

A **tool** is any function or capability that an AI Agent is allowed to call during task execution.
The LLM itself is a text processor — it reads tokens and predicts the next ones. It cannot browse the internet, run
code, make API calls, send emails, or query a database on its own. Tools are what bridge this gap.

Think of it this way:

> The **LLM** is the brain. **Tools** are the hands, eyes, and phone. Without tools, the brain is locked in a room with
> no way to interact with the world.

### The simplest possible definition

```
Tool = name + description + function + parameters
```

- **Name** — what the LLM calls the tool (`web_search`, `send_email`)
- **Description** — tells the LLM *when* and *why* to use this tool (critical for accuracy)
- **Parameters** — typed inputs the LLM must supply (`query: string`, `url: string`)
- **Function** — the actual Python code that runs when the tool

### How a tool fits into the agent loop

```
User prompt
    ↓
LLM reads tool descriptions in system prompt
    ↓
LLM outputs structured tool call  ← this is NOT prose, it is JSON
    ↓
Agent executor intercepts and runs the real function
    ↓
Result returned as "Observation" to the LLM
    ↓
LLM reasons on real data → decides next step
    ↓
Repeat until task is done
```

> **Key insight:** The LLM never runs code. It only *describes* what to run. The executor does the actual work.

---

## Why Tools Are Important

### 1. They break the frozen knowledge barrier

An LLM's training data has a cutoff date. Ask it about yesterday's stock price or your internal documentation — it will
either say it doesn't know, or worse, confidently make something up (hallucinate).

Tools solve this by letting the agent **fetch live data at runtime** — from the web, your database, your vector store,
or any API.

| Without tools                    | With tools                                        |
|----------------------------------|---------------------------------------------------|
| "I don't know the current price" | Calls `web_search` → returns live data            |
| Hallucinates an outdated answer  | Calls `db_query` → returns your actual record     |
| Cannot read your internal docs   | Calls `vector_search` → retrieves relevant chunks |

### 2. They enable real-world actions

Reading is only half of it. Tools give agents the ability to **write to the world** — send an email, create a ticket,
post a Slack message, charge a payment, update a record.

Without tools, an agent can only recommend. With tools, it can *do*.

### 3. They make reasoning grounded

The **ReAct** loop (Thought → Action → **Observation** → Thought) depends entirely on tools. Without tool results coming
back as observations, the LLM is reasoning in the dark — producing plausible-sounding but unverified conclusions.

Tools are what close the Perceive → Decide → Act → Observe loop.

### 4. They are the primary safety boundary

You control which tools are registered. If `delete_production_database()` is not a registered tool, the agent cannot
call it — no matter what the user asks. Tool design is your first and most important guardrail.

### 5. They are composable

A complex agent can chain tool calls — search, then fetch, then compute, then notify. Each tool does one thing well.
Composing them enables workflows that would require entire pipelines to build manually.

---

## Types of Tools

There are 6 major categories. Every tool in every agent framework falls into one of these.

---

### Category 1 — Search & Retrieval Tools

**What they do:** Fetch information the LLM does not have. These solve the frozen knowledge and hallucination problems.

**When the agent uses them:** Whenever it needs current data, external facts, or proprietary internal information.

| Tool                 | What it does                                     |
|----------------------|--------------------------------------------------|
| `web_search`         | Search the internet for live results             |
| `web_fetch`          | Fetch and parse a specific URL                   |
| `vector_search`      | Semantic search over a vector store (RAG)        |
| `db_query`           | Run a SQL query on a relational database         |
| `document_retriever` | Retrieve relevant chunks from uploaded documents |

**Real-world example:**

An agent is asked: *"What did our CEO say about Q3 revenue in the last board meeting?"*

```
Thought: I need to search our internal document store for board meeting notes.
Action: vector_search
Action Input: {"query": "CEO Q3 revenue board meeting", "top_k": 3}
Observation: [
  {"chunk": "CEO stated Q3 revenue was ₹42Cr, up 18% YoY...", "source": "board_meeting_oct_2024.pdf"}
]
Thought: Found the relevant passage. I can now answer directly.
Final Answer: According to the October 2024 board meeting notes, the CEO reported...
```

**LangChain code:**

```python
from langchain.tools import tool


@tool
def vector_search(query: str, top_k: int = 3) -> str:
	"""Search internal documents for relevant information.
    Use this when the user asks about company data, policies, or past decisions."""
	results = vectorstore.similarity_search(query, k=top_k)
	return [{"chunk": r.page_content, "source": r.metadata["source"]} for r in results]
```

---

### Category 2 — Code & Compute Tools

**What they do:** Execute code or perform computation in a sandboxed environment. The LLM writes the logic; the tool
runs it and returns the output.

**When the agent uses them:** Data analysis, mathematical calculations, file transformations, anything requiring
programmatic logic.

| Tool              | What it does                                 |
|-------------------|----------------------------------------------|
| `python_repl`     | Execute Python code and return stdout/result |
| `javascript_eval` | Run JavaScript in a sandbox                  |
| `bash_exec`       | Execute shell commands in a container        |
| `math_eval`       | Evaluate mathematical expressions safely     |
| `sql_executor`    | Run validated SQL against a database         |

**Real-world example:**

An agent is asked: *"Analyse our sales CSV and tell me which region grew fastest last quarter."*

```
Thought: I need to load and analyse a CSV file. I'll use Python for this.
Action: python_repl
Action Input: {
  "code": "import pandas as pd\ndf = pd.read_csv('/data/sales_q3.csv')\ngrowth = df.groupby('region')['revenue'].sum()\nprint(growth.pct_change().sort_values(ascending=False))"
}
Observation: 
  South     0.34
  West      0.21
  North     0.08
  East     -0.03
Thought: South region grew 34% — the fastest. East declined slightly.
Final Answer: The South region grew fastest at 34% last quarter...
```

**LangChain code:**

```python
from langchain_experimental.tools import PythonREPLTool

python_tool = PythonREPLTool()
# Built-in — wraps Python exec() in a safe subprocess
```

---

### Category 3 — Action & Automation Tools

**What they do:** Cause real-world side effects. These are the tools that make agents genuinely autonomous — they send
messages, create records, and trigger workflows.

**When the agent uses them:** After reasoning is complete and a decision has been made. These are typically the *last*
tool call in a sequence.

| Tool              | What it does                       |
|-------------------|------------------------------------|
| `send_email`      | Send an email via SMTP or SendGrid |
| `slack_post`      | Post a message to a Slack channel  |
| `create_ticket`   | Create a Jira/GitHub/Linear issue  |
| `calendar_event`  | Create or update a calendar event  |
| `webhook_trigger` | Call any webhook with a payload    |

**Real-world example:**

An agent monitors pricing and is triggered automatically every morning:

```
Thought: CompetitorB dropped their price below ours. I need to alert the team.
Action: slack_post
Action Input: {
  "channel": "#pricing-alerts",
  "text": "Alert: CompetitorB reduced Growth plan $75 → $65/mo. Our Pro is $79. Review recommended."
}
Observation: {"ok": true, "ts": "1712345678.001"}
Thought: Alert sent successfully.
Final Answer: Slack alert posted to #pricing-alerts. No further action required.
```

**LangChain code:**

```python
from langchain.tools import tool


@tool
def slack_post(channel: str, text: str) -> dict:
	"""Post a message to a Slack channel.
    Use this to send alerts, summaries, or notifications to the team."""
	response = slack_client.chat_postMessage(channel=channel, text=text)
	return {"ok": response["ok"], "ts": response["ts"]}
```

> **Design principle:** Action tools should be used *last* in the reasoning chain, after the agent has verified its
> facts using retrieval tools. Never let an agent send an alert based on unverified data.

---

### Category 4 — File & Data Tools

**What they do:** Read from and write to files, spreadsheets, PDFs, and cloud storage. Essential for document-heavy
enterprise workflows.

**When the agent uses them:** When the input or output of a task is a file — parsing invoices, generating reports,
reading configuration, writing summaries.

| Tool                | What it does                                 |
|---------------------|----------------------------------------------|
| `read_file`         | Read file contents from disk or storage      |
| `write_file`        | Write or append content to a file            |
| `parse_pdf`         | Extract text, tables, and metadata from PDFs |
| `excel_read`        | Read data from Excel spreadsheets            |
| `s3_get` / `s3_put` | Download/upload files from AWS S3            |

**Real-world example:**

An agent is asked: *"Read the invoice PDF and extract the total amount and vendor name."*

```
Thought: I need to parse a PDF to extract structured data.
Action: parse_pdf
Action Input: {"file_path": "/uploads/invoice_nov2024.pdf"}
Observation: {
  "vendor": "Razorpay Technologies Pvt Ltd",
  "invoice_number": "INV-2024-0892",
  "total_amount": "₹1,24,500",
  "due_date": "2024-12-15"
}
Thought: Got the structured data. Now I'll write it to our tracker.
Action: write_file
Action Input: {"path": "invoice_log.csv", "content": "Razorpay,INV-2024-0892,124500,2024-12-15\n"}
Final Answer: Invoice from Razorpay for ₹1,24,500 extracted and logged. Due 15 Dec 2024.
```

---

### Category 5 — External API Tools

**What they do:** Wrap any third-party HTTP API as a callable tool — payments, maps, CRM, weather, analytics. Each tool
is a typed wrapper around one API endpoint.

**When the agent uses them:** When the task involves third-party data or services that don't have a dedicated tool
category — essentially, anything with a REST API.

| Tool              | What it does                              |
|-------------------|-------------------------------------------|
| `stripe_charge`   | Create a payment charge via Stripe        |
| `maps_geocode`    | Convert an address to lat/lon coordinates |
| `weather_get`     | Get current weather for a city            |
| `crm_lookup`      | Search a contact in Salesforce or HubSpot |
| `razorpay_refund` | Initiate a refund via Razorpay            |

**Real-world example:**

A customer support agent handles: *"The customer at order #8821 wants a refund — approve it."*

```
Thought: I need to look up the order first, then initiate the refund.
Action: crm_lookup
Action Input: {"order_id": "8821"}
Observation: {"customer": "Anjali Sharma", "amount": 2499, "status": "delivered", "eligible_for_refund": true}

Thought: Order is eligible. Initiating refund via Razorpay.
Action: razorpay_refund
Action Input: {"order_id": "8821", "amount": 2499, "reason": "customer_request"}
Observation: {"refund_id": "rfnd_NxP92k", "status": "initiated", "eta": "5-7 business days"}

Final Answer: Refund of ₹2,499 initiated for Anjali Sharma (order #8821). 
             Refund ID: rfnd_NxP92k. Expected in 5-7 business days.
```

---

### Category 6 — Agent Tools (Multi-Agent)

**What they do:** Let an agent spawn sub-agents, hand off tasks to specialist agents, or pause for human approval. This
is the foundation of multi-agent architectures.

**When the agent uses them:** When a task is too complex for one agent, when specialist knowledge is required, or when a
human must approve an action before it proceeds.

| Tool            | What it does                                         |
|-----------------|------------------------------------------------------|
| `spawn_agent`   | Launch a sub-agent with a specific task              |
| `handoff`       | Transfer control to a specialist agent               |
| `human_in_loop` | Pause and wait for human approval                    |
| `memory_write`  | Store a fact in long-term memory for future sessions |
| `memory_read`   | Retrieve a stored fact from long-term memory         |

**Real-world example:**

An orchestrator agent manages a sales pipeline:

```
Thought: This lead needs both technical qualification and commercial pricing.
         I'll spawn specialist agents for each.

Action: spawn_agent
Action Input: {
  "task": "Qualify this lead technically: they need real-time data pipelines at 10M events/day",
  "agent": "solutions-engineer-agent"
}
Observation: {"verdict": "technically_qualified", "recommended_tier": "enterprise"}

Action: spawn_agent
Action Input: {
  "task": "Generate a custom quote for enterprise tier, 10M events/day",
  "agent": "pricing-agent"
}
Observation: {"quote": "₹4,80,000/year", "discount_applied": "20%"}

Action: human_in_loop
Action Input: {"question": "Quote of ₹4,80,000/year ready. Approve to send to client?"}
Observation: {"approved": true, "approver": "sales_manager@company.com"}

Final Answer: Quote approved and sent to client.
```

---

## Summary — Tool Categories at a Glance

| Category            | Direction        | Solves                               | Example                        |
|---------------------|------------------|--------------------------------------|--------------------------------|
| Search & Retrieval  | Read from world  | Frozen knowledge, hallucination      | `web_search`, `vector_search`  |
| Code & Compute      | Run logic        | Complex calculations, data analysis  | `python_repl`, `sql_executor`  |
| Action & Automation | Write to world   | Autonomy, notifications, workflows   | `send_email`, `slack_post`     |
| File & Data         | Read/write files | Document workflows, reports          | `parse_pdf`, `write_file`      |
| External APIs       | Call third-party | Payments, CRM, maps, weather         | `stripe_charge`, `crm_lookup`  |
| Agent Tools         | Spawn agents     | Complex multi-step, specialist tasks | `spawn_agent`, `human_in_loop` |

---

## Tool Design Best Practices

These apply whether you are building with LangChain, LlamaIndex, or any other framework.

### 1. Write descriptions like instructions, not labels

The description is the only way the LLM decides when to use a tool. Be specific.

```python
# Bad — too vague
from langchain.tools import tool


@tool
def search(query: str) -> str:
	"""Search for information."""


# Good — tells the LLM exactly when to use it
@tool
def webSearch(query: str) -> str:
	"""Search the internet for current, real-time information.
    Use this when the user asks about recent events, live prices,
    or any data that may have changed since your training cutoff.
    Do NOT use this for historical facts you already know."""
```

### 2. Return structured data, not prose

Return JSON or typed objects — not strings. The LLM can reason over structured data far more reliably.

```python
# Bad
return "The price is $65 per month"

# Good
return {"price": 65, "currency": "USD", "plan": "Growth", "billing": "monthly"}
```

### 3. Keep tools focused — one tool, one job

Do not build a single `do_everything` tool. Small, composable tools are more reliable and easier to debug.

```python
from langchain.tools import tool


# Bad
@tool
def manage_slack(action: str, channel: str, message: str = "", user: str = ""):
	"""Send messages, invite users, or create channels in Slack."""


# Good — separate tools for separate actions
@tool
def slack_post_message(channel: str, text: str) -> dict: ...


@tool
def slack_invite_user(channel: str, user_id: str) -> dict: ...
```

### 4. Always validate and handle errors

Tool failures should return informative error messages, not raise exceptions. The LLM can then decide to retry or take a
different path.

```python
from langchain.tools import tool


@tool
def web_search(query: str) -> str:
	"""Search the internet for current information."""
	try:
		results = search_api.query(query)
		return results[:3]
	except RateLimitError:
		return {"error": "Rate limit hit. Try again in 60 seconds."}
	except Exception as e:
		return {"error": f"Search failed: {str(e)}"}
```

### 5. Log every tool call

In production, log the tool name, arguments, and result for every call. This is your audit trail and your primary
debugging surface.

---

## Best Practices

- LLMs are trained heavily on the programming language at which you are working on. In case of **Python**, we
  should mostly use snake_case naming convention because they naturally associate with Python.
  In case of **TypeScript**, you should use camelCasing.
- Variables, methods, classes should have proper/self-explanatory names which helps LLMs in better selection accuracy.
- We should perefer to use Pydantic Models(if feasible) while working with Python and proper description of each
  varibles to make LLMs produce expected results with accuracy.
- Docstring for each method, class, enums, models should be concise so that LLM can interpret from them.

```text
"""Add two integers and return their sum."""
```