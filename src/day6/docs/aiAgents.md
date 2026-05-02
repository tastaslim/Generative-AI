# Problem

- LLMs are powerful—but on their own, they’re basically stateless text generators. The only task which LLM can do on
  its own is given a text input, it will generate some text output.
- It can not take actions on its own like calling some APIs, querying databases, doing google search for something which
  is not available in LLM knowledge base.

### ❌ No persistence

It doesn’t remember past steps unless you pass everything again(Like we have doing in our practice codes, where we are
passing all history + current prompt to it for each query.)

### ❌ No decision loops

It won’t say:
"Hmm, this failed, let me retry differently."

### ❌ No tool usage (natively)

It can suggest calling an API, but it can’t actually do it.

### ❌ No planning

It doesn’t break big problems into structured steps unless guided.

### ❌ No execution

It can write code—but it won’t run it.

---

In real world AI Applications, we need all of these capabilities where our system should be smart enough to make
decisions, execute codes, make API calls, can interact with tools(e.g. MCP tools), correct the mistakes it made etc.

# AI Agents

AI Agents come into picture to help with above problem. They can do all of above tasks.

### 1. Planning

Agents can break problems into steps:

“Search → analyze → summarize → verify”

### 2. Memory

They maintain:

Short-term memory (conversation)
Long-term memory (vector DB, logs, state)

### 3. Tool usage

Agents can actually do things:

Call APIs
Query databases
Run code
Trigger workflows

Frameworks like LangChain and AutoGPT help orchestrate this.

### 4. Iteration & feedback loops

Agents can:

Retry on failure
Evaluate outputs
Improve results

### 5. Autonomy (to a degree)

Instead of saying "Write an email", You can say, "Find top 5 candidates, analyze resumes, draft outreach, send emails"
And the agent coordinates everything.

```text
LLM = Brain
Agent = Brain + Memory + Hands + Decision system
```