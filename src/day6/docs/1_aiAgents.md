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
Agent = An AI Agent = LLM (brain) + Tools (hands) + Memory (notepad) + Loop (autonomy)
```

![LLM vs Agent](llm_vs_agent_architecture.svg)

---

# ReAct(Reasoning + Acting)

ReAct is one of the most important patterns in agentic AI. it is what turns an LLM from “just answering” into something
that can reason + act in a loop. ReAct is the control loop that makes LLMs behave like agents instead of chatbots.

**Thought → Action → Observation → Thought → Action → Observation → Thought → Action →... → Final Answer**

```text
Who is the CEO of Tesla and what is his age?

Thought: I need CEO of Tesla
Action: search("Tesla CEO")
Observation: Elon Musk is CEO of Tesla

Thought: Now I need his age
Action: search("Elon Musk age")
Observation: 54

Final Answer: Elon Musk, 54 years old
```

## Why ReAct is powerful

### 1. Breaks complex problems

Instead of guessing everything in one go, it

- decomposes tasks
- reduces hallucination

### 2. Enables tool usage

- It works with APIs, databases, search engines, code execution.
- It is used heavily in frameworks like LangChain.

### 3. Iterative correction

If something fails:

- **Observation**: API failed
- **Thought**: Try alternative API

### 4.Transparent reasoning

You can see how the model is thinking (useful for debugging). You see these in most LLM, where they keep on printing
something like:

thinking: some content
thought for 1 seconds
observation: I need to do this that blah blah blah

```text
while not done:
    think = LLM(context)
    action = parse(think)
    observation = execute(action)
    context += think + observation
```

## Where ReAct is used

- Coding agents (debug → run → fix)
- Research agents (search → read → summarize)
- Support bots (query → fetch → respond)
- DevOps agents (detect → analyze → act)

## Agents vs ReAct

- **Agent** is the system. **ReAct** is the strategy the LLM uses to think inside that system.
- You can build an agent without **ReAct** — it will just call tools more blindly, make more mistakes, and produce no
  auditable reasoning trace. ReAct is what gives agents the ability to self-correct mid-task.

![ReaAct vs Agent](ReAct_Agent.png)


