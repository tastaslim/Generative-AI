from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph


def manage_todo_task(agent: CompiledStateGraph):
    while True:
        prompt = input("User: ")
        if prompt in ("quit", "bye", "exit"):
            print("Goodbye!👋")
            break
        response = agent.invoke(
            input={"messages": {"role": "user", "content": prompt}},
            config=RunnableConfig(configurable={"thread_id": 1}),
        )
        result = response["messages"][-1].content
        print(f"{result}")


if __name__ == "__main__":
    load_dotenv()
    db = SQLDatabase.from_uri("sqlite:///my_todo_task.sqlite")
    db.run("""CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK(status IN ('completed', 'in_progress', 'pending')) default 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)
    # 1 LLM
    llm_provider: BaseChatModel = ChatOpenAI(model="gpt-5.4")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm_provider)
    # 2 TOOLS
    tools = toolkit.get_tools()
    # 3 SYSTEM_PROMPT
    system_prompt = """ 
    You are an a Intelligent System which interacts which SQL database to manage tasks.
    TASK RULES:
    1. LIMIT SELECT queries to 10 results max with ORDER BY created_at DESC
    2. After CREATE/UPDATE/DELETE, confirm with SELECT query
    3. If a user requests a list of tasks, present te output in a structured table format to the user.
    
    CRUD OPERATIONS:
    CREATE: INSERT INTO tasks (title, description, status, created_at)
    READ: SELECT * FROM tasks ORDER BY created_at DESC LIMIT 10;
    UPDATE: UPDATE tasks SET status = ? WHERE id = ? or title = ?;
    DELETE: DELETE FROM tasks WHERE id = ? or title = ?;
    
    TABLE schema: id, title, description, status(pending/in_progress/completed), created_at
    """
    memory = MemorySaver()
    # 4 AGENT
    todo_task_agent: CompiledStateGraph = create_agent(
        model=llm_provider,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory,
    )
    manage_todo_task(todo_task_agent)
