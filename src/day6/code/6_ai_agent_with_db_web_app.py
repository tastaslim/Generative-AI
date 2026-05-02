import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph


@st.cache_resource
def get_agent(
    _llm_provider, _tools, _prompt
):  # The _ prefix tells Streamlit to skip hashing that argument. All three params likely contain Pydantic objects with functions internally, so prefix all of them to be safe.
    memory = MemorySaver()
    # 4 AGENT
    agent: CompiledStateGraph = create_agent(
        model=_llm_provider,
        tools=_tools,
        system_prompt=_prompt,
        checkpointer=memory,
    )
    return agent


def manage_todo_task(agent: CompiledStateGraph):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)

    prompt = st.chat_input("I am your task manager")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("ai"):
            with st.spinner("Processing"):
                response = agent.invoke(
                    input={"messages": {"role": "user", "content": prompt}},
                    config=RunnableConfig(configurable={"thread_id": 1}),
                )
                result = response["messages"][-1].content
                st.markdown(result)
                st.session_state.messages.append({"role": "ai", "content": result})


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
    open_ai_llm_provider: BaseChatModel = ChatOpenAI(model="gpt-5.4")
    toolkit = SQLDatabaseToolkit(db=db, llm=open_ai_llm_provider)
    # 2 TOOLS
    sql_database_tools = toolkit.get_tools()
    # 3 SYSTEM_PROMPT
    system_prompt = """ 
    You are an a Intelligent System which interacts which SQL database to manage tasks.
    TASK RULES:
    1. LIMIT SELECT queries to 10 results max with ORDER BY created_at
    2. After CREATE/UPDATE/DELETE, confirm with SELECT query
    3. If a user requests a list of tasks, present te output in a structured table format to the user.

    CRUD OPERATIONS:
    CREATE: INSERT INTO tasks (title, description, status, created_at)
    READ: SELECT * FROM tasks ORDER BY created_at LIMIT 10;
    UPDATE: UPDATE tasks SET status = ? WHERE id = ? or title = ?;
    DELETE: DELETE FROM tasks WHERE id = ? or title = ?;

    TABLE schema: id, title, description, status(pending/in_progress/completed), created_at
    """
    st.title("TODO Task Manager")
    todo_task_agent = get_agent(open_ai_llm_provider, sql_database_tools, system_prompt)
    manage_todo_task(todo_task_agent)
