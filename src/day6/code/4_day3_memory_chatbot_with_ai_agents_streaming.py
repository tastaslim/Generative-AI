from typing import Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph


def agentic_chat_bot(agent: CompiledStateGraph):
    # Render existing history
    for message_history in st.session_state.message_history:
        role: str = message_history["role"]
        content: str = message_history["content"]
        st.chat_message(role).markdown(content)

    prompt = st.chat_input("Ask me something")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.message_history.append({"role": "user", "content": prompt})
        # ---- Without streaming ------

        # agent_response = agent.invoke(
        #     input={
        #         "messages": [
        #             {
        #                 "role": "user",
        #                 "content": prompt,
        #             }
        #         ]
        #     },
        #     config=RunnableConfig(configurable={"thread_id": 1}),
        # )
        # message: str = agent_response["messages"][-1].content
        # st.chat_message("ai").markdown(message)
        # st.session_state.message_history.append({"role": "ai", "content": message})

        # -----  With streaming -----

        agent_response_iterator: Iterator = agent.stream(
            input={"messages": [{"role": "user", "content": prompt}]},
            config=RunnableConfig(configurable={"thread_id": 1}),
            stream_mode="messages",
        )
        ai_container = st.chat_message("ai")
        with ai_container:
            space = st.empty()
            message = ""
            for chunk in agent_response_iterator:
                message += chunk[0].content
                space.write(message)

        st.session_state.message_history.append({"role": "ai", "content": message})


if __name__ == "__main__":
    load_dotenv()
    groq_llm: ChatGroq = ChatGroq(model="llama-3.3-70b-versatile")
    google_serper_wrapper = GoogleSerperAPIWrapper()
    tools = [google_serper_wrapper.run]
    if "memory" not in st.session_state:
        st.session_state.memory = MemorySaver()
        st.session_state.message_history = []

    search_agent = create_agent(
        model=groq_llm,
        tools=tools,
        system_prompt="You are an Agent who can do google search for given task and find results",
        checkpointer=st.session_state.memory,
    )
    st.title("My Search AI Assistant")
    agentic_chat_bot(search_agent)
