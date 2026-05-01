from typing import Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


def chatBotWithStreamlitStreaming(llmProvider: BaseChatModel):
    """
    Runs a Streamlit-based QA chatbot with persistent chat history and streaming.
    :param llmProvider: The LLM backend to invoke.
    :type llmProvider: BaseChatModel
    """
    outputParser = StrOutputParser()
    # Render existing history
    for message in st.session_state.messages:
        role: str = message["role"]
        content: str = message["content"]
        st.chat_message(role).markdown(content)

    query = st.chat_input("Please enter your question")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        chain = llmProvider | outputParser
        with st.chat_message("ai"):
            # replaced .invoke() with .stream()
            responseIterator: Iterator = chain.stream(st.session_state.messages)
            # replaced .markdown() with .write_stream()
            aiResponse = st.write_stream(responseIterator)  # type: ignore

        # ✅ Pass full session history as context — this is the memory
        st.session_state.messages.append({"role": "ai", "content": aiResponse})


if __name__ == "__main__":
    load_dotenv()
    groqLLM: ChatGroq = ChatGroq(model="llama-3.3-70b-versatile")
    # Initialize only once — not on every rerun
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.title("💕AI Powered ChatBot💕")
    st.markdown("My QnA Chat Bot Powered by LLMs like Groq, OpenAI, Gemini etc.")
    if st.button("Send Balloons!"):
        st.balloons()
    chatBotWithStreamlitStreaming(groqLLM)
