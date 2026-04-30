import streamlit as st
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


def BasicBotWithMemory(llmProvider: BaseChatModel):
    chatHistory = []
    while True:
        query = input("User: ")
        if query.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!👋")
            break
        chatHistory.append({"role": "user", "content": query})
        chain = llmProvider | StrOutputParser()
        response = chain.invoke(chatHistory)
        chatHistory.append({"role": "ai", "content": response})
        print(f"AI: {response}")


def chatBotQA(llmProvider: BaseChatModel):
    """
    Runs a simple terminal-based QA chatbot loop.
    :param llmProvider: The LLM backend to invoke.
    :type llmProvider: BaseChatModel
    """
    outputParser = StrOutputParser()
    while True:
        question: str = input("User: ")
        if question in ["exit", "quit", "bye"]:
            print("Goodbye! 👋")
            break
        chain = llmProvider | outputParser
        answer: str = chain.invoke(question)
        print(f"AI: {answer}")


def chatBotQAWithStreamlit(llmProvider: BaseChatModel):
    """
    Runs a Streamlit-based QA chatbot with persistent chat history.
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
        # ✅ Pass full session history as context — this is the memory
        aiResponse: str = chain.invoke(st.session_state.messages)
        st.chat_message("ai").markdown(aiResponse)
        st.session_state.messages.append({"role": "ai", "content": aiResponse})


if __name__ == "__main__":
    load_dotenv()
    groqLLM: ChatGroq = ChatGroq(model="llama-3.3-70b-versatile")
    # BasicBotWithMemory(groqLLM)
    # Initialize only once — not on every rerun
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.title("💕AI Powered ChatBot💕")
    st.markdown("My QnA Chat Bot Powered by LLMs like Groq, OpenAI, Gemini etc.")
    if st.button("Send Balloons!"):
        st.balloons()
    chatBotQAWithStreamlit(groqLLM)
