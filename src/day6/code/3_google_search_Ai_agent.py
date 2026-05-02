from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import (
    MemorySaver,
)  # from langgraph.checkpoint.memory import InMemorySaver ==> Both are exactly same


def google_serper_without_memory(llm_provider: BaseChatModel, search_provider):
    search_agent = create_agent(
        model=llm_provider,
        tools=[search_provider.run],
        system_prompt="You are an agent who can search for any question on google.",
    )
    # Currently our agent does not have any memory hence, it can't answer questions where we use pronouns.
    # In next tutorial, we will provide memory to our agents.
    """
	User: Who is the PM of India?
	AI: The current Prime Minister of India is Shri Narendra Modi, who was sworn in for his third term on June 9, 2024.
	User: What is his age?
	AI: Could you please provide more context or specify who you are referring to?
	"""
    while True:
        prompt: str = input("User: ")
        if prompt.lower() in ("quit", "exit", "bye"):
            print("Goodbye!👋")
            break

        response = search_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        # for resp in response["messages"]:
        #     print(resp)
        #     print()
        # print(
        #     len(response["messages"])
        # )  # 4 -> Answer was give in 4 steps (Think, Action, Observation, Result) -> ReAct
        print(f'AI: {response["messages"][-1].content}')


def google_serper_with_memory(llm_provider: BaseChatModel, search_provider):
    search_agent = create_agent(
        model=llm_provider,
        tools=[search_provider.run],
        system_prompt="You are an agent who can search for any question on google.",
        checkpointer=MemorySaver(),
    )
    while True:
        chat_id: str = input("Chat id: ")
        prompt: str = input("User: ")
        if prompt.lower() in ("quit", "exit", "bye"):
            print("Goodbye!👋")
            break

        response = search_agent.invoke(
            input={"messages": [{"role": "user", "content": prompt}]},
            config=RunnableConfig(configurable={"thread_id": chat_id}),
        )
        # for resp in response["messages"]:
        #     print(resp)
        #     print()
        # print(
        #     len(response["messages"])
        # )  # 4 -> Answer was give in 4 steps (Think, Action, Observation, Result) -> ReAct
        print(f'AI: {response["messages"][-1].content}')


def google_serper_with_memory_and_streaming(
    llm_provider: BaseChatModel, search_provider
):
    search_agent = create_agent(
        model=llm_provider,
        tools=[search_provider.run],
        system_prompt="You are an agent who can search for any question on google.",
        checkpointer=MemorySaver(),
    )
    while True:
        chat_id: str = input("Chat id: ")
        prompt: str = input("User: ")
        if prompt.lower() in ("quit", "exit", "bye"):
            print("Goodbye!👋")
            break

        response = search_agent.stream(
            input={"messages": [{"role": "user", "content": prompt}]},
            config=RunnableConfig(configurable={"thread_id": chat_id}),
            stream_mode="messages",
        )

        for token, metadata in response:
            print(token.content, end="", flush=True)


if __name__ == "__main__":
    load_dotenv()
    google_serper = GoogleSerperAPIWrapper()
    # google_Serper has one of the tool called .run() which would go and perform google search
    # print(google_serper.run("Who is the Bernardo Silva"))
    openai_model_provider: BaseChatModel = ChatOpenAI(model="gpt-4o-mini")
    # google_serper_without_memory(openai_model_provider)
    # google_serper_with_memory(openai_model_provider, google_serper)
    google_serper_with_memory_and_streaming(openai_model_provider, google_serper)
