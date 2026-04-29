from dotenv import load_dotenv
from langchain_groq import ChatGroq


def ChatBotQA(llmProvider: ChatGroq):
    while True:
        userInput: str = input("User:")
        if userInput in ("quit", "bye", "close"):
            print("Goodbye 👋")
            break
        output = llmProvider.invoke(userInput)
        print(f"AI: {output.content}")


if __name__ == "__main__":
    load_dotenv()
    groqLLM = ChatGroq(model="llama-3.3-70b-versatile")
    ChatBotQA(groqLLM)
