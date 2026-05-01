from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama


def OllamaPractice(llmProvider: BaseChatModel):
    outputParser = StrOutputParser()
    chain: Runnable = llmProvider | outputParser
    output = chain.invoke("Who is the Prime Minister of India?")
    print(output)


if __name__ == "__main__":
    gemma4LLMProvider: BaseChatModel = ChatOllama(model="gemma4:e2b")
    OllamaPractice(gemma4LLMProvider)
