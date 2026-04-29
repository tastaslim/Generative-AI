from concurrent.futures import ThreadPoolExecutor
from typing import List

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class LLMPromptTemplate(BaseModel):
    llmProvider: BaseChatModel
    prompt: PromptValue


def getStaticPrompt() -> PromptValue:
    # Static Prompts
    staticPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful coding assistant. You are a Senior Python Developer.",
            ),
            ("ai", "Sure! I can help you with Python, Java, and more."),
            ("user", "Explain Python with an example."),
        ]
    )
    formattedPrompt: PromptValue = staticPrompt.invoke({})
    return formattedPrompt


def getDynamicPrompts() -> List[PromptValue]:
    # Dynamic Prompts
    dynamicPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful coding assistant. You are a Senior Python Developer.",
            ),
            ("ai", "Sure! I can help you with Python, Java, and more."),
            ("user", "Explain {topic} with an example."),
        ]
    )
    ragPrompt: PromptValue = dynamicPrompt.invoke({"topic": "RAG"})
    genAiPrompt: PromptValue = dynamicPrompt.invoke({"topic": "Gen AI"})
    return [ragPrompt, genAiPrompt]


def runStaticPrompt(llmProvider):
    staticPrompt: PromptValue = getStaticPrompt()
    pythonOutput: AIMessage = llmProvider.invoke(staticPrompt)
    print(pythonOutput.content)


def runDynamicPrompt(llmProvider):
    dynamicPrompts: List[PromptValue] = getDynamicPrompts()
    for dynamicPrompt in dynamicPrompts:
        dynamicPromptOutput: AIMessage = llmProvider.invoke(dynamicPrompt)
        print(dynamicPromptOutput.content)


def runPrompt(llmPromptTemplate: LLMPromptTemplate) -> str | list[str]:
    llmProvider: BaseChatModel = llmPromptTemplate.llmProvider
    prompt: PromptValue = llmPromptTemplate.prompt
    output: AIMessage = llmProvider.invoke(prompt)
    return output.content


def runMultiplePrompts(llmProvider: BaseChatModel):
    staticPrompt: PromptValue = getStaticPrompt()
    dynamicPrompts: List[PromptValue] = getDynamicPrompts()
    prompts = [staticPrompt, *dynamicPrompts]
    with ThreadPoolExecutor(max_workers=3) as executor:
        result = list(
            executor.map(
                runPrompt,
                (
                    LLMPromptTemplate(llmProvider=llmProvider, prompt=prompt)
                    for prompt in prompts
                ),
            )
        )

    for res in result:
        print(res)


if __name__ == "__main__":
    load_dotenv()
    groqLLMProvider: ChatGroq = ChatGroq(model="llama-3.1-8b-instant")
    openAILLMProvider: ChatOpenAI = ChatOpenAI(model="gpt-5.4-mini")
    runMultiplePrompts(groqLLMProvider)
