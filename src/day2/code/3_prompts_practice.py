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
    """
    A wrapper model that binds an LLM provider with a formatted prompt.

    Used to pass both the provider and prompt together as a single unit,
    especially useful when running multiple prompts concurrently.

    :param llmProvider: The LLM backend to invoke (e.g. ChatGroq, ChatOpenAI).
    :type llmProvider: BaseChatModel
    :param prompt: The formatted prompt ready to be sent to the LLM.
    :type prompt: PromptValue
    """

    llmProvider: BaseChatModel
    prompt: PromptValue


def getStaticPrompt() -> PromptValue:
    """
    Builds and returns a static prompt with no dynamic placeholders.

    The prompt is pre-configured with a system role, an AI introduction,
    and a fixed user query asking to explain Python with an example.

    :return: A formatted prompt ready to be sent to an LLM.
    :rtype: PromptValue
    """
    # staticPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "You are a helpful coding assistant. You are a Senior Python Developer.",
    #         ),
    #         ("ai", "Sure! I can help you with Python, Java, and more."),
    #         ("user", "Explain Python with an example."),
    #     ]
    # )
    recommendedWayOfStaticPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            {
                "role": "system",
                "content": "You are a helpful coding assistant. You are a Senior Python Developer.",
            },
            {
                "role": "ai",
                "content": "Sure! I can help you with Python, Java, and more.",
            },
            {
                "role": "user",
                "content": "Explain Python with an example.",
            },
        ]
    )
    formattedPrompt: PromptValue = recommendedWayOfStaticPrompt.invoke({})
    return formattedPrompt


def getDynamicPrompts() -> List[PromptValue]:
    """
    Builds and returns a list of dynamic prompts with different topics injected.

    Uses a shared prompt template with a ``{topic}`` placeholder,
    and invokes it with multiple topics to generate separate formatted prompts.

    :return: A list of formatted prompts, one per topic.
    :rtype: List[PromptValue]
    """
    # dynamicPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "You are a helpful coding assistant. You are a Senior Python Developer.",
    #         ),
    #         ("ai", "Sure! I can help you with Python, Java, and more."),
    #         ("user", "Explain {topic} with an example."),
    #     ]
    # )
    recommendedWayOfDynamicPrompt: ChatPromptTemplate = (
        ChatPromptTemplate.from_messages(
            [
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant. You are a Senior Python Developer.",
                },
                {
                    "role": "ai",
                    "content": "Sure! I can help you with Python, Java, and more.",
                },
                {"role": "user", "content": "Explain {topic} with an example."},
            ]
        )
    )
    ragPrompt: PromptValue = recommendedWayOfDynamicPrompt.invoke({"topic": "RAG"})
    genAiPrompt: PromptValue = recommendedWayOfDynamicPrompt.invoke({"topic": "Gen AI"})
    return [ragPrompt, genAiPrompt]


def runStaticPrompt(llmProvider: BaseChatModel) -> None:
    """
    Runs the static prompt against the given LLM provider and prints the output.

    :param llmProvider: The LLM backend to invoke.
    :type llmProvider: BaseChatModel
    """
    staticPrompt: PromptValue = getStaticPrompt()
    pythonOutput: AIMessage = llmProvider.invoke(staticPrompt)
    print(pythonOutput.content)


def runDynamicPrompt(llmProvider: BaseChatModel) -> None:
    """
    Runs all dynamic prompts sequentially against the given LLM provider
    and prints each output.

    :param llmProvider: The LLM backend to invoke.
    :type llmProvider: BaseChatModel
    """
    dynamicPrompts: List[PromptValue] = getDynamicPrompts()
    for dynamicPrompt in dynamicPrompts:
        dynamicPromptOutput: AIMessage = llmProvider.invoke(dynamicPrompt)
        print(dynamicPromptOutput.content)


def runPrompt(llmPromptTemplate: LLMPromptTemplate) -> str | list[str]:
    """
    Invokes the LLM provider with the given prompt and returns the response content.

    Designed to be used as a callable in concurrent execution (e.g. ThreadPoolExecutor).

    :param llmPromptTemplate: A wrapper containing the LLM provider and formatted prompt.
    :type llmPromptTemplate: LLMPromptTemplate
    :return: The LLM response content as a string or list of strings.
    :rtype: str | list[str]
    """
    llmProvider: BaseChatModel = llmPromptTemplate.llmProvider
    prompt: PromptValue = llmPromptTemplate.prompt
    output: AIMessage = llmProvider.invoke(prompt)
    return output.content


def runMultiplePrompts(llmProvider: BaseChatModel) -> None:
    """
    Runs static and dynamic prompts concurrently using a thread pool
    and prints all results.

    Combines the static prompt and all dynamic prompts into a single list,
    then executes them in parallel using ``ThreadPoolExecutor`` with a max
    of 3 concurrent workers.

    :param llmProvider: The LLM backend to invoke for all prompts.
    :type llmProvider: BaseChatModel
    """
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
    openAILLMProvider: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini")
    runMultiplePrompts(groqLLMProvider)
