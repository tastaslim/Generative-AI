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
                    "content": "You are a helpful coding assistant. You are a Senior Python Developer who answers in {language}",
                },
                {
                    "role": "ai",
                    "content": "Sure! I can help you with Python, Java, and more.",
                },
                {"role": "user", "content": "Explain {topic} with an example."},
            ]
        )
    )
    ragPrompt: PromptValue = recommendedWayOfDynamicPrompt.invoke(
        {"topic": "RAG", "language": "English"}
    )
    genAiPrompt: PromptValue = recommendedWayOfDynamicPrompt.invoke(
        {"topic": "Gen AI", "language": "Hindi"}
    )
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


# Now let's do this right way, using chaining in LangChain
def runStaticPromptChain(llmProvider: BaseChatModel) -> None:
    """
    Runs the static prompt using LangChain's pipe operator.

    Chains the static prompt template directly with the LLM provider
    and an output parser using the ``|`` operator.

    :param llmProvider: The LLM backend to invoke.
    :type llmProvider: BaseChatModel
    """
    staticPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
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
    chain = staticPrompt | llmProvider
    print(chain.invoke({}))


def runDynamicPromptChain(llmProvider: BaseChatModel) -> None:
    """
    Runs dynamic prompts using LangChain's pipe operator with batch execution.

    Chains the dynamic prompt template with the LLM provider and output parser,
    then runs all topic-language combinations in a single ``batch()`` call.

    :param llmProvider: The LLM backend to invoke.
    :type llmProvider: BaseChatModel
    """
    dynamicPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            {
                "role": "system",
                "content": "You are a helpful coding assistant. You are a Senior Python Developer who answers in {language}.",
            },
            {
                "role": "ai",
                "content": "Sure! I can help you with Python, Java, and more.",
            },
            {
                "role": "user",
                "content": "Explain {topic} with an example.",
            },
        ]
    )
    staticPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
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
    """
    chain = staticPrompt | dynamicPrompt | llmProvider
    
    Above will give error because Every component in a chain has
    an expected input and output type. Here in chaining the output of first runnable will be passed to next runnable right.
    dynamicPrompt expects a dict but receives a PromptValue — type mismatch. ❌. Think of it like USB ports:
    staticPrompt outputs a Type-C plug (PromptValue).
    dynamicPrompt only accepts a USB-A socket (dict).
    They don't connect — wrong shape.
    llmProvider accepts Type-C (PromptValue) — so it connects perfectly.
    """
    staticChain = staticPrompt | llmProvider
    dynamicChain = dynamicPrompt | llmProvider
    print(staticChain.invoke({}))
    results = dynamicChain.batch(
        [
            {"topic": "RAG", "language": "English"},
            {"topic": "Gen AI", "language": "Hindi"},
        ]
    )
    for result in results:
        print(result)


def runMultiplePromptsChain(llmProvider: BaseChatModel) -> None:
    """
    Runs static and dynamic prompts concurrently using LangChain's batch execution.

    Chains the prompt template with the LLM and output parser using ``|``,
    then runs all prompts in a single ``batch()`` call — replacing the need
    for manual ``ThreadPoolExecutor`` and ``LLMPromptTemplate`` wrapping.

    :param llmProvider: The LLM backend to invoke for all prompts.
    :type llmProvider: BaseChatModel
    """
    dynamicPrompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            {
                "role": "system",
                "content": "You are a helpful coding assistant. You are a Senior Python Developer who answers in {language}.",
            },
            {
                "role": "ai",
                "content": "Sure! I can help you with Python, Java, and more.",
            },
            {
                "role": "user",
                "content": "Explain {topic} with an example.",
            },
        ]
    )
    chain = dynamicPrompt | llmProvider
    results = chain.batch(  # Because chain will also be runnable
        [
            {"topic": "Python", "language": "English"},  # static equivalent
            {"topic": "RAG", "language": "English"},
            {"topic": "Gen AI", "language": "Hindi"},
        ]
    )

    for result in results:
        print(result)


if __name__ == "__main__":
    load_dotenv()
    groqLLMProvider: ChatGroq = ChatGroq(model="llama-3.1-8b-instant")
    openAILLMProvider: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini")
    # runMultiplePrompts(groqLLMProvider)
    # runStaticPromptChain(groqLLMProvider)
    runDynamicPromptChain(groqLLMProvider)
