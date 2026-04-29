from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable, RunnableLambda
from langchain_groq import ChatGroq


def transformCase(result: str):
    return result.upper()


def runPrompts(llmProvider: BaseChatModel):
    promptTemplate: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            {"role": "system", "content": "You are Python Developer"},
            {"role": "ai", "content": "Sure, I will help you with Python queries."},
            {"role": "user", "content": "Define Variables in Python in {language}"},
        ]
    )
    outputParser = StrOutputParser()
    upperCaseRunnable = RunnableLambda(transformCase)
    chaining: RunnableSerializable = (
        promptTemplate
        | llmProvider
        | outputParser
        | upperCaseRunnable  # You see the power, we can do anything in chains now based on result
    )
    outputs = chaining.batch([{"language": "Hindi"}, {"language": "French"}])
    """
    Now with outputParser in chaining, we don't need to do output.content because outputParser will take care of it.
    for output in outputs:
        print(output.content)
    """
    for output in outputs:
        print(output)


if __name__ == "__main__":
    load_dotenv()
    groqLLM: ChatGroq = ChatGroq(model="llama-3.3-70b-versatile")
    runPrompts(groqLLM)
