from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_openai import ChatOpenAI

load_dotenv()

# llm = ChatOpenAI(model="gpt-5.5")
# response: AIMessage = llm.invoke(input="Who was the winner of IPL 2025?")
# outputMessage: str | list[str | dict] = response.content
# print(outputMessage)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# response: AIMessage = llm.invoke(
#     input="How to find the largest number in an array? Provide Python code."
# )
# print(response.content)

# If you see above, we are only passing inputs in string, but we can pass all type of inputs. So in Gem AI, we don't prefer inputs like above,
# instead inputs are taken as prompts.

prompts = [
    ("system", "You are a Senior Java Developer."),
    ("user", "How to sort the array?"),
]

response: AIMessage = llm.invoke(prompts)
print(response.content)
