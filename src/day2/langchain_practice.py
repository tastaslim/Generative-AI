from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-5.4-mini-2026-03-17")
response = llm.invoke(input = "Who is the PM of India?")
print(response)


