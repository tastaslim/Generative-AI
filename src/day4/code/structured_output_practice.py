from dotenv import load_dotenv

load_dotenv()

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Optional, List, cast

load_dotenv()


class Address(BaseModel):
    """Physical address of the person."""

    city: str = Field(description="City name")
    country: str = Field(description="Country name")


class Person(BaseModel):
    """Represents a person extracted from text."""

    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(default=None, description="Age if mentioned")
    skills: List[str] = Field(default=[], description="List of technical skills")
    address: Address = Field(description="Where the person lives")


prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": "You are an expert at extracting structured information from text.",
        },
        {
            "role": "user",
            "content": "{input}",
        },
    ]
)

groqLLM = ChatGroq(model="llama-3.3-70b-versatile")
structuredLLM = groqLLM.with_structured_output(Person)
chain = prompt | structuredLLM
while True:
    inputPrompt: str = input("User: ")
    response: Person = cast(
        Person,
        chain.invoke(
            {
                "input": "Alice is a Python and ML engineer from London, UK. She is 28 years old."
            }
        ),
    )
    print(response.address, response.skills)
