# Structured Output

LLM always returns plain text(string) as output which is unpredictable, hard to use in code.

```text
"The user is John, he is 30 years old and lives in New York."  ❌
```

- Generally, we need structured output, predictable, typed, validated data so that we can work on the returned data in
  our Gen AI applications. We can use **Pydantic**, **Enums**, **TypedDict** to achieve this. **Pydantic** is prefered
  for Production.
- Always add **Field(description=...)** in pydantic Models— LLM uses it to understand what to extract
- **with_structured_output()** handles prompt engineering, JSON parsing, and Pydantic validation — all in one call.

```python
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq


class Address(BaseModel):
    city: str
    country: str


class Person(BaseModel):
    name: str = Field(description="Full name")
    age: Optional[int] = Field(description="Age if mentioned")
    skills: List[str] = Field(description="List of skills")
    address: Address = Field(description="Where they live")


llmProvider: ChatGroq = ChatGroq(model="llama-3.3-70b-versatile")
structuredLLM = llmProvider.with_structured_output(Person)
response: Person = structuredLLM.invoke("Alice is a Python and ML engineer from London, UK. She's 26.")

print(response.name)  # Alice
print(response.skills)  # ['Python', 'ML']
print(response.address.city)  # London
```

```text
User text
    ↓
prompt (formats input)
    ↓
llm.with_structured_output(Schema)   ← tells LLM to return JSON matching schema
    ↓
Pydantic object (type-safe, validated)
    ↓
response.field  ✅
```
