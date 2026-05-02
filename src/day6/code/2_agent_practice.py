from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class AddNumber(BaseModel):
    """The model for adding numbers"""

    a: int = Field(description="First number")
    b: int = Field(description="Second number")


@tool("add_number", args_schema=AddNumber)
def add_number(a: int, b: int) -> int:
    """Adds two numbers and returns the sum"""
    return a + b


@tool("square_number")
def square_number(number: int) -> int:
    """Calculate and return square of number"""
    return number**2


def agent_practice(llmProvider: BaseChatModel):
    number_adder_tool = create_agent(
        model=llmProvider, tools=[add_number, square_number]
    )
    response = number_adder_tool.invoke(
        {"messages": [{"role": "user", "content": "What is the square of number 2+3"}]}
    )
    return response


if __name__ == "__main__":
    groqLlmProvider: BaseChatModel = ChatOpenAI(model="gpt-4o-mini")
    result = agent_practice(groqLlmProvider)
    # for resp in result["messages"]:
    #     print(resp)
    #     print()

    resp: List[HumanMessage] = result["messages"]
    answer = resp[-1].content
    print(answer)

    # for resp in result["messages"]:
    #     print(resp)
    #     print()
    # The output of above for loop would be the following. Here if you observe, the element in the "messages" list is result.
    # So response["messages"][-1].content will give result right.

    """
    ---- ReAct Pattern -----
    # Think
    content='What is 2+3' additional_kwargs={} response_metadata={} id='5ed0b2a4-b096-40e7-a0fa-a7f659d1c17e'
    
    # Action
    content='' additional_kwargs={'tool_calls': [{'id': '5gf79sf6x', 'function': {'arguments': '{"a":2,"b":3}', 'name': 'add_number'}, 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 289, 'total_tokens': 308, 'completion_time': 0.053982318, 'completion_tokens_details': None, 'prompt_time': 0.02860303, 'prompt_tokens_details': None, 'queue_time': 0.054717614, 'total_time': 0.082585348}, 'model_name': 'llama-3.3-70b-versatile', 'system_fingerprint': 'fp_dae98b5ecb', 'service_tier': 'on_demand', 'finish_reason': 'tool_calls', 'logprobs': None, 'model_provider': 'groq'} id='lc_run--019de7a4-7074-7da3-a144-a1bf75f84141-0' tool_calls=[{'name': 'add_number', 'args': {'a': 2, 'b': 3}, 'id': '5gf79sf6x', 'type': 'tool_call'}] invalid_tool_calls=[] usage_metadata={'input_tokens': 289, 'output_tokens': 19, 'total_tokens': 308}
    
    # Observation
    content='5' name='add_number' id='8cf631dc-230d-4674-a56c-a6e6615b8d47' tool_call_id='5gf79sf6x'
    
    # Result
    content='The answer is 5.' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 319, 'total_tokens': 326, 'completion_time': 0.04211905, 'completion_tokens_details': None, 'prompt_time': 0.015611856, 'prompt_tokens_details': None, 'queue_time': 0.162087812, 'total_time': 0.057730906}, 'model_name': 'llama-3.3-70b-versatile', 'system_fingerprint': 'fp_3272ea2d91', 'service_tier': 'on_demand', 'finish_reason': 'stop', 'logprobs': None, 'model_provider': 'groq'} id='lc_run--019de7a4-71b3-7742-9a1d-b6b225c010b9-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 319, 'output_tokens': 7, 'total_tokens': 326}
    """
