from typing import Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama


# from langchain_groq import ChatGroq
# from typing import List
# from pydantic import BaseModel, Field

# This code is not streaming
# class Movie(BaseModel):
#     """Represents a movie extracted from text"""
#
#     title: str = Field(description="Hindi Movie title")
#     collection: float = Field(description="Movie worldwide collection in Indian rupees")
#     director: str = Field(description="Movie director")
#     genre: str = Field(description="Movie genre")
#
#
# class AllMovies(BaseModel):
#     """Represents list of movies extracted from text"""
#
#     movies: List[Movie] = Field(description="List of Indian movies")
#
#
# def moviesQnA(llmProvider: BaseChatModel) -> AllMovies:
#     structuredLLMProvider: Runnable = llmProvider.with_structured_output(AllMovies)
#     prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
#         [
#             {
#                 "role": "system",
#                 "content": "You are best movie rater and critic in the Indian Movie industry.",
#             },
#             {"role": "ai", "content": "{input}"},
#         ]
#     )
#     chain = prompt | structuredLLMProvider
#     userInput = input("User:")
#     response: AllMovies = chain.invoke({"input": userInput})
#     return response
#
#


def getMovieStory(llmProvider: BaseChatModel) -> Iterator:
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            {
                "role": "system",
                "content": "You are an exceptional movie storyteller with more than 5000 words.",  # Large answer prompt to showcase streaming
            },
            {
                "role": "user",
                "content": "{input}",
            },
        ]
    )
    chain: Runnable = prompt | llmProvider | StrOutputParser()
    userInput: str = input("User: ")
    dataStream: Iterator = chain.stream({"input": userInput})
    for chunk in dataStream:
        yield chunk


if __name__ == "__main__":
    gemma4LLMProvider: BaseChatModel = ChatOllama(model="gemma4:e2b")
    movieGenerator = getMovieStory(gemma4LLMProvider)
    for movie in movieGenerator:
        print(movie, end="")
