from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

"""
chunk_size and chunk_overlap varies from LLMs to LLMs. These would mainly depend on the maximum context window of an LLM
"""


def run_pdf_splitter(file_path: str) -> List[Document]:
    """Split PDF files into multiple documents."""
    pdf_document_loader = PyPDFLoader(file_path=file_path)
    pdf_content = pdf_document_loader.load()
    pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
    split_texts = pdf_splitter.split_documents(pdf_content)
    return split_texts


def run_md_splitter(file_path: str) -> List[Document]:
    """Split Markdown files into multiple documents."""
    markdown_document_loader = TextLoader(file_path=file_path)
    markdown_content = markdown_document_loader.load()
    markdown_splitter = MarkdownTextSplitter(chunk_size=50, chunk_overlap=20)
    split_texts = markdown_splitter.split_documents(markdown_content)
    return split_texts


def run_txt_splitter(input_text: str) -> List[str]:
    """Split text files into multiple documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
    split_texts = text_splitter.split_text(input_text)
    return split_texts


if __name__ == "__main__":
    load_dotenv()
