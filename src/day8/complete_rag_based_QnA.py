from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    load_dotenv()
    pdf_loader = PyPDFLoader(
        file_path="../day7/images/sample_pdf.pdf"
    )  # DocumentLoader
    text_splitter = RecursiveCharacterTextSplitter()  # Chunking/Text Splitters
    vector_embedding = OpenAIEmbeddings()  # Vector Embedding
    vector_store = Chroma()  # Vector Store
