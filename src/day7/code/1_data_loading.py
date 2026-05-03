from dotenv import load_dotenv
from langchain_community.document_loaders import (
    WikipediaLoader,
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
)


def load_pdf_content(file_path):
    """Load PDF file and return content"""
    pdf_loader = PyPDFLoader(file_path)
    pdf_chunks = pdf_loader.load()
    for pdf_chunk in pdf_chunks:
        print(f"Metadata:\n{pdf_chunk.metadata}\n")
        print(f"Content:\n{pdf_chunk.page_content}\n")


def load_csv_content(file_path):
    """Load CSV file and return content"""
    csv_loader = CSVLoader(file_path)
    csv_chunks = csv_loader.load()
    for csv_chunk in csv_chunks:
        print(f"Metadata:\n{csv_chunk.metadata}\n")
        print(f"Content:\n{csv_chunk.page_content}\n")


def load_web_content(web_url):
    """Load Web contents and return it"""
    web_loader = WebBaseLoader(web_url)
    web_chunks = web_loader.load()
    for web_chunk in web_chunks:
        print(f"Metadata:\n{web_chunk.metadata}\n")
        print(f"Content:\n{web_chunk.page_content}\n")


def load_wiki_content(query: str, max_docs: int):
    """Load Wikipedia contents and return it"""
    wiki_loader = WikipediaLoader(query=query, load_max_docs=max_docs)
    wiki_chunks = wiki_loader.load()
    for wiki_chunk in wiki_chunks:
        print(f"Metadata:\n{wiki_chunk.metadata}\n")
        print(f"Content:\n{wiki_chunk.page_content}\n")


if __name__ == "__main__":
    load_dotenv()
    load_pdf_content("../images/sample_pdf.pdf")
    load_csv_content("../images/sample_csv.csv")
    load_web_content(
        web_url="https://www.educative.io/courses/advanced-rag-techniques/what-is-rag?utm_campaign=brand_educative&utm_source=google&utm_medium=ppc&utm_content=performance_max_india&utm_term=&aff=K3Zq&utm_term=&utm_campaign=%5BNew%5D+Performance+Max&utm_source=adwords&utm_medium=ppc&hsa_acc=5451446008&hsa_cam=18931439518&hsa_grp=&hsa_ad=&hsa_src=x&hsa_tgt=&hsa_kw=&hsa_mt=&hsa_net=adwords&hsa_ver=3&gad_source=1&gad_campaignid=18924941403&gbraid=0AAAAADfWLuRHcy3P5keaa4WDfaXSek1vY&gclid=CjwKCAjwwdbPBhBgEiwAxBRA4R3JNb8TyeLhMOLjkQl9JBOrTpfOSd4Y6PpTIdA5_RWzNlwh7GvAgRoCvzUQAvD_BwE"
    )
    load_wiki_content(query="Gen AI", max_docs=5)
