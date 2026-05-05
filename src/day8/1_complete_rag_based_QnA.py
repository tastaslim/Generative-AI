import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PERSIST_DIR = "./chroma_langchain_db"


def is_store_ready() -> bool:
    """Check if a persisted Chroma DB already exists on disk."""
    return os.path.isdir(CHROMA_PERSIST_DIR) and any(os.scandir(CHROMA_PERSIST_DIR))


def load_vector_store() -> Chroma:
    """Load an existing persisted Chroma vector store — no re-embedding."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )


def rag_preprocessing_pipeline(file_path: str):
    # 1. DocumentLoader
    pdf_loader = PyPDFLoader(file_path=file_path)
    pdf_documents = pdf_loader.load()

    # 2. Chunking/Text Splitters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    split_chunks = text_splitter.split_documents(documents=pdf_documents)

    # 3. Vector Embedding
    vector_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # 4. Vector Store
    vector_store = Chroma.from_documents(
        documents=split_chunks,
        persist_directory="./chroma_langchain_db",
        embedding=vector_embeddings,
    )
    return vector_store


def get_context(vector_store, query: str):
    context = ""  # Step 2 ==> Build Context
    documents = vector_store.similarity_search(query=query, k=15)
    for document in documents:
        context += f"{document.page_content}\n"

    return {
        "context": context,
        "query": query,
    }


def rag_pipeline_data_retrieval(
    vector_store: Chroma,
    query: str,
    ptemplate: PromptTemplate,
    llm_provider: BaseChatModel,
):
    query_context = get_context(vector_store=vector_store, query=query)
    rag_chain = ptemplate | llm_provider
    resp = rag_chain.invoke(query_context)
    return resp.content


def run_qa_loop(vector_store: Chroma, prompt: PromptTemplate, llm: BaseChatModel):
    """
    Stateless Q&A loop. Context is retrieved fresh per query from
    the persisted vector store — no re-embedding between sessions.
    Type 'exit' or 'quit' to stop.
    """
    print("\nRAG Q&A ready. Type 'exit' to quit.\n")
    while True:
        query = input("Question: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        answer = rag_pipeline_data_retrieval(
            vector_store=vector_store,
            query=query,
            ptemplate=prompt,
            llm_provider=llm,
        )
        print(answer)


if __name__ == "__main__":
    load_dotenv()
    file = "TheCloudMigrationHandbook.pdf"
    if is_store_ready():
        print("Loading existing vector store...")
        v_store = load_vector_store()
    else:
        print("No store found. Running preprocessing pipeline...")
        v_store = rag_preprocessing_pipeline(file_path=file)
        print("Preprocessing complete. Store persisted.")

    ## - Query Pipeline ####
    prompt_template = PromptTemplate.from_template("""
        You are a helpful assistant and provide answers based on the context of the user query. 
        If you don't know the answer, then you can say, I don't know. 
        Context: {context}
        Question: {query}
        """)

    openai_llm_provider = ChatOpenAI(model="gpt-5.4")
    run_qa_loop(vector_store=v_store, prompt=prompt_template, llm=openai_llm_provider)
