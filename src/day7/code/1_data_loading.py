from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader

if __name__ == "__main__":
    load_dotenv()
    # Read and load text based files
    # text_loader = TextLoader("../docs/2_RAG.md")  # Read and load text based files
    # Read and load PDF based files
    # pdf_loader = PyPDFLoader("../images/sample_pdf.pdf")
    # csv_loader = CSVLoader("../images/sample_csv.csv")
    # web_loader = WebBaseLoader(
    #     web_path="https://www.educative.io/courses/advanced-rag-techniques/what-is-rag?utm_campaign=brand_educative&utm_source=google&utm_medium=ppc&utm_content=performance_max_india&utm_term=&aff=K3Zq&utm_term=&utm_campaign=%5BNew%5D+Performance+Max&utm_source=adwords&utm_medium=ppc&hsa_acc=5451446008&hsa_cam=18931439518&hsa_grp=&hsa_ad=&hsa_src=x&hsa_tgt=&hsa_kw=&hsa_mt=&hsa_net=adwords&hsa_ver=3&gad_source=1&gad_campaignid=18924941403&gbraid=0AAAAADfWLuRHcy3P5keaa4WDfaXSek1vY&gclid=CjwKCAjwwdbPBhBgEiwAxBRA4R3JNb8TyeLhMOLjkQl9JBOrTpfOSd4Y6PpTIdA5_RWzNlwh7GvAgRoCvzUQAvD_BwE"
    # )

    wikipedia_loader = WikipediaLoader(query="Gen AI", load_max_docs=2)

    # pdf_content = pdf_loader.load()
    # csv_content = csv_loader.load()
    # web_content = web_loader.load()
    wikipedia_content = wikipedia_loader.load()

    # print(f"Length: {len(pdf_content)}")
    # This is an array because you get the content of documents based on number of pages by default
    for response in wikipedia_content:
        print(f"Metadata:\n{response.metadata}\n")
        print(f"Content:\n{response.page_content}\n")
