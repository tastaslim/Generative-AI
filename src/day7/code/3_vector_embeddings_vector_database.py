from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

if __name__ == "__main__":
    load_dotenv()
    # Provide OpenAI embeddings model
    # https://docs.langchain.com/oss/python/integrations/embeddings#similarity-metrics

    # Each embedding models come with different vector lengths. For example By default, the length of the embedding vector
    # is 1536 for text-embedding-3-small or 3072 for text-embedding-3-large

    texts = ["""
        Till now we have covered the Preprocessing of Data, Data Chunking techniques and also what is vector database. 
        Now, let’s talk about semantic search. But first, we need to understand vector embeddings, the key to making 
        it work.
        
        Vector embeddings may sound complex, but they’re simply numeric representations of data that capture important 
        features and relationships. Let’s dive into the world of vector embeddings to understand how they work and why 
        they’re essential."""]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  #
    vector = embeddings.embed_documents(texts=texts)
    # print(len(vector[0]))  # 1536
    # print(vector)

    # Now, Vector embedding is created, we need to store the embeddings in vector database
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally
    )
    vector_store = Chroma.from_texts(texts=texts)
    query = "vectors embeddings"
    results = vector_store.search(query=query, search_type="similarity")
    for result in results:
        print(result.page_content)
        print(result.metadata)
