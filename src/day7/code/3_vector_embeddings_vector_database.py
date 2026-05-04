from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

if __name__ == "__main__":
    load_dotenv()
    # Provide OpenAI embeddings model
    texts = ["""
        Till now we have covered the Preprocessing of Data, Data Chunking techniques and also what is vector database. 
        Now, let’s talk about semantic search. But first, we need to understand vector embeddings, the key to making 
        it work.
        
        Vector embeddings may sound complex, but they’re simply numeric representations of data that capture important 
        features and relationships. Let’s dive into the world of vector embeddings to understand how they work and why 
        they’re essential."""]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector = embeddings.embed_documents(texts=texts)
    print(len(vector[0]))
    print(vector)
