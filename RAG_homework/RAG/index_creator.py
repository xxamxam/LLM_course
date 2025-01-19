from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader

from RAG.vectorizer import DBVectirizer

def load_index(file_path = './data_for_ml.csv', model_name = "ai-forever/sbert_large_nlu_ru"):
    loader = CSVLoader(file_path=file_path,
        content_columns="section_text",
        metadata_columns=["section_name"],
        csv_args={
        'delimiter': ',',
        # 'quotechar': '"',
        'fieldnames': ["section_name","section_text"]
    })

    docs = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    documents = text_splitter.split_documents(docs)

    embeddings = DBVectirizer(model_name=model_name)

    vector = FAISS.from_documents(documents, embeddings)


    for doc in documents:
        key_text, val_text = documents[0].metadata["section_name"], documents[0].page_content
        key_emb = embeddings.embed_query(key_text)
        # val_emb = embeddings.embed_query(val_text)
        vector.add_texts([val_text], embeddings=[key_emb])


    retriever = vector.as_retriever(
        search_type="mmr",
        # search_kwargs={"k": 3, "fetch_k": 3, "lambda_mult": 0.5},
        search_kwargs={"k": 3,"fetch_k": 20, "lambda_mult": 0.5},
    )
    
    return retriever