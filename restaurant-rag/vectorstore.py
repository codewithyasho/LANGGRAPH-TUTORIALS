from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd
import os

df = pd.read_csv(
    r"C:\Users\yasho\OneDrive\Desktop\LANGGRAPH_TUT\restaurant-rag\restaurant_reviews.csv")
# print(df.head())

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# creating faiss vector store
vectorstore = FAISS.from_documents(
    documents=[
        Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
        )
        for index, row in df.iterrows()
    ],
    embedding=embeddings
)

vectorstore.save_local("faiss_index")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
