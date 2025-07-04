import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


@st.cache_resource
def get_settings():
    from config import settings
    return settings

settings = get_settings()

def process_documents(uploaded_files, embeddings, temp_dir=settings.TEMP_DIR):
    """Process uploaded documents and create a FAISS vector store."""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    documents = []
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.endswith('.txt') or uploaded_file.name.endswith('.md'):
            loader = TextLoader(temp_path)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue
        
        documents.extend(loader.load())

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    if embeddings:
        vectorstore = FAISS.from_documents(texts, embeddings)
        return vectorstore
    return None