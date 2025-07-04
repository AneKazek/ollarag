import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from core.model_loader import load_embedding_model

@st.cache_data(show_spinner="Memproses dokumen...")
def process_documents(files):
    """
    Memproses file yang diunggah: memuat, membagi, dan membuat vector store.
    """
    if not files:
        return None

    all_docs = []
    # Gunakan direktori sementara untuk menyimpan file yang diunggah
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Pilih loader berdasarkan ekstensi file
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif uploaded_file.name.endswith((".txt", ".md")):
                loader = TextLoader(file_path)
            else:
                st.warning(f"Format file '{uploaded_file.name}' tidak didukung. Dilewati.")
                continue
            
            all_docs.extend(loader.load())

    if not all_docs:
        return None

    # Bagi dokumen menjadi potongan-potongan kecil
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)

    # Buat vector store menggunakan FAISS
    embedding_model = load_embedding_model()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
    
    return vectorstore