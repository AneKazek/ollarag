import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
import os

@st.cache_resource(show_spinner="Memuat model embedding...")
def load_embedding_model():
    """
    Memuat model embedding dari Hugging Face.
    Model ini digunakan untuk mengubah teks menjadi vektor.
    Hasilnya di-cache agar tidak perlu dimuat ulang setiap saat.
    """
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

@st.cache_resource(show_spinner="Memuat LLM...")
def load_llm(model_path, n_gpu_layers, n_ctx):
    """
    Memuat model bahasa besar (LLM) dari file GGUF lokal.
    Menggunakan LlamaCpp untuk inferensi yang efisien.
    Hasilnya di-cache untuk model path yang sama.
    """
    if not os.path.exists(model_path):
        st.error(f"Error: File model tidak ditemukan di '{model_path}'")
        return None
    try:
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            temperature=0.2,
            max_tokens=4096,
            n_batch=512,
            verbose=False,
            streaming=True,
        )
        return llm
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None