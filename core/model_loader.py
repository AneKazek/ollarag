import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import settings

@st.cache_resource
def load_llm(model_path, n_gpu_layers, n_ctx):
    """Load the LlamaCpp model."""
    try:
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=512,
            n_ctx=n_ctx,
            f16_kv=True,
            verbose=True,
        )
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None

@st.cache_resource
def load_embedding_model(model_name):
    """Load the embedding model."""
    try:
        model_path = settings.AVAILABLE_EMBEDDING_MODELS.get(model_name)
        if not model_path:
            st.error(f"Model embedding '{model_name}' tidak ditemukan.")
            return None
        embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs={'device': 'cpu'})
        return embeddings
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None