# app.py
# Kode utama untuk aplikasi Streamlit Project Athena

import streamlit as st
import os
from core.model_loader import load_embedding_model, load_llm
from core.document_processor import process_documents
from agents.ollama_rag_agent import create_ollama_rag_agent, convert_chat_history
from langchain_core.messages import HumanMessage, AIMessage

def main():
    # --- Konfigurasi Halaman Streamlit ---
    st.set_page_config(
        page_title="Project Athena",
        page_icon="🧠",
        layout="wide"
    )



# --- Antarmuka Pengguna (UI) ---

st.title("🧠 OllaRAG: Asisten AI Lokal Anda")
st.caption("Didukung oleh LangChain, LlamaCpp, dan Streamlit")

# Inisialisasi session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# Sidebar untuk pengaturan
with st.sidebar:
    st.header("🚀 Pengaturan")
    
    # Pengaturan Model LLM
    st.subheader("Model Bahasa (LLM)")
    model_path = st.text_input(
        "Path ke Model GGUF Anda", 
        help="Contoh: C:/models/Llama-3.1-8B-Instruct.Q4_K_M.gguf"
    )
    n_gpu_layers = st.number_input("Lapisan GPU (n_gpu_layers)", min_value=-1, value=-1, help="-1 untuk offload semua lapisan yang memungkinkan.")
    n_ctx = st.number_input("Ukuran Konteks (n_ctx)", min_value=512, value=4096)

    if st.button("Muat Model LLM"):
        if model_path:
            st.session_state.llm = load_llm(model_path, n_gpu_layers, n_ctx)
            if st.session_state.llm:
                st.success("Model LLM berhasil dimuat!")
        else:
            st.warning("Silakan masukkan path ke model GGUF.")

    # Pengaturan Basis Pengetahuan (Dokumen)
    st.subheader("Basis Pengetahuan Lokal")
    uploaded_files = st.file_uploader(
        "Unggah dokumen Anda (.pdf, .txt, .md)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    if st.button("Proses Dokumen"):
        if uploaded_files:
            st.session_state.vectorstore = process_documents(uploaded_files)
            st.success("Dokumen berhasil diproses dan diindeks!")
        else:
            st.warning("Silakan unggah setidaknya satu dokumen.")
            
    # Pengaturan Akses Internet
    st.subheader("Akses Internet")
    tavily_api_key = st.text_input(
        "Tavily AI API Key", 
        type="password", 
        help="Dapatkan kunci API gratis dari tavily.com untuk mengaktifkan pencarian web."
    )


# Tampilkan riwayat obrolan
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Lihat Sumber"):
                for source in message["sources"]:
                    st.info(f"**Sumber:** {source.metadata.get('source', 'N/A')}\n\n**Kutipan:**\n\n> {source.page_content}")


# Input dari pengguna
if prompt := st.chat_input("Tanyakan sesuatu pada dokumen Anda atau internet..."):
    # Tambahkan pesan pengguna ke riwayat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tampilkan pesan asisten saat sedang memproses
    with st.chat_message("assistant"):
        # Periksa apakah semua komponen siap
        if not st.session_state.llm:
            st.error("Harap muat model LLM terlebih dahulu dari sidebar.")
        elif not st.session_state.vectorstore and not tavily_api_key:
            st.error("Harap proses dokumen atau sediakan Tavily API key di sidebar.")
        else:
            with st.spinner("Athena sedang berpikir..."):
                agent_executor = create_ollama_rag_agent(
                    st.session_state.llm,
                    st.session_state.vectorstore,
                    tavily_api_key
                )
                
                chat_history = convert_chat_history(st.session_state.messages[:-1])
                
                # Jalankan agent dan stream outputnya
                try:
                    full_response, unique_sources = run_ollama_rag_agent(
                        agent_executor,
                        prompt,
                        chat_history,
                        st.session_state.vectorstore
                    )

                    assistant_message = {"role": "assistant", "content": full_response, "sources": unique_sources}
                    st.session_state.messages.append(assistant_message)
                    
                    # Tampilkan sumber di bawah jawaban yang sudah lengkap
                    if unique_sources:
                        with st.expander("Lihat Sumber"):
                            for source in unique_sources:
                                st.info(f"**Sumber:** {source.metadata.get('source', 'N/A')}\n\n**Kutipan:**\n\n> {source.page_content}")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menjalankan agent: {e}")
                    st.exception(e)

