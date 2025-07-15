# OllaRAG: Your Local AI Assistant

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green?style=for-the-badge&logo=langchain&logoColor=white)
![LlamaCpp](https://img.shields.io/badge/LlamaCpp-0.2.0-orange?style=for-the-badge&logo=llama&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-0.1.0-purple?style=for-the-badge&logo=ollama&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-VectorStore-yellow?style=for-the-badge&logo=vectorworks&logoColor=white)
![Tavily AI](https://img.shields.io/badge/Tavily%20AI-Search-red?style=for-the-badge&logo=google&logoColor=white)

OllaRAG is a powerful and versatile local AI assistant built using Streamlit, LangChain, and LlamaCpp. It allows you to interact with your documents and the internet using a local Large Language Model (LLM) powered by Ollama, ensuring your data remains private and secure on your machine.

## ✨ Features

- **Local LLM Integration**: Utilize your own GGUF models via LlamaCpp for private and offline AI interactions.
- **Document Q&A**: Upload and query various document types (.pdf, .txt, .md) to get answers directly from your local knowledge base.
- **Dynamic Embedding Model Selection**: Choose from a variety of HuggingFace embedding models to best suit your document processing needs.
- **Web Search Capabilities**: Integrate with Tavily AI for real-time information retrieval from the internet.
- **Intuitive User Interface**: A clean and responsive Streamlit interface for seamless interaction.
- **Clean Code Architecture**: Organized codebase with clear separation of concerns for easy maintenance and future expansion.

## 🚀 Technologies Used

- **Streamlit**: For building the interactive web application.
- **LangChain**: For orchestrating the RAG (Retrieval Augmented Generation) pipeline, agent creation, and tool utilization.
- **LlamaCpp**: For running GGUF-formatted Large Language Models locally.
- **Ollama**: For serving and managing local LLMs.
- **HuggingFace Embeddings**: For generating document embeddings.
- **FAISS**: For efficient similarity search and vector storage of document embeddings.
- **Tavily AI**: For intelligent web search capabilities.
- **Python**: The core programming language.

## ⚙️ Setup and Installation

Follow these steps to get OllaRAG up and running on your local machine.

### Prerequisites

- Python 3.9+
- Git
- Ollama (installed and running with your desired LLM, e.g., `ollama run llama3.1`)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ollarag.git
   cd ollarag
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # On Windows
   # source venv/bin/activate # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your settings:**
   Open `config/settings.py` and update the following:
   - `DEFAULT_MODEL_PATH`: Set this to the absolute path of your GGUF model file (e.g., `C:/models/llama-3.1-8b-instruct.Q4_K_M.gguf`).
   - `TAVILY_API_KEY`: If you plan to use web search, obtain a free API key from [Tavily AI](https://tavily.com/) and paste it here.

## 🏃 How to Run

Once the setup is complete, you can run the Streamlit application:

```bash
streamlit run main.py
```

Your browser should automatically open to the OllaRAG interface (usually `http://localhost:8501`).

## 🚀 Usage

1.  **Load LLM**: In the sidebar, provide the path to your GGUF model and click "Load LLM".
2.  **Process Documents**: Upload your PDF, TXT, or MD files. Select an embedding model and click "Process Documents" to create a local knowledge base.
3.  **Enter Tavily API Key (Optional)**: If you want to enable web search, enter your Tavily AI API key.
4.  **Start Chatting**: Type your questions in the chat input. OllaRAG will use your loaded documents and/or web search to provide answers.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or inquiries, please reach out to [your-email@example.com].

---

**OllaRAG** - *Empowering your local AI interactions.*