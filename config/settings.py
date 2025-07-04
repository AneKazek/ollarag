# LLM Model Settings
DEFAULT_MODEL_PATH = "path/to/your/default/model.gguf"  # Path to the GGUF model file
DEFAULT_N_GPU_LAYERS = 32  # Number of layers to offload to GPU
DEFAULT_N_CTX = 2048  # Context window size

# Embedding Model Settings
AVAILABLE_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",  # Fast and good for general purpose
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",  # Larger, more powerful model
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5"   # Base version of BGE
}
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # The default embedding model to use

# API Keys
TAVILY_API_KEY = "your_tavily_api_key_here"  # API key for Tavily search

# Directory Settings
TEMP_DIR = "./temp_docs"  # Temporary directory for uploaded files