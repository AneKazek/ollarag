import os
from langchain_community.llms import LlamaCpp

model_path = "models\gemma-3-finetune.Q8_0.gguf" # Ganti dengan path model Anda
n_gpu_layers = -1 # Sesuaikan jika Anda menggunakan GPU
n_ctx = 4096

print(f"Checking if model exists at: {model_path}")
if not os.path.exists(model_path):
    print(f"Error: Model file not found at '{model_path}'")
else:
    try:
        print(f"Attempting to load LLM from {model_path}...")
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            temperature=0.2,
            max_tokens=4096,
            n_batch=512,
            verbose=True, # Set verbose to True for more output
            streaming=False,
        )
        print("LLM loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()