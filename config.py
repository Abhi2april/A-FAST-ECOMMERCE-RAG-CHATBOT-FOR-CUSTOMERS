import os
import requests
# Load API Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# URLs
NGROK_URL = "https://7ed5-103-47-74-66.ngrok-free.app"

# Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHROMA_PERSIST_DIR = "chroma_db"
