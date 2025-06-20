import os
import requests
import getpass
if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for groq: ")

# URLs
# NGROK_URL = "https://7ed5-103-47-74-66.ngrok-free.app"

# Model
#EMBEDDING_MODEL_NAME = "thenlper/gte-small"
EMBEDDING_MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
CHROMA_PERSIST_DIR = "chroma_db"
