import time
import shutil
from config import CHROMA_PERSIST_DIR

def test_rag(qa, query):
    print(f"Query: {query}\n")
    start = time.time()
    result = qa.invoke(query)
    end = time.time()
    print(f"Inference time: {round(end - start, 3)} sec.")
    print("\nResult:", result)
    