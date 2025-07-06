from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from config import EMBEDDING_MODEL_NAME
import torch
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )

logger = logging.getLogger(__name__)


def build_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  
        chunk_overlap=500,  
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  
        keep_separator=True
    )
    
    split_docs = []
    for doc in docs:
        splits = text_splitter.split_text(doc.page_content)
        for split in splits:
            # Create new document with same metadata
            new_doc = doc.copy()
            new_doc.page_content = split
            split_docs.append(new_doc)


    logger.info(f"Created {len(split_docs)} chunks from {len(docs)} documents")

    logger.info("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    logger.info("Creating in-memory vector store...")
    
    # Use InMemoryVectorStore instead of Chroma
    vectordb = InMemoryVectorStore.from_documents(
        documents=split_docs,
        embedding=embedding_model
    )
    
    logger.info(f"Vector store created with {len(split_docs)} documents")
    return vectordb
