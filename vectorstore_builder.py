from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from config import EMBEDDING_MODEL_NAME
import torch

def build_vectorstore(docs):
    # Create text splitter that preserves metadata
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separator="\n",
        keep_separator=True
    )
    
    # Split documents while preserving metadata
    split_docs = []
    for doc in docs:
        splits = text_splitter.split_text(doc.page_content)
        for split in splits:
            # Create new document with same metadata
            new_doc = doc.copy()
            new_doc.page_content = split
            split_docs.append(new_doc)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    # Create vector store with metadata
    vectordb = InMemoryVectorStore.from_documents(
        documents=split_docs,
        embedding=embedding_model
    #    persist_directory=CHROMA_PERSIST_DIR
    )
    return vectordb
