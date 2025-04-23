from dotenv import load_dotenv
import os
from data_loader import load_data
from doc_processor import create_documents
from vectorstore_builder import build_vectorstore
from model_config import setup_model, test_model
from query_analyzer import QueryAnalyzer

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def format_context(docs):
    """Format the retrieved documents into a readable context"""
    formatted_context = ""
    for i, doc in enumerate(docs, 1):
        formatted_context += f"\nDocument {i}:\n{doc.page_content}\n"
        formatted_context += f"Metadata: {doc.metadata}\n"
    return formatted_context

def main():
    # setup Groq model
    llm = setup_model()
    
    # initialize query analyzer
    query_analyzer = QueryAnalyzer(llm=llm)
    
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records")
    
    print("Creating documents...")
    docs = create_documents(df)
    print(f"Created {len(docs)} documents")
    
    print("Building vector store...")
    vectordb = build_vectorstore(docs)
    print("Vector store built successfully")
    
    # test the model with a simple prompt
    test_prompt = "Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words."
    test_model(llm, test_prompt)
    
    query = "What product did the customer with ID 55433 buy?"
    print(f"\nSearching for relevant documents for query: {query}")
    
    filters = query_analyzer.analyze_query(query)
    print(f"Extracted filters: {filters}")
    
    filter_func = query_analyzer.create_filter_function(filters)
    
    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 10,
            "filter": filter_func
        }
    )
    
    relevant_docs = retriever.get_relevant_documents(query)
    print(f"Found {len(relevant_docs)} relevant documents")
    
    if not relevant_docs:
        print("No relevant documents found for the given query.")
        return
    
    context = format_context(relevant_docs)
    
    prompt = f"""
    Query: {query}
    Relevant Context:
    {context}

    Instructions: 
    1. Analyze the provided context carefully
    2. Answer the query using only the information from the context
    3. If the information is not available in the context, say "I cannot find the requested information in the available data"
    4. Be specific and include relevant details from the context in your response"""
    
    print("\nGenerating response...")
    result = test_model(llm, prompt)
    print("\nFinal Answer:", result)

if __name__ == "__main__":
    main()
