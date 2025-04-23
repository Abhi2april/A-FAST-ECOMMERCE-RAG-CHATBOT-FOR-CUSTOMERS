import streamlit as st
import os
from dotenv import load_dotenv
from main import setup_model, load_data, create_documents, build_vectorstore, format_context
from query_analyzer import QueryAnalyzer

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput>div>div>input {
        font-size: 1.1rem;
    }
    .stMarkdown {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vectordb = None
    st.session_state.llm = None
    st.session_state.query_analyzer = None

def initialize_system():
    """Initialize the RAG system components"""
    with st.spinner("Initializing the system..."):
        # Setup Groq model
        st.session_state.llm = setup_model()
        
        # Initialize query analyzer
        st.session_state.query_analyzer = QueryAnalyzer(llm=st.session_state.llm)
        
        # Load and process data
        df = load_data()
        docs = create_documents(df)
        st.session_state.vectordb = build_vectorstore(docs)
        
        st.session_state.initialized = True

# Sidebar
with st.sidebar:
    st.title("RAG System")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This is a Retrieval-Augmented Generation (RAG) system that:
    - Processes your queries
    - Retrieves relevant documents
    - Generates accurate answers
    """)
    
    if st.button("Initialize System"):
        initialize_system()

# Main content
st.title("🔍 RAG System Interface")

if not st.session_state.initialized:
    st.info("Please click 'Initialize System' in the sidebar to start.")
    st.stop()

# Query input
query = st.text_input(
    "Enter your query:",
    placeholder="What would you like to know?",
    key="query_input"
)

if query:
    with st.spinner("Processing your query..."):
        # Analyze query and get filters
        filters = st.session_state.query_analyzer.analyze_query(query)
        filter_func = st.session_state.query_analyzer.create_filter_function(filters)
        
        # Retrieve relevant documents
        retriever = st.session_state.vectordb.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": filter_func
            }
        )
        
        relevant_docs = retriever.get_relevant_documents(query)
        
        if not relevant_docs:
            st.warning("No relevant documents found for your query.")
            st.stop()
        
        # Display retrieved documents
        st.markdown("### 📄 Retrieved Documents")
        for i, doc in enumerate(relevant_docs, 1):
            with st.expander(f"Document {i}"):
                st.markdown(doc.page_content)
                st.markdown(f"**Metadata:** {doc.metadata}")
        
        # Format context and generate response
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
        
        # Generate and display response
        with st.spinner("Generating response..."):
            response = st.session_state.llm.invoke(prompt)
            
            st.markdown("### 💡 Response")
            st.markdown(response) 